import os
import subprocess
import time
import re
from fastapi import FastAPI, Request, Header
from faster_whisper import WhisperModel
import uvicorn
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# --- CONFIGURATION ---
STT_DEVICE = os.getenv("STT_DEVICE", "cuda")
THREADS = int(os.getenv("STT_NUM_THREADS", 4))
MODEL_NAME = os.getenv("STT_MODEL", "large-v3-turbo")
COMPUTE_TYPE = os.getenv("STT_COMPUTE_TYPE", "float16")
BEAM_SIZE = int(os.getenv("STT_BEAM_SIZE", 3))
WAKE_WORD = "bobby"

LOCAL_MODEL_PATH = "/app/models/whisper_models"

print(f"🚀 Initialisation Whisper : {MODEL_NAME} sur {STT_DEVICE}")

model = WhisperModel(
    MODEL_NAME, 
    device=STT_DEVICE, 
    compute_type=COMPUTE_TYPE, 
    cpu_threads=THREADS,
    download_root=LOCAL_MODEL_PATH
)

print("✅ Modèle prêt.")

@app.get("/health")
async def health():
    return {"status": "ready"}

@app.post("/transcribe")
async def transcribe(request: Request, x_user_id: str = Header(None)):
    user_id = x_user_id or "unknown"
    ts = int(time.time() * 1000)
    raw_path = f"/tmp/audio_{user_id}_{ts}.raw"
    wav_path = f"{raw_path}.wav"

    with open(raw_path, "wb") as f:
        async for chunk in request.stream():
            f.write(chunk)

    try:
        # Conversion FFmpeg
        subprocess.run([
            'ffmpeg', '-y', '-f', 's16le', '-ar', '48000', '-ac', '2',
            '-i', raw_path, '-af', 'aresample=resampler=soxr', '-ar', '16000', 
            '-ac', '1', '-preset', 'superfast', wav_path
        ], check=True, capture_output=True)

        # Transcription avec paramètres de confiance
        segments, _ = model.transcribe(
            wav_path, 
            beam_size=BEAM_SIZE, 
            language="fr",
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=700, speech_pad_ms=200),
            temperature=0, 
            condition_on_previous_text=False,
            initial_prompt=f"L'utilisateur s'adresse à son assistant {WAKE_WORD.capitalize()}."
        )
        
        # Filtrage par confiance (logprob) pour éviter les faux positifs sonores
        valid_segments = []
        for s in segments:
            # avg_logprob: plus c'est proche de 0, plus l'IA est sûre d'elle.
            # -0.5 est un bon compromis pour rejeter les hallucinations.
            if s.avg_logprob > -0.5:
                valid_segments.append(s.text)
        
        full_text = " ".join(valid_segments).strip()

        # --- LOGIQUE WAKE WORD (BOBBY) ---
        # On utilise une regex pour détecter "bobby" même avec de la ponctuation
        # Ex: "Bobby, fais ça." ou "Hé Bobby!"
        pattern = rf"\b{WAKE_WORD}\b"
        match = re.search(pattern, full_text, re.IGNORECASE)

        if match:
            # On récupère ce qui est APRÈS le mot "Bobby"
            # Si l'utilisateur dit "Hé Bobby quelle heure est-il", on garde "quelle heure est-il"
            start_index = match.end()
            command_text = full_text[start_index:].strip(",.?! ")
            
            # Si le texte après Bobby est trop court, on renvoie quand même tout le texte
            # (Au cas où l'utilisateur dit juste "Bobby ?")
            final_text = command_text if len(command_text) > 1 else full_text

            print(f"🎯 [DETECTION {user_id}]: {final_text}")
            return {"text": final_text, "detected": True}
        
        # Si Bobby n'est pas détecté, on renvoie du vide
        return {"text": "", "detected": False}

    except Exception as e:
        print(f"❌ Erreur: {e}")
        return {"text": "", "error": str(e)}
    finally:
        if os.path.exists(raw_path): os.remove(raw_path)
        if os.path.exists(wav_path): os.remove(wav_path)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=3000)