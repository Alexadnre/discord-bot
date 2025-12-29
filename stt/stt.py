import os
import subprocess
import time
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
BEAM_SIZE = int(os.getenv("STT_BEAM_SIZE", 3)) # Réduit à 3 pour plus de vitesse, 5 est souvent overkill

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
        # OPTI 1: Conversion FFmpeg ultra-rapide
        # -preset superfast réduit la charge CPU
        subprocess.run([
            'ffmpeg', '-y', '-f', 's16le', '-ar', '48000', '-ac', '2',
            '-i', raw_path, '-af', 'aresample=resampler=soxr', '-ar', '16000', 
            '-ac', '1', '-preset', 'superfast', wav_path
        ], check=True, capture_output=True)

        # OPTI 2: Paramètres de transcription stricts
        segments, _ = model.transcribe(
            wav_path, 
            beam_size=BEAM_SIZE, 
            language="fr",
            # --- ANTI-HALLUCINATION ---
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=700, # Plus long pour ignorer les bruits
                speech_pad_ms=200            # Marge réduite pour être plus précis
            ),
            # Supprime les délires quand le modèle hésite
            temperature=0, 
            # Si le score de probabilité est trop bas, on ignore (évite les Vanessa/Colbot)
            no_speech_threshold=0.6,
            log_prob_threshold=-1.0,
            # Indispensable pour éviter les boucles infinies de texte
            condition_on_previous_text=False,
            # --- VITESSE ---
            initial_prompt="Transcription d'une conversation audio propre." # Aide le modèle à démarrer sur le bon ton
        )
        
        text = " ".join([segment.text for segment in segments]).strip()

        # Nettoyage des petites hallucinations résiduelles (mots de 1-2 lettres seuls)
        if len(text) <= 2:
            text = ""

        if text:
            print(f"📝 [{user_id}]: {text}")
        
        return {"text": text}

    except Exception as e:
        print(f"❌ Erreur: {e}")
        return {"text": "", "error": str(e)}
    finally:
        if os.path.exists(raw_path): os.remove(raw_path)
        if os.path.exists(wav_path): os.remove(wav_path)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=3000)