import os
import subprocess
import time
import re
import httpx
from fastapi import FastAPI, Request, Header
from faster_whisper import WhisperModel
import uvicorn
from threading import Lock
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
LLM_URL = os.getenv("LLM_URL", "http://llm:8000/generate")

# --- GESTION ÉTAT & CONCURRENCE ---
stt_lock = Lock()
LAST_REQUEST_TIME = 0.0
DEBOUNCE_DELAY = 2.0  # On ignore les requêtes identiques à moins de 2s d'intervalle

LOCAL_MODEL_PATH = "/app/models/whisper_models"

print(f"🚀 Initialisation Whisper : {MODEL_NAME} sur {STT_DEVICE}")
model = WhisperModel(
    MODEL_NAME, 
    device=STT_DEVICE, 
    compute_type=COMPUTE_TYPE, 
    cpu_threads=THREADS,
    download_root=LOCAL_MODEL_PATH
)
print("✅ Modèle STT prêt.")

@app.get("/health")
async def health():
    return {"status": "ready"}

@app.post("/transcribe")
async def transcribe(request: Request, x_user_id: str = Header(None)):
    global LAST_REQUEST_TIME
    user_id = x_user_id or "unknown"
    ts = int(time.time() * 1000)
    raw_path = f"/tmp/audio_{user_id}_{ts}.raw"
    wav_path = f"{raw_path}.wav"

    # Réception du flux audio
    with open(raw_path, "wb") as f:
        async for chunk in request.stream():
            f.write(chunk)

    # On utilise un verrou pour éviter que deux transcriptions ne se battent pour le GPU
    with stt_lock:
        try:
            # 1. Conversion FFmpeg (48kHz Stereo Raw -> 16kHz Mono Wav)
            subprocess.run([
                'ffmpeg', '-y', '-f', 's16le', '-ar', '48000', '-ac', '2',
                '-i', raw_path, '-af', 'aresample=resampler=soxr', '-ar', '16000', 
                '-ac', '1', '-preset', 'superfast', wav_path
            ], check=True, capture_output=True)

            # 2. Transcription via Faster-Whisper
            segments, _ = model.transcribe(
                wav_path, 
                beam_size=BEAM_SIZE, 
                language="fr",
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=700, speech_pad_ms=200),
                temperature=0,
                initial_prompt=f"L'utilisateur appelle son assistant {WAKE_WORD}."
            )
            
            valid_segments = [s.text for s in segments if s.avg_logprob > -0.5]
            full_text = " ".join(valid_segments).strip()

            # 3. Logique de détection du Wake Word et Debounce
            pattern = rf"(?i)\b{WAKE_WORD}\b"
            match = re.search(pattern, full_text)

            if match:
                current_time = time.time()
                # ANTI-REBOND : Si on a déjà traité une demande il y a moins de X secondes
                if (current_time - LAST_REQUEST_TIME) < DEBOUNCE_DELAY:
                    print(f"⚠️ [STT] Doublon détecté pour {user_id}, requête ignorée.")
                    return {"text": "", "detected": False}

                LAST_REQUEST_TIME = current_time

                # Extraction de la commande
                clean_query = re.sub(pattern, "", full_text, flags=re.IGNORECASE).strip(",.?! ")
                final_query = clean_query if len(clean_query) > 1 else full_text
                
                print(f"🧠 [STT] Question valide : {final_query}")

                # 4. Envoi au LLM
                async with httpx.AsyncClient() as client:
                    # Note : On ne met PAS <|begin_of_text|> car le LLM l'ajoute souvent lui-même
                    prompt_format = (
                        "<|start_header_id|>system<|end_header_id|>\n\n"
                        "Tu es BobbY, un assistant Discord. Réponds en une seule phrase courte.<|eot_id|>"
                        "<|start_header_id|>user<|end_header_id|>\n\n"
                        f"{final_query}<|eot_id|>"
                        "<|start_header_id|>assistant<|end_header_id|>\n\n"
                    )

                    try:
                        llm_response = await client.post(
                            LLM_URL, 
                            json={
                                "prompt": prompt_format, 
                                "max_tokens": 64, 
                                "temperature": 0.4, 
                                "stop": ["<|eot_id|>", "\n"]
                            },
                            timeout=60.0
                        )
                        
                        if llm_response.status_code == 200:
                            llm_answer = llm_response.json().get("text", "").strip()
                            print(f"🤖 [LLM]: {llm_answer}")
                            return {"text": llm_answer, "detected": True}
                        else:
                            return {"text": "Erreur serveur LLM.", "detected": True}
                    
                    except Exception as e:
                        print(f"❌ Erreur appel LLM: {e}")
                        return {"text": "Le cerveau est débranché.", "detected": True}

            return {"text": "", "detected": False}

        except Exception as e:
            print(f"❌ Erreur STT : {e}")
            return {"text": "", "error": str(e)}
        finally:
            # Nettoyage
            for p in [raw_path, wav_path]:
                if os.path.exists(p): os.remove(p)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=3000)