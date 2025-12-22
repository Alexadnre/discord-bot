import os
import subprocess
from fastapi import FastAPI, Request, Header
from faster_whisper import WhisperModel
import uvicorn
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# Récupération des config depuis le .env
STT_DEVICE = os.getenv("STT_DEVICE", "cpu")
THREADS = int(os.getenv("STT_NUM_THREADS", 2))

print(f"🚀 Initialisation Whisper sur {STT_DEVICE} avec {THREADS} threads...")
# Le modèle est chargé ICI (avant le démarrage du serveur)
model = WhisperModel("base", device=STT_DEVICE, compute_type="int8", cpu_threads=THREADS)
print("✅ Modèle chargé et prêt à transcrire.")

# --- AJOUT DE LA ROUTE HEALTH ---
@app.get("/health")
async def health():
    """Route utilisée par Docker pour savoir si le modèle est prêt"""
    return {"status": "ready"}
# --------------------------------

@app.post("/transcribe")
async def transcribe(request: Request, x_user_id: str = Header(None)):
    user_id = x_user_id or "unknown"
    # Utiliser un timestamp pour éviter les collisions si l'utilisateur parle vite
    import time
    ts = int(time.time() * 1000)
    raw_path = f"/tmp/audio_{user_id}_{ts}.raw"
    wav_path = f"{raw_path}.wav"

    with open(raw_path, "wb") as f:
        async for chunk in request.stream():
            f.write(chunk)

    try:
        # Conversion 48kHz Stereo -> 16kHz Mono
        subprocess.run([
            'ffmpeg', '-y', '-f', 's16le', '-ar', '48000', '-ac', '2',
            '-i', raw_path, '-ar', '16000', '-ac', '1', wav_path
        ], check=True, capture_output=True)

        segments, _ = model.transcribe(wav_path, beam_size=5, language="fr")
        text = " ".join([segment.text for segment in segments]).strip()

        print(f"📝 [{user_id}]: {text}")
        
        return {"text": text}
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return {"text": "", "error": str(e)}
    finally:
        # On nettoie toujours les fichiers, même en cas d'erreur
        if os.path.exists(raw_path): os.remove(raw_path)
        if os.path.exists(wav_path): os.remove(wav_path)

if __name__ == "__main__":
    # Port 3000 pour correspondre à ton Docker Compose
    uvicorn.run(app, host="0.0.0.0", port=3000)