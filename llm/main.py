import os
from fastapi import FastAPI
from pydantic import BaseModel
from llama_cpp import Llama

app = FastAPI()

# Configuration
MODEL_PATH = os.getenv("MODEL_PATH", "/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf")
THREADS = int(os.getenv("LLM_THREADS", "4"))
# Nombre de couches à placer sur GPU (0 = CPU only). Llama-cpp-python acceptera
# le paramètre `n_gpu_layers` si le backend a été compilé avec CUDA.
N_GPU_LAYERS = int(os.getenv("N_GPU_LAYERS", "0"))

print(f"Loading Llama model...")
# n_ctx=2048 permet d'avoir une mémoire de conversation suffisante
if N_GPU_LAYERS and N_GPU_LAYERS > 0:
    print(f"Initializing Llama with {N_GPU_LAYERS} GPU layers...")
    llm = Llama(model_path=MODEL_PATH, n_threads=THREADS, n_ctx=2048, n_gpu_layers=N_GPU_LAYERS,offload_kqv=True)
else:
    llm = Llama(model_path=MODEL_PATH, n_threads=THREADS, n_ctx=2048)
print("Model loaded")

class GenRequest(BaseModel):
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.7


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/generate")
def generate(req: GenRequest):
    try:
        # On force la réinitialisation de l'état interne
        llm.reset() 
        
        res = llm.create_completion(
            prompt=req.prompt,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            stop=["<|eot_id|>", "User:"]
        )
        return {"text": res["choices"][0]["text"].strip()}
    except Exception as e:
        return {"text": "", "error": str(e)}
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
