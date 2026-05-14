import os
import time
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from backend.model_handler import get_model_handler

app = FastAPI(title="Emotion Recognition API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(_ROOT, "best_model.pth"))
handler = None


@app.on_event("startup")
def startup():
    global handler
    print(f"Loading model from {MODEL_PATH}...")
    handler = get_model_handler(MODEL_PATH)
    print(f"Model loaded on {handler.device}")


@app.get("/api/health")
def health():
    return {"status": "ok", "model_loaded": handler is not None}


@app.post("/api/predict")
def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "Only image files are supported")

    try:
        image_bytes = file.file.read()
    except Exception:
        raise HTTPException(400, "Failed to read uploaded file")

    if len(image_bytes) > 10 * 1024 * 1024:
        raise HTTPException(413, "Image too large (max 10MB)")

    t_start = time.time()
    emotion, confidence, probabilities = handler.predict(image_bytes)
    elapsed = round((time.time() - t_start) * 1000)

    return {
        "emotion": emotion,
        "confidence": confidence,
        "probabilities": probabilities,
        "processing_time_ms": elapsed,
    }
