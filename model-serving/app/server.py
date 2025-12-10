"""
Model Serving API (FastAPI).

Exposes the finetuned model for real-time inference.
"""

from fastapi import FastAPI, HTTPException
import torch
from .schemas import PredictionRequest, PredictionResponse
from .load_model import load_finetuned_model
from .inference import InferenceEngine

app = FastAPI(title="TikTok Harmful Content Detection API")

# Global state
engine = None

@app.on_event("startup")
def load_engine():
    global engine
    try:
        model = load_finetuned_model(device_str="cuda" if torch.cuda.is_available() else "cpu")
        device = next(model.parameters()).device
        engine = InferenceEngine(model, device)
        print("Inference Engine loaded successfully.")
    except Exception as e:
        print(f"FAILED to load model: {e}")
        # Dont crash app, but health check will fail logic if we added one
        
@app.get("/health")
def health_check():
    if engine is None:
        return {"status": "loading_or_failed"}
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        result = engine.predict(request)
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
