# Model Serving API

A high-performance **FastAPI** application to serve the trained Multimodal Classifier.

## Features
- **Endpoint**: `POST /predict`
- **Backends**: 
    - **Local**: Loads features from file system (if precomputed).
    - **MinIO**: Can fetch features on-demand.
- **Model Loader**: Automatically pulls the finetuned model from Hugging Face Hub (`funa21/tiktok-vn-finetune`) if not found locally.

## API Specification

**Request**:
```json
{
  "video_id": "7234...",
  "use_minio": true
}
```

**Response**:
```json
{
  "video_id": "7234...",
  "label_name": "Harmful",
  "confidence": 0.98,
  "probabilities": {
    "Safe": 0.02,
    "Harmful": 0.98
  }
}
```

## Deployment
Dockerized for easy deployment.
```bash
docker build -t tiktok-serving -f model-serving/Dockerfile .
docker run -p 8000:8000 tiktok-serving
```
