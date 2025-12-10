from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class PredictionRequest(BaseModel):
    video_id: str
    # Option 1: Paths to features (if precomputed and accessible locally/network)
    video_feat_path: Optional[str] = None
    audio_feat_path: Optional[str] = None
    metadata_feat_path: Optional[str] = None
    
    # Option 2: Force reload from MinIO (slower but robust)
    use_minio: bool = False
    
    class Config:
        json_schema_extra = {
            "example": {
                "video_id": "7231234567",
                "use_minio": True
            }
        }

class PredictionResponse(BaseModel):
    video_id: str
    label_id: int
    label_name: str
    confidence: float
    probabilities: Dict[str, float]
    processing_time_ms: float
