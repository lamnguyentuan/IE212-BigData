"""
Inference Logic.

Orchestrates loading features and running the model.
"""

import sys
import torch
import numpy as np
import time
from pathlib import Path
from typing import Dict, Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from common.features.multimodal_feature_builder import MultimodalFeatureBuilder
from common.features.feature_schema import MultimodalFeatureRow
from .schemas import PredictionRequest, PredictionResponse

class InferenceEngine:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        # Helper to load features compatible with model training
        self.feature_builder = MultimodalFeatureBuilder()
        
        self.labels_map = {0: "Safe", 1: "Harmful"} # Adjust based on 2-class training

    def predict(self, request: PredictionRequest) -> PredictionResponse:
        start_time = time.time()
        
        # 1. Load Features
        # For serving, we usually read from Silver (precomputed) 
        # or we might need to run raw preprocessing (expensive).
        # Assuming for this phase we fetch precomputed features from local/MinIO silver.
        
        # If paths provided explicitly:
        # (This path needs MultimodalFeatureBuilder logic to be flexible, but 
        # FeatureBuilder expects a strict directory structure silver/{vid}...)
        
        # Simplest approach: reuse build_row_from_dir if we have a local dir
        # We assume data is synced to `tiktok-data/silver/{video_id}` usually.
        
        vid = request.video_id
        silver_dir = self.feature_builder.silver_dir() / vid
        
        if not silver_dir.exists() or not any(silver_dir.iterdir()):
             # Try to fetch from MinIO
             try:
                 print(f"Fetching features for {vid} from MinIO...")
                 from common.utils.minio_utils import MinioConfig, MinioClientWrapper
                 # We assume bucket is tiktok-data for now, or load from env
                 cfg = MinioConfig.from_env(bucket="tiktok-data")
                 mc = MinioClientWrapper(cfg)
                 
                 # Download silver files using helper
                 mc.download_silver_files(vid, self.feature_builder.base_dir)
                 print(f"Downloaded silver files for {vid}.")
                 
             except Exception as e:
                 if request.use_minio:
                    # If explicitly requested and failed, raise error
                    raise FileNotFoundError(f"Features for {vid} not found locally and MinIO fetch failed: {e}")
                 else:
                     # If use_minio=False (default in request), we still tried as fallback, but report local missing
                     print(f"MinIO fallback failed: {e}")
                     raise FileNotFoundError(f"Features for {vid} not found locally")

        row = self.feature_builder.build_row_from_dir(vid, silver_dir, None)
        if row is None:
             raise ValueError(f"Failed to build feature row for {vid}")
             
        # 2. Prepare Tensors
        # Add batch dimension
        v = torch.tensor(row.video_emb, dtype=torch.float32).unsqueeze(0).to(self.device)
        a = torch.tensor(row.audio_emb, dtype=torch.float32).unsqueeze(0).to(self.device)
        t = torch.tensor(row.text_emb, dtype=torch.float32).unsqueeze(0).to(self.device)
        m = torch.tensor(row.metadata_numeric, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # 3. Model Forward
        with torch.no_grad():
            logits = self.model(v, a, t, m)
            probs = torch.softmax(logits, dim=1)
            
        # 4. Result
        top_p, top_class = torch.max(probs, dim=1)
        score = top_p.item()
        label_id = top_class.item()
        label_name = self.labels_map.get(label_id, "Unknown")
        
        # Detailed probabilities
        probs_dict = {
            self.labels_map.get(i, str(i)): p.item() 
            for i, p in enumerate(probs[0])
        }
        
        duration = (time.time() - start_time) * 1000
        
        return PredictionResponse(
            video_id=vid,
            label_id=label_id,
            label_name=label_name,
            confidence=score,
            probabilities=probs_dict,
            processing_time_ms=duration
        )
