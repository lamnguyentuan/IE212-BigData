from __future__ import annotations
from pathlib import Path

# Default paths cho Medallion layout (có thể override bằng configs/paths.yaml)
DEFAULT_DATA_ROOT = Path("tiktok-data")
DEFAULT_BRONZE_SUBDIR = "bronze"
DEFAULT_SILVER_SUBDIR = "silver"
DEFAULT_GOLD_SUBDIR = "gold"

# Một số default khác (đồng bộ với preprocess_config.yaml)
DEFAULT_AUDIO_SAMPLE_RATE = 16000
DEFAULT_NUM_FRAMES = 16
DEFAULT_FRAME_SIZE = (224, 224)

# Tên bucket MinIO mặc định (có thể đổi trong env / config)
DEFAULT_MINIO_BUCKET = "tiktok-data"

# Env vars để config MinIO (optional)
ENV_MINIO_ENDPOINT = "MINIO_ENDPOINT"
ENV_MINIO_ACCESS_KEY = "MINIO_ACCESS_KEY"
ENV_MINIO_SECRET_KEY = "MINIO_SECRET_KEY"
ENV_MINIO_SECURE = "MINIO_SECURE"  # "0" or "1"
