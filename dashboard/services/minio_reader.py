"""
MinIO Reader Service.

Helper to generate presigned URLs for video playback.
"""

from minio import Minio
from datetime import timedelta
import os

# Config
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MINIO_SECURE = os.getenv("MINIO_SECURE", "False").lower() == "true"
BUCKET_NAME = "tiktok-data" # Adjust if variable

def get_minio_client():
    return Minio(
        MINIO_ENDPOINT.replace("http://", "").replace("https://", ""),
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=MINIO_SECURE
    )

def get_video_url(video_id: str, bucket: str = "tiktok-data") -> str:
    """
    Generate a presigned URL for the video file.
    Try locations: 
      - bronze/{video_id}/video.mp4 (raw crawl)
      - silver/{video_id}/video.mp4 (processed, if stored)
    """
    client = get_minio_client()
    
    # Bronze usually has the playback file
    object_name = f"bronze/{video_id}/video.mp4"
    
    try:
        # Check existence stat
        client.stat_object(bucket, object_name)
        url = client.get_presigned_url(
            "GET",
            bucket,
            object_name,
            expires=timedelta(hours=1),
        )
        return url
    except Exception as e:
        # Try silver if bronze missing? Or just return None
        return None
