from minio import Minio
import os
import mimetypes
from pathlib import Path
from dotenv import load_dotenv

# Import standardized config
import sys
# Add root to sys.path to ensure common imports work if running from subdir
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from common.utils.minio_utils import MinioConfig, MinioClientWrapper

load_dotenv()

def get_minio_client(config_name: str = "config_tiktok.yaml"):
    """
    Unified MinIO client loader.
    Ignores config_name and uses environment variables + common defaults.
    """
    # Prefer env vars, fallback to default "tiktok-data"
    bucket_name = os.getenv("MINIO_BUCKET", "tiktok-data")
    
    cfg = MinioConfig.from_env(bucket=bucket_name)
    
    # Ensure client is initialized and bucket exists
    wrapper = MinioClientWrapper(cfg)
    
    # Return raw client and bucket name to match legacy signature
    # wrapper._client is the raw Minio object
    return wrapper._client, cfg.bucket

def upload_file(client: Minio, bucket: str, object_name: str, file_path: str) -> bool:
    """
    Legacy helper kept for compatibility.
    """
    try:
        content_type, _ = mimetypes.guess_type(file_path)
        if content_type is None:
            content_type = "application/octet-stream"

        client.fput_object(
            bucket_name=bucket,
            object_name=object_name,
            file_path=file_path,
            content_type=content_type
        )
        return True
    except Exception as e:
        print(f"Failed to upload {file_path} to {bucket}/{object_name}: {e}")
        return False
