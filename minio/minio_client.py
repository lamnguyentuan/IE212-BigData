from minio import Minio
import yaml
import os
import mimetypes
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE = Path(__file__).parent

def get_minio_client(config_name: str = "config_tiktok.yaml"):
    """
    Load config and return (MinioClient, bucket_name).
    """
    cfg_path = BASE / config_name
    if not cfg_path.exists():
        raise FileNotFoundError(f"MinIO config not found: {cfg_path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Allow env var override for secret key and endpoint
    endpoint = os.getenv("MINIO_ENDPOINT") or cfg["endpoint"]
    access_key = os.getenv("MINIO_ACCESS_KEY") or cfg["access_key"]
    secret = os.getenv("MINIO_SECRET_KEY") or cfg.get("secret_key")
    
    if not secret:
        # Fallback for local dev if set in simple config, 
        # though ideally secrets shouldn't be in yaml.
        print(f"WARNING: No MINIO_SECRET_KEY found for {config_name}. Using default 'minioadmin' if not provided.")
        secret = "minioadmin"

    client = Minio(
        endpoint,
        access_key=access_key,
        secret_key=secret,
        secure=cfg.get("secure", False),
    )

    bucket = cfg["bucket"]
    if not client.bucket_exists(bucket):
        try:
            client.make_bucket(bucket)
            print(f"Created bucket: {bucket}")
        except Exception as e:
            print(f"Error creating bucket {bucket}: {e}")

    return client, bucket

def upload_file(client: Minio, bucket: str, object_name: str, file_path: str) -> bool:
    """
    Helper to upload a single file with auto mime-type detection.
    Returns True if successful, False otherwise.
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

def upload_directory(client: Minio, bucket: str, local_dir: str, prefix: str = "") -> None:
    """
    Recursively upload a directory to MinIO.
    """
    p = Path(local_dir)
    if not p.is_dir():
        print(f"Not a directory: {local_dir}")
        return

    for f in p.rglob("*"):
        if f.is_file():
            # Create relative path for object name
            rel_path = f.relative_to(p)
            object_name = str(Path(prefix) / rel_path)
            
            upload_file(client, bucket, object_name, str(f))
            print(f"Uploaded: {object_name}")
