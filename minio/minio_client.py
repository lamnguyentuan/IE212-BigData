from minio import Minio
import yaml
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE = Path(__file__).parent

def get_minio_client(config_name: str = "config_tiktok.yaml"):
    cfg_path = BASE / config_name
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    client = Minio(
        cfg["endpoint"],
        access_key=cfg["access_key"],
        secret_key=os.getenv("MINIO_SECRET_KEY"),
        secure=cfg["secure"],
    )

    bucket = cfg["bucket"]
    if not client.bucket_exists(bucket):
        client.make_bucket(bucket)

    return client, bucket
