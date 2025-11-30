from minio import Minio
import yaml
from pathlib import Path

BASE = Path(__file__).parent

def get_minio_client(config_name: str = "config_tiktok.yaml"):
    cfg_path = BASE / config_name
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    client = Minio(
        cfg["endpoint"],
        access_key=cfg["access_key"],
        secret_key=cfg["secret_key"],
        secure=cfg["secure"],
    )

    bucket = cfg["bucket"]
    if not client.bucket_exists(bucket):
        client.make_bucket(bucket)

    return client, bucket
