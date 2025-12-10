# 🗂 MinIO Configuration

Using **Medallion Architecture** (Bronze/Silver/Gold) on 2 buckets:
- `tiktok-data` for crawled data.
- `tikharm` for the static dataset.

## `minio_client.py`
Provides helpers:
- `get_minio_client(config_name)`: Returns `(client, bucket)`.
- `upload_file(client, bucket, object_name, path)`: Auto-detects mime-type.
- `upload_directory(client, bucket, local_dir)`: Recursive upload.

## Configs
- `config_tiktok.yaml`
- `config_tikharm.yaml`
