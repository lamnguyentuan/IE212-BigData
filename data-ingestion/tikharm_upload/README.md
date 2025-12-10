# TikHarm Dataset Uploader

Utility scripts to ingest the TikHarm dataset into the project's Data Lake.

## Purpose
The TikHarm dataset provides labeled training data for:
- Adult Content
- Harmful Behavior
- Safe Content
- Physical Harm/Suicide

This uploader moves the data from a local directory structure into the **Bronze Layer** on MinIO, standardizing the format for downstream ETL processing.

## Usage

```bash
python upload_tikharm_to_minio.py
```

## Configuration
Update the `TIKHARM_ROOT` variable in the script or provide it via environment variables if the dataset location changes.

## Output
Generates a `tikharm_manifest.json` mapping all uploaded videos to their S3 paths.
