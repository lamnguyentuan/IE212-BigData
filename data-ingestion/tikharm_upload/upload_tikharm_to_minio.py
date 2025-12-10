"""
Upload TikHarm dataset to MinIO + Generate Manifest.

Enhancements:
- Uses `minio_client.upload_file` helper.
- Generates a `dataset_manifest.json` locally and uploads to MinIO root or bronze.
"""

import json
import io
import sys
from pathlib import Path
from typing import List, Dict

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "minio"))

from minio_client import get_minio_client, upload_file
from data_pipeline.utils.logger import get_logger

logger = get_logger("tikharm-upload")
SRC_ROOT = ROOT / "offline-training" / "datasets" / "TikHarm"

SPLITS = ["train", "val", "test"]
VIDEO_EXTS = {".mp4", ".mov", ".avi"}

def upload_and_index():
    client, bucket = get_minio_client("config_tikharm.yaml")
    
    if not SRC_ROOT.exists():
        logger.error(f"Source root not found: {SRC_ROOT}")
        return

    manifest: List[Dict] = []

    for split in SPLITS:
        split_dir = SRC_ROOT / split
        if not split_dir.exists(): 
            continue
            
        for label_dir in split_dir.iterdir():
            if not label_dir.is_dir(): continue
            
            label_name = label_dir.name
            
            for vid_file in label_dir.rglob("*"):
                if vid_file.suffix.lower() not in VIDEO_EXTS:
                    continue
                
                # Create ID
                # Simple ID strategy: split_label_filename (sanitized)
                safe_name = vid_file.stem.replace(" ", "_")
                clean_label = label_name.replace(" ", "_").lower()
                video_id = f"{split}_{clean_label}_{safe_name}"
                
                # 1. Upload Video
                obj_vid = f"bronze/{video_id}/video.mp4"
                if upload_file(client, bucket, obj_vid, str(vid_file)):
                    logger.info(f"Uploaded {video_id}")
                else:
                    logger.error(f"Failed {video_id}")
                    continue

                # 2. Upload Metadata
                meta = {
                    "video_id": video_id,
                    "split": split,
                    "label": label_name,
                    "original_path": str(vid_file.relative_to(SRC_ROOT))
                }
                
                obj_meta = f"bronze/{video_id}/metadata.json"
                try:
                    meta_bytes = json.dumps(meta).encode('utf-8')
                    client.put_object(bucket, obj_meta, io.BytesIO(meta_bytes), len(meta_bytes), content_type="application/json")
                except Exception as e:
                    logger.error(f"Meta upload failed: {e}")

                manifest.append(meta)

    # 3. Write Manifest
    manifest_path = ROOT / "tikharm_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    
    logger.info(f"Manifest written to {manifest_path}")
    
    # Optional: Upload manifest to MinIO
    upload_file(client, bucket, "dataset_manifest.json", str(manifest_path))
    logger.info("Manifest uploaded to MinIO root.")

if __name__ == "__main__":
    upload_and_index()
