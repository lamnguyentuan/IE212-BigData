"""
Package tikharm_upload

Chức năng:
- Tiền xử lý logic đặt tên video_id cho TikHarm
- Upload trực tiếp toàn bộ video TikHarm từ local lên MinIO (layer bronze)
  theo layout:

    bucket (từ config_tikharm.yaml) /
    └── bronze/
        └── {video_id}/
            ├── video.mp4
            └── metadata.json

Trong đó:
- video_id = {split}_{label_slug}_{running_index}
  ví dụ: train_safe_000001, val_harmful_000010, test_adult_000123
"""

from .upload_tikharm_to_minio import upload_tikharm_to_minio  # noqa: F401

__all__ = ["upload_tikharm_to_minio"]
