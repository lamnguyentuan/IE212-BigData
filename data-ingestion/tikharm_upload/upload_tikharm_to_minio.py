"""
Upload TikHarm dataset lên MinIO (không tạo bản preprocessed local)

Dataset gốc (local):

offline-training/datasets/TikHarm/
├── train/
│   ├── Adult Content/
│   ├── Harmful Content/
│   ├── Safe/
│   └── Suicide/
├── val/
└── test/

Script này sẽ:
- Duyệt qua các split: train / val / test
- Duyệt qua các label folder: "Adult Content", "Harmful Content", "Safe", "Suicide"
- Với mỗi file video, sinh ra một video_id theo format:
    {split}_{label_slug}_{running_index}
  VD:
    train_safe_000001, val_adult_000010, test_harmful_000123, ...
- Upload trực tiếp lên MinIO (bucket trong config_tikharm.yaml) với layout:

tikharm/
└── bronze/
    └── {video_id}/
        ├── video.mp4
        └── metadata.json

Trong đó metadata.json chứa:
- video_id
- split
- label_raw (vd: "Adult Content")
- label (slug: adult/harmful/safe/suicide)
- original_filename
- original_path (tương đối so với TikHarm root)

Không tạo thêm bất kỳ thư mục preprocessed nào trên local.
"""

from __future__ import annotations

import io
import json
import logging
import mimetypes
import os
import sys
from pathlib import Path
from typing import Dict, List

# Thiết lập ROOT = thư mục IE212-BigData/
ROOT = Path(__file__).resolve().parents[2]

# Cho phép import minio_client từ thư mục minio/
sys.path.append(str(ROOT / "minio"))
from minio_client import get_minio_client  # type: ignore

# Thư mục TikHarm gốc trên local
SRC_ROOT = ROOT / "offline-training" / "datasets" / "TikHarm"

# Các split và mapping label
SPLITS: List[str] = ["train", "val", "test"]

LABEL_SLUGS: Dict[str, str] = {
    "Adult Content": "adult",
    "Harmful Content": "harmful",
    "Safe": "safe",
    "Suicide": "suicide",
}

# Định dạng video hợp lệ (nếu dataset chỉ có .mp4 thì vẫn ok)
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".flv", ".webm"}


def setup_logging() -> None:
    """Thiết lập logging cho quá trình upload."""
    log_dir = ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_dir / "tikharm_upload.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def validate_local_root(root: Path) -> None:
    """Kiểm tra thư mục gốc TikHarm có tồn tại không."""
    if not root.exists():
        raise FileNotFoundError(
            f"Không tìm thấy thư mục TikHarm gốc: {root}\n"
            f"Hãy chắc chắn dataset nằm ở offline-training/datasets/TikHarm"
        )


def upload_tikharm_to_minio(config_name: str = "config_tikharm.yaml") -> None:
    """
    Tiền xử lý + upload TikHarm trực tiếp lên MinIO.

    - Không ghi ra thư mục preprocessed local
    - Mỗi video ứng với một "video_id" duy nhất
    - MinIO layout:
        bronze/{video_id}/video.mp4
        bronze/{video_id}/metadata.json
    """
    client, bucket = get_minio_client(config_name)
    logging.info(f"[MINIO] Using bucket: {bucket}")
    logging.info(f"[SRC_ROOT] {SRC_ROOT}")

    # Duyệt lần lượt các split: train / val / test
    for split in SPLITS:
        split_dir = SRC_ROOT / split
        if not split_dir.exists():
            logging.warning(f"[SKIP] Split folder not found: {split_dir}")
            continue

        logging.info(f"=== Processing split: {split} ===")

        # Bộ đếm cho từng label_slug để tạo index 000001, 000002, ...
        counters: Dict[str, int] = {slug: 0 for slug in LABEL_SLUGS.values()}

        # Duyệt các thư mục label: "Adult Content", "Safe", ...
        for label_dir in split_dir.iterdir():
            if not label_dir.is_dir():
                continue

            label_raw = label_dir.name
            label_slug = LABEL_SLUGS.get(label_raw)

            if label_slug is None:
                logging.warning(f"[SKIP] Unknown label folder: {label_raw}")
                continue

            logging.info(f"  Label '{label_raw}' -> slug '{label_slug}'")

            # Duyệt tất cả file video bên trong label_dir (đệ quy)
            for f in label_dir.rglob("*"):
                if not f.is_file():
                    continue

                if f.suffix.lower() not in VIDEO_EXTS:
                    continue

                counters[label_slug] += 1
                idx = counters[label_slug]

                video_id = f"{split}_{label_slug}_{idx:06d}"
                logging.info(f"    [{label_slug}] #{idx} -> video_id={video_id}")

                # -------------------------
                # 1) Upload video
                # -------------------------
                object_video = f"bronze/{video_id}/video.mp4"

                content_type, _ = mimetypes.guess_type(str(f))
                if content_type is None:
                    content_type = "application/octet-stream"

                logging.info(
                    f"       Upload video: {f} -> s3://{bucket}/{object_video}"
                )
                client.fput_object(
                    bucket_name=bucket,
                    object_name=object_video,
                    file_path=str(f),
                    content_type=content_type,
                )

                # -------------------------
                # 2) Tạo metadata và upload metadata.json
                # -------------------------
                rel_path = f.relative_to(SRC_ROOT)

                metadata = {
                    "video_id": video_id,
                    "split": split,
                    "label_raw": label_raw,
                    "label": label_slug,
                    "original_filename": f.name,
                    "original_path": str(rel_path),
                    "source": "TikHarm",
                }

                metadata_bytes = json.dumps(
                    metadata, indent=2, ensure_ascii=False
                ).encode("utf-8")
                metadata_stream = io.BytesIO(metadata_bytes)

                object_meta = f"bronze/{video_id}/metadata.json"
                logging.info(
                    f"       Upload metadata.json -> s3://{bucket}/{object_meta}"
                )
                client.put_object(
                    bucket_name=bucket,
                    object_name=object_meta,
                    data=metadata_stream,
                    length=len(metadata_bytes),
                    content_type="application/json",
                )

        logging.info(f"=== DONE split: {split} ===")

    logging.info("=== FINISHED TikHarm upload to MinIO (bronze) ===")


def main() -> None:
    setup_logging()
    validate_local_root(SRC_ROOT)
    upload_tikharm_to_minio(config_name="config_tikharm.yaml")


if __name__ == "__main__":
    main()
