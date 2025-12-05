from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import os

from minio import Minio  # pip install minio

from .constants import (
    DEFAULT_MINIO_BUCKET,
    ENV_MINIO_ACCESS_KEY,
    ENV_MINIO_ENDPOINT,
    ENV_MINIO_SECRET_KEY,
    ENV_MINIO_SECURE,
)
from .file_io import ensure_dir
from .logging_utils import get_logger


@dataclass
class MinioConfig:
    """
    Cấu hình MinIO. Có thể lấy từ env hoặc truyền tay.

    - endpoint: ví dụ "localhost:9000"
    - access_key / secret_key: credentials
    - secure: False nếu chạy HTTP, True nếu HTTPS
    - bucket: tên bucket (vd: "tiktok-data")
    """
    endpoint: str
    access_key: str
    secret_key: str
    secure: bool = False
    bucket: str = DEFAULT_MINIO_BUCKET

    @classmethod
    def from_env(cls, bucket: Optional[str] = None) -> "MinioConfig":
        endpoint = os.getenv(ENV_MINIO_ENDPOINT, "localhost:9000")
        access_key = os.getenv(ENV_MINIO_ACCESS_KEY, "minioadmin")
        secret_key = os.getenv(ENV_MINIO_SECRET_KEY, "minioadmin")
        secure_str = os.getenv(ENV_MINIO_SECURE, "0")
        secure = secure_str not in ("0", "false", "False", "no", "")

        return cls(
            endpoint=endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure,
            bucket=bucket or DEFAULT_MINIO_BUCKET,
        )


@dataclass
class MinioClientWrapper:
    """
    Wrapper mỏng cho MinIO client, tập trung cho use-case Medallion:

    - bronze/{video_id}/video.mp4
    - bronze/{video_id}/metadata.json
    - silver/{video_id}/...
    - gold/{video_id}/...

    Sử dụng:
        cfg = MinioConfig.from_env(bucket="tiktok-data")
        mc = MinioClientWrapper(cfg)

        mc.download_object("bronze/123/video.mp4", "tiktok-data/bronze/123/video.mp4")
    """
    config: MinioConfig
    _client: Minio = field(init=False)
    _logger_name: str = "minio"

    def __post_init__(self):
        self._client = Minio(
            endpoint=self.config.endpoint,
            access_key=self.config.access_key,
            secret_key=self.config.secret_key,
            secure=self.config.secure,
        )
        self.logger = get_logger(self._logger_name)

    # -------- low-level API --------

    def download_object(self, object_name: str, local_path: Path) -> None:
        """
        Download object từ bucket về local path.

        object_name ví dụ: "bronze/123456/video.mp4"
        """
        ensure_dir(local_path.parent)
        self.logger.info(f"[MinIO] GET s3://{self.config.bucket}/{object_name} -> {local_path}")
        self._client.fget_object(
            bucket_name=self.config.bucket,
            object_name=object_name,
            file_path=str(local_path),
        )

    def upload_file(self, local_path: Path, object_name: str) -> None:
        """
        Upload local file lên bucket.

        object_name ví dụ: "silver/123456/audio.wav"
        """
        self.logger.info(f"[MinIO] PUT {local_path} -> s3://{self.config.bucket}/{object_name}")
        self._client.fput_object(
            bucket_name=self.config.bucket,
            object_name=object_name,
            file_path=str(local_path),
        )

    # -------- convenience methods cho Medallion --------

    def download_bronze_video(self, video_id: str, data_root: Path) -> Path:
        """
        Download:
          - bronze/{video_id}/video.mp4
          - bronze/{video_id}/metadata.json

        về local:
          data_root/bronze/{video_id}/...
        """
        bronze_dir = data_root / "bronze" / video_id
        ensure_dir(bronze_dir)

        video_obj = f"bronze/{video_id}/video.mp4"
        meta_obj = f"bronze/{video_id}/metadata.json"

        video_path = bronze_dir / "video.mp4"
        meta_path = bronze_dir / "metadata.json"

        try:
            self.download_object(video_obj, video_path)
        except Exception as e:
            self.logger.error(f"[MinIO] Failed to download {video_obj}: {e}")
            raise

        try:
            self.download_object(meta_obj, meta_path)
        except Exception as e:
            self.logger.error(f"[MinIO] Failed to download {meta_obj}: {e}")
            # metadata có thể thiếu, tuỳ bạn muốn raise hay chỉ warning
            raise

        return bronze_dir

    def upload_silver_dir(self, video_id: str, data_root: Path) -> None:
        """
        Upload toàn bộ nội dung silver/{video_id}/ lên MinIO dưới prefix:
          silver/{video_id}/...
        """
        silver_dir = data_root / "silver" / video_id
        if not silver_dir.exists():
            self.logger.warning(f"[MinIO] silver dir not found: {silver_dir}")
            return

        for path in silver_dir.rglob("*"):
            if path.is_dir():
                continue
            rel = path.relative_to(data_root)
            object_name = str(rel).replace("\\", "/")  # Windows compat
            self.upload_file(path, object_name)
