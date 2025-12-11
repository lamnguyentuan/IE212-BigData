from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import numpy as np
import os

from common.features.feature_schema import MultimodalFeatureRow
from common.utils.minio_utils import MinioConfig, MinioClientWrapper   # ✨ import MinIO wrapper


@dataclass
class FeatureSaver:
    base_dir: Path = Path("tiktok-data")
    gold_name: str = "gold"

    # các flag cho MinIO (đọc từ paths.yaml hoặc env)
    use_minio: bool = False
    upload_gold: bool = False
    minio_bucket: Optional[str] = None

    def gold_dir(self) -> Path:
        return self.base_dir / self.gold_name

    def save_npz(self, dataset_name: str, rows: List[MultimodalFeatureRow]) -> Path:
        if not rows:
            raise ValueError("No rows to save in FeatureSaver.save_npz")

        gold_dir = self.gold_dir()
        gold_dir.mkdir(parents=True, exist_ok=True)

        out_path = gold_dir / f"{dataset_name}.npz"

        # gộp field
        video_ids = [r.video_id for r in rows]
        video_embs = np.stack([r.video_emb for r in rows], axis=0)
        audio_embs = np.stack([r.audio_emb for r in rows], axis=0)
        text_embs = np.stack([r.text_emb for r in rows], axis=0)
        metadata_numeric = np.stack([r.metadata_numeric for r in rows], axis=0)
        labels = np.array([r.label for r in rows], dtype=np.int64)

        np.savez_compressed(
            out_path,
            video_ids=np.array(video_ids),
            video_embs=video_embs,
            audio_embs=audio_embs,
            text_embs=text_embs,
            metadata_numeric=metadata_numeric,
            labels=labels,
        )

        # ✨ Upload lên MinIO nếu bật
        if self.use_minio and self.upload_gold and self.minio_bucket:
            cfg = MinioConfig.from_env(bucket=self.minio_bucket)
            client = MinioClientWrapper(cfg)
            remote_key = f"{self.gold_name}/{out_path.name}"
            print(f"[FeatureSaver] [MinIO] PUT {out_path} -> s3://{self.minio_bucket}/{remote_key}")
            client.upload_file(out_path, remote_key)

        return out_path
