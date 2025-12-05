from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Iterable, Dict, Any

import numpy as np

from .feature_schema import MultimodalFeatureRow


@dataclass
class FeatureSaver:
    """
    Lưu dataset multimodal thành 1 file .npz duy nhất ở GOLD layer.

    - base_dir: thư mục gốc data (vd: tiktok-data)
    - gold_name: tên folder gold (mặc định: 'gold')
    """

    base_dir: Path = Path("tiktok-data")
    gold_name: str = "gold"

    def gold_dir(self) -> Path:
        return self.base_dir / self.gold_name

    def save_npz(
        self,
        dataset_name: str,
        rows: Iterable[MultimodalFeatureRow],
    ) -> Path:
        """
        Lưu dataset multimodal thành:
            gold/{dataset_name}.npz

        Trong file .npz gồm:
            - video_ids: array shape (N,)
            - video_emb: shape (N, D_v)
            - audio_emb: shape (N, D_a)
            - text_emb:  shape (N, D_t)
            - metadata_numeric: shape (N, F)
            - fused:     shape (N, D_v + D_a + D_t + F)
            - labels:    shape (N,) hoặc toàn -1 nếu không có
        """
        rows = list(rows)
        if not rows:
            raise ValueError("No rows to save in FeatureSaver.save_npz")

        video_ids = np.array([r.video_id for r in rows])
        video_emb = np.stack([r.video_emb for r in rows], axis=0)
        audio_emb = np.stack([r.audio_emb for r in rows], axis=0)
        text_emb = np.stack([r.text_emb for r in rows], axis=0)
        metadata_numeric = np.stack([r.metadata_numeric for r in rows], axis=0)
        fused = np.stack([r.fused_vector() for r in rows], axis=0)

        labels_list: List[int] = []
        for r in rows:
            if r.label is None:
                labels_list.append(-1)
            else:
                labels_list.append(int(r.label))
        labels = np.array(labels_list, dtype=np.int64)

        out_dir = self.gold_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{dataset_name}.npz"

        np.savez_compressed(
            out_path,
            video_ids=video_ids,
            video_emb=video_emb,
            audio_emb=audio_emb,
            text_emb=text_emb,
            metadata_numeric=metadata_numeric,
            fused=fused,
            labels=labels,
        )
        print(f"[FeatureSaver] Saved dataset to {out_path}")
        return out_path
