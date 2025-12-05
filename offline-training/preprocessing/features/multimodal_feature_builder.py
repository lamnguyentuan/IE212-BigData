from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Iterable, Dict, Any, Optional

import json
import numpy as np

from .feature_schema import MultimodalFeatureRow


@dataclass
class MultimodalFeatureBuilder:
    """
    Builder cho dataset multimodal từ Silver/Gold.

    Giả định:
      - Silver/{video_id}/video_embedding.npy
      - Silver/{video_id}/audio_embedding.npy
      - Silver/{video_id}/metadata_features.npz
          + numeric_scaled
          + desc_emb
          + tags_emb
          + comments_emb
      - (Optional) Gold/{video_id}/label.json  {"label": 0 or 1}
    """
    base_dir: Path = Path("tiktok-data")
    silver_name: str = "silver"
    gold_name: str = "gold"

    def silver_dir(self) -> Path:
        return self.base_dir / self.silver_name

    def gold_dir(self) -> Path:
        return self.base_dir / self.gold_name

    def list_video_ids(self) -> List[str]:
        sdir = self.silver_dir()
        if not sdir.exists():
            return []
        return sorted(
            p.name
            for p in sdir.iterdir()
            if p.is_dir()
            and (p / "video_embedding.npy").exists()
            and (p / "audio_embedding.npy").exists()
            and (p / "metadata_features.npz").exists()
        )

    def _load_label(self, video_id: str) -> Optional[int]:
        """
        Cố gắng đọc label theo thứ tự ưu tiên:
          1) gold/{video_id}/label.json
          2) silver/{video_id}/label.json
          3) bronze/{video_id}/label.json (nếu bạn muốn thêm sau)
        Nếu không có -> None.
        """
        candidates = [
            self.gold_dir() / video_id / "label.json",
            self.silver_dir() / video_id / "label.json",
        ]
        for p in candidates:
            if p.exists():
                try:
                    obj = json.loads(p.read_text(encoding="utf-8"))
                    val = obj.get("label", None)
                    if val is None:
                        continue
                    return int(val)
                except Exception as e:
                    print(f"[MultimodalFeatureBuilder] Warning: cannot read label from {p}: {e}")
        return None

    def build_row_for_video(self, video_id: str) -> MultimodalFeatureRow:
        """
        Load embedding từng modality cho 1 video_id và build 1 row.
        """
        sdir = self.silver_dir() / video_id

        # Video embedding
        video_emb_path = sdir / "video_embedding.npy"
        audio_emb_path = sdir / "audio_embedding.npy"
        meta_feat_path = sdir / "metadata_features.npz"

        if not video_emb_path.exists():
            raise FileNotFoundError(f"video_embedding.npy not found for {video_id}")
        if not audio_emb_path.exists():
            raise FileNotFoundError(f"audio_embedding.npy not found for {video_id}")
        if not meta_feat_path.exists():
            raise FileNotFoundError(f"metadata_features.npz not found for {video_id}")

        video_emb = np.load(video_emb_path)
        audio_emb = np.load(audio_emb_path)

        meta_npz = np.load(meta_feat_path)
        numeric_scaled = meta_npz["numeric_scaled"]
        # Lấy text embedding: mình chọn comments_emb (phản ánh phản ứng người xem),
        # bạn có thể đổi thành desc_emb nếu thích.
        if "comments_emb" in meta_npz:
            text_emb = meta_npz["comments_emb"]
        elif "desc_emb" in meta_npz:
            text_emb = meta_npz["desc_emb"]
        else:
            raise KeyError("metadata_features.npz must contain comments_emb or desc_emb")

        label = self._load_label(video_id)

        return MultimodalFeatureRow(
            video_id=video_id,
            video_emb=video_emb,
            audio_emb=audio_emb,
            text_emb=text_emb,
            metadata_numeric=numeric_scaled,
            label=label,
        )

    def build_rows(self, video_ids: Optional[Iterable[str]] = None) -> List[MultimodalFeatureRow]:
        """
        Build list MultimodalFeatureRow cho toàn bộ video_ids (hoặc auto detect).
        """
        if video_ids is None:
            video_ids = self.list_video_ids()

        rows: List[MultimodalFeatureRow] = []
        for vid in video_ids:
            try:
                print(f"[MultimodalFeatureBuilder] Building row for video_id={vid}")
                row = self.build_row_for_video(vid)
                rows.append(row)
            except Exception as e:
                print(f"[MultimodalFeatureBuilder] ERROR for {vid}: {e}")
        return rows
