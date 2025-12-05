from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any

import json
import numpy as np

from .feature_schema import MultimodalFeatureRow


@dataclass
class MultimodalFeatureBuilder:
    """
    Build multimodal feature rows từ các file đã được preprocess trong SILVER.

    Giả định cấu trúc:

      base_dir/
        ├── bronze/
        │     └── {video_id}/metadata.json      # TikHarm / tiktok-data metadata
        ├── silver/
        │     └── {video_id}/
        │           ├── audio_embedding.npy
        │           ├── video_embedding.npy
        │           └── metadata_features.npz   # output từ MetadataPreprocessor
        └── gold/
              └── *.npz (output của FeatureSaver)

    Label space chung (4 lớp):

        0 = safe
        1 = adult content
        2 = harmful content (generic / not safe)
        3 = suicide / self-harm
    """

    base_dir: Path = Path("tiktok-data")
    silver_name: str = "silver"
    gold_name: str = "gold"
    bronze_name: str = "bronze"   # 💥 QUAN TRỌNG: phải có field này

    def __post_init__(self) -> None:
        # Đảm bảo base_dir là Path
        if not isinstance(self.base_dir, Path):
            self.base_dir = Path(self.base_dir)

    # ----------------- convenience dirs -----------------

    def silver_dir(self) -> Path:
        return self.base_dir / self.silver_name

    def gold_dir(self) -> Path:
        return self.base_dir / self.gold_name

    # ----------------- low-level loader helpers -----------------

    def _load_npy(self, path: Path) -> Optional[np.ndarray]:
        if not path.exists():
            return None
        try:
            return np.load(path)
        except Exception as e:
            print(f"[MultimodalFeatureBuilder] WARNING: cannot load npy {path}: {e}")
            return None

    def _load_npz(self, path: Path) -> Optional[Dict[str, Any]]:
        if not path.exists():
            return None
        try:
            data = np.load(path)
            return {k: data[k] for k in data.files}
        except Exception as e:
            print(f"[MultimodalFeatureBuilder] WARNING: cannot load npz {path}: {e}")
            return None

    # ----------------- label handling -----------------

    def _map_label_string_to_id(self, label_str: str) -> int:
        """
        Map string label từ TikHarm / tiktok-data về id 0..3.
        """
        s = label_str.strip().lower()

        # SAFE
        if s in ["safe"]:
            return 0

        # ADULT CONTENT
        if s in ["adult", "adult content", "adult-content"]:
            return 1

        # SUICIDE / SELF-HARM
        if s in ["suicide", "self-harm", "self harm", "suicidal"]:
            return 3

        # HARMFUL (generic, bao gồm not safe)
        if s in ["harmful", "harmful content", "not safe", "unsafe"]:
            return 2

        # Default: harmful generic
        return 2

    def _load_label(self, video_id: str) -> Optional[int]:
        """
        Ưu tiên label theo thứ tự:

          1) gold/{video_id}/label.json      (nếu bạn tạo tay)
          2) silver/{video_id}/label.json    (nếu có)
          3) bronze/{video_id}/metadata.json (TikHarm / tiktok-data)
        """
        # 1) gold/silver label.json
        gold_label = self.gold_dir() / video_id / "label.json"
        silver_label = self.silver_dir() / video_id / "label.json"
        for p in [gold_label, silver_label]:
            if p.exists():
                try:
                    obj = json.loads(p.read_text(encoding="utf-8"))
                    val = obj.get("label", None)
                    if val is None:
                        continue
                    if isinstance(val, int):
                        return val
                    return self._map_label_string_to_id(str(val))
                except Exception as e:
                    print(f"[MultimodalFeatureBuilder] WARNING: cannot parse label from {p}: {e}")

        # 2) metadata trong bronze/{video_id}/metadata.json
        meta_path = self.base_dir / self.bronze_name / video_id / "metadata.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                raw_label = meta.get("label") or meta.get("label_raw")
                if raw_label is None:
                    return None
                return self._map_label_string_to_id(str(raw_label))
            except Exception as e:
                print(f"[MultimodalFeatureBuilder] WARNING: cannot parse label from {meta_path}: {e}")
                return None

        return None

    # ----------------- build single row -----------------

    def _build_single_row(self, video_id: str) -> Optional[MultimodalFeatureRow]:
        """
        Build 1 MultimodalFeatureRow từ silver/{video_id} và bronze/{video_id}/metadata.json.
        """
        silver_vid_dir = self.silver_dir() / video_id

        video_emb_path = silver_vid_dir / "video_embedding.npy"
        audio_emb_path = silver_vid_dir / "audio_embedding.npy"
        meta_feat_path = silver_vid_dir / "metadata_features.npz"

        if not video_emb_path.exists():
            print(f"[MultimodalFeatureBuilder] WARNING: missing video_embedding for {video_id}")
            return None
        if not audio_emb_path.exists():
            print(f"[MultimodalFeatureBuilder] WARNING: missing audio_embedding for {video_id}")
            return None
        if not meta_feat_path.exists():
            print(f"[MultimodalFeatureBuilder] WARNING: missing metadata_features.npz for {video_id}")
            return None

        video_emb = self._load_npy(video_emb_path)
        audio_emb = self._load_npy(audio_emb_path)
        meta_dict = self._load_npz(meta_feat_path)

        if video_emb is None or audio_emb is None or meta_dict is None:
            print(f"[MultimodalFeatureBuilder] WARNING: failed to load embeddings/meta for {video_id}")
            return None

        numeric_scaled = meta_dict.get("numeric_scaled")
        comments_emb = meta_dict.get("comments_emb")

        if numeric_scaled is None or comments_emb is None:
            print(f"[MultimodalFeatureBuilder] WARNING: invalid keys in metadata_features.npz for {video_id}")
            return None

        text_emb = comments_emb
        label = self._load_label(video_id)
        if label is None:
            label = -1

        row = MultimodalFeatureRow(
            video_id=video_id,
            video_emb=video_emb.astype(np.float32),
            audio_emb=audio_emb.astype(np.float32),
            text_emb=text_emb.astype(np.float32),
            metadata_numeric=numeric_scaled.astype(np.float32),
            label=int(label),
        )
        return row

    # ----------------- public API -----------------

    def build_rows(self, video_ids: List[str]) -> List[MultimodalFeatureRow]:
        rows: List[MultimodalFeatureRow] = []
        for vid in video_ids:
            print(f"[MultimodalFeatureBuilder] Building row for video_id={vid}")
            try:
                row = self._build_single_row(vid)
                if row is not None:
                    rows.append(row)
            except Exception as e:
                print(f"[MultimodalFeatureBuilder] ERROR for {vid}: {e}")
        return rows
