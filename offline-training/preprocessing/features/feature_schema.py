from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class MultimodalFeatureRow:
    """
    1 sample multimodal cho 1 video_id.

    - video_emb:   embedding từ TimeSformer  (video/video_encoder_timesformer.py)
    - audio_emb:   embedding từ Wav2Vec2     (audio/audio_encoder_wav2vec.py)
    - text_emb:    text embedding (ví dụ: comments_emb hoặc desc_emb từ metadata)
    - metadata_numeric: numeric_scaled từ metadata (log, tỉ lệ, stats comment, toxic...)
    - label:       Safe / Not Safe (0/1) nếu có, None nếu chưa gán
    """
    video_id: str
    video_emb: np.ndarray
    audio_emb: np.ndarray
    text_emb: np.ndarray
    metadata_numeric: np.ndarray
    label: Optional[int] = None

    def fused_vector(self) -> np.ndarray:
        """
        Trả về vector concat: [video | audio | text | metadata_numeric].
        """
        return np.concatenate(
            [self.video_emb, self.audio_emb, self.text_emb, self.metadata_numeric],
            axis=0,
        )
