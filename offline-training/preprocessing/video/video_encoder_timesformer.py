from __future__ import annotations
from dataclasses import dataclass
import torch
import numpy as np
from transformers import TimesformerModel, TimesformerConfig


@dataclass
class TimeSformerVideoEncoder:
    """
    Encoder video dùng TimeSformer (HuggingFace).
    Input: tensor (T, C, H, W)
    Output: embedding vector (H,)
    """
    model_name: str = "facebook/timesformer-base-finetuned-k400"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __post_init__(self):
        self.model = TimesformerModel.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        self.hidden_size = self.model.config.hidden_size

    @torch.no_grad()
    def encode(self, frames: torch.Tensor) -> np.ndarray:
        """
        frames shape: (T, C, H, W)
        HuggingFace muốn input shape: (B, T, C, H, W)
        """
        frames = frames.unsqueeze(0).to(self.device)
        out = self.model(pixel_values=frames)

        # out.last_hidden_state: (B, T*patches, H)
        # Pooling: mean hoặc lấy CLS token nếu có.
        emb = out.last_hidden_state.mean(dim=1)[0]  # (H,)
        return emb.cpu().numpy().astype(np.float32)
