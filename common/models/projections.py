"""
Feature Projection Modules.

Projects diverse input embedding dimensions to a common fusion dimension.
"""

import torch
import torch.nn as nn
from typing import Optional

class FeatureProjector(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.LayerNorm(output_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class ModelProjections(nn.Module):
    """
    Holds projectors for all modalities.
    """
    def __init__(
        self,
        video_dim: int,
        audio_dim: int,
        text_dim: int,
        meta_dim: int,
        fusion_dim: int = 768,
        dropout: float = 0.1
    ):
        super().__init__()
        self.video_proj = FeatureProjector(video_dim, fusion_dim, dropout)
        self.audio_proj = FeatureProjector(audio_dim, fusion_dim, dropout)
        self.text_proj = FeatureProjector(text_dim, fusion_dim, dropout)
        self.meta_proj = FeatureProjector(meta_dim, fusion_dim, dropout)
        
    def forward(
        self,
        video_emb: torch.Tensor,
        audio_emb: torch.Tensor,
        text_emb: torch.Tensor,
        meta_num: torch.Tensor
    ):
        """
        Projects all inputs to (Batch, fusion_dim).
        """
        v = self.video_proj(video_emb)
        a = self.audio_proj(audio_emb)
        t = self.text_proj(text_emb)
        m = self.meta_proj(meta_num)
        return v, a, t, m
