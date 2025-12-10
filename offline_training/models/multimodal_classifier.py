"""
Multimodal Classifier.

Integrates Projections and Fusion to classify video content.
"""

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin # ✨ Import Mixin

from .projections import ModelProjections
from .fusion import FusionModule

class MultimodalClassifier(nn.Module, PyTorchModelHubMixin): # ✨ Inherit
    def __init__(
        self,
        video_dim: int = 768,    # TimeSformer base
        audio_dim: int = 768,    # Wav2Vec2 base
        text_dim: int = 768,     # PhoBERT base
        meta_dim: int = 25,      # Normalized numeric features
        fusion_dim: int = 768,
        num_classes: int = 4,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.projections = ModelProjections(
            video_dim=video_dim,
            audio_dim=audio_dim,
            text_dim=text_dim,
            meta_dim=meta_dim,
            fusion_dim=fusion_dim,
            dropout=dropout
        )
        
        self.fusion = FusionModule(
            embedding_dim=fusion_dim,
            num_modalities=4, # V, A, T, M
            dropout=dropout
        )
        
        self.classifier_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.GELU(),
            nn.Linear(fusion_dim // 2, num_classes)
        )

    def forward(
        self,
        video_emb: torch.Tensor,
        audio_emb: torch.Tensor,
        text_emb: torch.Tensor,
        meta_num: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            video_emb: (B, video_dim)
            audio_emb: (B, audio_dim)
            text_emb: (B, text_dim)
            meta_num: (B, meta_dim)
        Returns:
            logits: (B, num_classes)
        """
        # 1. Project to common dim
        pv, pa, pt, pm = self.projections(video_emb, audio_emb, text_emb, meta_num)
        
        # 2. Stack inputs for fusion: (B, 4, fusion_dim)
        stack = torch.stack([pv, pa, pt, pm], dim=1)
        
        # 3. Fuse
        fused = self.fusion(stack)
        
        # 4. Classify
        logits = self.classifier_head(fused)
        return logits
