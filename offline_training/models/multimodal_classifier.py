"""
Multimodal Classifier.

# CHANGELOG: added cross-attention fusion
Integrates Projections and Cross-Attention Fusion to classify video content.
"""

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

from .projections import ModelProjections
from .fusion import CrossModalFusion

class MultimodalClassifier(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        video_dim: int = 768,    # TimeSformer base
        audio_dim: int = 768,    # Wav2Vec2 base
        text_dim: int = 768,     # PhoBERT base
        meta_dim: int = 25,      # Normalized numeric features
        fusion_dim: int = 768,
        num_classes: int = 4,
        dropout: float = 0.2,
        fusion_heads: int = 8,   # ✨ New
        fusion_layers: int = 2   # ✨ New
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
        
        # ✨ Use CrossModalFusion
        self.fusion = CrossModalFusion(
            embedding_dim=fusion_dim,
            num_heads=fusion_heads,
            num_layers=fusion_layers,
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
        
        # 2. Prepare for Cross-Attention
        # We need Sequence Dimensions. 
        # Since inputs are currently 1D vectors [B, D], we unsqueeze to [B, 1, D].
        
        # Query: Text + Metadata
        q_text = pt.unsqueeze(1) # (B, 1, D)
        q_meta = pm.unsqueeze(1) # (B, 1, D)
        query = torch.cat([q_text, q_meta], dim=1) # (B, 2, D)
        
        # Key/Value: Video + Audio
        k_video = pv.unsqueeze(1) # (B, 1, D)
        k_audio = pa.unsqueeze(1) # (B, 1, D)
        kv = torch.cat([k_video, k_audio], dim=1) # (B, 2, D)
        
        # 3. Fuse
        fused = self.fusion(query_emb=query, kv_emb=kv)
        
        # 4. Classify
        logits = self.classifier_head(fused)
        return logits
