"""
Cross-Modal Fusion Module.

Combines projected features using Attention or Concatenation.
"""

import torch
import torch.nn as nn

class FusionModule(nn.Module):
    def __init__(self, embedding_dim: int, num_modalities: int = 4, dropout: float = 0.1):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_modalities = num_modalities
        
        # Simple implementation: Self-Attention over modalities
        # Input: (Batch, NumModalities, EmbeddingDim)
        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(embedding_dim * num_modalities, embedding_dim),
            nn.GELU(),
            nn.LayerNorm(embedding_dim)
        )

    def forward(self, features_stack: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features_stack: (Batch, NumModalities, EmbeddingDim)
        Returns:
            fused_vector: (Batch, EmbeddingDim)
        """
        # Self-Attention acting as cross-modal interaction
        attn_out, _ = self.attention(features_stack, features_stack, features_stack)
        
        # Residual + Norm
        x = self.norm(features_stack + self.dropout(attn_out))
        
        # Flatten and project down
        # (Batch, NumModalities * EmbeddingDim)
        B = x.size(0)
        x_flat = x.reshape(B, -1)
        
        fused = self.fusion_mlp(x_flat)
        return fused
