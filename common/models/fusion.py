"""
Cross-Modal Fusion Module.

# CHANGELOG: added cross-attention fusion
Combines projected features using Cross-Attention.
Text/Metadata acts as Query, Video/Audio as Key/Value.
"""

import torch
import torch.nn as nn

class CrossModalFusion(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # Cross-Attention Layer: Query=Text+Meta, Key/Value=Video+Audio
        # We wrap it in a TransformerDecoder-like structure or just stack MultiheadAttention
        
        # Using a custom block for clarity:
        # Layer 1: CA -> Add&Norm -> FFN -> Add&Norm
        self.layers = nn.ModuleList([
            CrossAttentionLayer(embedding_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Pooling / Output projection
        self.norm_out = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        query_emb: torch.Tensor, # (Batch, SeqQ, Dim) - Text/Meta
        kv_emb: torch.Tensor,    # (Batch, SeqKV, Dim) - Video/Audio
        key_padding_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            query_emb: Text/Metadata features acting as Query.
            kv_emb: Video/Audio features acting as Key/Value.
        Returns:
            fused_vector: (Batch, Dim) - Pooled representation.
        """
        x = query_emb
        
        for layer in self.layers:
            x = layer(x, kv_emb, key_padding_mask)
            
        x = self.norm_out(x)
        
        # Pool: simple mean pooling over the Query sequence (Text)
        # Assuming x is (Batch, SeqQ, Dim)
        # If padded, we should be careful, but here we assume inputs are mostly valid
        fused = x.mean(dim=1) 
        
        return fused

class CrossAttentionLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_model * 4, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, q: torch.Tensor, k: torch.Tensor, key_padding_mask: torch.Tensor = None):
        # Cross Attention
        # q: (B, Sq, D), k: (B, Sk, D)
        attn_out, _ = self.attn(q, k, k, key_padding_mask=key_padding_mask)
        x = q + self.dropout1(attn_out)
        x = self.norm1(x)
        
        # FFN
        ff_out = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = x + self.dropout2(ff_out)
        x = self.norm2(x)
        
        return x
