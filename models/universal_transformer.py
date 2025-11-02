#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UniversalTransformer — Scientific Knowledge Learner
---------------------------------------------------
Custom-built transformer for the Humanoid Scientist Brain.
Trains from scratch, no pretrained weights.
Designed for continual online learning and scientific reasoning.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Positional Encoding
# -----------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """Add positional encoding to embeddings."""
        return x + self.pe[:, :x.size(1)]


# -----------------------------
# Multi-Head Self-Attention Block
# -----------------------------
class TransformerBlock(nn.Module):
    def __init__(self, d_model, heads, ff_hidden_mult=4, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=heads, dropout=dropout, batch_first=True)
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_hidden_mult * d_model),
            nn.GELU(),
            nn.Linear(ff_hidden_mult * d_model, d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_out, _ = self.attention(x, x, x, attn_mask=mask, need_weights=False)
        x = self.layernorm1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.layernorm2(x + self.dropout(ff_out))
        return x


# -----------------------------
# Universal Transformer
# -----------------------------
class UniversalTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 1024,
        n_heads: int = 8,
        n_layers: int = 12,
        ff_mult: int = 4,
        dropout: float = 0.1,
        max_len: int = 2048,
        num_classes: int = 2
    ):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = PositionalEncoding(d_model, max_len=max_len)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, ff_hidden_mult=ff_mult, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # Output heads
        self.cls_head = nn.Linear(d_model, num_classes)
        self.projection = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        # x: (batch, seq_len)
        x = self.token_embed(x)
        x = self.pos_embed(x)
        for layer in self.layers:
            x = layer(x, mask)
        x = self.norm(x)
        cls_token = x[:, 0]  # Use first token for global understanding
        logits = self.cls_head(cls_token)
        embeddings = self.projection(cls_token)
        return logits, embeddings


# -----------------------------
# Model Initialization Helper
# -----------------------------
def build_universal_transformer(vocab_size: int) -> UniversalTransformer:
    model = UniversalTransformer(
        vocab_size=vocab_size,
        d_model=1024,   # deep scientific representation
        n_heads=8,
        n_layers=12,
        ff_mult=4,
        dropout=0.1,
        max_len=2048,
        num_classes=2,
    )
    print(f"🧠 Built UniversalTransformer: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")
    return model


if __name__ == "__main__":
    model = build_universal_transformer(vocab_size=32000)
    sample = torch.randint(0, 32000, (2, 512))
    logits, emb = model(sample)
    print("Output:", logits.shape, emb.shape)
