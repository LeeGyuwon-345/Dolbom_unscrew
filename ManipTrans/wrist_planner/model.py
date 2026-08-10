"""High-level Wrist Planner — Transformer BC (Chen et al. 2024, obj-dex).

객체 목표궤적 window G_{t:t+T} (각 프레임 18차원) → 양손 손목궤적 window a^W_{t:t+T} (18차원).
인코더-온리 Transformer(입출력 window 동일 길이) + 태스크(카테고리) 임베딩.
"""
import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=64):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class WristPlannerTransformer(nn.Module):
    def __init__(self, obj_dim=18, wrist_dim=18, d_model=256, nhead=4,
                 layers=4, ff=512, n_cat=1, dropout=0.0):
        super().__init__()
        self.in_proj = nn.Linear(obj_dim, d_model)
        self.cat_emb = nn.Embedding(n_cat, d_model)
        self.pos = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, ff, dropout, activation="gelu", batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, wrist_dim)
        )

    def forward(self, obj, cat):
        # obj (B,T,obj_dim), cat (B,)
        h = self.in_proj(obj) + self.cat_emb(cat).unsqueeze(1)
        h = self.pos(h)
        h = self.encoder(h)
        return self.head(h)  # (B,T,wrist_dim)
