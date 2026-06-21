"""
hybrid_cnn_transformer.py — Hibrit CNN-Transformer modeli (Ana Katkı).

Mimari (HCTC + Ansari et al. referans alınarak tasarlandı):
  Stage A — CNN        : Lokal / kısa vadeli özellik çıkarımı
  Stage B — Transformer: Global / uzun vadeli bağımlılık modelleme
  Stage C — FC         : Sınıflandırma başlığı

Giriş: ham I/Q verisi (batch, 2, 1024)
Çıkış: sınıf olasılıkları (batch, 24)
"""

import math
import torch
import torch.nn as nn
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ── Stage A: CNN Feature Extractor ───────────────────────────────────────────

class ConvBlock(nn.Module):
    """Conv1d + BatchNorm + ReLU + Dropout."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int, dropout: float = 0.3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel,
                      padding=kernel // 2, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )

    def forward(self, x):
        return self.block(x)


class CNNFeatureExtractor(nn.Module):
    """
    Stage A: Ham I/Q → lokal özellik haritası.
    Çıkış shape: (batch, d_model, seq_len) — Transformer'a beslenecek
    """

    def __init__(
        self,
        in_channels: int  = config.INPUT_CHANNELS,
        filters: list     = config.HYBRID_CNN_FILTERS,
        kernels: list     = config.HYBRID_CNN_KERNELS,
        dropout: float    = config.HYBRID_CNN_DROPOUT,
        d_model: int      = config.TRANSFORMER_D_MODEL,
    ):
        super().__init__()
        assert len(filters) == len(kernels), "filters ve kernels aynı uzunlukta olmalı"

        layers = []
        in_ch = in_channels
        for out_ch, k in zip(filters, kernels):
            layers.append(ConvBlock(in_ch, out_ch, kernel=k, dropout=dropout))
            in_ch = out_ch
        self.conv_stack = nn.Sequential(*layers)

        # Son filtre sayısını d_model'e projeksiyon
        self.proj = nn.Conv1d(filters[-1], d_model, kernel_size=1)

        # Boyutu küçült — uzun dizi Transformer için pahalı
        self.pool = nn.AvgPool1d(kernel_size=8, stride=8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 2, 1024)
        feat = self.conv_stack(x)   # (B, filters[-1], 1024)
        feat = self.proj(feat)      # (B, d_model, 1024)
        feat = self.pool(feat)      # (B, d_model, 128)
        return feat


# ── Stage B: Transformer Encoder ─────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    """Sinüs tabanlı konumsal kodlama (Vaswani et al. 2017)."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        pe = pe.unsqueeze(0)   # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, seq_len, d_model)
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TransformerEncoder(nn.Module):
    """
    Stage B: Çok başlı öz dikkat (multi-head self-attention) ile
    uzun vadeli bağımlılıkları yakala.
    """

    def __init__(
        self,
        d_model:  int   = config.TRANSFORMER_D_MODEL,
        nhead:    int   = config.TRANSFORMER_NHEAD,
        dim_ff:   int   = config.TRANSFORMER_DIM_FF,
        n_layers: int   = config.TRANSFORMER_LAYERS,
        dropout:  float = config.TRANSFORMER_DROPOUT,
    ):
        super().__init__()
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,   # (B, seq, d_model) formatı
            norm_first=True,    # Pre-LN — eğitim kararlılığı için
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, d_model, seq_len)
        x = x.permute(0, 2, 1)   # → (B, seq_len, d_model)
        x = self.pos_enc(x)
        x = self.encoder(x)
        x = self.norm(x)
        # Global ortalama — sabit boyutlu temsil
        x = x.mean(dim=1)         # → (B, d_model)
        return x


# ── Stage C: Classifier Head ──────────────────────────────────────────────────

class ClassifierHead(nn.Module):
    """Stage C: Tam bağlantılı sınıflandırma katmanları."""

    def __init__(
        self,
        d_model:     int   = config.TRANSFORMER_D_MODEL,
        fc_dim_1:    int   = config.FC_DIM_1,
        fc_dim_2:    int   = config.FC_DIM_2,
        num_classes: int   = config.NUM_CLASSES,
        dropout:     float = config.FC_DROPOUT,
    ):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_model, fc_dim_1),
            nn.BatchNorm1d(fc_dim_1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc_dim_1, fc_dim_2),
            nn.BatchNorm1d(fc_dim_2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc_dim_2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


# ── Ana Model ─────────────────────────────────────────────────────────────────

class HybridCNNTransformer(nn.Module):
    """
    Hibrit CNN-Transformer Modeli (Ana Katkı).

    Giriş  : (batch, 2, 1024) — ham I/Q verisi
    Çıkış  : (batch, 24)      — sınıf log-olasılıkları (Softmax dışarıda)
    """

    def __init__(self):
        super().__init__()
        self.stage_a = CNNFeatureExtractor()
        self.stage_b = TransformerEncoder()
        self.stage_c = ClassifierHead()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.stage_a(x)   # (B, d_model, seq_len)
        feat = self.stage_b(feat) # (B, d_model)
        out  = self.stage_c(feat) # (B, num_classes)
        return out


# ── Transformer-Only Model ────────────────────────────────────────────────────

class TransformerOnly(nn.Module):
    """
    CNN aşaması olmadan ham I/Q verisini doğrudan Transformer'a besler.

    Giriş  : (batch, 2, 1024) — ham I/Q verisi
    Çıkış  : (batch, 24)      — sınıf log-olasılıkları (Softmax dışarıda)

    Boru hattı:
      (B, 2, 1024)
        → Conv1d(kernel=8, stride=8) → (B, d_model, 128)   # downsample
        → permute → (B, 128, d_model)
        → Positional Encoding
        → Transformer Encoder
        → Global Average Pooling → (B, d_model)
        → FC Head → (B, 24)
    """

    def __init__(
        self,
        in_channels: int   = config.INPUT_CHANNELS,
        d_model:     int   = config.TRANSFORMER_D_MODEL,
        nhead:       int   = config.TRANSFORMER_NHEAD,
        n_layers:    int   = config.TRANSFORMER_LAYERS,
        dim_ff:      int   = config.TRANSFORMER_DIM_FF,
        dropout:     float = config.TRANSFORMER_DROPOUT,
    ):
        super().__init__()
        # Ham I/Q'yu d_model kanalına projekle ve aynı anda 8x downsample et
        # (B, 2, 1024) → (B, d_model, 128)
        self.input_proj = nn.Conv1d(in_channels, d_model, kernel_size=8, stride=8)

        # Dizi uzunluğu 1024/8 = 128'e indiği için max_len=128
        self.pos_enc = PositionalEncoding(
            d_model, max_len=config.SIGNAL_LENGTH // 8, dropout=dropout
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

        self.head = ClassifierHead(d_model=d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 2, 1024)
        x = self.input_proj(x)   # → (B, d_model, 128)
        x = x.permute(0, 2, 1)   # → (B, 128, d_model)
        x = self.pos_enc(x)      # positional encoding
        x = self.encoder(x)      # transformer encoder
        x = self.norm(x)
        x = x.mean(dim=1)        # global average pooling → (B, d_model)
        return self.head(x)      # → (B, num_classes)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    x = torch.randn(4, config.INPUT_CHANNELS, config.SIGNAL_LENGTH)

    hybrid = HybridCNNTransformer()
    print(f"Hybrid CNN-Transformer — Parametre sayisi: {count_parameters(hybrid):,}")
    print(f"Giris: {x.shape}  →  Cikis: {hybrid(x).shape}")

    feat_a = hybrid.stage_a(x)
    print(f"Stage A cikisi: {feat_a.shape}")
    feat_b = hybrid.stage_b(feat_a)
    print(f"Stage B cikisi: {feat_b.shape}")

    print()
    transformer = TransformerOnly()
    print(f"Transformer-Only — Parametre sayisi: {count_parameters(transformer):,}")
    print(f"Giris: {x.shape}  →  Cikis: {transformer(x).shape}")
