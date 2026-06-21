"""
cnn_baseline.py — 1D CNN referans modeli.
O'Shea & West (2018) "Over-the-Air Deep Learning" mimarisinden uyarlanmıştır.
"""

import torch
import torch.nn as nn
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class CNNBlock(nn.Module):
    """Conv1d + BatchNorm + ReLU + MaxPool + Dropout bloğu."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel, padding=kernel // 2),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(p=0.1),
        )

    def forward(self, x):
        return self.block(x)


class CNNBaseline(nn.Module):
    """
    Baseline 1D CNN modeli.
    Giriş : (batch, 2, 1024)  — 2 kanal: I ve Q
    Çıkış : (batch, num_classes)  — sınıf olasılıkları
    """

    def __init__(
        self,
        num_classes: int = config.NUM_CLASSES,
        in_channels: int = config.INPUT_CHANNELS,
        filters: int = config.CNN_FILTERS,
        depth: int = config.CNN_DEPTH,
        fc_dim: int = config.CNN_FC_DIM,
        dropout: float = config.CNN_DROPOUT,
    ):
        super().__init__()

        # Evrişim katmanları: her adımda uzunluk yarıya iner (1024 → 8)
        layers = [CNNBlock(in_channels, filters)]
        for _ in range(depth - 1):
            layers.append(CNNBlock(filters, filters))
        self.conv_body = nn.Sequential(*layers)

        # Global Average Pooling → sabit boyut
        self.gap = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(filters, fc_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc_dim, fc_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 2, 1024)
        feat = self.conv_body(x)   # (B, filters, L')
        feat = self.gap(feat)      # (B, filters, 1)
        return self.classifier(feat)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = CNNBaseline()
    print(f"CNN Baseline — Parametre sayisi: {count_parameters(model):,}")
    x = torch.randn(4, config.INPUT_CHANNELS, config.SIGNAL_LENGTH)
    out = model(x)
    print(f"Giris: {x.shape}  →  Cikis: {out.shape}")
