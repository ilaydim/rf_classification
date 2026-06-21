"""
eda.py — Keşifsel Veri Analizi
Veriyi RAM'e ALMADAN sadece küçük dilimler okur.
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

def main():
    path = config.DATASET_PATH
    if not os.path.exists(path):
        print(f"HATA: Dataset bulunamadı: {path}")
        return

    print("=" * 55)
    print("RadioML 2018.01A — Keşifsel Veri Analizi")
    print("=" * 55)

    with h5py.File(path, "r") as f:
        # Sadece shape bilgisi — RAM kullanmaz
        x_shape = f["X"].shape
        y_shape = f["Y"].shape
        z_shape = f["Z"].shape

        print(f"\nX (sinyal) : {x_shape}")
        print(f"Y (etiket) : {y_shape}")
        print(f"Z (SNR)    : {z_shape}")
        print(f"\nToplam örnek : {x_shape[0]:,}")
        print(f"Sinyal uzunluğu: {x_shape[1]} sample")
        print(f"Kanal sayısı : {x_shape[2]} (I ve Q)")

        # SNR değerlerini küçük dilimde oku
        z_sample = f["Z"][:500000, 0]
        snr_unique = np.unique(z_sample)
        print(f"\nSNR aralığı  : {snr_unique.min():.0f} dB → {snr_unique.max():.0f} dB")
        print(f"SNR seviyeleri: {len(snr_unique)} adet")

        # Her sınıftan sadece 1 örnek çek — toplam 24 örnek
        # Dataset sınıflara göre blok blok sıralı, her blok N/24 örnek içeriyor
        print("\nHer modülasyondan 1 örnek okunuyor...")
        n_total = f["X"].shape[0]
        block_size = n_total // config.NUM_CLASSES  # her sınıf bloğu ~106k örnek
        samples = {}
        for cls_idx in range(config.NUM_CLASSES):
            sample_idx = cls_idx * block_size  # her bloktan ilk örnek
            x_sample = f["X"][sample_idx]      # (1024, 2) — sadece 1 satır okur
            samples[cls_idx] = x_sample

    print(f"✓ {len(samples)} sınıftan örnek okundu\n")

    # ── Grafik 1: Her modülasyondan örnek sinyal ──────────────────────
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    fig, axes = plt.subplots(4, 6, figsize=(18, 10))
    axes = axes.flatten()

    for i in range(config.NUM_CLASSES):
        if i not in samples:
            continue
        x = samples[i]
        I = x[:200, 0]
        Q = x[:200, 1]
        axes[i].plot(I, color="#2196F3", alpha=0.8, linewidth=0.8, label="I")
        axes[i].plot(Q, color="#FF9800", alpha=0.8, linewidth=0.8, label="Q")
        axes[i].set_title(config.MODULATION_CLASSES[i], fontsize=9, fontweight="bold")
        axes[i].set_xticks([]); axes[i].set_yticks([])

    axes[0].legend(fontsize=7, loc="upper right")
    plt.suptitle("RadioML 2018.01A — Her Modülasyondan Örnek I/Q Sinyali", fontsize=13)
    plt.tight_layout()

    save_path = os.path.join(config.RESULTS_DIR, "eda_signals.png")
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    print(f"✓ Grafik kaydedildi: {save_path}")
    plt.show()

    # ── Grafik 2: SNR dağılımı ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    snr_vals, snr_counts = np.unique(z_sample, return_counts=True)
    ax.bar(snr_vals, snr_counts, width=1.5, color="#4CAF50", edgecolor="white")
    ax.set_xlabel("SNR (dB)", fontsize=12)
    ax.set_ylabel("Örnek sayısı (ilk 10k'dan)", fontsize=12)
    ax.set_title("SNR Dağılımı", fontsize=13)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()

    save_path2 = os.path.join(config.RESULTS_DIR, "eda_snr_dist.png")
    plt.savefig(save_path2, dpi=120, bbox_inches="tight")
    print(f"✓ Grafik kaydedildi: {save_path2}")
    plt.show()

    print("\n✓ EDA tamamlandı!")

if __name__ == "__main__":
    main()