"""
evaluate.py — Eğitilmiş modeli değerlendir, grafikler üret.

Kullanım:
  python evaluate.py --model cnn
  python evaluate.py --model hybrid
  python evaluate.py --model hybrid --compare cnn   # ikisini karşılaştır
"""

import argparse
import json
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # GUI gerektirmeyen backend
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

import config
from data.dataset import get_dataloaders
from models.cnn_baseline import CNNBaseline
from models.hybrid_cnn_transformer import HybridCNNTransformer, TransformerOnly


def load_model(model_name: str) -> nn.Module:
    ckpt_path = Path(config.SAVE_DIR) / f"best_{model_name}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint bulunamadi: {ckpt_path}\nOnce train.py calistirin.")

    if model_name == "cnn":
        model = CNNBaseline()
    elif model_name == "transformer":
        model = TransformerOnly()
    else:
        model = HybridCNNTransformer()
    ckpt = torch.load(ckpt_path, map_location=config.DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(config.DEVICE)
    model.eval()
    print(f"{model_name.upper()} yuklendi — epoch={ckpt['epoch']}, val_acc={ckpt['val_acc']:.4f}")
    return model


@torch.no_grad()
def predict_all(model, loader):
    """Tüm test setindeki tahmin ve gerçek etiketleri + SNR'ları topla."""
    all_preds, all_labels, all_snrs = [], [], []
    for x, y, snr in tqdm(loader, desc="  Tahmin", ncols=80, leave=False):
        x = x.to(config.DEVICE, non_blocking=True)
        logits = model(x)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(y.numpy())
        all_snrs.append(snr.numpy())
    return (
        np.concatenate(all_preds),
        np.concatenate(all_labels),
        np.concatenate(all_snrs),
    )


def accuracy_per_snr(preds, labels, snrs):
    """Her SNR seviyesi için doğruluk hesapla."""
    accs = {}
    for snr_val in config.SNR_LEVELS:
        mask = np.isclose(snrs, snr_val)
        if mask.sum() == 0:
            accs[snr_val] = 0.0
            continue
        accs[snr_val] = (preds[mask] == labels[mask]).mean()
    return accs


def plot_snr_accuracy(results: dict, save_path: str):
    """SNR bazlı doğruluk grafiği (birden fazla model karşılaştırması)."""
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {"cnn": "#4C72B0", "hybrid": "#DD8452"}
    linestyles = {"cnn": "--", "hybrid": "-"}

    for model_name, acc_dict in results.items():
        snrs = sorted(acc_dict.keys())
        accs = [acc_dict[s] * 100 for s in snrs]
        ax.plot(snrs, accs,
                label=model_name.upper(),
                color=colors.get(model_name, "gray"),
                linestyle=linestyles.get(model_name, "-"),
                linewidth=2, marker="o", markersize=4)

    ax.set_xlabel("SNR (dB)", fontsize=13)
    ax.set_ylabel("Dogruluk (%)", fontsize=13)
    ax.set_title("SNR Bazli Siniflandirma Basarisi — RadioML 2018.01A", fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-21, 31)
    ax.set_ylim(0, 105)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close()
    print(f"  SNR grafigi kaydedildi: {save_path}")


def plot_snr_comparison(cnn_accs: dict, transformer_accs: dict, hybrid_accs: dict, save_path: str):
    """Three-model SNR accuracy comparison in academic style."""
    plt.rcParams.update({
        "font.family": "serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig, ax = plt.subplots(figsize=(10, 5))

    snrs = sorted(cnn_accs.keys())

    ax.plot(snrs, [cnn_accs[s] * 100 for s in snrs],
            color="black", linestyle="--", linewidth=1.8,
            label="CNN Only")
    ax.plot(snrs, [transformer_accs[s] * 100 for s in snrs],
            color="black", linestyle=":", linewidth=1.8,
            label="Transformer Only")
    ax.plot(snrs, [hybrid_accs[s] * 100 for s in snrs],
            color="#008080", linestyle="-", linewidth=3.0,
            label="Hybrid (Proposed)")

    ax.set_xlabel("SNR (dB)", fontsize=13)
    ax.set_ylabel("Accuracy (%)", fontsize=13)
    ax.set_title("SNR-Based Classification Accuracy — RadioML 2018.01A", fontsize=14)
    ax.set_xlim(-20, 30)
    ax.set_ylim(0, 105)
    ax.set_xticks(range(-20, 31, 5))
    ax.legend(fontsize=12, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close()
    plt.rcParams.update(plt.rcParamsDefault)
    print(f"  Karsilastirma grafigi kaydedildi: {save_path}")


def plot_confusion_matrix(preds, labels, class_names, snr_filter, save_path: str):
    """Belirli bir SNR seviyesinde confusion matrix."""
    mask = np.isclose(snr_filter[0], snr_filter[1]) if isinstance(snr_filter, tuple) else np.ones(len(preds), dtype=bool)
    cm = confusion_matrix(labels, preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(cm_norm, annot=True, fmt=".2f",
                xticklabels=class_names, yticklabels=class_names,
                cmap="Blues", ax=ax, vmin=0, vmax=1,
                linewidths=0.3, linecolor="gray",
                annot_kws={"size": 7})
    ax.set_xlabel("Tahmin edilen", fontsize=12)
    ax.set_ylabel("Gercek", fontsize=12)
    ax.set_title("Confusion Matrix (Tum SNR)", fontsize=13)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Confusion matrix kaydedildi: {save_path}")


def plot_training_history(model_name: str, save_path: str):
    """Eğitim sürecindeki kayıp ve doğruluk eğrilerini çiz."""
    history_path = Path(config.RESULTS_DIR) / f"history_{model_name}.json"
    if not history_path.exists():
        print(f"  Gecmis dosyasi bulunamadi: {history_path}")
        return

    with open(history_path) as f:
        hist = json.load(f)

    epochs = range(1, len(hist["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, hist["train_loss"], label="Train", color="#4C72B0")
    axes[0].plot(epochs, hist["val_loss"],   label="Val",   color="#DD8452")
    axes[0].set_title(f"{model_name.upper()} — Kayip", fontsize=12)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Kayip")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    tr_acc = [a * 100 for a in hist["train_acc"]]
    vl_acc = [a * 100 for a in hist["val_acc"]]
    axes[1].plot(epochs, tr_acc, label="Train", color="#4C72B0")
    axes[1].plot(epochs, vl_acc, label="Val",   color="#DD8452")
    axes[1].set_title(f"{model_name.upper()} — Dogruluk", fontsize=12)
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Dogruluk (%)")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Egitim grafigi kaydedildi: {save_path}")


def evaluate_model(model_name: str, test_loader, class_names) -> dict:
    model = load_model(model_name)
    preds, labels, snrs = predict_all(model, test_loader)

    snr_accs = accuracy_per_snr(preds, labels, snrs)
    overall  = (preds == labels).mean()

    print(f"\n{model_name.upper()} — Genel dogruluk: {overall*100:.2f}%")
    print(f"  0 dB dogrulugu  : {snr_accs.get(0, 0)*100:.2f}%")
    print(f"  10 dB dogrulugu : {snr_accs.get(10, 0)*100:.2f}%")

    # Confusion matrix
    cm_path = Path(config.RESULTS_DIR) / f"confusion_{model_name}.png"
    plot_confusion_matrix(preds, labels, class_names, None, str(cm_path))

    # Eğitim grafiği
    hist_path = Path(config.RESULTS_DIR) / f"training_{model_name}.png"
    plot_training_history(model_name, str(hist_path))

    return snr_accs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RF Siniflandirma — Degerlendirme")
    parser.add_argument("--model",   type=str, default="hybrid",
                        choices=["cnn", "hybrid", "transformer"])
    parser.add_argument("--compare", action="store_true",
                        help="CNN, Transformer ve Hybrid ucunu karsilastir")
    parser.add_argument("--batch_size",  type=int, default=config.BATCH_SIZE)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    print("Test loader hazirlaniyor...")
    (_, _, _, test_l), class_names = get_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    if args.compare:
        print("\n=== 3-Model karsilastirmasi basliyor ===")
        cnn_accs         = evaluate_model("cnn",         test_l, class_names)
        transformer_accs = evaluate_model("transformer", test_l, class_names)
        hybrid_accs      = evaluate_model("hybrid",      test_l, class_names)

        comp_path = Path(config.RESULTS_DIR) / "snr_comparison.png"
        plot_snr_comparison(cnn_accs, transformer_accs, hybrid_accs, str(comp_path))
    else:
        snr_results = {args.model: evaluate_model(args.model, test_l, class_names)}
        snr_plot_path = Path(config.RESULTS_DIR) / "snr_accuracy.png"
        plot_snr_accuracy(snr_results, str(snr_plot_path))

    print(f"\nTum sonuclar '{config.RESULTS_DIR}' klasorune kaydedildi.")
