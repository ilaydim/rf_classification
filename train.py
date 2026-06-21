"""
train.py — Eğitim döngüsü, resume (kaldığı yerden devam) destekli.

Kullanım:
  python train.py --model cnn          # CNN eğit
  python train.py --model hybrid       # Hybrid eğit
  python train.py --model transformer  # Transformer-Only eğit
  python train.py --model cnn --resume # Kaldığı yerden devam et
"""

import argparse
import os
import json
import time

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from data.dataset import get_dataloaders
from models.cnn_baseline import CNNBaseline
from models.hybrid_cnn_transformer import HybridCNNTransformer, TransformerOnly


def get_model(model_name: str):
    if model_name == "cnn":
        return CNNBaseline()
    elif model_name == "hybrid":
        return HybridCNNTransformer()
    elif model_name == "transformer":
        return TransformerOnly()
    else:
        raise ValueError(f"Bilinmeyen model: {model_name}")


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_val_loss, history):
    """Her epoch sonunda tam durum kaydeder — resume için."""
    torch.save({
        "epoch"         : epoch,
        "model_state"   : model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "best_val_loss" : best_val_loss,
        "history"       : history,
    }, path)


def load_checkpoint(path, model, optimizer, scheduler):
    """Kaydedilen durumu yükle."""
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    scheduler.load_state_dict(ckpt["scheduler_state"])
    return ckpt["epoch"], ckpt["best_val_loss"], ckpt["history"]


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, total_correct, total_samples = 0.0, 0, 0

    pbar = tqdm(loader, desc="Egitim", leave=False)
    for x, y, _ in pbar:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        preds = logits.argmax(dim=1)
        total_loss    += loss.item() * len(x)
        total_correct += (preds == y).sum().item()
        total_samples += len(x)
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0

    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        preds = logits.argmax(dim=1)
        total_loss    += loss.item() * len(x)
        total_correct += (preds == y).sum().item()
        total_samples += len(x)

    return total_loss / total_samples, total_correct / total_samples


def train(model_name: str, resume: bool = False):
    # ── Klasörler ─────────────────────────────────────────────────────────
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    # Checkpoint yolları
    resume_path = os.path.join(config.CHECKPOINT_DIR, f"resume_{model_name}.pt")
    best_path   = os.path.join(config.CHECKPOINT_DIR, f"best_{model_name}.pt")

    # ── Cihaz ─────────────────────────────────────────────────────────────
    device = torch.device(config.DEVICE)

    # ── Veri ──────────────────────────────────────────────────────────────
    (train_ds, train_loader, val_loader, _), _ = get_dataloaders()

    # ── Model ─────────────────────────────────────────────────────────────
    model = get_model(model_name).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # ── Optimizer ve Scheduler ────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE,
                      weight_decay=config.WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode="min",
                                  factor=config.LR_FACTOR,
                                  patience=config.LR_PATIENCE)

    # ── Resume: kaldığı yerden devam ──────────────────────────────────────
    start_epoch    = 1
    best_val_loss  = float("inf")
    no_improve     = 0
    history        = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    if resume and os.path.exists(resume_path):
        print(f"[Resume] Checkpoint bulundu: {resume_path}")
        start_epoch, best_val_loss, history = load_checkpoint(
            resume_path, model, optimizer, scheduler
        )
        start_epoch += 1  # bir sonraki epoch'tan devam et
        no_improve = 0
        print(f"[Resume] Epoch {start_epoch}'den devam ediliyor...")
    elif resume and not os.path.exists(resume_path):
        print("[Resume] Checkpoint bulunamadı, baştan başlanıyor.")

    # ── Bilgi ekranı ───────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  Model    : {model_name.upper()}")
    print(f"  Cihaz    : {device}")
    print(f"  Epochs   : {start_epoch} → {config.NUM_EPOCHS}")
    print(f"  Batch    : {config.BATCH_SIZE}")
    print(f"  Parametre: {total_params:,}")
    print(f"{'='*55}\n")

    # ── Eğitim döngüsü ────────────────────────────────────────────────────
    for epoch in range(start_epoch, config.NUM_EPOCHS + 1):
        t0 = time.time()

        # Her epoch başında veriyi karıştır
        train_ds.shuffle_epoch()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss,   val_acc   = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch:>3}/{config.NUM_EPOCHS} | "
            f"Tr: loss={train_loss:.4f} acc={train_acc:.4f} | "
            f"Val: loss={val_loss:.4f} acc={val_acc:.4f} | "
            f"LR={lr:.2e} | {elapsed:.1f}s"
        )

        # Her epoch sonunda resume checkpoint kaydet
        save_checkpoint(resume_path, model, optimizer, scheduler,
                        epoch, best_val_loss, history)

        # Her epoch sonunda history'yi kaydet (Ctrl+C olursa kaybolmasin)
        results_path = os.path.join(config.RESULTS_DIR, f"history_{model_name}.json")
        with open(results_path, "w") as f:
            json.dump(history, f, indent=2)

        # En iyi modeli ayrıca kaydet
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            torch.save({
                "epoch"      : epoch,
                "model_name" : model_name,
                "model_state": model.state_dict(),
                "val_loss"   : val_loss,
                "val_acc"    : val_acc,
            }, best_path)
            print(f"  ✓ En iyi model kaydedildi (val_acc={val_acc:.4f})")
        else:
            no_improve += 1
            if no_improve >= config.EARLY_STOP_PATIENCE:
                print(f"\n[EarlyStop] {config.EARLY_STOP_PATIENCE} epoch iyileşme yok, durdu.")
                break

    # Sonuçları kaydet
    results_path = os.path.join(config.RESULTS_DIR, f"history_{model_name}.json")
    with open(results_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nEgitim tamamlandi.")
    print(f"En iyi model  : {best_path}")
    print(f"Sonuclar      : {results_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="hybrid",
                        choices=["cnn", "hybrid", "transformer"])
    parser.add_argument("--resume", action="store_true",
                        help="Kaldığı yerden devam et")
    args = parser.parse_args()
    train(args.model, resume=args.resume)


if __name__ == "__main__":
    main()