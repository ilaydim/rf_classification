"""
dataset.py — RadioML 2018.01A için PyTorch Dataset ve DataLoader.

Dataset yapısı (HDF5):
  X : (2_555_904, 1024, 2)  — float32, I/Q sinyal
  Y : (2_555_904, 24)       — float32, one-hot etiket
  Z : (2_555_904, 1)        — float32, SNR (dB)

Shuffle stratejisi:
  Her epoch başında shuffle_epoch() çağrılır.
  Epoch genelinde karıştırılmış sıra oluşturulur; HDF5 okuması
  her batch içinde sıralı indeksle yapılır, sonuçlar karışık sıraya
  geri map edilir (sorted-read → remap).
"""

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

_SPLIT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "split_indices.npz")


class RadioMLDataset(Dataset):
    """
    HDF5 dosyasından lazy-loading yapan Dataset.

    shuffle_epoch() her epoch başında çağrılmalıdır.
    HDF5 okuması her batch içinde sıralı yapılır (performans),
    sonuçlar orijinal karışık sıraya geri map edilir.
    """

    def __init__(self, hdf5_path: str, indices: np.ndarray,
                 batch_size: int, normalize: bool = True):
        self.hdf5_path  = hdf5_path
        self.indices    = indices          # ham indeksler — sıralanmaz
        self.batch_size = batch_size
        self.normalize  = normalize
        # Birinci epoch'tan önce identity sıra; shuffle_epoch() ile güncellenir
        self.epoch_read_order = np.arange(len(indices), dtype=np.int64)

    def shuffle_epoch(self):
        """
        Epoch başında karıştırma + batch-içi HDF5 sort + remap.

        1. Tüm N indeks rastgele karıştırılır (perm).
        2. Her batch penceresi içinde HDF5 indeksleri sıralanır
           → batch içinde sequential HDF5 erişimi.
        3. epoch_read_order[i], DataLoader'ın __getitem__(i) çağrısında
           hangi orijinal indeksin okunacağını belirtir.
        """
        N, BS = len(self.indices), self.batch_size
        perm = np.random.permutation(N)

        read_order = np.empty(N, dtype=np.int64)
        for b_start in range(0, N, BS):
            b_end          = min(b_start + BS, N)
            batch_pos      = perm[b_start:b_end]          # bu batch'in epoch pozisyonları
            sort_within    = np.argsort(self.indices[batch_pos])  # HDF5'e göre sırala
            read_order[b_start:b_end] = batch_pos[sort_within]   # sıralı okunacak pozisyonlar

        self.epoch_read_order = read_order

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = int(self.indices[self.epoch_read_order[idx]])

        with h5py.File(self.hdf5_path, "r") as f:
            x        = f["X"][real_idx]        # (1024, 2)
            y_onehot = f["Y"][real_idx]         # (24,)
            snr      = float(f["Z"][real_idx, 0])

        x = x.astype(np.float32).T             # (1024, 2) → (2, 1024)

        if self.normalize:
            x = x / (np.abs(x).max() + 1e-8)

        label = int(np.argmax(y_onehot))

        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(label, dtype=torch.long),
            torch.tensor(snr, dtype=torch.float32),
        )


def _compute_and_save_splits(hdf5_path: str) -> tuple:
    """HDF5'ten stratified split hesaplar ve split_indices.npz'e kaydeder."""
    with h5py.File(hdf5_path, "r") as f:
        n_total = f["X"].shape[0]

    print("Etiketler okunuyor (bir kez)...")
    chunk  = 50_000
    labels = np.empty(n_total, dtype=np.int8)
    with h5py.File(hdf5_path, "r") as f:
        for start in range(0, n_total, chunk):
            end = min(start + chunk, n_total)
            labels[start:end] = np.argmax(f["Y"][start:end], axis=1)
    print("Etiketler hazır.")

    all_idx = np.arange(n_total)
    train_val_idx, test_idx = train_test_split(
        all_idx, test_size=config.TEST_RATIO,
        stratify=labels, random_state=config.RANDOM_SEED,
    )
    val_ratio_adj = config.VAL_RATIO / (config.TRAIN_RATIO + config.VAL_RATIO)
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=val_ratio_adj,
        stratify=labels[train_val_idx], random_state=config.RANDOM_SEED,
    )

    np.savez(_SPLIT_PATH, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)
    print(f"Split indeksleri kaydedildi: {_SPLIT_PATH}")
    return train_idx, val_idx, test_idx


def get_dataloaders(
    hdf5_path: str  = config.DATASET_PATH,
    batch_size: int = config.BATCH_SIZE,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> tuple:
    """
    Train / Val / Test DataLoader ve train Dataset döndürür.
    train_ds her epoch başında shuffle_epoch() çağrılmalı.

    İlk çalıştırmada split hesaplanır ve data/split_indices.npz'e kaydedilir.
    Sonraki çalıştırmalarda doğrudan dosyadan yüklenir.

    Returns:
        (train_ds, train_loader, val_loader, test_loader), class_names
    """

    if os.path.exists(_SPLIT_PATH):
        print(f"Split indeksleri yükleniyor: {_SPLIT_PATH}")
        data      = np.load(_SPLIT_PATH)
        train_idx = data["train_idx"]
        val_idx   = data["val_idx"]
        test_idx  = data["test_idx"]
    else:
        train_idx, val_idx, test_idx = _compute_and_save_splits(hdf5_path)

    print(f"Train: {len(train_idx):,}  Val: {len(val_idx):,}  Test: {len(test_idx):,}")

    train_ds = RadioMLDataset(hdf5_path, train_idx, batch_size, normalize=True)
    val_ds   = RadioMLDataset(hdf5_path, val_idx,   batch_size, normalize=True)
    test_ds  = RadioMLDataset(hdf5_path, test_idx,  batch_size, normalize=True)

    kwargs = dict(batch_size=batch_size, num_workers=num_workers,
                  pin_memory=pin_memory, shuffle=False)
    train_loader = DataLoader(train_ds, **kwargs)
    val_loader   = DataLoader(val_ds,   **kwargs)
    test_loader  = DataLoader(test_ds,  **kwargs)

    return (train_ds, train_loader, val_loader, test_loader), config.MODULATION_CLASSES
