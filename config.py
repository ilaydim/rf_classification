"""
config.py — Tüm hiperparametreler tek yerden yönetilir.
Kaynak: HCTC (Ruikar et al. 2024) + Ansari et al. 2025
"""
import os
import torch

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR     = os.path.join(BASE_DIR, "data")
DATASET_PATH = os.path.join(DATA_DIR, "GOLD_XYZ_OSC.0001_1024.hdf5")
SAVE_DIR     = os.path.join(BASE_DIR, "checkpoints")
CHECKPOINT_DIR = SAVE_DIR
RESULTS_DIR  = os.path.join(BASE_DIR, "results")

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Dataset ───────────────────────────────────────────────────────────────────
MODULATION_CLASSES = [
    "OOK", "4ASK", "8ASK", "BPSK", "QPSK", "8PSK",
    "16PSK", "32PSK", "16APSK", "32APSK", "64APSK", "128APSK",
    "16QAM", "32QAM", "64QAM", "128QAM", "256QAM",
    "AM-SSB-WC", "AM-SSB-SC", "AM-DSB-WC", "AM-DSB-SC",
    "FM", "GMSK", "OQPSK"
]
NUM_CLASSES    = len(MODULATION_CLASSES)  # 24
SIGNAL_LENGTH  = 1024
INPUT_CHANNELS = 2   # I ve Q

HDF5_X_KEY = "X"   # (N, 1024, 2)
HDF5_Y_KEY = "Y"   # (N, 24) one-hot
HDF5_Z_KEY = "Z"   # (N, 1)  SNR

SNR_LEVELS = list(range(-20, 32, 2))   # 26 seviye

# ── Split ─────────────────────────────────────────────────────────────────────
TRAIN_RATIO = 0.60
VAL_RATIO   = 0.20
TEST_RATIO  = 0.20
RANDOM_SEED = 42

# ── Training ──────────────────────────────────────────────────────────────────
BATCH_SIZE    = 256
NUM_EPOCHS          = 100
LEARNING_RATE = 3e-4
WEIGHT_DECAY        = 1e-4
LR_PATIENCE         = 8
LR_FACTOR           = 0.5
MIN_LR              = 1e-6
EARLY_STOP_PATIENCE = 20

# ── CNN Baseline ──────────────────────────────────────────────────────────────
CNN_FILTERS  = 64
CNN_KERNEL   = 3
CNN_DEPTH    = 7
CNN_FC_DIM   = 128
CNN_DROPOUT  = 0.5

# ── Hybrid CNN-Transformer ────────────────────────────────────────────────────
HYBRID_CNN_FILTERS  = [64, 128, 256, 256]
HYBRID_CNN_KERNELS  = [7, 5, 3, 3]
HYBRID_CNN_DROPOUT  = 0.3

TRANSFORMER_D_MODEL = 256
TRANSFORMER_NHEAD   = 8
TRANSFORMER_DIM_FF  = 512
TRANSFORMER_LAYERS  = 4
TRANSFORMER_DROPOUT = 0.1

FC_DIM_1   = 512
FC_DIM_2   = 256
FC_DROPOUT = 0.3

# ── Device ────────────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

if __name__ == "__main__":
    print(f"Dataset     : {DATASET_PATH}")
    print(f"Sinif sayisi: {NUM_CLASSES}")
    print(f"SNR araligi : {SNR_LEVELS[0]} dB -> {SNR_LEVELS[-1]} dB")
    print(f"Cihaz       : {DEVICE}")
