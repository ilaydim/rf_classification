"""
download_dataset.py — Dataset indirme yönergeleri ve doğrulama.

Çalıştır:  python data/download_dataset.py
"""
import os, sys

DATASET_FILENAME = "GOLD_XYZ_OSC.0001_1024.hdf5"
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(DATA_DIR, DATASET_FILENAME)

print("""
╔══════════════════════════════════════════════════════════════╗
║     RadioML 2018.01A — Dataset İndirme Yönergesi            ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  YÖNTEM 1 — Kaggle Web (önerilen):                          ║
║    1. https://www.kaggle.com/datasets/pinakigupta95/         ║
║       radioml-201801a adresine git                           ║
║    2. "Download" butonuna tıkla                              ║
║    3. GOLD_XYZ_OSC.0001_1024.hdf5 dosyasını                 ║
║       rf_classification/data/ klasörüne taşı                 ║
║                                                              ║
║  YÖNTEM 2 — Kaggle CLI:                                     ║
║    pip install kaggle                                        ║
║    kaggle datasets download pinakigupta95/radioml-201801a   ║
║    unzip radioml-201801a.zip -d data/                        ║
║                                                              ║
║  YÖNTEM 3 — DeepSig resmi site:                             ║
║    https://www.deepsig.ai/datasets                           ║
║                                                              ║
║  Dosya boyutu: ~7 GB                                        ║
║  Beklenen yol: rf_classification/data/                       ║
║                GOLD_XYZ_OSC.0001_1024.hdf5                  ║
╚══════════════════════════════════════════════════════════════╝
""")

if os.path.exists(DATASET_PATH):
    size_gb = os.path.getsize(DATASET_PATH) / 1e9
    print(f"✓ Dataset mevcut: {DATASET_PATH} ({size_gb:.2f} GB)")

    # Hızlı doğrulama
    try:
        import h5py
        with h5py.File(DATASET_PATH, "r") as f:
            x_shape = f["X"].shape
            y_shape = f["Y"].shape
            z_shape = f["Z"].shape
        print(f"✓ HDF5 gecerli:")
        print(f"  X (sinyal) : {x_shape}")
        print(f"  Y (etiket) : {y_shape}")
        print(f"  Z (SNR)    : {z_shape}")
        print(f"\nDataset kullanima hazir!")
    except Exception as e:
        print(f"✗ HDF5 okuma hatasi: {e}")
else:
    print(f"✗ Dataset bulunamadi: {DATASET_PATH}")
    print("Yukaridaki yontemlerden birini kullanarak indirin.")
