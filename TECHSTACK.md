# Techstack & Proje Özeti
**CMPE 490 — AI Based RF Signal Classification for Electronic Warfare Scenarios**
TED University | Spring 2026 | İlayda Dim

---

## 1. Kullanılan Teknolojiler

| Katman | Araç / Kütüphane | Versiyon | Kullanım Amacı |
|---|---|---|---|
| Dil | Python | 3.9 (venv) | Tüm codebase |
| Deep Learning | PyTorch | ≥ 2.0 | Model, eğitim, inference |
| Veri okuma | h5py | ≥ 3.8 | HDF5 formatındaki dataset |
| Sayısal hesap | NumPy | ≥ 1.24 | Dizi işlemleri, split, normalize |
| ML Yardımcıları | scikit-learn | ≥ 1.3 | Stratified split, confusion matrix, classification report |
| Görselleştirme | Matplotlib | ≥ 3.7 | Eğitim eğrileri, SNR grafikleri |
| Görselleştirme | Seaborn | ≥ 0.12 | Confusion matrix heatmap |
| Progress bar | tqdm | ≥ 4.65 | Epoch / batch ilerleme |
| Tablo / analiz | pandas | ≥ 2.0 | EDA yardımcısı |

Cihaz sıralaması: **CUDA → MPS (Apple Silicon) → CPU** (otomatik algılanır)

---

## 2. Dataset — RadioML 2018.01A

| Özellik | Değer |
|---|---|
| Toplam örnek | 2 555 904 |
| Sinyal uzunluğu | 1024 sample |
| Giriş kanalları | 2 (I — In-phase, Q — Quadrature) |
| Etiket sayısı | 24 modülasyon sınıfı |
| SNR aralığı | −20 dB → +30 dB (2 dB adım, 26 seviye) |
| Depolama formatı | HDF5 (`.hdf5`) |
| HDF5 anahtarları | `X` (N,1024,2) · `Y` (N,24) one-hot · `Z` (N,1) SNR |

**24 Modülasyon Sınıfı:**
OOK, 4ASK, 8ASK, BPSK, QPSK, 8PSK, 16PSK, 32PSK,
16APSK, 32APSK, 64APSK, 128APSK, 16QAM, 32QAM, 64QAM, 128QAM, 256QAM,
AM-SSB-WC, AM-SSB-SC, AM-DSB-WC, AM-DSB-SC, FM, GMSK, OQPSK

---

## 3. Veri Hattı (data/dataset.py)

- **Lazy loading:** HDF5 dosyası hiçbir zaman tamamen RAM'e çekilmez; her `__getitem__` çağrısında tek satır okunur.
- **Stratified split:** Scikit-learn ile sınıf dağılımı korunarak %60 / %20 / %20 bölünmesi yapılır. Sonuç `data/split_indices.npz`'e kaydedilir; sonraki çalıştırmalarda yeniden hesaplanmaz.
- **Shuffle stratejisi:** Her epoch başında `shuffle_epoch()` çağrılır. Batch içi HDF5 indeksleri sıralı okunur (I/O optimizasyonu), ardından karışık sıraya geri map edilir.
- **Normalizasyon:** Her örnek kendi mutlak maksimumuna bölünür (`x / (|x|.max + ε)`).

---

## 4. Modeller

### 4.1 CNN Baseline (`models/cnn_baseline.py`)
*Referans: O'Shea & West (2018) "Over-the-Air Deep Learning"*

```
Giriş (B, 2, 1024)
  └─ 7 × CNNBlock(Conv1d → BN → ReLU → MaxPool → Dropout)
       (her adımda uzunluk yarıya iner: 1024 → 8)
  └─ Global Average Pooling → (B, 64, 1)
  └─ Flatten
  └─ Linear(64→128) → ReLU → Dropout(0.5)
  └─ Linear(128→128) → ReLU → Dropout(0.5)
  └─ Linear(128→24)
```

| Hiperparametre | Değer |
|---|---|
| Filtre sayısı | 64 |
| Kernel boyutu | 3 |
| Derinlik | 7 blok |
| FC boyutu | 128 |
| Dropout | 0.5 |

---

### 4.2 Hibrit CNN-Transformer (`models/hybrid_cnn_transformer.py`)
*Ana katkı — Referans: HCTC (Ruikar et al. 2024) + Ansari et al. 2025*

Üç aşamalı mimari:

```
Giriş (B, 2, 1024)
  │
  ├─ Stage A — CNN Feature Extractor
  │    4 × ConvBlock(Conv1d → BN → ReLU → Dropout)
  │    Filtreler : 64 → 128 → 256 → 256
  │    Kernellar : 7 → 5 → 3 → 3
  │    Projeksiyon: Conv1d(256 → d_model=256, kernel=1)
  │    AvgPool(8×) → (B, 256, 128)
  │
  ├─ Stage B — Transformer Encoder
  │    Sinüs tabanlı Positional Encoding (Vaswani et al. 2017)
  │    4 × TransformerEncoderLayer
  │      d_model=256, nhead=8, dim_ff=512
  │      Pre-LN (norm_first=True — kararlı eğitim)
  │    Global Average Pooling → (B, 256)
  │
  └─ Stage C — FC Classifier
       Linear(256→512) → BN → ReLU → Dropout(0.3)
       Linear(512→256) → BN → ReLU → Dropout(0.3)
       Linear(256→24)
```

| Hiperparametre | Değer |
|---|---|
| d_model | 256 |
| Transformer head sayısı | 8 |
| Transformer katman sayısı | 4 |
| Feed-forward boyutu | 512 |
| CNN dropout | 0.3 |
| Transformer dropout | 0.1 |
| FC dropout | 0.3 |

---

### 4.3 Transformer-Only (`models/hybrid_cnn_transformer.py` → `TransformerOnly`)
*Ablasyon / karşılaştırma modeli*

```
Giriş (B, 2, 1024)
  └─ Conv1d(2 → 256, kernel=8, stride=8) → (B, 256, 128)  # downsample
  └─ Positional Encoding
  └─ 4 × TransformerEncoderLayer (aynı konfigürasyon)
  └─ Global Average Pooling → (B, 256)
  └─ FC Head → (B, 24)
```

---

## 5. Eğitim Konfigürasyonu (config.py / train.py)

| Parametre | Değer |
|---|---|
| Optimizer | AdamW |
| Öğrenme hızı | 3 × 10⁻⁴ |
| Weight decay | 1 × 10⁻⁴ |
| Loss | CrossEntropyLoss (label smoothing = 0.1) |
| LR Scheduler | ReduceLROnPlateau (patience=8, factor=0.5, min_lr=1e-6) |
| Early stopping | patience = 20 epoch |
| Batch size | 256 |
| Maks epoch | 100 |
| Gradient clipping | max_norm = 1.0 |
| Veri bölümü | 60 / 20 / 20 (stratified) |
| Random seed | 42 |
| Resume | Her epoch sonunda checkpoint kaydedilir |

---

## 6. Eğitim Sonuçları

| Model | Çalışılan Epoch | En İyi Val. Doğruluğu |
|---|---|---|
| CNN Baseline | 100 | **54.69 %** |
| Transformer-Only | 4 | **55.17 %** |
| Hibrit CNN-Transformer | 23 | **58.28 %** |

> Hibrit model erken durdurulmuştur (23 epoch); eğitim tamamlandığında daha yüksek doğruluk beklenmektedir.

---

## 7. Dosya Yapısı

```
rf_classification/
├── config.py                        # Tüm hiperparametreler ve yollar
├── train.py                         # Eğitim döngüsü (resume destekli)
├── evaluate.py                      # Test değerlendirmesi + grafik üretimi
├── eda.py                           # Keşifsel veri analizi (RAM'e almadan)
├── requirements.txt                 # Python bağımlılıkları
│
├── data/
│   ├── dataset.py                   # RadioMLDataset + get_dataloaders()
│   ├── download_dataset.py          # Dataset indirme yönergeleri
│   ├── split_indices.npz            # Stratified split indeksleri (önbellek)
│   └── GOLD_XYZ_OSC.0001_1024.hdf5 # RadioML 2018.01A verisi
│
├── models/
│   ├── __init__.py
│   ├── cnn_baseline.py              # CNNBaseline
│   └── hybrid_cnn_transformer.py   # HybridCNNTransformer, TransformerOnly
│
├── checkpoints/
│   ├── best_cnn.pt                  # En iyi CNN ağırlıkları
│   ├── best_hybrid.pt               # En iyi Hybrid ağırlıkları
│   ├── best_transformer.pt          # En iyi Transformer ağırlıkları
│   ├── resume_cnn.pt                # Resume checkpoint — CNN
│   ├── resume_hybrid.pt             # Resume checkpoint — Hybrid
│   └── resume_transformer.pt        # Resume checkpoint — Transformer
│
└── results/
    ├── history_{model}.json         # Epoch bazlı loss / acc geçmişi
    ├── training_{model}.png         # Eğitim eğrisi grafikleri
    ├── confusion_{model}.png        # Confusion matrix ısı haritası
    ├── snr_accuracy.png             # SNR bazlı doğruluk karşılaştırması
    ├── eda_signals.png              # 24 sınıftan örnek I/Q sinyali
    └── eda_snr_dist.png             # SNR dağılım grafiği
```

---

## 8. Temel Komutlar

```bash
# Ortam kurulumu
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Keşifsel veri analizi
python eda.py

# Eğitim
python train.py --model cnn          # Baseline CNN
python train.py --model hybrid       # Hibrit CNN-Transformer
python train.py --model transformer  # Transformer-Only
python train.py --model hybrid --resume   # Kaldığı yerden devam

# Değerlendirme
python evaluate.py --model hybrid --compare cnn
```

---

## 9. Referanslar

- O'Shea, T. & West, N. (2018). *Over-the-Air Deep Learning Based Radio Signal Classification.*
- Vaswani, A. et al. (2017). *Attention Is All You Need.*
- Ruikar, D. et al. (2024). *HCTC — Hybrid CNN-Transformer for RF Classification.*
- Ansari, M. et al. (2025). *Transformer-Based Automatic Modulation Classification.*
- DeepSig Inc. (2018). *RadioML 2018.01A Dataset.*
