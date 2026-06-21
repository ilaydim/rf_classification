# AI Based RF Signal Classification for Electronic Warfare Scenarios
**TED University – CMPE 490 | Spring 2026 | İlayda Dim**

## Dataset
RadioML 2018.01A — `data/` klasörüne indir:
```
python data/download_dataset.py
```

## Kurulum
```bash
conda create -n rf_class python=3.10 && conda activate rf_class
pip install -r requirements.txt
```

## Eğitim
```bash
python train.py --model cnn      # Baseline CNN
python train.py --model hybrid   # Hibrit CNN-Transformer
```

## Değerlendirme
```bash
python evaluate.py --model hybrid --compare cnn
```

## Proje Yapısı
```
rf_classification/
├── config.py                        # Tüm hiperparametreler
├── train.py                         # Eğitim döngüsü
├── evaluate.py                      # Değerlendirme ve grafikler
├── data/
│   ├── dataset.py                   # PyTorch Dataset + DataLoader
│   └── download_dataset.py          # Dataset indirme yönergeleri
├── models/
│   ├── cnn_baseline.py              # Baseline 1D CNN
│   └── hybrid_cnn_transformer.py   # Ana hibrit model
├── checkpoints/                     # Kaydedilen modeller
└── results/                         # Grafikler ve sonuçlar
```
