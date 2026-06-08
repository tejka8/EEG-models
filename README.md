# EEG-FM-Bench

> A reproducible benchmarking framework for comparing modern deep learning architectures on EEG-based ADHD classification, with support for both clinical 19-channel EEG (10-20 system) and the consumer Neurosity Crown 8-channel headset.

---

## Overview

EEG-FM-Bench evaluates three deep learning architectures on the task of ADHD classification from EEG recordings under identical conditions:

| Model | Type | Pretrained |
|-------|------|------------|
| **NeuroGPT** | EEG Conformer + GPT transformer foundation model |  Yes |
| **EEGPT** | Channel-as-token transformer |  No (trained from scratch) |
| **EEGNet** | Compact CNN baseline (braindecode) |  No |

All models share an identical preprocessing pipeline, train/validation/test split, and evaluation metrics, enabling fair head-to-head comparison. A final ensemble decision is computed via majority vote across the three models.

The project also includes an interactive web dashboard (FastAPI + Vanilla JS) that auto-detects the input EEG format (19-channel clinical or 8-channel Crown) and routes to the matching set of trained models.

---

## Key Results

### Clinical EEG (19 channels, 10-20 system)

| Model | Test AUROC | Accuracy | Balanced Acc | AUC-PR |
|-------|:---------:|:--------:|:------------:|:------:|
| **NeuroGPT** | **0.995** | 94.5% | 94.9% | 0.994 |
| EEGPT | 0.896 | 78.8% | 80.8% | 0.866 |
| EEGNet | 0.893 | 67.1% | 70.5% | 0.811 |

### Neurosity Crown (8 channels, 10-10 system)

| Model | Test AUROC | Accuracy | Balanced Acc | AUC-PR |
|-------|:---------:|:--------:|:------------:|:------:|
| **NeuroGPT** | **0.763** | 58.3% | 62.5% | 0.738 |
| EEGPT | 0.704 | 66.4% | 68.5% | 0.703 |
| EEGNet | 0.700 | 66.5% | 68.5% | 0.687 |

The ~0.19–0.23 AUROC drop on the 8-channel configuration quantifies the impact of reduced channel count — particularly the loss of frontal FP1/FP2 electrodes that are highly informative for ADHD detection.

---

## Dataset

- **Subjects**: 121 children (61 ADHD, 60 controls), aged 7–12
- **System**: 19-channel 10-20 montage at 128 Hz
- **Task**: Visual attention paradigm
- **Diagnosis**: Confirmed by an experienced psychiatrist using DSM-IV criteria
- **Split**: Fixed 70/15/15 train/validation/test split (seed = 42), stratified by class

### Crown 8-channel Variant

Derived from the original 19-channel dataset via **MNE spherical spline interpolation** (Perrin et al., 1989) using the `standard_1005` montage. Of the 8 Crown channels (F5, F6, C3, C4, CP3, CP4, PO3, PO4), only C3 and C4 overlap directly with the 19-channel set; the remaining 6 are spatially interpolated.

---

## Preprocessing Pipeline

Identical for all models and both formats:

1. µV → V conversion (`× 1e-6`)
2. Set MNE montage (`standard_1005`)
3. Highpass filter at 0.1 Hz
4. Notch filter at 50 Hz
5. Resample to 256 Hz
6. Non-overlapping windows of 4 seconds (1024 samples)
7. `get_data(units='uV')` for final values in µV

Final prediction per recording is computed via majority vote across all windows within each model, then a second-level majority vote across the three models.

---

## Project Structure

```
EEG-FM-Bench/
├── baseline/                    # Training entry points & model configs
│   ├── eegpt/
│   ├── eegnet/
│   └── neurogpt/
├── data/
│   ├── dataset/                 # Dataset classes
│   │   ├── adhd.py              # 19-channel ADHD
│   │   └── adhd_crown.py        # 8-channel Crown variant
│   └── processor/               # Preprocessing wrappers
├── conf/preproc/                # Preprocessing YAML configs
├── assets/data/raw/             # Raw EEG CSVs (per-subject)
│   ├── ADHD/
│   └── ADHD_Crown/
├── dashboard/                   # Web application
│   ├── backend/
│   │   ├── main.py              # FastAPI entry point
│   │   ├── model_loader.py      # Model loading (6 models: 3×19ch + 3×8ch)
│   │   ├── preprocessor.py      # Auto-detecting preprocessor
│   │   └── inference.py         # Inference engine
│   ├── frontend/
│   │   └── index.html           # Dashboard UI
│   └── models/                  # Checkpoint files (.pt)
├── crown_dataset_builder.py     # 19ch → 8ch conversion script
└── README.md
```

---

## Dashboard

The dashboard provides an end-to-end interface for analyzing single EEG recordings:

- **Auto format detection**: 19-channel ADHD vs 8-channel Crown
- **Ensemble prediction**: Combined decision from all 3 models with confidence and reliability scoring
- **Per-model breakdown**: Individual predictions, confidences, and per-window classifications
- **Signal visualization**: First 10–30 seconds of the selected channels (Chart.js)
- **Brainwave bands**: Delta, theta, alpha, beta, gamma power with Normal/High status
- **History**: Local browser storage of past analyses with CSV export
- **Models reference**: Detailed metrics and architecture info for each model, toggleable between 19ch and 8ch variants

### Running the Dashboard

```bash
# From the dashboard directory
cd dashboard
uvicorn backend.main:app --reload --port 8000
```

Then open http://localhost:8000 in your browser. The first startup takes ~15 seconds while all 6 models are loaded into memory.

---

## Quick Start

### Prerequisites

- Python 3.10+
- PyTorch 2.0+ (with CUDA recommended for training)
- ~2 GB RAM for inference (all 6 models loaded)
- ~10 GB GPU memory for training

### Installation

```bash
git clone https://github.com/jovanaristevska/EEG-models.git
cd EEG-FM-Bench
pip install -r requirements.txt
```

### Training a Model

```bash
# Train NeuroGPT on 19-channel ADHD
python baseline_main.py conf_file=baseline/neurogpt/neurogpt.yaml model_type=neurogpt

# Train NeuroGPT on 8-channel Crown variant
python baseline_main.py conf_file=baseline/neurogpt/neurogpt_crown.yaml model_type=neurogpt
```

### Generating the Crown Dataset

```bash
# Converts 19-channel ADHD CSVs into 8-channel Crown CSVs via spherical interpolation
python crown_dataset_builder.py
```

### Running Inference via Dashboard

```bash
cd dashboard
uvicorn backend.main:app --reload --port 8000
# Open http://localhost:8000 and upload a CSV
```

---

## Tech Stack

| Component | Tool |
|-----------|------|
| Deep learning | PyTorch 2.x |
| EEG models | NeuroGPT, EEGPT (custom), EEGNet (braindecode) |
| Preprocessing | MNE-Python |
| Training | Azure GPU cluster (NVIDIA T4) |
| Experiment tracking | Weights & Biases |
| Backend | FastAPI + Uvicorn |
| Frontend | Vanilla HTML/JS + Chart.js |

---

## Reproducibility

- All experiments use a fixed random seed (`42`) for the train/validation/test split
- Identical preprocessing for all models (highpass → notch → resample → window)
- Same evaluation metrics (AUROC, AUC-PR, accuracy, balanced accuracy)
- Per-epoch checkpoints saved during training
- Training logs and per-epoch metrics available on Weights & Biases

---

## Limitations

This is a research prototype, not a diagnostic tool. Important caveats:

- Trained on 121 children aged 7–12 — performance outside this range is unverified
- Performance has not been validated across diverse populations or EEG devices
- The 8-channel Crown variant uses spatially interpolated data from clinical recordings, not raw recordings from an actual Crown device
- Always consult a qualified clinician for ADHD diagnosis

---

## Citation

If you use this work, please cite:

```bibtex
@thesis{ristevska2026eegfmbench,
  title  = {Сyстем за компаративна евалуација на EEG модели за ADHD класификација},
  author = {Ристевска, Јована},
  school = {Faculty of Computer Science and Engineering, Ss. Cyril and Methodius University in Skopje},
  year   = {2026},
  type   = {Diploma thesis}
}
```

---

## License

This project is released under the MIT License.

EEG data used in this project comes from a publicly available ADHD dataset. Please refer to the original dataset's license for usage terms.

---

## Author

**Јована Ристевска** (Jovana Ristevska)
Faculty of Computer Science and Engineering, Skopje
2026

## Acknowledgments

- The original ADHD EEG dataset providers
- The authors of NeuroGPT, EEGPT, and EEGNet for their open-source models
- braindecode and MNE-Python communities for their excellent EEG tooling
