# ECG Upsampling — Neural Network Pipeline

Super-resolution of ECG signals via deep learning: **100 Hz → 500 Hz** (×5 upsampling factor).

---

## Motivation

Consumer-grade and wearable ECG devices often record at reduced sampling rates (100–125 Hz) to save power and storage, losing high-frequency morphological detail critical for clinical diagnosis (e.g., HF component of HRV, fine P-wave structure, notch detection). This project trains a neural network to recover the full 500 Hz bandwidth from 100 Hz recordings.

---

## Project Phases

| Phase | Description | Status |
|-------|-------------|--------|
| **1** | Supervised dataset construction (degrading pipeline + QC) | In progress |
| 2 | Neural network architecture design | Planned |
| 3 | Training loop, losses, optimizer | Planned |
| 4 | Morphological loss (PQRST-aware) | Planned |
| 5 | Evaluation and benchmarking | Planned |

---

## Phase 1 — Dataset Construction

Builds paired samples **(x_LR, x_HR)** where:
- **x_HR**: original 500 Hz ECG ground truth  
- **x_LR**: physically-motivated degraded version at 100 Hz

The degrading chain **D = Q ∘ N ∘ S ∘ F** applies:
1. Kaiser FIR anti-aliasing filter (fc = 50 Hz, As ≥ 80 dB)
2. Decimation ×5
3. Composite noise (thermal + powerline 50 Hz + baseline wander)
4. 12-bit ADC quantization

See [`phase1_dataset_preparation/README.md`](phase1_dataset_preparation/README.md) for full details.

### Datasets Used

| Dataset | Frequency | Patients | Leads | Role |
|---------|-----------|----------|-------|------|
| [PTB-XL](https://physionet.org/content/ptb-xl/1.0.3/) | 500 Hz | 18 885 | 12 | Primary training source |
| [PTB Diagnostic](https://physionet.org/content/ptbdb/1.0.0/) | 1000 Hz | 290 | 15 | High-fidelity pairs |
| [MIT-BIH Arrhythmia](https://physionet.org/content/mitdb/1.0.0/) | 360 Hz | 47 | 2 | Arrhythmia diversity / pre-training |
| [CPSC2018](http://2018.icbeb.org/Challenge.html) | 500 Hz | 6 877 | 12 | Demographic diversity |

### Expected Output (PTB-XL, leads I+II)

| Split | Patients | Windows (est.) |
|-------|----------|----------------|
| Train | ~15 100 | ~300 000 |
| Val | ~1 900 | ~38 000 |
| Test | ~1 900 | ~38 000 |

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download datasets (see phase1_dataset_preparation/README.md)

# 3. Run Phase 1 pipeline
cd phase1_dataset_preparation
python scripts/run_phase1.py --config config/config.yaml

# 4. Verify output
python scripts/verify_dataset.py --data_dir data/processed
```

---

## Requirements

- Python 3.10+
- See `requirements.txt` for full dependency list

---

## Repository Structure

```
Upsampling-physiological-signals/
├── CLAUDE.md                         ← AI assistant context
├── README.md                         ← this file
├── requirements.txt
└── phase1_dataset_preparation/       ← Phase 1: dataset pipeline
    ├── README.md
    ├── config/config.yaml
    ├── data/
    │   ├── raw/                      ← downloaded datasets go here
    │   └── processed/                ← HDF5 output
    ├── src/                          ← pipeline modules
    └── scripts/                      ← entry points
```
