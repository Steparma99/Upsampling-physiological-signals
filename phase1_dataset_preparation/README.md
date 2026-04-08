# Phase 1 — Supervised Dataset Construction

Builds the paired dataset **(x_LR, x_HR)** for supervised training of the ECG upsampling network.

- **x_HR**: original ECG at f_HR = 500 Hz (ground truth)  
- **x_LR**: physically-degraded version at f_LR = 100 Hz  
- **Upsampling factor**: R = f_HR / f_LR = 5

---

## 1. Datasets

### 1.1 Selection Criteria

A dataset is suitable if it simultaneously satisfies:
1. **f_s ≥ f_HR = 500 Hz** — real high-resolution ground truth
2. **Reliable PQRST annotations** — needed for morphological loss in Phase 4
3. **Clinical diversity** — healthy subjects, arrhythmias, demographic variety

### 1.2 Download Instructions

#### PTB-XL (primary, 500 Hz, 18 885 patients)
```bash
# Requires PhysioNet account (free)
wget -r -N -c -np https://physionet.org/files/ptb-xl/1.0.3/ -P data/raw/ptbxl/
# OR use the physionet client:
pip install physionet-client
physionet-client dl --dataset ptb-xl --version 1.0.3 -d data/raw/ptbxl/
```

#### PTB Diagnostic ECG (1000 Hz, 290 patients)
```bash
wget -r -N -c -np https://physionet.org/files/ptbdb/1.0.0/ -P data/raw/ptb_diagnostic/
```

#### MIT-BIH Arrhythmia (360 Hz, 47 patients)
```bash
wget -r -N -c -np https://physionet.org/files/mitdb/1.0.0/ -P data/raw/mitbih/
# OR via wfdb Python:
python -c "import wfdb; wfdb.dl_database('mitdb', 'data/raw/mitbih/')"
```

#### CPSC2018 (500 Hz, 6 877 patients)
```
Download from: http://2018.icbeb.org/Challenge.html
Extract to: data/raw/cpsc2018/
Expected structure:
  data/raw/cpsc2018/
    ├── TrainingSet1/   (WFDB records)
    └── ...
```

### 1.3 Usage Strategy

| Dataset | f_s | Supervised Role | Notes |
|---------|-----|----------------|-------|
| PTB-XL | 500 Hz | Primary (×5) | Leads I and II by default |
| PTB Diagnostic | 1000 Hz | High-fidelity (×10 → resampled to 500 Hz as x_HR) | |
| MIT-BIH | 360 Hz | Unsupervised pre-training only | f_s < 500 Hz, pseudo-pairs only |
| CPSC2018 | 500 Hz | Supplementary (×5) | Asian demographic |

> **MIT-BIH note**: since f_s = 360 Hz < f_HR = 500 Hz, it cannot provide real 500 Hz ground truth. It is used for unsupervised pre-training by creating pseudo-pairs at 360 Hz → 72 Hz (×5).

---

## 2. Degrading Pipeline — D = Q ∘ N ∘ S ∘ F

Each x_HR goes through four sequential stages to produce x_LR:

### Stage F — Anti-Aliasing Filter
FIR linear-phase filter (Kaiser window, Type I) removing all energy above the LR Nyquist frequency f_N^LR = f_LR / 2 = 50 Hz.

| Parameter | Value |
|-----------|-------|
| Cutoff f_c (normalized) | 50/500 = 0.1 |
| Stopband attenuation A_s | ≥ 80 dB |
| Kaiser β | 0.1102 × (80 − 8.7) ≈ 7.857 |
| Phase | Linear (zero-phase via `filtfilt`) |

### Stage S — Decimation
Downsample by R = 5 (stride): x↓[m] = x̃[5m]

### Stage N — Composite Noise
Noise model simulates real acquisition imperfections:

| Component | Model |
|-----------|-------|
| Thermal (white Gaussian) | SNR_dB ~ U(25, 45) dB |
| Powerline interference | 50 Hz sinusoid, amplitude ~ U(0, 0.05) σ_x |
| Baseline wander | 3 sinusoids, f_k ~ U(0.05, 0.8) Hz, A_k ~ U(0, 0.1) σ_x |

### Stage Q — Quantization
12-bit ADC on ±5 mV range:  
Δ = 10 mV / 4095 ≈ 2.44 μV  
(Can be disabled in config — negligible effect for ADC ≥ 12 bit)

---

## 3. Segmentation

- **Window duration**: T_w = 10 s
- **L_HR** = 10 × 500 = 5000 samples
- **L_LR** = 10 × 100 = 1000 samples
- **Overlap**: 50% (stride = 5 s = 2500 HR samples)

### Z-Score Normalization
Computed from x_HR statistics, applied identically to both x_HR and x_LR to preserve the linear scale relationship:

```
μ_w = mean(x_HR)
σ_w = std(x_HR)
x̂_HR = (x_HR − μ_w) / σ_w
x̂_LR = (x_LR − μ_w) / σ_w
```

---

## 4. Artifact Rejection

A window is discarded if **any** of the following criteria is met:

| Criterion | Condition to reject |
|-----------|---------------------|
| Saturation | max(|x̂_HR|) > 5 |
| Variance | σ_w < 0.05 mV or σ_w > 5 mV |
| SQI (Welch) | power([0.5–45 Hz]) / total power < 0.80 |
| RMSSD (Pan-Tompkins) | RMSSD of successive RR diff > 300 ms |

---

## 5. Dataset Split

Split is performed **by patient** (not by window) to prevent data leakage:

- Train: 80% of patients
- Validation: 10% of patients
- Test: 10% of patients

---

## 6. Output Format

Each split is stored as an HDF5 file: `data/processed/{train,val,test}.h5`

| Dataset | Shape | Dtype | Description |
|---------|-------|-------|-------------|
| `x_hr` | (N, 5000) | float32 | Normalized HR windows |
| `x_lr` | (N, 1000) | float32 | Normalized LR windows |
| `r_peaks` | (N,) variable | int32 | R-peak indices in HR samples |
| `mu_w` | (N,) | float64 | Per-window mean (for denormalization) |
| `sigma_w` | (N,) | float64 | Per-window std (for denormalization) |
| `record_ids` | (N,) | string | Source record identifier |
| `window_idxs` | (N,) | int32 | Window index within record |

---

## 7. Running the Pipeline

```bash
# From phase1_dataset_preparation/
python scripts/run_phase1.py --config config/config.yaml

# Options:
#   --config CONFIG     Path to config YAML (default: config/config.yaml)
#   --datasets          Comma-separated list of datasets to process (default: all enabled)
#   --output_dir DIR    Override output directory
#   --dry_run           Process first 10 records only (for testing)
```

### Sanity Check

```bash
python scripts/verify_dataset.py --data_dir data/processed
# Prints statistics and saves diagnostic plots to data/processed/plots/
```

---

## 8. Module Reference

| Module | Responsibility |
|--------|---------------|
| `src/degrading.py` | Anti-aliasing filter + decimation + noise + quantization |
| `src/segmentation.py` | Window extraction + z-score normalization |
| `src/artifact_rejection.py` | All 4 quality checks |
| `src/pan_tompkins.py` | R-peak detection |
| `src/dataset_builder.py` | Per-recording orchestration + HDF5 save |
| `src/loaders/ptbxl_loader.py` | PTB-XL WFDB loader |
| `src/loaders/ptb_diagnostic_loader.py` | PTB Diagnostic WFDB loader |
| `src/loaders/mitbih_loader.py` | MIT-BIH WFDB loader |
| `src/loaders/cpsc2018_loader.py` | CPSC2018 WFDB loader |
| `scripts/run_phase1.py` | Main entry point |
| `scripts/verify_dataset.py` | Post-build diagnostics |
