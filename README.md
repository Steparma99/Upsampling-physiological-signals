# ECG Super-Resolution — 100 Hz → 500 Hz Neural Upsampling

A two-phase deep learning pipeline for reconstructing high-rate (500 Hz) ECG signals from low-rate (100 Hz) inputs. Frequency content above the input Nyquist limit is genuinely synthesized, not interpolated.

---

## Table of Contents

- [Motivation](#motivation)
- [Architecture Overview](#architecture-overview)
- [Training Pipeline](#training-pipeline)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Dataset Setup](#dataset-setup)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Evaluation](#evaluation)
- [Results](#results)

---

## Motivation

Consumer-grade and wearable ECG devices typically record at 100–125 Hz to conserve battery and storage. This discards all spectral content above 50–60 Hz, including:

- High-frequency components of HRV
- Fine P-wave notch structure (interatrial conduction)
- High-slope portions of QRS useful for His-Purkinje activation timing
- Epsilon waves and late potentials in cardiomyopathies

This project trains a neural network to recover the full 500 Hz bandwidth from 100 Hz recordings, using PTB-XL native dual-rate pairs as the primary supervision signal.

---

## Architecture Overview

```
x_LR (B, 1000)  ──┐
d_map (B, 1000) ──┘ X_in (B, 2, 1000)
                        │
                 ┌──────▼───────┐        f_signal (12) ─→ MLP ─┐
                 │  Multi-Scale │        f_meta   ( 7) ─→ MLP ─┘─→ f_emb (256)
                 │ CNN Encoder  │─────────────────────────────────────────┐
                 │  4 res blocks│                                         │
                 └──────┬───────┘                                         │
                  skips │  patch_seq (B, 20, 256)                         │
                        │                                                 │
                 ┌──────▼───────────────────────────────────────┐         │
                 │          Transformer Encoder                 │◄────────┘
                 │  [f_emb | patch_1 … patch_20] — pre-LN, 6L  │
                 └──────┬───────────────────────────────────────┘
                        │  z_cls (B,256)  z_patches (B,20,256)
                        │
                 ┌──────▼───────┐
                 │  U-Net Decoder (Phase B)                     │
                 │  SubPixel ×5 → ×5 → ×2 → ×5 + FiLM + skips │
                 └──────┬───────┘
                        │
                   x̂_HR (B, 5000)
```

### Encoder

- **Input** — `X_in = [x_LR ‖ d_map]` where `d_map` is the normalized distance to the nearest R-peak at each sample
- **Multi-scale CNN** — 4 residual blocks, each with 3 parallel branches (kernels 3 / 7 / 15) capturing fast QRS transitions, QRS morphology, and slow T/P waves respectively; branches fused via 1×1 conv
- **Patchification** — strided conv produces P=20 temporal patches
- **Transformer** — 6-layer pre-LN bidirectional Transformer; physiological CLS token (f_emb) at position 0, patches at positions 1–20; sinusoidal positional encoding

### Conditioning Vectors

**f_signal** (12 features, computed from x_LR only — no HR leakage):  
`[μ, σ, HR, RMSSD, ΔRR, Skewness, Kurtosis, mean_QRS_area, spectral_slope, ZCR, QTc, T_polarity]`

**f_meta** (7 features, patient metadata):  
`[age_norm, sex, NORM, MI, STTC, CD, HYP]`  
Multi-hot diagnosis. Missing values set to 0/0.5/zeros. During training, metadata is zeroed jointly with configurable probability to ensure robustness to missing inputs.

Both vectors projected via independent MLPs and concatenated to form a 256-dim physiological embedding that conditions the Transformer and decoder at every stage.

### U-Net Decoder (Phase B)

- 4 upsampling stages via **SubPixel (pixel shuffle) convolution** — no transposed convolutions (avoids checkerboard artifacts)
- **FiLM conditioning** at each stage using z_cls: `γ * LN(u) + β`
- Skip connections from CNN encoder at Stage 3 (the only decoder stage at matching temporal resolution N_LR=1000)
- Final 1-channel output at 5000 samples (500 Hz × 10 s)

### PatchGAN Discriminator (Phase B)

- 3 strided convolutional layers with Instance Normalization and LeakyReLU
- Per-patch real/fake output map — no sigmoid (hinge loss)
- Activated only after a reconstruction-only warmup phase

---

## Training Pipeline

### Phase A — Self-Supervised Pre-Training

The encoder learns ECG representations from x_LR alone, without any LR→HR pairs.

**SimCLR (NT-Xent):**  
Two augmented views of each signal are passed through the encoder and a projection head. The contrastive loss pulls same-sample views together and pushes different-sample views apart. L2 normalization is applied inside the loss, not as a model layer.

**MAE (Masked Autoencoder):**  
40% of temporal patches are masked (rhythm-aware: QRS patches have 2× the baseline masking probability). The physiological CLS token (position 0) is never masked. A lightweight MAE decoder reconstructs masked patches; the loss combines per-patch z-score MSE (using target statistics) with a weighted FFT component.

**Augmentations:**  
Gaussian noise · amplitude scaling · baseline wander · powerline interference · segment masking · time warping (with d_map recomputation). Strength is ramped via a cosine curriculum over the first 20 epochs.

**Output:** `encoder.pt` — CNN blocks, Transformer, and feature projection MLPs.  
Projection head and MAE decoder are discarded.

### Phase B — Supervised Fine-Tuning

Uses native PTB-XL LR/HR pairs (100 Hz and 500 Hz, no artificial downsampling).

**Generator loss** (weighted sum):

| Component | Weight | Purpose |
|-----------|--------|---------|
| L1 reconstruction | 10.0 | Pixel-wise fidelity (dominant) |
| FFT magnitude L1 | 1.0 | Forces spectral synthesis above LR Nyquist |
| Multi-scale STFT | 1.0 | QRS detail + global P/T structure |
| Morphological | 5.0 | Focused MSE on PQRST landmark windows |
| Adversarial (hinge) | 1.0 | Perceptual realism |
| Feature matching | 2.0 | GAN training stability |

**Transfer strategy:**  
Encoder frozen for the first 20 epochs. Then progressively unfrozen with layer-wise LR decay (deeper layers get lower LR). Adversarial loss also activates at epoch 20, so GAN gradients always flow through a fully trainable generator.

**Output:** `encoder_finetuned.pt` + `decoder.pt`

---

## Repository Structure

```
ecg-superresolution/
├── README.md
├── requirements.txt
├── configs/
│   └── default.yaml              # All hyperparameters (never hardcoded in model files)
├── src/
│   ├── data/
│   │   ├── preprocessing.py      # Normalization, LR/HR pair construction, R-distance map
│   │   ├── features.py           # f_signal and f_meta extraction
│   │   ├── augmentations.py      # ECG-specific augmentations for contrastive learning
│   │   └── dataset.py            # PTBXLDataset, build_training_tuple, collate_fn
│   ├── models/
│   │   ├── encoder.py            # Multi-scale CNN + Transformer encoder
│   │   ├── decoder.py            # U-Net with SubPixel conv + FiLM conditioning
│   │   ├── discriminator.py      # PatchGAN discriminator
│   │   ├── projection_head.py    # SimCLR projection head (Phase A only)
│   │   ├── mae_decoder.py        # MAE decoder (Phase A only)
│   │   └── film.py               # FiLM conditioning module
│   ├── losses/
│   │   ├── contrastive.py        # NT-Xent loss
│   │   ├── mae_loss.py           # MAE reconstruction + FFT loss
│   │   ├── reconstruction.py     # L1, FFT, multi-scale STFT
│   │   ├── morphological.py      # PQRST landmark loss
│   │   └── adversarial.py        # Hinge loss + feature matching
│   ├── training/
│   │   ├── phase_a.py            # Pre-training loop (SimCLR + MAE)
│   │   └── phase_b.py            # Fine-tuning loop (Generator + Discriminator)
│   └── utils/
│       └── metrics.py            # SNR, PRD, RMSE, SSIM evaluation
├── scripts/
│   ├── train_phase_a.py          # Phase A entry point
│   ├── train_phase_b.py          # Phase B entry point
│   └── evaluate.py               # Offline evaluation and comparison plots
└── tests/
    ├── conftest.py
    ├── test_dimensions.py        # Synthetic forward pass, tensor shape verification
    └── test_dataset.py           # Data integrity: normalization, d_map, metadata dropout
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/Steparma99/Upsampling-physiological-signals.git
cd Upsampling-physiological-signals

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate          # Linux/macOS
# venv\Scripts\Activate.ps1       # Windows PowerShell

# Install dependencies
pip install -r requirements.txt
```

**Requirements:** Python 3.10+ · PyTorch 2.2+ · CUDA recommended for training

---

## Dataset Setup

### PTB-XL (Primary — required)

PTB-XL provides native 100 Hz and 500 Hz recordings for the same patients — the only dataset for which no artificial downsampling is applied.

```bash
# Download from PhysioNet (requires physionet credentials or wget)
wget -r -N -c -np \
    https://physionet.org/files/ptb-xl/1.0.3/ \
    -P ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/
```

Or download manually from: https://physionet.org/content/ptb-xl/1.0.3/

The dataset root must contain `ptbxl_database.csv`, `records100/`, and `records500/`.

**Splits** (stratified by patient, PTB-XL fold assignment):

| Split | Folds | Patients (approx.) |
|-------|-------|-------------------|
| Train | 1–8   | ~15 100           |
| Val   | 9     | ~1 890            |
| Test  | 10    | ~1 890            |

### Additional Datasets (optional)

| Dataset | Hz | Records | Use |
|---------|----|---------|-----|
| [MIT-BIH Arrhythmia](https://physionet.org/content/mitdb/1.0.0/) | 360 | 48 | Arrhythmia diversity (simulated downsampling) |
| [CPSC2018](http://2018.icbeb.org/Challenge.html) | 500 | 6 877 | Demographic diversity |

For non-native datasets, FIR anti-aliasing (Kaiser, f_c=50 Hz, A_s≥80 dB) is applied **before** decimation.

---

## Quick Start

### Run Tests

```bash
python -m pytest tests/ -v
```

All 32 tests should pass. Key checks: tensor shapes end-to-end, no HR leakage in normalization, all-or-nothing metadata dropout, d_map non-negativity, annotation padding sentinel.

### Phase A — Self-Supervised Pre-Training

```bash
python scripts/train_phase_a.py \
    --data ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3 \
    --save_dir runs/phase_a \
    --gpu 0 \
    --seed 42
```

Checkpoints are saved every epoch to `runs/phase_a/last_phase_a.pt`. Best validation loss is saved as `runs/phase_a/best_phase_a.pt`. The final encoder weights are exported to `runs/phase_a/encoder.pt`.

Resume an interrupted run:

```bash
python scripts/train_phase_a.py \
    --data ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3 \
    --save_dir runs/phase_a \
    --resume runs/phase_a/last_phase_a.pt
```

### Phase B — Supervised Fine-Tuning

```bash
python scripts/train_phase_b.py \
    --data ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3 \
    --encoder_ckpt runs/phase_a/encoder.pt \
    --save_dir runs/phase_b \
    --gpu 0 \
    --seed 42
```

Outputs: `runs/phase_b/encoder_finetuned.pt` and `runs/phase_b/decoder.pt`.

### Disable W&B

```bash
python scripts/train_phase_a.py ... --no_wandb
```

---

## Configuration

All hyperparameters are stored in `configs/default.yaml` and loaded via [OmegaConf](https://omegaconf.readthedocs.io/). Model files never hardcode dimensions or loss weights.

Key parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model.d_model` | 256 | Encoder/decoder channel width |
| `model.encoder.n_residual_blocks` | 4 | CNN depth |
| `model.transformer.n_layers` | 6 | Transformer depth |
| `phase_a.batch_size` | 256 | Contrastive learning benefits from large batches |
| `phase_a.simclr.temperature` | 0.07 | NT-Xent temperature |
| `phase_a.mae.masking_ratio` | 0.40 | Fraction of patches masked during MAE |
| `phase_b.encoder_freeze_epochs` | 20 | Epochs before progressive encoder unfreezing |
| `data.p_drop_meta` | 0.30 | Probability of zeroing entire f_meta during training |
| `data.channel` | 0 | ECG lead index (0 = Lead I for PTB-XL) |

---

## Evaluation

```bash
python scripts/evaluate.py \
    --encoder_ckpt runs/phase_b/encoder_finetuned.pt \
    --decoder_ckpt runs/phase_b/decoder.pt \
    --data ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3 \
    --split test \
    --save_dir runs/eval
```

Reported metrics:

| Metric | Description |
|--------|-------------|
| **SNR** (dB) | Signal-to-noise ratio of reconstructed vs. ground-truth HR signal |
| **PRD** (%) | Percent root-mean-square difference |
| **RMSE** | Root mean square error (normalized) |
| **RMSE (mV)** | RMSE in physical units after denormalization |
| **SSIM** | Structural similarity index |

Save comparison plots (LR input / HR predicted / HR ground truth):

```bash
python scripts/evaluate.py \
    --encoder_ckpt runs/phase_b/encoder_finetuned.pt \
    --decoder_ckpt runs/phase_b/decoder.pt \
    --data ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3 \
    --split test \
    --save_plots runs/eval_plots \
    --n_plots 32
```

---

## Results

> Training in progress — results will be published here upon completion.

---

## License

This project uses PTB-XL data released under the [Open Data Commons Attribution License (ODC-By) v1.0](https://physionet.org/content/ptb-xl/1.0.3/LICENSE.txt).  
Code in this repository is released under the MIT License.
