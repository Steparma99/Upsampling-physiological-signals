"""
verify_pipeline.py -- End-to-end sanity check for src/data pipeline.

Loads a real 10-second ECG window from a local dataset (PTB-XL when available),
runs it through the complete preprocessing, feature extraction, augmentation,
and dataset assembly pipeline, then prints every intermediate result and renders
a multi-panel diagnostic plot.

Run from project root:
    python -m src.data.verify_pipeline

Or directly:
    python src/data/verify_pipeline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# ---- Import handling: works both as module (-m) and as direct script --------
try:
    from .preprocessing import (
        FS_HR, FS_LR, L_HR, L_LR,
        apply_normalization, compute_r_distance_map,
        denormalize, detect_r_peaks, downsample_signal,
        zscore_normalize,
    )
    from .features import (
        DIAGNOSTIC_CLASSES, F_META_DIM, F_SIGNAL_DIM,
        build_f_meta, build_f_signal,
    )
    from .augmentations import ECGAugmentation
    from .dataset import PTBXLDataset, SOURCE_NATIVE, build_training_tuple, _ecg_collate_fn
except ImportError:
    _root = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(_root))
    from src.data.preprocessing import (
        FS_HR, FS_LR, L_HR, L_LR,
        apply_normalization, compute_r_distance_map,
        denormalize, detect_r_peaks, downsample_signal,
        zscore_normalize,
    )
    from src.data.features import (
        DIAGNOSTIC_CLASSES, F_META_DIM, F_SIGNAL_DIM,
        build_f_meta, build_f_signal,
    )
    from src.data.augmentations import ECGAugmentation
    from src.data.dataset import PTBXLDataset, SOURCE_NATIVE, build_training_tuple, _ecg_collate_fn

import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless-safe; change to "TkAgg" for interactive
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ---- Formatting helpers -----------------------------------------------------

SEP  = "=" * 62
SEP2 = "-" * 62


def _header(title: str) -> None:
    print(f"\n{SEP}\n  {title}\n{SEP}")


def _ok(msg: str) -> None:
    print(f"  [OK]   {msg}")


def _check(cond: bool, msg: str) -> None:
    tag = "[OK]  " if cond else "[FAIL]"
    print(f"  {tag} {msg}")
    if not cond:
        raise AssertionError(f"Check failed: {msg}")


# =============================================================================
# STEP 1 -- Real ECG loading
# =============================================================================

def _default_ptbxl_root() -> Path:
    return Path(__file__).resolve().parents[2] / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"


def _load_dataset_ecg(sample_idx: int = 0, split: str = "train") -> tuple[
    np.ndarray,
    np.ndarray,
    dict[str, object],
]:
    """Load one real ECG example from the local PTB-XL dataset."""
    root = _default_ptbxl_root()
    if not root.exists():
        raise FileNotFoundError(
            "PTB-XL root not found. Expected local dataset at "
            f"{root}"
        )

    dataset = PTBXLDataset(root=root, split=split, channel=0, p_drop_meta=0.0, seed=42)
    if len(dataset) == 0:
        raise RuntimeError(f"PTBXLDataset at {root} is empty for split={split!r}.")

    idx = int(np.clip(sample_idx, 0, len(dataset) - 1))
    signal_hr, signal_lr, meta = dataset._load_record(idx)
    row = dataset._metadata.iloc[idx]
    info: dict[str, object] = {
        "dataset_name": "PTB-XL",
        "dataset_root": str(root),
        "split": split,
        "sample_idx": idx,
        "ecg_id": int(row.name),
        "filename_hr": str(row["filename_hr"]),
        "filename_lr": str(row["filename_lr"]),
        "age": meta.get("age"),
        "sex": meta.get("sex"),
        "diagnostic_labels": meta.get("diagnostic_labels"),
    }
    return signal_hr, signal_lr, info


# =============================================================================
# Main verification routine
# =============================================================================

def run_verification() -> None:

    # ---- 1. Real dataset ECG -----------------------------------------------
    _header("1 / 7 -- REAL ECG LOADING FROM DATASET")
    sig_hr, sig_lr, sample_info = _load_dataset_ecg(sample_idx=0, split="train")

    _check(sig_hr.shape == (L_HR,),
           f"HR shape {sig_hr.shape} == ({L_HR},)")
    _check(sig_lr.shape == (L_LR,),
           f"LR shape {sig_lr.shape} == ({L_LR},)")
    print(f"  Dataset: {sample_info['dataset_name']}  split={sample_info['split']}  "
          f"sample_idx={sample_info['sample_idx']}  ecg_id={sample_info['ecg_id']}")
    print(f"  HR file: {sample_info['filename_hr']}")
    print(f"  LR file: {sample_info['filename_lr']}")
    print(f"  HR: {L_HR} samples @ {FS_HR} Hz   "
          f"range [{sig_hr.min():.3f}, {sig_hr.max():.3f}]")
    print(f"  LR: {L_LR} samples @ {FS_LR} Hz   "
          f"range [{sig_lr.min():.3f}, {sig_lr.max():.3f}]")
    print(f"  First 12 HR samples: {np.round(sig_hr[:12], 5).tolist()}")
    print(f"  First 12 LR samples: {np.round(sig_lr[:12], 5).tolist()}")

    # ---- 2. Normalisation --------------------------------------------------
    _header("2 / 7 -- Z-SCORE NORMALISATION")
    x_lr_hat, mu_w, sigma_w = zscore_normalize(sig_lr)
    x_hr_hat = apply_normalization(sig_hr, mu_w, sigma_w)

    _check(x_lr_hat.dtype == np.float32, "x_LR_hat dtype = float32")
    _check(x_hr_hat.dtype == np.float32, "x_HR_hat dtype = float32")
    _check(abs(float(x_lr_hat.mean())) < 1e-4,
           f"|mean(x_LR_hat)| < 1e-4  (got {abs(float(x_lr_hat.mean())):.2e})")
    _check(abs(float(x_lr_hat.std()) - 1.0) < 0.01,
           f"std(x_LR_hat) ~ 1  (got {x_lr_hat.std():.5f})")

    print(f"  mu_w    = {mu_w:.6f}")
    print(f"  sigma_w = {sigma_w:.6f}")
    print(f"  x_LR_hat:  mean={x_lr_hat.mean():.4f}  std={x_lr_hat.std():.4f}"
          f"  range [{x_lr_hat.min():.2f}, {x_lr_hat.max():.2f}]")
    print(f"  x_HR_hat:  mean={x_hr_hat.mean():.4f}  std={x_hr_hat.std():.4f}"
          f"  range [{x_hr_hat.min():.2f}, {x_hr_hat.max():.2f}]")

    sig_lr_rec = denormalize(x_lr_hat, mu_w, sigma_w)
    max_err = float(np.max(np.abs(sig_lr_rec - sig_lr)))
    _check(max_err < 1e-4,
           f"denormalize() round-trip error < 1e-4  (got {max_err:.2e})")

    # ---- 3. R-peaks + d_map ------------------------------------------------
    _header("3 / 7 -- R-PEAK DETECTION & R-DISTANCE MAP")
    r_peaks = detect_r_peaks(sig_lr, fs=FS_LR)
    d_map   = compute_r_distance_map(r_peaks, length=L_LR)

    _check(len(r_peaks) > 0, f"R-peaks detected: {len(r_peaks)}")
    _check(d_map.shape == (L_LR,),
           f"d_map shape {d_map.shape} == ({L_LR},)")
    _check(float(d_map.min()) >= 0.0,
           f"d_map values >= 0  (min={d_map.min():.4f})")
    _check(float(d_map[r_peaks[0]]) < 0.01,
           f"d_map ~ 0 at first R-peak  (got {d_map[r_peaks[0]]:.4f})")

    hr_bpm = 60.0 / float(np.mean(np.diff(r_peaks) / FS_LR))
    print(f"  R-peaks: {len(r_peaks)}   "
          f"first 5 positions: {r_peaks[:5].tolist()}")
    print(f"  Heart rate from peaks: {hr_bpm:.1f} bpm")
    print(f"  d_map:  min={d_map.min():.4f}  max={d_map.max():.4f}"
          f"  mean={d_map.mean():.4f}")

    # ---- 4. f_signal -------------------------------------------------------
    _header("4 / 7 -- PHYSIOLOGICAL FEATURE VECTOR (f_signal)")
    f_signal = build_f_signal(sig_lr, mu_w, sigma_w, r_peaks, fs=FS_LR)

    _check(f_signal.shape == (F_SIGNAL_DIM,),
           f"f_signal shape {f_signal.shape} == ({F_SIGNAL_DIM},)")
    _check(f_signal.dtype == np.float32, "f_signal dtype = float32")
    _check(np.all(np.isfinite(f_signal)),
           "all f_signal values are finite (no NaN/Inf)")

    feature_names = [
        "mu_w          (global mean)",
        "sigma_w       (global std)",
        "HR            (bpm)",
        "RMSSD         (ms)",
        "delta_RR      (relative)",
        "Skewness      (Fisher)",
        "Kurtosis      (excess)",
        "QRS area      (mean, V*s)",
        "Spectral slope(log-log)",
        "ZCR           (per sample)",
        "QTc           (Bazett, s)",
        "T polarity    (+1/-1)",
    ]
    print(f"\n  {'idx':<5} {'Feature':<35} {'Value':>12}")
    print(f"  {'-' * 55}")
    for i, (name, val) in enumerate(zip(feature_names, f_signal)):
        print(f"  [{i:>2}]  {name:<35} {val:>12.5f}")

    # ---- 5. f_meta ---------------------------------------------------------
    _header("5 / 7 -- METADATA VECTOR (f_meta)")
    age = sample_info["age"]
    sex = sample_info["sex"]
    diag = sample_info["diagnostic_labels"]

    f_meta_full    = build_f_meta(age, sex, diag, p_drop=0.0)
    f_meta_dropped = build_f_meta(age, sex, diag, p_drop=1.0,
                                  rng=np.random.default_rng(0))

    _check(f_meta_full.shape == (F_META_DIM,),
           f"f_meta shape {f_meta_full.shape} == ({F_META_DIM},)")
    _check(f_meta_full.dtype == np.float32, "f_meta dtype = float32")
    _check(np.all(f_meta_dropped == 0.0),
           "p_drop=1.0 -> entire f_meta zeroed jointly (all-or-nothing)")

    meta_names    = ["age/100", "sex"] + list(DIAGNOSTIC_CLASSES)
    age_scaled = 0.0 if age is None else float(age) / 100.0
    sex_value = 0.0 if sex is None else float(sex)
    diag_dict = diag if isinstance(diag, dict) else {}
    meta_expected = [age_scaled, sex_value] + [float(diag_dict.get(c, False)) for c in DIAGNOSTIC_CLASSES]

    print(f"\n  {'idx':<5} {'Field':<12} {'Got':>8}   {'Expected':>8}")
    print(f"  {'-' * 40}")
    for i, (name, val, exp) in enumerate(zip(meta_names, f_meta_full, meta_expected)):
        ok = "[OK]" if abs(float(val) - exp) < 1e-5 else "[FAIL]"
        print(f"  [{i}]   {name:<12} {val:>8.4f}   {exp:>8.4f}  {ok}")

    print(f"\n  Raw metadata from dataset:")
    print(f"    age={age}  sex={sex}  diagnostics={diag_dict}")
    print(f"\n  p_drop=1.0 result: {f_meta_dropped.tolist()}  (all zeros)")
    print(f"\n  NOTE: in training the Dataset uses p_drop=0.3 (from configs/default.yaml)")
    print(f"        at inference p_drop=0.0 (Dataset default) -- no dropout")

    # ---- 6. Full training tuple --------------------------------------------
    _header("6 / 7 -- FULL TRAINING TUPLE (build_training_tuple)")
    tup = build_training_tuple(
        signal_hr=sig_hr,
        signal_lr=sig_lr,
        age=age,
        sex=sex,
        diagnostic_labels=diag,
        pqrst_annotations=None,
        source_flag=SOURCE_NATIVE,
        p_drop_meta=0.0,
    )
    x_lr_t, x_hr_t, d_map_t, fsig_t, fmeta_t, pqrst, mu_t, sigma_t, flag = tup

    checks = [
        ("x_LR_hat",  x_lr_t,  (L_LR,)),
        ("x_HR_hat",  x_hr_t,  (L_HR,)),
        ("d_map",     d_map_t, (L_LR,)),
        ("f_signal",  fsig_t,  (F_SIGNAL_DIM,)),
        ("f_meta",    fmeta_t, (F_META_DIM,)),
    ]
    for name, arr, exp in checks:
        _check(arr.shape == exp,
               f"{name:<12} shape {arr.shape} == {exp}")

    _check(pqrst is None,
           "pqrst_annotations = None  (no annotations available in this dataset loader)")
    _check(abs(float(mu_t) - float(np.float32(mu_w))) < 1e-6,
           f"mu_w preserved in tuple  ({float(mu_t):.6f})")
    _check(flag == SOURCE_NATIVE,
           f"source_flag = SOURCE_NATIVE ({SOURCE_NATIVE})")

    print(f"\n  Tuple element types & shapes:")
    print(f"    x_LR_hat : {x_lr_t.dtype}  {x_lr_t.shape}")
    print(f"    x_HR_hat : {x_hr_t.dtype}  {x_hr_t.shape}")
    print(f"    d_map    : {d_map_t.dtype}  {d_map_t.shape}")
    print(f"    f_signal : {fsig_t.dtype}  {fsig_t.shape}")
    print(f"    f_meta   : {fmeta_t.dtype}  {fmeta_t.shape}")
    print(f"    pqrst    : {pqrst}")
    print(f"    mu_w     : {float(mu_t):.5f}  (float32)")
    print(f"    sigma_w  : {float(sigma_t):.5f}  (float32)")
    print(f"    flag     : {flag}  (0=native, 1=simulated)")

    # Collate 2 samples into a mini-batch
    batch = _ecg_collate_fn([tup, tup])
    _check(batch[0].shape == (2, L_LR),
           f"collated x_lr  {batch[0].shape} == (2, {L_LR})")
    _check(batch[1].shape == (2, L_HR),
           f"collated x_hr  {batch[1].shape} == (2, {L_HR})")
    _check(batch[3].shape == (2, F_SIGNAL_DIM),
           f"collated f_sig {batch[3].shape} == (2, {F_SIGNAL_DIM})")
    _ok("collate_fn: 2-sample mini-batch assembled correctly")

    # ---- 7. Augmentations -- two SimCLR views ------------------------------
    _header("7 / 7 -- AUGMENTATIONS (ECGAugmentation, SimCLR two views)")
    aug_params = {
        "gaussian_noise":    {"p": 1.0, "snr_db_min": 30.0, "snr_db_max": 40.0},
        "amplitude_scaling": {"p": 1.0, "scale_min": 0.8,   "scale_max": 1.2},
        "baseline_wander":   {"p": 1.0, "freq_min": 0.1,    "freq_max": 0.5,
                              "amplitude_frac": 0.08},
        "powerline_noise":   {"p": 1.0, "freq": 50.0,       "amplitude_frac": 0.05},
        "segment_masking":   {"p": 1.0, "max_mask_frac": 0.05},
        "time_warping":      {"p": 1.0, "warp_frac": 0.03,  "n_knots": 4},
    }
    aug = ECGAugmentation(params=aug_params, fs=FS_LR, seed=42)
    view_a, view_b, dmap_a, dmap_b, peaks_a, peaks_b = aug(sig_lr, r_peaks)

    _check(view_a.shape == (L_LR,), f"view_a shape {view_a.shape}")
    _check(view_b.shape == (L_LR,), f"view_b shape {view_b.shape}")
    _check(dmap_a.shape == (L_LR,), f"d_map_a shape {dmap_a.shape}")
    _check(not np.allclose(view_a, view_b),
           "view_a != view_b  (independent random augmentation draws)")
    _check(len(peaks_a) > 0, f"R-peaks re-detected in view_a: {len(peaks_a)}")
    _check(len(peaks_b) > 0, f"R-peaks re-detected in view_b: {len(peaks_b)}")

    print(f"  view_a:  mean={view_a.mean():.4f}  std={view_a.std():.4f}"
          f"  R-peaks={len(peaks_a)}")
    print(f"  view_b:  mean={view_b.mean():.4f}  std={view_b.std():.4f}"
          f"  R-peaks={len(peaks_b)}")
    print(f"  d_map_a: min={dmap_a.min():.4f}  max={dmap_a.max():.4f}")

    # ---- Summary -----------------------------------------------------------
    print(f"\n{SEP}")
    print("  ALL CHECKS PASSED -- pipeline is correctly wired.")
    print(SEP)

    # ---- Plot --------------------------------------------------------------
    _plot_diagnostics(
        sig_hr=sig_hr,
        x_lr_hat=x_lr_hat,
        r_peaks=r_peaks,
        d_map=d_map,
        f_signal=f_signal,
        f_meta_full=f_meta_full,
        view_a=view_a,
        view_b=view_b,
        feature_names=feature_names,
        meta_names=meta_names,
        sample_info=sample_info,
    )


# =============================================================================
# Diagnostic plot
# =============================================================================

def _plot_diagnostics(
    sig_hr, x_lr_hat, r_peaks, d_map,
    f_signal, f_meta_full,
    view_a, view_b,
    feature_names, meta_names, sample_info,
) -> None:
    t_lr = np.arange(L_LR) / FS_LR
    t_hr = np.arange(L_HR) / FS_HR

    fig = plt.figure(figsize=(16, 18))
    fig.suptitle(
        "ECG Super-Resolution Pipeline -- Verification on Real Dataset ECG",
        fontsize=14, fontweight="bold",
    )
    gs = gridspec.GridSpec(5, 2, figure=fig, hspace=0.58, wspace=0.35)

    # Panel 1: Raw HR signal
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(t_hr, sig_hr, color="#2196F3", linewidth=0.8, label="x_HR  (500 Hz, raw)")
    ax1.set_title(
        f"Raw HR signal ({sample_info['dataset_name']} ecg_id={sample_info['ecg_id']}, 500 Hz)",
        fontsize=10,
    )
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude (a.u.)")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Normalised LR + R-peaks
    ax2 = fig.add_subplot(gs[1, :])
    ax2.plot(t_lr, x_lr_hat, color="#4CAF50", linewidth=0.9,
             label="x_LR_hat  (z-score normalised, 100 Hz)")
    if len(r_peaks) > 0:
        ax2.scatter(r_peaks / FS_LR, x_lr_hat[r_peaks],
                    color="red", s=50, zorder=5,
                    label=f"R-peaks  (n={len(r_peaks)})")
    ax2.set_title("Normalised LR signal + detected R-peaks", fontsize=10)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("z-score")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Panel 3: d_map
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.plot(t_lr, d_map, color="#FF9800", linewidth=0.9)
    ax3.set_title("R-distance map  d_map[m] = min|m-r_k| / RR_mean", fontsize=9)
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Normalised distance")
    ax3.set_ylim(bottom=0)
    ax3.grid(True, alpha=0.3)

    # Panel 4: Two SimCLR views
    ax4 = fig.add_subplot(gs[2, 1])
    ax4.plot(t_lr, view_a, color="#9C27B0", linewidth=0.7, alpha=0.85, label="view A")
    ax4.plot(t_lr, view_b, color="#F44336", linewidth=0.7, alpha=0.85, label="view B")
    ax4.set_title("SimCLR augmented views (all 6 augmentations active)", fontsize=9)
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("Amplitude")
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    # Panel 5: f_signal bar chart
    ax5 = fig.add_subplot(gs[3, :])
    short_names = [
        "mu_w", "sigma_w", "HR\n(bpm)", "RMSSD\n(ms)", "dRR",
        "Skew", "Kurt", "QRS\narea", "Slope", "ZCR", "QTc\n(s)", "T_pol",
    ]
    colors5 = ["#2196F3" if v >= 0 else "#F44336" for v in f_signal]
    bars5 = ax5.bar(range(F_SIGNAL_DIM), f_signal, color=colors5,
                    edgecolor="white", linewidth=0.5)
    ax5.set_xticks(range(F_SIGNAL_DIM))
    ax5.set_xticklabels(short_names, fontsize=8)
    ax5.set_title(
        f"f_signal in R^{F_SIGNAL_DIM} -- Physiological conditioning vector",
        fontsize=10,
    )
    ax5.set_ylabel("Value")
    ax5.axhline(0, color="black", linewidth=0.5)
    ax5.grid(True, axis="y", alpha=0.3)
    for bar, val in zip(bars5, f_signal):
        va = "bottom" if float(val) >= 0 else "top"
        ax5.text(
            bar.get_x() + bar.get_width() / 2,
            float(bar.get_height()),
            f"{val:.2f}", ha="center", va=va, fontsize=7,
        )

    # Panel 6: f_meta bar chart
    ax6 = fig.add_subplot(gs[4, :])
    colors6 = ["#FF9800" if v > 0 else "#BDBDBD" for v in f_meta_full]
    bars6 = ax6.bar(range(F_META_DIM), f_meta_full, color=colors6,
                    edgecolor="white", linewidth=0.5)
    ax6.set_xticks(range(F_META_DIM))
    ax6.set_xticklabels(meta_names, fontsize=9)
    ax6.set_ylim(-0.1, 1.35)
    ax6.set_title(
        f"f_meta in R^{F_META_DIM} -- Metadata vector"
        f"  (p_drop=0.3 in training: entire vector zeroed jointly)",
        fontsize=10,
    )
    ax6.set_ylabel("Value")
    ax6.grid(True, axis="y", alpha=0.3)
    for bar, val in zip(bars6, f_meta_full):
        ax6.text(
            bar.get_x() + bar.get_width() / 2,
            float(val) + 0.04,
            f"{val:.2f}", ha="center", va="bottom", fontsize=9,
        )

    out_path = Path("pipeline_verification.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n  Plot saved -> {out_path.resolve()}")
    plt.close(fig)


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    run_verification()
