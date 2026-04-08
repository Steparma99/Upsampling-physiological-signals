"""
Phase 1 — Dataset Verification and Diagnostic Plots

Runs sanity checks on the built HDF5 dataset and produces diagnostic plots.

Usage
-----
From phase1_dataset_preparation/:

    python scripts/verify_dataset.py --data_dir data/processed

Output
------
  Printed summary statistics for each split.
  Plots saved to: data/processed/plots/
    - signal_examples_{split}.png   (random x_HR / x_LR pairs)
    - filter_response.png           (anti-aliasing FIR frequency response)
    - noise_spectrum.png            (averaged PSD of x_HR vs x_LR)
    - rr_intervals.png              (RR interval histogram)
    - rejection_rates.png           (bar chart of rejection reasons)
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

# Ensure Unicode output works on Windows terminals (cp1252 → utf-8)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import h5py
import numpy as np

_PHASE1_DIR = Path(__file__).resolve().parents[1]

logger = logging.getLogger("verify_dataset")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)


# ---------------------------------------------------------------------------
# HDF5 inspection helpers
# ---------------------------------------------------------------------------

def print_split_stats(h5_path: Path) -> dict:
    """Print key statistics for one HDF5 split file."""
    if not h5_path.exists():
        logger.warning("File not found: %s", h5_path)
        return {}

    with h5py.File(h5_path, "r") as f:
        n = f.attrs.get("n_windows", f["x_hr"].shape[0])
        L_HR = f.attrs.get("L_HR", f["x_hr"].shape[1])
        L_LR = f.attrs.get("L_LR", f["x_lr"].shape[1])
        fs_hr = f.attrs.get("fs_hr", 500)
        fs_lr = f.attrs.get("fs_lr", 100)

        # Sample statistics from first 1000 windows
        n_sample = min(1000, n)
        x_hr = f["x_hr"][:n_sample]  # (n_sample, L_HR)
        x_lr = f["x_lr"][:n_sample]  # (n_sample, L_LR)
        mu_w = f["mu_w"][:n_sample]
        sigma_w = f["sigma_w"][:n_sample]

        # R-peak statistics
        all_rr = []
        for i in range(min(200, n)):
            peaks = f["r_peaks"][i]
            if len(peaks) >= 2:
                rr = np.diff(peaks) / float(fs_hr) * 1000.0  # ms
                all_rr.extend(rr.tolist())

        unique_records = np.unique(f["record_ids"][:])

    stats = {
        "file": str(h5_path),
        "n_windows": int(n),
        "L_HR": int(L_HR),
        "L_LR": int(L_LR),
        "fs_hr": int(fs_hr),
        "fs_lr": int(fs_lr),
        "n_unique_records": len(unique_records),
        "x_hr_mean": float(np.mean(x_hr)),
        "x_hr_std": float(np.mean(np.std(x_hr, axis=1))),
        "x_lr_mean": float(np.mean(x_lr)),
        "x_lr_std": float(np.mean(np.std(x_lr, axis=1))),
        "sigma_w_mean_mv": float(np.mean(sigma_w)),
        "sigma_w_p5_mv": float(np.percentile(sigma_w, 5)),
        "sigma_w_p95_mv": float(np.percentile(sigma_w, 95)),
        "rr_mean_ms": float(np.mean(all_rr)) if all_rr else None,
        "rr_std_ms": float(np.std(all_rr)) if all_rr else None,
        "heart_rate_est_bpm": float(60000.0 / np.mean(all_rr)) if all_rr else None,
    }

    print(f"\n{'='*60}")
    print(f"  Split: {h5_path.stem}")
    print(f"{'='*60}")
    print(f"  Windows            : {stats['n_windows']:>10,}")
    print(f"  Unique records     : {stats['n_unique_records']:>10,}")
    print(f"  L_HR / L_LR        : {stats['L_HR']} / {stats['L_LR']} samples")
    print(f"  fs_hr / fs_lr      : {stats['fs_hr']} / {stats['fs_lr']} Hz")
    print(f"  x_HR mean (z-score): {stats['x_hr_mean']:>+10.4f}")
    print(f"  x_HR std  (z-score): {stats['x_hr_std']:>10.4f}  (should be ~1.0)")
    print(f"  x_LR mean (z-score): {stats['x_lr_mean']:>+10.4f}")
    print(f"  σ_w (mV): "
          f"mean={stats['sigma_w_mean_mv']:.3f}  "
          f"p5={stats['sigma_w_p5_mv']:.3f}  "
          f"p95={stats['sigma_w_p95_mv']:.3f}")
    if stats["rr_mean_ms"]:
        print(f"  RR interval (ms)   : {stats['rr_mean_ms']:.1f} ± {stats['rr_std_ms']:.1f}")
        print(f"  Est. heart rate    : {stats['heart_rate_est_bpm']:.1f} bpm")

    return stats


# ---------------------------------------------------------------------------
# Plotting helpers (matplotlib)
# ---------------------------------------------------------------------------

def plot_signal_examples(
    h5_path: Path,
    plot_dir: Path,
    n_examples: int = 4,
    fs_hr: int = 500,
    fs_lr: int = 100,
) -> None:
    """Plot random (x_HR, x_LR) pairs with R-peaks overlaid."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available — skipping signal plots")
        return

    if not h5_path.exists():
        return

    with h5py.File(h5_path, "r") as f:
        n = f["x_hr"].shape[0]
        indices = np.random.default_rng(0).choice(n, size=min(n_examples, n), replace=False)
        x_hr_all = f["x_hr"][indices]
        x_lr_all = f["x_lr"][indices]
        r_peaks_all = [f["r_peaks"][i] for i in indices]

    L_HR = x_hr_all.shape[1]
    L_LR = x_lr_all.shape[1]
    t_hr = np.arange(L_HR) / fs_hr
    t_lr = np.arange(L_LR) / fs_lr

    fig, axes = plt.subplots(n_examples, 1, figsize=(14, 3 * n_examples), sharex=False)
    if n_examples == 1:
        axes = [axes]

    for ax, x_hr, x_lr, r_peaks in zip(axes, x_hr_all, x_lr_all, r_peaks_all):
        ax.plot(t_hr, x_hr, color="steelblue", lw=0.8, label="x_HR (500 Hz)", alpha=0.9)
        ax.plot(t_lr, x_lr, color="tomato", lw=1.0, label="x_LR (100 Hz)", alpha=0.7)
        if len(r_peaks) > 0:
            ax.scatter(
                r_peaks / fs_hr, x_hr[r_peaks],
                color="green", s=25, zorder=5, label="R-peaks",
            )
        ax.set_ylabel("Amplitude (z-score)")
        ax.legend(loc="upper right", fontsize=7)
        ax.grid(True, lw=0.3)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"Signal examples — {h5_path.stem}", fontsize=11)
    fig.tight_layout()

    out_path = plot_dir / f"signal_examples_{h5_path.stem}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved: %s", out_path)


def plot_filter_response(plot_dir: Path, fs_hr: int = 500) -> None:
    """Plot the Kaiser FIR anti-aliasing filter frequency response."""
    try:
        import matplotlib.pyplot as plt
        from scipy import signal as sp_signal
    except ImportError:
        return

    # Re-design filter with spec parameters
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.degrading import design_antialiasing_filter

    h = design_antialiasing_filter(fs_hr=float(fs_hr), fs_lr=100.0)
    w, H = sp_signal.freqz(h, worN=4096, fs=float(fs_hr))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))

    ax1.plot(w, 20 * np.log10(np.abs(H) + 1e-15), color="steelblue")
    ax1.axvline(50.0, color="tomato", ls="--", lw=1, label="f_N^LR = 50 Hz")
    ax1.axhline(-80.0, color="gray", ls=":", lw=0.8, label="−80 dB target")
    ax1.set_xlim(0, fs_hr // 2)
    ax1.set_ylim(-120, 5)
    ax1.set_xlabel("Frequency (Hz)")
    ax1.set_ylabel("Magnitude (dB)")
    ax1.set_title(f"Kaiser FIR Anti-Aliasing Filter  ({len(h)} taps)")
    ax1.legend()
    ax1.grid(True, lw=0.3)

    ax2.plot(w[:len(w)//4], 20 * np.log10(np.abs(H[:len(H)//4]) + 1e-15),
             color="steelblue")
    ax2.axvline(50.0, color="tomato", ls="--", lw=1, label="f_N^LR = 50 Hz")
    ax2.set_xlim(0, 80)
    ax2.set_ylim(-100, 5)
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Magnitude (dB)")
    ax2.set_title("Zoom: transition band region")
    ax2.legend()
    ax2.grid(True, lw=0.3)

    fig.tight_layout()
    out_path = plot_dir / "filter_response.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved: %s", out_path)


def plot_noise_spectra(h5_path: Path, plot_dir: Path, fs_hr: int = 500, fs_lr: int = 100) -> None:
    """Plot averaged PSD of x_HR and x_LR for a random subset of windows."""
    try:
        import matplotlib.pyplot as plt
        from scipy import signal as sp_signal
    except ImportError:
        return

    if not h5_path.exists():
        return

    with h5py.File(h5_path, "r") as f:
        n = min(200, f["x_hr"].shape[0])
        idx = np.arange(n)
        x_hr = f["x_hr"][idx]
        x_lr = f["x_lr"][idx]

    # Average Welch PSD
    from scipy.signal import welch
    f_hr_arr, psd_hr = welch(x_hr[0], fs=fs_hr, nperseg=256)
    f_lr_arr, psd_lr = welch(x_lr[0], fs=fs_lr, nperseg=64)
    for i in range(1, n):
        _, p = welch(x_hr[i], fs=fs_hr, nperseg=256)
        psd_hr += p
        _, p = welch(x_lr[i], fs=fs_lr, nperseg=64)
        psd_lr += p
    psd_hr /= n
    psd_lr /= n

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.semilogy(f_hr_arr, psd_hr, color="steelblue", label="x_HR (500 Hz)")
    ax.semilogy(f_lr_arr, psd_lr, color="tomato", label="x_LR (100 Hz)")
    ax.axvline(50.0, color="gray", ls="--", lw=0.8, label="50 Hz cutoff")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (a.u.)")
    ax.set_title(f"Averaged Power Spectral Density — {h5_path.stem}")
    ax.legend()
    ax.grid(True, lw=0.3)
    fig.tight_layout()

    out_path = plot_dir / f"noise_spectrum_{h5_path.stem}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved: %s", out_path)


def plot_rejection_rates(stats_json_path: Path, plot_dir: Path) -> None:
    """Bar chart of rejection reasons from pipeline_stats.json."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    if not stats_json_path.exists():
        return

    with open(stats_json_path) as f:
        data = json.load(f)

    pipe = data.get("pipeline_stats", {})
    total = pipe.get("total", 1)
    reasons = ["saturation", "variance", "sqi", "rmssd"]
    counts = [pipe.get(f"rejected_{r}", 0) for r in reasons]
    pcts = [100.0 * c / max(total, 1) for c in counts]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(reasons, pcts, color=["#e74c3c", "#e67e22", "#3498db", "#2ecc71"])
    ax.set_ylabel("% of total windows")
    ax.set_title(f"Artifact rejection breakdown  (total={total:,})")
    for bar, pct in zip(bars, pcts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            f"{pct:.1f}%", ha="center", va="bottom", fontsize=9,
        )
    ax.grid(True, axis="y", lw=0.3)
    fig.tight_layout()

    out_path = plot_dir / "rejection_rates.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved: %s", out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 1 — dataset verification")
    parser.add_argument(
        "--data_dir",
        default=str(_PHASE1_DIR / "data" / "processed"),
        help="Directory with HDF5 files (default: <phase1_dir>/data/processed)",
    )
    parser.add_argument(
        "--no_plots",
        action="store_true",
        help="Skip generating diagnostic plots",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)

    if not data_dir.exists():
        logger.error("Data directory not found: %s", data_dir)
        sys.exit(1)

    plot_dir = data_dir / "plots"
    if not args.no_plots:
        plot_dir.mkdir(exist_ok=True)
        # Ensure src is importable
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    all_stats = {}

    # Inspect supervised splits
    for split_name in ("train", "val", "test"):
        h5_path = data_dir / f"{split_name}.h5"
        stats = print_split_stats(h5_path)
        if stats:
            all_stats[split_name] = stats
            if not args.no_plots:
                plot_signal_examples(h5_path, plot_dir)
                plot_noise_spectra(h5_path, plot_dir)

    # Inspect pretraining split
    pretrain_path = data_dir / "mitbih_pretrain.h5"
    stats = print_split_stats(pretrain_path)
    if stats:
        all_stats["mitbih_pretrain"] = stats

    # Global diagnostics
    if not args.no_plots:
        plot_filter_response(plot_dir)
        stats_json = data_dir / "pipeline_stats.json"
        plot_rejection_rates(stats_json, plot_dir)

    # Summary
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    total_windows = sum(s.get("n_windows", 0) for s in all_stats.values())
    print(f"  Total windows across all splits : {total_windows:,}")
    for split, s in all_stats.items():
        print(f"  {split:<20s} : {s.get('n_windows', 0):>10,} windows")
    print()


if __name__ == "__main__":
    main()
