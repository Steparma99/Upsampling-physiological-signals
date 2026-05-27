"""
evaluate.py — Offline evaluation of a trained ECG super-resolution model.

Loads a fine-tuned encoder + U-Net decoder, runs inference on a data split,
computes SNR / PRD / RMSE / SSIM, and optionally saves comparison plots.

Usage
-----
    python scripts/evaluate.py \\
        --encoder_ckpt runs/phase_b/encoder_finetuned.pt \\
        --decoder_ckpt runs/phase_b/decoder.pt \\
        --data /path/to/ptbxl \\
        --split test

    # Save comparison plots (LR upsampled | HR predicted | HR ground truth):
    python scripts/evaluate.py \\
        --encoder_ckpt runs/phase_b/encoder_finetuned.pt \\
        --decoder_ckpt runs/phase_b/decoder.pt \\
        --data /path/to/ptbxl \\
        --split test \\
        --save_plots runs/eval_plots \\
        --n_plots 16

Output
------
    Prints a table of aggregated metrics to stdout.
    Saves results as JSON to <save_dir>/metrics_<split>.json (if --save_dir set).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

# ── Make project root importable regardless of working directory ──────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import PTBXLDataset, get_dataloader
from src.models.decoder import UNetDecoder
from src.models.encoder import ECGEncoder
from src.utils.metrics import evaluate_batch

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# =============================================================================
# Argument parsing
# =============================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate ECG super-resolution model on a data split",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--encoder_ckpt", type=str, required=True,
        help="Path to encoder_finetuned.pt (or encoder.pt for Phase-A-only eval).",
    )
    p.add_argument(
        "--decoder_ckpt", type=str, required=True,
        help="Path to decoder.pt saved by train_phase_b.py.",
    )
    p.add_argument(
        "--data", type=str, required=True,
        help="Path to PTB-XL root directory.",
    )
    p.add_argument(
        "--split", type=str, default="test", choices=["train", "val", "test"],
        help="Dataset split to evaluate.",
    )
    p.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to OmegaConf config YAML.",
    )
    p.add_argument(
        "--gpu", type=int, default=0,
        help="CUDA device index.  Use -1 to force CPU.",
    )
    p.add_argument(
        "--batch_size", type=int, default=32,
        help="Evaluation batch size.",
    )
    p.add_argument(
        "--save_dir", type=str, default=None,
        help="Directory for metrics JSON (optional).",
    )
    p.add_argument(
        "--save_plots", type=str, default=None,
        help="Directory for comparison plots (optional).  Requires matplotlib.",
    )
    p.add_argument(
        "--n_plots", type=int, default=8,
        help="Number of example comparison plots to save.",
    )
    p.add_argument(
        "--num_workers", type=int, default=0,
        help="DataLoader worker processes.",
    )
    return p.parse_args()


# =============================================================================
# Plot helper
# =============================================================================

def _save_comparison_plot(
    x_lr: np.ndarray,     # (L_LR,)  normalised
    x_hr_pred: np.ndarray,# (L_HR,)  normalised
    x_hr_true: np.ndarray,# (L_HR,)  normalised
    mu_w: float,
    sigma_w: float,
    path: Path,
    fs_lr: int = 100,
    fs_hr: int = 500,
) -> None:
    """Save a three-panel comparison plot for one ECG window."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available — skipping plot.")
        return

    # Denormalise for display (millivolts scale)
    x_lr_raw   = x_lr      * sigma_w + mu_w
    x_hr_pred_raw = x_hr_pred * sigma_w + mu_w
    x_hr_true_raw = x_hr_true * sigma_w + mu_w

    t_lr = np.arange(len(x_lr_raw))   / fs_lr
    t_hr = np.arange(len(x_hr_pred_raw)) / fs_hr

    fig, axes = plt.subplots(3, 1, figsize=(14, 6), sharex=True)
    axes[0].plot(t_lr, x_lr_raw,       color="steelblue",  linewidth=0.8)
    axes[0].set_title(f"LR input  ({fs_lr} Hz)",   fontsize=9)
    axes[1].plot(t_hr, x_hr_pred_raw,  color="darkorange", linewidth=0.6)
    axes[1].set_title(f"HR predicted ({fs_hr} Hz)", fontsize=9)
    axes[2].plot(t_hr, x_hr_true_raw,  color="forestgreen", linewidth=0.6)
    axes[2].set_title(f"HR ground truth ({fs_hr} Hz)", fontsize=9)

    for ax in axes:
        ax.set_ylabel("Amplitude")
        ax.margins(x=0)
    axes[-1].set_xlabel("Time (s)")

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = _parse_args()

    # ── Config ────────────────────────────────────────────────────────────────
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    cfg = OmegaConf.load(cfg_path)

    # ── Device ───────────────────────────────────────────────────────────────
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
    logger.info("Evaluating on device: %s", device)

    # ── Dataset ───────────────────────────────────────────────────────────────
    data_root = Path(args.data)
    dataset = PTBXLDataset(
        root=data_root,
        split=args.split,
        channel=cfg.data.channel,
        p_drop_meta=0.0,   # No dropout at inference
    )
    loader = get_dataloader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    logger.info("Evaluating on '%s' split — %d records", args.split, len(dataset))

    # ── Load models ───────────────────────────────────────────────────────────
    encoder = ECGEncoder(cfg).to(device)
    decoder = UNetDecoder(cfg).to(device)

    for ckpt_path, model, name in [
        (args.encoder_ckpt, encoder, "Encoder"),
        (args.decoder_ckpt, decoder, "Decoder"),
    ]:
        path = Path(ckpt_path)
        if not path.exists():
            raise FileNotFoundError(f"{name} checkpoint not found: {path}")
        state = torch.load(path, map_location="cpu", weights_only=True)
        # Accept bare state-dicts or dicts with key 'encoder'/'decoder'/'state_dict'
        if isinstance(state, dict) and not any(
            k.startswith(("cnn_encoder", "transformer", "feature_mlp",
                          "stages", "head", "up"))
            for k in state
        ):
            for key in ("encoder", "decoder", "state_dict", "model_state_dict"):
                if key in state:
                    state = state[key]
                    break
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            logger.warning("%s — missing keys: %s", name, missing[:5])
        if unexpected:
            logger.warning("%s — unexpected keys: %s", name, unexpected[:5])
        logger.info("Loaded %s from %s", name, path)

    encoder.eval()
    decoder.eval()

    # ── Physical constants (same as preprocessing.py) ─────────────────────────
    from src.data.preprocessing import FS_LR, FS_HR

    # ── Evaluation loop ───────────────────────────────────────────────────────
    all_metrics: list[dict[str, float]] = []
    plots_saved = 0
    plot_dir = Path(args.save_plots) if args.save_plots else None

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            x_lr, x_hr, d_map, f_signal, f_meta, _ann, mu_w, sigma_w, _flag = batch

            x_lr     = x_lr.to(device,     non_blocking=True)
            x_hr     = x_hr.to(device,     non_blocking=True)
            d_map    = d_map.to(device,     non_blocking=True)
            f_signal = f_signal.to(device,  non_blocking=True)
            f_meta   = f_meta.to(device,    non_blocking=True)
            mu_w_d   = mu_w.to(device,      non_blocking=True)
            sigma_w_d = sigma_w.to(device,  non_blocking=True)

            # ── Forward pass ─────────────────────────────────────────────────
            X_in = torch.stack([x_lr, d_map], dim=1)          # (B, 2, L_LR)
            _, z_cls, z_patches, skips = encoder(X_in, f_signal, f_meta)
            x_hat = decoder(z_patches, z_cls, skips)           # (B, L_HR)

            # ── Metrics ───────────────────────────────────────────────────────
            metrics = evaluate_batch(x_hat, x_hr, mu_w_d, sigma_w_d)
            all_metrics.append(metrics)

            # ── Optional plots ────────────────────────────────────────────────
            if plot_dir is not None and plots_saved < args.n_plots:
                B = x_lr.shape[0]
                for b in range(B):
                    if plots_saved >= args.n_plots:
                        break
                    _save_comparison_plot(
                        x_lr=x_lr[b].cpu().numpy(),
                        x_hr_pred=x_hat[b].cpu().numpy(),
                        x_hr_true=x_hr[b].cpu().numpy(),
                        mu_w=float(mu_w[b]),
                        sigma_w=float(sigma_w[b]),
                        path=plot_dir / f"sample_{batch_idx:04d}_{b:02d}.png",
                        fs_lr=FS_LR,
                        fs_hr=FS_HR,
                    )
                    plots_saved += 1

    # ── Aggregate metrics ─────────────────────────────────────────────────────
    keys = list(all_metrics[0].keys())
    aggregated: dict[str, float] = {
        k: float(np.mean([m[k] for m in all_metrics if k in m]))
        for k in keys
    }

    # ── Print results ─────────────────────────────────────────────────────────
    sep = "─" * 52
    print(f"\n{sep}")
    print(f"  Evaluation  |  split: {args.split}  |  N batches: {len(all_metrics)}")
    print(sep)
    print(f"  {'Metric':<18}  {'Value':>10}")
    print(sep)
    for k, v in aggregated.items():
        unit = " dB" if "snr" in k else " %" if "prd" in k else "   " if "ssim" in k else " mV" if "mv" in k else "   "
        print(f"  {k:<18}  {v:>10.4f}{unit}")
    print(f"{sep}\n")

    # ── Save results JSON ─────────────────────────────────────────────────────
    if args.save_dir:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        out_path = save_dir / f"metrics_{args.split}.json"
        with open(out_path, "w") as f:
            json.dump(
                {
                    "split":  args.split,
                    "n_batches": len(all_metrics),
                    "metrics": aggregated,
                    "encoder_ckpt": args.encoder_ckpt,
                    "decoder_ckpt": args.decoder_ckpt,
                },
                f, indent=2,
            )
        logger.info("Metrics saved to %s", out_path)

    if plot_dir is not None:
        logger.info("Saved %d comparison plots to %s", plots_saved, plot_dir)


if __name__ == "__main__":
    main()
