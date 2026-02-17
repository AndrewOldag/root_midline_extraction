"""
Generate a lightweight summary of training results for remote review.

Loads the best model, runs evaluation on the validation set, generates
prediction visualizations, and packages everything into summary/ at
the project root -- ready to git add, commit, and push.

Produces summary/ containing:
    - summary_report.json    (checkpoint metadata, training stats, eval metrics)
    - training_curves.png    (copied from logs)
    - training_log.csv       (copied from logs)
    - predictions/           (prediction overlay PNGs from evaluation)
    - prediction_montage.png (grid of downscaled prediction images)
    - thumbnails/            (individual downscaled prediction PNGs)

Usage:
    python summarize_output.py
    python summarize_output.py --max-vis 20       # more prediction images
    python summarize_output.py --skip-eval         # skip evaluation, just summarize existing files
"""

import argparse
import csv
import json
import math
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
import utils
from data_loader import create_dataloaders
from evaluate import dice_coefficient, qc_distance_error, qc_mse
from model import build_model

SUMMARY_DIR = config.PROJECT_ROOT / "summary"
THUMBNAIL_DIR = SUMMARY_DIR / "thumbnails"
PREDICTIONS_SAVE_DIR = SUMMARY_DIR / "predictions"
THUMBNAIL_WIDTH = 400


# =============================================================================
# Evaluation  (generates prediction images + metrics)
# =============================================================================


@torch.no_grad()
def run_evaluation(
    max_vis: int = 20,
) -> Dict:
    """
    Load best model, evaluate on validation set, save prediction images
    directly into summary/predictions/.

    Returns dict with per-sample metrics and summary statistics.
    """
    device = utils.get_device()

    # Load data
    print("  Loading validation data...")
    _, val_loader = create_dataloaders()
    print(f"  Validation samples: {len(val_loader.dataset)}")

    # Find best checkpoint
    ckpt_path = config.CHECKPOINT_DIR / "best_model.pth"
    if not ckpt_path.exists():
        pth_files = sorted(config.CHECKPOINT_DIR.glob("*.pth"))
        if not pth_files:
            return {"error": "No checkpoint found"}
        ckpt_path = pth_files[-1]

    print(f"  Loading model from {ckpt_path.name}...")
    model = build_model(device)
    checkpoint = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    ckpt_epoch = checkpoint.get("epoch", "?")
    ckpt_val_loss = checkpoint.get("val_loss", "?")
    print(f"  Checkpoint: epoch {ckpt_epoch}, val_loss={ckpt_val_loss}")

    # Evaluate + generate visualisation images
    PREDICTIONS_SAVE_DIR.mkdir(parents=True, exist_ok=True)

    all_dice = []
    all_qc_dist = []
    all_qc_mse = []
    all_filenames = []
    vis_count = 0

    for batch in tqdm(val_loader, desc="  Evaluating"):
        images = batch["image"].to(device)
        pred = model(images)

        midline_preds = torch.sigmoid(pred["midline"]).cpu().numpy()
        qc_preds = torch.sigmoid(pred["qc"]).cpu().numpy()
        midline_targets = batch["midline_mask"].numpy()
        qc_targets = batch["qc_heatmap"].numpy()

        for i in range(images.size(0)):
            mp = midline_preds[i, 0]
            mt = midline_targets[i, 0]
            qp = qc_preds[i, 0]
            qt = qc_targets[i, 0]
            fname = batch["filename"][i]

            d = dice_coefficient(mp, mt)
            dist = qc_distance_error(qp, qt)
            mse = qc_mse(qp, qt)

            all_dice.append(d)
            all_qc_dist.append(dist)
            all_qc_mse.append(mse)
            all_filenames.append(fname)

            if vis_count < max_vis:
                img_np = batch["image"][i, 0].numpy()
                utils.plot_prediction_overlay(
                    image=img_np,
                    pred_midline=mp,
                    pred_qc=qp,
                    gt_midline=mt,
                    gt_qc=qt,
                    title=f"{fname} | Dice={d:.3f} | QC dist={dist:.1f}px",
                    save_path=str(PREDICTIONS_SAVE_DIR / f"{fname}_prediction.png"),
                )
                vis_count += 1

    dice_arr = np.array(all_dice)
    dist_arr = np.array(all_qc_dist)
    mse_arr = np.array(all_qc_mse)

    metrics = {
        "checkpoint_used": ckpt_path.name,
        "checkpoint_epoch": ckpt_epoch,
        "num_samples": len(all_dice),
        "num_visualizations": vis_count,
        "dice": {
            "mean": round(float(dice_arr.mean()), 4),
            "std": round(float(dice_arr.std()), 4),
            "min": round(float(dice_arr.min()), 4),
            "max": round(float(dice_arr.max()), 4),
        },
        "qc_distance_px": {
            "mean": round(float(dist_arr.mean()), 2),
            "std": round(float(dist_arr.std()), 2),
            "min": round(float(dist_arr.min()), 2),
            "max": round(float(dist_arr.max()), 2),
            "median": round(float(np.median(dist_arr)), 2),
        },
        "qc_mse": {
            "mean": round(float(mse_arr.mean()), 6),
        },
        "per_sample": [
            {
                "filename": fn,
                "dice": round(d, 4),
                "qc_dist_px": round(qd, 2),
            }
            for fn, d, qd in zip(all_filenames, all_dice, all_qc_dist)
        ],
    }

    print(f"  Dice:  mean={metrics['dice']['mean']:.4f}  "
          f"min={metrics['dice']['min']:.4f}  max={metrics['dice']['max']:.4f}")
    print(f"  QC dist: mean={metrics['qc_distance_px']['mean']:.1f}px  "
          f"median={metrics['qc_distance_px']['median']:.1f}px")
    print(f"  Saved {vis_count} prediction images")

    return metrics


# =============================================================================
# Checkpoint metadata
# =============================================================================


def extract_checkpoint_metadata() -> list:
    """Read epoch and val_loss from each .pth checkpoint without loading weights."""
    checkpoint_dir = config.CHECKPOINT_DIR
    if not checkpoint_dir.exists():
        return []

    metadata = []
    for pth_file in sorted(checkpoint_dir.glob("*.pth")):
        try:
            ckpt = torch.load(str(pth_file), map_location="cpu", weights_only=False)
            metadata.append({
                "file": pth_file.name,
                "size_mb": round(pth_file.stat().st_size / (1024 * 1024), 1),
                "epoch": ckpt.get("epoch", None),
                "val_loss": round(ckpt.get("val_loss", float("nan")), 6),
            })
        except Exception as exc:
            metadata.append({
                "file": pth_file.name,
                "size_mb": round(pth_file.stat().st_size / (1024 * 1024), 1),
                "error": str(exc),
            })
    return metadata


# =============================================================================
# Training log summary
# =============================================================================


def summarize_training_log() -> dict:
    """Parse training_log.csv and return key stats."""
    log_path = config.LOG_DIR / "training_log.csv"
    if not log_path.exists():
        return {"error": "training_log.csv not found"}

    rows = []
    with open(log_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        return {"error": "training_log.csv is empty"}

    last = rows[-1]
    total_epochs = len(rows)
    best_val_row = min(rows, key=lambda r: float(r.get("val_total", "inf")))

    return {
        "total_epochs_logged": total_epochs,
        "final_epoch": {
            "epoch": last.get("epoch"),
            "train_total": last.get("train_total"),
            "val_total": last.get("val_total"),
            "train_midline": last.get("train_midline"),
            "train_qc": last.get("train_qc"),
            "val_midline": last.get("val_midline"),
            "val_qc": last.get("val_qc"),
            "lr": last.get("lr"),
        },
        "best_val_epoch": {
            "epoch": best_val_row.get("epoch"),
            "val_total": best_val_row.get("val_total"),
            "train_total": best_val_row.get("train_total"),
        },
        "configured_max_epochs": config.EPOCHS,
        "early_stopped": total_epochs < config.EPOCHS,
    }


# =============================================================================
# File inventory
# =============================================================================


def build_file_inventory() -> dict:
    """Count and measure files in each output subdirectory."""
    inventory = {}
    for subdir_name in ("checkpoints", "logs", "predictions"):
        subdir = config.OUTPUT_DIR / subdir_name
        if not subdir.exists():
            inventory[subdir_name] = {"exists": False}
            continue
        files = list(subdir.iterdir())
        total_bytes = sum(f.stat().st_size for f in files if f.is_file())
        inventory[subdir_name] = {
            "exists": True,
            "file_count": len([f for f in files if f.is_file()]),
            "total_size_mb": round(total_bytes / (1024 * 1024), 1),
        }
    return inventory


# =============================================================================
# Thumbnails and montage
# =============================================================================


def generate_thumbnails() -> list:
    """Downscale prediction images from summary/predictions/ into thumbnails/."""
    if not PREDICTIONS_SAVE_DIR.exists():
        return []

    image_files = sorted(PREDICTIONS_SAVE_DIR.glob("*.png"))
    if not image_files:
        return []

    THUMBNAIL_DIR.mkdir(parents=True, exist_ok=True)
    thumb_paths = []

    for img_path in image_files:
        img = Image.open(img_path)
        ratio = THUMBNAIL_WIDTH / img.width
        new_h = int(img.height * ratio)
        thumb = img.resize((THUMBNAIL_WIDTH, new_h), Image.LANCZOS)

        out_path = THUMBNAIL_DIR / img_path.name
        thumb.save(str(out_path), optimize=True)
        thumb_paths.append(out_path)

    return thumb_paths


def build_montage(thumb_paths: list) -> None:
    """Tile thumbnail images into a single montage grid."""
    if not thumb_paths:
        return

    images = [Image.open(p) for p in thumb_paths]
    n = len(images)

    cols = min(n, 4)
    rows = math.ceil(n / cols)

    thumb_w = images[0].width
    thumb_h = images[0].height
    padding = 6

    montage_w = cols * thumb_w + (cols + 1) * padding
    montage_h = rows * thumb_h + (rows + 1) * padding
    montage = Image.new("RGB", (montage_w, montage_h), color=(30, 30, 30))

    for idx, img in enumerate(images):
        row = idx // cols
        col = idx % cols
        x = padding + col * (thumb_w + padding)
        y = padding + row * (thumb_h + padding)
        montage.paste(img, (x, y))

    montage.save(str(SUMMARY_DIR / "prediction_montage.png"), optimize=True)
    print(f"  Montage saved ({cols}x{rows} grid, {n} images)")


# =============================================================================
# Copy small log files
# =============================================================================


def copy_log_files() -> list:
    """Copy training_curves.png and training_log.csv into summary/."""
    copied = []
    for filename in ("training_curves.png", "training_log.csv"):
        src = config.LOG_DIR / filename
        if src.exists():
            shutil.copy2(str(src), str(SUMMARY_DIR / filename))
            copied.append(filename)
    return copied


# =============================================================================
# Main
# =============================================================================


def main(max_vis: int = 20, skip_eval: bool = False) -> None:
    print("=" * 60)
    print("Output Summary Generator")
    print("=" * 60)

    if not config.OUTPUT_DIR.exists():
        print(f"ERROR: output directory not found at {config.OUTPUT_DIR}")
        return

    # Clean previous summary and recreate
    if SUMMARY_DIR.exists():
        shutil.rmtree(SUMMARY_DIR)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Run evaluation (model + data -> prediction images + metrics)
    eval_metrics = None
    if not skip_eval:
        print("\n[1/5] Running evaluation on validation set...")
        try:
            eval_metrics = run_evaluation(max_vis=max_vis)
        except Exception as exc:
            print(f"  Evaluation failed: {exc}")
            print("  Continuing with remaining summary steps...")
            eval_metrics = {"error": str(exc)}
    else:
        print("\n[1/5] Skipping evaluation (--skip-eval)")

    # 2. Checkpoint metadata
    print("\n[2/5] Extracting checkpoint metadata...")
    checkpoint_meta = extract_checkpoint_metadata()
    if checkpoint_meta:
        for m in checkpoint_meta:
            epoch_str = f"epoch {m.get('epoch', '?')}" if "epoch" in m else "?"
            print(f"  {m['file']:40s}  {m['size_mb']:>7.1f} MB  ({epoch_str})")
    else:
        print("  No checkpoints found.")

    # 3. Training log summary
    print("\n[3/5] Summarizing training log...")
    training_summary = summarize_training_log()
    if "error" not in training_summary:
        print(f"  Epochs logged: {training_summary['total_epochs_logged']}")
        print(f"  Best val loss: {training_summary['best_val_epoch']['val_total']} "
              f"(epoch {training_summary['best_val_epoch']['epoch']})")
        print(f"  Early stopped: {training_summary['early_stopped']}")
    else:
        print(f"  {training_summary['error']}")

    # 4. Thumbnails and montage from prediction images
    print("\n[4/5] Generating thumbnails and montage...")
    thumb_paths = generate_thumbnails()
    if thumb_paths:
        print(f"  Created {len(thumb_paths)} thumbnails ({THUMBNAIL_WIDTH}px wide)")
        build_montage(thumb_paths)
    else:
        print("  No prediction images to thumbnail.")

    # 5. Copy log files and write report
    print("\n[5/5] Copying log files and writing summary report...")
    copied = copy_log_files()
    for f in copied:
        print(f"  Copied {f}")

    inventory = build_file_inventory()

    report = {
        "generated_at": datetime.now().isoformat(),
        "output_dir": str(config.OUTPUT_DIR),
        "file_inventory": inventory,
        "checkpoints": checkpoint_meta,
        "training": training_summary,
        "evaluation": eval_metrics,
    }

    report_path = SUMMARY_DIR / "summary_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Report written to {report_path}")

    # Final size estimate
    summary_bytes = sum(
        f.stat().st_size for f in SUMMARY_DIR.rglob("*") if f.is_file()
    )
    print(f"\n{'=' * 60}")
    print(f"Summary folder: {SUMMARY_DIR}")
    print(f"Total size:     {summary_bytes / (1024 * 1024):.2f} MB")
    print(f"{'=' * 60}")
    print("\nDone! You can now:")
    print(f"  git add summary/")
    print(f"  git commit -m \"Update training summary\"")
    print(f"  git push")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate training summary for remote review")
    parser.add_argument(
        "--max-vis", type=int, default=20,
        help="Max number of prediction images to generate (default: 20)",
    )
    parser.add_argument(
        "--skip-eval", action="store_true",
        help="Skip evaluation; only summarize existing log/checkpoint files",
    )
    args = parser.parse_args()
    main(max_vis=args.max_vis, skip_eval=args.skip_eval)
