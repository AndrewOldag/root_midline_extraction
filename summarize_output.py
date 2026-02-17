"""
Generate a lightweight summary of the output/ folder for remote review.

Produces output/summary/ containing:
    - summary_report.json   (checkpoint metadata, training stats, per-prediction info)
    - training_curves.png   (copied from logs)
    - training_log.csv      (copied from logs)
    - prediction_montage.png (grid of downscaled overlay images)
    - thumbnails/           (individual downscaled overlay PNGs)

Usage:
    python summarize_output.py
"""

import csv
import json
import math
import shutil
from datetime import datetime
from pathlib import Path

from PIL import Image

import config

SUMMARY_DIR = config.OUTPUT_DIR / "summary"
THUMBNAIL_DIR = SUMMARY_DIR / "thumbnails"
THUMBNAIL_WIDTH = 400  # pixels -- target width for each thumbnail


# =============================================================================
# Checkpoint metadata
# =============================================================================


def extract_checkpoint_metadata() -> list:
    """Read epoch and val_loss from each .pth checkpoint without loading weights."""
    import torch

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
# Prediction stats
# =============================================================================


def collect_prediction_stats() -> dict:
    """Collect stats from all _coords.json files in predictions/."""
    pred_dir = config.PREDICTIONS_DIR
    if not pred_dir.exists():
        return {"error": "predictions/ directory not found"}

    json_files = sorted(pred_dir.glob("*_coords.json"))
    if not json_files:
        return {"count": 0, "samples": []}

    samples = []
    for jf in json_files:
        with open(jf) as f:
            data = json.load(f)
        samples.append({
            "filename": data.get("filename", jf.stem),
            "qc_point": data.get("qc_point"),
            "midline_num_points": data.get("midline_num_points", 0),
            "crop_bbox": data.get("crop_bbox"),
        })

    midline_counts = [s["midline_num_points"] for s in samples]

    return {
        "count": len(samples),
        "midline_points_min": min(midline_counts) if midline_counts else None,
        "midline_points_max": max(midline_counts) if midline_counts else None,
        "midline_points_mean": round(sum(midline_counts) / len(midline_counts), 1) if midline_counts else None,
        "samples": samples,
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
    """Downscale all _overlay.png images and save to thumbnails/."""
    pred_dir = config.PREDICTIONS_DIR
    if not pred_dir.exists():
        return []

    overlay_files = sorted(pred_dir.glob("*_overlay.png"))
    if not overlay_files:
        return []

    THUMBNAIL_DIR.mkdir(parents=True, exist_ok=True)
    thumb_paths = []

    for overlay in overlay_files:
        img = Image.open(overlay)
        ratio = THUMBNAIL_WIDTH / img.width
        new_h = int(img.height * ratio)
        thumb = img.resize((THUMBNAIL_WIDTH, new_h), Image.LANCZOS)

        out_path = THUMBNAIL_DIR / overlay.name
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


def main() -> None:
    print("=" * 60)
    print("Output Summary Generator")
    print("=" * 60)

    if not config.OUTPUT_DIR.exists():
        print(f"ERROR: output directory not found at {config.OUTPUT_DIR}")
        return

    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Checkpoint metadata
    print("\n[1/5] Extracting checkpoint metadata...")
    checkpoint_meta = extract_checkpoint_metadata()
    if checkpoint_meta:
        for m in checkpoint_meta:
            epoch_str = f"epoch {m.get('epoch', '?')}" if "epoch" in m else "?"
            print(f"  {m['file']:40s}  {m['size_mb']:>7.1f} MB  ({epoch_str})")
    else:
        print("  No checkpoints found.")

    # 2. Training log summary
    print("\n[2/5] Summarizing training log...")
    training_summary = summarize_training_log()
    if "error" not in training_summary:
        print(f"  Epochs logged: {training_summary['total_epochs_logged']}")
        print(f"  Best val loss: {training_summary['best_val_epoch']['val_total']} "
              f"(epoch {training_summary['best_val_epoch']['epoch']})")
        print(f"  Early stopped: {training_summary['early_stopped']}")
    else:
        print(f"  {training_summary['error']}")

    # 3. Prediction stats
    print("\n[3/5] Collecting prediction stats...")
    prediction_stats = collect_prediction_stats()
    if "error" not in prediction_stats:
        print(f"  Predictions: {prediction_stats['count']}")
        if prediction_stats["count"] > 0:
            print(f"  Midline points: min={prediction_stats['midline_points_min']}, "
                  f"max={prediction_stats['midline_points_max']}, "
                  f"mean={prediction_stats['midline_points_mean']}")
    else:
        print(f"  {prediction_stats['error']}")

    # 4. Thumbnails and montage
    print("\n[4/5] Generating thumbnails and montage...")
    thumb_paths = generate_thumbnails()
    if thumb_paths:
        print(f"  Created {len(thumb_paths)} thumbnails ({THUMBNAIL_WIDTH}px wide)")
        build_montage(thumb_paths)
    else:
        print("  No overlay images found to thumbnail.")

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
        "predictions": prediction_stats,
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
    print("\nDone! You can now git add, commit, and push output/summary/")


if __name__ == "__main__":
    main()
