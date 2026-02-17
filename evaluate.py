"""
Evaluation metrics and visualization for the root midline / QC model.

Metrics:
    - Dice coefficient for midline segmentation
    - Euclidean distance error for QC localization
    - Mean Squared Error for QC heatmap

Usage:
    python evaluate.py --checkpoint output/checkpoints/best_model.pth
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
import preprocessing
import utils
from data_loader import create_dataloaders
from model import DualHeadUNet, build_model


# =============================================================================
# Metrics
# =============================================================================


def dice_coefficient(
    pred: np.ndarray, target: np.ndarray, threshold: float = 0.5, smooth: float = 1.0
) -> float:
    """
    Compute Dice coefficient between prediction and target.

    Args:
        pred: Predicted mask (H, W), values in [0, 1].
        target: Ground truth mask (H, W), values in {0, 1}.
        threshold: Threshold to binarize prediction.
        smooth: Smoothing term to avoid division by zero.

    Returns:
        Dice score in [0, 1].
    """
    pred_bin = (pred > threshold).astype(np.float32)
    target_bin = target.astype(np.float32)

    intersection = (pred_bin * target_bin).sum()
    return (2.0 * intersection + smooth) / (pred_bin.sum() + target_bin.sum() + smooth)


def qc_distance_error(
    pred_heatmap: np.ndarray, target_heatmap: np.ndarray
) -> float:
    """
    Compute Euclidean distance between predicted and ground truth QC points.

    The QC point is extracted as the peak (argmax) of each heatmap.

    Args:
        pred_heatmap: Predicted QC heatmap (H, W), values in [0, 1].
        target_heatmap: Ground truth QC heatmap (H, W), values in [0, 1].

    Returns:
        Euclidean distance in pixels (at the network output resolution).
    """
    pred_peak = np.unravel_index(np.argmax(pred_heatmap), pred_heatmap.shape)
    target_peak = np.unravel_index(np.argmax(target_heatmap), target_heatmap.shape)

    dist = np.sqrt(
        (pred_peak[0] - target_peak[0]) ** 2 + (pred_peak[1] - target_peak[1]) ** 2
    )
    return float(dist)


def qc_mse(pred_heatmap: np.ndarray, target_heatmap: np.ndarray) -> float:
    """Compute MSE between predicted and target QC heatmaps."""
    return float(np.mean((pred_heatmap - target_heatmap) ** 2))


# =============================================================================
# Evaluation
# =============================================================================


@torch.no_grad()
def evaluate_model(
    model: DualHeadUNet,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, List[float]]:
    """
    Evaluate the model on a DataLoader and compute per-sample metrics.

    Args:
        model: Trained model.
        loader: DataLoader (typically validation set).
        device: Computation device.

    Returns:
        Dict with lists of per-sample metrics:
            - 'dice': Dice coefficients for midline
            - 'qc_dist': QC distance errors (pixels)
            - 'qc_mse': QC heatmap MSE values
            - 'filenames': Sample filenames
    """
    model.eval()
    results = {
        "dice": [],
        "qc_dist": [],
        "qc_mse": [],
        "filenames": [],
    }

    for batch in tqdm(loader, desc="Evaluating"):
        images = batch["image"].to(device)
        pred = model(images)

        # Move predictions to CPU numpy
        midline_preds = torch.sigmoid(pred["midline"]).cpu().numpy()
        qc_preds = torch.sigmoid(pred["qc"]).cpu().numpy()
        midline_targets = batch["midline_mask"].numpy()
        qc_targets = batch["qc_heatmap"].numpy()

        for i in range(images.size(0)):
            mp = midline_preds[i, 0]  # (H, W)
            mt = midline_targets[i, 0]
            qp = qc_preds[i, 0]
            qt = qc_targets[i, 0]

            results["dice"].append(dice_coefficient(mp, mt))
            results["qc_dist"].append(qc_distance_error(qp, qt))
            results["qc_mse"].append(qc_mse(qp, qt))
            results["filenames"].append(batch["filename"][i])

    return results


def print_metrics(results: Dict[str, List[float]]) -> None:
    """Print summary statistics of evaluation results."""
    dices = np.array(results["dice"])
    qc_dists = np.array(results["qc_dist"])
    qc_mses = np.array(results["qc_mse"])

    print("\n" + "=" * 50)
    print("Evaluation Results")
    print("=" * 50)
    print(f"  Samples evaluated: {len(dices)}")
    print()
    print("  Midline Segmentation (Dice):")
    print(f"    Mean:   {dices.mean():.4f}")
    print(f"    Std:    {dices.std():.4f}")
    print(f"    Min:    {dices.min():.4f}")
    print(f"    Max:    {dices.max():.4f}")
    print()
    print("  QC Localization (Distance in pixels at network resolution):")
    print(f"    Mean:   {qc_dists.mean():.2f} px")
    print(f"    Std:    {qc_dists.std():.2f} px")
    print(f"    Min:    {qc_dists.min():.2f} px")
    print(f"    Max:    {qc_dists.max():.2f} px")
    print(f"    Median: {np.median(qc_dists):.2f} px")
    print()
    print("  QC Heatmap (MSE):")
    print(f"    Mean:   {qc_mses.mean():.6f}")
    print("=" * 50)


# =============================================================================
# Visualization
# =============================================================================


@torch.no_grad()
def visualize_predictions(
    model: DualHeadUNet,
    loader: DataLoader,
    device: torch.device,
    save_dir: Path = config.PREDICTIONS_DIR,
    max_samples: int = 10,
) -> None:
    """
    Generate and save prediction visualizations.

    Args:
        model: Trained model.
        loader: DataLoader.
        device: Computation device.
        save_dir: Directory to save visualizations.
        max_samples: Maximum number of samples to visualize.
    """
    model.eval()
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for batch in loader:
        if count >= max_samples:
            break

        images = batch["image"].to(device)
        pred = model(images)

        midline_preds = torch.sigmoid(pred["midline"]).cpu().numpy()
        qc_preds = torch.sigmoid(pred["qc"]).cpu().numpy()

        for i in range(images.size(0)):
            if count >= max_samples:
                break

            img = batch["image"][i, 0].numpy()  # (H, W)
            mp = midline_preds[i, 0]
            qp = qc_preds[i, 0]
            mt = batch["midline_mask"][i, 0].numpy()
            qt = batch["qc_heatmap"][i, 0].numpy()
            fname = batch["filename"][i]

            # Compute per-sample metrics for the title
            dice = dice_coefficient(mp, mt)
            dist = qc_distance_error(qp, qt)

            utils.plot_prediction_overlay(
                image=img,
                pred_midline=mp,
                pred_qc=qp,
                gt_midline=mt,
                gt_qc=qt,
                title=f"{fname} | Dice={dice:.3f} | QC dist={dist:.1f}px",
                save_path=str(save_dir / f"{fname}_prediction.png"),
            )
            count += 1

    print(f"Saved {count} prediction visualizations to {save_dir}")


# =============================================================================
# Main
# =============================================================================


def main(checkpoint_path: str) -> None:
    """
    Run full evaluation pipeline.

    Args:
        checkpoint_path: Path to model checkpoint.
    """
    utils.set_seed()
    device = utils.get_device()
    print(f"Using device: {device}")

    # Load data
    print("\nLoading data...")
    _, val_loader = create_dataloaders()

    # Load model
    print("\nLoading model...")
    model = build_model(device)
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    epoch = checkpoint.get("epoch", "?")
    val_loss = checkpoint.get("val_loss", "?")
    print(f"Loaded checkpoint from epoch {epoch} (val_loss={val_loss})")

    # Evaluate
    print("\nRunning evaluation on validation set...")
    results = evaluate_model(model, val_loader, device)
    print_metrics(results)

    # Visualize
    print("\nGenerating prediction visualizations...")
    visualize_predictions(model, val_loader, device)

    print("\nDone!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate root midline/QC model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(config.CHECKPOINT_DIR / "best_model.pth"),
        help="Path to model checkpoint (default: output/checkpoints/best_model.pth)",
    )
    parser.add_argument(
        "--max-vis",
        type=int,
        default=10,
        help="Maximum number of samples to visualize (default: 10)",
    )
    args = parser.parse_args()

    main(args.checkpoint)
