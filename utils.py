"""
Utility functions for root midline extraction pipeline.
Includes meta.mat parsing, filename mapping, seed setting, and plotting helpers.
"""

import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch

import config


def set_seed(seed: int = config.SEED) -> None:
    """Set random seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Get the best available device (CUDA > CPU)."""
    if config.DEVICE == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_meta(meta_path: Optional[Path] = None) -> Dict:
    """
    Parse meta.mat (MATLAB v7.3 / HDF5) to extract train/val indices
    and dataset parameters.

    Returns:
        dict with keys:
            - 'train_indices': list of 0-based integer indices
            - 'val_indices': list of 0-based integer indices
            - 'line_thickness_px': float
            - 'qc_sigma_px': float
    """
    if meta_path is None:
        meta_path = config.META_MAT_PATH

    with h5py.File(str(meta_path), "r") as f:
        # Train/val indices (MATLAB is 1-based, convert to 0-based)
        train_idx = f["trainIdx"][:, 0].astype(int) - 1
        val_idx = f["valIdx"][:, 0].astype(int) - 1

        # Dataset parameters
        line_thickness = float(f["lineThicknessPx"][0, 0])
        qc_sigma = float(f["qcSigmaPx"][0, 0])

    return {
        "train_indices": train_idx.tolist(),
        "val_indices": val_idx.tolist(),
        "line_thickness_px": line_thickness,
        "qc_sigma_px": qc_sigma,
    }


def get_filenames(masks_dir: Optional[Path] = None) -> List[str]:
    """
    Get sorted list of base filenames from the midline_masks directory.
    These filenames are used to match across images, masks, and QC heatmaps.

    Returns:
        Sorted list of filenames (e.g., 'Basler_...0000.png')
    """
    if masks_dir is None:
        masks_dir = config.MIDLINE_MASKS_DIR

    files = sorted([
        f for f in os.listdir(masks_dir)
        if f.lower().endswith(".png")
    ])
    return files


def build_file_lists(
    filenames: List[str],
    indices: List[int],
) -> Tuple[List[Path], List[Path], List[Path]]:
    """
    Build lists of file paths for images, masks, and QC heatmaps
    for a given set of indices.

    Args:
        filenames: Full sorted list of base filenames
        indices: 0-based indices to select

    Returns:
        (image_paths, mask_paths, qc_paths) - lists of Path objects
    """
    image_paths = []
    mask_paths = []
    qc_paths = []

    for idx in indices:
        fname = filenames[idx]
        stem = Path(fname).stem

        # Raw image path -- try multiple extensions since masks are .png
        # but raw images may be .tiff, .tif, .png, etc.
        img_path = None
        for ext in config.IMAGE_EXTENSIONS:
            candidate = config.IMAGES_DIR / f"{stem}{ext}"
            if candidate.exists():
                img_path = candidate
                break

        if img_path is None:
            # Fall back to same name as mask (e.g., .png)
            img_path = config.IMAGES_DIR / fname

        if not img_path.exists() and config.USE_PREVIEW_FALLBACK:
            # Try preview fallback: add _preview suffix
            preview_name = f"{stem}_preview.png"
            preview_path = config.PREVIEW_DIR / preview_name
            if preview_path.exists():
                img_path = preview_path

        image_paths.append(img_path)
        mask_paths.append(config.MIDLINE_MASKS_DIR / fname)
        qc_paths.append(config.QC_HEATMAPS_DIR / fname)

    return image_paths, mask_paths, qc_paths


def ensure_dirs() -> None:
    """Create output directories if they don't exist."""
    for d in [config.OUTPUT_DIR, config.CHECKPOINT_DIR,
              config.LOG_DIR, config.PREDICTIONS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Plotting helpers
# =============================================================================


def plot_sample(
    image: np.ndarray,
    midline_mask: np.ndarray,
    qc_heatmap: np.ndarray,
    title: str = "",
    save_path: Optional[str] = None,
) -> None:
    """
    Plot an image with its midline mask and QC heatmap side by side.

    Args:
        image: Grayscale image (H, W) or (H, W, 1)
        midline_mask: Binary mask (H, W), values in [0, 1]
        qc_heatmap: Heatmap (H, W), values in [0, 1]
        title: Optional title
        save_path: If provided, save to this path instead of showing
    """
    if image.ndim == 3 and image.shape[-1] == 1:
        image = image[:, :, 0]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    axes[1].imshow(midline_mask, cmap="gray")
    axes[1].set_title("Midline Mask")
    axes[1].axis("off")

    axes[2].imshow(qc_heatmap, cmap="hot")
    axes[2].set_title("QC Heatmap")
    axes[2].axis("off")

    if title:
        fig.suptitle(title)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def _overlay_qc_heatmap(ax, heatmap: np.ndarray, threshold: float = 0.1) -> None:
    """
    Overlay a QC heatmap on an axes, masking out low values so only
    the actual peak region is visible (avoids the red-haze problem).
    Also marks the peak location with a crosshair.
    """
    masked = np.ma.masked_where(heatmap < threshold, heatmap)
    ax.imshow(masked, cmap="hot", alpha=0.65, vmin=0, vmax=1)

    peak_yx = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    ax.plot(
        peak_yx[1], peak_yx[0], "c+",
        markersize=14, markeredgewidth=2.5,
    )


def plot_prediction_overlay(
    image: np.ndarray,
    pred_midline: np.ndarray,
    pred_qc: np.ndarray,
    gt_midline: Optional[np.ndarray] = None,
    gt_qc: Optional[np.ndarray] = None,
    title: str = "",
    save_path: Optional[str] = None,
) -> None:
    """
    Plot prediction overlaid on the original image.

    Args:
        image: Grayscale image (H, W)
        pred_midline: Predicted midline mask (H, W), values in [0, 1]
        pred_qc: Predicted QC heatmap (H, W), values in [0, 1]
        gt_midline: Optional ground truth midline mask
        gt_qc: Optional ground truth QC heatmap
        title: Optional title
        save_path: If provided, save to this path
    """
    n_cols = 2 if gt_midline is None else 3
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 6))

    # Predicted midline overlay
    axes[0].imshow(image, cmap="gray")
    midline_masked = np.ma.masked_where(pred_midline < 0.3, pred_midline)
    axes[0].imshow(midline_masked, cmap="Greens", alpha=0.6, vmin=0, vmax=1)
    axes[0].set_title("Predicted Midline")
    axes[0].axis("off")

    # Predicted QC overlay (masked to show only the peak region)
    axes[1].imshow(image, cmap="gray")
    _overlay_qc_heatmap(axes[1], pred_qc)
    axes[1].set_title("Predicted QC")
    axes[1].axis("off")

    if gt_midline is not None and n_cols > 2:
        axes[2].imshow(image, cmap="gray")
        gt_mid_masked = np.ma.masked_where(gt_midline < 0.3, gt_midline)
        axes[2].imshow(gt_mid_masked, cmap="Greens", alpha=0.5, vmin=0, vmax=1)
        if gt_qc is not None:
            _overlay_qc_heatmap(axes[2], gt_qc)
        axes[2].set_title("Ground Truth")
        axes[2].axis("off")

    if title:
        fig.suptitle(title, fontsize=11)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_midline_losses: Optional[List[float]] = None,
    train_qc_losses: Optional[List[float]] = None,
    save_path: Optional[str] = None,
) -> None:
    """Plot training and validation loss curves."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, label="Train Total", linewidth=2)
    ax.plot(epochs, val_losses, label="Val Total", linewidth=2)

    if train_midline_losses:
        ax.plot(epochs, train_midline_losses, "--", label="Train Midline", alpha=0.7)
    if train_qc_losses:
        ax.plot(epochs, train_qc_losses, "--", label="Train QC", alpha=0.7)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
