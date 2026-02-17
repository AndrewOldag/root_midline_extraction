"""
Training loop for the dual-head U-Net root midline / QC detection model.

Usage:
    python train.py

Trains the model using the dataset in dataset/dataset/, saves checkpoints
and training logs to output/.
"""

import csv
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
import utils
from data_loader import create_dataloaders
from model import CombinedLoss, DualHeadUNet, build_model


def train_one_epoch(
    model: DualHeadUNet,
    loader: DataLoader,
    criterion: CombinedLoss,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> Dict[str, float]:
    """
    Train for one epoch.

    Returns:
        Dict with 'total_loss', 'midline_loss', 'qc_loss' (averaged over batches).
    """
    model.train()
    running_total = 0.0
    running_midline = 0.0
    running_qc = 0.0
    n_batches = 0

    pbar = tqdm(loader, desc="Train", leave=False)
    for batch in pbar:
        images = batch["image"].to(device)
        targets = {
            "midline_mask": batch["midline_mask"].to(device),
            "qc_heatmap": batch["qc_heatmap"].to(device),
        }

        # Forward pass
        predictions = model(images)
        total_loss, midline_loss, qc_loss = criterion(predictions, targets)

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Accumulate
        running_total += total_loss.item()
        running_midline += midline_loss.item()
        running_qc += qc_loss.item()
        n_batches += 1

        pbar.set_postfix({
            "loss": f"{total_loss.item():.4f}",
            "mid": f"{midline_loss.item():.4f}",
            "qc": f"{qc_loss.item():.4f}",
        })

    return {
        "total_loss": running_total / max(n_batches, 1),
        "midline_loss": running_midline / max(n_batches, 1),
        "qc_loss": running_qc / max(n_batches, 1),
    }


@torch.no_grad()
def validate(
    model: DualHeadUNet,
    loader: DataLoader,
    criterion: CombinedLoss,
    device: torch.device,
) -> Dict[str, float]:
    """
    Run validation.

    Returns:
        Dict with 'total_loss', 'midline_loss', 'qc_loss' (averaged over batches).
    """
    model.eval()
    running_total = 0.0
    running_midline = 0.0
    running_qc = 0.0
    n_batches = 0

    pbar = tqdm(loader, desc="Val  ", leave=False)
    for batch in pbar:
        images = batch["image"].to(device)
        targets = {
            "midline_mask": batch["midline_mask"].to(device),
            "qc_heatmap": batch["qc_heatmap"].to(device),
        }

        predictions = model(images)
        total_loss, midline_loss, qc_loss = criterion(predictions, targets)

        running_total += total_loss.item()
        running_midline += midline_loss.item()
        running_qc += qc_loss.item()
        n_batches += 1

    return {
        "total_loss": running_total / max(n_batches, 1),
        "midline_loss": running_midline / max(n_batches, 1),
        "qc_loss": running_qc / max(n_batches, 1),
    }


def save_checkpoint(
    model: DualHeadUNet,
    optimizer: optim.Optimizer,
    scheduler,
    epoch: int,
    val_loss: float,
    path: Path,
) -> None:
    """Save a training checkpoint."""
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "val_loss": val_loss,
        },
        str(path),
    )


def load_checkpoint(
    path: Path,
    model: DualHeadUNet,
    optimizer: optim.Optimizer = None,
    scheduler=None,
) -> int:
    """
    Load a checkpoint. Returns the epoch number.
    """
    checkpoint = torch.load(str(path), weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and checkpoint["scheduler_state_dict"] is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint["epoch"]


class CSVLogger:
    """Simple CSV logger for training metrics."""

    def __init__(self, path: Path, fieldnames: List[str]):
        self.path = path
        self.fieldnames = fieldnames
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

    def log(self, row: Dict) -> None:
        with open(self.path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row)


def train(resume_from: str = None) -> None:
    """
    Main training function.

    Args:
        resume_from: Path to checkpoint to resume from (optional).
    """
    # Setup
    utils.set_seed()
    utils.ensure_dirs()
    device = utils.get_device()
    print(f"Using device: {device}")

    # Data
    print("\nLoading data...")
    train_loader, val_loader = create_dataloaders()
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches:   {len(val_loader)}")

    # Model
    print("\nBuilding model...")
    model = build_model(device)

    # Loss
    criterion = CombinedLoss(lambda_qc=config.LAMBDA_QC)

    # Optimizer (differential learning rate: encoder slower, decoders faster)
    optimizer = optim.AdamW(
        [
            {"params": model.get_encoder_params(), "lr": config.LEARNING_RATE * 0.1},
            {"params": model.get_decoder_params(), "lr": config.LEARNING_RATE},
        ],
        weight_decay=config.WEIGHT_DECAY,
    )

    # Scheduler
    if config.SCHEDULER == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.EPOCHS, eta_min=1e-6
        )
    elif config.SCHEDULER == "onecycle":
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[config.LEARNING_RATE * 0.1, config.LEARNING_RATE],
            epochs=config.EPOCHS,
            steps_per_epoch=len(train_loader),
        )
    else:
        scheduler = None

    # Resume if requested
    start_epoch = 0
    if resume_from is not None:
        start_epoch = load_checkpoint(
            Path(resume_from), model, optimizer, scheduler
        )
        print(f"Resumed from epoch {start_epoch}")

    # Logger
    logger = CSVLogger(
        config.LOG_DIR / "training_log.csv",
        fieldnames=[
            "epoch", "train_total", "train_midline", "train_qc",
            "val_total", "val_midline", "val_qc", "lr", "time_s",
        ],
    )

    # Training loop
    best_val_loss = float("inf")
    patience_counter = 0
    history = {
        "train_total": [], "val_total": [],
        "train_midline": [], "train_qc": [],
    }

    print(f"\nStarting training for {config.EPOCHS} epochs...")
    print("=" * 70)

    for epoch in range(start_epoch, config.EPOCHS):
        epoch_start = time.time()

        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validate
        val_metrics = validate(model, val_loader, criterion, device)

        # Step scheduler
        if scheduler is not None and config.SCHEDULER == "cosine":
            scheduler.step()

        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[1]["lr"]  # Decoder LR

        # Log
        history["train_total"].append(train_metrics["total_loss"])
        history["val_total"].append(val_metrics["total_loss"])
        history["train_midline"].append(train_metrics["midline_loss"])
        history["train_qc"].append(train_metrics["qc_loss"])

        logger.log({
            "epoch": epoch + 1,
            "train_total": f"{train_metrics['total_loss']:.6f}",
            "train_midline": f"{train_metrics['midline_loss']:.6f}",
            "train_qc": f"{train_metrics['qc_loss']:.6f}",
            "val_total": f"{val_metrics['total_loss']:.6f}",
            "val_midline": f"{val_metrics['midline_loss']:.6f}",
            "val_qc": f"{val_metrics['qc_loss']:.6f}",
            "lr": f"{current_lr:.8f}",
            "time_s": f"{epoch_time:.1f}",
        })

        # Print progress
        print(
            f"Epoch {epoch+1:3d}/{config.EPOCHS} | "
            f"Train: {train_metrics['total_loss']:.4f} "
            f"(mid={train_metrics['midline_loss']:.4f}, qc={train_metrics['qc_loss']:.4f}) | "
            f"Val: {val_metrics['total_loss']:.4f} "
            f"(mid={val_metrics['midline_loss']:.4f}, qc={val_metrics['qc_loss']:.4f}) | "
            f"LR: {current_lr:.6f} | "
            f"Time: {epoch_time:.1f}s"
        )

        # Save best model
        if val_metrics["total_loss"] < best_val_loss:
            best_val_loss = val_metrics["total_loss"]
            patience_counter = 0
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                val_metrics["total_loss"],
                config.CHECKPOINT_DIR / "best_model.pth",
            )
            print(f"  -> New best model saved (val_loss={best_val_loss:.6f})")
        else:
            patience_counter += 1

        # Save periodic checkpoint every 25 epochs
        if (epoch + 1) % 25 == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                val_metrics["total_loss"],
                config.CHECKPOINT_DIR / f"checkpoint_epoch_{epoch+1}.pth",
            )

        # Early stopping
        if patience_counter >= config.PATIENCE:
            print(f"\nEarly stopping at epoch {epoch+1} (no improvement for {config.PATIENCE} epochs)")
            break

    # Save final model
    save_checkpoint(
        model, optimizer, scheduler, epoch,
        val_metrics["total_loss"],
        config.CHECKPOINT_DIR / "final_model.pth",
    )

    # Plot training curves
    utils.plot_training_curves(
        history["train_total"],
        history["val_total"],
        history["train_midline"],
        history["train_qc"],
        save_path=str(config.LOG_DIR / "training_curves.png"),
    )

    print("=" * 70)
    print(f"Training complete. Best val loss: {best_val_loss:.6f}")
    print(f"Checkpoints saved to: {config.CHECKPOINT_DIR}")
    print(f"Training log saved to: {config.LOG_DIR / 'training_log.csv'}")
    print(f"Training curves saved to: {config.LOG_DIR / 'training_curves.png'}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train root midline/QC model")
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override number of epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Override batch size"
    )
    parser.add_argument(
        "--lr", type=float, default=None,
        help="Override learning rate"
    )
    parser.add_argument(
        "--image-size", type=int, default=None,
        help="Override image size (square, e.g. 256 for 256x256). "
             "Smaller = faster training. Default: 512"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Quick smoke test: 4 train + 2 val samples, 2 epochs, "
             "128x128 images, CPU only. Tests full pipeline without GPU or "
             "real images (uses preview fallback)."
    )
    args = parser.parse_args()

    # Apply dry-run overrides
    if args.dry_run:
        config.DRY_RUN = True
        config.EPOCHS = config.DRY_RUN_EPOCHS
        config.BATCH_SIZE = config.DRY_RUN_BATCH_SIZE
        config.IMAGE_SIZE = config.DRY_RUN_IMAGE_SIZE
        config.NUM_WORKERS = 0
        config.DEVICE = "cpu"
        config.USE_PREVIEW_FALLBACK = True
        # Disable ImageNet pretrained weights to skip download if offline
        config.ENCODER_WEIGHTS = None
        print("=" * 70)
        print("DRY RUN MODE")
        print("  Testing full pipeline with minimal data and settings.")
        print(f"  Samples: {config.DRY_RUN_TRAIN_SAMPLES} train, {config.DRY_RUN_VAL_SAMPLES} val")
        print(f"  Epochs: {config.EPOCHS}, Batch size: {config.BATCH_SIZE}")
        print(f"  Image size: {config.IMAGE_SIZE}, Device: CPU")
        print("=" * 70)

    # Apply manual overrides (these take priority over dry-run defaults)
    if args.epochs is not None:
        config.EPOCHS = args.epochs
    if args.batch_size is not None:
        config.BATCH_SIZE = args.batch_size
    if args.lr is not None:
        config.LEARNING_RATE = args.lr
    if args.image_size is not None:
        config.IMAGE_SIZE = (args.image_size, args.image_size)

    train(resume_from=args.resume)
