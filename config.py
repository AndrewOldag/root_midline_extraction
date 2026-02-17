"""
Central configuration for root midline extraction training pipeline.
All hyperparameters, paths, and training settings in one place.
"""

import os
from pathlib import Path

# =============================================================================
# Paths
# =============================================================================

# Project root (directory containing this file)
PROJECT_ROOT = Path(os.path.dirname(os.path.abspath(__file__)))

# Dataset paths
DATASET_ROOT = PROJECT_ROOT / "dataset" / "dataset"
IMAGES_DIR = DATASET_ROOT / "images"          # Raw microscope images (added by user)
MIDLINE_MASKS_DIR = DATASET_ROOT / "midline_masks"
QC_HEATMAPS_DIR = DATASET_ROOT / "qc_heatmaps"
PREVIEW_DIR = DATASET_ROOT / "preview"
META_MAT_PATH = DATASET_ROOT / "meta.mat"

# Supported raw image extensions (checked in order when resolving image paths)
IMAGE_EXTENSIONS = [".tiff", ".tif", ".png", ".jpg", ".jpeg", ".bmp"]

# Output paths
OUTPUT_DIR = PROJECT_ROOT / "output"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
LOG_DIR = OUTPUT_DIR / "logs"
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"

# =============================================================================
# Pre-processing: Classical Root Cropping
# =============================================================================

# Gaussian blur kernel size for root detection (must be odd)
CROP_BLUR_KERNEL = 51

# Padding around detected root bounding box (fraction of bbox size)
CROP_PADDING_FRACTION = 0.15

# Minimum padding in pixels on each side
CROP_PADDING_MIN_PX = 100

# Morphological kernel sizes for closing and opening
CROP_MORPH_CLOSE_KERNEL = 51
CROP_MORPH_OPEN_KERNEL = 21

# =============================================================================
# Model
# =============================================================================

# Network input size (after cropping and resizing)
IMAGE_SIZE = (512, 512)  # (H, W)

# Encoder backbone (from segmentation-models-pytorch)
ENCODER_NAME = "resnet34"
ENCODER_WEIGHTS = "imagenet"

# Number of input channels (grayscale = 1)
IN_CHANNELS = 1

# =============================================================================
# Training
# =============================================================================

# Basic training params
BATCH_SIZE = 4
NUM_WORKERS = 0   # 0 is safest on Windows; set to 2-4 on Linux with GPU
EPOCHS = 200
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

# Loss weights
# Combined loss = L_midline + LAMBDA_QC * L_qc
LAMBDA_QC = 1.0

# Scheduler
SCHEDULER = "cosine"  # "cosine" or "onecycle"

# Early stopping patience (epochs with no val improvement)
PATIENCE = 30

# =============================================================================
# Data Augmentation
# =============================================================================

# Rotation range (degrees)
AUG_ROTATION_LIMIT = 30

# Shift/scale/rotate limits
AUG_SHIFT_LIMIT = 0.1
AUG_SCALE_LIMIT = 0.15

# Elastic transform params
AUG_ELASTIC_ALPHA = 120
AUG_ELASTIC_SIGMA = 6

# Brightness/contrast jitter
AUG_BRIGHTNESS_LIMIT = 0.2
AUG_CONTRAST_LIMIT = 0.2

# =============================================================================
# Miscellaneous
# =============================================================================

# Random seed for reproducibility
SEED = 42

# Device (auto-detect)
DEVICE = "cuda"  # Will fall back to CPU in code if CUDA not available

# Use preview images as fallback when raw images are not available
# Extracts blue channel (least affected by green/red annotations)
USE_PREVIEW_FALLBACK = True

# Logging frequency (batches)
LOG_INTERVAL = 10

# =============================================================================
# Dry-Run / Smoke Test
# =============================================================================

# When True, overrides settings for a quick pipeline test:
#   - 4 train samples, 2 val samples
#   - 2 epochs, batch_size=2, image_size=128x128
#   - num_workers=0 (avoids multiprocessing issues)
#   - Forces CPU
# Run with:  python train.py --dry-run
DRY_RUN = False
DRY_RUN_TRAIN_SAMPLES = 4
DRY_RUN_VAL_SAMPLES = 2
DRY_RUN_IMAGE_SIZE = (128, 128)
DRY_RUN_EPOCHS = 2
DRY_RUN_BATCH_SIZE = 2
