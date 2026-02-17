"""
Dataset and DataLoader for root midline extraction.

Loads image / midline mask / QC heatmap triples, applies classical root
cropping, data augmentation, and normalization.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import config
import preprocessing
import utils


class RootDataset(Dataset):
    """
    PyTorch Dataset for root midline / QC detection.

    Each sample consists of:
        - Input image (grayscale)
        - Midline mask (binary, 0 or 1)
        - QC heatmap (continuous, 0 to 1)

    The classical root-region crop is applied before augmentation.
    """

    def __init__(
        self,
        image_paths: List[Path],
        mask_paths: List[Path],
        qc_paths: List[Path],
        transform: Optional[A.Compose] = None,
        target_size: Tuple[int, int] = config.IMAGE_SIZE,
        is_preview_fallback: bool = False,
        use_mask_bbox: bool = True,
    ):
        """
        Args:
            image_paths: Paths to input images (raw or preview fallback).
            mask_paths: Paths to midline mask PNGs.
            qc_paths: Paths to QC heatmap PNGs.
            transform: Albumentations augmentation pipeline.
            target_size: (H, W) network input size.
            is_preview_fallback: If True, extract blue channel from preview images.
            use_mask_bbox: If True, derive crop bbox from the midline mask
                (guarantees root is in crop). Use True for training, False for inference.
        """
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.qc_paths = qc_paths
        self.transform = transform
        self.target_size = target_size
        self.is_preview_fallback = is_preview_fallback
        self.use_mask_bbox = use_mask_bbox

    def __len__(self) -> int:
        return len(self.image_paths)

    def _load_image(self, path: Path) -> np.ndarray:
        """Load input image as grayscale uint8 (H, W)."""
        if not path.exists():
            raise FileNotFoundError(
                f"Image not found: {path}\n"
                f"Please add raw images to {config.IMAGES_DIR}/ "
                f"or set USE_PREVIEW_FALLBACK=True in config.py"
            )

        if self.is_preview_fallback or "_preview" in path.name:
            # Preview images are RGB with green/red annotations.
            # Blue channel is least affected.
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is None:
                raise IOError(f"Failed to read image: {path}")
            gray = img[:, :, 0]  # Blue channel (BGR format in OpenCV)
        else:
            # Raw grayscale image -- try OpenCV first, fall back to Pillow
            # (OpenCV can fail on some TIFF formats)
            gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                # Pillow fallback for TIFF and other formats
                from PIL import Image as PILImage
                pil_img = PILImage.open(str(path))
                gray = np.array(pil_img.convert("L"))

        # Ensure uint8 (TIFF images may be uint16)
        if gray.dtype != np.uint8:
            # Normalize to uint8 range
            gray = (gray.astype(np.float32) / gray.max() * 255).astype(np.uint8)

        return gray

    def _load_mask(self, path: Path) -> np.ndarray:
        """Load midline mask as binary float32 (H, W), values in {0, 1}."""
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise IOError(f"Failed to read mask: {path}")
        # Normalize: 0/255 -> 0.0/1.0
        return (mask / 255.0).astype(np.float32)

    def _load_qc_heatmap(self, path: Path) -> np.ndarray:
        """Load QC heatmap as float32 (H, W), values in [0, 1]."""
        # QC heatmaps are 16-bit PNGs
        qc = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if qc is None:
            raise IOError(f"Failed to read QC heatmap: {path}")
        # Normalize from uint16 [0, 65535] to float32 [0, 1]
        return (qc.astype(np.float32) / 65535.0)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Load raw data
        image = self._load_image(self.image_paths[idx])
        mask = self._load_mask(self.mask_paths[idx])
        qc = self._load_qc_heatmap(self.qc_paths[idx])

        # Root-region crop + resize
        # During training, use mask-based bbox for guaranteed correctness.
        # During inference, use classical detection.
        image, mask, qc, bbox = preprocessing.preprocess_sample(
            image, mask, qc, target_size=self.target_size,
            use_mask_bbox=self.use_mask_bbox,
        )

        # Ensure mask stays binary after resizing
        mask = (mask > 0.5).astype(np.float32)

        # Re-normalize QC after resize (peak might have changed)
        qc_max = qc.max()
        if qc_max > 0:
            qc = qc / qc_max

        # Apply augmentations (synchronized across image, mask, qc)
        if self.transform is not None:
            transformed = self.transform(
                image=image,
                masks=[mask, qc],
            )
            image = transformed["image"]
            mask = transformed["masks"][0]
            qc = transformed["masks"][1]

        # Convert to float32 tensors
        # Image: normalize to [0, 1], add channel dim -> (1, H, W)
        image = image.astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image).unsqueeze(0)

        # Mask: already float32 [0,1] -> (1, H, W)
        mask_tensor = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)

        # QC: already float32 [0,1] -> (1, H, W)
        qc_tensor = torch.from_numpy(qc.astype(np.float32)).unsqueeze(0)

        return {
            "image": image_tensor,
            "midline_mask": mask_tensor,
            "qc_heatmap": qc_tensor,
            "bbox": torch.tensor(bbox, dtype=torch.int32),
            "filename": self.image_paths[idx].stem,
        }


# =============================================================================
# Augmentation pipelines
# =============================================================================


def get_train_augmentation() -> A.Compose:
    """
    Augmentation pipeline for training.
    All spatial transforms are applied identically to image and masks.
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Affine(
            translate_percent={"x": (-config.AUG_SHIFT_LIMIT, config.AUG_SHIFT_LIMIT),
                               "y": (-config.AUG_SHIFT_LIMIT, config.AUG_SHIFT_LIMIT)},
            scale=(1 - config.AUG_SCALE_LIMIT, 1 + config.AUG_SCALE_LIMIT),
            rotate=(-config.AUG_ROTATION_LIMIT, config.AUG_ROTATION_LIMIT),
            border_mode=cv2.BORDER_CONSTANT,
            fill=0,
            fill_mask=0,
            p=0.7,
        ),
        A.ElasticTransform(
            alpha=config.AUG_ELASTIC_ALPHA,
            sigma=config.AUG_ELASTIC_SIGMA,
            border_mode=cv2.BORDER_CONSTANT,
            fill=0,
            fill_mask=0,
            p=0.3,
        ),
        A.RandomBrightnessContrast(
            brightness_limit=config.AUG_BRIGHTNESS_LIMIT,
            contrast_limit=config.AUG_CONTRAST_LIMIT,
            p=0.5,
        ),
        A.GaussNoise(p=0.2),
    ])


def get_val_augmentation() -> A.Compose:
    """Validation pipeline: no augmentation, just ensure consistent format."""
    return A.Compose([])


# =============================================================================
# DataLoader factory
# =============================================================================


def create_dataloaders(
    batch_size: int = config.BATCH_SIZE,
    num_workers: int = config.NUM_WORKERS,
    target_size: Tuple[int, int] = config.IMAGE_SIZE,
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation DataLoaders.

    Reads meta.mat for train/val split, builds file lists,
    and wraps in DataLoaders with appropriate augmentation.

    Args:
        batch_size: Batch size for both loaders.
        num_workers: Number of data loading workers.
        target_size: (H, W) network input size.
        max_train_samples: If set, limit training set to this many samples
            (useful for dry-run / smoke testing).
        max_val_samples: If set, limit validation set to this many samples.

    Returns:
        (train_loader, val_loader)
    """
    # Apply dry-run overrides if active
    if config.DRY_RUN:
        batch_size = config.DRY_RUN_BATCH_SIZE
        num_workers = 0
        target_size = config.DRY_RUN_IMAGE_SIZE
        max_train_samples = max_train_samples or config.DRY_RUN_TRAIN_SAMPLES
        max_val_samples = max_val_samples or config.DRY_RUN_VAL_SAMPLES

    # Load metadata
    meta = utils.load_meta()
    filenames = utils.get_filenames()

    train_indices = meta["train_indices"]
    val_indices = meta["val_indices"]

    # Limit samples if requested
    if max_train_samples is not None:
        train_indices = train_indices[:max_train_samples]
    if max_val_samples is not None:
        val_indices = val_indices[:max_val_samples]

    print(f"Dataset: {len(filenames)} total samples")
    print(f"  Train: {len(train_indices)} samples"
          + (f" (limited from {len(meta['train_indices'])})" if max_train_samples else ""))
    print(f"  Val:   {len(val_indices)} samples"
          + (f" (limited from {len(meta['val_indices'])})" if max_val_samples else ""))

    # Check if raw images exist (real files, not broken symlinks)
    raw_images_usable = False
    if config.IMAGES_DIR.exists():
        # Check that at least one file is actually readable (not a broken symlink)
        for f in os.listdir(config.IMAGES_DIR):
            candidate = config.IMAGES_DIR / f
            if candidate.is_file():  # is_file() returns False for broken symlinks
                raw_images_usable = True
                break

    is_preview_fallback = not raw_images_usable and config.USE_PREVIEW_FALLBACK

    if not raw_images_usable:
        if config.USE_PREVIEW_FALLBACK:
            n_previews = len(os.listdir(config.PREVIEW_DIR)) if config.PREVIEW_DIR.exists() else 0
            print(
                f"\nWARNING: Raw images not usable in {config.IMAGES_DIR}.\n"
                f"Using preview images as fallback (blue channel extraction).\n"
                f"Note: only {n_previews} preview images available "
                f"out of {len(filenames)} total samples.\n"
            )
        else:
            raise FileNotFoundError(
                f"Raw images not found in {config.IMAGES_DIR}.\n"
                f"Please add raw microscope images or set USE_PREVIEW_FALLBACK=True."
            )

    # Build file lists for train and val
    train_imgs, train_masks, train_qcs = utils.build_file_lists(
        filenames, train_indices
    )
    val_imgs, val_masks, val_qcs = utils.build_file_lists(
        filenames, val_indices
    )

    # Filter out samples where the image file doesn't exist or is unreadable
    # (handles broken symlinks, missing preview images, etc.)
    def filter_existing(imgs, masks, qcs):
        filtered_imgs, filtered_masks, filtered_qcs = [], [], []
        for im, ma, qc in zip(imgs, masks, qcs):
            if im.is_file():
                filtered_imgs.append(im)
                filtered_masks.append(ma)
                filtered_qcs.append(qc)
        return filtered_imgs, filtered_masks, filtered_qcs

    orig_train = len(train_imgs)
    orig_val = len(val_imgs)
    train_imgs, train_masks, train_qcs = filter_existing(train_imgs, train_masks, train_qcs)
    val_imgs, val_masks, val_qcs = filter_existing(val_imgs, val_masks, val_qcs)

    if len(train_imgs) < orig_train or len(val_imgs) < orig_val:
        print(f"  After filtering missing images: {len(train_imgs)} train, {len(val_imgs)} val")

    if len(train_imgs) == 0:
        raise RuntimeError(
            "No usable training images found. Make sure raw images are in "
            f"{config.IMAGES_DIR}/ or that preview images exist for fallback."
        )

    # Create datasets
    # use_mask_bbox=True ensures the crop is derived from ground truth masks
    # (guaranteed to contain the root), avoiding classical detection failures.
    train_dataset = RootDataset(
        image_paths=train_imgs,
        mask_paths=train_masks,
        qc_paths=train_qcs,
        transform=get_train_augmentation(),
        target_size=target_size,
        is_preview_fallback=is_preview_fallback,
        use_mask_bbox=True,
    )

    val_dataset = RootDataset(
        image_paths=val_imgs,
        mask_paths=val_masks,
        qc_paths=val_qcs,
        transform=get_val_augmentation(),
        target_size=target_size,
        is_preview_fallback=is_preview_fallback,
        use_mask_bbox=True,
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=len(train_dataset) > batch_size,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    return train_loader, val_loader
