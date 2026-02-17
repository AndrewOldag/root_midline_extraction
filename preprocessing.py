"""
Classical image processing for root region detection and cropping.

Pipeline: Gaussian blur -> invert -> Otsu threshold -> morphological ops
          -> largest connected component -> bounding box + padding -> crop

This isolates the root from the background before feeding to the neural network,
reducing the active image area by 3-10x.
"""

import warnings
from typing import Optional, Tuple

import cv2
import numpy as np

import config


def find_root_bbox(
    image: np.ndarray,
    blur_kernel: int = config.CROP_BLUR_KERNEL,
    morph_close_kernel: int = config.CROP_MORPH_CLOSE_KERNEL,
    morph_open_kernel: int = config.CROP_MORPH_OPEN_KERNEL,
    padding_fraction: float = config.CROP_PADDING_FRACTION,
    padding_min_px: int = config.CROP_PADDING_MIN_PX,
) -> Tuple[int, int, int, int]:
    """
    Detect the root region in a microscope image using classical methods.

    The root is darker than the background, so we invert and threshold.

    Args:
        image: Grayscale image as numpy array (H, W), uint8 or uint16.
        blur_kernel: Gaussian blur kernel size (must be odd).
        morph_close_kernel: Morphological closing kernel size.
        morph_open_kernel: Morphological opening kernel size.
        padding_fraction: Fraction of bbox size to add as padding on each side.
        padding_min_px: Minimum padding in pixels on each side.

    Returns:
        (x, y, w, h) bounding box of the root region with padding.
        Coordinates are clamped to image bounds.
    """
    h_img, w_img = image.shape[:2]

    # Ensure grayscale uint8
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    if gray.dtype != np.uint8:
        # Normalize to uint8 (e.g., from uint16)
        gray = (gray.astype(np.float32) / gray.max() * 255).astype(np.uint8)

    # Step 1: Gaussian blur to smooth out noise
    blurred = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)

    # Step 2: Invert (root is dark -> becomes bright) and Otsu threshold
    inverted = 255 - blurred
    _, binary = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Step 3: Morphological closing to fill gaps in root
    close_kern = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (morph_close_kernel, morph_close_kernel)
    )
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kern)

    # Step 4: Morphological opening to remove small noise
    open_kern = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (morph_open_kernel, morph_open_kernel)
    )
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, open_kern)

    # Step 5: Find the largest connected component (the root)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        opened, connectivity=8
    )

    if num_labels <= 1:
        # No components found -- fall back to full image
        warnings.warn(
            "Root detection failed: no connected components found. "
            "Returning full image bounding box."
        )
        return (0, 0, w_img, h_img)

    # Ignore background (label 0), find largest by area
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = np.argmax(areas) + 1  # +1 because we skipped background
    x = stats[largest_label, cv2.CC_STAT_LEFT]
    y = stats[largest_label, cv2.CC_STAT_TOP]
    w = stats[largest_label, cv2.CC_STAT_WIDTH]
    h = stats[largest_label, cv2.CC_STAT_HEIGHT]

    # Step 6: Add padding
    pad_x = max(int(w * padding_fraction), padding_min_px)
    pad_y = max(int(h * padding_fraction), padding_min_px)

    x = max(0, x - pad_x)
    y = max(0, y - pad_y)
    w = min(w_img - x, w + 2 * pad_x)
    h = min(h_img - y, h + 2 * pad_y)

    return (x, y, w, h)


def bbox_from_mask(
    mask: np.ndarray,
    padding_fraction: float = config.CROP_PADDING_FRACTION,
    padding_min_px: int = config.CROP_PADDING_MIN_PX,
) -> Tuple[int, int, int, int]:
    """
    Compute a bounding box from a ground truth mask.

    Use this during training for guaranteed-correct crops.
    Falls back to full image if mask is empty.

    Args:
        mask: Binary mask (H, W), any dtype (nonzero = foreground).
        padding_fraction: Fraction of bbox size to add as padding.
        padding_min_px: Minimum padding in pixels.

    Returns:
        (x, y, w, h) bounding box with padding, clamped to image bounds.
    """
    h_img, w_img = mask.shape[:2]

    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        warnings.warn("Empty mask provided to bbox_from_mask. Returning full image.")
        return (0, 0, w_img, h_img)

    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())
    w = x_max - x_min + 1
    h = y_max - y_min + 1

    # Add padding
    pad_x = max(int(w * padding_fraction), padding_min_px)
    pad_y = max(int(h * padding_fraction), padding_min_px)

    x = max(0, x_min - pad_x)
    y = max(0, y_min - pad_y)
    w = min(w_img - x, (x_max + 1 + pad_x) - x)
    h = min(h_img - y, (y_max + 1 + pad_y) - y)

    return (x, y, w, h)


def crop_to_root(
    image: np.ndarray,
    midline_mask: Optional[np.ndarray] = None,
    qc_heatmap: Optional[np.ndarray] = None,
    bbox: Optional[Tuple[int, int, int, int]] = None,
    use_mask_bbox: bool = False,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Tuple[int, int, int, int]]:
    """
    Crop an image (and optionally its mask and heatmap) to the root region.

    During training (use_mask_bbox=True), the bbox is derived from the midline
    mask for guaranteed correctness. During inference, classical detection is used.

    Args:
        image: Input image (H, W) or (H, W, C).
        midline_mask: Optional midline mask (H, W).
        qc_heatmap: Optional QC heatmap (H, W).
        bbox: Optional pre-computed bounding box (x, y, w, h).
        use_mask_bbox: If True and midline_mask is provided, derive bbox from mask.

    Returns:
        (cropped_image, cropped_mask, cropped_qc, bbox)
        mask and qc are None if not provided.
    """
    if bbox is None:
        if use_mask_bbox and midline_mask is not None:
            bbox = bbox_from_mask(midline_mask)
        else:
            bbox = find_root_bbox(image)

    x, y, w, h = bbox

    cropped_image = image[y : y + h, x : x + w]
    cropped_mask = midline_mask[y : y + h, x : x + w] if midline_mask is not None else None
    cropped_qc = qc_heatmap[y : y + h, x : x + w] if qc_heatmap is not None else None

    return cropped_image, cropped_mask, cropped_qc, bbox


def resize_crop(
    crop: np.ndarray,
    target_size: Tuple[int, int] = config.IMAGE_SIZE,
    interpolation: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    """
    Resize a cropped image to the target network input size.

    Args:
        crop: Cropped image or mask (H, W) or (H, W, C).
        target_size: (H, W) target size.
        interpolation: OpenCV interpolation method.
            Use cv2.INTER_LINEAR for images, cv2.INTER_NEAREST for masks.

    Returns:
        Resized array.
    """
    # cv2.resize expects (width, height)
    target_wh = (target_size[1], target_size[0])
    return cv2.resize(crop, target_wh, interpolation=interpolation)


def preprocess_sample(
    image: np.ndarray,
    midline_mask: Optional[np.ndarray] = None,
    qc_heatmap: Optional[np.ndarray] = None,
    target_size: Tuple[int, int] = config.IMAGE_SIZE,
    use_mask_bbox: bool = False,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Tuple[int, int, int, int]]:
    """
    Full pre-processing pipeline: detect root -> crop -> resize.

    Args:
        image: Raw microscope image (H, W) or (H, W, C).
        midline_mask: Ground truth midline mask (H, W), optional.
        qc_heatmap: Ground truth QC heatmap (H, W), optional.
        target_size: (H, W) network input size.
        use_mask_bbox: If True, derive crop bbox from the midline mask
            (use during training for guaranteed correctness).

    Returns:
        (resized_image, resized_mask, resized_qc, bbox)
        bbox is in original image coordinates for inverse mapping.
    """
    # Crop to root region
    cropped_image, cropped_mask, cropped_qc, bbox = crop_to_root(
        image, midline_mask, qc_heatmap, use_mask_bbox=use_mask_bbox
    )

    # Resize image (bilinear)
    resized_image = resize_crop(cropped_image, target_size, cv2.INTER_LINEAR)

    # Resize mask (nearest neighbor to preserve binary values)
    resized_mask = None
    if cropped_mask is not None:
        resized_mask = resize_crop(cropped_mask, target_size, cv2.INTER_NEAREST)

    # Resize QC heatmap (bilinear to preserve smooth Gaussian)
    resized_qc = None
    if cropped_qc is not None:
        resized_qc = resize_crop(cropped_qc, target_size, cv2.INTER_LINEAR)

    return resized_image, resized_mask, resized_qc, bbox


def map_predictions_to_original(
    pred: np.ndarray,
    bbox: Tuple[int, int, int, int],
    original_size: Tuple[int, int],
    interpolation: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    """
    Map a prediction from network output space back to original image coordinates.

    Args:
        pred: Prediction array at network output size (H, W).
        bbox: (x, y, w, h) bounding box used during cropping.
        original_size: (H, W) of the original full-resolution image.
        interpolation: Interpolation method for resizing.

    Returns:
        Full-resolution prediction array with the prediction placed
        at the correct location.
    """
    x, y, w, h = bbox
    orig_h, orig_w = original_size

    # Resize prediction back to crop size
    pred_crop = cv2.resize(pred, (w, h), interpolation=interpolation)

    # Place in full-resolution canvas
    full_pred = np.zeros((orig_h, orig_w), dtype=pred_crop.dtype)
    full_pred[y : y + h, x : x + w] = pred_crop

    return full_pred


def extract_qc_point(
    qc_heatmap: np.ndarray,
    bbox: Optional[Tuple[int, int, int, int]] = None,
    crop_size: Optional[Tuple[int, int]] = None,
) -> Tuple[int, int]:
    """
    Extract the QC point location as the peak of the heatmap.

    Args:
        qc_heatmap: QC heatmap (H, W), either at network output size or original.
        bbox: If provided, map the point back to original image coordinates.
        crop_size: (H, W) of the crop before resizing (needed for coordinate mapping).

    Returns:
        (x, y) coordinates of the QC point in original image space.
    """
    # Find peak location
    peak_yx = np.unravel_index(np.argmax(qc_heatmap), qc_heatmap.shape)
    peak_y, peak_x = peak_yx

    if bbox is not None and crop_size is not None:
        bx, by, bw, bh = bbox
        net_h, net_w = qc_heatmap.shape[:2]
        crop_h, crop_w = crop_size

        # Scale from network coords to crop coords
        peak_x = int(peak_x * crop_w / net_w)
        peak_y = int(peak_y * crop_h / net_h)

        # Offset to original image coords
        peak_x += bx
        peak_y += by

    return (peak_x, peak_y)
