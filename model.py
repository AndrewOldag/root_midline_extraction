"""
Dual-Head U-Net model for simultaneous midline segmentation and QC heatmap prediction.

Architecture:
    - Shared ResNet34 encoder (pretrained on ImageNet)
    - Two independent U-Net decoder heads:
        1. Midline decoder -> binary mask (sigmoid)
        2. QC decoder -> continuous heatmap (sigmoid)

Uses segmentation-models-pytorch (smp) for the backbone.
"""

from typing import Dict, List, Tuple

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn

import config


class UNetDecoder(nn.Module):
    """
    A U-Net decoder that takes encoder features and produces a single-channel output.

    Uses the decoder implementation from segmentation-models-pytorch internally,
    extracted from a full smp.Unet model.
    """

    def __init__(
        self,
        encoder_channels: Tuple[int, ...],
        decoder_channels: Tuple[int, ...] = (256, 128, 64, 32, 16),
    ):
        super().__init__()

        # Build a temporary smp Unet to extract decoder + segmentation head
        # We only need the decoder and head, not the encoder
        temp_unet = smp.Unet(
            encoder_name=config.ENCODER_NAME,
            encoder_weights=None,  # No weights needed, we just want the architecture
            in_channels=config.IN_CHANNELS,
            classes=1,
            decoder_channels=list(decoder_channels),
        )

        self.decoder = temp_unet.decoder
        self.segmentation_head = temp_unet.segmentation_head

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: List of encoder feature maps (from deepest to shallowest).

        Returns:
            Output tensor (B, 1, H, W).
        """
        decoder_output = self.decoder(*features)
        output = self.segmentation_head(decoder_output)
        return output


class DualHeadUNet(nn.Module):
    """
    Dual-head U-Net with a shared encoder and two independent decoders.

    - Midline head: outputs binary segmentation mask
    - QC head: outputs continuous heatmap

    Both outputs go through sigmoid activation.
    """

    def __init__(
        self,
        encoder_name: str = config.ENCODER_NAME,
        encoder_weights: str = config.ENCODER_WEIGHTS,
        in_channels: int = config.IN_CHANNELS,
        decoder_channels: Tuple[int, ...] = (256, 128, 64, 32, 16),
    ):
        super().__init__()

        # Build the shared encoder from an smp Unet
        base_model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=1,
            decoder_channels=list(decoder_channels),
        )

        # Extract the shared encoder
        self.encoder = base_model.encoder

        # Get encoder output channel sizes for decoder construction
        encoder_channels = self.encoder.out_channels

        # Midline decoder head
        self.midline_decoder = base_model.decoder
        self.midline_head = base_model.segmentation_head

        # QC decoder head (separate instance with fresh weights)
        qc_model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=None,  # Fresh decoder weights
            in_channels=in_channels,
            classes=1,
            decoder_channels=list(decoder_channels),
        )
        self.qc_decoder = qc_model.decoder
        self.qc_head = qc_model.segmentation_head

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input tensor (B, 1, H, W).

        Returns:
            Dict with:
                'midline': predicted midline mask (B, 1, H, W), after sigmoid
                'qc': predicted QC heatmap (B, 1, H, W), after sigmoid
        """
        # Shared encoder
        features = self.encoder(x)

        # Midline decoder
        midline_out = self.midline_decoder(features)
        midline_out = self.midline_head(midline_out)

        # QC decoder
        qc_out = self.qc_decoder(features)
        qc_out = self.qc_head(qc_out)

        return {
            "midline": midline_out,
            "qc": qc_out,
        }

    def get_encoder_params(self):
        """Get encoder parameters (for differential learning rates)."""
        return self.encoder.parameters()

    def get_decoder_params(self):
        """Get all decoder parameters (both heads)."""
        params = list(self.midline_decoder.parameters()) + \
                 list(self.midline_head.parameters()) + \
                 list(self.qc_decoder.parameters()) + \
                 list(self.qc_head.parameters())
        return params


# =============================================================================
# Loss functions
# =============================================================================


class DiceLoss(nn.Module):
    """Dice loss for binary segmentation."""

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(pred)
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)

        intersection = (pred_flat * target_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (
            pred_flat.sum() + target_flat.sum() + self.smooth
        )
        return 1.0 - dice


class DiceBCELoss(nn.Module):
    """Combined Dice + BCE loss for midline segmentation."""

    def __init__(self, dice_weight: float = 0.5, bce_weight: float = 0.5):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (
            self.dice_weight * self.dice_loss(pred, target)
            + self.bce_weight * self.bce_loss(pred, target)
        )


class CombinedLoss(nn.Module):
    """
    Combined loss for the dual-head model.

    L = L_midline + lambda_qc * L_qc

    Where:
        L_midline = DiceBCE(midline_pred, midline_target)
        L_qc = MSE(qc_pred, qc_target)
    """

    def __init__(self, lambda_qc: float = config.LAMBDA_QC):
        super().__init__()
        self.lambda_qc = lambda_qc
        self.midline_loss_fn = DiceBCELoss()
        self.qc_loss_fn = nn.MSELoss()

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute combined loss.

        Args:
            predictions: Dict with 'midline' and 'qc' tensors (logits, before sigmoid).
            targets: Dict with 'midline_mask' and 'qc_heatmap' tensors.

        Returns:
            (total_loss, midline_loss, qc_loss)
        """
        midline_loss = self.midline_loss_fn(
            predictions["midline"], targets["midline_mask"]
        )

        # QC loss uses sigmoid of predictions since target is in [0, 1]
        qc_pred_sigmoid = torch.sigmoid(predictions["qc"])
        qc_loss = self.qc_loss_fn(qc_pred_sigmoid, targets["qc_heatmap"])

        total_loss = midline_loss + self.lambda_qc * qc_loss

        return total_loss, midline_loss, qc_loss


def build_model(device: torch.device = None) -> DualHeadUNet:
    """
    Build and return the dual-head U-Net model.

    Args:
        device: Device to place model on. Auto-detects if None.

    Returns:
        DualHeadUNet model on the specified device.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DualHeadUNet(
        encoder_name=config.ENCODER_NAME,
        encoder_weights=config.ENCODER_WEIGHTS,
        in_channels=config.IN_CHANNELS,
    )

    model = model.to(device)

    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: DualHeadUNet ({config.ENCODER_NAME})")
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Device: {device}")

    return model
