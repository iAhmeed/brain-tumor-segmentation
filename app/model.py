"""
Brain Tumor Segmentation — Model Definition

U-Net with ResNet34 encoder, trained on BraTS 2020 dataset.
- Input:  4-channel MRI slices (T1, T1ce, T2, FLAIR), shape [B, 4, 192, 192]
- Output: 3-channel logits (WT, TC, ET), shape [B, 3, 192, 192]
"""

import torch
import segmentation_models_pytorch as smp


def build_model():
    """Build the U-Net model matching the training configuration."""
    return smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=4,
        classes=3,
        activation=None,
    )


def load_model(ckpt_path: str, device: str = "cpu"):
    """
    Load trained model weights from a checkpoint file.

    Args:
        ckpt_path: Path to the .pth checkpoint file.
        device: Device to load the model onto ('cpu' or 'cuda').

    Returns:
        Model in eval mode on the specified device.
    """
    model = build_model()
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model.to(device)
