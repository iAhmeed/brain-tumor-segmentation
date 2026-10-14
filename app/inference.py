"""
Brain Tumor Segmentation — Inference Engine

Runs forward pass on a single MRI slice and computes Dice scores
when ground-truth masks are provided.

Tumor regions:
    WT (Whole Tumor):      All tumor tissue (labels 1, 2, 3)
    TC (Tumor Core):       Necrotic core + enhancing (labels 1, 3)
    ET (Enhancing Tumor):  Enhancing tumor only (label 3)
"""

import numpy as np
import torch

# Tumor region names — order matches model output channels
REGIONS = ["WT", "TC", "ET"]


def predict(model, input_array: np.ndarray, device: str = "cpu"):
    """
    Run inference on a single preprocessed MRI slice.

    Args:
        model: Loaded U-Net model in eval mode.
        input_array: NumPy array of shape [4, H, W], float32.
                     Channels: T1, T1ce, T2, FLAIR (z-score normalized).
        device: Device string ('cpu' or 'cuda').

    Returns:
        Dictionary with binary masks for each tumor region:
        {
            "WT": np.ndarray (H, W), binary uint8,
            "TC": np.ndarray (H, W), binary uint8,
            "ET": np.ndarray (H, W), binary uint8,
        }
    """
    # Prepare input tensor: [1, 4, H, W]
    tensor = torch.from_numpy(input_array).unsqueeze(0).float().to(device)

    with torch.no_grad():
        logits = model(tensor)  # [1, 3, H, W]

    # Sigmoid → threshold at 0.5 → binary masks
    probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()  # [3, H, W]
    binary_masks = (probs > 0.5).astype(np.uint8)

    return {
        region: binary_masks[i]
        for i, region in enumerate(REGIONS)
    }


def compute_dice(pred_masks: dict, gt_masks: dict, smooth: float = 1e-6):
    """
    Compute Dice coefficient per tumor region.

    Args:
        pred_masks: Dict of predicted binary masks {region: ndarray (H, W)}.
        gt_masks:   Dict of ground-truth binary masks {region: ndarray (H, W)}.
        smooth:     Smoothing constant to avoid division by zero.

    Returns:
        Dictionary with Dice scores:
        {"WT": float, "TC": float, "ET": float, "mean": float}
    """
    scores = {}
    for region in REGIONS:
        p = pred_masks[region].flatten().astype(np.float32)
        t = gt_masks[region].flatten().astype(np.float32)
        intersection = (p * t).sum()
        dice = (2.0 * intersection + smooth) / (p.sum() + t.sum() + smooth)
        scores[region] = round(float(dice), 4)

    scores["mean"] = round(
        sum(scores[r] for r in REGIONS) / len(REGIONS), 4
    )
    return scores


def run_inference(model, input_array: np.ndarray, gt_array: np.ndarray = None,
                  device: str = "cpu"):
    """
    Full inference pipeline: prediction + optional Dice computation.

    Args:
        model:       Loaded U-Net model in eval mode.
        input_array: NumPy array [4, H, W], float32 — preprocessed MRI slice.
        gt_array:    Optional NumPy array [3, H, W], float32 — ground-truth masks
                     (channels: WT, TC, ET as binary 0/1). If None, Dice scores
                     are not computed.
        device:      Device string ('cpu' or 'cuda').

    Returns:
        (masks_dict, dice_dict)
        - masks_dict: {"WT": ndarray, "TC": ndarray, "ET": ndarray}
        - dice_dict:  {"WT": float, ..., "mean": float} or None if no GT
    """
    masks = predict(model, input_array, device)

    dice = None
    if gt_array is not None:
        gt_masks = {
            region: gt_array[i].astype(np.uint8)
            for i, region in enumerate(REGIONS)
        }
        dice = compute_dice(masks, gt_masks)

    return masks, dice
