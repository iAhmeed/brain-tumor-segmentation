"""
Brain Tumor Segmentation — Preprocessing Module

Handles loading and preprocessing for:
1. 4 raw BraTS NIfTI files (T1, T1ce, T2, FLAIR) + optional segmentation file.
2. Pre-extracted .npy slices (shape [4, H, W]) + optional ground truth .npy.
"""

import base64
import io
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import nibabel as nib
import numpy as np
from PIL import Image


def zscore(v: np.ndarray) -> np.ndarray:
    """
    Z-score normalize over non-zero voxels (matching BraTS training pipeline).
    """
    m = v > 0
    out = np.zeros_like(v, dtype=np.float32)
    if m.any():
        out[m] = (v[m] - v[m].mean()) / (v[m].std() + 1e-8)
    return out


def make_preview_image(x_slice: np.ndarray) -> Image.Image:
    """
    Generate a grayscale preview image from the FLAIR channel (channel 3)
    or T1 (channel 0) if FLAIR has no contrast.
    """
    channel = x_slice[3].astype(np.float32)
    cmin, cmax = channel.min(), channel.max()
    if cmax > cmin:
        norm = (channel - cmin) / (cmax - cmin) * 255.0
    else:
        norm = np.zeros_like(channel)
    return Image.fromarray(norm.astype(np.uint8), mode="L")


def image_to_base64_data_uri(image: Image.Image) -> str:
    """Convert a PIL Image to a base64 PNG data URI."""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def center_crop_or_pad(arr: np.ndarray, target_size: int = 192) -> np.ndarray:
    """
    Crop or pad a 2D array to (target_size, target_size).
    Matches BraTS standard 240x240 -> [24:216, 24:216] = 192x192.
    """
    h, w = arr.shape[-2], arr.shape[-1]
    if h == target_size and w == target_size:
        return arr

    if h >= target_size and w >= target_size:
        start_y = (h - target_size) // 2
        start_x = (w - target_size) // 2
        return arr[..., start_y : start_y + target_size, start_x : start_x + target_size]

    # Pad if smaller
    pad_y = max(0, target_size - h)
    pad_x = max(0, target_size - w)
    pad_top = pad_y // 2
    pad_bottom = pad_y - pad_top
    pad_left = pad_x // 2
    pad_right = pad_x - pad_left

    if arr.ndim == 2:
        return np.pad(arr, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="constant")
    elif arr.ndim == 3:
        return np.pad(arr, ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right)), mode="constant")
    return arr


def preprocess_nifti_volumes(
    t1_path: Path,
    t1ce_path: Path,
    t2_path: Path,
    flair_path: Path,
    seg_path: Optional[Path] = None,
    slice_idx: Optional[int] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], int, Image.Image]:
    """
    Preprocess 4 NIfTI volumes:
    - Load volumes and z-score normalize.
    - Select optimal axial slice (highest tumor content if seg given, or largest brain area).
    - Center crop to 192x192.
    - Return (input_slice [4, 192, 192], gt_slice [3, 192, 192] or None, slice_idx, preview_img).
    """
    # Load volumes
    t1_vol = nib.load(str(t1_path)).get_fdata(dtype=np.float32)
    t1ce_vol = nib.load(str(t1ce_path)).get_fdata(dtype=np.float32)
    t2_vol = nib.load(str(t2_path)).get_fdata(dtype=np.float32)
    flair_vol = nib.load(str(flair_path)).get_fdata(dtype=np.float32)

    seg_vol = None
    if seg_path is not None and seg_path.exists():
        seg_vol = nib.load(str(seg_path)).get_fdata().astype(np.uint8)
        seg_vol[seg_vol == 4] = 3

    num_slices = flair_vol.shape[2]

    # Pick slice if not explicitly given
    if slice_idx is None or slice_idx < 0 or slice_idx >= num_slices:
        if seg_vol is not None:
            # Pick slice with highest tumor count
            tumor_counts = [(seg_vol[:, :, s] > 0).sum() for s in range(num_slices)]
            best_slice = int(np.argmax(tumor_counts))
            if tumor_counts[best_slice] == 0:
                best_slice = num_slices // 2
        else:
            # Pick slice with largest brain cross section
            brain_counts = [(flair_vol[:, :, s] > 0).sum() for s in range(num_slices)]
            best_slice = int(np.argmax(brain_counts))
            if brain_counts[best_slice] == 0:
                best_slice = num_slices // 2
        slice_idx = best_slice

    # Z-score normalize non-zero voxels for each volume
    norm_t1 = zscore(t1_vol[:, :, slice_idx])
    norm_t1ce = zscore(t1ce_vol[:, :, slice_idx])
    norm_t2 = zscore(t2_vol[:, :, slice_idx])
    norm_flair = zscore(flair_vol[:, :, slice_idx])

    # Crop to 192x192
    c_t1 = center_crop_or_pad(norm_t1, target_size=192)
    c_t1ce = center_crop_or_pad(norm_t1ce, target_size=192)
    c_t2 = center_crop_or_pad(norm_t2, target_size=192)
    c_flair = center_crop_or_pad(norm_flair, target_size=192)

    input_array = np.stack([c_t1, c_t1ce, c_t2, c_flair], axis=0).astype(np.float32)

    gt_array = None
    if seg_vol is not None:
        raw_seg = center_crop_or_pad(seg_vol[:, :, slice_idx], target_size=192)
        wt = (raw_seg > 0).astype(np.float32)
        tc = ((raw_seg == 1) | (raw_seg == 3)).astype(np.float32)
        et = (raw_seg == 3).astype(np.float32)
        gt_array = np.stack([wt, tc, et], axis=0)

    preview_img = make_preview_image(input_array)
    return input_array, gt_array, slice_idx, preview_img


def preprocess_npy_array(
    input_array: np.ndarray,
    gt_array: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], Image.Image]:
    """
    Validate and format an uploaded .npy array.
    Expected input shape: [4, H, W] (or [H, W, 4]).
    """
    if input_array.ndim == 3 and input_array.shape[-1] == 4 and input_array.shape[0] != 4:
        input_array = np.transpose(input_array, (2, 0, 1))

    if input_array.ndim != 3 or input_array.shape[0] != 4:
        raise ValueError(
            f"Invalid input shape: expected [4, H, W], got {input_array.shape}"
        )

    # Crop/pad to 192x192 if necessary
    if input_array.shape[1] != 192 or input_array.shape[2] != 192:
        input_array = center_crop_or_pad(input_array, target_size=192)

    input_array = input_array.astype(np.float32)

    if gt_array is not None:
        if gt_array.ndim == 2:
            # Single channel label map: 0, 1, 2, 3
            raw_seg = center_crop_or_pad(gt_array, target_size=192)
            wt = (raw_seg > 0).astype(np.float32)
            tc = ((raw_seg == 1) | (raw_seg == 3)).astype(np.float32)
            et = (raw_seg == 3).astype(np.float32)
            gt_array = np.stack([wt, tc, et], axis=0)
        elif gt_array.ndim == 3:
            if gt_array.shape[-1] == 3 and gt_array.shape[0] != 3:
                gt_array = np.transpose(gt_array, (2, 0, 1))
            if gt_array.shape[1] != 192 or gt_array.shape[2] != 192:
                gt_array = center_crop_or_pad(gt_array, target_size=192)
            gt_array = gt_array.astype(np.float32)
        else:
            gt_array = None

    preview_img = make_preview_image(input_array)
    return input_array, gt_array, preview_img
