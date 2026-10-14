"""
Extract demo sample slices from the BraTS cache for the web app.

Usage:
    python scripts/extract_demo_samples.py --cache /path/to/brats_cache --output demo_samples --count 5

This script is meant to be run ONCE by the developer (wherever the BraTS
cache is available, e.g. on Kaggle). The generated demo_samples/ directory
should then be committed to the repository so that anyone cloning the repo
can run the demo without needing BraTS data.

Cache format (from data-preparation notebook):
    {patient_name}_x.npy — shape [num_slices, 4, 192, 192], float16
    {patient_name}_y.npy — shape [num_slices, 192, 192], uint8
    Segmentation labels: 0=background, 1=necrotic core, 2=edema, 3=enhancing
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def find_best_slices(cache_dir: Path, count: int = 5, min_tumor_pixels: int = 500):
    """
    Scan the BraTS cache and pick slices with the most tumor content,
    spread across different patients for diversity.

    Returns a list of dicts: [{patient, slice_idx, tumor_pixels}, ...]
    """
    candidates = []

    x_files = sorted(cache_dir.glob("*_x.npy"))
    print(f"  Found {len(x_files)} patients in cache.")

    for x_path in x_files:
        patient_name = x_path.stem.replace("_x", "")
        y_path = cache_dir / f"{patient_name}_y.npy"

        if not y_path.exists():
            continue

        y = np.load(y_path, mmap_mode="r")  # [num_slices, H, W]

        for i in range(len(y)):
            tumor_pixels = int((y[i] > 0).sum())
            if tumor_pixels >= min_tumor_pixels:
                candidates.append({
                    "patient": patient_name,
                    "slice_idx": i,
                    "tumor_pixels": tumor_pixels,
                })

    # Sort by tumor content descending
    candidates.sort(key=lambda c: c["tumor_pixels"], reverse=True)

    # Pick best slice from each unique patient first (for diversity)
    selected = []
    used_patients = set()

    for c in candidates:
        if len(selected) >= count:
            break
        if c["patient"] not in used_patients:
            selected.append(c)
            used_patients.add(c["patient"])

    # If still need more, allow duplicates from the same patient
    if len(selected) < count:
        for c in candidates:
            if len(selected) >= count:
                break
            if c not in selected:
                selected.append(c)

    return selected[:count]


def make_preview_png(x_slice: np.ndarray) -> Image.Image:
    """
    Create a grayscale preview image from the FLAIR channel (index 3).
    FLAIR typically shows tumors most clearly.
    """
    channel = x_slice[3].astype(np.float32)

    # Normalize to 0–255
    cmin, cmax = channel.min(), channel.max()
    if cmax > cmin:
        channel = (channel - cmin) / (cmax - cmin) * 255.0
    else:
        channel = np.zeros_like(channel)

    return Image.fromarray(channel.astype(np.uint8), mode="L")


def build_gt_masks(y_slice: np.ndarray) -> np.ndarray:
    """
    Convert raw segmentation labels to the 3-channel binary mask format
    used by the model (matching CachedBraTS dataset class from notebook).

    Channels:
        0 = WT (Whole Tumor):      y > 0
        1 = TC (Tumor Core):       (y == 1) | (y == 3)
        2 = ET (Enhancing Tumor):  y == 3
    """
    return np.stack([
        y_slice > 0,
        (y_slice == 1) | (y_slice == 3),
        y_slice == 3,
    ]).astype(np.float32)


def generate_synthetic_samples(output_dir: Path, count: int = 5):
    """
    Generate realistic synthetic brain MRI slices with simulated tumors
    for local demo and testing when real BraTS cache data is not locally available.
    """
    print(f"\nGenerating {count} synthetic brain MRI demo samples...\n")
    manifest = []
    
    np.random.seed(42)
    H, W = 192, 192
    y_grid, x_grid = np.ogrid[:H, :W]

    case_configs = [
        {"name": "Frontal Lobe High-Grade Glioma", "center": (75, 115), "r_wt": 24, "r_tc": 15, "r_et": 9},
        {"name": "Parietal Glioblastoma (Multifocal/Ring)", "center": (115, 75), "r_wt": 28, "r_tc": 18, "r_et": 12},
        {"name": "Temporal Astrocytoma (Edema Dominant)", "center": (120, 125), "r_wt": 26, "r_tc": 13, "r_et": 7},
        {"name": "Occipital Mass (Solid Enhancing Core)", "center": (135, 95), "r_wt": 20, "r_tc": 14, "r_et": 10},
        {"name": "Deep Periventricular Glioma", "center": (90, 85), "r_wt": 22, "r_tc": 14, "r_et": 8},
    ]

    for idx in range(1, count + 1):
        sample_id = f"{idx:02d}"
        cfg = case_configs[(idx - 1) % len(case_configs)]
        
        # Brain mask: ellipse
        cy_brain, cx_brain = 96, 96
        ry_brain, rx_brain = 72, 60
        dist_brain = ((y_grid - cy_brain) / ry_brain) ** 2 + ((x_grid - cx_brain) / rx_brain) ** 2
        brain_mask = dist_brain <= 1.0

        # Ventricles: two small central ellipses
        v1 = (((y_grid - 90) / 20) ** 2 + ((x_grid - 90) / 7) ** 2) <= 1.0
        v2 = (((y_grid - 90) / 20) ** 2 + ((x_grid - 102) / 7) ** 2) <= 1.0
        ventricles = (v1 | v2) & brain_mask

        # Tumor regions: WT (outer), TC (intermediate), ET (core/rim)
        ty, tx = cfg["center"]
        dist_tumor = np.sqrt((y_grid - ty) ** 2 + (x_grid - tx) ** 2)
        
        # Add slight boundary irregularity
        noise_wt = np.sin(y_grid * 0.2) * np.cos(x_grid * 0.2) * 2.0
        wt_mask = (dist_tumor <= (cfg["r_wt"] + noise_wt)) & brain_mask
        tc_mask = (dist_tumor <= cfg["r_tc"]) & wt_mask
        et_mask = (dist_tumor <= cfg["r_et"]) & tc_mask
        
        # Label map: 0=bg, 1=necrotic/non-enhancing (TC but not ET), 2=edema (WT but not TC), 3=enhancing (ET)
        y_slice = np.zeros((H, W), dtype=np.uint8)
        y_slice[wt_mask] = 2  # Edema
        y_slice[tc_mask] = 1  # Necrotic / non-enhancing core
        y_slice[et_mask] = 3  # Enhancing tumor

        tumor_pixels = int((y_slice > 0).sum())

        # Construct 4 modalities: 0: T1, 1: T1ce, 2: T2, 3: FLAIR
        channels = []
        for mod_idx, mod_name in enumerate(["t1", "t1ce", "t2", "flair"]):
            vol = np.zeros((H, W), dtype=np.float32)
            
            # Base brain parenchyma intensity
            if mod_name == "t1":
                vol[brain_mask] = 600.0 + np.random.normal(0, 15, size=vol.shape)[brain_mask]
                vol[ventricles] = 150.0  # CSF dark on T1
                vol[y_slice == 1] = 300.0  # Necrosis hypointense on T1
                vol[y_slice == 2] = 450.0  # Edema mildly hypointense
                vol[y_slice == 3] = 650.0
            elif mod_name == "t1ce":
                vol[brain_mask] = 600.0 + np.random.normal(0, 15, size=vol.shape)[brain_mask]
                vol[ventricles] = 150.0
                vol[y_slice == 1] = 300.0  # Necrotic core
                vol[y_slice == 2] = 480.0  # Edema
                vol[y_slice == 3] = 1100.0 # ENHANCING rim/core (bright on T1ce)
            elif mod_name == "t2":
                vol[brain_mask] = 500.0 + np.random.normal(0, 20, size=vol.shape)[brain_mask]
                vol[ventricles] = 1100.0 # CSF bright on T2
                vol[y_slice == 1] = 950.0
                vol[y_slice == 2] = 1200.0 # Edema very bright on T2
                vol[y_slice == 3] = 900.0
            elif mod_name == "flair":
                vol[brain_mask] = 550.0 + np.random.normal(0, 20, size=vol.shape)[brain_mask]
                vol[ventricles] = 120.0  # CSF attenuated (dark) on FLAIR
                vol[y_slice == 1] = 900.0
                vol[y_slice == 2] = 1300.0 # Edema/tumor hyperintense on FLAIR
                vol[y_slice == 3] = 1150.0

            # Z-score normalize non-zero voxels (matching BraTS pipeline)
            m = vol > 0
            if m.any():
                vol[m] = (vol[m] - vol[m].mean()) / (vol[m].std() + 1e-8)
            channels.append(vol)

        x_slice = np.stack(channels, axis=0).astype(np.float32)  # [4, 192, 192]
        gt_masks = build_gt_masks(y_slice)                     # [3, 192, 192]

        # Save files
        np.save(output_dir / f"sample_{sample_id}_input.npy", x_slice)
        np.save(output_dir / f"sample_{sample_id}_gt.npy", gt_masks)

        preview = make_preview_png(x_slice)
        preview_filename = f"sample_{sample_id}_preview.png"
        preview.save(output_dir / preview_filename)

        patient_label = f"Synthetic_{sample_id}"
        print(f"  Sample {sample_id}: {cfg['name']} ({tumor_pixels} tumor voxels)")

        manifest.append({
            "id": sample_id,
            "label": f"Sample {sample_id} — {cfg['name']}",
            "preview": preview_filename,
            "patient": patient_label,
            "slice_idx": 80,
            "tumor_pixels": tumor_pixels,
        })

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nDone! {len(manifest)} synthetic samples saved to: {output_dir}/")
    print(f"Manifest: {manifest_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract demo samples from BraTS cache or generate synthetic test samples"
    )
    parser.add_argument(
        "--cache", type=str, default=None,
        help="Path to the BraTS cache directory containing *_x.npy and *_y.npy files"
    )
    parser.add_argument(
        "--synthetic", action="store_true",
        help="Generate realistic synthetic brain MRI samples for testing without BraTS dataset"
    )
    parser.add_argument(
        "--output", type=str, default="demo_samples",
        help="Output directory for demo samples (default: demo_samples)"
    )
    parser.add_argument(
        "--count", type=int, default=5,
        help="Number of samples to extract (default: 5)"
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.synthetic or args.cache is None:
        if args.cache is None and not args.synthetic:
            print("No --cache specified; falling back to generating synthetic demo samples.")
            print("(To extract from real BraTS data, provide --cache /path/to/cache)\n")
        generate_synthetic_samples(output_dir, count=args.count)
        return

    cache_dir = Path(args.cache)
    if not cache_dir.exists():
        print(f"Error: Cache directory not found: {cache_dir}")
        print("Tip: Use --synthetic to generate demo test samples without BraTS cache.")
        return

    print(f"Scanning cache at: {cache_dir}")
    selected = find_best_slices(cache_dir, count=args.count)

    if not selected:
        print("Error: No suitable slices found in cache.")
        return

    print(f"\nExtracting {len(selected)} samples...\n")

    manifest = []

    for idx, sample in enumerate(selected, start=1):
        sample_id = f"{idx:02d}"
        patient = sample["patient"]
        slice_idx = sample["slice_idx"]

        print(f"  Sample {sample_id}: {patient}, slice {slice_idx} "
              f"({sample['tumor_pixels']} tumor pixels)")

        # Load data
        x = np.load(cache_dir / f"{patient}_x.npy", mmap_mode="r")
        y = np.load(cache_dir / f"{patient}_y.npy", mmap_mode="r")

        x_slice = x[slice_idx].astype(np.float32)  # [4, 192, 192]
        y_slice = y[slice_idx]                       # [192, 192]
        gt_masks = build_gt_masks(y_slice)            # [3, 192, 192]

        # Save input and ground-truth
        np.save(output_dir / f"sample_{sample_id}_input.npy", x_slice)
        np.save(output_dir / f"sample_{sample_id}_gt.npy", gt_masks)

        # Save preview PNG (FLAIR channel)
        preview = make_preview_png(x_slice)
        preview_filename = f"sample_{sample_id}_preview.png"
        preview.save(output_dir / preview_filename)

        manifest.append({
            "id": sample_id,
            "label": f"Sample {sample_id} — {patient}",
            "preview": preview_filename,
            "patient": patient,
            "slice_idx": slice_idx,
            "tumor_pixels": sample["tumor_pixels"],
        })

    # Save manifest
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nDone! {len(manifest)} samples saved to: {output_dir}/")
    print(f"Manifest: {manifest_path}")
    print("\nCommit the demo_samples/ directory to your repository.")


if __name__ == "__main__":
    main()
