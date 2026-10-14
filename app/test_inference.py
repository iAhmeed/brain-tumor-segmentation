"""
Smoke test for the inference engine.

Usage (from project root):
    python -m app.test_inference

Tests that the model loads correctly and can run inference
on a dummy input tensor + real demo samples (if available).
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

from app.model import load_model
from app.inference import run_inference, REGIONS


def test_with_dummy_data(ckpt_path: str, device: str):
    """Test inference with a random dummy tensor."""
    print("=" * 50)
    print("Inference Engine — Smoke Test")
    print("=" * 50)

    # Load model
    print(f"\nDevice: {device}")
    print(f"Loading model from: {ckpt_path}")
    model = load_model(ckpt_path, device=device)
    print("Model loaded successfully.")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")

    # Create dummy input (random noise simulating a 4-channel MRI slice)
    print("\nRunning inference on dummy input [4, 192, 192]...")
    dummy_input = np.random.randn(4, 192, 192).astype(np.float32)

    # Also create a dummy GT for testing Dice computation
    dummy_gt = np.random.randint(0, 2, size=(3, 192, 192)).astype(np.float32)

    masks, dice = run_inference(model, dummy_input, gt_array=dummy_gt, device=device)

    print("\nPredicted masks:")
    for region in REGIONS:
        mask = masks[region]
        pixel_count = int(mask.sum())
        print(f"  {region}: shape {mask.shape}, positive pixels: {pixel_count}")

    print("\nDice scores (vs random GT — expect ~0.5):")
    for key, val in dice.items():
        print(f"  {key}: {val:.4f}")

    # Test without GT
    print("\nRunning inference without ground truth...")
    masks2, dice2 = run_inference(model, dummy_input, device=device)
    assert dice2 is None, "Dice should be None when no GT is provided"
    print("  Dice correctly returned as None.")

    print("\n  All dummy-data tests passed!")
    return model


def test_with_demo_samples(model, device: str):
    """Test inference with actual demo samples if available."""
    demo_dir = Path("demo_samples")
    manifest_path = demo_dir / "manifest.json"

    if not manifest_path.exists():
        print("\nNo demo samples found — skipping real data test.")
        print("Run scripts/extract_demo_samples.py to generate them.")
        return

    with open(manifest_path) as f:
        manifest = json.load(f)

    print("\n" + "=" * 50)
    print("Testing with demo samples")
    print("=" * 50)

    for sample in manifest:
        sid = sample["id"]
        input_path = demo_dir / f"sample_{sid}_input.npy"
        gt_path = demo_dir / f"sample_{sid}_gt.npy"

        x = np.load(input_path)
        gt = np.load(gt_path)

        masks, dice = run_inference(model, x, gt_array=gt, device=device)

        print(f"\n  {sample['label']}:")
        for region in REGIONS:
            print(f"    {region}: dice={dice[region]:.4f}, "
                  f"pred_pixels={int(masks[region].sum())}")
        print(f"    Mean: {dice['mean']:.4f}")

    print("\n  All demo sample tests passed!")


def main():
    ckpt_path = Path("ckpt/best.pth")
    if not ckpt_path.exists():
        print(f"Error: Checkpoint not found at {ckpt_path}")
        print("Make sure you have the trained model in ckpt/best.pth")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = test_with_dummy_data(str(ckpt_path), device)
    test_with_demo_samples(model, device)

    print("\n" + "=" * 50)
    print("All tests passed!")
    print("=" * 50)


if __name__ == "__main__":
    main()
