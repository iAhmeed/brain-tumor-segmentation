"""
Brain Tumor Segmentation — FastAPI Backend Application

Endpoints:
- GET  /health                  Health check & model status
- GET  /samples                 List available bundled demo samples
- GET  /samples/{id}/preview    Get thumbnail image for a sample
- POST /predict/sample/{id}     Run inference on a bundled demo sample
- POST /predict/upload-npy      Run inference on an uploaded .npy slice
- POST /predict/upload-nii      Run inference on uploaded 4-channel NIfTI files
- POST /predict/upload          Unified endpoint accepting either .npy or NIfTI
"""

import io
import json
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.inference import REGIONS, run_inference
from app.model import load_model
from app.preprocess import (
    image_to_base64_data_uri,
    make_preview_image,
    preprocess_nifti_volumes,
    preprocess_npy_array,
)

BASE_DIR = Path(__file__).resolve().parent.parent
CKPT_PATH = BASE_DIR / "ckpt" / "best.pth"
DEMO_DIR = BASE_DIR / "demo_samples"
MANIFEST_PATH = DEMO_DIR / "manifest.json"

# Global state for loaded model and device
state = {
    "model": None,
    "device": "cpu",
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    state["device"] = device
    print(f"Initializing Brain Tumor Segmentation API on device: {device}")

    if not CKPT_PATH.exists():
        print(f"WARNING: Checkpoint file not found at {CKPT_PATH}")
    else:
        print(f"Loading checkpoint from: {CKPT_PATH}")
        try:
            state["model"] = load_model(str(CKPT_PATH), device=device)
            print("Model loaded successfully into memory.")
        except Exception as e:
            print(f"Error loading model: {e}")

    yield
    print("Shutting down API server...")


app = FastAPI(
    title="Brain Tumor Segmentation API",
    description="FastAPI service serving U-Net (ResNet34) for BraTS brain tumor segmentation",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for local dev & frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



def ensure_model_loaded():
    if state["model"] is None:
        device = state["device"]
        if CKPT_PATH.exists():
            state["model"] = load_model(str(CKPT_PATH), device=device)
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Model checkpoint not found at {CKPT_PATH}",
            )
    return state["model"], state["device"]


@app.get("/health")
def health_check():
    """Health check endpoint showing model and hardware status."""
    model, device = ensure_model_loaded()
    return {
        "status": "ok",
        "device": device,
        "cuda_available": torch.cuda.is_available(),
        "checkpoint_loaded": model is not None,
        "checkpoint_path": str(CKPT_PATH),
    }


@app.get("/samples")
def get_samples():
    """List all available demo samples with their metadata and preview URLs."""
    if not MANIFEST_PATH.exists():
        return []

    try:
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        for item in manifest:
            item["preview_url"] = f"/samples/{item['id']}/preview"
        return manifest
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read demo manifest: {e}")


@app.get("/samples/{sample_id}/preview")
def get_sample_preview(sample_id: str):
    """Return the thumbnail preview PNG for a bundled sample."""
    sample_id = sample_id.zfill(2)
    preview_file = DEMO_DIR / f"sample_{sample_id}_preview.png"
    if not preview_file.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Preview image for sample '{sample_id}' not found",
        )
    return FileResponse(str(preview_file), media_type="image/png")


@app.post("/predict/sample/{sample_id}")
def predict_sample(sample_id: str):
    """
    Run inference on a bundled demo sample.
    Computes segmentation masks and Dice score against the bundled ground truth.
    """
    sample_id = sample_id.zfill(2)
    input_path = DEMO_DIR / f"sample_{sample_id}_input.npy"
    gt_path = DEMO_DIR / f"sample_{sample_id}_gt.npy"
    preview_path = DEMO_DIR / f"sample_{sample_id}_preview.png"

    if not input_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Demo sample '{sample_id}' not found on disk",
        )

    model, device = ensure_model_loaded()

    try:
        input_array = np.load(str(input_path)).astype(np.float32)
        gt_array = np.load(str(gt_path)).astype(np.float32) if gt_path.exists() else None

        masks, dice = run_inference(model, input_array, gt_array=gt_array, device=device)

        # Build base64 preview
        preview_img = make_preview_image(input_array)
        preview_base64 = image_to_base64_data_uri(preview_img)

        return {
            "sample_id": sample_id,
            "masks": {r: masks[r].tolist() for r in REGIONS},
            "dice": dice,
            "preview_base64": preview_base64,
            "width": int(masks["WT"].shape[1]),
            "height": int(masks["WT"].shape[0]),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")


@app.post("/predict/upload-npy")
async def predict_upload_npy(
    input_file: UploadFile = File(..., description="4-channel .npy MRI slice [4, H, W]"),
    gt_file: Optional[UploadFile] = File(None, description="Optional ground truth .npy [3, H, W]"),
):
    """Run inference on an uploaded .npy array slice."""
    model, device = ensure_model_loaded()

    try:
        input_bytes = await input_file.read()
        raw_input = np.load(io.BytesIO(input_bytes))

        raw_gt = None
        if gt_file is not None:
            gt_bytes = await gt_file.read()
            raw_gt = np.load(io.BytesIO(gt_bytes))

        input_array, gt_array, preview_img = preprocess_npy_array(raw_input, raw_gt)
        masks, dice = run_inference(model, input_array, gt_array=gt_array, device=device)
        preview_base64 = image_to_base64_data_uri(preview_img)

        return {
            "source": "upload-npy",
            "masks": {r: masks[r].tolist() for r in REGIONS},
            "dice": dice,
            "preview_base64": preview_base64,
            "width": int(masks["WT"].shape[1]),
            "height": int(masks["WT"].shape[0]),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process .npy upload: {e}")


@app.post("/predict/upload-nii")
async def predict_upload_nii(
    t1: UploadFile = File(..., description="T1 NIfTI file (.nii or .nii.gz)"),
    t1ce: UploadFile = File(..., description="T1ce NIfTI file (.nii or .nii.gz)"),
    t2: UploadFile = File(..., description="T2 NIfTI file (.nii or .nii.gz)"),
    flair: UploadFile = File(..., description="FLAIR NIfTI file (.nii or .nii.gz)"),
    seg: Optional[UploadFile] = File(None, description="Optional BraTS segmentation file"),
    slice_idx: Optional[int] = Form(None, description="Optional axial slice index"),
):
    """
    Run inference on 4 uploaded BraTS MRI scans (T1, T1ce, T2, FLAIR).
    Preprocesses volumes, picks the best tumor/brain slice, and segments it.
    """
    model, device = ensure_model_loaded()

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Save uploaded files to disk for nibabel
        async def save_upload(upload_file: UploadFile, filename: str) -> Path:
            ext = ".nii.gz" if upload_file.filename.endswith(".nii.gz") else ".nii"
            dest = tmp_path / f"{filename}{ext}"
            content = await upload_file.read()
            with open(dest, "wb") as f:
                f.write(content)
            return dest

        try:
            t1_path = await save_upload(t1, "t1")
            t1ce_path = await save_upload(t1ce, "t1ce")
            t2_path = await save_upload(t2, "t2")
            flair_path = await save_upload(flair, "flair")
            seg_path = await save_upload(seg, "seg") if seg is not None else None

            input_slice, gt_slice, selected_slice, preview_img = preprocess_nifti_volumes(
                t1_path, t1ce_path, t2_path, flair_path, seg_path, slice_idx=slice_idx
            )

            masks, dice = run_inference(model, input_slice, gt_array=gt_slice, device=device)
            preview_base64 = image_to_base64_data_uri(preview_img)

            return {
                "source": "upload-nii",
                "slice_idx": selected_slice,
                "masks": {r: masks[r].tolist() for r in REGIONS},
                "dice": dice,
                "preview_base64": preview_base64,
                "width": int(masks["WT"].shape[1]),
                "height": int(masks["WT"].shape[0]),
            }
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to process NIfTI upload: {e}")


@app.post("/predict/upload")
async def predict_unified_upload(
    # Option A: .npy file
    input_npy: Optional[UploadFile] = File(None),
    gt_npy: Optional[UploadFile] = File(None),
    # Option B: 4 NIfTI files
    t1: Optional[UploadFile] = File(None),
    t1ce: Optional[UploadFile] = File(None),
    t2: Optional[UploadFile] = File(None),
    flair: Optional[UploadFile] = File(None),
    seg: Optional[UploadFile] = File(None),
    slice_idx: Optional[int] = Form(None),
):
    """
    Unified upload endpoint:
    - If input_npy is provided, processes as .npy slice.
    - If t1, t1ce, t2, flair are provided, processes as 4 NIfTI scans.
    """
    if input_npy is not None and input_npy.filename:
        return await predict_upload_npy(input_file=input_npy, gt_file=gt_npy)

    if t1 is not None and t1ce is not None and t2 is not None and flair is not None:
        return await predict_upload_nii(
            t1=t1, t1ce=t1ce, t2=t2, flair=flair, seg=seg, slice_idx=slice_idx
        )

    raise HTTPException(
        status_code=400,
        detail="Invalid upload. Provide either 'input_npy' or all 4 NIfTI files ('t1', 't1ce', 't2', 'flair').",
    )


# Mount frontend directory for SPA serving (must be mounted after API routes)
frontend_dir = BASE_DIR / "frontend"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
