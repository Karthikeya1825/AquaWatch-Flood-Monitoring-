"""
AquaWatch - Standalone Model Inference Script

Run inference on any Sentinel-2 GeoTIFF or standard image using the trained U-Net model.

Usage:
  # Predict water mask for a single Sentinel-2 GeoTIFF:
  python infer.py --input ../data/raw/S2Hand/India_900498_S2Hand.tif --output results/india_prediction.png

  # Compare with NDWI side-by-side:
  python infer.py --input ../data/raw/S2Hand/India_900498_S2Hand.tif --compare_ndwi --output results/india_comparison.png
"""

import os
import sys
import argparse
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image
import torch
import matplotlib.pyplot as plt

# Internal imports
from model import build_model
from ndwi_baseline import compute_ndwi
from dataset import DEFAULT_BANDS


def load_input_image(image_path: Path):
    """
    Load an image (GeoTIFF or RGB) and format it as a 4-channel tensor (RGB + NIR)
    normalized to [0.0, 1.0].
    """
    ext = image_path.suffix.lower()
    if ext in [".tif", ".tiff"]:
        raw = tifffile.imread(str(image_path))
        if raw.ndim == 3 and raw.shape[2] == 13:
            raw = np.transpose(raw, (2, 0, 1))

        if raw.shape[0] >= 8:
            # 13-band Sentinel-2: B2, B3, B4, B8
            selected = raw[[1, 2, 3, 7], :, :].astype(np.float32)
            # Reflectance scale factor (10,000x)
            tensor_np = np.clip(selected / 10000.0, 0.0, 1.0)
            green_raw = raw[2].astype(np.float32)
            nir_raw = raw[7].astype(np.float32)
            r = raw[3].astype(np.float32)
            g = raw[2].astype(np.float32)
            b = raw[1].astype(np.float32)
        elif raw.shape[0] >= 4:
            selected = raw[:4, :, :].astype(np.float32)
            tensor_np = np.clip(selected / 10000.0, 0.0, 1.0)
            green_raw = raw[1].astype(np.float32)
            nir_raw = raw[3].astype(np.float32)
            r, g, b = raw[0], raw[1], raw[2]
        else:
            # Replicate channels for 3-band
            tensor_np = np.repeat(raw[:3, :, :].astype(np.float32) / 255.0, 1, axis=0)
            tensor_np = np.concatenate([tensor_np, tensor_np[:1]], axis=0)
            green_raw = raw[1].astype(np.float32)
            nir_raw = raw[0].astype(np.float32)
            r, g, b = raw[0], raw[1], raw[2]
    else:
        # Standard PNG/JPEG
        pil_img = Image.open(str(image_path)).convert("RGB")
        arr = np.array(pil_img).transpose((2, 0, 1)).astype(np.float32) / 255.0
        tensor_np = np.concatenate([arr, arr[1:2]], axis=0) # Synthetic NIR
        green_raw = arr[1]
        nir_raw = arr[0]
        r, g, b = arr[0], arr[1], arr[2]

    # Create RGB preview
    rgb = np.stack([r, g, b], axis=-1).astype(np.float32)
    p2, p98 = np.percentile(rgb, 2.0), np.percentile(rgb, 98.0)
    if p98 <= p2: p98 = p2 + 1.0
    rgb_preview = (np.clip((rgb - p2) / (p98 - p2), 0.0, 1.0) * 255).astype(np.uint8)

    ndwi_map = compute_ndwi(green_raw, nir_raw)
    return tensor_np, rgb_preview, ndwi_map


def run_inference(
    input_path: Path,
    output_path: Path,
    checkpoint_path: Path,
    threshold: float = 0.5,
    compare_ndwi: bool = False,
    device_str: str = "auto"
):
    print("=" * 70)
    print("AquaWatch: Sentinel-2 Satellite Water Segmentation Inference")
    print(f"Input: {input_path}")
    print("=" * 70)

    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    # 1. Load Model
    model = build_model("unet_resnet34", in_channels=4, classes=1, encoder_weights=None).to(device)
    if checkpoint_path.exists():
        ckpt = torch.load(str(checkpoint_path), map_location=device)
        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        else:
            model.load_state_dict(ckpt)
        print(f"Loaded checkpoint from: {checkpoint_path}")
    else:
        print(f"WARNING: Checkpoint {checkpoint_path} not found. Running with baseline weights.")

    model.eval()

    # 2. Process Input
    tensor_np, rgb_preview, ndwi_map = load_input_image(input_path)
    img_tensor = torch.from_numpy(tensor_np).unsqueeze(0).float().to(device)

    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.sigmoid(logits).squeeze().cpu().numpy()
        pred_mask = (probs > threshold).astype(np.uint8)

    # 3. Calculate Quantitative Statistics
    total_pixels = pred_mask.size
    water_pixels = int(np.sum(pred_mask == 1))
    water_pct = (water_pixels / total_pixels) * 100.0
    # 1 pixel at 10m resolution = 100 m² = 0.0001 km²
    area_km2 = (water_pixels * 100.0) / 1_000_000.0

    print("\nQuantitative Prediction Results:")
    print(f"  • Image Dimensions:   {pred_mask.shape[1]} x {pred_mask.shape[0]} px")
    print(f"  • Water Pixels:       {water_pixels:,} / {total_pixels:,}")
    print(f"  • Water Coverage:     {water_pct:.2f}%")
    print(f"  • Water Surface Area: {area_km2:.3f} km²")

    # 4. Colorize and Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    h, w = pred_mask.shape
    colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
    colored_mask[pred_mask == 0] = [45, 55, 72]      # Dark Slate Land
    colored_mask[pred_mask == 1] = [29, 140, 248]    # Primary Water Blue

    if compare_ndwi:
        ndwi_pred = (ndwi_map > 0.0).astype(np.uint8)
        ndwi_colored = np.zeros((h, w, 3), dtype=np.uint8)
        ndwi_colored[ndwi_pred == 0] = [45, 55, 72]
        ndwi_colored[ndwi_pred == 1] = [25, 195, 177]  # Teal

        fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=150)
        fig.patch.set_facecolor('#0B1F33')

        axes[0].imshow(rgb_preview)
        axes[0].set_title(f"True Color (RGB)\n{input_path.name}", color='#F6FAFD', fontsize=11)
        axes[0].axis('off')

        axes[1].imshow(ndwi_colored)
        axes[1].set_title("NDWI Baseline (> 0.0)\nMcFeeters 1996", color='#19C3B1', fontsize=11)
        axes[1].axis('off')

        axes[2].imshow(colored_mask)
        axes[2].set_title(f"AquaWatch U-Net Prediction\nArea: {area_km2:.3f} km² ({water_pct:.1f}%)", color='#1D8CF8', fontsize=11)
        axes[2].axis('off')

        plt.tight_layout()
        plt.savefig(str(output_path), facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
        plt.close()
    else:
        Image.fromarray(colored_mask).save(str(output_path))

    print(f"\nSaved prediction to: {output_path}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="AquaWatch U-Net Inference")
    parser.add_argument("--input", type=str, required=True, help="Input GeoTIFF or image file")
    parser.add_argument("--output", type=str, default="results/prediction.png", help="Output PNG path")
    parser.add_argument("--checkpoint", type=str, default="../results/checkpoints/best_model.pt", help="Model checkpoint")
    parser.add_argument("--threshold", type=float, default=0.5, help="Sigmoid classification threshold")
    parser.add_argument("--compare_ndwi", action="store_true", help="Generate side-by-side NDWI comparison")
    parser.add_argument("--device", type=str, default="auto", help="Compute device ('cuda', 'cpu', 'auto')")
    args = parser.parse_args()

    inp = Path(args.input).resolve()
    out = Path(args.output).resolve()
    ckpt = Path(args.checkpoint).resolve()

    if not inp.exists():
        print(f"Error: Input file '{inp}' does not exist.", file=sys.stderr)
        sys.exit(1)

    run_inference(inp, out, ckpt, threshold=args.threshold, compare_ndwi=args.compare_ndwi, device_str=args.device)


if __name__ == "__main__":
    main()
