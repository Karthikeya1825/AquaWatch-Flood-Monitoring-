"""
AquaWatch - Dataset Verification & Sample Visualizer
Inspects downloaded Sentinel-2 and LabelHand GeoTIFF chips, computes pixel statistics,
and generates a multi-panel visualization of sample image/mask pairs.
"""

import os
import sys
import glob
from pathlib import Path
import numpy as np
import tifffile
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def load_s2_bands(s2_path: Path):
    """
    Load Sentinel-2 13-band image using tifffile.
    Returns: array of shape (13, 512, 512) float32 normalized or raw uint16.
    Sentinel-2 Bands:
      0: Coastal (B1)
      1: Blue (B2)
      2: Green (B3)
      3: Red (B4)
      4: RedEdge-1 (B5)
      5: RedEdge-2 (B6)
      6: RedEdge-3 (B7)
      7: NIR (B8)
      8: Narrow NIR (B8A)
      9: Water Vapor (B9)
      10: Cirrus (B10)
      11: SWIR-1 (B11)
      12: SWIR-2 (B12)
    """
    arr = tifffile.imread(str(s2_path))
    # Shape might be (13, 512, 512) or (512, 512, 13)
    if arr.ndim == 3 and arr.shape[2] == 13:
        arr = np.transpose(arr, (2, 0, 1))
    return arr


def load_label_mask(label_path: Path):
    """
    Load LabelHand ground truth GeoTIFF.
    Values:
      -1: No data / cloud / invalid
       0: Land (not water)
       1: Water (flood / permanent water)
    """
    arr = tifffile.imread(str(label_path))
    if arr.ndim == 3:
        arr = arr[0]
    return arr.astype(np.int16)


def make_rgb_composite(s2_arr: np.ndarray, clip_percentile: float = 98.0):
    """
    Extract RGB bands (Red: B4 idx 3, Green: B3 idx 2, Blue: B2 idx 1)
    and apply percentile-based contrast stretch for visual clarity.
    """
    r = s2_arr[3].astype(np.float32)
    g = s2_arr[2].astype(np.float32)
    b = s2_arr[1].astype(np.float32)

    rgb = np.stack([r, g, b], axis=-1)
    # Sentinel-2 L1C TOA reflectance: 0-10,000 scale, typical surface 0-3000
    p_low = np.percentile(rgb, 2.0)
    p_high = np.percentile(rgb, clip_percentile)
    if p_high <= p_low:
        p_high = p_low + 1.0
    rgb_stretched = np.clip((rgb - p_low) / (p_high - p_low), 0.0, 1.0)
    return rgb_stretched


def make_false_color_composite(s2_arr: np.ndarray):
    """
    False-color NIR-Red-Green composite (B8, B4, B3)
    Water appears dark/black, vegetation bright red.
    """
    nir = s2_arr[7].astype(np.float32)
    r = s2_arr[3].astype(np.float32)
    g = s2_arr[2].astype(np.float32)

    fc = np.stack([nir, r, g], axis=-1)
    p_low = np.percentile(fc, 2.0)
    p_high = np.percentile(fc, 98.0)
    if p_high <= p_low:
        p_high = p_low + 1.0
    return np.clip((fc - p_low) / (p_high - p_low), 0.0, 1.0)


def colorize_label_mask(label_arr: np.ndarray):
    """
    Create an RGB representation of the ground truth label:
      -1: Light Gray/Yellow hash (Invalid / Cloud) -> [180, 180, 180]
       0: Dark Muted Greenish Gray (Land) -> [50, 60, 50]
       1: Vibrant Water Blue (Water) -> [29, 140, 248] (#1D8CF8)
    """
    h, w = label_arr.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)

    # Land: Dark background
    colored[label_arr == 0] = [45, 55, 72]      # Slate / dark land
    # Water: Primary blue
    colored[label_arr == 1] = [29, 140, 248]    # AquaWatch primary water blue
    # Invalid / Cloud (-1)
    colored[label_arr == -1] = [230, 200, 100]  # Amber/Yellow for no-data
    return colored


def verify_and_analyze(data_dir: Path, output_fig_path: Path):
    s2_dir = data_dir / "raw" / "S2Hand"
    label_dir = data_dir / "raw" / "LabelHand"

    s2_files = sorted(list(s2_dir.glob("*.tif")))
    if not s2_files:
        print(f"No files found in {s2_dir}. Run download_sen1floods11.py first!")
        return

    print(f"\nAnalyzing {len(s2_files)} downloaded sample files...")
    stats = []
    paired_samples = []

    for s2_file in s2_files:
        lbl_name = s2_file.name.replace("_S2Hand.tif", "_LabelHand.tif")
        lbl_file = label_dir / lbl_name
        if not lbl_file.exists():
            continue

        s2_arr = load_s2_bands(s2_file)
        lbl_arr = load_label_mask(lbl_file)

        total_pixels = lbl_arr.size
        water_pixels = int(np.sum(lbl_arr == 1))
        land_pixels = int(np.sum(lbl_arr == 0))
        nodata_pixels = int(np.sum(lbl_arr == -1))
        valid_pixels = water_pixels + land_pixels
        water_pct = (water_pixels / valid_pixels * 100.0) if valid_pixels > 0 else 0.0

        event_name = s2_file.name.split("_")[0]
        chip_id = s2_file.name.split("_")[1]

        chip_stat = {
            "s2_name": s2_file.name,
            "event": event_name,
            "chip_id": chip_id,
            "shape_s2": s2_arr.shape,
            "shape_lbl": lbl_arr.shape,
            "water_px": water_pixels,
            "land_px": land_pixels,
            "nodata_px": nodata_pixels,
            "water_pct": water_pct,
            "s2_arr": s2_arr,
            "lbl_arr": lbl_arr
        }
        stats.append(chip_stat)
        paired_samples.append(chip_stat)

    # Print markdown table of sample stats
    print("\n" + "=" * 90)
    print("DATASET VERIFICATION & PIXEL STATISTICS")
    print("=" * 90)
    print(f"{'Chip Name':<28} | {'Event':<10} | {'Water Px':<9} | {'Land Px':<9} | {'NoData Px':<9} | {'Water %':<7}")
    print("-" * 90)
    for s in stats:
        print(f"{s['s2_name']:<28} | {s['event']:<10} | {s['water_px']:<9} | {s['land_px']:<9} | {s['nodata_px']:<9} | {s['water_pct']:>6.2f}%")
    print("-" * 90)

    tot_water = sum(s["water_px"] for s in stats)
    tot_land = sum(s["land_px"] for s in stats)
    tot_nodata = sum(s["nodata_px"] for s in stats)
    tot_valid = tot_water + tot_land
    agg_water_pct = (tot_water / tot_valid * 100.0) if tot_valid > 0 else 0.0

    print(f"{'TOTAL / AGGREGATE':<28} | {'ALL':<10} | {tot_water:<9} | {tot_land:<9} | {tot_nodata:<9} | {agg_water_pct:>6.2f}%")
    print("=" * 90)

    # Generate multi-panel figure for up to 5 pairs
    num_to_plot = min(5, len(paired_samples))
    fig, axes = plt.subplots(num_to_plot, 3, figsize=(14, 3.8 * num_to_plot), dpi=150)
    if num_to_plot == 1:
        axes = np.expand_dims(axes, 0)

    fig.patch.set_facecolor('#0B1F33')
    plt.suptitle("AquaWatch - Sen1Floods11 Hand-Labeled Sample Pairs Verification\n"
                 "Columns: True Color (RGB: B4, B3, B2) | False Color (NIR, R, G) | Ground Truth Mask (Blue=Water, Gray=Land, Yellow=NoData)",
                 color='#F6FAFD', fontsize=12, fontweight='bold', y=0.995)

    for i in range(num_to_plot):
        sample = paired_samples[i]
        s2_arr = sample["s2_arr"]
        lbl_arr = sample["lbl_arr"]

        rgb = make_rgb_composite(s2_arr)
        fc = make_false_color_composite(s2_arr)
        lbl_col = colorize_label_mask(lbl_arr)

        # 1. RGB
        axes[i, 0].imshow(rgb)
        axes[i, 0].set_title(f"RGB True Color: {sample['event']} (Chip #{sample['chip_id']})",
                             color='#F6FAFD', fontsize=10, pad=6)
        axes[i, 0].axis('off')

        # 2. False Color
        axes[i, 1].imshow(fc)
        axes[i, 1].set_title(f"NIR Composite (B8/B4/B3) - Water Absorbs NIR",
                             color='#19C3B1', fontsize=10, pad=6)
        axes[i, 1].axis('off')

        # 3. Ground Truth Mask
        axes[i, 2].imshow(lbl_col)
        axes[i, 2].set_title(f"Hand Label: {sample['water_pct']:.1f}% Water ({sample['water_px']} px)",
                             color='#1D8CF8', fontsize=10, pad=6)
        axes[i, 2].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    output_fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_fig_path), facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
    plt.close()
    print(f"\nVerification figure saved to: {output_fig_path}")


if __name__ == "__main__":
    current_dir = Path(__file__).resolve().parent
    base_data = current_dir
    if not (base_data / "raw").exists():
        base_data = current_dir.parent / "data"

    fig_out = current_dir.parent / "results" / "figures" / "phase1_sample_verification.png"
    verify_and_analyze(base_data, fig_out)
