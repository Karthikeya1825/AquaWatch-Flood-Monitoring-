"""
AquaWatch - Normalized Difference Water Index (NDWI) Classical Baselines
Computes water segmentation baselines using:
1. Fixed NDWI threshold (NDWI > 0.0, McFeeters 1996)
2. Otsu automated thresholding on NDWI

Sentinel-2 bands used:
  Green: Band 3 (Index 2 in 13-band, Index 1 in default RGB+NIR [B2, B3, B4, B8])
  NIR:   Band 8 (Index 7 in 13-band, Index 3 in default RGB+NIR [B2, B3, B4, B8])

Formula:
  NDWI = (Green - NIR) / (Green + NIR + epsilon)
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import tifffile
import matplotlib.pyplot as plt
from tqdm import tqdm


def compute_ndwi(green: np.ndarray, nir: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """
    Compute Normalized Difference Water Index (NDWI).
    Values range from -1.0 to +1.0.
    """
    g = green.astype(np.float32)
    n = nir.astype(np.float32)
    denom = g + n + eps
    ndwi = (g - n) / denom
    return np.clip(ndwi, -1.0, 1.0)


def otsu_threshold(image: np.ndarray, mask: np.ndarray = None, num_bins: int = 256) -> float:
    """
    Compute Otsu threshold on an NDWI image [-1.0, 1.0] for valid pixels.
    """
    if mask is not None:
        valid_pixels = image[mask]
    else:
        valid_pixels = image.ravel()

    if len(valid_pixels) == 0:
        return 0.0

    # Rescale [-1, 1] to [0, 255]
    scaled = np.clip((valid_pixels + 1.0) * 127.5, 0, 255).astype(np.uint8)
    hist, bin_edges = np.histogram(scaled, bins=num_bins, range=(0, 256))
    hist = hist.astype(np.float32)
    total = hist.sum()
    if total == 0:
        return 0.0

    current_max = 0.0
    threshold = 0
    sum_total = np.dot(np.arange(num_bins), hist)
    sum_b = 0.0
    w_b = 0.0

    for i in range(num_bins):
        w_b += hist[i]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += i * hist[i]
        m_b = sum_b / w_b
        m_f = (sum_total - sum_b) / w_f
        between_var = w_b * w_f * (m_b - m_f) ** 2
        if between_var > current_max:
            current_max = between_var
            threshold = i

    # Convert back from [0, 255] to [-1.0, 1.0]
    return (threshold / 127.5) - 1.0


def compute_metrics(tp: int, fp: int, fn: int, tn: int) -> Dict[str, float]:
    """Calculate IoU, F1/Dice, Precision, Recall, and Accuracy."""
    denom_iou = tp + fp + fn
    iou = float(tp / denom_iou) if denom_iou > 0 else 0.0

    denom_f1 = 2 * tp + fp + fn
    f1 = float((2 * tp) / denom_f1) if denom_f1 > 0 else 0.0

    denom_prec = tp + fp
    precision = float(tp / denom_prec) if denom_prec > 0 else 0.0

    denom_rec = tp + fn
    recall = float(tp / denom_rec) if denom_rec > 0 else 0.0

    total_valid = tp + tn + fp + fn
    accuracy = float((tp + tn) / total_valid) if total_valid > 0 else 0.0

    return {
        "iou": round(iou, 4),
        "f1": round(f1, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "accuracy": round(accuracy, 4),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn)
    }


def evaluate_ndwi_on_dataset(data_dir: Path, split: str = "test"):
    """
    Evaluate both NDWI (fixed threshold 0.0) and NDWI (Otsu adaptive)
    across the full split.
    """
    s2_dir = data_dir / "raw" / "S2Hand"
    label_dir = data_dir / "raw" / "LabelHand"
    split_csv = data_dir / "splits" / f"flood_{split}_data.csv"

    if not split_csv.exists():
        raise FileNotFoundError(f"Split CSV not found: {split_csv}")

    pairs = []
    with open(split_csv, mode="r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 2:
                s2_file = parts[0].strip().replace("_S1Hand.tif", "_S2Hand.tif")
                lbl_file = parts[1].strip()
                if (s2_dir / s2_file).exists() and (label_dir / lbl_file).exists():
                    pairs.append((s2_dir / s2_file, label_dir / lbl_file, s2_file))

    print(f"Evaluating NDWI Baselines on {len(pairs)} test image chips...")

    # Aggregate counters
    tp_zero, fp_zero, fn_zero, tn_zero = 0, 0, 0, 0
    tp_otsu, fp_otsu, fn_otsu, tn_otsu = 0, 0, 0, 0

    per_event_stats = {}
    per_chip_metrics = []

    for s2_path, lbl_path, s2_name in tqdm(pairs, desc="Computing NDWI"):
        s2_arr = tifffile.imread(str(s2_path))
        if s2_arr.ndim == 3 and s2_arr.shape[2] == 13:
            s2_arr = np.transpose(s2_arr, (2, 0, 1))

        # Green is Band 3 (index 2), NIR is Band 8 (index 7)
        green = s2_arr[2].astype(np.float32)
        nir = s2_arr[7].astype(np.float32)
        ndwi = compute_ndwi(green, nir)

        lbl_arr = tifffile.imread(str(lbl_path))
        if lbl_arr.ndim == 3:
            lbl_arr = lbl_arr[0]
        lbl_arr = lbl_arr.astype(np.int16)

        # Valid mask: ignore -1 (clouds / no data)
        valid_mask = (lbl_arr != -1)
        gt_water = (lbl_arr == 1)

        # Baseline 1: Fixed Threshold (NDWI > 0.0)
        pred_zero = (ndwi > 0.0) & valid_mask

        # Baseline 2: Otsu Threshold
        otsu_th = otsu_threshold(ndwi, mask=valid_mask)
        pred_otsu = (ndwi > otsu_th) & valid_mask

        # Compute counts for zero threshold
        c_tp_z = int(np.sum(pred_zero & gt_water))
        c_fp_z = int(np.sum(pred_zero & (~gt_water) & valid_mask))
        c_fn_z = int(np.sum((~pred_zero) & gt_water & valid_mask))
        c_tn_z = int(np.sum((~pred_zero) & (~gt_water) & valid_mask))

        tp_zero += c_tp_z
        fp_zero += c_fp_z
        fn_zero += c_fn_z
        tn_zero += c_tn_z

        # Compute counts for otsu threshold
        c_tp_o = int(np.sum(pred_otsu & gt_water))
        c_fp_o = int(np.sum(pred_otsu & (~gt_water) & valid_mask))
        c_fn_o = int(np.sum((~pred_otsu) & gt_water & valid_mask))
        c_tn_o = int(np.sum((~pred_otsu) & (~gt_water) & valid_mask))

        tp_otsu += c_tp_o
        fp_otsu += c_fp_o
        fn_otsu += c_fn_o
        tn_otsu += c_tn_o

        event = s2_name.split("_")[0]
        if event not in per_event_stats:
            per_event_stats[event] = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
        per_event_stats[event]["tp"] += c_tp_z
        per_event_stats[event]["fp"] += c_fp_z
        per_event_stats[event]["fn"] += c_fn_z
        per_event_stats[event]["tn"] += c_tn_z

        chip_metric = compute_metrics(c_tp_z, c_fp_z, c_fn_z, c_tn_z)
        chip_metric["chip_name"] = s2_name
        chip_metric["event"] = event
        chip_metric["otsu_threshold"] = round(float(otsu_th), 4)
        per_chip_metrics.append(chip_metric)

    # Compute overall metrics
    metrics_zero = compute_metrics(tp_zero, fp_zero, fn_zero, tn_zero)
    metrics_otsu = compute_metrics(tp_otsu, fp_otsu, fn_otsu, tn_otsu)

    event_metrics = {}
    for ev, c in per_event_stats.items():
        event_metrics[ev] = compute_metrics(c["tp"], c["fp"], c["fn"], c["tn"])

    results = {
        "dataset": "Sen1Floods11",
        "split": split,
        "total_test_chips": len(pairs),
        "ndwi_fixed_zero": metrics_zero,
        "ndwi_otsu": metrics_otsu,
        "per_event_metrics": event_metrics,
        "per_chip_metrics": per_chip_metrics
    }

    # Print summary table
    print("\n" + "=" * 80)
    print("REAL TEST SET BASELINE METRICS (Sen1Floods11 Hand-Labeled Test Split - 90 Chips)")
    print("=" * 80)
    print(f"{'Method / Baseline':<25} | {'IoU (Water)':<12} | {'F1 / Dice':<10} | {'Precision':<10} | {'Recall':<10} | {'Accuracy':<10}")
    print("-" * 80)
    print(f"{'NDWI (Threshold > 0.0)':<25} | {metrics_zero['iou']:<12.4f} | {metrics_zero['f1']:<10.4f} | {metrics_zero['precision']:<10.4f} | {metrics_zero['recall']:<10.4f} | {metrics_zero['accuracy']:<10.4f}")
    print(f"{'NDWI (Otsu Adaptive)':<25} | {metrics_otsu['iou']:<12.4f} | {metrics_otsu['f1']:<10.4f} | {metrics_otsu['precision']:<10.4f} | {metrics_otsu['recall']:<10.4f} | {metrics_otsu['accuracy']:<10.4f}")
    print("=" * 80)

    print("\nGeographic Regional Breakdown (NDWI > 0):")
    print("-" * 80)
    print(f"{'Region':<15} | {'IoU':<8} | {'F1':<8} | {'Precision':<10} | {'Recall':<10}")
    print("-" * 80)
    for ev, m in sorted(event_metrics.items()):
        print(f"{ev:<15} | {m['iou']:<8.4f} | {m['f1']:<8.4f} | {m['precision']:<10.4f} | {m['recall']:<10.4f}")
    print("-" * 80)

    # Save to results/baseline_metrics.json
    results_dir = data_dir.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_file = results_dir / "baseline_metrics.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved real baseline evaluation metrics to: {out_file}")

    # Plot confusion matrix and regional comparison
    plot_baseline_summary(results, results_dir / "figures" / "phase2_ndwi_baseline_results.png")

    return results


def plot_baseline_summary(results: Dict, output_path: Path):
    """Plot visual comparison of NDWI baselines and regional IoUs."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    events = list(results["per_event_metrics"].keys())
    ious = [results["per_event_metrics"][e]["iou"] for e in events]
    f1s = [results["per_event_metrics"][e]["f1"] for e in events]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=150)
    fig.patch.set_facecolor('#0B1F33')

    # Plot 1: Regional IoU Bar Chart
    ax1 = axes[0]
    ax1.set_facecolor('#0F2438')
    bars = ax1.bar(events, ious, color='#1D8CF8', edgecolor='#19C3B1', linewidth=1.2)
    ax1.set_title("NDWI Baseline IoU by Geographic Region", color='#F6FAFD', fontsize=12, pad=10)
    ax1.set_ylabel("IoU (Water Class)", color='#F6FAFD', fontsize=10)
    ax1.set_ylim(0, 1.0)
    ax1.tick_params(axis='x', rotation=45, colors='#F6FAFD')
    ax1.tick_params(axis='y', colors='#F6FAFD')
    ax1.grid(axis='y', linestyle='--', alpha=0.3, color='#5B7083')
    for bar in bars:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., h + 0.02, f"{h:.2f}",
                 ha='center', va='bottom', color='#F6FAFD', fontsize=8, fontweight='bold')

    # Plot 2: Confusion Matrix for NDWI (Threshold 0.0)
    ax2 = axes[1]
    ax2.set_facecolor('#0F2438')
    m0 = results["ndwi_fixed_zero"]
    cm = np.array([
        [m0["tn"], m0["fp"]],
        [m0["fn"], m0["tp"]]
    ])
    # Normalize by row
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    im = ax2.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
    ax2.set_title(f"Normalized Confusion Matrix (NDWI > 0.0)\nOverall Accuracy: {m0['accuracy']*100:.2f}% | Test IoU: {m0['iou']:.4f}",
                  color='#F6FAFD', fontsize=11, pad=10)
    ax2.set_xticks([0, 1])
    ax2.set_yticks([0, 1])
    ax2.set_xticklabels(["Pred Land", "Pred Water"], color='#F6FAFD', fontsize=10)
    ax2.set_yticklabels(["True Land", "True Water"], color='#F6FAFD', fontsize=10)

    for i in range(2):
        for j in range(2):
            raw_val = cm[i, j]
            pct = cm_norm[i, j] * 100.0
            text_color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax2.text(j, i, f"{raw_val:,}\n({pct:.1f}%)",
                     ha="center", va="center", color=text_color, fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(str(output_path), facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
    plt.close()
    print(f"Baseline comparison figure saved to: {output_path}")


if __name__ == "__main__":
    current_dir = Path(__file__).resolve().parent
    base_data = current_dir.parent / "data"
    evaluate_ndwi_on_dataset(base_data, split="test")
