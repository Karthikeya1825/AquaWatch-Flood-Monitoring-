"""
AquaWatch - Sen1Floods11 Hand-Labeled Dataset Downloader
Downloads the official hand-labeled Sentinel-2 images (S2Hand) and corresponding
ground truth masks (LabelHand) from Google Cloud Storage public mirror.

Dataset Source:
  Cloud to Street Sen1Floods11 v1.1
  Bucket: gs://sen1floods11
  Public HTTPS base: https://storage.googleapis.com/sen1floods11/

Usage:
  # Download 5 sample chips for quick verification & visualization:
  python download_sen1floods11.py --sample 5

  # Download the hand-labeled test split only:
  python download_sen1floods11.py --split test

  # Download all hand-labeled splits (train: 252, valid: 89, test: 90):
  python download_sen1floods11.py --split all
"""

import os
import sys
import csv
import argparse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Base URLs
GCS_BASE = "https://storage.googleapis.com/sen1floods11/v1.1"
SPLIT_URL_BASE = f"{GCS_BASE}/splits/flood_handlabeled"
S2_URL_BASE = f"{GCS_BASE}/data/flood_events/HandLabeled/S2Hand"
LABEL_URL_BASE = f"{GCS_BASE}/data/flood_events/HandLabeled/LabelHand"

SPLIT_FILES = {
    "train": "flood_train_data.csv",
    "valid": "flood_valid_data.csv",
    "test": "flood_test_data.csv",
    "bolivia": "flood_bolivia_data.csv"
}


def download_file(url: str, dest_path: Path, min_size: int = 100) -> bool:
    """Download a file with verification of existing non-empty files."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists() and dest_path.stat().st_size > min_size:
        return True  # already exists and non-empty

    temp_path = dest_path.with_suffix(dest_path.suffix + ".part")
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 AquaWatch-Downloader/1.0"}
        )
        with urllib.request.urlopen(req, timeout=30) as response, open(temp_path, "wb") as out_file:
            chunk_size = 64 * 1024
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                out_file.write(chunk)
        if temp_path.exists() and temp_path.stat().st_size > min_size:
            temp_path.replace(dest_path)
            return True
        else:
            if temp_path.exists():
                temp_path.unlink()
            return False
    except Exception as e:
        if temp_path.exists():
            temp_path.unlink()
        print(f"Error downloading {url}: {e}", file=sys.stderr)
        return False


def fetch_splits(splits_dir: Path):
    """Download official CSV split definitions if not already present."""
    splits_dir.mkdir(parents=True, exist_ok=True)
    for split_key, csv_name in SPLIT_FILES.items():
        dest = splits_dir / csv_name
        if not dest.exists() or dest.stat().st_size < 100:
            url = f"{SPLIT_URL_BASE}/{csv_name}"
            print(f"Fetching split definition: {csv_name}...")
            ok = download_file(url, dest)
            if not ok:
                raise RuntimeError(f"Failed to download split file from {url}")


def parse_split_csv(csv_path: Path):
    """
    Parse a split CSV file.
    The official CSV format has:
      Ghana_103272_S1Hand.tif,Ghana_103272_LabelHand.tif
    We convert the first column to S2:
      Ghana_103272_S2Hand.tif,Ghana_103272_LabelHand.tif
    """
    pairs = []
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or len(row) < 2:
                continue
            s1_filename = row[0].strip()
            label_filename = row[1].strip()
            s2_filename = s1_filename.replace("_S1Hand.tif", "_S2Hand.tif")
            pairs.append((s2_filename, label_filename))
    return pairs


def download_chip_pair(pair, s2_dir: Path, label_dir: Path):
    """Download both S2 image and corresponding ground truth label mask."""
    s2_file, label_file = pair
    s2_url = f"{S2_URL_BASE}/{s2_file}"
    label_url = f"{LABEL_URL_BASE}/{label_file}"

    s2_dest = s2_dir / s2_file
    label_dest = label_dir / label_file

    ok_s2 = download_file(s2_url, s2_dest, min_size=100)
    ok_label = download_file(label_url, label_dest, min_size=100)

    return s2_file, label_file, (ok_s2 and ok_label)


def main():
    parser = argparse.ArgumentParser(description="AquaWatch Sen1Floods11 Hand-Labeled Downloader")
    parser.add_argument("--split", type=str, default="sample", choices=["train", "valid", "test", "all", "sample"],
                        help="Which split to download (default: sample)")
    parser.add_argument("--sample", type=int, default=5,
                        help="Number of diverse sample chips to download if --split sample is used (default: 5)")
    parser.add_argument("--data_dir", type=str, default="../data",
                        help="Root data directory (relative or absolute)")
    parser.add_argument("--max_workers", type=int, default=6,
                        help="Concurrent download threads")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    splits_dir = data_dir / "splits"
    s2_dir = data_dir / "raw" / "S2Hand"
    label_dir = data_dir / "raw" / "LabelHand"

    print("=" * 70)
    print("AquaWatch: Sen1Floods11 Hand-Labeled Subset Downloader")
    print(f"Data directory: {data_dir}")
    print("=" * 70)

    # 1. Fetch official split CSVs
    fetch_splits(splits_dir)

    # 2. Collect target pairs
    all_pairs = []
    splits_to_process = []
    if args.split == "all":
        splits_to_process = ["train", "valid", "test"]
    elif args.split == "sample":
        splits_to_process = ["test"]
    else:
        splits_to_process = [args.split]

    for sp in splits_to_process:
        csv_path = splits_dir / SPLIT_FILES[sp]
        pairs = parse_split_csv(csv_path)
        all_pairs.extend(pairs)
        print(f"Found {len(pairs)} chip pairs in split '{sp}'.")

    if args.split == "sample":
        # Select diverse geographic samples (e.g. Spain, USA, Ghana, Paraguay, etc.)
        seen_events = set()
        sample_pairs = []
        for pair in all_pairs:
            event = pair[0].split("_")[0]
            if event not in seen_events:
                seen_events.add(event)
                sample_pairs.append(pair)
            if len(sample_pairs) >= args.sample:
                break
        # If we need more, pad from remaining
        if len(sample_pairs) < args.sample:
            for pair in all_pairs:
                if pair not in sample_pairs:
                    sample_pairs.append(pair)
                if len(sample_pairs) >= args.sample:
                    break
        target_pairs = sample_pairs
        print(f"\nSelected {len(target_pairs)} diverse sample pairs for verification:")
        for s2_name, lbl_name in target_pairs:
            print(f"  • S2: {s2_name} | Label: {lbl_name}")
    else:
        target_pairs = all_pairs

    print(f"\nStarting parallel download of {len(target_pairs)} pairs ({len(target_pairs)*2} files)...")

    succeeded = 0
    failed = 0
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(download_chip_pair, p, s2_dir, label_dir): p for p in target_pairs}
        for future in as_completed(futures):
            s2_file, label_file, ok = future.result()
            if ok:
                succeeded += 1
                print(f"  [OK] {s2_file} & {label_file}")
            else:
                failed += 1
                print(f"  [FAIL] {s2_file} or {label_file}", file=sys.stderr)

    print("\n" + "=" * 70)
    print(f"Download completed: {succeeded}/{len(target_pairs)} pairs downloaded successfully.")
    if failed > 0:
        print(f"WARNING: {failed} downloads failed. Check network connection and re-run.")
    print("=" * 70)


if __name__ == "__main__":
    main()
