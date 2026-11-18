"""
Download and verify Multi-class Weather Dataset.
Supports:
1. Automated download from open GitHub mirror (ndb796/weather_dataset).
2. Existing local Kaggle extracted dataset in data/weather_raw/.
"""

import os
import shutil
import urllib.request
import zipfile
from pathlib import Path

MIRROR_ZIP_URL = "https://github.com/ndb796/weather_dataset/archive/refs/heads/master.zip"


def download_and_extract_weather_data(data_dir: Path = None):
    if data_dir is None:
        data_dir = Path(__file__).resolve().parent / "weather_raw"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Check if already extracted
    classes = [d for d in os.listdir(data_dir) if (data_dir / d).is_dir() and not d.startswith('.')]
    if len(classes) >= 4:
        print(f"Weather dataset already exists at {data_dir} with classes: {classes}")
        verify_dataset(data_dir)
        return data_dir

    # Check if user placed a kaggle zip
    kaggle_zips = list(data_dir.parent.glob("*weather*.zip")) + list(data_dir.glob("*.zip"))
    if kaggle_zips:
        chosen_zip = kaggle_zips[0]
        print(f"Found local zip archive: {chosen_zip}. Extracting...")
        with zipfile.ZipFile(chosen_zip, 'r') as zf:
            zf.extractall(data_dir)
        verify_dataset(data_dir)
        return data_dir

    # Download from open mirror
    print(f"Downloading Multi-class Weather Dataset from open mirror: {MIRROR_ZIP_URL}...")
    tmp_zip = data_dir.parent / "weather_mirror_temp.zip"
    req = urllib.request.Request(MIRROR_ZIP_URL, headers={"User-Agent": "AquaWatch/1.0"})
    with urllib.request.urlopen(req) as resp, open(tmp_zip, "wb") as out_f:
        shutil.copyfileobj(resp, out_f)

    print(f"Downloaded {os.path.getsize(tmp_zip) / (1024*1024):.1f} MB. Extracting...")
    with zipfile.ZipFile(tmp_zip, 'r') as zf:
        zf.extractall(data_dir)

    # Clean up temp zip
    if tmp_zip.exists():
        tmp_zip.unlink()

    # Reorganize if extracted into subfolder
    subdirs = [d for d in data_dir.iterdir() if d.is_dir()]
    if len(subdirs) == 1 and ("weather_dataset" in subdirs[0].name.lower() or "master" in subdirs[0].name.lower()):
        nested = subdirs[0]
        # Check if train/test exist inside
        if (nested / "train").exists():
            for split_dir in ["train", "test"]:
                for cls_dir in (nested / split_dir).iterdir():
                    if cls_dir.is_dir():
                        target_cls = data_dir / cls_dir.name.lower()
                        target_cls.mkdir(exist_ok=True)
                        for img_f in cls_dir.glob("*.*"):
                            dest = target_cls / f"{split_dir}_{img_f.name}"
                            if not dest.exists():
                                shutil.copy2(img_f, dest)
            shutil.rmtree(nested)

    verify_dataset(data_dir)
    return data_dir


def verify_dataset(data_dir: Path):
    print("\n--- WEATHER DATASET VERIFICATION ---")
    counts = {}
    valid_exts = {".jpg", ".jpeg", ".png", ".webp"}

    # Look for class folders directly in data_dir or inside train/test
    class_dirs = [d for d in data_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]

    # If dataset has train/test subfolders
    if any(d.name in ["train", "test"] for d in class_dirs):
        for split in ["train", "test"]:
            sp_dir = data_dir / split
            if sp_dir.exists():
                for cdir in sp_dir.iterdir():
                    if cdir.is_dir():
                        c_name = cdir.name.lower()
                        imgs = [f for f in cdir.glob("*.*") if f.suffix.lower() in valid_exts]
                        counts[c_name] = counts.get(c_name, 0) + len(imgs)
    else:
        for cdir in class_dirs:
            c_name = cdir.name.lower()
            imgs = [f for f in cdir.glob("*.*") if f.suffix.lower() in valid_exts]
            counts[c_name] = len(imgs)

    total_images = sum(counts.values())
    print(f"Total Valid Images: {total_images}")
    for c, n in sorted(counts.items()):
        print(f"  - Class '{c}': {n} images ({n/max(1, total_images)*100:.1f}%)")

    return counts


if __name__ == "__main__":
    download_and_extract_weather_data()
