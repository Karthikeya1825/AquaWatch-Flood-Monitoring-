"""
AquaWatch - PyTorch Dataset & Transforms for Sen1Floods11 Sentinel-2 Imagery

Handles:
- Loading multi-spectral Sentinel-2 GeoTIFFs (13 bands or selected subset: RGB + NIR)
- Scaling 10,000x TOA reflectance into standard range [0.0, 1.0]
- Loading hand-labeled ground truth masks (1 = Water, 0 = Land, -1 = NoData / Cloud)
- Geospatial & spectral data augmentations via Albumentations
"""

import os
import csv
from pathlib import Path
from typing import List, Optional, Tuple, Callable

import numpy as np
import tifffile
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Sentinel-2 Standard Band Mapping (0-indexed)
# 0: B1 (Coastal Aerosol)
# 1: B2 (Blue)
# 2: B3 (Green)
# 3: B4 (Red)
# 4: B5 (RedEdge-1)
# 5: B6 (RedEdge-2)
# 6: B7 (RedEdge-3)
# 7: B8 (NIR - Broad)
# 8: B8A (NIR - Narrow)
# 9: B9 (Water Vapor)
# 10: B10 (Cirrus)
# 11: B11 (SWIR-1)
# 12: B12 (SWIR-2)
DEFAULT_BANDS = [1, 2, 3, 7]  # Blue, Green, Red, NIR (RGB + NIR)


def get_train_transforms(image_size: int = 512) -> A.Compose:
    """
    Data augmentations for training:
    - Random Horizontal and Vertical flips
    - Random 90 degree rotations
    - Subtle ShiftScaleRotate (preserving valid footprint)
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.08,
            rotate_limit=15,
            border_mode=0,
            p=0.5
        ),
    ])


def get_valid_transforms(image_size: int = 512) -> Optional[A.Compose]:
    """Validation / Test transform: deterministic evaluation."""
    return None


class Sen1Floods11Dataset(Dataset):
    """
    PyTorch Dataset for Sen1Floods11 Hand-Labeled Sentinel-2 Chips.
    """
    def __init__(
        self,
        data_dir: str or Path,
        split: str = "test",
        bands: List[int] = DEFAULT_BANDS,
        transform: Optional[A.Compose] = None,
        scale_factor: float = 10000.0,
        return_metadata: bool = True
    ):
        super().__init__()
        self.data_dir = Path(data_dir).resolve()
        self.split = split
        self.bands = bands
        self.transform = transform
        self.scale_factor = scale_factor
        self.return_metadata = return_metadata

        self.s2_dir = self.data_dir / "raw" / "S2Hand"
        self.label_dir = self.data_dir / "raw" / "LabelHand"
        self.split_csv = self.data_dir / "splits" / f"flood_{split}_data.csv"

        if not self.split_csv.exists():
            raise FileNotFoundError(f"Split CSV not found at: {self.split_csv}")

        # Parse pairs and keep those that exist on disk
        self.pairs = []
        with open(self.split_csv, mode="r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row or len(row) < 2:
                    continue
                s1_name = row[0].strip()
                lbl_name = row[1].strip()
                s2_name = s1_name.replace("_S1Hand.tif", "_S2Hand.tif")

                s2_path = self.s2_dir / s2_name
                lbl_path = self.label_dir / lbl_name
                if s2_path.exists() and lbl_path.exists():
                    self.pairs.append((s2_path, lbl_path, s2_name))

        if len(self.pairs) == 0:
            raise RuntimeError(
                f"No downloaded file pairs found for split '{split}' in {self.data_dir}. "
                f"Please run download_sen1floods11.py first."
            )

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        s2_path, lbl_path, chip_name = self.pairs[idx]

        # 1. Read S2 Multi-band GeoTIFF
        s2_raw = tifffile.imread(str(s2_path))
        # Ensure channel dimension is first: (C, H, W)
        if s2_raw.ndim == 3 and s2_raw.shape[2] == 13:
            s2_raw = np.transpose(s2_raw, (2, 0, 1))
        elif s2_raw.ndim == 2:
            s2_raw = np.expand_dims(s2_raw, 0)

        # Select target bands
        if self.bands is not None:
            s2_selected = s2_raw[self.bands, :, :]
        else:
            s2_selected = s2_raw

        # Transpose to (H, W, C) for Albumentations and normalization
        img_hwc = np.transpose(s2_selected, (1, 2, 0)).astype(np.float32)

        # Scale reflectance: 0-10,000 -> [0.0, 1.0]
        img_hwc = img_hwc / self.scale_factor
        img_hwc = np.clip(img_hwc, 0.0, 1.0)

        # 2. Read Ground Truth Mask
        lbl_raw = tifffile.imread(str(lbl_path))
        if lbl_raw.ndim == 3:
            lbl_raw = lbl_raw[0]
        mask_hw = lbl_raw.astype(np.float32)  # values: 1, 0, -1

        # 3. Apply Albumentations transform if provided
        if self.transform is not None:
            augmented = self.transform(image=img_hwc, mask=mask_hw)
            img_hwc = augmented["image"]
            mask_hw = augmented["mask"]

        # 4. Convert to PyTorch Tensors
        # Image: (C, H, W)
        img_tensor = torch.from_numpy(np.transpose(img_hwc, (2, 0, 1))).float()
        # Mask: (1, H, W)
        mask_tensor = torch.from_numpy(mask_hw).unsqueeze(0).float()

        if self.return_metadata:
            return {
                "image": img_tensor,
                "mask": mask_tensor,
                "chip_name": chip_name,
                "event": chip_name.split("_")[0],
                "chip_id": chip_name.split("_")[1]
            }
        return img_tensor, mask_tensor


def get_dataloader(
    data_dir: str or Path,
    split: str = "test",
    bands: List[int] = DEFAULT_BANDS,
    batch_size: int = 4,
    shuffle: bool = False,
    num_workers: int = 0
) -> DataLoader:
    """Create a DataLoader for the requested split."""
    transform = get_train_transforms() if split == "train" else get_valid_transforms()
    dataset = Sen1Floods11Dataset(
        data_dir=data_dir,
        split=split,
        bands=bands,
        transform=transform,
        return_metadata=True
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )


if __name__ == "__main__":
    # Test DataLoader execution
    script_dir = Path(__file__).resolve().parent
    data_path = script_dir.parent / "data"
    loader = get_dataloader(data_path, split="test", batch_size=4)
    print(f"Loaded Sen1Floods11 test split: {len(loader.dataset)} sample pairs.")
    batch = next(iter(loader))
    print(f"Batch image shape: {batch['image'].shape} | dtype: {batch['image'].dtype}")
    print(f"Batch mask shape: {batch['mask'].shape} | dtype: {batch['mask'].dtype}")
    print(f"Batch sample chip: {batch['chip_name'][0]}")
    print("Dataset verification successful!")
