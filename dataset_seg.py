"""
BraTS 2021 segmentation dataset for SA-DMAE fine-tuning.

Returns (x_slices, seg_masks) where:
  x_slices  : (n_slices, 3, 224, 224)  — T1ce/T2/FLAIR, values in [0, 1]
  seg_masks : (3, 224, 224)            — WT / TC / ET binary float masks

BraTS label convention:
  1 = NCR (Necrotic core)
  2 = ED  (Peritumoral edema)
  4 = ET  (Enhancing tumor)

  WT (Whole Tumor)   = 1 + 2 + 4  (all non-zero)
  TC (Tumor Core)    = 1 + 4
  ET (Enhancing)     = 4 only
"""

import random
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from dataset_medical import (
    load_volume,
    normalize_volume,
    find_tumor_center_z,
    extract_slice_stack,
)


def make_brats_masks(seg_slice: np.ndarray) -> torch.Tensor:
    """Convert a 2-D BraTS seg slice to 3-channel binary mask tensor.

    seg_slice: (H, W) numpy array with BraTS labels {0, 1, 2, 4}
    Returns  : (3, H, W) float32 tensor  [WT, TC, ET]
    """
    wt = (seg_slice > 0).astype(np.float32)
    tc = ((seg_slice == 1) | (seg_slice == 4)).astype(np.float32)
    et = (seg_slice == 4).astype(np.float32)
    return torch.from_numpy(np.stack([wt, tc, et], axis=0))   # (3, H, W)


class BraTSSegDataset(Dataset):
    """BraTS 2021 dataset for supervised segmentation fine-tuning.

    Args:
        root      : BraTS2021 root directory (contains case sub-folders)
        n_slices  : number of consecutive axial slices per sample (must be odd)
        img_size  : spatial resolution to resize to
        split     : 'train' or 'val'
        val_ratio : fraction reserved for validation
        seed      : random seed for reproducible split
    """

    def __init__(
        self,
        root: str,
        n_slices: int = 3,
        img_size: int = 224,
        split: str = "train",
        val_ratio: float = 0.2,
        seed: int = 42,
    ):
        assert split in ("train", "val"), "split must be 'train' or 'val'"
        self.root     = Path(root)
        self.n_slices = n_slices
        self.img_size = img_size

        cases = self._scan_cases()
        rng = random.Random(seed)
        rng.shuffle(cases)
        n_val = int(len(cases) * val_ratio)
        self.cases = cases[:n_val] if split == "val" else cases[n_val:]

        if len(self.cases) == 0:
            raise RuntimeError(f"No BraTS cases found under {root}")

    def _scan_cases(self) -> List[dict]:
        cases = []
        for case_dir in sorted(self.root.iterdir()):
            if not case_dir.is_dir():
                continue
            cid = case_dir.name
            t1ce = case_dir / f"{cid}_t1ce.nii.gz"
            t2   = case_dir / f"{cid}_t2.nii.gz"
            fl   = case_dir / f"{cid}_flair.nii.gz"
            seg  = case_dir / f"{cid}_seg.nii.gz"
            if all(p.exists() for p in [t1ce, t2, fl, seg]):
                cases.append({
                    "case_id": cid,
                    "t1ce": str(t1ce),
                    "t2":   str(t2),
                    "flair": str(fl),
                    "seg":  str(seg),
                })
        return cases

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        info = self.cases[idx]

        # Load & normalise modality volumes
        vols = [
            normalize_volume(load_volume(info["t1ce"])),
            normalize_volume(load_volume(info["t2"])),
            normalize_volume(load_volume(info["flair"])),
        ]

        # Load seg volume for center Z + mask extraction
        seg_vol = load_volume(info["seg"])          # (H, W, D)  float32

        # Tumor center Z
        center_z = find_tumor_center_z(seg_vol)

        # Image slices: (n_slices, 3, H, W)
        x_slices = extract_slice_stack(vols, center_z, self.n_slices, (self.img_size, self.img_size))

        # Segmentation mask for center slice only
        seg_center = seg_vol[:, :, center_z]        # (H, W)
        seg_mask   = make_brats_masks(seg_center)   # (3, H, W)

        # Resize seg mask to img_size
        seg_mask = F.interpolate(
            seg_mask.unsqueeze(0),                  # (1, 3, H, W)
            size=(self.img_size, self.img_size),
            mode="nearest",
        ).squeeze(0)                                # (3, H, W)

        return x_slices, seg_mask
