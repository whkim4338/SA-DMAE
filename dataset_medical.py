"""
Medical image dataset for SA-DMAE (2.5D).

Supports BraTS 2021 and EDG datasets (both NIfTI format).
Each sample returns 3 consecutive axial slices centered on the tumor ROI,
stacked as (n_slices, C, H, W) where C = len(modalities).

BraTS 2021 directory layout expected:
    root/
    └── BraTS2021_00000/
        ├── BraTS2021_00000_flair.nii.gz
        ├── BraTS2021_00000_t1ce.nii.gz
        ├── BraTS2021_00000_t2.nii.gz
        └── BraTS2021_00000_seg.nii.gz   ← segmentation mask

EDG directory layout expected (same convention):
    root/
    └── EDG_00000/
        ├── EDG_00000_flair.nii.gz
        ├── EDG_00000_t1ce.nii.gz
        ├── EDG_00000_t2.nii.gz
        └── EDG_00000_seg.nii.gz
"""

import os
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F


# Modalities used as input channels (order = channel order in tensor)
DEFAULT_MODALITIES = ["t1ce", "t2", "flair"]


# ── Utility functions ────────────────────────────────────────────────────────

def load_volume(path: str) -> np.ndarray:
    """Load a NIfTI volume and return it as (H, W, D) float32 numpy array."""
    img = nib.load(path)
    return img.get_fdata(dtype=np.float32)


def normalize_volume(vol: np.ndarray, low_pct: float = 1.0, high_pct: float = 99.0) -> np.ndarray:
    """Percentile-based normalization to [0, 1], robust to MRI outliers.

    Only non-zero voxels are used to compute percentiles (avoids background bias).
    """
    nonzero = vol[vol > 0]
    if nonzero.size == 0:
        return vol
    lo = np.percentile(nonzero, low_pct)
    hi = np.percentile(nonzero, high_pct)
    if hi == lo:
        return np.zeros_like(vol)
    vol = np.clip(vol, lo, hi)
    return (vol - lo) / (hi - lo)


def find_tumor_center_z(seg: np.ndarray) -> int:
    """Return the axial (Z) index of the tumor center of mass.

    Uses the midpoint of the Z-range that contains any tumor label,
    which is robust and avoids a scipy dependency.

    seg: (H, W, D) segmentation volume — BraTS labels 1/2/4 or any nonzero.
    """
    tumor_mask = seg > 0                          # whole tumor
    z_with_tumor = np.where(tumor_mask.any(axis=(0, 1)))[0]
    if len(z_with_tumor) == 0:
        return seg.shape[2] // 2                  # fallback: volume midpoint
    return int(z_with_tumor[len(z_with_tumor) // 2])


def extract_slice_stack(
    volumes: List[np.ndarray],
    center_z: int,
    n_slices: int = 3,
    target_size: Tuple[int, int] = (224, 224),
) -> torch.Tensor:
    """Extract n_slices consecutive axial slices centred on center_z.

    Handles boundary cases by clamping (edge slices are repeated if needed).

    volumes   : list of (H, W, D) arrays, one per modality
    center_z  : axial index of the tumor center
    n_slices  : number of slices (must be odd)
    target_size: (H, W) to resize each slice to

    Returns: (n_slices, C, H, W) float32 tensor  — values in [0, 1]
    """
    assert n_slices % 2 == 1, "n_slices must be odd"
    half = n_slices // 2
    D = volumes[0].shape[2]

    # Clamp slice indices to valid range
    z_indices = [min(max(center_z + offset, 0), D - 1) for offset in range(-half, half + 1)]

    slices = []
    for z in z_indices:
        # Stack modalities as channels for this axial slice: (C, H, W)
        channels = [vol[:, :, z] for vol in volumes]           # each (H, W)
        slice_tensor = torch.from_numpy(np.stack(channels, axis=0))  # (C, H, W)

        # Resize to target_size
        slice_tensor = F.interpolate(
            slice_tensor.unsqueeze(0),                         # (1, C, H, W)
            size=target_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)                                           # (C, H, W)

        slices.append(slice_tensor)

    return torch.stack(slices, dim=0)                          # (n_slices, C, H, W)


# ── Dataset ──────────────────────────────────────────────────────────────────

class MedicalSliceDataset(Dataset):
    """PyTorch Dataset for 2.5D medical image slices centred on tumor ROI.

    Works for both BraTS 2021 and EDG datasets.

    Args:
        root        : path to dataset root directory
        modalities  : list of modality suffixes to stack as channels
        n_slices    : number of consecutive axial slices per sample
        target_size : spatial resolution each slice is resized to
        seg_suffix  : filename suffix for the ROI file (e.g. "seg" or "mask")
        mask_type   : "tumor" — use ROI file to find tumor center (default)
                      "brain" — ROI is a brain mask (not tumor), use volume midpoint
        transform   : optional transform applied to the final tensor
    """

    def __init__(
        self,
        root: str,
        modalities: List[str] = DEFAULT_MODALITIES,
        n_slices: int = 3,
        target_size: Tuple[int, int] = (224, 224),
        seg_suffix: str = "seg",
        mask_type: str = "tumor",
        transform=None,
    ):
        assert mask_type in ("tumor", "brain"), \
            "mask_type must be 'tumor' or 'brain'"

        self.root        = Path(root)
        self.modalities  = modalities
        self.n_slices    = n_slices
        self.target_size = target_size
        self.seg_suffix  = seg_suffix
        self.mask_type   = mask_type
        self.transform   = transform

        self.samples = self._scan_cases()
        if len(self.samples) == 0:
            raise RuntimeError(f"No valid cases found under {root}")

    def _scan_cases(self) -> List[dict]:
        """Walk root directory and collect cases that have all required files."""
        cases = []
        for case_dir in sorted(self.root.iterdir()):
            if not case_dir.is_dir():
                continue
            case_id = case_dir.name
            info = self._build_paths(case_dir, case_id)
            if info is not None:
                cases.append(info)
        return cases

    def _build_paths(self, case_dir: Path, case_id: str) -> Optional[dict]:
        """Return file paths for a case, or None if any required file is missing."""
        paths = {}

        # Modality files
        for mod in self.modalities:
            p = case_dir / f"{case_id}_{mod}.nii.gz"
            if not p.exists():
                return None
            paths[mod] = str(p)

        # ROI file: try configured suffix first, then fallback to the other
        fallback = "mask" if self.seg_suffix == "seg" else "seg"
        roi_path = case_dir / f"{case_id}_{self.seg_suffix}.nii.gz"
        if not roi_path.exists():
            roi_path = case_dir / f"{case_id}_{fallback}.nii.gz"
        paths["roi"]     = str(roi_path) if roi_path.exists() else None
        paths["case_id"] = case_id
        return paths

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        info = self.samples[idx]

        # Load and normalise each modality volume
        volumes = [
            normalize_volume(load_volume(info[mod]))
            for mod in self.modalities
        ]

        # Determine center_z
        if info["roi"] is None or self.mask_type == "brain":
            # No tumor ROI → use volume midpoint
            center_z = volumes[0].shape[2] // 2  # type: ignore[union-attr]
        else:
            roi = load_volume(info["roi"])
            center_z = find_tumor_center_z(roi)

        # Extract (n_slices, C, H, W) tensor
        x = extract_slice_stack(volumes, center_z, self.n_slices, self.target_size)

        if self.transform is not None:
            x = self.transform(x)

        return x, info["case_id"]


# ── Convenience constructors ─────────────────────────────────────────────────

def get_brats2021_dataset(root: str, **kwargs) -> MedicalSliceDataset:
    """BraTS 2021 dataset using T1ce + T2 + FLAIR."""
    return MedicalSliceDataset(root, modalities=DEFAULT_MODALITIES, **kwargs)


def get_edg_dataset(root: str, mask_type: str = "tumor", **kwargs) -> MedicalSliceDataset:
    """EDG dataset.

    Args:
        mask_type: "tumor" if mask file marks tumor region (use for center_z)
                   "brain" if mask file marks whole brain (fall back to midpoint)
    """
    return MedicalSliceDataset(
        root, modalities=DEFAULT_MODALITIES,
        seg_suffix="mask", mask_type=mask_type,
        **kwargs,
    )


def get_combined_dataset(brats_root: str, ucsf_root: str, **kwargs):
    """Concatenate BraTS 2021 and UCSF-PDGM into a single dataset."""
    from torch.utils.data import ConcatDataset
    brats = get_brats2021_dataset(brats_root, **kwargs)
    ucsf  = UCSFPDGMDataset(ucsf_root, **kwargs)
    return ConcatDataset([brats, ucsf])


# ── UCSF-PDGM Dataset ────────────────────────────────────────────────────────

class UCSFPDGMDataset(Dataset):
    """PyTorch Dataset for UCSF-PDGM (Kaggle version).

    File structure per case:
        {case_id}_nifti/
        ├── {case_id}_FLAIR_bias.nii/{case_id}_FLAIR_bias.nii
        ├── {case_id}_T1c_bias.nii/{case_id}_T1gad_bias.nii
        ├── {case_id}_T2.nii.gz                               ← optional
        └── {case_id}_tumor_segmentation.nii/
            └── {case_id}_tumor_segmentation.nii

    Channel order (same as BraTS): T1ce / T2 / FLAIR
    T2 missing → filled with zeros (same shape as FLAIR)

    Args:
        root        : path to dataset root (contains *_nifti folders)
        n_slices    : number of consecutive axial slices per sample
        target_size : spatial resolution each slice is resized to
        transform   : optional transform
    """

    def __init__(
        self,
        root: str,
        n_slices: int = 3,
        target_size: Tuple[int, int] = (224, 224),
        transform=None,
    ):
        self.root        = Path(root)
        self.n_slices    = n_slices
        self.target_size = target_size
        self.transform   = transform

        self.samples = self._scan_cases()
        if len(self.samples) == 0:
            raise RuntimeError(f"No valid UCSF-PDGM cases found under {root}")

    def _find_file(self, case_dir: Path, pattern: str) -> Optional[Path]:
        """Glob for first *file* matching pattern (skips directories)."""
        matches = [p for p in case_dir.rglob(pattern) if p.is_file()]
        return matches[0] if matches else None

    def _scan_cases(self) -> List[dict]:
        cases = []
        for case_dir in sorted(self.root.iterdir()):
            if not case_dir.is_dir():
                continue
            info = self._build_paths(case_dir)
            if info is not None:
                cases.append(info)
        return cases

    def _build_paths(self, case_dir: Path) -> Optional[dict]:
        """Return file paths; return None if any required file is missing.

        T2가 없는 케이스는 데이터 품질 일관성을 위해 스킵.
        (501개 중 178개가 T2 없음 → zeros 채움 시 35% 오염 우려)
        """
        flair = self._find_file(case_dir, "*FLAIR_bias.nii")
        t1ce  = self._find_file(case_dir, "*T1gad_bias.nii")
        t2    = self._find_file(case_dir, "*_T2.nii.gz")
        seg   = self._find_file(case_dir, "*tumor_segmentation.nii")

        # FLAIR, T1ce, T2, seg 모두 필수
        if any(f is None for f in [flair, t1ce, t2, seg]):
            return None

        return {
            "case_id" : case_dir.name,
            "flair"   : str(flair),
            "t1ce"    : str(t1ce),
            "t2"      : str(t2),
            "seg"     : str(seg),
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        info = self.samples[idx]

        # 로드 & 정규화 — 채널 순서: T1ce / T2 / FLAIR (BraTS와 동일)
        volumes = [
            normalize_volume(load_volume(info["t1ce"])),
            normalize_volume(load_volume(info["t2"])),
            normalize_volume(load_volume(info["flair"])),
        ]

        # 종양 center_z
        seg      = load_volume(info["seg"])
        center_z = find_tumor_center_z(seg)

        # (n_slices, 3, H, W) 텐서 추출
        x = extract_slice_stack(volumes, center_z, self.n_slices, self.target_size)

        if self.transform is not None:
            x = self.transform(x)

        return x, info["case_id"]
