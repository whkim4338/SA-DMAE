"""
BraTS 2021 segmentation dataset for SA-DMAE fine-tuning.

하이브리드 방식:
  - 이미지: 전처리된 .pt 파일에서 로드 (빠름)
  - seg 마스크: seg NIfTI에서 center slice만 추출

Returns (x_slices, seg_masks) where:
  x_slices  : (n_slices, 3, 224, 224)  — .pt에서 로드 (이미 정규화됨)
  seg_masks : (3, 224, 224)            — WT / TC / ET binary float masks

BraTS label convention:
  1 = NCR (Necrotic core)
  2 = ED  (Peritumoral edema)
  4 = ET  (Enhancing tumor)

  WT (Whole Tumor)   = 1 + 2 + 4
  TC (Tumor Core)    = 1 + 4
  ET (Enhancing)     = 4 only
"""

import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from dataset_medical import find_tumor_center_z, load_volume


def make_brats_masks(seg_slice: np.ndarray) -> torch.Tensor:
    """BraTS seg 슬라이스 → WT/TC/ET 3채널 binary mask.

    seg_slice: (H, W) numpy array, BraTS labels {0, 1, 2, 4}
    Returns  : (3, H, W) float32 tensor  [WT, TC, ET]
    """
    wt = (seg_slice > 0).astype(np.float32)
    tc = ((seg_slice == 1) | (seg_slice == 4)).astype(np.float32)
    et = (seg_slice == 4).astype(np.float32)
    return torch.from_numpy(np.stack([wt, tc, et], axis=0))


class BraTSSegDataset(Dataset):
    """BraTS 2021 segmentation fine-tuning dataset.

    이미지는 전처리된 .pt 파일에서 로드하고,
    seg 마스크만 NIfTI에서 center slice를 추출해 사용.
    → NIfTI 4개 로드 대비 ~4배 빠름.

    Args:
        pt_dir    : .pt 파일이 있는 디렉터리 (colab_preprocess.ipynb로 생성)
        nifti_dir : BraTS2021 NIfTI 원본 디렉터리 (seg 파일 추출용)
        img_size  : seg mask resize 해상도
        split     : 'train' or 'val'
        val_ratio : validation 비율
        seed      : 재현성 시드
    """

    def __init__(
        self,
        pt_dir: str,
        nifti_dir: str,
        img_size: int = 224,
        split: str = "train",
        val_ratio: float = 0.2,
        seed: int = 42,
    ):
        assert split in ("train", "val")
        self.img_size = img_size

        samples = self._scan(Path(pt_dir), Path(nifti_dir))
        rng = random.Random(seed)
        rng.shuffle(samples)
        n_val = int(len(samples) * val_ratio)
        self.samples = samples[:n_val] if split == "val" else samples[n_val:]

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No matched samples found.\n"
                f"  pt_dir   : {pt_dir}\n"
                f"  nifti_dir: {nifti_dir}\n"
                f"Check that .pt filenames match BraTS case IDs."
            )

    def _scan(self, pt_dir: Path, nifti_dir: Path) -> List[dict]:
        """BraTS .pt 파일과 seg NIfTI를 매칭."""
        samples = []
        for pt_path in sorted(pt_dir.glob("*.pt")):
            case_id  = pt_path.stem               # e.g. BraTS2021_00000
            case_dir = nifti_dir / case_id
            seg_path = case_dir / f"{case_id}_seg.nii.gz"
            if seg_path.exists():
                samples.append({"pt": pt_path, "seg": seg_path, "case_id": case_id})
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        info = self.samples[idx]

        # ── 이미지: .pt에서 로드 (n_slices, 3, 224, 224) ────────────────────
        x_slices = torch.load(info["pt"], map_location="cpu", weights_only=True)
        # .pt shape: (n_slices, C, H, W), values in [0, 1]

        # ── seg 마스크: NIfTI center slice만 추출 ────────────────────────────
        seg_vol  = load_volume(str(info["seg"]))       # (H, W, D)
        center_z = find_tumor_center_z(seg_vol)
        seg_slice = seg_vol[:, :, center_z]            # (H, W)

        seg_mask = make_brats_masks(seg_slice)         # (3, H, W)
        seg_mask = F.interpolate(
            seg_mask.unsqueeze(0),
            size=(self.img_size, self.img_size),
            mode="nearest",
        ).squeeze(0)                                   # (3, H, W)

        return x_slices, seg_mask
