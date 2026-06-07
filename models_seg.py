"""
Segmentation fine-tuning model for SA-DMAE.

SA-DMAE pre-trained encoder + lightweight segmentation decoder.
Supports BraTS 2021 WT/TC/ET 3-class binary segmentation.

Usage:
    encoder = sa_dmae_vit_base_patch16(n_slices=3, axial_depth=4)
    # load pre-trained weights ...
    model = SADMAESegmentation(encoder, num_classes=3)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SegDecoder(nn.Module):
    """Progressive upsampling decoder: patch tokens → segmentation mask.

    Input : (B, embed_dim, 14, 14)  — reshaped patch tokens
    Output: (B, num_classes, 224, 224)

    4× bilinear upsampling stages (14 → 28 → 56 → 112 → 224).
    """

    def __init__(self, embed_dim: int = 768, num_classes: int = 3):
        super().__init__()
        self.decoder = nn.Sequential(
            # 14 → 28
            nn.ConvTranspose2d(embed_dim, 256, kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.GELU(),
            # 28 → 56
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.GELU(),
            # 56 → 112
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.GELU(),
            # 112 → 224
            nn.ConvTranspose2d(64, num_classes, kernel_size=2, stride=2),
        )

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        """
        patch_tokens: (B, L, D)  where L = H_p * W_p  (e.g. 196 for 14×14)
        Returns     : (B, num_classes, H, W)
        """
        B, L, D = patch_tokens.shape
        h = w = int(L ** 0.5)
        x = patch_tokens.permute(0, 2, 1).reshape(B, D, h, w)
        return self.decoder(x)


class SADMAESegmentation(nn.Module):
    """SA-DMAE encoder + segmentation decoder.

    Args:
        encoder      : pre-trained SpatialAxialDMAE instance
        num_classes  : number of segmentation classes (default 3: WT/TC/ET)
        freeze_encoder: if True, encoder weights are frozen
    """

    CLASSES = ["WT", "TC", "ET"]  # Whole Tumor / Tumor Core / Enhancing Tumor

    def __init__(self, encoder, num_classes: int = 3, freeze_encoder: bool = False):
        super().__init__()
        self.encoder = encoder
        embed_dim = encoder.pos_embed.shape[-1]
        self.seg_decoder = SegDecoder(embed_dim=embed_dim, num_classes=num_classes)
        self.num_classes = num_classes

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def forward(self, x_slices: torch.Tensor) -> torch.Tensor:
        """
        x_slices: (B, n_slices, C, H, W)  values in [0, 1]
        Returns : (B, num_classes, H, W)   raw logits (no sigmoid)
        """
        latent = self.encoder.encode(x_slices)   # (B, L+1, D)
        patch_tokens = latent[:, 1:, :]           # remove CLS  → (B, 196, D)
        logits = self.seg_decoder(patch_tokens)   # (B, num_classes, 224, 224)
        return logits


# ── Loss functions ────────────────────────────────────────────────────────────

# BraTS 클래스 불균형 보정: 종양 픽셀(~5%)에 가중치 부여 [WT, TC, ET]
# 배경 대비 종양 픽셀 비율의 역수 — 실험적으로 튜닝된 값
POS_WEIGHT = torch.tensor([10.0, 15.0, 20.0])  # WT / TC / ET


def dice_loss(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-5) -> torch.Tensor:
    """Soft Dice loss.

    pred  : (B, C, H, W)  raw logits
    target: (B, C, H, W)  binary float masks
    smooth: 작은 값 사용 — 종양이 없는 슬라이스에서 trivial solution 방지
    """
    pred = pred.sigmoid()
    intersection = (pred * target).sum(dim=(-2, -1))
    union = pred.sum(dim=(-2, -1)) + target.sum(dim=(-2, -1))
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice.mean()


def seg_loss(logits: torch.Tensor, target: torch.Tensor,
             dice_only: bool = True) -> torch.Tensor:
    """Segmentation loss.

    dice_only=True (기본값): Dice loss만 사용.
      → 클래스 불균형에 강건, encoder freeze + fine-tuning에 적합.
    dice_only=False: Dice + weighted BCE 혼합.
    """
    d = dice_loss(logits, target)
    if dice_only:
        return d

    pw = POS_WEIGHT.to(logits.device).view(1, -1, 1, 1).expand_as(logits)
    b  = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pw)
    return 0.5 * d + 0.5 * b


# ── Metrics ───────────────────────────────────────────────────────────────────

@torch.no_grad()
def dice_score(logits: torch.Tensor, target: torch.Tensor,
               threshold: float = 0.5, smooth: float = 1e-5) -> torch.Tensor:
    """Per-class Dice score.

    Returns: (C,) tensor — one value per class
    """
    pred = (logits.sigmoid() > threshold).float()
    intersection = (pred * target).sum(dim=(-2, -1))
    union = pred.sum(dim=(-2, -1)) + target.sum(dim=(-2, -1))
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice.mean(dim=0)   # average over batch → (C,)
