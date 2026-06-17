"""
Segmentation fine-tuning script for SA-DMAE / DMAE.

Loads a pre-trained SA-DMAE checkpoint and fine-tunes on BraTS 2021
tumor segmentation (WT / TC / ET).

Example:
    # SA-DMAE encoder
    python main_finetune_seg.py \
        --data_path /path/to/BraTS2021 \
        --resume    /path/to/sa_dmae_best.pth \
        --n_slices  3 --axial_depth 4 \
        --output_dir ./output_seg_sa

    # DMAE encoder (baseline)
    python main_finetune_seg.py \
        --data_path /path/to/BraTS2021 \
        --resume    /path/to/dmae_best.pth \
        --n_slices  1 --axial_depth 0 \
        --output_dir ./output_seg_dmae
"""

import argparse
import json
import math
import os
import sys
import time
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import models_sa_dmae
from dataset_seg import BraTSSegDataset
from models_seg import SADMAESegmentation, dice_score, seg_loss


# ── Argument Parsing ─────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser("SA-DMAE Segmentation Fine-tuning")

    # Data
    p.add_argument("--pt_dir",      required=True,  help="전처리된 BraTS .pt 파일 디렉터리")
    p.add_argument("--nifti_dir",   required=True,  help="BraTS2021 NIfTI 원본 디렉터리 (seg 추출용)")
    p.add_argument("--n_slices",    type=int, default=3)
    p.add_argument("--img_size",    type=int, default=224)
    p.add_argument("--val_ratio",   type=float, default=0.2)
    p.add_argument("--num_workers", type=int, default=2)

    # Model
    p.add_argument("--model",       default="sa_dmae_vit_base_patch16")
    p.add_argument("--axial_depth", type=int, default=2)
    p.add_argument("--sigma",       type=float, default=0.25)
    p.add_argument("--num_classes", type=int, default=3, help="WT/TC/ET")
    p.add_argument("--freeze_encoder", action="store_true", default=True,
                   help="Freeze encoder weights, train decoder only (권장)")
    p.add_argument("--no_freeze_encoder", dest="freeze_encoder", action="store_false",
                   help="Encoder까지 fine-tuning (비권장: pre-trained 망가질 수 있음)")
    p.add_argument("--dice_only", action="store_true", default=True,
                   help="Dice loss만 사용 (기본값, class imbalance 강건)")
    p.add_argument("--no_dice_only", dest="dice_only", action="store_false",
                   help="Dice + weighted BCE 혼합 loss 사용")

    # Training
    p.add_argument("--epochs",      type=int, default=50)
    p.add_argument("--batch_size",  type=int, default=8)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--min_lr",      type=float, default=1e-6)
    p.add_argument("--weight_decay",type=float, default=0.05)
    p.add_argument("--patience",    type=int, default=15,
                   help="Early stopping patience (0=disabled)")
    p.add_argument("--warmup_epochs", type=int, default=5)

    # Checkpoint / output
    p.add_argument("--resume",      default="", help="Pre-trained encoder checkpoint")
    p.add_argument("--resume_seg",  default="", help="Resume seg fine-tuning from checkpoint")
    p.add_argument("--output_dir",  default="./output_seg")
    p.add_argument("--log_dir",     default="")
    p.add_argument("--save_every",  type=int, default=10)
    p.add_argument("--device",      default="cuda")

    return p.parse_args()


# ── LR Schedule ──────────────────────────────────────────────────────────────

def cosine_lr(optimizer, epoch, warmup_epochs, total_epochs, lr, min_lr):
    if epoch < warmup_epochs:
        cur_lr = lr * epoch / max(1, warmup_epochs)
    else:
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        cur_lr = min_lr + 0.5 * (lr - min_lr) * (1.0 + math.cos(math.pi * progress))
    for pg in optimizer.param_groups:
        pg["lr"] = cur_lr
    return cur_lr


# ── Train / Eval loops ───────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, device, epoch, args):
    model.train()
    total_loss = 0.0
    n = 0
    start = time.time()

    for step, (x_slices, seg_masks) in enumerate(loader):
        x_slices  = x_slices.to(device, non_blocking=True)
        seg_masks = seg_masks.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=(device == "cuda")):
            logits = model(x_slices)
            loss   = seg_loss(logits, seg_masks, dice_only=args.dice_only)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        n += 1

        elapsed = time.time() - start
        eta     = elapsed / (step + 1) * (len(loader) - step - 1)
        print(
            f"\r  [Train] step {step+1}/{len(loader)} | "
            f"loss {loss.item():.4f} | "
            f"ETA {eta:.0f}s   ",
            end="", flush=True,
        )

    print()
    return total_loss / n


@torch.no_grad()
def evaluate(model, loader, device, dice_only=True):
    model.eval()
    total_loss = 0.0
    dice_sum   = torch.zeros(3, device=device)
    n = 0

    for x_slices, seg_masks in loader:
        x_slices  = x_slices.to(device, non_blocking=True)
        seg_masks = seg_masks.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=(device == "cuda")):
            logits = model(x_slices)
            loss   = seg_loss(logits, seg_masks, dice_only=dice_only)

        total_loss += loss.item()
        dice_sum   += dice_score(logits, seg_masks)
        n += 1

    return total_loss / n, (dice_sum / n).cpu()   # (scalar, (3,))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = get_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    log_dir = args.log_dir or args.output_dir
    writer  = SummaryWriter(log_dir=log_dir)

    # ── Dataset ──────────────────────────────────────────────────────────────
    train_ds = BraTSSegDataset(args.pt_dir, args.nifti_dir,
                               img_size=args.img_size, split="train",
                               val_ratio=args.val_ratio)
    val_ds   = BraTSSegDataset(args.pt_dir, args.nifti_dir,
                               img_size=args.img_size, split="val",
                               val_ratio=args.val_ratio)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=args.num_workers,
                              pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers,
                              pin_memory=True)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    # ── Encoder ──────────────────────────────────────────────────────────────
    model_fn = getattr(models_sa_dmae, args.model)
    encoder  = model_fn(
        n_slices=args.n_slices,
        axial_depth=args.axial_depth,
        sigma=args.sigma,
    )

    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        state = ckpt.get("model", ckpt)
        msg = encoder.load_state_dict(state, strict=False)
        print(f"Loaded encoder from {args.resume}")
        print(f"  missing : {len(msg.missing_keys)} | unexpected: {len(msg.unexpected_keys)}")
    else:
        print("[!] No pre-trained checkpoint — training from random init")

    # ── Segmentation Model ───────────────────────────────────────────────────
    model = SADMAESegmentation(
        encoder=encoder,
        num_classes=args.num_classes,
        freeze_encoder=args.freeze_encoder,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {n_params/1e6:.1f}M")

    # ── Optimiser ────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay,
    )

    # ── Resume seg fine-tuning ───────────────────────────────────────────────
    start_epoch  = 0
    best_dice    = 0.0
    patience_cnt = 0

    if args.resume_seg:
        seg_ckpt = torch.load(args.resume_seg, map_location="cpu", weights_only=False)
        model.load_state_dict(seg_ckpt["model"])
        optimizer.load_state_dict(seg_ckpt["optimizer"])
        start_epoch  = seg_ckpt["epoch"] + 1
        best_dice    = seg_ckpt.get("best_dice", 0.0)
        patience_cnt = seg_ckpt.get("patience_cnt", 0)
        print(f"Resumed from {args.resume_seg} (epoch {start_epoch}, best_dice {best_dice:.4f})")

    # ── Training Loop ────────────────────────────────────────────────────────
    log_path = Path(args.output_dir) / "log_seg.txt"
    start_total = time.time()

    print(f"\nDevice: {device}")
    print(f"Epochs: {args.epochs} | Batch: {args.batch_size} | LR: {args.lr}")
    print("=" * 65)

    for epoch in range(start_epoch, args.epochs):
        cur_lr = cosine_lr(optimizer, epoch, args.warmup_epochs,
                           args.epochs, args.lr, args.min_lr)

        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch, args)
        val_loss, val_dice = evaluate(model, val_loader, device, dice_only=args.dice_only)
        mean_dice = val_dice.mean().item()
        epoch_time = time.time() - t0

        # Logging
        writer.add_scalar("train/loss",   train_loss, epoch)
        writer.add_scalar("val/loss",     val_loss,   epoch)
        writer.add_scalar("val/dice_WT",  val_dice[0].item(), epoch)
        writer.add_scalar("val/dice_TC",  val_dice[1].item(), epoch)
        writer.add_scalar("val/dice_ET",  val_dice[2].item(), epoch)
        writer.add_scalar("val/dice_mean",mean_dice,  epoch)
        writer.add_scalar("train/lr",     cur_lr,     epoch)

        elapsed = time.time() - start_total
        eta     = elapsed / (epoch + 1) * (args.epochs - epoch - 1)
        print(
            f"Epoch {epoch+1:3d}/{args.epochs} | "
            f"train {train_loss:.4f} | val {val_loss:.4f} | "
            f"Dice WT={val_dice[0]:.4f} TC={val_dice[1]:.4f} ET={val_dice[2]:.4f} "
            f"mean={mean_dice:.4f} | "
            f"{epoch_time:.0f}s | ETA {eta/60:.1f}min"
        )

        # Save log
        log = dict(epoch=epoch, train_loss=train_loss, val_loss=val_loss,
                   dice_wt=val_dice[0].item(), dice_tc=val_dice[1].item(),
                   dice_et=val_dice[2].item(), dice_mean=mean_dice, lr=cur_lr)
        with open(log_path, "a") as f:
            f.write(json.dumps(log) + "\n")

        # Best checkpoint (by mean Dice)
        if mean_dice > best_dice + 1e-4:
            best_dice    = mean_dice
            patience_cnt = 0
            ckpt_path = Path(args.output_dir) / "checkpoint-best.pth"
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "dice": best_dice, "args": args}, ckpt_path)
            print(f"  ✓ Best checkpoint saved (mean Dice={best_dice:.4f})")
        else:
            patience_cnt += 1

        # Periodic checkpoint (includes optimizer for resume)
        if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "best_dice": best_dice,
                "patience_cnt": patience_cnt,
                "args": args,
            }, Path(args.output_dir) / f"checkpoint-{epoch+1}.pth")

        # Early stopping
        if args.patience > 0 and patience_cnt >= args.patience:
            print(f"\nEarly stopping at epoch {epoch+1} (patience={args.patience})")
            break

    # Final checkpoint
    torch.save({"model": model.state_dict(), "epoch": epoch, "args": args},
               Path(args.output_dir) / "checkpoint-final.pth")

    print("=" * 65)
    print(f"Best mean Dice: {best_dice:.4f}")
    writer.close()


if __name__ == "__main__":
    main()
