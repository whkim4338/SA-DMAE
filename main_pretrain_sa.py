"""
SA-DMAE pre-training script.

Differences from original main_pretrain.py:
  - Uses MedicalSliceDataset (NIfTI) or PreprocessedDataset (.pt files from Colab)
  - Model: SpatialAxialDMAE  (models_sa_dmae.py)
  - Input shape: (B, n_slices, C, H, W)  instead of (B, C, H, W)
  - Train / Val split (default 90/10) with per-epoch val loss
  - Rich console output: step progress + epoch summary + ETA
  - CPU-friendly: AMP disabled on CPU
  - timm 1.x compatible optimizer factory

Usage (preprocessed .pt files, BraTS + UCSF):
    python main_pretrain_sa.py \\
        --data_path  ./pt_slices/brats \\
        --ucsf_path  ./pt_slices/ucsf \\
        --preprocessed \\
        --device cuda \\
        --epochs 200 --batch_size 16

Usage (NIfTI, single dataset):
    python main_pretrain_sa.py --data_path /path/to/BraTS2021
"""

import argparse
import datetime
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler, random_split
from torch.utils.tensorboard import SummaryWriter

import timm.optim as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

import models_sa_dmae
from dataset_medical import get_brats2021_dataset, get_combined_dataset


# ── Preprocessed dataset (.pt files saved by Colab) ─────────────────────────

class PreprocessedDataset(Dataset):
    """Loads pre-extracted (n_slices, C, H, W) tensors saved as .pt files."""

    def __init__(self, data_path: str):
        self.files = sorted(Path(data_path).glob("*.pt"))
        if len(self.files) == 0:
            raise RuntimeError(f"No .pt files found in {data_path}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        x = torch.load(self.files[idx], map_location="cpu")
        return x, self.files[idx].stem


# ── Console helpers ───────────────────────────────────────────────────────────

def _fmt_time(seconds: float) -> str:
    return str(datetime.timedelta(seconds=int(seconds)))


def _print_step(epoch, total_epochs, step, total_steps, loss, lr):
    """In-place step progress bar printed to the same line."""
    bar_len = 30
    filled  = int(bar_len * (step + 1) / total_steps)
    bar     = "█" * filled + "░" * (bar_len - filled)
    pct     = 100 * (step + 1) / total_steps
    print(
        f"\rEpoch [{epoch+1:>3}/{total_epochs}] "
        f"|{bar}| {pct:5.1f}%  "
        f"step {step+1:>4}/{total_steps}  "
        f"loss: {loss:.4f}  lr: {lr:.2e}",
        end="", flush=True,
    )


def _print_epoch_summary(epoch, total_epochs, train_loss, val_loss,
                         lr, epoch_time, elapsed, eta):
    """Clear summary block printed after each epoch."""
    sep = "─" * 65
    print(f"\r{sep}")
    val_str = f"{val_loss:.4f}" if val_loss is not None else "  N/A "
    print(
        f"  Epoch [{epoch+1:>3}/{total_epochs}]  "
        f"train_loss: {train_loss:.4f}  │  "
        f"val_loss: {val_str}  │  "
        f"lr: {lr:.2e}"
    )
    print(
        f"  epoch time: {_fmt_time(epoch_time)}  │  "
        f"elapsed: {_fmt_time(elapsed)}  │  "
        f"ETA: {_fmt_time(eta)}"
    )
    print(sep)


# ── Argument parser ───────────────────────────────────────────────────────────

def get_args_parser():
    parser = argparse.ArgumentParser("SA-DMAE pre-training", add_help=False)

    # Training
    parser.add_argument("--batch_size",    default=4,    type=int)
    parser.add_argument("--epochs",        default=200,  type=int)
    parser.add_argument("--accum_iter",    default=4,    type=int,
                        help="Gradient accumulation steps")

    # Model
    parser.add_argument("--model",         default="sa_dmae_vit_base_patch16", type=str)
    parser.add_argument("--input_size",    default=224,  type=int)
    parser.add_argument("--mask_ratio",    default=0.75, type=float)
    parser.add_argument("--norm_pix_loss", action="store_true")
    parser.set_defaults(norm_pix_loss=False)
    parser.add_argument("--sigma",         default=0.25, type=float,
                        help="Gaussian noise std applied to center slice")

    # SA-DMAE specific
    parser.add_argument("--n_slices",      default=3,    type=int)
    parser.add_argument("--axial_depth",   default=4,    type=int)

    # Optimizer
    parser.add_argument("--weight_decay",  default=0.05,  type=float)
    parser.add_argument("--lr",            default=None,  type=float)
    parser.add_argument("--blr",           default=1e-3,  type=float,
                        help="Base lr: actual_lr = blr * eff_batch / 256")
    parser.add_argument("--min_lr",        default=0.0,   type=float)
    parser.add_argument("--warmup_epochs", default=20,    type=int)

    # Dataset
    parser.add_argument("--data_path",    default="",  type=str,
                        help="BraTS root dir (NIfTI) or .pt dir (--preprocessed)")
    parser.add_argument("--ucsf_path",    default="",  type=str,
                        help="UCSF-PDGM .pt dir — merged with data_path when --preprocessed")
    parser.add_argument("--preprocessed", action="store_true",
                        help="Load pre-extracted .pt slices instead of raw NIfTI")
    parser.add_argument("--val_ratio",    default=0.1,  type=float,
                        help="Fraction of data held out for validation (default: 0.1)")
    parser.add_argument("--modalities",   default=["t1ce", "t2", "flair"], nargs="+")
    parser.add_argument("--num_workers",  default=4,    type=int,
                        help="DataLoader workers (use 0 on Windows if errors occur)")
    parser.add_argument("--pin_mem",      action="store_true")
    parser.set_defaults(pin_mem=False)

    # Logging / checkpointing
    parser.add_argument("--output_dir",   default="./output_sa", type=str)
    parser.add_argument("--log_dir",      default="./output_sa", type=str)
    parser.add_argument("--save_every",   default=20,   type=int,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--resume",       default="",   type=str)
    parser.add_argument("--start_epoch",  default=0,    type=int)
    parser.add_argument("--seed",         default=0,    type=int)
    parser.add_argument("--device",       default="cuda", type=str)

    # Distributed (optional)
    parser.add_argument("--world_size",   default=1,    type=int)
    parser.add_argument("--local_rank",   default=-1,   type=int)
    parser.add_argument("--dist_on_itp",  action="store_true")
    parser.add_argument("--dist_url",     default="env://", type=str)

    return parser


# ── Learning rate schedule ────────────────────────────────────────────────────

def _adjust_learning_rate(optimizer, progress, args):
    """Cosine decay with linear warmup. progress = epoch + step/steps_per_epoch."""
    if progress < args.warmup_epochs:
        lr = args.lr * progress / args.warmup_epochs
    else:
        t  = (progress - args.warmup_epochs) / (args.epochs - args.warmup_epochs)
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * (1.0 + math.cos(math.pi * t))
    for pg in optimizer.param_groups:
        pg["lr"] = lr * pg.get("lr_scale", 1.0)


# ── Train one epoch ───────────────────────────────────────────────────────────

def train_one_epoch(model, data_loader, optimizer, device, epoch,
                    loss_scaler, log_writer, args):
    model.train()
    use_amp     = device.type == "cuda"
    total_steps = len(data_loader)
    running_loss = 0.0
    optimizer.zero_grad()

    for step, (samples, _) in enumerate(data_loader):
        # LR schedule (per-iteration)
        if step % args.accum_iter == 0:
            _adjust_learning_rate(optimizer, step / total_steps + epoch, args)

        samples = samples.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            loss, _, _ = model(samples, mask_ratio=args.mask_ratio)

        if not torch.isfinite(loss):
            print(f"\nLoss is {loss.item()}, stopping training")
            raise RuntimeError("Non-finite loss")

        loss_scaled = loss / args.accum_iter
        loss_scaler(
            loss_scaled, optimizer,
            parameters=model.parameters(),
            update_grad=((step + 1) % args.accum_iter == 0),
        )
        if (step + 1) % args.accum_iter == 0:
            optimizer.zero_grad()

        if device.type == "cuda":
            torch.cuda.synchronize()

        cur_loss = loss.item()
        running_loss += cur_loss
        lr = optimizer.param_groups[0]["lr"]

        # ── 콘솔: step 진행률 (같은 줄 덮어쓰기) ─────────────────────────
        _print_step(epoch, args.epochs, step, total_steps, cur_loss, lr)

        # TensorBoard
        if log_writer is not None and (step + 1) % args.accum_iter == 0:
            epoch_1000x = int((step / total_steps + epoch) * 1000)
            log_writer.add_scalar("train/loss_step", cur_loss, epoch_1000x)
            log_writer.add_scalar("train/lr",        lr,       epoch_1000x)

    avg_loss = running_loss / total_steps
    return {"loss": avg_loss, "lr": lr}


# ── Validation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, data_loader, device, args):
    model.eval()
    use_amp    = device.type == "cuda"
    total_loss = 0.0

    for samples, _ in data_loader:
        samples = samples.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            loss, _, _ = model(samples, mask_ratio=args.mask_ratio)
        total_loss += loss.item()

    return total_loss / len(data_loader)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    misc.init_distributed_mode(args)

    device = torch.device(
        args.device if (torch.cuda.is_available() or args.device == "cpu") else "cpu"
    )
    if device.type == "cpu":
        print("[!] CUDA not available — running on CPU (training will be slow)")

    torch.manual_seed(args.seed + misc.get_rank())
    np.random.seed(args.seed + misc.get_rank())
    if device.type == "cuda":
        cudnn.benchmark = True

    # ── Dataset 로드 ─────────────────────────────────────────────────────────
    if args.preprocessed:
        dataset_brats = PreprocessedDataset(args.data_path)
        if args.ucsf_path:
            from torch.utils.data import ConcatDataset
            dataset_ucsf = PreprocessedDataset(args.ucsf_path)
            full_dataset = ConcatDataset([dataset_brats, dataset_ucsf])
            print(f"[Dataset] BraTS: {len(dataset_brats)}  UCSF: {len(dataset_ucsf)}  "
                  f"Total: {len(full_dataset)}")
        else:
            full_dataset = dataset_brats
            print(f"[Dataset] {len(full_dataset)} samples from {args.data_path}")
    else:
        full_dataset = get_brats2021_dataset(
            args.data_path,
            modalities=args.modalities, n_slices=args.n_slices,
            target_size=(args.input_size, args.input_size),
        )
        print(f"[Dataset] BraTS 2021 NIfTI: {len(full_dataset)} samples")

    # ── Train / Val split ─────────────────────────────────────────────────────
    n_total = len(full_dataset)
    n_val   = max(1, int(n_total * args.val_ratio))
    n_train = n_total - n_val
    dataset_train, dataset_val = random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )
    print(f"[Split]   Train: {n_train}  Val: {n_val}  (val_ratio={args.val_ratio})")

    loader_train = DataLoader(
        dataset_train,
        sampler=RandomSampler(dataset_train),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    loader_val = DataLoader(
        dataset_val,
        sampler=SequentialSampler(dataset_val),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    # ── 모델 ─────────────────────────────────────────────────────────────────
    model = models_sa_dmae.__dict__[args.model](
        norm_pix_loss=args.norm_pix_loss,
        sigma=args.sigma,
        n_slices=args.n_slices,
        axial_depth=args.axial_depth,
    )
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[Model]   {args.model}  ({n_params:.1f}M params)")

    # ── Optimizer ─────────────────────────────────────────────────────────────
    eff_batch = args.batch_size * args.accum_iter * misc.get_world_size()
    if args.lr is None:
        args.lr = args.blr * eff_batch / 256
    print(f"[Optim]   eff_batch={eff_batch}  lr={args.lr:.2e}  "
          f"warmup={args.warmup_epochs} epochs")

    try:
        param_groups = optim_factory.param_groups_weight_decay(model, args.weight_decay)
    except AttributeError:
        param_groups = optim_factory.add_weight_decay(model, args.weight_decay)

    optimizer    = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    loss_scaler  = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model,
                    optimizer=optimizer, loss_scaler=loss_scaler)

    # ── TensorBoard ───────────────────────────────────────────────────────────
    log_writer = None
    if args.log_dir and misc.is_main_process():
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)

    # ── 학습 시작 헤더 ────────────────────────────────────────────────────────
    print("=" * 65)
    print(f"  SA-DMAE Pre-training")
    print(f"  Epochs: {args.epochs}  |  Device: {device}  |  "
          f"Batch: {args.batch_size} x accum {args.accum_iter}")
    print(f"  sigma={args.sigma}  mask_ratio={args.mask_ratio}  "
          f"axial_depth={args.axial_depth}")
    print("=" * 65)

    start_time   = time.time()
    best_val     = float("inf")

    for epoch in range(args.start_epoch, args.epochs):
        epoch_start = time.time()

        # Train
        train_stats = train_one_epoch(
            model, loader_train, optimizer, device, epoch,
            loss_scaler, log_writer, args,
        )

        # Val
        val_loss = evaluate(model, loader_val, device, args)

        epoch_time = time.time() - epoch_start
        elapsed    = time.time() - start_time
        remaining  = elapsed / (epoch - args.start_epoch + 1) * (args.epochs - epoch - 1)

        # ── 콘솔: epoch 요약 출력 ─────────────────────────────────────────
        _print_epoch_summary(
            epoch, args.epochs,
            train_stats["loss"], val_loss,
            train_stats["lr"],
            epoch_time, elapsed, remaining,
        )

        # ── Best val loss 추적 ────────────────────────────────────────────
        if val_loss < best_val:
            best_val = val_loss
            if args.output_dir:
                misc.save_model(
                    args=args, model=model, model_without_ddp=model,
                    optimizer=optimizer, loss_scaler=loss_scaler,
                    epoch=epoch, suffix="best",
                )
            print(f"  [Best] val_loss improved → {best_val:.4f}  (checkpoint saved)")

        # ── Periodic checkpoint ───────────────────────────────────────────
        if args.output_dir and (epoch % args.save_every == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model,
                optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch,
            )

        # ── TensorBoard: epoch 단위 ───────────────────────────────────────
        if log_writer is not None:
            log_writer.add_scalar("train/loss_epoch", train_stats["loss"], epoch)
            log_writer.add_scalar("val/loss_epoch",   val_loss,            epoch)
            log_writer.flush()

        # ── log.txt ───────────────────────────────────────────────────────
        log_stats = {
            "epoch":      epoch,
            "train_loss": train_stats["loss"],
            "val_loss":   val_loss,
            "lr":         train_stats["lr"],
        }
        if args.output_dir and misc.is_main_process():
            with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total = _fmt_time(time.time() - start_time)
    print(f"\nTraining complete. Total time: {total}  |  Best val loss: {best_val:.4f}")


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
