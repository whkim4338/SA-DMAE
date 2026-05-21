"""
SA-DMAE pre-training script.

Differences from original main_pretrain.py:
  - Uses MedicalSliceDataset (NIfTI) or PreprocessedDataset (.pt files from Colab)
  - Model: SpatialAxialDMAE  (models_sa_dmae.py)
  - Input shape: (B, n_slices, C, H, W)  instead of (B, C, H, W)
  - CPU-friendly: AMP disabled on CPU, distributed optional
  - timm 1.x compatible optimizer factory

Usage (NIfTI, single dataset):
    python main_pretrain_sa.py --data_path /path/to/BraTS2021

Usage (preprocessed .pt files):
    python main_pretrain_sa.py --data_path /path/to/pt_files --preprocessed

Usage (combined BraTS + EDG):
    python main_pretrain_sa.py \\
        --data_path /path/to/BraTS2021 \\
        --edg_path  /path/to/EDG
"""

import argparse
import datetime
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torch.utils.tensorboard import SummaryWriter

import timm.optim as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

import models_sa_dmae
from dataset_medical import get_brats2021_dataset, get_edg_dataset, get_combined_dataset


# ── Preprocessed dataset (.pt files saved by Colab) ─────────────────────────

class PreprocessedDataset(Dataset):
    """Loads pre-extracted (n_slices, C, H, W) tensors saved as .pt files.

    Expected layout:
        data_path/
        ├── case_00000.pt
        ├── case_00001.pt
        └── ...
    """

    def __init__(self, data_path: str):
        self.files = sorted(Path(data_path).glob("*.pt"))
        if len(self.files) == 0:
            raise RuntimeError(f"No .pt files found in {data_path}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        x = torch.load(self.files[idx], map_location="cpu")
        return x, self.files[idx].stem


# ── Argument parser ──────────────────────────────────────────────────────────

def get_args_parser():
    parser = argparse.ArgumentParser("SA-DMAE pre-training", add_help=False)

    # Training
    parser.add_argument("--batch_size",   default=4,   type=int)
    parser.add_argument("--epochs",       default=200, type=int)
    parser.add_argument("--accum_iter",   default=4,   type=int,
                        help="Gradient accumulation steps (effective_bs = batch_size * accum_iter)")

    # Model
    parser.add_argument("--model",        default="sa_dmae_vit_base_patch16", type=str)
    parser.add_argument("--input_size",   default=224,  type=int)
    parser.add_argument("--mask_ratio",   default=0.75, type=float)
    parser.add_argument("--norm_pix_loss", action="store_true")
    parser.set_defaults(norm_pix_loss=False)
    parser.add_argument("--sigma",        default=0.25, type=float,
                        help="Gaussian noise std applied to center slice")

    # SA-DMAE specific
    parser.add_argument("--n_slices",     default=3,   type=int,
                        help="Number of consecutive axial slices per sample (must be odd)")
    parser.add_argument("--axial_depth",  default=4,   type=int,
                        help="Number of late ViT blocks that include AxialAttention")

    # Optimizer
    parser.add_argument("--weight_decay", default=0.05,  type=float)
    parser.add_argument("--lr",           default=None,  type=float)
    parser.add_argument("--blr",          default=1e-3,  type=float,
                        help="Base lr: actual_lr = blr * effective_batch_size / 256")
    parser.add_argument("--min_lr",       default=0.0,   type=float)
    parser.add_argument("--warmup_epochs", default=20,   type=int)

    # Dataset
    parser.add_argument("--data_path",    default="",  type=str,
                        help="BraTS root dir (NIfTI) or .pt directory (--preprocessed)")
    parser.add_argument("--edg_path",     default="",  type=str,
                        help="EDG root dir — combined with data_path when provided")
    parser.add_argument("--preprocessed", action="store_true",
                        help="Load pre-extracted .pt slices instead of raw NIfTI")
    parser.add_argument("--modalities",   default=["t1ce", "t2", "flair"], nargs="+")
    parser.add_argument("--num_workers",  default=2,   type=int)
    parser.add_argument("--pin_mem",      action="store_true")
    parser.set_defaults(pin_mem=False)

    # Logging / checkpointing
    parser.add_argument("--output_dir",   default="./output_sa", type=str)
    parser.add_argument("--log_dir",      default="./output_sa", type=str)
    parser.add_argument("--save_every",   default=20,  type=int,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--resume",       default="",  type=str)
    parser.add_argument("--start_epoch",  default=0,   type=int)
    parser.add_argument("--seed",         default=0,   type=int)
    parser.add_argument("--device",       default="cuda", type=str)

    # Distributed (optional — single-process fine too)
    parser.add_argument("--world_size",   default=1,   type=int)
    parser.add_argument("--local_rank",   default=-1,  type=int)
    parser.add_argument("--dist_on_itp",  action="store_true")
    parser.add_argument("--dist_url",     default="env://", type=str)

    return parser


# ── Training loop ────────────────────────────────────────────────────────────

def train_one_epoch(model, data_loader, optimizer, device, epoch,
                    loss_scaler, log_writer, args):
    model.train()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch [{epoch}]"

    use_amp = device.type == "cuda"
    optimizer.zero_grad()

    for step, (samples, _) in enumerate(metric_logger.log_every(data_loader, 20, header)):
        # lr schedule: per-iteration cosine warmup
        if step % args.accum_iter == 0:
            _adjust_learning_rate(optimizer, step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)  # (B, n_slices, C, H, W)

        with torch.cuda.amp.autocast(enabled=use_amp):
            loss, _, _ = model(samples, mask_ratio=args.mask_ratio)

        if not torch.isfinite(loss):
            print(f"Loss is {loss.item()}, stopping training")
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

        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if log_writer is not None and (step + 1) % args.accum_iter == 0:
            epoch_1000x = int((step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar("train_loss", loss.item(), epoch_1000x)
            log_writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch_1000x)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def _adjust_learning_rate(optimizer, progress, args):
    """Cosine decay with linear warmup (progress = epoch + step/steps_per_epoch)."""
    if progress < args.warmup_epochs:
        lr = args.lr * progress / args.warmup_epochs
    else:
        import math
        t = (progress - args.warmup_epochs) / (args.epochs - args.warmup_epochs)
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * (1 + math.cos(math.pi * t))
    for pg in optimizer.param_groups:
        pg["lr"] = lr * pg.get("lr_scale", 1.0)


# ── Main ─────────────────────────────────────────────────────────────────────

def main(args):
    misc.init_distributed_mode(args)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu"
                          else "cpu")
    if device.type == "cpu":
        print("CUDA not available — running on CPU")

    # Reproducibility
    torch.manual_seed(args.seed + misc.get_rank())
    np.random.seed(args.seed + misc.get_rank())
    if device.type == "cuda":
        cudnn.benchmark = True

    # ── Dataset ──────────────────────────────────────────────────────────────
    if args.preprocessed:
        dataset_train = PreprocessedDataset(args.data_path)
        print(f"Preprocessed dataset: {len(dataset_train)} samples from {args.data_path}")
    elif args.edg_path:
        dataset_train = get_combined_dataset(
            args.data_path, args.edg_path,
            modalities=args.modalities, n_slices=args.n_slices,
            target_size=(args.input_size, args.input_size),
        )
        print(f"Combined dataset (BraTS + EDG): {len(dataset_train)} samples")
    else:
        dataset_train = get_brats2021_dataset(
            args.data_path,
            modalities=args.modalities, n_slices=args.n_slices,
            target_size=(args.input_size, args.input_size),
        )
        print(f"BraTS 2021 dataset: {len(dataset_train)} samples")

    sampler = RandomSampler(dataset_train)
    data_loader = DataLoader(
        dataset_train, sampler=sampler,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = models_sa_dmae.__dict__[args.model](
        norm_pix_loss=args.norm_pix_loss,
        sigma=args.sigma,
        n_slices=args.n_slices,
        axial_depth=args.axial_depth,
    )
    model.to(device)
    print(f"Model: {args.model}  |  params: {sum(p.numel() for p in model.parameters()):,}")

    # ── Optimizer ─────────────────────────────────────────────────────────────
    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256
    print(f"Effective batch size: {eff_batch_size}  |  lr: {args.lr:.2e}")

    # timm 1.x API: param_groups_weight_decay (replaces add_weight_decay)
    try:
        param_groups = optim_factory.param_groups_weight_decay(
            model, args.weight_decay
        )
    except AttributeError:
        param_groups = optim_factory.add_weight_decay(model, args.weight_decay)

    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model,
                    optimizer=optimizer, loss_scaler=loss_scaler)

    # ── Logging ───────────────────────────────────────────────────────────────
    log_writer = None
    if args.log_dir and misc.is_main_process():
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)

    # ── Training loop ──────────────────────────────────────────────────────────
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs):
        train_stats = train_one_epoch(
            model, data_loader, optimizer, device, epoch,
            loss_scaler, log_writer, args,
        )

        # Checkpoint
        if args.output_dir and (epoch % args.save_every == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model,
                optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch,
            )

        # Log
        log_stats = {**{f"train_{k}": v for k, v in train_stats.items()}, "epoch": epoch}
        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    print(f"Training time: {total}")


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
