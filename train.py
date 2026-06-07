"""Training completo del denoiser.

Esempi:
    python train.py --dataset cifar10 --epochs 30 --model autoencoder
    python train.py --dataset custom --image-dir ./mie_foto/ --epochs 50
    python train.py --dataset cifar10 --model unet --epochs 30
"""

import argparse
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from models import build_model
from utils import (CIFAR10NoisyDataset, CustomImageDataset,
                   calculate_psnr, calculate_ssim)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_args():
    p = argparse.ArgumentParser(description="Training del Denoising Autoencoder")
    p.add_argument("--dataset", choices=["cifar10", "custom"], default="cifar10")
    p.add_argument("--image-dir", default="./data/images",
                   help="Cartella immagini (solo con --dataset custom)")
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--model", choices=["autoencoder", "unet"], default="autoencoder")
    p.add_argument("--noise-type", default="gaussian",
                   choices=["gaussian", "salt_pepper", "poisson", "speckle"])
    p.add_argument("--noise-level", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-split", type=float, default=0.1)
    p.add_argument("--num-workers", type=int, default=0,
                   help="DataLoader workers (0 = sicuro su Windows)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default="./checkpoints")
    return p.parse_args()


def build_dataset(args):
    if args.dataset == "cifar10":
        return CIFAR10NoisyDataset(
            image_size=args.image_size, noise_type=args.noise_type,
            noise_level=args.noise_level, train=True,
        )
    return CustomImageDataset(
        image_dir=args.image_dir, image_size=args.image_size,
        noise_type=args.noise_type, noise_level=args.noise_level,
    )


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum, psnr_sum, ssim_sum, n = 0.0, 0.0, 0.0, 0
    for noisy, clean in loader:
        noisy, clean = noisy.to(device), clean.to(device)
        out = model(noisy)
        loss_sum += criterion(out, clean).item()
        psnr_sum += calculate_psnr(clean, out)
        ssim_sum += calculate_ssim(clean, out)
        n += 1
    return loss_sum / n, psnr_sum / n, ssim_sum / n


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = get_device()
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Device: {device} │ Modello: {args.model} │ Dataset: {args.dataset}")

    # ── Dati: split train/val ──
    full = build_dataset(args)
    n_val = max(1, int(len(full) * args.val_split))
    n_train = len(full) - n_val
    train_ds, val_ds = random_split(
        full, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)
    print(f"Campioni: {n_train} train / {n_val} val")

    # ── Modello / ottimizzazione ──
    model = build_model(args.model).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"Parametri: {params:,}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3)
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # ── Loop ──
    best_val = float("inf")
    ckpt_path = os.path.join(args.output_dir, f"{args.model}.pth")
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for noisy, clean in train_loader:
            noisy, clean = noisy.to(device), clean.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                out = model(noisy)
                loss = criterion(out, clean)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        val_loss, val_psnr, val_ssim = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        flag = ""
        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "model": args.model,
                "channels": 3,
                "image_size": args.image_size,
                "state_dict": model.state_dict(),
            }, ckpt_path)
            flag = "  ✓ best"

        print(f"Epoca {epoch:3d}/{args.epochs} │ train {train_loss:.5f} │ "
              f"val {val_loss:.5f} │ PSNR {val_psnr:5.1f} dB │ "
              f"SSIM {val_ssim:.3f}{flag}")

    print(f"\nCompletato in {time.time() - start:.1f}s")
    print(f"Miglior checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
