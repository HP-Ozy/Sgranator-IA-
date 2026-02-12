import argparse
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models import DenoisingAutoencoder
from utils import CIFAR10NoisyDataset, show_grid, calculate_psnr


def main():
    parser = argparse.ArgumentParser(description="🚀 Demo rapida del denoiser")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--noise-level", type=float, default=0.15)
    parser.add_argument("--noise-type", type=str, default="gaussian",
                        choices=["gaussian", "salt_pepper", "poisson", "speckle"])
    parser.add_argument("--image-size", type=int, default=64,
                        help="Dimensione immagini (64 per demo veloce)")
    args = parser.parse_args()

    
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"🖥️ Device: {device}")

    # ── Dataset ──────────────────────────────────────────────────
    print("\n📥 Scaricamento CIFAR-10...")
    dataset = CIFAR10NoisyDataset(
        image_size=args.image_size,
        noise_type=args.noise_type,
        noise_level=args.noise_level,
    )
    loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2)

    # ── Modello ──────────────────────────────────────────────────
    model = DenoisingAutoencoder(channels=3, features=[32, 64, 128]).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"🧠 Modello: {params:,} parametri")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # ── Training ──────────────────────────────────────────
    print(f"\n🏋️ Training per {args.epochs} epoche...\n")
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0
        for noisy, clean in loader:
            noisy, clean = noisy.to(device), clean.to(device)
            output = model(noisy)

            if output.shape != clean.shape:
                clean = nn.functional.interpolate(
                    clean, size=output.shape[2:], mode="bilinear", align_corners=False
                )

            loss = criterion(output, clean)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        print(f"  Epoca {epoch}/{args.epochs} │ Loss: {avg_loss:.6f}")

    elapsed = time.time() - start
    print(f"\n⏱ Training completato in {elapsed:.1f}s")

   
    print("\n🖼️ Generazione risultati...")
    model.eval()
    with torch.no_grad():
        noisy_batch, clean_batch = next(iter(loader))
        noisy_batch = noisy_batch.to(device)
        denoised_batch = model(noisy_batch).cpu()

        if denoised_batch.shape != clean_batch.shape:
            clean_batch = nn.functional.interpolate(
                clean_batch, size=denoised_batch.shape[2:],
                mode="bilinear", align_corners=False
            )

        # Calcola metriche
        psnr_noisy = calculate_psnr(clean_batch[:8], noisy_batch[:8].cpu())
        psnr_denoised = calculate_psnr(clean_batch[:8], denoised_batch[:8])

        print(f"\n📊 Metriche:")
        print(f"   PSNR rumorosa:  {psnr_noisy:.1f} dB")
        print(f"   PSNR denoisata: {psnr_denoised:.1f} dB")
        print(f"   Miglioramento:  +{psnr_denoised - psnr_noisy:.1f} dB ✨")

        
        show_grid(
            noisy_batch.cpu(), clean_batch, denoised_batch,
            n_samples=4, save_path="./demo_results.png"
        )

    print("\n✅ Demo completata! Risultati salvati in demo_results.png")
    print("   Per un training completo, usa: python train.py --epochs 50")


if __name__ == "__main__":
    main()
