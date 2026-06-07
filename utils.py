"""Dataset, rumore, metriche e visualizzazione.

Tutto lavora su tensori float in [0, 1] con shape [C,H,W] o [N,C,H,W].
"""

import os

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from PIL import Image


# ──────────────────────────────────────────────────────────────────
# Rumore — ogni funzione prende un'immagine pulita e ne restituisce
# una rumorosa, mantenendo i valori in [0, 1].
# ──────────────────────────────────────────────────────────────────
def add_noise(img, noise_type="gaussian", noise_level=0.1):
    if noise_type == "gaussian":
        # Grana da ISO alto: rumore additivo normale.
        noisy = img + torch.randn_like(img) * noise_level

    elif noise_type == "salt_pepper":
        # Pixel difettosi: una frazione `noise_level` va a 0 o 1.
        noisy = img.clone()
        probs = torch.rand_like(img)
        noisy[probs < noise_level / 2] = 0.0
        noisy[probs > 1 - noise_level / 2] = 1.0

    elif noise_type == "poisson":
        # Rumore quantistico (shot noise), dipendente dal segnale.
        scale = max(1.0, 255.0 * (1.0 - noise_level))
        noisy = torch.poisson(img * scale) / scale

    elif noise_type == "speckle":
        # Rumore moltiplicativo (radar/ultrasuoni).
        noisy = img + img * torch.randn_like(img) * noise_level

    else:
        raise ValueError(f"Tipo di rumore sconosciuto: {noise_type!r}")

    return noisy.clamp(0.0, 1.0)


# ──────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────
class CIFAR10NoisyDataset(Dataset):
    """CIFAR-10 con rumore sintetico generato al volo.

    Restituisce coppie (noisy, clean). Il rumore è ricampionato a ogni
    accesso, così la rete vede realizzazioni di rumore sempre diverse.
    """

    def __init__(self, image_size=64, noise_type="gaussian", noise_level=0.1,
                 train=True, root="./data"):
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
        self.base = datasets.CIFAR10(root=root, train=train, download=True)
        self.noise_type = noise_type
        self.noise_level = noise_level

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, _ = self.base[idx]
        clean = self.transform(img)
        noisy = add_noise(clean, self.noise_type, self.noise_level)
        return noisy, clean


class CustomImageDataset(Dataset):
    """Immagini da una cartella locale, con rumore sintetico al volo."""

    EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

    def __init__(self, image_dir, image_size=128, noise_type="gaussian",
                 noise_level=0.1):
        self.paths = [
            os.path.join(image_dir, f)
            for f in sorted(os.listdir(image_dir))
            if f.lower().endswith(self.EXTS)
        ]
        if not self.paths:
            raise FileNotFoundError(f"Nessuna immagine in {image_dir!r}")
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
        self.noise_type = noise_type
        self.noise_level = noise_level

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        clean = self.transform(img)
        noisy = add_noise(clean, self.noise_type, self.noise_level)
        return noisy, clean


# ──────────────────────────────────────────────────────────────────
# Metriche
# ──────────────────────────────────────────────────────────────────
def calculate_psnr(img1, img2, max_val=1.0):
    """PSNR medio (dB) tra due batch/immagini in [0, max_val]."""
    mse = F.mse_loss(img1, img2).item()
    if mse == 0:
        return float("inf")
    return 10.0 * torch.log10(torch.tensor(max_val ** 2 / mse)).item()


def _gaussian_window(window_size, sigma, channels, device):
    coords = torch.arange(window_size, dtype=torch.float32, device=device)
    coords -= window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    window_2d = g[:, None] @ g[None, :]
    return window_2d.expand(channels, 1, window_size, window_size).contiguous()


def calculate_ssim(img1, img2, window_size=11, sigma=1.5, max_val=1.0):
    """SSIM medio (0..1) — implementazione torch, nessuna dipendenza extra."""
    if img1.dim() == 3:
        img1, img2 = img1.unsqueeze(0), img2.unsqueeze(0)
    channels = img1.size(1)
    window = _gaussian_window(window_size, sigma, channels, img1.device)
    pad = window_size // 2

    mu1 = F.conv2d(img1, window, padding=pad, groups=channels)
    mu2 = F.conv2d(img2, window, padding=pad, groups=channels)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=channels) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=pad, groups=channels) - mu1_mu2

    c1, c2 = (0.01 * max_val) ** 2, (0.03 * max_val) ** 2
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / \
               ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return ssim_map.mean().item()


# ──────────────────────────────────────────────────────────────────
# Visualizzazione
# ──────────────────────────────────────────────────────────────────
def show_grid(noisy, clean, denoised, n_samples=4, save_path=None):
    """Salva/mostra una griglia: righe = Rumorosa / Pulita / Denoisata."""
    import matplotlib
    if save_path:
        matplotlib.use("Agg")  # backend headless: nessuna finestra richiesta
    import matplotlib.pyplot as plt

    n = min(n_samples, noisy.size(0))
    rows = [("Rumorosa", noisy), ("Pulita", clean), ("Denoisata", denoised)]
    fig, axes = plt.subplots(3, n, figsize=(2.5 * n, 7.5))
    if n == 1:
        axes = axes.reshape(3, 1)

    for r, (label, batch) in enumerate(rows):
        for c in range(n):
            img = batch[c].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
            axes[r, c].imshow(img)
            axes[r, c].axis("off")
            if c == 0:
                axes[r, c].set_ylabel(label, fontsize=12)
        axes[r, 0].axis("on")
        axes[r, 0].set_xticks([])
        axes[r, 0].set_yticks([])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
