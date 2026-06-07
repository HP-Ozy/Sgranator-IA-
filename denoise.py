"""Inferenza: pulisce immagini reali usando un checkpoint addestrato.

Esempi:
    python denoise.py --input foto_rumorosa.jpg --output foto_pulita.png
    python denoise.py --input ./foto_rumorose/ --output ./foto_pulite/
    python denoise.py --input foto.jpg --compare
"""

import argparse
import os

import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image

from models import build_model

EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(checkpoint, model_name, device):
    ckpt = torch.load(checkpoint, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        model_name = ckpt.get("model", model_name)
        state = ckpt["state_dict"]
    else:  # checkpoint "grezzo" col solo state_dict
        state = ckpt
    model = build_model(model_name, channels=ckpt.get("channels", 3)
                        if isinstance(ckpt, dict) else 3).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, model_name


def _pad_to_multiple(x, m=8):
    """Padding riflesso fino a lato multiplo di `m` (richiesto dai modelli)."""
    _, _, h, w = x.shape
    ph, pw = (m - h % m) % m, (m - w % m) % m
    x = F.pad(x, (0, pw, 0, ph), mode="reflect")
    return x, h, w


@torch.no_grad()
def denoise_image(model, path, device):
    """Pulisce un'immagine di dimensione arbitraria e ritorna (noisy, clean)."""
    img = Image.open(path).convert("RGB")
    noisy = transforms.ToTensor()(img).unsqueeze(0).to(device)
    padded, h, w = _pad_to_multiple(noisy)
    out = model(padded)[:, :, :h, :w]  # ritaglia al formato originale
    return noisy.squeeze(0).cpu(), out.squeeze(0).clamp(0, 1).cpu()


def collect_inputs(path):
    if os.path.isdir(path):
        return [os.path.join(path, f) for f in sorted(os.listdir(path))
                if f.lower().endswith(EXTS)]
    return [path]


def output_path(args, in_path):
    if os.path.isdir(args.input):
        out_dir = args.output or (args.input.rstrip("/\\") + "_pulite")
        os.makedirs(out_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(in_path))[0]
        return os.path.join(out_dir, f"{base}_denoised.png")
    if args.output:
        return args.output
    base = os.path.splitext(in_path)[0]
    return f"{base}_denoised.png"


def main():
    p = argparse.ArgumentParser(description="Pulisci le tue foto col denoiser")
    p.add_argument("--input", required=True, help="File immagine o cartella")
    p.add_argument("--output", help="File/cartella di output")
    p.add_argument("--model", choices=["autoencoder", "unet"], default="autoencoder")
    p.add_argument("--checkpoint", help="Path del .pth (default ./checkpoints/<model>.pth)")
    p.add_argument("--compare", action="store_true",
                   help="Salva confronto rumorosa|pulita affiancate")
    args = p.parse_args()

    device = get_device()
    ckpt = args.checkpoint or os.path.join("./checkpoints", f"{args.model}.pth")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(
            f"Checkpoint non trovato: {ckpt}. Allena prima con train.py.")
    model, model_name = load_model(ckpt, args.model, device)
    print(f"Device: {device} │ Modello: {model_name} │ Checkpoint: {ckpt}")

    inputs = collect_inputs(args.input)
    print(f"Immagini da elaborare: {len(inputs)}")

    for in_path in inputs:
        noisy, clean = denoise_image(model, in_path, device)
        out_path = output_path(args, in_path)
        if args.compare:
            grid = torch.cat([noisy, clean], dim=2)  # affianca orizzontalmente
            cmp_path = os.path.splitext(out_path)[0] + "_compare.png"
            save_image(grid, cmp_path)
            print(f"  {in_path} -> {cmp_path}")
        else:
            save_image(clean, out_path)
            print(f"  {in_path} -> {out_path}")

    print("Fatto.")


if __name__ == "__main__":
    main()
