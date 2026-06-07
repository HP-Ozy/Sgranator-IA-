"""Architetture per il denoising.

Due modelli, stessa interfaccia (input [N,3,H,W] in [0,1] -> output stessa shape):

- DenoisingAutoencoder: encoder/decoder simmetrico, leggero (~600K param).
- UNet: skip connections, preserva i dettagli fini (~7.7M param).

Entrambi sono *size-preserving* per immagini con lato multiplo di 8, quindi
non serve alcun fallback di interpolazione a valle.
"""

import torch
import torch.nn as nn


def _conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class DenoisingAutoencoder(nn.Module):
    """Autoencoder convoluzionale per denoising.

    L'encoder comprime con conv stride-2, il decoder ricostruisce con
    ConvTranspose stride-2: per ogni livello la dimensione si dimezza e poi
    raddoppia esattamente, quindi output.shape == input.shape.
    """

    def __init__(self, channels=3, features=(32, 64, 128)):
        super().__init__()
        features = list(features)

        # ── Encoder: ogni step dimezza H,W e porta a `f` canali ──
        enc = []
        in_ch = channels
        for f in features:
            enc.append(nn.Sequential(
                nn.Conv2d(in_ch, f, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(f),
                nn.ReLU(inplace=True),
            ))
            in_ch = f
        self.encoder = nn.ModuleList(enc)

        # ── Bottleneck ──
        bottleneck = features[-1] * 2
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_ch, bottleneck, kernel_size=3, padding=1),
            nn.BatchNorm2d(bottleneck),
            nn.ReLU(inplace=True),
        )
        in_ch = bottleneck

        # ── Decoder: ogni step raddoppia H,W ──
        dec = []
        for f in reversed(features):
            dec.append(nn.Sequential(
                nn.ConvTranspose2d(in_ch, f, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(f),
                nn.ReLU(inplace=True),
            ))
            in_ch = f
        self.decoder = nn.ModuleList(dec)

        # Testa finale: torna ai canali originali, sigmoid per restare in [0,1]
        self.head = nn.Sequential(
            nn.Conv2d(in_ch, channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        for block in self.encoder:
            x = block(x)
        x = self.bottleneck(x)
        for block in self.decoder:
            x = block(x)
        return self.head(x)


class UNet(nn.Module):
    """U-Net per denoising con skip connections.

    Le skip connections passano i dettagli ad alta frequenza dall'encoder
    al decoder, evitando la perdita di nitidezza tipica dell'autoencoder puro.
    """

    def __init__(self, channels=3, features=(64, 128, 256, 512)):
        super().__init__()
        features = list(features)

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(2)

        # Encoder
        in_ch = channels
        for f in features:
            self.downs.append(_conv_block(in_ch, f))
            in_ch = f

        # Bottleneck
        self.bottleneck = _conv_block(features[-1], features[-1] * 2)

        # Decoder (up-conv + conv block che fonde la skip)
        for f in reversed(features):
            self.ups.append(nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2))
            self.ups.append(_conv_block(f * 2, f))

        self.head = nn.Sequential(
            nn.Conv2d(features[0], channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skips = skips[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)              # up-conv
            skip = skips[idx // 2]
            # Allinea eventuali mismatch di 1px dovuti al pooling
            if x.shape[-2:] != skip.shape[-2:]:
                x = nn.functional.interpolate(x, size=skip.shape[-2:], mode="nearest")
            x = torch.cat((skip, x), dim=1)
            x = self.ups[idx + 1](x)          # conv block

        return self.head(x)


def build_model(name="autoencoder", channels=3):
    """Factory usata da train.py / denoise.py."""
    name = name.lower()
    if name == "autoencoder":
        return DenoisingAutoencoder(channels=channels, features=(32, 64, 128))
    if name == "unet":
        # base ridotta: ~7.7M param, gira bene anche su GPU da 4GB
        return UNet(channels=channels, features=(32, 64, 128, 256))
    raise ValueError(f"Modello sconosciuto: {name!r} (usa 'autoencoder' o 'unet')")


if __name__ == "__main__":
    # Smoke test: verifica forward + size preservation + conteggio parametri.
    x = torch.rand(2, 3, 64, 64)
    for name in ("autoencoder", "unet"):
        m = build_model(name)
        y = m(x)
        n = sum(p.numel() for p in m.parameters())
        assert y.shape == x.shape, f"{name}: shape {y.shape} != {x.shape}"
        print(f"{name:12s} ok | out {tuple(y.shape)} | {n:,} parametri")
