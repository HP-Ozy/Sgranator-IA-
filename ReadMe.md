# 🧹 PyTorch Photo Denoiser

![kokk](https://github.com/user-attachments/assets/7b1161f6-efc9-4e04-bd9c-4e13d302be0c)

Un **Denoising Autoencoder** costruito da zero con PyTorch che pulisce foto sgranate e rumorose. Risultati visivi immediati, perfetto per imparare come funzionano le reti neurali applicati a problemi reali.

---

## 🎯 Cosa fa

Prende un'immagine con rumore (grana, pixel difettosi, artefatti) e la restituisce pulita. La rete impara a **distinguere il segnale dal rumore** comprimendo l'immagine in una rappresentazione compatta e ricostruendola.

## 🧠 Come funziona

```
Immagine rumorosa ──→ [ ENCODER ] ──→ Bottleneck ──→ [ DECODER ] ──→ Immagine pulita
   (3 canali)          Conv layers     (256 ch)      ConvTranspose     (3 canali)
```

L'**encoder** comprime l'immagine scartando le informazioni meno importanti (incluso il rumore). Il **decoder** ricostruisce l'immagine usando solo le feature essenziali. Il risultato? Un'immagine più pulita dell'originale rumoroso.

### Due architetture disponibili

| Modello | Parametri | Pro | Contro |
|---------|-----------|-----|--------|
| **Autoencoder** | ~1.1M | Veloce, semplice, ottimo per imparare | Perde dettagli fini |
| **U-Net** | ~7.7M | Skip connections preservano i dettagli | Più lento, più VRAM |

> Entrambi i modelli sono *size-preserving*: l'output ha sempre la stessa
> risoluzione dell'input (per lati multipli di 8). In inferenza le immagini
> di dimensione arbitraria vengono gestite con padding riflesso + crop.

### 1. Setup

```bash
git clone https://github.com/HP-Ozy/Sgranator-IA-.git
cd Sgranator-IA-
pip install -r requirements.txt
```

### 2. Demo istantanea (2 minuti!)

```bash
python demo.py
```

Scarica automaticamente CIFAR-10, addestra un modello per 5 epoche e mostra i risultati. Nessun dataset necessario!

### 3. Training completo

```bash
# Con CIFAR-10 (50k immagini, download automatico)
python train.py --dataset cifar10 --epochs 30 --model autoencoder

# Con le TUE immagini
python train.py --dataset custom --image-dir ./mie_foto/ --epochs 50

# Con U-Net (risultati migliori, più lento)
python train.py --dataset cifar10 --model unet --epochs 30
```

### 4. Pulisci le tue foto

```bash
# Singola immagine
python denoise.py --input foto_rumorosa.jpg --output foto_pulita.png

# Cartella intera
python denoise.py --input ./foto_rumorose/ --output ./foto_pulite/

# Con confronto side-by-side
python denoise.py --input foto.jpg --compare
```

## Tipi di rumore supportati

Il progetto supporta 4 tipi di rumore, ognuno simula un problema reale:

| Tipo | Flag | Cosa simula | Esempio reale |
|------|------|-------------|---------------|
| **Gaussiano** | `--noise-type gaussian` | Grana da ISO alto | Foto notturne con smartphone |
| **Sale & Pepe** | `--noise-type salt_pepper` | Pixel difettosi | Sensore danneggiato |
| **Poisson** | `--noise-type poisson` | Rumore quantistico | Fotografia scientifica |
| **Speckle** | `--noise-type speckle` | Rumore moltiplicativo | Immagini radar/ultrasuoni |

```bash
# Prova diversi tipi di rumore
python train.py --noise-type salt_pepper --noise-level 0.08
python train.py --noise-type gaussian --noise-level 0.2
```

##  Metriche

Il modello viene valutato con:

- **PSNR** (Peak Signal-to-Noise Ratio): misura la qualità pixel per pixel. Sopra 30 dB è generalmente buono.
- **SSIM** (Structural Similarity): misura la somiglianza strutturale percepita. Più vicino a 1 = meglio.

Risultati tipici dopo 30 epoche su CIFAR-10 (rumore gaussiano, σ=0.1):

| Modello | PSNR rumorosa | PSNR denoisata | Miglioramento |
|---------|---------------|----------------|---------------|
| Autoencoder | ~20 dB | ~27 dB | +7 dB |
| U-Net | ~20 dB | ~30 dB | +10 dB |

## ⚙️ Tutti i parametri

### train.py

| Parametro | Default | Descrizione |
|-----------|---------|-------------|
| `--dataset` | `cifar10` | `cifar10` o `custom` |
| `--image-dir` | `./data/images` | Cartella immagini (solo con `custom`) |
| `--image-size` | `128` | Dimensione resize delle immagini |
| `--model` | `autoencoder` | `autoencoder` o `unet` |
| `--noise-type` | `gaussian` | Tipo di rumore |
| `--noise-level` | `0.1` | Intensità del rumore (0.0 - 1.0) |
| `--epochs` | `20` | Numero di epoche |
| `--batch-size` | `32` | Dimensione del batch |
| `--lr` | `0.001` | Learning rate |
| `--val-split` | `0.1` | Frazione usata per la validazione |
| `--num-workers` | `0` | Worker del DataLoader (0 = sicuro su Windows) |
| `--seed` | `42` | Seed per riproducibilità |
| `--output-dir` | `./checkpoints` | Dove salvare il modello |

Durante il training viene salvato automaticamente il **miglior** checkpoint
(in base alla val loss) in `--output-dir/<model>.pth`, con LR scheduler e
mixed precision (AMP) attivi su GPU.

### denoise.py

| Parametro | Default | Descrizione |
|-----------|---------|-------------|
| `--input` | *(obbligatorio)* | File immagine o cartella |
| `--output` | auto | File/cartella di output |
| `--model` | `autoencoder` | `autoencoder` o `unet` |
| `--checkpoint` | `./checkpoints/<model>.pth` | Path del modello addestrato |
| `--compare` | `False` | Salva confronto rumorosa \| pulita affiancate |

## 📁 Struttura del progetto

```
.
├── demo.py            # Demo 2 minuti su CIFAR-10
├── train.py          # Training completo (cifar10 / custom)
├── denoise.py        # Inferenza su immagini reali
├── models.py         # DenoisingAutoencoder + U-Net
├── utils.py          # Dataset, rumore, metriche (PSNR/SSIM), griglia
└── requirements.txt
```

