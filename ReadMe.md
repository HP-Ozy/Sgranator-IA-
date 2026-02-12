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
| **Autoencoder** | ~590K | Veloce, semplice, ottimo per imparare | Perde dettagli fini |
| **U-Net** | ~7.7M | Skip connections preservano i dettagli | Più lento, più VRAM |

### 1. Setup

```bash
git clone https://github.com/tuo-username/pytorch-photo-denoiser.git
cd pytorch-photo-denoiser
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
| `--output-dir` | `./checkpoints` | Dove salvare il modello |

## 🤔 Limitazioni (e cosa imparare da esse)

Questo progetto è pensato per **imparare**, quindi ha limiti voluti:

1. **Risoluzione bassa** — Il modello lavora a 128x128. Per risoluzioni maggiori servono architetture più complesse (diffusion models, etc.)
2. **Rumore artificiale** — Il modello è addestrato su rumore sintetico. Il rumore reale delle fotocamere è più complesso e variegato.
3. **Nessun rumore specifico** — Un modello professionale sarebbe addestrato su coppie di foto reali (con/senza rumore) dello stesso sensore.
4. **Perdita di dettagli** — L'autoencoder base tende a "sfocare" leggermente. La U-Net è meglio grazie alle skip connections.

**Prossimi passi per chi vuole andare oltre:**
- Implementare le Perceptual Loss (confronto nello spazio delle feature, non dei pixel)
- Provare architetture residuali (DnCNN)
- Usare dataset di rumore reale (SIDD, DND)
- Aggiungere training con mixed noise (più tipi contemporaneamente)

## 🖥️ Requisiti hardware

| Hardware | Tempo per 20 epoche (CIFAR-10) | Note |
|----------|-------------------------------|------|
| GPU NVIDIA (RTX 3060+) | ~3-5 min | Consigliato |
| Apple Silicon (M1/M2) | ~5-10 min | Supporto MPS |
| Solo CPU | ~20-40 min | Funziona, ma lento |

## 📜 Licenza

MIT — Usa, modifica e condividi liberamente!

---

<p align="center">
  Fatto con ❤️ e PyTorch
  <br><br>
  ⭐ Se ti è utile, lascia una stella!
</p>
