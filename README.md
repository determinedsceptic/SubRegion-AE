# SubRegion-AE

Latent space compression for ocean reanalysis data in the Kuroshio Extension, targeting downstream data assimilation.

## Problem

Compressing high-dimensional ocean state fields (101 channels: 25-level U, V, T, S + SSH) into a compact latent space that is:
1. **High compression ratio** (~389x) with low reconstruction error (R² > 0.99)
2. **Sign-flip free** — continuous autoencoders suffer from sign/contrast ambiguity where reconstructed fields have correct spatial structure but inverted values
3. **DA-friendly** — smooth, continuous latent space suitable for ensemble-based data assimilation (EnKF, etc.)

## Method

**DCAE-VAE**: Deep Convolutional AutoEncoder with a VAE bottleneck.

- **Encoder**: Stem Conv → 3 EncoderStages (ResBlock + SpatialAttention + 2x downsample) → RMSNorm → mu/logvar heads
- **Bottleneck**: VAE reparameterization (z = mu + sigma * epsilon), KL regularization keeps latent space smooth
- **Decoder**: Symmetric DecoderStages (2x upsample + ResBlock + SpatialAttention) → Final Conv
- **Loss**: Masked reconstruction (L1+L2) + FFT spectral loss + KL divergence + structured latent loss (DC-AE 1.5)

### Sign-Flip Solution

The sign-flip problem arises from three interacting factors:
1. **Latent sign degeneracy**: both z and -z can be valid encodings without regularization
2. **FFT loss is sign-invariant**: |FFT(x)| = |FFT(-x)|, providing no corrective gradient
3. **Unregularized latent space**: disconnected clusters allow different sign conventions

**KL regularization** (kl_weight=1e-2) forces the posterior close to N(0,1), ensuring the latent space is dense and continuous around the origin. The decoder must learn a consistent mapping in this high-density region, eliminating sign-flip discontinuities. Additionally, reducing FFT weight (0.5 → 0.1) lets reconstruction loss dominate, providing stronger corrective gradients.

## Data

**GLORYS12 Kuroshio Extension Subregion** (25°N–45.75°N, 150°E–174.9°E)

| Property | Value |
|----------|-------|
| Variables | U, V, T, S (25 depth levels each) + SSH |
| Channels | 101 |
| Grid | 250 × 300 |
| Train / Val / Test | 1993–2017 / 2018 / 2019–2020 |
| Preprocessing | Per-channel min-max → [0,1] |

Data is stored on the remote server (`test1`) and is not included in this repository.

## Model Comparison

| | DCAE-VAE | VQ-VAE |
|---|---|---|
| Latent type | Continuous (float32) | Discrete (codebook index) |
| Latent shape | [16, 32, 38] | [128, 32, 38] / indices [32, 38] |
| Compression ratio | ~389x | ~49x (continuous) / ~18,124x (indices) |
| Sign-flip | Solved (KL) | N/A (discrete) |
| DA compatibility | Linear operations, gradients | No interpolation/gradients |

## Repository Structure

```
SubRegion-AE/
├── model/
│   ├── dcae.py              # DCAE-VAE (main model)
│   ├── vqvae.py             # VQ-VAE baseline
│   └── vae.py               # Standard VAE
├── data/
│   ├── dataset.py           # OceanRawDataset, build_dataset(), load_constants()
│   └── data_utils.py        # normalize_fn / denormalize_fn
├── train/
│   ├── train_dcae.py        # DCAE training (DDP, AMP)
│   └── train_vqvae.py       # VQ-VAE training
├── eval/
│   ├── save_latent.py       # Offline latent space export
│   ├── eval_flip.py         # Flip metric evaluation
│   └── eval_multi_seed.py   # Multi-seed comparison
├── script/
│   ├── shell/               # Training launch scripts (torchrun)
│   └── notebook/            # Evaluation notebooks (eval_dcae.ipynb, eval_vqvae.ipynb)
├── utils/
│   ├── checkpoint.py        # Save/load/auto-resume checkpoints
│   └── flip_metrics.py      # FLIP metric computation
├── config.py                # DatasetConfig from datasets.yaml
├── datasets.yaml            # Dataset paths and metadata
└── sync.sh                  # rsync to remote server
```

## Quick Start

### Install

```bash
pip install -r requirements.txt
```

### Train

```bash
# DCAE-VAE (default: 6 GPU, 500 epochs)
bash script/shell/run_dcae_single.sh

# Override parameters
FFT_WEIGHT=0.1 KL_WEIGHT=1e-2 bash script/shell/run_dcae_single.sh

# VQ-VAE baseline
bash script/shell/run_vqvae_single.sh
```

### Evaluate

Use the Jupyter notebooks in `script/notebook/`:
- `eval_dcae.ipynb` — reconstruction metrics (physical units), per-variable breakdown, correlation analysis
- `eval_vqvae.ipynb` — VQ-VAE evaluation

### Export Latent Space

```bash
python eval/save_latent.py \
    --ckpt-path output/glorys12_kuroshio_extension/<tag>/best_model.pth \
    --output-dir output/latent/<tag> \
    --splits train val test
```

Outputs one `{YYYYMMDD}.pt` per day with shape `[16, 32, 38]`.

## Training Configuration

Key hyperparameters (DCAE-VAE):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `base_channels` | 64 | Base feature channels |
| `channel_multipliers` | 1 2 4 8 | Per-stage multipliers (3 stages, 8x spatial downsample) |
| `latent_channels` | 16 | Bottleneck channels |
| `fft_weight` | 0.1 | FFT spectral loss weight |
| `kl_weight` | 1e-2 | KL divergence weight |
| `structured_weight` | 1.0 | DC-AE 1.5 structured latent loss weight |
| `lr` | 2e-4 | Learning rate (AdamW + CosineAnnealing) |

## Development Workflow

1. Edit code locally
2. `./sync.sh` to rsync to remote server
3. Run training on remote: `bash script/shell/run_dcae_single.sh`
4. Evaluate with notebooks on remote
