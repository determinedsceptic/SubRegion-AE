"""
eval_flip.py  —  量化 DCAE 重建中的局部对比度翻转（sign/contrast flip）

核心思路：
  1. 用高斯模糊提取局部背景场  bg = GaussianBlur(field, sigma)
  2. 局部异常  anom = field - bg
  3. 逐像素比较 sign(anom_true) vs sign(anom_recon)
  4. 输出：per-channel flip_rate、空间翻转 map、汇总指标

用法：
  # 单卡
  python3 eval/eval_flip.py --ckpt /path/to/best_model.pth [--sigma 8 16 32] [--split val]
  # 多卡
  torchrun --nproc_per_node 6 eval/eval_flip.py --ckpt /path/to/best_model.pth
"""

import os
import sys
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from model.dcae import DCAE
from data.dataset import build_dataset, load_constants
from data.data_utils import normalize_fn
from config import get_dataset_config
from utils.flip_metrics import compute_flip_metrics


# ═══════════════════════════════════════════════════════════════════════════════
# Distributed utilities
# ═══════════════════════════════════════════════════════════════════════════════

def setup_distributed(args):
    if "LOCAL_RANK" in os.environ:
        args.local_rank = int(os.environ["LOCAL_RANK"])

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
    else:
        args.rank = 0
        args.world_size = 1

    args.distributed = args.world_size > 1

    if torch.cuda.is_available():
        torch.cuda.set_device(args.local_rank)
        args.device = f"cuda:{args.local_rank}"

    if args.distributed:
        dist.init_process_group(backend="nccl", init_method="env://")
        dist.barrier()


def cleanup_distributed(args):
    if args.distributed and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def is_rank0(args):
    return args.rank == 0


# ═══════════════════════════════════════════════════════════════════════════════
# Model loading & inference
# ═══════════════════════════════════════════════════════════════════════════════

def load_model(ckpt_path, in_channels, device):
    ckpt = torch.load(ckpt_path, map_location=device)

    # 从 checkpoint 中提取模型超参（如果存在）
    args_dict = ckpt.get('args', {})
    base_channels = args_dict.get('base_channels', 64)
    channel_multipliers = args_dict.get('channel_multipliers', [1, 2, 4, 8])
    latent_channels = args_dict.get('latent_channels', 16)
    num_res_blocks = args_dict.get('num_res_blocks', 2)
    attention_resolutions = args_dict.get('attention_resolutions', [1, 2])
    num_heads = args_dict.get('num_heads', 8)

    model = DCAE(
        in_channels=in_channels,
        base_channels=base_channels,
        channel_multipliers=channel_multipliers,
        latent_channels=latent_channels,
        num_res_blocks=num_res_blocks,
        attention_resolutions=attention_resolutions,
        num_heads=num_heads,
    ).to(device)

    state = ckpt.get('model_state_dict', ckpt.get('model', ckpt))
    state = {(k[7:] if k.startswith('module.') else k): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval()

    print(f"[rank {dist.get_rank() if dist.is_initialized() else 0}] "
          f"Loaded model from {ckpt_path}")
    print(f"  base_channels={base_channels}, channel_multipliers={channel_multipliers}, "
          f"latent_channels={latent_channels}")
    return model


@torch.inference_mode()
def run_inference_and_eval(model, loader, mu, sigma_norm, mask, device,
                           sigmas, vis_samples, max_batches=None):
    """
    Batch-wise inference + flip metric computation to avoid OOM.

    Returns per-sigma dict with accumulated stats and vis data.
    """
    mask_cpu = mask.cpu()

    # per-sigma accumulators
    sigma_data = {}
    for s in sigmas:
        sigma_data[s] = {
            'flip_rates': [],    # list of [B, C]
            'local_corrs': [],   # list of [B, C]
            # vis data: {global_sample_idx: (flip_map, anom_true, anom_recon)}
            'vis': {},
        }

    sample_offset = 0
    vis_set = set(vis_samples)

    for step, batch in enumerate(loader):
        if max_batches is not None and step >= max_batches:
            break
        raw = batch.to(device=device, dtype=torch.float32)
        x_norm = normalize_fn(raw, mu=mu, sigma=sigma_norm)
        recon_norm = model(x_norm)

        x_cpu = x_norm.cpu()
        recon_cpu = recon_norm.cpu()
        B = x_cpu.shape[0]

        for s in sigmas:
            flip_map, flip_rate, local_corr, anom_true, anom_recon = \
                compute_flip_metrics(x_cpu, recon_cpu, mask_cpu, s)

            sigma_data[s]['flip_rates'].append(flip_rate)
            sigma_data[s]['local_corrs'].append(local_corr)

            # save vis data for requested samples
            for local_i in range(B):
                global_i = sample_offset + local_i
                if global_i in vis_set:
                    sigma_data[s]['vis'][global_i] = (
                        flip_map[local_i:local_i+1],
                        anom_true[local_i:local_i+1],
                        anom_recon[local_i:local_i+1],
                    )

        sample_offset += B

    # concat accumulated stats
    for s in sigmas:
        if sigma_data[s]['flip_rates']:
            sigma_data[s]['flip_rates'] = torch.cat(sigma_data[s]['flip_rates'], dim=0)
            sigma_data[s]['local_corrs'] = torch.cat(sigma_data[s]['local_corrs'], dim=0)
        else:
            # This rank got no data (unlikely but handle gracefully)
            sigma_data[s]['flip_rates'] = torch.empty(0)
            sigma_data[s]['local_corrs'] = torch.empty(0)

    return sigma_data, sample_offset, mask_cpu


# ═══════════════════════════════════════════════════════════════════════════════
# Distributed gather utilities
# ═══════════════════════════════════════════════════════════════════════════════

def gather_tensor(tensor, args):
    """Gather variable-length tensors from all ranks to rank 0.

    Returns concatenated tensor on rank 0, None on other ranks.
    """
    if not args.distributed:
        return tensor

    device = torch.device(args.device)

    # Communicate local sizes
    local_n = torch.tensor([tensor.shape[0]], dtype=torch.long, device=device)
    all_sizes = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(args.world_size)]
    dist.all_gather(all_sizes, local_n)
    all_sizes = [s.item() for s in all_sizes]
    max_n = max(all_sizes)

    if max_n == 0:
        return tensor

    # Pad to max size for all_gather
    C = tensor.shape[1] if tensor.ndim == 2 else 1
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(1)

    padded = torch.zeros(max_n, C, dtype=tensor.dtype, device=device)
    if tensor.shape[0] > 0:
        padded[:tensor.shape[0]] = tensor.to(device)

    gathered = [torch.zeros_like(padded) for _ in range(args.world_size)]
    dist.all_gather(gathered, padded)

    if is_rank0(args):
        parts = [g[:all_sizes[i]].cpu() for i, g in enumerate(gathered)]
        return torch.cat(parts, dim=0)
    return None


def gather_vis_data(local_vis, args):
    """Gather vis dicts from all ranks to rank 0.

    Each rank has a dict {global_sample_idx: (flip_map, anom_true, anom_recon)}.
    We use a simple approach: serialize keys, send counts, then transfer tensors.
    For simplicity (vis data is small), we use gatherv-style with object gather.
    """
    if not args.distributed:
        return local_vis

    # Use gather_object (available in PyTorch >= 1.8)
    gathered = [None] * args.world_size if is_rank0(args) else None
    dist.gather_object(local_vis, gathered, dst=0)

    if is_rank0(args):
        merged = {}
        for d in gathered:
            merged.update(d)
        return merged
    return {}


# ═══════════════════════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════════════════════

CHANNEL_LABELS = {}
_vars = ['U', 'V', 'T', 'S']
for vi, var in enumerate(_vars):
    for lev in range(25):
        CHANNEL_LABELS[vi * 25 + lev] = f"{var}_L{lev}"
CHANNEL_LABELS[100] = "SSH"


def _ch_label(ch):
    return CHANNEL_LABELS.get(ch, f"CH{ch}")


def plot_per_channel_flip_rate(flip_rate_mean, sigma, save_dir):
    """Bar chart: per-channel mean flip rate."""
    C = len(flip_rate_mean)
    fig, ax = plt.subplots(figsize=(max(14, C * 0.15), 4))
    colors = ['#d62728' if v > 0.3 else '#ff7f0e' if v > 0.15 else '#2ca02c'
              for v in flip_rate_mean]
    ax.bar(range(C), flip_rate_mean, color=colors, width=1.0, edgecolor='none')
    ax.set_xlabel('Channel index')
    ax.set_ylabel('Flip rate')
    ax.set_title(f'Per-channel flip rate (σ={sigma})')
    ax.set_xlim(-0.5, C - 0.5)
    ax.axhline(0.15, color='orange', ls='--', lw=0.8, label='0.15')
    ax.axhline(0.30, color='red',    ls='--', lw=0.8, label='0.30')
    ax.legend(loc='upper right', fontsize=8)

    # 标注 top-5 翻转通道
    top5 = np.argsort(flip_rate_mean)[-5:][::-1]
    for ch in top5:
        if flip_rate_mean[ch] > 0.1:
            ax.annotate(_ch_label(ch), (ch, flip_rate_mean[ch]),
                        fontsize=7, ha='center', va='bottom')

    plt.tight_layout()
    path = os.path.join(save_dir, f'flip_rate_per_channel_sigma{sigma}.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_per_channel_correlation(corr_mean, sigma, save_dir):
    """Bar chart: per-channel local anomaly correlation."""
    C = len(corr_mean)
    fig, ax = plt.subplots(figsize=(max(14, C * 0.15), 4))
    colors = ['#d62728' if v < 0 else '#2ca02c' for v in corr_mean]
    ax.bar(range(C), corr_mean, color=colors, width=1.0, edgecolor='none')
    ax.set_xlabel('Channel index')
    ax.set_ylabel('Local anomaly correlation')
    ax.set_title(f'Per-channel local anomaly Pearson correlation (σ={sigma})')
    ax.set_xlim(-0.5, C - 0.5)
    ax.set_ylim(-1.05, 1.05)
    ax.axhline(0, color='black', ls='-', lw=0.5)

    # 标注负相关通道
    neg_chs = np.where(corr_mean < 0)[0]
    for ch in neg_chs:
        ax.annotate(_ch_label(ch), (ch, corr_mean[ch]),
                    fontsize=7, ha='center', va='top')

    plt.tight_layout()
    path = os.path.join(save_dir, f'local_corr_per_channel_sigma{sigma}.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_flip_maps(flip_map, anom_true, anom_recon, mask_2d, sigma,
                   channels, sample_idx, save_dir):
    """
    For selected channels, show:
      col 0: anom_true
      col 1: anom_recon
      col 2: flip map (red=flipped, blue=consistent, gray=land)
    """
    n = len(channels)
    fig, axes = plt.subplots(n, 3, figsize=(14, 3.5 * n))
    if n == 1:
        axes = axes[np.newaxis]

    for r, ch in enumerate(channels):
        m2d = mask_2d[ch].numpy() if mask_2d.ndim == 3 else mask_2d.numpy()

        at = anom_true[sample_idx, ch].numpy().copy()
        ar = anom_recon[sample_idx, ch].numpy().copy()
        fm = flip_map[sample_idx, ch].numpy().copy()

        at[~m2d.astype(bool)] = np.nan
        ar[~m2d.astype(bool)] = np.nan

        vmax = max(np.nanpercentile(np.abs(at), 99),
                   np.nanpercentile(np.abs(ar), 99), 1e-8)
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

        im0 = axes[r, 0].imshow(at, cmap='RdBu_r', norm=norm)
        axes[r, 0].set_title(f'{_ch_label(ch)} anom_true')
        plt.colorbar(im0, ax=axes[r, 0], fraction=0.046, pad=0.04)

        im1 = axes[r, 1].imshow(ar, cmap='RdBu_r', norm=norm)
        axes[r, 1].set_title(f'{_ch_label(ch)} anom_recon')
        plt.colorbar(im1, ax=axes[r, 1], fraction=0.046, pad=0.04)

        # flip map: 0=consistent(blue), 1=flipped(red), nan=land
        fm_vis = np.full_like(at, np.nan)
        fm_vis[m2d.astype(bool)] = fm[m2d.astype(bool)].astype(float)
        im2 = axes[r, 2].imshow(fm_vis, cmap='RdYlGn_r', vmin=0, vmax=1,
                                interpolation='nearest')
        axes[r, 2].set_title(f'{_ch_label(ch)} flip map')
        plt.colorbar(im2, ax=axes[r, 2], fraction=0.046, pad=0.04)

        for ax in axes[r]:
            ax.set_xticks([])
            ax.set_yticks([])

    plt.suptitle(f'Flip analysis (σ={sigma}, sample={sample_idx})', fontsize=13)
    plt.tight_layout()
    path = os.path.join(save_dir, f'flip_maps_sigma{sigma}_sample{sample_idx}.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate DCAE sign/contrast flip")
    p.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint")
    p.add_argument("--data-name", type=str, default="glorys12_kuroshio_extension")
    p.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    p.add_argument("--sigma", nargs="+", type=float, default=[8, 16, 32],
                   help="Gaussian blur sigma(s) for local background extraction")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-batches", type=int, default=None,
                   help="Limit number of batches (None=all)")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save-dir", type=str, default=None,
                   help="Output directory; default=same dir as checkpoint")
    p.add_argument("--vis-samples", nargs="+", type=int, default=[0],
                   help="Sample indices to visualize (global indices)")
    p.add_argument("--vis-top-n", type=int, default=8,
                   help="Show top-N flipped channels in spatial maps")
    p.add_argument("--local_rank", "--local-rank", type=int, default=0,
                   help="Local rank for distributed evaluation (set by torchrun)")
    return p.parse_args()


def main():
    args = parse_args()
    setup_distributed(args)
    device = torch.device(args.device)

    try:
        if args.save_dir is None:
            args.save_dir = os.path.join(os.path.dirname(args.ckpt), 'flip_eval')
        if is_rank0(args):
            os.makedirs(args.save_dir, exist_ok=True)

        # ── data ──
        dataset_config = get_dataset_config(args.data_name)
        date_map = {
            'train': dataset_config.train_date_range,
            'val':   dataset_config.val_date_range,
            'test':  dataset_config.test_date_range,
        }
        dataset = build_dataset(dataset_config.raw_data_dir, date_map[args.split])
        total_samples = len(dataset)

        sampler = DistributedSampler(
            dataset,
            num_replicas=args.world_size,
            rank=args.rank,
            shuffle=False,
            drop_last=False,
        ) if args.distributed else None

        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            sampler=sampler,
                            num_workers=args.num_workers, pin_memory=False)

        constants = load_constants(dataset_config.constant_dir)
        mu_norm    = constants[0][..., None, None].to(device)
        sigma_norm = constants[1][..., None, None].to(device)
        mask       = constants[-1].float()  # [H, W] or [C, H, W]

        if is_rank0(args):
            print(f"Dataset [{args.split}]: {total_samples} samples, "
                  f"world_size={args.world_size}")
            print(f"Channels: {dataset_config.num_channels}, mask shape: {tuple(mask.shape)}")

        # ── model ──
        model = load_model(args.ckpt, dataset_config.num_channels, device)

        # ── batch-wise inference + evaluation ──
        if is_rank0(args):
            print("Running inference + evaluation (batch-wise)...")

        # For vis_samples: each rank tracks its own local sample offset.
        # DistributedSampler assigns global indices to ranks in a round-robin/chunked way.
        # We pass all vis_samples to each rank; only the rank that has the corresponding
        # global index will actually store vis data.
        # We need to map global vis indices to local indices for each rank.
        if args.distributed:
            # DistributedSampler indices for this rank
            sampler.set_epoch(0)  # deterministic
            rank_indices = list(sampler)
            global_to_local = {g: l for l, g in enumerate(rank_indices)}
            local_vis_samples = [global_to_local[v] for v in args.vis_samples
                                 if v in global_to_local]
        else:
            local_vis_samples = args.vis_samples

        sigma_data, N_local, mask_cpu = run_inference_and_eval(
            model, loader, mu_norm, sigma_norm, mask, device,
            sigmas=args.sigma, vis_samples=local_vis_samples,
            max_batches=args.max_batches,
        )
        C = dataset_config.num_channels

        if is_rank0(args):
            print(f"  Rank 0 processed {N_local} samples")

        # ── gather results across ranks ──
        for sigma in args.sigma:
            sd = sigma_data[sigma]

            # Gather flip_rates and local_corrs
            gathered_fr = gather_tensor(sd['flip_rates'], args)
            gathered_lc = gather_tensor(sd['local_corrs'], args)

            if is_rank0(args):
                # Truncate to actual dataset size (DistributedSampler may pad)
                effective_n = min(gathered_fr.shape[0], total_samples)
                sd['flip_rates'] = gathered_fr[:effective_n]
                sd['local_corrs'] = gathered_lc[:effective_n]

            # Gather vis data
            # Map local vis indices back to global for gather
            if args.distributed:
                global_vis = {}
                for local_idx, data in sd['vis'].items():
                    if local_idx < len(rank_indices):
                        global_idx = rank_indices[local_idx]
                        global_vis[global_idx] = data
                sd['vis'] = gather_vis_data(global_vis, args)
            # else: vis already uses global indices

        # ── Only rank 0 does reporting / plotting / CSV ──
        if not is_rank0(args):
            return

        N = min(total_samples,
                sigma_data[args.sigma[0]]['flip_rates'].shape[0])
        print(f"  Total processed: {N} samples")

        # ── report & plot for each sigma ──
        all_records = []

        for sigma in args.sigma:
            print(f"\n{'='*60}")
            print(f"  σ = {sigma}")
            print(f"{'='*60}")

            sd = sigma_data[sigma]
            flip_rate_np = sd['flip_rates'].numpy()     # [N, C]
            local_corr_np = sd['local_corrs'].numpy()   # [N, C]
            fr_mean = flip_rate_np.mean(axis=0)          # [C]
            lc_mean = local_corr_np.mean(axis=0)         # [C]

            print(f"  Overall flip rate:  {fr_mean.mean():.4f}")
            print(f"  Overall local corr: {lc_mean.mean():.4f}")

            # top flipped channels
            top_idx = np.argsort(fr_mean)[::-1][:args.vis_top_n]
            print(f"  Top-{args.vis_top_n} flipped channels:")
            for ch in top_idx:
                print(f"    {_ch_label(ch):>8s} (ch={ch:3d}): "
                      f"flip_rate={fr_mean[ch]:.4f}, corr={lc_mean[ch]:.4f}")

            # ── save CSV ──
            df_rows = []
            for ch in range(C):
                df_rows.append({
                    'sigma': sigma,
                    'channel': ch,
                    'label': _ch_label(ch),
                    'flip_rate_mean': fr_mean[ch],
                    'flip_rate_std': flip_rate_np[:, ch].std(),
                    'local_corr_mean': lc_mean[ch],
                    'local_corr_std': local_corr_np[:, ch].std(),
                })
            all_records.extend(df_rows)

            # ── plots ──
            plot_per_channel_flip_rate(fr_mean, sigma, args.save_dir)
            plot_per_channel_correlation(lc_mean, sigma, args.save_dir)

            # spatial maps for selected samples
            vis_channels = list(top_idx[:min(args.vis_top_n, 6)])
            for si in args.vis_samples:
                if si in sd['vis']:
                    fm, at, ar = sd['vis'][si]
                    plot_flip_maps(fm, at, ar, mask_cpu,
                                   sigma, vis_channels, 0, args.save_dir)

        # ── save combined CSV ──
        df = pd.DataFrame(all_records)
        csv_path = os.path.join(args.save_dir, 'flip_metrics.csv')
        df.to_csv(csv_path, index=False, float_format='%.6f')
        print(f"\nSaved metrics to {csv_path}")

        # ── summary across sigmas ──
        print(f"\n{'='*60}")
        print("Summary: mean flip rate by sigma")
        print(df.groupby('sigma')['flip_rate_mean'].agg(['mean', 'max']).round(4))
        print(f"{'='*60}")

    finally:
        cleanup_distributed(args)


if __name__ == "__main__":
    main()
