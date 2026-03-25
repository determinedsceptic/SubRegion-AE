"""
eval_multi_seed.py  —  多种子训练结果对比分析

功能：
  1. 扫描多个 seed checkpoint 目录，对每个运行 flip 评估
  2. 输出 per-channel flip_rate 分布（mean ± std across seeds）
  3. 识别结构性翻转通道 vs 随机翻转通道
  4. 通道异常方差 vs flip_rate 散点图 + Pearson 相关系数
  5. 训练过程中 flip_rate 随 epoch 变化曲线（从 TensorBoard 读取）

用法：
  # 单卡
  python3 eval/eval_multi_seed.py \
      --run-dirs output/data/tag_seed42 output/data/tag_seed123 ... \
      --sigma 16.0 --split val --save-dir output/multi_seed_analysis
  # 多卡
  torchrun --nproc_per_node 6 eval/eval_multi_seed.py \
      --run-dirs output/data/tag_seed42 output/data/tag_seed123 ... \
      --sigma 16.0 --split val --save-dir output/multi_seed_analysis
"""

import os
import sys
import argparse

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

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


def print_rank0(args, *items, **kwargs):
    if is_rank0(args):
        print(*items, **kwargs)


def gather_tensor(tensor, args):
    """Gather variable-length [N, C] tensors from all ranks, return concat on rank 0."""
    if not args.distributed:
        return tensor

    device = torch.device(args.device)

    local_n = torch.tensor([tensor.shape[0]], dtype=torch.long, device=device)
    all_sizes = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(args.world_size)]
    dist.all_gather(all_sizes, local_n)
    all_sizes = [s.item() for s in all_sizes]
    max_n = max(all_sizes)

    if max_n == 0:
        return tensor

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


# ═══════════════════════════════════════════════════════════════════════════════
# Channel labels
# ═══════════════════════════════════════════════════════════════════════════════

CHANNEL_LABELS = {}
_vars = ['U', 'V', 'T', 'S']
for vi, var in enumerate(_vars):
    for lev in range(25):
        CHANNEL_LABELS[vi * 25 + lev] = f"{var}_L{lev}"
CHANNEL_LABELS[100] = "SSH"


def _ch_label(ch):
    return CHANNEL_LABELS.get(ch, f"CH{ch}")


# ═══════════════════════════════════════════════════════════════════════════════
# Model loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_model(ckpt_path, in_channels, device):
    ckpt = torch.load(ckpt_path, map_location=device)
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
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# Inference (distributed-aware, batch-wise flip computation)
# ═══════════════════════════════════════════════════════════════════════════════

@torch.inference_mode()
def run_inference_and_eval(model, loader, mu, sigma_norm, mask, device,
                           sigma, max_batches=None):
    """
    Batch-wise inference + flip metric on this rank's data subset.
    Returns flip_rate [N_local, C], local_corr [N_local, C], and
    optionally x_all [N_local, C, H, W] for variance analysis.
    """
    mask_cpu = mask.cpu()
    flip_rates = []
    local_corrs = []
    xs_for_ref = []
    collect_x_ref = True  # only first seed needs x_ref

    for step, batch in enumerate(loader):
        if max_batches is not None and step >= max_batches:
            break
        raw = batch.to(device=device, dtype=torch.float32)
        x_norm = normalize_fn(raw, mu=mu, sigma=sigma_norm)
        recon_norm = model(x_norm)

        x_cpu = x_norm.cpu()
        recon_cpu = recon_norm.cpu()

        _, flip_rate, local_corr, _, _ = compute_flip_metrics(
            x_cpu, recon_cpu, mask_cpu, sigma)
        flip_rates.append(flip_rate)
        local_corrs.append(local_corr)

        if collect_x_ref:
            xs_for_ref.append(x_cpu)

    if flip_rates:
        flip_rates = torch.cat(flip_rates, dim=0)
        local_corrs = torch.cat(local_corrs, dim=0)
    else:
        flip_rates = torch.empty(0)
        local_corrs = torch.empty(0)

    x_ref = torch.cat(xs_for_ref, dim=0) if xs_for_ref else None
    return flip_rates, local_corrs, x_ref


# ═══════════════════════════════════════════════════════════════════════════════
# TensorBoard flip curve extraction
# ═══════════════════════════════════════════════════════════════════════════════

def load_tb_flip_curves(run_dir):
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        print("  Warning: tensorboard not installed, skipping TB curve extraction.")
        return None

    tb_dir = os.path.join(run_dir, 'tb')
    if not os.path.isdir(tb_dir):
        return None

    ea = EventAccumulator(tb_dir)
    ea.Reload()

    result = {}
    for tag in ea.Tags().get('scalars', []):
        if tag.startswith('flip/'):
            events = ea.Scalars(tag)
            result[tag] = [(e.step, e.value) for e in events]
    return result if result else None


# ═══════════════════════════════════════════════════════════════════════════════
# Analysis & Plotting
# ═══════════════════════════════════════════════════════════════════════════════

def plot_flip_rate_distribution(all_flip_rates, save_dir):
    seed_labels = sorted(all_flip_rates.keys())
    rates = np.stack([all_flip_rates[s] for s in seed_labels], axis=0)
    mean = rates.mean(axis=0)
    std = rates.std(axis=0)
    C = mean.shape[0]

    fig, ax = plt.subplots(figsize=(max(14, C * 0.15), 5))
    x = np.arange(C)
    ax.bar(x, mean, yerr=std, color='steelblue', width=1.0, edgecolor='none',
           capsize=1, ecolor='gray', alpha=0.8)
    ax.set_xlabel('Channel index')
    ax.set_ylabel('Flip rate (mean ± std across seeds)')
    ax.set_title(f'Per-channel flip rate distribution ({len(seed_labels)} seeds)')
    ax.set_xlim(-0.5, C - 0.5)
    ax.axhline(0.15, color='orange', ls='--', lw=0.8, label='0.15')
    ax.axhline(0.30, color='red', ls='--', lw=0.8, label='0.30')
    ax.legend(loc='upper right', fontsize=8)

    plt.tight_layout()
    path = os.path.join(save_dir, 'multi_seed_flip_rate_distribution.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def classify_channels(all_flip_rates, threshold=0.15):
    seed_labels = sorted(all_flip_rates.keys())
    rates = np.stack([all_flip_rates[s] for s in seed_labels], axis=0)
    C = rates.shape[1]

    rows = []
    for ch in range(C):
        ch_rates = rates[:, ch]
        n_flipped = (ch_rates > threshold).sum()
        if n_flipped == len(seed_labels):
            category = 'structural'
        elif n_flipped == 0:
            category = 'stable'
        else:
            category = 'random'
        rows.append({
            'channel': ch,
            'label': _ch_label(ch),
            'mean_flip_rate': ch_rates.mean(),
            'std_flip_rate': ch_rates.std(),
            'n_seeds_flipped': int(n_flipped),
            'total_seeds': len(seed_labels),
            'category': category,
        })
    return pd.DataFrame(rows)


def plot_variance_vs_flip(x_all, mask, all_flip_rates, save_dir):
    m = mask.float()
    while m.ndim < x_all.ndim:
        m = m.unsqueeze(0)
    m = m.expand_as(x_all)

    B, C, H, W = x_all.shape
    x_flat = (x_all * m).view(B, C, -1)
    n_valid = m.view(B, C, -1).sum(dim=2).clamp_min(1)
    x_mean = x_flat.sum(dim=2) / n_valid
    x_var = ((x_flat - x_mean[:, :, None]).pow(2) * m.view(B, C, -1)).sum(dim=2) / n_valid
    ch_variance = x_var.mean(dim=0).numpy()

    seed_labels = sorted(all_flip_rates.keys())
    rates = np.stack([all_flip_rates[s] for s in seed_labels], axis=0)
    mean_flip = rates.mean(axis=0)

    r, p = stats.pearsonr(ch_variance, mean_flip)

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = []
    for ch in range(C):
        if ch < 25:
            colors.append('tab:blue')
        elif ch < 50:
            colors.append('tab:orange')
        elif ch < 75:
            colors.append('tab:green')
        elif ch < 100:
            colors.append('tab:red')
        else:
            colors.append('tab:purple')

    ax.scatter(ch_variance, mean_flip, c=colors, s=20, alpha=0.7)
    ax.set_xlabel('Channel variance (normalized data)')
    ax.set_ylabel('Mean flip rate across seeds')
    ax.set_title(f'Channel variance vs flip rate\nPearson r={r:.4f}, p={p:.4e}')

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='tab:blue', label='U', markersize=8),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='tab:orange', label='V', markersize=8),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='tab:green', label='T', markersize=8),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='tab:red', label='S', markersize=8),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='tab:purple', label='SSH', markersize=8),
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    path = os.path.join(save_dir, 'variance_vs_flip_rate.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")
    print(f"  Pearson r={r:.4f}, p={p:.4e}")


def plot_flip_over_training(all_tb_curves, save_dir):
    fig, ax = plt.subplots(figsize=(10, 5))
    has_data = False

    for seed_label in sorted(all_tb_curves.keys()):
        curves = all_tb_curves[seed_label]
        if curves is None or 'flip/overall' not in curves:
            continue
        data = curves['flip/overall']
        steps, values = zip(*data)
        ax.plot(steps, values, label=seed_label, alpha=0.8)
        has_data = True

    if not has_data:
        plt.close(fig)
        print("  No flip/overall TensorBoard data found, skipping training curve plot.")
        return

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Overall flip rate')
    ax.set_title('Flip rate over training (per seed)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, 'flip_rate_over_training.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Multi-seed DCAE flip analysis")
    p.add_argument("--run-dirs", nargs="+", required=True,
                   help="Checkpoint directories for each seed run")
    p.add_argument("--data-name", type=str, default="glorys12_kuroshio_extension")
    p.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    p.add_argument("--sigma", type=float, default=16.0,
                   help="Gaussian blur sigma for flip evaluation")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-batches", type=int, default=None,
                   help="Limit batches per seed (None=all)")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save-dir", type=str, default="output/multi_seed_analysis",
                   help="Output directory for analysis results")
    p.add_argument("--ckpt-name", type=str, default="best_model.pth",
                   help="Checkpoint filename to load from each run dir")
    p.add_argument("--flip-threshold", type=float, default=0.15,
                   help="Threshold for classifying a channel as 'flipped'")
    p.add_argument("--local_rank", "--local-rank", type=int, default=0,
                   help="Local rank for distributed evaluation (set by torchrun)")
    return p.parse_args()


def main():
    args = parse_args()
    setup_distributed(args)
    device = torch.device(args.device)

    try:
        if is_rank0(args):
            os.makedirs(args.save_dir, exist_ok=True)

        # ── data (shared across all seeds) ──
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
        mask       = constants[-1].float()

        print_rank0(args, f"Dataset [{args.split}]: {total_samples} samples, "
                          f"world_size={args.world_size}")
        print_rank0(args, f"Sigma: {args.sigma}")
        print_rank0(args, f"Run dirs: {len(args.run_dirs)}")

        # ── evaluate each seed ──
        all_flip_rates = {}
        all_tb_curves = {}
        x_ref = None

        for run_dir in args.run_dirs:
            seed_label = os.path.basename(run_dir.rstrip('/'))
            ckpt_path = os.path.join(run_dir, args.ckpt_name)

            if not os.path.exists(ckpt_path):
                print_rank0(args, f"\n  WARNING: {ckpt_path} not found, skipping.")
                continue

            print_rank0(args, f"\n{'='*60}")
            print_rank0(args, f"  {seed_label}: {ckpt_path}")
            print_rank0(args, f"{'='*60}")

            model = load_model(ckpt_path, dataset_config.num_channels, device)

            is_first_seed = (x_ref is None)
            flip_rates_local, local_corrs_local, x_ref_local = \
                run_inference_and_eval(
                    model, loader, mu_norm, sigma_norm, mask, device,
                    sigma=args.sigma, max_batches=args.max_batches,
                )

            # Gather across ranks
            flip_rates_all = gather_tensor(flip_rates_local, args)

            if is_rank0(args):
                effective_n = min(flip_rates_all.shape[0], total_samples)
                flip_rates_all = flip_rates_all[:effective_n]

                fr_mean = flip_rates_all.mean(dim=0).numpy()
                all_flip_rates[seed_label] = fr_mean
                print(f"  Overall flip rate: {fr_mean.mean():.4f} "
                      f"({effective_n} samples)")

            # Save x_ref from first seed (only rank 0 needs it for variance plot)
            if is_first_seed and is_rank0(args) and x_ref_local is not None:
                # For variance analysis, we only need a moderate amount of data
                # Gather x_ref would be too expensive; rank 0's local subset is enough
                x_ref = x_ref_local

            # Load TB curves (only rank 0)
            if is_rank0(args):
                tb_curves = load_tb_flip_curves(run_dir)
                all_tb_curves[seed_label] = tb_curves

            del model, flip_rates_local, local_corrs_local, x_ref_local
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Sync all ranks before loading next model
            if args.distributed:
                dist.barrier()

        # ── Analysis (rank 0 only) ──
        if not is_rank0(args):
            return

        if not all_flip_rates:
            print("No valid runs found. Exiting.")
            return

        print(f"\n{'='*60}")
        print("  Multi-seed analysis")
        print(f"{'='*60}")

        plot_flip_rate_distribution(all_flip_rates, args.save_dir)

        df_class = classify_channels(all_flip_rates, threshold=args.flip_threshold)
        csv_path = os.path.join(args.save_dir, 'channel_classification.csv')
        df_class.to_csv(csv_path, index=False, float_format='%.6f')
        print(f"\n  Channel classification saved to {csv_path}")

        n_structural = (df_class['category'] == 'structural').sum()
        n_random = (df_class['category'] == 'random').sum()
        n_stable = (df_class['category'] == 'stable').sum()
        print(f"  Structural: {n_structural}, Random: {n_random}, Stable: {n_stable}")

        structural = df_class[df_class['category'] == 'structural']
        if len(structural) > 0:
            print(f"\n  Structural flip channels (flipped in ALL seeds):")
            for _, row in structural.iterrows():
                print(f"    {row['label']:>8s} (ch={row['channel']:3d}): "
                      f"mean={row['mean_flip_rate']:.4f} ± {row['std_flip_rate']:.4f}")

        random_chs = df_class[df_class['category'] == 'random']
        if len(random_chs) > 0:
            print(f"\n  Random flip channels (flipped in SOME seeds):")
            for _, row in random_chs.iterrows():
                print(f"    {row['label']:>8s} (ch={row['channel']:3d}): "
                      f"mean={row['mean_flip_rate']:.4f} ± {row['std_flip_rate']:.4f}, "
                      f"flipped in {row['n_seeds_flipped']}/{row['total_seeds']} seeds")

        if x_ref is not None:
            plot_variance_vs_flip(x_ref, mask.cpu(), all_flip_rates, args.save_dir)

        plot_flip_over_training(all_tb_curves, args.save_dir)

        rows = []
        for seed_label in sorted(all_flip_rates.keys()):
            fr = all_flip_rates[seed_label]
            for ch in range(len(fr)):
                rows.append({
                    'seed': seed_label,
                    'channel': ch,
                    'label': _ch_label(ch),
                    'flip_rate': fr[ch],
                })
        df_all = pd.DataFrame(rows)
        all_csv_path = os.path.join(args.save_dir, 'all_seeds_flip_rates.csv')
        df_all.to_csv(all_csv_path, index=False, float_format='%.6f')
        print(f"\n  All seeds flip rates saved to {all_csv_path}")

        print(f"\nDone. Results in {args.save_dir}")

    finally:
        cleanup_distributed(args)


if __name__ == "__main__":
    main()
