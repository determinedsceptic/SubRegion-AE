"""
fix_flip.py  —  诊断并修复 DCAE 输出通道的符号翻转

原理：
  decoder 最后一层是 Conv2d(base_ch, 101, 1)，纯线性映射。
  对于翻转通道 c（Pearson corr < 0），直接取反 weight[c] 和 bias[c]
  即可精确翻转该通道输出，无需重训练。

用法：
  python3 eval/fix_flip.py --ckpt /path/to/best_model.pth [--corr-threshold -0.3] [--split val]

输出：
  - 修复后的 checkpoint: {ckpt_dir}/best_model_fixed.pth
  - 修复前后的 per-channel correlation 对比
"""

import os
import sys
import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from model.dcae import DCAE
from data.dataset import build_dataset, load_constants
from data.data_utils import normalize_fn
from config import get_dataset_config


CHANNEL_LABELS = {}
for vi, var in enumerate(['U', 'V', 'T', 'S']):
    for lev in range(25):
        CHANNEL_LABELS[vi * 25 + lev] = f"{var}_L{lev}"
CHANNEL_LABELS[100] = "SSH"


def load_model(ckpt_path, in_channels, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    args_dict = ckpt.get('args', {})

    model = DCAE(
        in_channels=in_channels,
        base_channels=args_dict.get('base_channels', 64),
        channel_multipliers=args_dict.get('channel_multipliers', [1, 2, 4, 8]),
        latent_channels=args_dict.get('latent_channels', 16),
        num_res_blocks=args_dict.get('num_res_blocks', 2),
        attention_resolutions=args_dict.get('attention_resolutions', [1, 2]),
        num_heads=args_dict.get('num_heads', 8),
    ).to(device)

    state = ckpt.get('model_state_dict', ckpt.get('model', ckpt))
    state = {(k[7:] if k.startswith('module.') else k): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, ckpt


@torch.inference_mode()
def compute_channel_corr(model, loader, mu, sigma_norm, mask, device, max_batches=None):
    """Compute per-channel Pearson correlation between input and reconstruction.

    Returns: corr_mean [C], corr_all [N, C]
    """
    mask_expanded = (mask > 0).float()  # [C, H, W] or [H, W]
    all_corrs = []

    for step, batch in enumerate(loader):
        if max_batches is not None and step >= max_batches:
            break
        raw = batch.to(device=device, dtype=torch.float32)
        x = normalize_fn(raw, mu=mu, sigma=sigma_norm)
        recon = model(x)

        B, C, H, W = x.shape
        x_clean = torch.nan_to_num(x, nan=0.0)

        if mask_expanded.ndim < x_clean.ndim:
            m = mask_expanded.unsqueeze(0).expand_as(x_clean)
        else:
            m = mask_expanded.unsqueeze(0).expand_as(x_clean)
        valid = m  # [B, C, H, W]

        x_flat = x_clean.reshape(B, C, -1)
        r_flat = recon.reshape(B, C, -1)
        v_flat = valid.reshape(B, C, -1)

        n = v_flat.sum(dim=2).clamp_min(1.0)
        x_mean = (x_flat * v_flat).sum(dim=2) / n
        r_mean = (r_flat * v_flat).sum(dim=2) / n

        x_c = (x_flat - x_mean.unsqueeze(2)) * v_flat
        r_c = (r_flat - r_mean.unsqueeze(2)) * v_flat

        cov = (x_c * r_c).sum(dim=2) / n
        x_std = (x_c.pow(2).sum(dim=2) / n).sqrt().clamp_min(1e-8)
        r_std = (r_c.pow(2).sum(dim=2) / n).sqrt().clamp_min(1e-8)

        corr = cov / (x_std * r_std)  # [B, C]
        all_corrs.append(corr.cpu())

    all_corrs = torch.cat(all_corrs, dim=0)  # [N, C]
    return all_corrs.mean(dim=0), all_corrs


def flip_output_channels(model, channels_to_flip):
    """Negate weight and bias of final Conv2d for specified output channels."""
    final_conv = model.final[1]  # nn.Conv2d(base_ch, in_channels, 1)
    assert isinstance(final_conv, torch.nn.Conv2d), \
        f"Expected Conv2d, got {type(final_conv)}"

    for ch in channels_to_flip:
        final_conv.weight.data[ch] *= -1
        if final_conv.bias is not None:
            final_conv.bias.data[ch] *= -1


def parse_args():
    p = argparse.ArgumentParser(description="Diagnose and fix DCAE channel sign flips")
    p.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint")
    p.add_argument("--data-name", type=str, default="glorys12_kuroshio_extension")
    p.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    p.add_argument("--corr-threshold", type=float, default=-0.3,
                   help="Channels with mean correlation below this are flipped (default: -0.3)")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-batches", type=int, default=None,
                   help="Limit batches for diagnosis (None=all)")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output", type=str, default=None,
                   help="Output checkpoint path; default={ckpt_dir}/{ckpt_name}_fixed.pth")
    p.add_argument("--dry-run", action="store_true",
                   help="Only diagnose, don't fix or save")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    # ── data ──
    dataset_config = get_dataset_config(args.data_name)
    date_map = {
        'train': dataset_config.train_date_range,
        'val':   dataset_config.val_date_range,
        'test':  dataset_config.test_date_range,
    }
    dataset = build_dataset(dataset_config.raw_data_dir, date_map[args.split])
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=False)

    constants = load_constants(dataset_config.constant_dir)
    mu = constants[0][..., None, None].to(device)
    sigma_norm = constants[1][..., None, None].to(device)
    mask = constants[-1].float().to(device)

    # ── load model ──
    model, ckpt = load_model(args.ckpt, dataset_config.num_channels, device)

    # ── Step 1: diagnose ──
    print("=" * 60)
    print("Step 1: Computing per-channel correlation (before fix)")
    print("=" * 60)
    corr_mean_before, _ = compute_channel_corr(
        model, loader, mu, sigma_norm, mask, device, args.max_batches)

    C = corr_mean_before.shape[0]
    flipped_channels = []
    for ch in range(C):
        label = CHANNEL_LABELS.get(ch, f"CH{ch}")
        corr_val = corr_mean_before[ch].item()
        if corr_val < args.corr_threshold:
            flipped_channels.append(ch)
            print(f"  [FLIP] {label:>8s} (ch={ch:3d}): corr = {corr_val:+.4f}")

    print(f"\nFound {len(flipped_channels)} flipped channels "
          f"(threshold={args.corr_threshold})")

    if not flipped_channels:
        print("No channels to fix. Exiting.")
        return

    if args.dry_run:
        print("\n--dry-run: skipping fix and save.")
        return

    # ── Step 2: fix ──
    print("\n" + "=" * 60)
    print("Step 2: Flipping final Conv2d weights for affected channels")
    print("=" * 60)
    flip_output_channels(model, flipped_channels)
    print(f"  Flipped {len(flipped_channels)} channels: {flipped_channels}")

    # ── Step 3: verify ──
    print("\n" + "=" * 60)
    print("Step 3: Verifying per-channel correlation (after fix)")
    print("=" * 60)
    corr_mean_after, _ = compute_channel_corr(
        model, loader, mu, sigma_norm, mask, device, args.max_batches)

    improved = 0
    for ch in flipped_channels:
        label = CHANNEL_LABELS.get(ch, f"CH{ch}")
        before = corr_mean_before[ch].item()
        after = corr_mean_after[ch].item()
        delta = after - before
        status = "OK" if after > 0 else "WARN"
        print(f"  [{status}] {label:>8s} (ch={ch:3d}): "
              f"{before:+.4f} → {after:+.4f}  (Δ={delta:+.4f})")
        if after > before:
            improved += 1

    overall_before = corr_mean_before.mean().item()
    overall_after = corr_mean_after.mean().item()
    print(f"\n  Overall mean corr: {overall_before:+.4f} → {overall_after:+.4f}")
    print(f"  Improved: {improved}/{len(flipped_channels)} channels")

    # ── Step 4: save ──
    if args.output is None:
        base, ext = os.path.splitext(args.ckpt)
        output_path = f"{base}_fixed{ext}"
    else:
        output_path = args.output

    # Update the checkpoint with fixed weights
    ckpt['model_state_dict'] = model.state_dict()
    ckpt['flipped_channels'] = flipped_channels
    torch.save(ckpt, output_path)
    print(f"\nSaved fixed checkpoint to {output_path}")


if __name__ == "__main__":
    main()
