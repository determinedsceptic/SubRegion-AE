"""
离线保存 DCAE 潜空间表示。

对指定 split 的每个样本，编码为 mu（确定性潜变量）并保存为 .pt 文件。
输出目录结构与原始数据一致：每日一个 {YYYYMMDD}.pt，shape [latent_channels, H', W']。

用法：
    python eval/save_latent.py \
        --data-name glorys12_kuroshio_extension \
        --ckpt-path output/glorys12_kuroshio_extension/dcae2_.../best_model.pth \
        --splits train val test \
        --output-dir output/latent/glorys12_kuroshio_extension/dcae2_...
"""

import sys
import os
import argparse

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from model.dcae import DCAE
from data.dataset import load_constants
from data.data_utils import normalize_fn
from config import get_dataset_config


def parse_args():
    parser = argparse.ArgumentParser(description="Save DCAE latent representations offline")
    parser.add_argument("--data-name", type=str, default="glorys12_kuroshio_extension")
    parser.add_argument("--ckpt-path", type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save latent .pt files")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"],
                        help="Which splits to process (default: train val test)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # 模型结构参数（需要与训练一致）
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--channel-multipliers", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--latent-channels", type=int, default=16)
    parser.add_argument("--num-res-blocks", type=int, default=2)
    parser.add_argument("--attention-resolutions", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--num-heads", type=int, default=8)

    return parser.parse_args()


def load_model(args, in_channels, device):
    model = DCAE(
        in_channels=in_channels,
        base_channels=args.base_channels,
        channel_multipliers=args.channel_multipliers,
        latent_channels=args.latent_channels,
        num_res_blocks=args.num_res_blocks,
        attention_resolutions=args.attention_resolutions,
        num_heads=args.num_heads,
    ).to(device)

    ckpt = torch.load(args.ckpt_path, map_location=device)
    state = ckpt.get('model_state_dict', ckpt.get('model', ckpt))
    state = {(k[7:] if k.startswith('module.') else k): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def main():
    args = parse_args()
    device = torch.device(args.device)

    dataset_config = get_dataset_config(args.data_name)
    constants = load_constants(dataset_config.constant_dir)
    mu = constants[0][..., None, None].to(device)
    sigma = constants[1][..., None, None].to(device)

    model = load_model(args, dataset_config.num_channels, device)
    print(f"Model loaded from {args.ckpt_path}")

    # 打印一次潜空间 shape
    with torch.inference_mode():
        dummy = torch.randn(1, dataset_config.num_channels, *dataset_config.grid_size, device=device)
        dummy_z = model.encode(normalize_fn(dummy, mu=mu, sigma=sigma))
        print(f"Latent shape: {tuple(dummy_z.shape[1:])} (per sample)")
        del dummy, dummy_z

    split_ranges = {
        'train': dataset_config.train_date_range,
        'val':   dataset_config.val_date_range,
        'test':  dataset_config.test_date_range,
    }

    for split in args.splits:
        if split not in split_ranges:
            print(f"Warning: unknown split '{split}', skipping")
            continue

        date_range = split_ranges[split]
        dates = pd.date_range(start=date_range[0], end=date_range[1], freq="D")
        files = [os.path.join(dataset_config.raw_data_dir, f"{d.strftime('%Y%m%d')}.pt") for d in dates]

        split_dir = os.path.join(args.output_dir, split)
        os.makedirs(split_dir, exist_ok=True)

        print(f"\n[{split}] {len(dates)} samples → {split_dir}")

        # 逐 batch 处理，但按单样本保存
        saved = 0
        with torch.inference_mode():
            for i in tqdm(range(0, len(files), args.batch_size), desc=split):
                batch_files = files[i:i + args.batch_size]
                batch_dates = dates[i:i + args.batch_size]

                # 加载 batch
                tensors = []
                valid_dates = []
                for f, d in zip(batch_files, batch_dates):
                    try:
                        tensors.append(torch.load(f))
                        valid_dates.append(d)
                    except (FileNotFoundError, RuntimeError) as e:
                        print(f"  Skip {f}: {e}")

                if not tensors:
                    continue

                raw = torch.stack(tensors).to(device=device, dtype=torch.float32)
                x_norm = normalize_fn(raw, mu=mu, sigma=sigma)
                z = model.encode(x_norm)  # [B, latent_channels, H', W']

                # 逐样本保存
                for j, d in enumerate(valid_dates):
                    fname = f"{d.strftime('%Y%m%d')}.pt"
                    torch.save(z[j].cpu(), os.path.join(split_dir, fname))
                    saved += 1

        print(f"  Saved {saved} latent files")

    print("\nDone.")


if __name__ == "__main__":
    main()
