"""
离线保存 DCAE 潜空间表示（支持多卡 DDP）。

扫描原始数据目录中的所有 .pt 文件，编码为 mu（确定性潜变量）并保存。
输出目录结构与原始数据一致：每日一个 {YYYYMMDD}.pt，shape [latent_channels, H', W']。

模型结构参数自动从 checkpoint 中的 args 字段读取，无需手动指定。
同时保存 metadata.json，包含还原回原空间所需的全部信息。

用法（单卡）：
    python eval/save_latent.py \
        --ckpt-path output/.../best_model.pth \
        --output-dir output/latent/...

用法（多卡）：
    torchrun --nproc_per_node 6 eval/save_latent.py \
        --ckpt-path output/.../best_model.pth \
        --output-dir output/latent/...
"""

import sys
import os
import json
import argparse

import torch
import torch.distributed as dist
from torch.cuda.amp import autocast
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from model.dcae import DCAE
from data.dataset import load_constants
from data.data_utils import normalize_fn
from config import get_dataset_config


class RawFileDataset(Dataset):
    """逐文件加载的 Dataset，保留文件名用于输出。"""
    def __init__(self, raw_dir, filenames):
        self.raw_dir = raw_dir
        self.filenames = filenames

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        data = torch.load(os.path.join(self.raw_dir, fname))
        return data, fname


def parse_args():
    parser = argparse.ArgumentParser(description="Save DCAE latent representations offline")
    parser.add_argument("--ckpt-path", type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save latent .pt files")
    parser.add_argument("--data-name", type=str, default=None,
                        help="Dataset name (default: read from checkpoint)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--local_rank", "--local-rank", type=int, default=0)
    return parser.parse_args()


def setup_distributed(args):
    if "LOCAL_RANK" in os.environ:
        args.local_rank = int(os.environ["LOCAL_RANK"])
    args.rank = int(os.environ.get("RANK", 0))
    args.world_size = int(os.environ.get("WORLD_SIZE", 1))
    args.distributed = args.world_size > 1

    if torch.cuda.is_available():
        torch.cuda.set_device(args.local_rank)
        args.device = f"cuda:{args.local_rank}"
    else:
        args.device = "cpu"

    if args.distributed:
        dist.init_process_group(backend="nccl", init_method="env://")


def cleanup_distributed(args):
    if args.distributed and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def is_rank0(args):
    return args.rank == 0


def load_model_from_ckpt(ckpt_path, device):
    """从 checkpoint 加载模型，自动读取训练时的架构参数。"""
    ckpt = torch.load(ckpt_path, map_location=device)

    train_args = ckpt.get('args', {})
    if not train_args:
        raise ValueError(
            f"Checkpoint {ckpt_path} 中没有 'args' 字段，"
            "无法自动推断模型结构。请使用包含 args 的 checkpoint。"
        )

    model = DCAE(
        in_channels=train_args.get('in_channels', 101),
        base_channels=train_args['base_channels'],
        channel_multipliers=train_args['channel_multipliers'],
        latent_channels=train_args['latent_channels'],
        num_res_blocks=train_args['num_res_blocks'],
        attention_resolutions=train_args['attention_resolutions'],
        num_heads=train_args['num_heads'],
    ).to(device)

    state = ckpt.get('model_state_dict', ckpt.get('model', ckpt))
    state = {(k[7:] if k.startswith('module.') else k): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval()

    return model, train_args


def save_metadata(output_dir, train_args, ckpt_path, latent_shape, grid_size):
    """保存 metadata.json，包含还原回原空间所需的全部信息。"""
    metadata = {
        'ckpt_path': os.path.abspath(ckpt_path),
        'latent_shape': list(latent_shape),
        'grid_size': list(grid_size),
        'model': {
            'base_channels': train_args['base_channels'],
            'channel_multipliers': train_args['channel_multipliers'],
            'latent_channels': train_args['latent_channels'],
            'num_res_blocks': train_args['num_res_blocks'],
            'attention_resolutions': train_args['attention_resolutions'],
            'num_heads': train_args['num_heads'],
        },
        'data': {
            'data_name': train_args.get('data_name', 'unknown'),
            'in_channels': train_args.get('in_channels', 101),
        },
        'usage': (
            'To decode: load DCAE with the "model" params above, '
            'load checkpoint from ckpt_path, then call '
            'model.decode(z, target_shape=grid_size).'
        ),
    }

    path = os.path.join(output_dir, 'metadata.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"Saved metadata to {path}")


def main():
    args = parse_args()
    setup_distributed(args)
    device = torch.device(args.device)

    model, train_args = load_model_from_ckpt(args.ckpt_path, device)
    data_name = args.data_name or train_args.get('data_name', 'glorys12_kuroshio_extension')

    if is_rank0(args):
        print(f"Model loaded from {args.ckpt_path}")
        print(f"  Architecture: base_channels={train_args['base_channels']}, "
              f"channel_multipliers={train_args['channel_multipliers']}, "
              f"latent_channels={train_args['latent_channels']}")
        if args.distributed:
            print(f"  Running on {args.world_size} GPUs")

    dataset_config = get_dataset_config(data_name)
    constants = load_constants(dataset_config.constant_dir)
    mu = constants[0][..., None, None].to(device)
    sigma = constants[1][..., None, None].to(device)

    # 获取潜空间 shape（rank 0 only）
    if is_rank0(args):
        with torch.inference_mode():
            dummy = torch.randn(1, dataset_config.num_channels, *dataset_config.grid_size, device=device)
            dummy_z = model.encode(normalize_fn(dummy, mu=mu, sigma=sigma))
            latent_shape = tuple(dummy_z.shape[1:])
            print(f"  Latent shape: {latent_shape}")
            del dummy, dummy_z

        os.makedirs(args.output_dir, exist_ok=True)
        save_metadata(args.output_dir, train_args, args.ckpt_path,
                      latent_shape, dataset_config.grid_size)

    if args.distributed:
        dist.barrier()  # 等 rank 0 创建目录和 metadata

    # 构建 dataset
    raw_dir = dataset_config.raw_data_dir
    all_files = sorted([f for f in os.listdir(raw_dir) if f.endswith('.pt')])

    if is_rank0(args):
        print(f"\nFound {len(all_files)} files in {raw_dir}")

    dataset = RawFileDataset(raw_dir, all_files)

    sampler = DistributedSampler(
        dataset, num_replicas=args.world_size, rank=args.rank, shuffle=False
    ) if args.distributed else None

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    saved = 0
    use_amp = torch.cuda.is_available()
    with torch.inference_mode():
        pbar = tqdm(loader, desc=f"encoding [rank {args.rank}]", disable=not is_rank0(args))
        for raw, fnames in pbar:
            raw = raw.to(device=device, dtype=torch.float32)
            x_norm = normalize_fn(raw, mu=mu, sigma=sigma)
            with autocast(enabled=use_amp):
                z = model.encode(x_norm)
            z = z.float()  # 确保保存为 float32

            for j, fname in enumerate(fnames):
                torch.save(z[j].cpu(), os.path.join(args.output_dir, fname))
                saved += 1

    if args.distributed:
        dist.barrier()

    if is_rank0(args):
        total = len([f for f in os.listdir(args.output_dir) if f.endswith('.pt')])
        print(f"\nSaved {total} latent files to {args.output_dir}")

    cleanup_distributed(args)


if __name__ == "__main__":
    main()
