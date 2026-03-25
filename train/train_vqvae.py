import sys
import os
import argparse
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm


current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, current_dir)

from model.vqvae import VQVAE
from data.dataset import build_dataset, load_constants
from data.data_utils import normalize_fn
from config import get_dataset_config
from utils.checkpoint import save_checkpoint, auto_resume_helper


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


def reduce_mean(value: torch.Tensor, args):
    if not args.distributed:
        return value
    reduced = value.detach().clone()
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    reduced /= args.world_size
    return reduced


def parse_args():
    parser = argparse.ArgumentParser(description="Train VQ-VAE for Ocean Data")
    parser.add_argument("--data-name", type=str, default="glorys12_kuroshio_extension_normalized_by_uvtsh", help="Dataset name")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--min-lr", type=float, default=1e-6, help="Minimum learning rate for cosine annealing")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--local_rank", "--local-rank", type=int, default=0, help="Local rank for distributed training")
    parser.add_argument("--tag", type=str, default="vqvae_training", help="Experiment tag")

    parser.add_argument("--hidden-dims", nargs="+", type=int, default=[64, 128, 256], help="Encoder/decoder hidden dims")
    parser.add_argument("--embedding-dim", type=int, default=128, help="Latent embedding dim")
    parser.add_argument("--num-embeddings", type=int, default=2048, help="Codebook size")
    parser.add_argument("--quantizer", type=str, default="ema", choices=["ema", "standard"], help="Quantizer type")
    parser.add_argument("--ema-decay", type=float, default=0.99, help="EMA decay if quantizer=ema")
    parser.add_argument("--commitment-cost", type=float, default=0.25, help="Commitment cost beta")
    parser.add_argument("--vq-weight", type=float, default=1.0, help="Global weight for VQ loss")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping max norm; <=0 disables")

    parser.add_argument("--experiment-version", type=str, default=None)
    parser.add_argument("--precond", type=str, default="edm")
    parser.add_argument("--arch", type=str, default="adm")
    parser.add_argument("--normalize-method", type=str, default="z-score")
    parser.add_argument("--condition", nargs='+', default=[])
    parser.add_argument("--target", type=str, default="raw")
    parser.add_argument("--loss", type=str, default="mse")
    parser.add_argument("--hyper-parameter-version", type=str, default="v1")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader num_workers per rank")
    parser.add_argument("--auto-resume", action="store_true")
    parser.add_argument("--resume-path", type=str, default=None)

    args = parser.parse_args()
    args.save_dir = os.path.join("output", args.data_name, args.tag)
    return args


def masked_mse_loss(recon_x, x, mask):
    x = torch.nan_to_num(x, nan=0.0)
    loss = F.mse_loss(recon_x, x, reduction='none')

    if mask.ndim < loss.ndim:
        while mask.ndim < loss.ndim:
            mask = mask.unsqueeze(0)

    valid_mask = (mask > 0).to(loss.dtype)
    loss = loss * valid_mask

    num_valid = valid_mask.sum()
    if num_valid > 0:
        return loss.sum() / num_valid
    return loss.sum()


def preprocess_raw_data(batch, normalization_stats, device):
    if isinstance(batch, dict):
        x = batch['raw_data'].to(device=device)
    else:
        x = batch.to(device=device)

    x = normalize_fn(x, **normalization_stats['ocean'])
    x = x.to(dtype=torch.float32)
    return x


def _extract_state_dict(checkpoint):
    for key in ['model_state_dict', 'model', 'net']:
        if key in checkpoint and checkpoint[key] is not None:
            return checkpoint[key]
    return None


def _extract_optimizer_state(checkpoint):
    for key in ['optimizer_state_dict', 'optimizer_state', 'optimizer']:
        if key in checkpoint and checkpoint[key] is not None:
            return checkpoint[key]
    return None


def _extract_scheduler_state(checkpoint):
    for key in ['scheduler_state_dict', 'scheduler_state', 'scheduler']:
        if key in checkpoint and checkpoint[key] is not None:
            return checkpoint[key]
    return None


def try_resume(args, model_without_ddp, optimizer, scheduler):
    resume_path = args.resume_path
    if args.auto_resume and not resume_path:
        resume_path = auto_resume_helper(args.save_dir)
        if resume_path is not None:
            print_rank0(args, f"Auto-resume enabled. Found checkpoint: {resume_path}")

    if not resume_path:
        return 0, float('inf')

    if not os.path.exists(resume_path):
        print_rank0(args, f"Resume path does not exist: {resume_path}")
        return 0, float('inf')

    print_rank0(args, f"Loading checkpoint from {resume_path}")
    checkpoint = torch.load(resume_path, map_location='cpu')

    model_state = _extract_state_dict(checkpoint)
    if model_state is None:
        print_rank0(args, "Warning: no model state in checkpoint, start from scratch.")
        return 0, float('inf')

    normalized_state = {}
    for key, value in model_state.items():
        if key.startswith('module.'):
            normalized_state[key[7:]] = value
        else:
            normalized_state[key] = value
    model_without_ddp.load_state_dict(normalized_state)

    optimizer_state = _extract_optimizer_state(checkpoint)
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)

    scheduler_state = _extract_scheduler_state(checkpoint)
    if scheduler_state is not None:
        scheduler.load_state_dict(scheduler_state)

    start_epoch = int(checkpoint.get('epoch', -1)) + 1
    best_val_loss = float(checkpoint.get('loss', checkpoint.get('min_loss', float('inf'))))

    if scheduler_state is None and start_epoch > 0:
        for _ in range(start_epoch):
            scheduler.step()

    print_rank0(args, f"Resumed at epoch {start_epoch}, best_val_loss={best_val_loss:.6f}")
    return start_epoch, best_val_loss


def train(args):
    setup_distributed(args)
    if is_rank0(args):
        os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device(args.device)
    print_rank0(args, f"Using device: {device}, rank/world_size={args.rank}/{args.world_size}")

    dataset_config = get_dataset_config(args.data_name)

    print_rank0(args, f"Loading datasets for {args.data_name}...")
    train_dataset = build_dataset(dataset_config.raw_data_dir, dataset_config.train_date_range)
    val_dataset   = build_dataset(dataset_config.raw_data_dir, dataset_config.val_date_range)

    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=args.world_size,
        rank=args.rank,
        shuffle=True,
        drop_last=True,
    ) if args.distributed else None

    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=args.world_size,
        rank=args.rank,
        shuffle=False,
        drop_last=False,
    ) if args.distributed else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    print_rank0(args, "Loading constants...")
    try:
        constants = load_constants(dataset_config.constant_dir)
        normed_ocean_mean = constants[0].to(device)
        normed_ocean_std = constants[1].to(device)
        mask = constants[-1].to(device).float()
        ocean_mu = normed_ocean_mean[..., None, None]
        ocean_sigma = normed_ocean_std[..., None, None]
        normalization_stats = {
            'ocean': {'mu': ocean_mu, 'sigma': ocean_sigma}
        }
        print_rank0(args, f"Mask loaded successfully. Shape: {mask.shape}")
    except Exception as e:
        print_rank0(args, f"Warning: Could not load constants/mask: {e}")
        print_rank0(args, "Using ones mask (all pixels valid).")
        mask = torch.ones(dataset_config.grid_size).to(device)
        normalization_stats = {
            'ocean': {'mu': torch.tensor(0.0, device=device), 'sigma': torch.tensor(1.0, device=device)}
        }

    in_channels = dataset_config.num_channels
    print_rank0(
        args,
        f"Initializing VQ-VAE with in_channels={in_channels}, embedding_dim={args.embedding_dim}, codebook={args.num_embeddings}",
    )

    model = VQVAE(
        in_channels=in_channels,
        hidden_dims=args.hidden_dims,
        embedding_dim=args.embedding_dim,
        num_embeddings=args.num_embeddings,
        commitment_cost=args.commitment_cost,
        quantizer=args.quantizer,
        ema_decay=args.ema_decay,
    ).to(device)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

    model_without_ddp = model.module if hasattr(model, "module") else model
    optimizer = torch.optim.AdamW(model_without_ddp.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.min_lr,
    )

    start_epoch, best_val_loss = try_resume(args, model_without_ddp, optimizer, scheduler)

    print_rank0(args, "Starting VQ-VAE training...")
    for epoch in range(start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        model.train()
        train_loss_accum = 0.0
        train_recon_accum = 0.0
        train_vq_accum = 0.0
        train_perplexity_accum = 0.0
        num_train_steps = 0

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{args.epochs} [Train]",
            disable=not is_rank0(args),
        )
        for batch in pbar:
            x = preprocess_raw_data(batch, normalization_stats, device)

            optimizer.zero_grad(set_to_none=True)
            recon_x, q_info = model(x)

            recon_loss = masked_mse_loss(recon_x, x, mask)
            vq_loss = q_info['vq_loss']
            total_loss = recon_loss + args.vq_weight * vq_loss

            total_loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model_without_ddp.parameters(), args.grad_clip)
            optimizer.step()

            train_loss_accum += total_loss.item()
            train_recon_accum += recon_loss.item()
            train_vq_accum += vq_loss.item()
            train_perplexity_accum += q_info['perplexity'].item()
            num_train_steps += 1

            if is_rank0(args):
                pbar.set_postfix({
                    'loss': total_loss.item(),
                    'recon': recon_loss.item(),
                    'vq': vq_loss.item(),
                    'ppl': q_info['perplexity'].item(),
                })

            del recon_x, q_info, recon_loss, vq_loss, total_loss, x

        local_train_loss = torch.tensor(train_loss_accum / max(num_train_steps, 1), device=device, dtype=torch.float32)
        local_train_recon = torch.tensor(train_recon_accum / max(num_train_steps, 1), device=device, dtype=torch.float32)
        local_train_vq = torch.tensor(train_vq_accum / max(num_train_steps, 1), device=device, dtype=torch.float32)
        local_train_ppl = torch.tensor(train_perplexity_accum / max(num_train_steps, 1), device=device, dtype=torch.float32)

        avg_train_loss = reduce_mean(local_train_loss, args).item()
        avg_train_recon = reduce_mean(local_train_recon, args).item()
        avg_train_vq = reduce_mean(local_train_vq, args).item()
        avg_train_ppl = reduce_mean(local_train_ppl, args).item()

        model.eval()
        val_loss_accum = 0.0
        val_recon_accum = 0.0
        val_vq_accum = 0.0
        val_perplexity_accum = 0.0
        num_val_steps = 0

        with torch.no_grad():
            val_pbar = tqdm(
                val_loader,
                desc=f"Epoch {epoch + 1}/{args.epochs} [Val]",
                disable=not is_rank0(args),
            )
            for batch in val_pbar:
                x = preprocess_raw_data(batch, normalization_stats, device)
                recon_x, q_info = model(x)

                recon_loss = masked_mse_loss(recon_x, x, mask)
                vq_loss = q_info['vq_loss']
                total_loss = recon_loss + args.vq_weight * vq_loss

                val_loss_accum += total_loss.item()
                val_recon_accum += recon_loss.item()
                val_vq_accum += vq_loss.item()
                val_perplexity_accum += q_info['perplexity'].item()
                num_val_steps += 1

                del recon_x, q_info, recon_loss, vq_loss, total_loss, x

        local_val_loss = torch.tensor(val_loss_accum / max(num_val_steps, 1), device=device, dtype=torch.float32)
        local_val_recon = torch.tensor(val_recon_accum / max(num_val_steps, 1), device=device, dtype=torch.float32)
        local_val_vq = torch.tensor(val_vq_accum / max(num_val_steps, 1), device=device, dtype=torch.float32)
        local_val_ppl = torch.tensor(val_perplexity_accum / max(num_val_steps, 1), device=device, dtype=torch.float32)

        avg_val_loss = reduce_mean(local_val_loss, args).item()
        avg_val_recon = reduce_mean(local_val_recon, args).item()
        avg_val_vq = reduce_mean(local_val_vq, args).item()
        avg_val_ppl = reduce_mean(local_val_ppl, args).item()

        print_rank0(
            args,
            f"Epoch {epoch + 1}: "
            f"Train loss/recon/vq/ppl={avg_train_loss:.6f}/{avg_train_recon:.6f}/{avg_train_vq:.6f}/{avg_train_ppl:.2f}, "
            f"Val loss/recon/vq/ppl={avg_val_loss:.6f}/{avg_val_recon:.6f}/{avg_val_vq:.6f}/{avg_val_ppl:.2f}, "
            f"LR={optimizer.param_groups[0]['lr']:.8f}",
        )

        if avg_val_loss < best_val_loss and is_rank0(args):
            best_val_loss = avg_val_loss
            save_path = os.path.join(args.save_dir, "best_model.pth")
            save_checkpoint({
                'epoch': epoch,
                'model_state_dict': model_without_ddp.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': best_val_loss,
                'args': vars(args),
            }, save_path)
            print_rank0(args, f"Saved best model to {save_path}")

        if (epoch + 1) % 10 == 0 and is_rank0(args):
            save_path = os.path.join(args.save_dir, f'ckpt_{epoch:03d}.pth')
            save_checkpoint({
                'epoch': epoch,
                'model_state_dict': model_without_ddp.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_val_loss,
                'args': vars(args),
            }, save_path)

        if is_rank0(args):
            save_path = os.path.join(args.save_dir, 'last_model.pth')
            save_checkpoint({
                'epoch': epoch,
                'model_state_dict': model_without_ddp.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': best_val_loss,
                'args': vars(args),
            }, save_path)

        scheduler.step()

    cleanup_distributed(args)


if __name__ == "__main__":
    args = parse_args()
    train(args)
