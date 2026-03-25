import sys
import os
import argparse
import json
import random
from datetime import datetime
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from model.dcae import DCAE
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


def log_run_config(args):
    if not is_rank0(args):
        return

    os.makedirs(args.save_dir, exist_ok=True)

    run_config = dict(vars(args))
    run_config.update({
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "project_root": current_dir,
        "working_dir": os.getcwd(),
        "python_executable": sys.executable,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    })

    json_path = os.path.join(args.save_dir, "run_config.json")
    txt_path = os.path.join(args.save_dir, "run_config.txt")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(run_config, f, ensure_ascii=False, indent=2)

    with open(txt_path, "w", encoding="utf-8") as f:
        for key in sorted(run_config.keys()):
            f.write(f"{key}: {run_config[key]}\n")

    print_rank0(args, "===== Run Configuration =====")
    for key in sorted(run_config.keys()):
        print_rank0(args, f"{key}: {run_config[key]}")
    print_rank0(args, "=============================")
    print_rank0(args, f"Saved run config to {json_path}")
    print_rank0(args, f"Saved run config to {txt_path}")


def reduce_mean(value: torch.Tensor, args):
    if not args.distributed:
        return value
    reduced = value.detach().clone()
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    reduced /= args.world_size
    return reduced


def parse_args():
    parser = argparse.ArgumentParser(description="Train DC-AE for Ocean Data")
    parser.add_argument("--data-name", type=str, default="glorys12_kuroshio_extension", help="Dataset name")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--min-lr", type=float, default=1e-6, help="Minimum learning rate for cosine annealing")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--local_rank", "--local-rank", type=int, default=0, help="Local rank for distributed training")
    parser.add_argument("--tag", type=str, default="dcae_training", help="Experiment tag")
    parser.add_argument("--save-dir", type=str, default=None, help="Output directory for checkpoints/logs; default=output/<data_name>/<tag>")

    parser.add_argument("--base-channels", type=int, default=64, help="Base channel count at stage 0")
    parser.add_argument("--channel-multipliers", nargs="+", type=int, default=[1, 2, 4], help="Per-stage channel multipliers; num_stages = len - 1")
    parser.add_argument("--latent-channels", type=int, default=16, help="Latent bottleneck channels")
    parser.add_argument("--num-res-blocks", type=int, default=2, help="ResBlocks per encoder/decoder stage")
    parser.add_argument("--attention-resolutions", nargs="+", type=int, default=[0, 1], help="Stage indices (0-based) where spatial attention is enabled")
    parser.add_argument("--num-heads", type=int, default=8, help="Number of attention heads in SpatialAttention")
    parser.add_argument("--fft-weight", type=float, default=0.5, help="Weight for FFT spectral loss term")
    parser.add_argument("--latent-reg", type=float, default=1e-3, help="L2 regularization weight on latent z magnitude; prevents unbounded Softplus outputs (0 disables)")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping max norm; <=0 disables")

    parser.add_argument("--structured-weight", type=float, default=1.0,
                        help="Weight for DC-AE 1.5 structured latent space loss (0 disables)")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader num_workers per rank")
    parser.add_argument("--auto-resume", action="store_true")
    parser.add_argument("--resume-path", type=str, default=None)

    args = parser.parse_args()

    num_stages = max(len(args.channel_multipliers) - 1, 0)
    valid_attention_indices = [idx for idx in args.attention_resolutions if 0 <= idx < num_stages]
    invalid_attention_indices = [idx for idx in args.attention_resolutions if idx not in valid_attention_indices]
    if invalid_attention_indices:
        print(
            f"Warning: ignoring invalid --attention-resolutions indices {invalid_attention_indices}; "
            f"valid range is 0..{max(num_stages - 1, 0)} for channel_multipliers={args.channel_multipliers}."
        )
    args.attention_resolutions = sorted(set(valid_attention_indices))

    if args.save_dir is None or str(args.save_dir).strip() == "":
        args.save_dir = os.path.join("output", args.data_name, args.tag)
    return args


def masked_recon_loss(recon_x, x, mask):
    x = torch.nan_to_num(x, nan=0.0)
    l1 = F.l1_loss(recon_x, x, reduction='none')
    l2 = F.mse_loss(recon_x, x, reduction='none')
    loss = 0.5 * l1 + 0.5 * l2

    if mask.ndim < loss.ndim:
        while mask.ndim < loss.ndim:
            mask = mask.unsqueeze(0)

    valid_mask = (mask > 0).to(loss.dtype)
    valid_mask = valid_mask.expand_as(loss)
    loss = loss * valid_mask

    num_valid = valid_mask.sum().clamp_min(1.0)
    return loss.sum() / num_valid


def fft_spectral_loss(recon_x: torch.Tensor, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
    """L1 loss on 2D FFT amplitude spectra. NaN pixels zeroed before transform."""
    x_clean = torch.nan_to_num(x, nan=0.0)
    recon_clean = torch.nan_to_num(recon_x, nan=0.0)

    if mask is not None:
        if mask.ndim < x_clean.ndim:
            while mask.ndim < x_clean.ndim:
                mask = mask.unsqueeze(0)
        valid = (mask > 0).to(x_clean.dtype)
        x_clean = x_clean * valid
        recon_clean = recon_clean * valid

    x_fft = torch.fft.rfft2(x_clean)
    r_fft = torch.fft.rfft2(recon_clean)
    return F.l1_loss(torch.abs(r_fft), torch.abs(x_fft))


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

    tb_writer = SummaryWriter(log_dir=os.path.join(args.save_dir, "tb")) if is_rank0(args) else None

    log_run_config(args)

    device = torch.device(args.device)
    scaler = GradScaler(enabled=torch.cuda.is_available())
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
        f"Initializing DC-AE with in_channels={in_channels}, "
        f"channel_multipliers={args.channel_multipliers}, "
        f"latent_channels={args.latent_channels}, "
        f"f={2 ** (len(args.channel_multipliers) - 1)}",
    )

    model = DCAE(
        in_channels=in_channels,
        base_channels=args.base_channels,
        channel_multipliers=args.channel_multipliers,
        latent_channels=args.latent_channels,
        num_res_blocks=args.num_res_blocks,
        attention_resolutions=args.attention_resolutions,
        num_heads=args.num_heads,
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

    print_rank0(args, "Starting DC-AE training...")
    try:
        for epoch in range(start_epoch, args.epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            model.train()
            train_loss_accum = 0.0
            train_recon_accum = 0.0
            train_fft_accum = 0.0
            num_train_steps = 0

            pbar = tqdm(
                train_loader,
                desc=f"Epoch {epoch + 1}/{args.epochs} [Train]",
                disable=not is_rank0(args),
            )
            for batch in pbar:
                x = preprocess_raw_data(batch, normalization_stats, device)

                optimizer.zero_grad(set_to_none=True)
                with autocast(enabled=torch.cuda.is_available()):
                    target_shape = (x.shape[2], x.shape[3])
                    recon_x, z = model(x, return_latent=True)
                    recon_loss = masked_recon_loss(recon_x, x, mask)
                    fft_loss = fft_spectral_loss(recon_x, x, mask)
                    latent_reg_loss = z.pow(2).mean()

                    # DC-AE 1.5: Structured Latent Space loss
                    # Randomly use only first c' channels to force channel ordering
                    c_prime = random.randint(1, args.latent_channels)
                    channel_mask = torch.zeros_like(z)
                    channel_mask[:, :c_prime] = 1.0
                    recon_partial = model_without_ddp.decode(z * channel_mask, target_shape)
                    structured_loss = masked_recon_loss(recon_partial, x, mask)

                    total_loss = (recon_loss
                                  + args.fft_weight * fft_loss
                                  + args.latent_reg * latent_reg_loss
                                  + args.structured_weight * structured_loss)

                scaler.scale(total_loss).backward()
                if args.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(model_without_ddp.parameters(), args.grad_clip)
                else:
                    scaler.unscale_(optimizer)
                    grad_norm = torch.norm(torch.stack([
                        p.grad.detach().norm(2) for p in model_without_ddp.parameters() if p.grad is not None
                    ]), 2) if any(p.grad is not None for p in model_without_ddp.parameters()) else None
                scaler.step(optimizer)
                scaler.update()

                train_loss_accum += total_loss.item()
                train_recon_accum += recon_loss.item()
                train_fft_accum += fft_loss.item()
                num_train_steps += 1

                if is_rank0(args):
                    pbar.set_postfix({
                        'loss': total_loss.item(),
                        'recon': recon_loss.item(),
                        'fft': fft_loss.item(),
                        'z_reg': latent_reg_loss.item(),
                        'struct': structured_loss.item(),
                    })

                    global_step = epoch * len(train_loader) + num_train_steps - 1
                    tb_writer.add_scalar('train_step/loss', total_loss.item(), global_step)
                    tb_writer.add_scalar('train_step/recon', recon_loss.item(), global_step)
                    tb_writer.add_scalar('train_step/fft', fft_loss.item(), global_step)
                    tb_writer.add_scalar('train_step/z_reg', latent_reg_loss.item(), global_step)
                    tb_writer.add_scalar('train_step/structured', structured_loss.item(), global_step)
                    tb_writer.add_scalar('train_step/lr', optimizer.param_groups[0]['lr'], global_step)
                    if grad_norm is not None:
                        tb_writer.add_scalar('train_step/grad_norm', float(grad_norm.item()), global_step)

                del recon_x, z, recon_partial, recon_loss, fft_loss, latent_reg_loss, structured_loss, total_loss, x

            # Free cached blocks before validation to reduce fragmentation/OOM.
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            local_train_loss = torch.tensor(train_loss_accum / max(num_train_steps, 1), device=device, dtype=torch.float32)
            local_train_recon = torch.tensor(train_recon_accum / max(num_train_steps, 1), device=device, dtype=torch.float32)
            local_train_fft = torch.tensor(train_fft_accum / max(num_train_steps, 1), device=device, dtype=torch.float32)

            avg_train_loss = reduce_mean(local_train_loss, args).item()
            avg_train_recon = reduce_mean(local_train_recon, args).item()
            avg_train_fft = reduce_mean(local_train_fft, args).item()

            model.eval()
            val_loss_accum = 0.0
            val_recon_accum = 0.0
            val_fft_accum = 0.0
            num_val_steps = 0

            with torch.inference_mode(), autocast(enabled=torch.cuda.is_available()):
                val_pbar = tqdm(
                    val_loader,
                    desc=f"Epoch {epoch + 1}/{args.epochs} [Val]",
                    disable=not is_rank0(args),
                )
                for batch in val_pbar:
                    x = preprocess_raw_data(batch, normalization_stats, device)
                    recon_x = model(x)

                    recon_loss = masked_recon_loss(recon_x, x, mask)
                    fft_loss = fft_spectral_loss(recon_x, x, mask)
                    total_loss = recon_loss + args.fft_weight * fft_loss

                    val_loss_accum += total_loss.item()
                    val_recon_accum += recon_loss.item()
                    val_fft_accum += fft_loss.item()
                    num_val_steps += 1

                    del recon_x, recon_loss, fft_loss, total_loss, x

            local_val_loss = torch.tensor(val_loss_accum / max(num_val_steps, 1), device=device, dtype=torch.float32)
            local_val_recon = torch.tensor(val_recon_accum / max(num_val_steps, 1), device=device, dtype=torch.float32)
            local_val_fft = torch.tensor(val_fft_accum / max(num_val_steps, 1), device=device, dtype=torch.float32)

            avg_val_loss = reduce_mean(local_val_loss, args).item()
            avg_val_recon = reduce_mean(local_val_recon, args).item()
            avg_val_fft = reduce_mean(local_val_fft, args).item()

            print_rank0(
                args,
                f"Epoch {epoch + 1}: "
                f"Train loss/recon/fft={avg_train_loss:.6f}/{avg_train_recon:.6f}/{avg_train_fft:.6f}, "
                f"Val loss/recon/fft={avg_val_loss:.6f}/{avg_val_recon:.6f}/{avg_val_fft:.6f}, "
                f"LR={optimizer.param_groups[0]['lr']:.8f}",
            )

            if tb_writer is not None:
                tb_writer.add_scalar('train/loss', avg_train_loss, epoch)
                tb_writer.add_scalar('train/recon', avg_train_recon, epoch)
                tb_writer.add_scalar('train/fft', avg_train_fft, epoch)
                tb_writer.add_scalar('valid/loss', avg_val_loss, epoch)
                tb_writer.add_scalar('valid/recon', avg_val_recon, epoch)
                tb_writer.add_scalar('valid/fft', avg_val_fft, epoch)
                tb_writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
                tb_writer.flush()

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

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        if tb_writer is not None:
            tb_writer.close()
        cleanup_distributed(args)


if __name__ == "__main__":
    args = parse_args()
    train(args)
