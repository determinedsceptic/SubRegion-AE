import os
import sys
import json
import argparse
import random
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.decomposition import IncrementalPCA

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from config import get_dataset_config
from data.dataset import build_dataset, load_constants
from data.data_utils import normalize_fn


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_split_date_range(dataset_config, split: str):
    split = split.lower()
    if split == "train":
        return dataset_config.train_date_range
    if split == "val":
        return dataset_config.val_date_range
    if split == "test":
        return dataset_config.test_date_range
    raise ValueError(f"Unsupported split: {split}")


def build_loader(dataset_config, split: str, batch_size: int, num_workers: int):
    date_range = get_split_date_range(dataset_config, split)
    dataset = build_dataset(dataset_config.raw_data_dir, date_range)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
    )
    return dataset, loader


def load_norm_and_mask(dataset_config, device: torch.device):
    constants = load_constants(dataset_config.constant_dir)
    normed_ocean_mean = constants[0].to(device)
    normed_ocean_std = constants[1].to(device)
    mask = constants[-1].to(device).float()

    mu = normed_ocean_mean[..., None, None]
    sigma = normed_ocean_std[..., None, None]

    return mu, sigma, mask


def flatten_features(x_norm: torch.Tensor, mask_flat: np.ndarray = None):
    # [B, C, H, W] -> [B, C*H*W]
    flat = x_norm.reshape(x_norm.shape[0], -1)
    if mask_flat is not None:
        flat = flat[:, mask_flat]
    return flat


def fit_incremental_pca(args, loader, mu, sigma, feature_mask_flat=None):
    ipca = IncrementalPCA(n_components=args.n_components, batch_size=args.ipca_batch_size)

    fitted_samples = 0
    min_fit_batch = args.n_components
    buffered_chunks = []
    buffered_count = 0

    pbar = tqdm(loader, desc="Fitting IncrementalPCA")
    for batch in pbar:
        x = batch.to(dtype=torch.float32)
        x_norm = normalize_fn(x, mu=mu.cpu(), sigma=sigma.cpu())
        x_np = flatten_features(x_norm, feature_mask_flat).cpu().numpy().astype(np.float32)

        if args.max_fit_samples > 0:
            remain = args.max_fit_samples - fitted_samples
            if remain <= 0:
                break
            x_np = x_np[:remain]

        if x_np.shape[0] == 0:
            continue

        buffered_chunks.append(x_np)
        buffered_count += x_np.shape[0]

        if buffered_count >= min_fit_batch:
            x_fit = np.concatenate(buffered_chunks, axis=0)
            ipca.partial_fit(x_fit)
            fitted_samples += x_fit.shape[0]
            buffered_chunks = []
            buffered_count = 0

        pbar.set_postfix({"fitted": fitted_samples})

        if args.max_fit_samples > 0 and fitted_samples >= args.max_fit_samples:
            break

    # If leftovers are enough for one more partial_fit, consume them as well.
    if buffered_count >= min_fit_batch:
        x_fit = np.concatenate(buffered_chunks, axis=0)
        ipca.partial_fit(x_fit)
        fitted_samples += x_fit.shape[0]

    if fitted_samples == 0:
        raise RuntimeError("No samples were used to fit PCA. Check dataset path or max_fit_samples.")

    return ipca, fitted_samples


def evaluate_reconstruction(args, loader, ipca, mu, sigma, mask, feature_mask_flat=None):
    mask = mask.to(dtype=torch.float32)
    while mask.ndim < 4:
        mask = mask.unsqueeze(0)

    total_sq = 0.0
    total_abs = 0.0
    total_cnt = 0.0
    eval_samples = 0

    C = mu.shape[0]
    H, W = mask.shape[-2], mask.shape[-1]
    full_dim = C * H * W

    pbar = tqdm(loader, desc="Evaluating PCA")
    for batch in pbar:
        x = batch.to(dtype=torch.float32)
        x_norm = normalize_fn(x, mu=mu.cpu(), sigma=sigma.cpu())

        x_flat = flatten_features(x_norm, feature_mask_flat).cpu().numpy().astype(np.float32)
        if args.max_eval_samples > 0:
            remain = args.max_eval_samples - eval_samples
            if remain <= 0:
                break
            x_flat = x_flat[:remain]
            x_norm = x_norm[:remain]

        if x_flat.shape[0] == 0:
            continue

        z = ipca.transform(x_flat)
        recon_flat = ipca.inverse_transform(z).astype(np.float32)

        if feature_mask_flat is not None:
            recon_full = np.zeros((recon_flat.shape[0], full_dim), dtype=np.float32)
            recon_full[:, feature_mask_flat] = recon_flat
            recon_flat = recon_full

        recon_norm = torch.from_numpy(recon_flat).reshape(-1, C, H, W)

        diff = recon_norm - x_norm
        valid = mask.expand_as(diff)
        total_sq += (diff.pow(2) * valid).sum().item()
        total_abs += (diff.abs() * valid).sum().item()
        total_cnt += valid.sum().item()

        eval_samples += x_norm.shape[0]
        pbar.set_postfix({"eval": eval_samples})

        if args.max_eval_samples > 0 and eval_samples >= args.max_eval_samples:
            break

    if total_cnt <= 0:
        raise RuntimeError("No valid ocean pixels found in mask for metric computation.")

    mse = total_sq / total_cnt
    rmse = float(np.sqrt(mse))
    mae = total_abs / total_cnt

    metrics = {
        "rmse_norm": rmse,
        "mae_norm": mae,
        "mse_norm": mse,
        "evaluated_samples": eval_samples,
    }
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description="PCA baseline for DCAE comparison")
    parser.add_argument("--data-name", type=str, default="glorys12_kuroshio_extension")
    parser.add_argument("--fit-split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--eval-split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--n-components", type=int, default=128, help="PCA latent dimension")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--ipca-batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-fit-samples", type=int, default=0, help="0 means use full split")
    parser.add_argument("--max-eval-samples", type=int, default=0, help="0 means use full split")
    parser.add_argument("--use-ocean-mask-only", action="store_true", help="Use only ocean pixels as PCA features")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tag", type=str, default="pca_baseline")
    parser.add_argument("--save-dir", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    dataset_config = get_dataset_config(args.data_name)
    device = torch.device("cpu")

    if args.save_dir is None or str(args.save_dir).strip() == "":
        args.save_dir = os.path.join("output", args.data_name, args.tag)
    os.makedirs(args.save_dir, exist_ok=True)

    fit_dataset, fit_loader = build_loader(dataset_config, args.fit_split, args.batch_size, args.num_workers)
    eval_dataset, eval_loader = build_loader(dataset_config, args.eval_split, args.batch_size, args.num_workers)

    mu, sigma, mask = load_norm_and_mask(dataset_config, device)

    feature_mask_flat = None
    if args.use_ocean_mask_only:
        C = dataset_config.num_channels
        mask_flat = mask.reshape(-1).bool().cpu().numpy()
        feature_mask_flat = np.tile(mask_flat, C)

    feature_dim = dataset_config.num_channels * dataset_config.grid_size[0] * dataset_config.grid_size[1]
    feature_dim_eff = int(feature_mask_flat.sum()) if feature_mask_flat is not None else feature_dim

    if args.n_components <= 0 or args.n_components > feature_dim_eff:
        raise ValueError(
            f"n_components must be in [1, {feature_dim_eff}], got {args.n_components}"
        )

    print("===== PCA Baseline Configuration =====")
    print(f"data_name: {args.data_name}")
    print(f"fit_split: {args.fit_split}, eval_split: {args.eval_split}")
    print(f"n_components: {args.n_components}")
    print(f"fit/eval dataset size: {len(fit_dataset)}/{len(eval_dataset)}")
    print(f"feature_dim(full/effective): {feature_dim}/{feature_dim_eff}")
    print(f"use_ocean_mask_only: {args.use_ocean_mask_only}")
    print("======================================")

    ipca, fitted_samples = fit_incremental_pca(
        args=args,
        loader=fit_loader,
        mu=mu,
        sigma=sigma,
        feature_mask_flat=feature_mask_flat,
    )

    metrics = evaluate_reconstruction(
        args=args,
        loader=eval_loader,
        ipca=ipca,
        mu=mu,
        sigma=sigma,
        mask=mask,
        feature_mask_flat=feature_mask_flat,
    )

    explained = float(np.sum(ipca.explained_variance_ratio_))
    compression_ratio_full = feature_dim / float(args.n_components)
    compression_ratio_eff = feature_dim_eff / float(args.n_components)

    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "data_name": args.data_name,
        "fit_split": args.fit_split,
        "eval_split": args.eval_split,
        "n_components": args.n_components,
        "fitted_samples": int(fitted_samples),
        "feature_dim_full": int(feature_dim),
        "feature_dim_effective": int(feature_dim_eff),
        "use_ocean_mask_only": bool(args.use_ocean_mask_only),
        "compression_ratio_full": float(compression_ratio_full),
        "compression_ratio_effective": float(compression_ratio_eff),
        "explained_variance_ratio_sum": explained,
        **metrics,
    }

    model_npz = os.path.join(args.save_dir, "pca_model.npz")
    summary_json = os.path.join(args.save_dir, "pca_metrics.json")

    np.savez_compressed(
        model_npz,
        components=ipca.components_.astype(np.float32),
        mean=ipca.mean_.astype(np.float32),
        explained_variance=ipca.explained_variance_.astype(np.float32),
        explained_variance_ratio=ipca.explained_variance_ratio_.astype(np.float32),
        singular_values=ipca.singular_values_.astype(np.float32),
        n_components=np.array([args.n_components], dtype=np.int32),
        feature_dim_full=np.array([feature_dim], dtype=np.int64),
        feature_dim_effective=np.array([feature_dim_eff], dtype=np.int64),
    )

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("===== PCA Baseline Results =====")
    for k, v in summary.items():
        print(f"{k}: {v}")
    print("================================")
    print(f"Saved PCA model to: {model_npz}")
    print(f"Saved metrics to: {summary_json}")


if __name__ == "__main__":
    main()
