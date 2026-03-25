"""
utils/flip_metrics.py  —  翻转指标核心计算函数

从 eval_flip.py 提取，供 train_dcae.py 和 eval_multi_seed.py 复用。
"""

import math
import torch
import torch.nn.functional as F


def _make_gaussian_kernel_1d(sigma: float) -> torch.Tensor:
    """Create a normalized 1-D Gaussian kernel, shape [K]."""
    radius = int(math.ceil(3 * sigma))
    size = 2 * radius + 1
    coords = torch.arange(size, dtype=torch.float32) - radius
    g = torch.exp(-coords.pow(2) / (2 * sigma ** 2))
    g = g / g.sum()
    return g


def _make_gaussian_kernel(sigma: float, channels: int) -> torch.Tensor:
    """Create a 2D Gaussian kernel for depthwise convolution."""
    radius = int(math.ceil(3 * sigma))
    size = 2 * radius + 1
    coords = torch.arange(size, dtype=torch.float32) - radius
    g1d = torch.exp(-coords.pow(2) / (2 * sigma ** 2))
    g2d = g1d[:, None] * g1d[None, :]
    g2d = g2d / g2d.sum()
    # [C_out, C_in/groups, kH, kW]  — depthwise: groups=channels
    kernel = g2d.unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1, 1)
    return kernel


def _norm_conv1d(x, m, kernel_1d, dim, C):
    """
    Normalized depthwise 1-D convolution along spatial dim.
    dim='h' → vertical (kernel shape [C,1,K,1]),
    dim='w' → horizontal (kernel shape [C,1,1,K]).
    """
    K = kernel_1d.shape[0]
    pad_size = K // 2
    if dim == 'w':
        kernel = kernel_1d.view(1, 1, 1, K).expand(C, 1, 1, K)
        pad_arg = [pad_size, pad_size, 0, 0]          # left, right, top, bot
    else:
        kernel = kernel_1d.view(1, 1, K, 1).expand(C, 1, K, 1)
        pad_arg = [0, 0, pad_size, pad_size]

    kernel = kernel.to(x.device, x.dtype)

    num = F.conv2d(F.pad(x * m, pad_arg, mode='reflect'),  kernel, groups=C)
    den = F.conv2d(F.pad(m,     pad_arg, mode='constant', value=0), kernel, groups=C)
    return num / den.clamp_min(1e-8)


def masked_gaussian_blur(
    x: torch.Tensor,
    mask: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    """
    Normalized convolution: 只在有效像素上做高斯平滑，避免海岸线偏差。
    使用可分离 1-D 卷积（先水平后垂直），内存 O(K) 而非 O(K²)。

    x    : [B, C, H, W]
    mask : broadcastable to [B, C, H, W], 1=valid 0=land
    """
    C = x.shape[1]
    g1d = _make_gaussian_kernel_1d(sigma)

    m = mask.to(x.dtype)
    while m.ndim < x.ndim:
        m = m.unsqueeze(0)
    m = m.expand_as(x)

    # pass 1: horizontal
    h = _norm_conv1d(x, m, g1d, dim='w', C=C)
    # pass 2: vertical (re-mask to avoid propagating land fill)
    return _norm_conv1d(h, m, g1d, dim='h', C=C)


def compute_flip_metrics(
    x: torch.Tensor,
    recon: torch.Tensor,
    mask: torch.Tensor,
    sigma: float,
):
    """
    Returns
    -------
    flip_map    : [B, C, H, W]  bool, True = flipped pixel
    flip_rate   : [B, C]        fraction of ocean pixels that flipped
    local_corr  : [B, C]        per-channel spatial correlation of local anomalies
    anom_true   : [B, C, H, W]  local anomaly of ground truth
    anom_recon  : [B, C, H, W]  local anomaly of reconstruction
    """
    bg_true  = masked_gaussian_blur(x,     mask, sigma)
    bg_recon = masked_gaussian_blur(recon, mask, sigma)

    anom_true  = x     - bg_true
    anom_recon = recon - bg_recon

    # mask
    m = mask.to(x.dtype)
    while m.ndim < x.ndim:
        m = m.unsqueeze(0)
    m = m.expand_as(x)
    m_bool = m > 0.5

    # ── flip map: 符号不一致 且 异常幅度足够大(避免噪声) ──
    # 阈值: 每通道异常标准差的 5%
    abs_anom = anom_true.abs()
    # per-channel std over ocean pixels
    B, C, H, W = x.shape
    anom_flat = (abs_anom * m).view(B, C, -1)
    n_valid = m.view(B, C, -1).sum(dim=2).clamp_min(1)
    anom_std = (anom_flat.pow(2).sum(dim=2) / n_valid).sqrt()  # [B, C]
    threshold = 0.05 * anom_std  # [B, C]

    significant = abs_anom > threshold[:, :, None, None]
    sign_disagree = (anom_true.sign() != anom_recon.sign())
    flip_map = sign_disagree & significant & m_bool  # [B, C, H, W]

    # ── flip rate ──
    n_significant = (significant & m_bool).view(B, C, -1).sum(dim=2).clamp_min(1).float()
    flip_count = flip_map.view(B, C, -1).sum(dim=2).float()
    flip_rate = flip_count / n_significant  # [B, C]

    # ── local correlation (Pearson) per channel ──
    at = (anom_true * m).view(B, C, -1)
    ar = (anom_recon * m).view(B, C, -1)
    at_mean = at.sum(2) / n_valid
    ar_mean = ar.sum(2) / n_valid
    at_c = at - at_mean[:, :, None]
    ar_c = ar - ar_mean[:, :, None]
    cov = (at_c * ar_c * m.view(B, C, -1)).sum(2) / n_valid
    std_t = ((at_c.pow(2) * m.view(B, C, -1)).sum(2) / n_valid).sqrt()
    std_r = ((ar_c.pow(2) * m.view(B, C, -1)).sum(2) / n_valid).sqrt()
    local_corr = cov / (std_t * std_r).clamp_min(1e-8)  # [B, C]

    return flip_map, flip_rate, local_corr, anom_true, anom_recon
