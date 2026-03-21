import torch
from typing import Union, Optional, List

# Type alias for cleaner signatures
NumType = Union[float, int]
NormParamType = Union[NumType, List[NumType], torch.Tensor, List[torch.Tensor]]


def normalize_fn(
    data: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    mu: NormParamType = 0.0,
    sigma: NormParamType = 1.0,
) -> torch.Tensor:
    """
    Normalizes data using the formula: (data - mu) / sigma.
    Supports sequential normalization if mu/sigma are lists.
    """
    # Ensure inputs are lists for consistent iteration
    mu_list = mu if isinstance(mu, list) else [mu]
    sigma_list = sigma if isinstance(sigma, list) else [sigma]

    if len(mu_list) != len(sigma_list):
        raise ValueError("Lengths of 'mu' and 'sigma' must match if provided as lists.")

    # Apply normalization steps sequentially
    for _mu, _sigma in zip(mu_list, sigma_list):
        data = (data - _mu) / _sigma

    # Handle numeric stability
    data = torch.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

    # Apply mask if provided
    if mask is not None:
        # Broadcast mask to match data shape if needed
        if mask.ndim < data.ndim:
            # Expand mask dimensions on the left (e.g., (H, W) -> (1, H, W))
            # or handle based on specific logic. Here assuming simple broadcasting works or
            # keeping original simple check.
            pass
        data = data * mask.bool()

    return data


def denormalize_fn(
    data: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    mu: NormParamType = 0.0,
    sigma: NormParamType = 1.0,
) -> torch.Tensor:
    """
    Denormalizes data using the formula: data * sigma + mu.
    Reverses the order of operations if mu/sigma are lists.
    """
    # Ensure inputs are lists
    mu_list = mu if isinstance(mu, list) else [mu]
    sigma_list = sigma if isinstance(sigma, list) else [sigma]

    if len(mu_list) != len(sigma_list):
        raise ValueError("Lengths of 'mu' and 'sigma' must match if provided as lists.")

    # Handle numeric stability
    data = torch.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

    # Apply denormalization in REVERSE order
    for _mu, _sigma in reversed(list(zip(mu_list, sigma_list))):
        data = data * _sigma + _mu

    # Apply mask if provided (set invalid regions to NaN)
    if mask is not None:
        # Simple broadcasting logic from original code refactored for readability
        mask_bool = mask.bool()
        if mask_bool.ndim != data.ndim:
            # Attempt to broadcast if dimensions don't match exactly
            # e.g. mask (H,W) applied to (C, H, W)
            if data.shape[-mask.ndim:] == mask.shape:
                # Broadcasting automatically handles this in assignment usually, 
                # but for boolean indexing we need matching shapes or broadcasted bool tensor.
                mask_bool = mask_bool.expand_as(data)
            else:
                mask_bool = mask_bool.unsqueeze(0).expand_as(data)

        data[~mask_bool] = torch.nan

    return data


# def get_anomaly_ocean(data: torch.Tensor, climatology: torch.Tensor) -> torch.Tensor:
#     """Computes anomaly by subtracting climatology from UV channels."""
#     anomaly = data.clone()

#     # Calculate channel index: (all_channels - ssh_channel) / 2 assuming SSH is 1 channel
#     # This logic assumes structure [SSH, U_params..., V_params...]? 
#     # Or [SSH, U, V]? Original logic: (shape - 1) // 2.
#     uv_start_idx = (anomaly.shape[-3] - 1) // 2

#     # Subtract climatology only from UV channels onward
#     anomaly[..., uv_start_idx:, :, :] = (
#         data[..., uv_start_idx:, :, :] - climatology[..., uv_start_idx:, :, :]
#     )
#     return anomaly


# def get_raw_ocean(anomaly: torch.Tensor, climatology: torch.Tensor) -> torch.Tensor:
#     """Restores raw data by adding climatology to UV channels."""
#     data = anomaly.clone()
#     uv_start_idx = (anomaly.shape[-3] - 1) // 2

#     data[..., uv_start_idx:, :, :] = (
#         anomaly[..., uv_start_idx:, :, :] + climatology[..., uv_start_idx:, :, :]
#     )
#     return data
