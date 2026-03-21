import os
from typing import List, Optional, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset


def load_constants(constant_dir: str) -> Tuple[torch.Tensor, ...]:
    """
    从 constant_dir 加载归一化统计量和 mask。

    Returns
    -------
    normed_ocean_mean, normed_ocean_std, raw_ocean_min, raw_ocean_max, depths, mask
    """
    def safe_load(filename, default_val):
        path = os.path.join(constant_dir, filename)
        try:
            return torch.load(path)
        except (FileNotFoundError, RuntimeError):
            return torch.tensor(default_val)

    normed_ocean_mean = safe_load("normed_ocean_mean.pt", [0.0])
    normed_ocean_std  = safe_load("normed_ocean_std.pt",  [1.0])
    raw_ocean_min     = safe_load("raw_ocean_min.pt",     [0.0])
    raw_ocean_max     = safe_load("raw_ocean_max.pt",     [1.0])
    depths            = safe_load("depths.pt",            [0.0])
    mask              = safe_load("mask.pt",              [1.0])

    return normed_ocean_mean, normed_ocean_std, raw_ocean_min, raw_ocean_max, depths, mask


class OceanRawDataset(Dataset):
    """只加载 raw_data 的最小数据集，用于 AE 训练。"""

    def __init__(self, raw_data_files: List[str]):
        self.files = raw_data_files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.load(self.files[idx])


def build_dataset(raw_data_dir: str, date_range: Tuple[str, str]) -> OceanRawDataset:
    """
    Parameters
    ----------
    raw_data_dir : 存放 {YYYYMMDD}.pt 的目录
    date_range   : (start, end) 字符串，如 ("1993-01-01", "2017-12-31")
    """
    start, end = date_range
    dates = pd.date_range(start=start, end=end, freq="D")
    files = [os.path.join(raw_data_dir, f"{d.strftime('%Y%m%d')}.pt") for d in dates]
    return OceanRawDataset(files)
