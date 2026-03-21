import os
import yaml
from dataclasses import dataclass, field
from typing import List, Tuple, Union

_current_dir = os.path.dirname(os.path.abspath(__file__))
_datasets_file = os.path.join(_current_dir, 'datasets.yaml')
_datasets_config = {}
if os.path.exists(_datasets_file):
    with open(_datasets_file, 'r') as f:
        _datasets_config = yaml.safe_load(f)


@dataclass
class DatasetConfig:
    data_name: str = 'glorys12_kuroshio_extension'

    raw_data_dir: str = ''
    constant_dir: str = ''
    atm_forcing_dir: str = ''
    climatology_dir: str = ''

    num_channels: int = 101
    num_forcing_channels: int = 9
    grid_size: Tuple[int, int] = (250, 300)

    train_date_range: Union[Tuple[str, str], List[Tuple[str, str]]] = ('1993-01-01', '2017-12-31')
    val_date_range:   Union[Tuple[str, str], List[Tuple[str, str]]] = ('2018-01-01', '2018-12-31')
    test_date_range:  Union[Tuple[str, str], List[Tuple[str, str]]] = ('2019-01-01', '2020-12-31')

    def __post_init__(self):
        if self.data_name in _datasets_config:
            cfg = _datasets_config[self.data_name]
            for key, value in cfg.items():
                if hasattr(self, key) and key != 'data_name':
                    setattr(self, key, value)
            # datasets.yaml 用 list 存 date range，统一转成 tuple
            for attr in ('train_date_range', 'val_date_range', 'test_date_range'):
                v = getattr(self, attr)
                if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str):
                    setattr(self, attr, tuple(v))


def get_dataset_config(data_name: str) -> DatasetConfig:
    return DatasetConfig(data_name=data_name)
