# SubRegion-AE

## Project Overview

研究 GLORYS12 黑潮延伸区海洋再分析数据的潜空间压缩方法，核心目标是解决 DCAE 在高压缩比（~400x）下出现的 **符号翻转（sign/contrast ambiguity）** 问题，同时保留其相比 VQ-VAE 的压缩率和重建精度优势。

**核心问题**：DCAE 的连续潜空间在编解码过程中存在符号/对比度歧义——重建结果空间结构正确但高低值关系反转，导致物理意义不一致。该现象对初始条件高度敏感。

## Data

- **数据集**：GLORYS12 黑潮延伸区子区域（25°N–45.75°N, 150°E–174.9°E）
- **变量**：101 通道 = 25层 × 4变量（U, V, T, S）+ 1层 SSH
- **网格**：250 × 300
- **时间划分**：Train 1993–2017 / Val 2018 / Test 2019–2020
- **预处理方式**（见 `datasets.yaml`）：
  - `glorys12_kuroshio_extension`：逐通道 min-max → [0,1]
  - `glorys12_kuroshio_extension_anomaly`：min-max 后减气候态
  - `glorys12_kuroshio_extension_normalized_by_uvtsh`：按变量（而非通道）全深度 min-max
- **数据格式**：每日一个 `{YYYYMMDD}.pt` 文件，shape `[101, 250, 300]`
- **数据存储在远程服务器**（test1），本地无数据文件

## Architecture

### Directory Structure

```
SubRegion-AE/
├── model/                  # 模型定义
│   ├── dcae.py             # Deep Convolutional AutoEncoder（核心）
│   ├── vqvae.py            # VQ-VAE 基线（标准 + EMA 量化）
│   └── vae.py              # 标准 VAE
├── data/
│   ├── dataset.py          # OceanRawDataset, build_dataset(), load_constants()
│   └── data_utils.py       # normalize_fn / denormalize_fn
├── utils/
│   ├── checkpoint.py       # save/load/auto_resume checkpoint
│   └── flip_metrics.py     # FLIP 指标计算
├── train/
│   ├── train_dcae.py       # DCAE 训练主脚本
│   └── train_vqvae.py      # VQ-VAE 训练脚本
├── eval/
│   ├── eval_flip.py        # 局部对比度翻转评估
│   └── eval_multi_seed.py  # 多种子训练结果对比分析
├── script/
│   ├── shell/              # 训练启动脚本（torchrun DDP）
│   └── notebook/           # 评估用 Jupyter notebooks
├── config.py               # DatasetConfig dataclass，从 datasets.yaml 加载
├── datasets.yaml           # 数据集路径与元信息
└── sync.sh                 # rsync 代码到远程服务器 test1
```

### DCAE (`model/dcae.py`)

- **编码器**：Stem Conv → N 个 EncoderStage（ResBlock + 可选 SpatialAttention + 2x 下采样）
- **Bottleneck**：RMSNorm → Conv1x1 → **Softplus**（强制 z > 0，消除符号翻转）
- **解码器**：对称的 DecoderStage（2x 上采样 + ResBlock + 可选 SpatialAttention）→ Final Conv
- **Padding**：reflect padding 保证空间尺寸可被 2^num_stages 整除
- 默认配置：base_channels=64, channel_multipliers=(1,2,4,8), latent_channels=16
- 空间压缩 16x → 潜空间 shape `[B, 16, ~16, ~19]`

### Training (`train/train_dcae.py`)

**损失函数**：
- `masked_recon_loss`：0.5×L1 + 0.5×L2，海陆 mask 加权
- `fft_spectral_loss`：2D FFT 振幅谱 L1 损失
- `latent_reg`：z 的 L2 正则，防止 Softplus 输出无界
- `structured_loss`（DC-AE 1.5）：随机选取前 c' 个通道解码，强制通道按重要性排序

**训练设置**：
- 分布式 DDP（torchrun），AMP 混合精度
- AdamW + CosineAnnealingLR
- 默认 6 GPU, batch_size=16, epochs=500, lr=2e-4

## Remote Environment

- 远程服务器别名：`test1`
- 远程项目路径：`/test1/hyj/SubRegion-AE`
- 数据路径前缀：`/test1/dataset/DA/subregion/glorys12_kuroshio_extension/`
- 使用 `sync.sh` 将本地代码 rsync 到远程

## Development Workflow

1. 本地编辑代码
2. `./sync.sh` 同步到远程服务器
3. 在远程运行 `bash script/shell/run_dcae_single.sh`
4. 用 `script/notebook/eval_dcae.ipynb` 评估结果

## Conventions

- Python 3，PyTorch
- 模型定义在 `model/` 目录，单数命名
- 训练脚本在项目根目录，命名 `train_{model}.py`
- 启动脚本在 `script/shell/`，命名 `run_{model}_single.sh`
- Checkpoint 保存在 `output/{data_name}/{tag}/`
- TensorBoard 日志在 checkpoint 目录下的 `tb/` 子目录
