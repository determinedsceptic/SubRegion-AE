import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        norm = x.pow(2).mean(dim=1, keepdim=True).add(self.eps).sqrt()
        return x / norm * self.weight[None, :, None, None]


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.norm1 = RMSNorm(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = RMSNorm(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.act = nn.SiLU()
        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act(self.norm1(x)))
        h = self.conv2(self.act(self.norm2(h)))
        return self.act(h + self.shortcut(x))


class SpatialAttention(nn.Module):
    def __init__(self, channels: int, num_heads: int = 8):
        super().__init__()
        self.norm = RMSNorm(channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        h = h.view(B, C, H * W).permute(0, 2, 1)  # [B, HW, C]
        h, _ = self.attn(h, h, h, need_weights=False)
        h = h.permute(0, 2, 1).view(B, C, H, W)
        return x + h


class EncoderStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_res_blocks: int,
        use_attention: bool = False,
        num_heads: int = 8,
    ):
        super().__init__()
        blocks = []
        cur = in_channels
        for _ in range(num_res_blocks):
            blocks.append(ResBlock(cur, out_channels))
            cur = out_channels
        if use_attention:
            blocks.append(SpatialAttention(out_channels, num_heads))
        self.blocks = nn.Sequential(*blocks)
        self.downsample = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.blocks(x)
        return self.downsample(x)


class DecoderStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_res_blocks: int,
        use_attention: bool = False,
        num_heads: int = 8,
    ):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
        )
        blocks = []
        for _ in range(num_res_blocks):
            blocks.append(ResBlock(out_channels, out_channels))
        if use_attention:
            blocks.append(SpatialAttention(out_channels, num_heads))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        return self.blocks(x)


class DCAE(nn.Module):
    def __init__(
        self,
        in_channels: int = 101,
        base_channels: int = 64,
        channel_multipliers=(1, 2, 4, 8),
        latent_channels: int = 16,
        num_res_blocks: int = 2,
        attention_resolutions=(1, 2),
        num_heads: int = 8,
    ):
        super().__init__()
        ch = [base_channels * m for m in channel_multipliers]
        num_stages = len(channel_multipliers) - 1

        # Validate attention_resolutions indices
        for idx in attention_resolutions:
            if idx >= num_stages:
                raise ValueError(
                    f"attention_resolutions index {idx} is out of range for "
                    f"num_stages={num_stages} (valid: 0..{num_stages - 1})"
                )

        self.num_stages = num_stages

        # Stem
        self.stem = nn.Conv2d(in_channels, ch[0], 3, padding=1)

        # Encoder stages
        enc_stages = []
        for i in range(num_stages):
            enc_stages.append(EncoderStage(
                in_channels=ch[i],
                out_channels=ch[i + 1],
                num_res_blocks=num_res_blocks,
                use_attention=(i in attention_resolutions),
                num_heads=num_heads,
            ))
        self.encoder_stages = nn.ModuleList(enc_stages)

        # Bottleneck — Log-Normal VAE: h → μ, log_σ → z = exp(μ + σ·ε) > 0
        self.encode_norm = RMSNorm(ch[-1])
        self.encode_mu = nn.Conv2d(ch[-1], latent_channels, 1)
        self.encode_logvar = nn.Conv2d(ch[-1], latent_channels, 1)
        self.decode_proj = nn.Conv2d(latent_channels, ch[-1], 1)

        # Decoder stages (symmetric, reversed)
        dec_stages = []
        for i in range(num_stages - 1, -1, -1):
            dec_stages.append(DecoderStage(
                in_channels=ch[i + 1],
                out_channels=ch[i],
                num_res_blocks=num_res_blocks,
                use_attention=(i in attention_resolutions),
                num_heads=num_heads,
            ))
        self.decoder_stages = nn.ModuleList(dec_stages)

        # Final output layer
        self.final = nn.Sequential(
            ResBlock(ch[0], ch[0]),
            nn.Conv2d(ch[0], in_channels, 1),
        )

    def _pad(self, x: torch.Tensor):
        """Pad to next multiple of 2^num_stages using reflection padding."""
        s = 2 ** self.num_stages
        H, W = x.shape[2], x.shape[3]
        pH = (s - H % s) % s
        pW = (s - W % s) % s
        if pH > 0 or pW > 0:
            # F.pad order: (left, right, top, bottom)
            x = F.pad(x, (0, pW, 0, pH), mode='reflect')
        return x, H, W

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode x to latent exp(mu) (deterministic, for inference/DA)."""
        x, _, _ = self._pad(x)
        h = self.stem(x)
        for stage in self.encoder_stages:
            h = stage(h)
        h = self.encode_norm(h)
        return torch.exp(self.encode_mu(h))

    def encode_distribution(self, x: torch.Tensor):
        """Encode x to (mu, logvar) for training with KL loss."""
        x, _, _ = self._pad(x)
        h = self.stem(x)
        for stage in self.encoder_stages:
            h = stage(h)
        h = self.encode_norm(h)
        return self.encode_mu(h), self.encode_logvar(h)

    def decode(self, z: torch.Tensor, target_shape) -> torch.Tensor:
        h = self.decode_proj(z)
        for stage in self.decoder_stages:
            h = stage(h)
        h = self.final(h)
        # Crop back to original spatial size (removes reflect-padded region)
        h = h[:, :, :target_shape[0], :target_shape[1]]
        return h

    def forward(self, x: torch.Tensor, return_latent: bool = False):
        target_shape = (x.shape[2], x.shape[3])
        x_padded, _, _ = self._pad(x)
        h = self.stem(x_padded)
        for stage in self.encoder_stages:
            h = stage(h)
        h = self.encode_norm(h)
        mu = self.encode_mu(h)
        logvar = self.encode_logvar(h)
        # Log-Normal reparameterization: z = exp(mu + sigma * epsilon) > 0 always
        if self.training:
            z = torch.exp(mu + torch.exp(0.5 * logvar) * torch.randn_like(mu))
        else:
            z = torch.exp(mu)
        h = self.decode_proj(z)
        for stage in self.decoder_stages:
            h = stage(h)
        h = self.final(h)
        recon = h[:, :, :target_shape[0], :target_shape[1]]
        if return_latent:
            return recon, mu, logvar, z
        return recon
