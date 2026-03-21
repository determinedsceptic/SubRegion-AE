"""
Ocean state encoder for the DCAE.

Architecture
------------
A stack of strided 2-D convolution blocks that progressively halve the spatial
resolution, followed by an adaptive average-pool and a linear projection to the
latent dimension.

Input format
------------
The encoder expects tensors of shape ``(B, C_in, H, W)`` where:

* ``B`` – batch size
* ``C_in`` – number of input channels, typically ``n_vars × n_levels``
  (e.g. 2 variables × 101 levels = 202 channels for temperature and salinity
  in the Kuroshio Extension experiments).
* ``H, W`` – horizontal grid dimensions.

Activation choice
-----------------
ELU is used throughout so that the *decoder* can also use ELU and still
produce D(z) ≠ D(−z) (see ``OceanDecoder``).  Symmetric activations such as
``tanh`` or ``sin`` would weaken the asymmetry we need in the decoder.

Batch normalisation is included after every convolution to stabilise training
with high-dimensional multi-channel input.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class ConvBlock(nn.Sequential):
    """Conv2d → BatchNorm2d → ELU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
    ) -> None:
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                      padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ELU(inplace=True),
        )


class OceanEncoder(nn.Module):
    """Convolutional encoder for multi-channel ocean state fields.

    Parameters
    ----------
    in_channels:
        Number of input channels (``n_vars × n_levels``).
    latent_dim:
        Dimension of the output latent vector.
    hidden_channels:
        Channel widths for successive convolution blocks.  Each block
        downsamples the spatial resolution by a factor of 2 (stride=2).
    """

    def __init__(
        self,
        in_channels: int,
        latent_dim: int,
        hidden_channels: List[int] | None = None,
    ) -> None:
        super().__init__()
        if hidden_channels is None:
            hidden_channels = [128, 256, 256, 512]

        layers: List[nn.Module] = []
        prev_ch = in_channels
        for ch in hidden_channels:
            layers.append(ConvBlock(prev_ch, ch, kernel_size=3, stride=2))
            prev_ch = ch

        self.conv_layers = nn.Sequential(*layers)
        # Pool to a fixed 4×4 spatial footprint regardless of input resolution.
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(prev_ch * 4 * 4, latent_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch of ocean state fields to latent codes.

        Parameters
        ----------
        x:
            Input tensor ``(B, C_in, H, W)``.

        Returns
        -------
        torch.Tensor
            Latent codes ``(B, latent_dim)`` — **no** activation applied so
            the codes are unconstrained real vectors.
        """
        h = self.conv_layers(x)
        h = self.pool(h)
        h = self.flatten(h)
        return self.fc(h)
