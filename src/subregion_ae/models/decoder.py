"""
Ocean state decoder for the DCAE.

Architecture
------------
A linear projection from the latent vector, followed by a reshape into a small
spatial feature map, and then a stack of transposed-convolution blocks that
progressively double the spatial resolution back to the original input size.

Asymmetric (ELU) activations — the key to sign-flip stability
--------------------------------------------------------------
Every intermediate transposed-convolution block uses **ELU** activation.
ELU(t) = t for t > 0  and  ELU(t) = exp(t) − 1 for t ≤ 0, so
ELU(t) ≠ −ELU(−t) in general.  This asymmetry propagates through the decoder,
ensuring  **D(z) ≠ D(−z)**  for almost all z.

Consequence: the reconstruction loss  L = ‖D(E(x)) − x‖²  is no longer
invariant under the sign flip  z → −z, so gradient descent has a clear signal
to converge to *one* sign convention rather than oscillating between z and −z.
This is the primary structural mechanism that eliminates the sign-flip
instability described in the problem statement.

The *final* block deliberately omits any activation to let the decoder output
cover the full real range of the ocean state variables.

Spatial resolution contract
----------------------------
The decoder is designed as the exact spatial mirror of ``OceanEncoder``:

  OceanEncoder applies N strided-conv blocks (stride=2) + AdaptiveAvgPool(4,4).
  OceanDecoder starts from 4×4 and applies N transposed-conv blocks (stride=2).

  After N blocks the output is  4 × 2^N × 4 × 2^N.
  A final bilinear ``Upsample`` adjusts to any target ``(H, W)`` that was
  passed in at construction time.

Input / output format
---------------------
``forward(z, output_shape)`` accepts an optional ``output_shape`` override so
that the decoder can be used with spatial resolutions not known at construction
time.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DeconvBlock(nn.Sequential):
    """ConvTranspose2d → BatchNorm2d → ELU (asymmetric activation)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
    ) -> None:
        super().__init__(
            nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size,
                stride=stride, padding=padding, bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ELU(inplace=True),  # asymmetric → D(z) ≠ D(−z)
        )


class OceanDecoder(nn.Module):
    """Convolutional decoder for multi-channel ocean state fields.

    Parameters
    ----------
    latent_dim:
        Dimension of the input latent vector.
    out_channels:
        Number of output channels (should equal the encoder's ``in_channels``).
    hidden_channels:
        Channel widths for successive transposed-convolution blocks (listed
        in *encoder* order; the decoder reverses them internally).  Must
        match the ``hidden_channels`` used in ``OceanEncoder``.
    output_shape:
        Expected spatial output ``(H, W)``.  The final ``Upsample`` layer
        maps the decoder's native resolution to this shape.
    """

    def __init__(
        self,
        latent_dim: int,
        out_channels: int,
        hidden_channels: List[int] | None = None,
        output_shape: Tuple[int, int] = (64, 64),
    ) -> None:
        super().__init__()
        if hidden_channels is None:
            hidden_channels = [128, 256, 256, 512]

        # Decode: start from the last (deepest) channel count.
        rev_channels = list(reversed(hidden_channels))
        first_ch = rev_channels[0]

        self.fc = nn.Linear(latent_dim, first_ch * 4 * 4)
        self.reshape_channels = first_ch

        blocks: List[nn.Module] = []
        prev_ch = first_ch
        for ch in rev_channels[1:]:
            blocks.append(DeconvBlock(prev_ch, ch))
            prev_ch = ch

        # Final block: no BN, no activation — full real range output.
        blocks.append(
            nn.ConvTranspose2d(
                prev_ch, out_channels, kernel_size=4, stride=2, padding=1,
            )
        )
        self.deconv_layers = nn.Sequential(*blocks)
        self.output_shape = output_shape

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.ConvTranspose2d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        z: torch.Tensor,
        output_shape: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """Decode latent codes to ocean state fields.

        Parameters
        ----------
        z:
            Latent codes ``(B, latent_dim)``.
        output_shape:
            Optional ``(H, W)`` override.  Falls back to ``self.output_shape``.

        Returns
        -------
        torch.Tensor
            Reconstructed ocean state fields ``(B, out_channels, H, W)``.
        """
        tgt_h, tgt_w = output_shape if output_shape is not None else self.output_shape

        h = self.fc(z)
        h = h.view(h.size(0), self.reshape_channels, 4, 4)
        h = self.deconv_layers(h)

        # Bilinear resize to the exact target resolution.
        if h.shape[-2:] != (tgt_h, tgt_w):
            h = F.interpolate(h, size=(tgt_h, tgt_w), mode="bilinear",
                              align_corners=False)
        return h
