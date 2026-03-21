"""
DCAE – Deep Convolutional AutoEncoder with sign-stabilized latent space.

This module provides the top-level ``DCAE`` class that combines:

* ``OceanEncoder``  – strided Conv2d blocks + linear projection.
* ``SignCanonicalizer`` – EMA-based per-dimension sign correction.
* ``OceanDecoder``  – transposed-Conv2d blocks with ELU activations.

Sign-flip instability and its resolution
-----------------------------------------
When a symmetric decoder (one where D(z) ≈ D(−z)) is used inside an ensemble
or variational data assimilation (DA) loop, the latent codes of different
ensemble members or DA iterates can land in opposite half-spaces of the latent
space.  Ensemble statistics (mean, covariance) then become meaningless and the
decoded physical fields contain spurious features.

This implementation resolves the problem through two complementary mechanisms:

1. **Asymmetric decoder activations (ELU)**
   The ``OceanDecoder`` uses ELU between every transposed-convolution block.
   Because ELU(t) ≠ −ELU(−t), the decoder mapping D satisfies D(z) ≠ D(−z)
   for almost all z.  This breaks the encoder symmetry at the objective level:
   the reconstruction loss L = ‖D(E(x)) − x‖² now penalises sign flips, so
   gradient descent converges to a unique sign convention.

2. **SignCanonicalizer (EMA-based per-dimension sign correction)**
   Even with an asymmetric decoder, different training runs or different
   ensemble members can, in principle, settle on opposite sign conventions.
   The ``SignCanonicalizer`` maintains a running EMA of the latent codes seen
   during training and, at every forward pass, corrects each latent dimension
   so that its sign matches the EMA reference.  During DA, the background
   state latent code is used as the reference so that all ensemble members
   are aligned with the background.

Usage
-----
Training::

    model = DCAE(in_channels=202, latent_dim=64, spatial_shape=(160, 180))
    x_hat, z = model(x)
    loss = F.mse_loss(x_hat, x) + sign_loss(z)  # see SignAlignmentLoss
    loss.backward()

Inference / encoding for DA::

    model.eval()
    z_b = model.encode(x_b)                          # background
    z_members = model.encode(x_members, ref_z=z_b)   # ensemble
    # ... assimilation update in latent space ...
    x_a = model.decode(z_updated, ref_z=z_b)          # back to physical space
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from subregion_ae.models.decoder import OceanDecoder
from subregion_ae.models.encoder import OceanEncoder
from subregion_ae.models.sign_correction import SignCanonicalizer


class DCAE(nn.Module):
    """Deep Convolutional AutoEncoder with sign-stabilized latent space.

    Parameters
    ----------
    in_channels:
        Number of input channels, i.e. ``n_vars × n_levels``.
        For the Kuroshio Extension experiments with temperature and salinity
        at 101 vertical levels this is 2 × 101 = 202.
    latent_dim:
        Dimensionality of the compressed latent space.
    spatial_shape:
        ``(H, W)`` of the input / output spatial grid.
    hidden_channels:
        Channel widths for successive encoder conv blocks (decoder uses the
        reverse order).  Default ``[128, 256, 256, 512]``.
    ema_momentum:
        EMA momentum for the ``SignCanonicalizer`` reference update.
        Smaller → slower, more stable reference; larger → faster adaptation.
    """

    def __init__(
        self,
        in_channels: int,
        latent_dim: int,
        spatial_shape: Tuple[int, int],
        hidden_channels: List[int] | None = None,
        ema_momentum: float = 0.05,
    ) -> None:
        super().__init__()

        if hidden_channels is None:
            hidden_channels = [128, 256, 256, 512]

        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.spatial_shape = spatial_shape

        self.encoder = OceanEncoder(
            in_channels=in_channels,
            latent_dim=latent_dim,
            hidden_channels=hidden_channels,
        )
        self.sign_canonicalizer = SignCanonicalizer(
            latent_dim=latent_dim,
            ema_momentum=ema_momentum,
        )
        self.decoder = OceanDecoder(
            latent_dim=latent_dim,
            out_channels=in_channels,
            hidden_channels=hidden_channels,
            output_shape=spatial_shape,
        )

    # ------------------------------------------------------------------
    # Core encode / decode helpers
    # ------------------------------------------------------------------

    def encode(
        self,
        x: torch.Tensor,
        ref_z: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode ocean state fields to sign-canonical latent codes.

        Parameters
        ----------
        x:
            Input tensor ``(B, C_in, H, W)``.
        ref_z:
            Optional reference latent code(s) for sign alignment.

            * During **training** leave this as ``None``: the internal EMA
              reference is used and updated automatically.
            * During **assimilation** pass the *background* latent code
              ``z_b = encode(x_b)`` so that every ensemble member is
              canonicalized against the same physical reference, preventing
              sign flips across the ensemble.

        Returns
        -------
        torch.Tensor
            Sign-corrected latent codes ``(B, latent_dim)``.
        """
        z_raw = self.encoder(x)
        return self.sign_canonicalizer(z_raw, reference_z=ref_z)

    def decode(
        self,
        z: torch.Tensor,
        ref_z: Optional[torch.Tensor] = None,
        output_shape: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """Decode latent codes to ocean state fields.

        Parameters
        ----------
        z:
            Latent codes ``(B, latent_dim)``.
        ref_z:
            Optional reference latent code for sign canonicalization.
            When provided, ``z`` is sign-corrected *before* decoding.  This
            is important inside a DA update loop where the optimization
            trajectory can sometimes flip latent signs: canonicalizing before
            every decode call prevents spurious reconstructions.
        output_shape:
            Optional ``(H, W)`` override for the spatial output size.

        Returns
        -------
        torch.Tensor
            Reconstructed ocean state fields ``(B, C_in, H, W)``.
        """
        if ref_z is not None:
            z = self.sign_canonicalizer.canonicalize(z, reference_z=ref_z)
        return self.decoder(z, output_shape=output_shape)

    # ------------------------------------------------------------------
    # Full forward pass (training)
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode then decode a batch of ocean state fields.

        Parameters
        ----------
        x:
            Input ocean state fields ``(B, C_in, H, W)``.

        Returns
        -------
        x_hat : torch.Tensor
            Reconstructed fields ``(B, C_in, H, W)``.
        z : torch.Tensor
            Sign-canonical latent codes ``(B, latent_dim)``.
            Returned so that the caller can compute auxiliary losses
            (e.g. ``SignAlignmentLoss``) without re-encoding.
        """
        z = self.encode(x)
        x_hat = self.decoder(z, output_shape=x.shape[-2:])
        return x_hat, z
