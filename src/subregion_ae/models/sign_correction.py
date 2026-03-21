"""
Sign Canonicalization for a stable DCAE latent space.

Background
----------
A convolutional autoencoder trained with a symmetric reconstruction loss
(e.g. MSE) can develop an approximate symmetry  D(z) ≈ D(−z)  in its decoder.
When this happens the encoder is free to map the same physical state **x** to
either **z** or **−z** depending on the random initialisation or on how the
optimisation trajectory evolves.  In an ensemble data assimilation (DA) loop
this "sign-flip" instability means that two ensemble members that represent
nearly identical ocean states can land in opposite latent half-spaces, making
ensemble statistics (mean, covariance) physically meaningless and causing
spurious structures after decoding.

Solution
--------
``SignCanonicalizer`` enforces a *canonical* sign convention for every latent
dimension by:

* Maintaining an **exponential moving average (EMA)** of latent codes seen
  during training as a reference direction ``self.reference``.
* Applying a **per-dimension sign correction** so that every encoded latent
  vector is aligned with the reference:  ``sign(z_i) == sign(reference_i)``.
* Providing a ``canonicalize`` method for use inside the DA loop, where the
  background-state latent code acts as the sign reference.  This ensures that
  all ensemble perturbations remain on the same side as the background,
  preventing sign flips during assimilation iterations.

Gradient flow
-------------
The sign correction is a multiplication by ±1.  Where ``z_i ≠ 0`` this is a
piecewise-constant function with derivative 0 almost everywhere, but it does
*not* block gradients — the correction factor is treated as a constant and the
gradient of the loss with respect to ``z_i`` passes through unchanged in
magnitude.  In practice this is sufficient for stable training because the
``SignAlignmentLoss`` provides an explicit gradient signal that steers the
encoder to produce codes that need *fewer* corrections over time.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional


class SignCanonicalizer(nn.Module):
    """Enforce a consistent sign convention across all latent codes.

    Parameters
    ----------
    latent_dim:
        Dimensionality of the latent space.
    ema_momentum:
        Momentum for the exponential moving average of the reference
        direction.  Smaller values → slower adaptation (more stable
        reference); larger values → faster adaptation.  Typical range
        0.01–0.10.

    Buffers (not learnable parameters)
    ------------------------------------
    reference:
        Running EMA of the latent codes seen during training.  Acts as
        the canonical "positive" direction for each latent dimension.
    _initialized:
        Boolean flag; set to True after the first training forward pass.
    """

    def __init__(self, latent_dim: int, ema_momentum: float = 0.05) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.ema_momentum = ema_momentum

        self.register_buffer("reference", torch.zeros(latent_dim))
        self.register_buffer("_initialized", torch.tensor(False))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _update_reference(self, z: torch.Tensor) -> None:
        """Update the EMA reference from the current batch of latent codes."""
        batch_mean = z.detach().mean(dim=0)
        if not self._initialized.item():
            self.reference.copy_(batch_mean)
            self._initialized.fill_(True)
        else:
            self.reference.lerp_(batch_mean, self.ema_momentum)

    @staticmethod
    def _sign_correction(z: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """Return a {−1, +1} correction tensor that aligns *z* with *ref*.

        For each element:
        * ``+1``  if ``z`` and ``ref`` already have the same sign.
        * ``−1``  if they have opposite signs (flip needed).
        * ``+1``  for any zero element (default: keep as-is).

        After applying the correction ``z_corrected = z * correction``,
        every element of ``z_corrected`` will have the same sign as the
        corresponding element of ``ref``.
        """
        product = z * ref
        correction = torch.sign(product)
        # Replace 0 with +1: zero elements are left unchanged.
        return torch.where(correction == 0, torch.ones_like(correction), correction)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward(
        self,
        z: torch.Tensor,
        reference_z: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply sign canonicalization and (during training) update the EMA.

        Parameters
        ----------
        z:
            Raw latent codes from the encoder, shape ``(batch, latent_dim)``.
        reference_z:
            Optional external reference latent code(s).

            * If provided (shape ``(batch, latent_dim)`` **or**
              ``(latent_dim,)``), each sample in ``z`` is aligned with the
              corresponding row of ``reference_z``.  Pass the *background*
              latent code here during data assimilation so that all ensemble
              members are canonicalized against the same physical reference.
            * If ``None``, the internal EMA reference is used.

        Returns
        -------
        torch.Tensor
            Sign-corrected latent codes with the same shape and magnitude
            as ``z`` but with signs aligned to ``reference_z`` (or the
            internal EMA reference).
        """
        if reference_z is not None:
            ref = reference_z
        else:
            # No external reference: use (and optionally update) the EMA reference.
            if self.training:
                # Update *before* applying correction so this batch's statistics
                # are incorporated into the reference immediately (including the
                # very first batch that initialises the reference).
                self._update_reference(z)
            if not self._initialized.item():
                # Still uninitialized (eval mode with no prior training call).
                return z
            ref = self.reference.unsqueeze(0)  # (1, latent_dim) → broadcast

        correction = self._sign_correction(z, ref)
        return z * correction

    def canonicalize(
        self,
        z: torch.Tensor,
        reference_z: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Convenience alias for inference / assimilation (non-training) use.

        Identical to :meth:`forward` but never updates the EMA reference.
        Use this inside the DA loop to canonicalize ensemble perturbations
        without accidentally modifying the trained model state.

        Parameters
        ----------
        z:
            Latent codes to canonicalize, shape ``(batch, latent_dim)``.
        reference_z:
            Reference latent code(s).  Typically the background latent code
            ``z_b = encode(x_b)`` so that all ensemble members are aligned
            with the background.

        Returns
        -------
        torch.Tensor
            Sign-corrected latent codes.
        """
        was_training = self.training
        self.eval()
        try:
            return self.forward(z, reference_z=reference_z)
        finally:
            self.train(was_training)
