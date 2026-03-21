"""
Sign alignment loss for stable DCAE training.

Purpose
-------
Even with an asymmetric decoder (ELU activations), the encoder gradient
landscape has many saddle points where the sign of a latent dimension can
be arbitrary.  The ``SignAlignmentLoss`` provides an explicit, differentiable
signal that steers the encoder to produce codes that are *already aligned*
with the running EMA reference stored inside a ``SignCanonicalizer``.

Conceptually the loss penalises encoder outputs where ``sign(z_i)`` would
disagree with ``sign(reference_i)`` (i.e. those that require a hard flip in
the canonicalization step).  This reduces the frequency of hard sign
corrections during training and produces a smoother, more consistent latent
space.

Loss formulation
----------------
For a batch of latent codes ``z`` (shape ``(B, D)``) and the EMA reference
``r`` (shape ``(D,)``), the loss is:

    L_sign = mean( ReLU( −z · r̂ )² )

where ``r̂ = r / (‖r‖ + ε)`` is the unit-normalized reference.

* ``z_i · r̂_i > 0``  → same sign → ReLU term is 0 → no penalty.
* ``z_i · r̂_i < 0``  → opposite sign → quadratic penalty proportional
  to how far the code is on the "wrong" side.

This is a *soft* constraint: it does not force the decoder to ignore negative
latent values, it simply provides a gradient signal that gradually steers the
encoder toward the sign convention established by the EMA reference.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from subregion_ae.models.sign_correction import SignCanonicalizer


class SignAlignmentLoss(nn.Module):
    """Differentiable penalty for latent sign misalignment.

    Parameters
    ----------
    sign_canonicalizer:
        The ``SignCanonicalizer`` instance attached to the DCAE model.
        The loss reads its ``reference`` buffer, so both must refer to the
        *same* object.
    weight:
        Scalar weight λ applied to the sign alignment loss before it is
        added to the reconstruction loss.  Typical values 0.001–0.01.
    eps:
        Small constant added to the reference norm for numerical stability.
    """

    def __init__(
        self,
        sign_canonicalizer: SignCanonicalizer,
        weight: float = 0.01,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.sign_canonicalizer = sign_canonicalizer
        self.weight = weight
        self.eps = eps

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Compute the weighted sign alignment loss.

        Parameters
        ----------
        z:
            Latent codes produced by the encoder (after ``SignCanonicalizer``
            has already been applied), shape ``(B, latent_dim)``.

        Returns
        -------
        torch.Tensor
            Scalar loss value: ``weight × mean( ReLU(−z · r̂)² )``.
            Returns ``torch.tensor(0.)`` before the EMA reference has been
            initialised (i.e. before the first training batch).
        """
        if not self.sign_canonicalizer._initialized.item():
            return z.new_tensor(0.0)

        ref = self.sign_canonicalizer.reference  # (latent_dim,)
        ref_norm = ref / (ref.norm() + self.eps)  # unit vector

        # Project each latent dimension onto the reference unit vector.
        # Shape: (B, latent_dim)
        projection = z * ref_norm.unsqueeze(0)

        # Penalise components that point *away* from the reference direction.
        penalty = F.relu(-projection).pow(2).mean()

        return self.weight * penalty
