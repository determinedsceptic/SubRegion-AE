"""
Data assimilation utilities with sign-stable latent-space encoding.

Overview
--------
This module provides helper functions for running data assimilation (DA)
experiments (e.g. Ensemble Kalman Filter, 4D-Var) in the DCAE latent space
with guaranteed sign consistency.

Sign-flip problem during assimilation
--------------------------------------
Even after training a sign-stable DCAE, sign flips can still emerge during
the DA update step because:

* In **EnKF**: the ensemble analysis update is linear in the latent space but
  can produce analysis members that drift to the "mirror" half-space when the
  background ensemble spread is large.
* In **4D-Var / gradient-based optimization**: the cost-function gradient can
  push a latent iterate across the sign boundary (z_i = 0 for some dimension i).

Solution: background-anchored sign canonicalization
----------------------------------------------------
All public functions in this module accept a ``background_z`` argument.
After every latent-space update the canonicalized background latent code is
used as a sign reference to correct the analysis latent codes:

    z_a_corrected = sign_canonicalizer.canonicalize(z_a, reference_z=z_b)

This ensures the decoded analysis fields are physically consistent with the
background regardless of how the optimization trajectory evolved.

Ensemble Kalman Filter (EnKF) in latent space
----------------------------------------------
``ensemble_enkf_update`` implements the standard EnKF analysis equations:

    z_a = z_b + K (y − H z_b)

where:
* z_b  – background ensemble matrix  (N × D)
* y    – observation vector           (p,)
* H    – observation operator         (p × D), linear in latent space
* K    – Kalman gain  K = P_b H^T (H P_b H^T + R)^{-1}
* P_b  – background latent covariance (D × D)
* R    – observation error covariance (p × p)
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from subregion_ae.models.dcae import DCAE
from subregion_ae.models.sign_correction import SignCanonicalizer


def encode_ensemble(
    model: DCAE,
    x_ensemble: torch.Tensor,
    x_background: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Encode an ensemble of ocean state fields to sign-canonical latent codes.

    All ensemble members are canonicalized against the background latent code
    so that ensemble statistics (mean, covariance) are computed in a
    sign-consistent latent space.

    Parameters
    ----------
    model:
        Trained ``DCAE`` instance (in eval mode).
    x_ensemble:
        Ensemble of ocean state fields, shape ``(N, C, H, W)`` where
        ``N`` is the ensemble size.
    x_background:
        Background (prior mean) ocean state field, shape ``(1, C, H, W)``
        or ``(C, H, W)``.  If ``None``, the first ensemble member is used as
        the background reference for sign canonicalization.

    Returns
    -------
    z_ensemble : torch.Tensor
        Sign-canonical latent codes for all ensemble members, shape
        ``(N, latent_dim)``.
    z_background : torch.Tensor
        Latent code of the background state, shape ``(1, latent_dim)``.
    """
    model.eval()
    with torch.no_grad():
        if x_background is None:
            x_background = x_ensemble[:1]

        if x_background.dim() == 3:
            x_background = x_background.unsqueeze(0)

        # Encode background — uses the model's internal EMA reference.
        z_background = model.encode(x_background)  # (1, D)

        # Encode all ensemble members, aligning their signs with z_background.
        z_ensemble = model.encode(x_ensemble, ref_z=z_background)  # (N, D)

    return z_ensemble, z_background


def ensemble_enkf_update(
    z_background: torch.Tensor,
    z_ensemble: torch.Tensor,
    y_obs: torch.Tensor,
    H: torch.Tensor,
    R: torch.Tensor,
    sign_canonicalizer: Optional[SignCanonicalizer] = None,
    inflation_factor: float = 1.0,
) -> torch.Tensor:
    """Standard deterministic EnKF analysis update in the latent space.

    Applies the ensemble Kalman update in the DCAE latent space and
    optionally sign-corrects the result to stay consistent with the
    background.

    Parameters
    ----------
    z_background:
        Background ensemble mean in latent space, shape ``(1, D)`` or
        ``(D,)``.
    z_ensemble:
        Background ensemble matrix, shape ``(N, D)``.
    y_obs:
        Observation vector, shape ``(p,)``.
    H:
        Linear observation operator in latent space, shape ``(p, D)``.
    R:
        Observation error covariance matrix, shape ``(p, p)``.
    sign_canonicalizer:
        ``SignCanonicalizer`` from the DCAE model.  When provided, each
        analysis ensemble member is canonicalized against ``z_background``
        after the update to prevent sign flips.
    inflation_factor:
        Multiplicative covariance inflation (>1 widens the ensemble spread).

    Returns
    -------
    torch.Tensor
        Analysis ensemble in latent space, shape ``(N, D)``.
    """
    z_b = z_background.squeeze(0)  # (D,)
    N, D = z_ensemble.shape

    # Ensemble anomalies (deviations from the ensemble mean).
    z_mean = z_ensemble.mean(dim=0)  # (D,)
    A = (z_ensemble - z_mean.unsqueeze(0)) * inflation_factor  # (N, D)

    # Background covariance (sample, unbiased): P_b = A^T A / (N-1)
    P_b = (A.T @ A) / max(N - 1, 1)  # (D, D)

    # Innovation: d = y - H z_mean
    y_pred = H @ z_mean  # (p,)
    innovation = y_obs - y_pred  # (p,)

    # Kalman gain: K = P_b H^T (H P_b H^T + R)^{-1}
    HP_b = H @ P_b  # (p, D)
    S = HP_b @ H.T + R  # (p, p)  — innovation covariance
    K = P_b @ H.T @ torch.linalg.inv(S)  # (D, p)

    # Deterministic EnKF: all members receive the same mean increment.
    increment = (K @ innovation).unsqueeze(0)  # (1, D)
    z_analysis = z_ensemble + increment  # (N, D)

    # Sign canonicalization: align analysis members with the background.
    if sign_canonicalizer is not None:
        z_analysis = sign_canonicalizer.canonicalize(
            z_analysis,
            reference_z=z_background,
        )

    return z_analysis


def decode_ensemble(
    model: DCAE,
    z_ensemble: torch.Tensor,
    z_background: torch.Tensor,
    output_shape: Optional[Tuple[int, int]] = None,
) -> torch.Tensor:
    """Decode an analysis ensemble from the latent space to physical space.

    Before decoding, each member is sign-canonicalized against the background
    latent code to guard against sign flips introduced during the DA update.

    Parameters
    ----------
    model:
        Trained ``DCAE`` instance (in eval mode).
    z_ensemble:
        Analysis ensemble latent codes, shape ``(N, latent_dim)``.
    z_background:
        Background latent code used as sign reference, shape
        ``(1, latent_dim)`` or ``(latent_dim,)``.
    output_shape:
        Optional ``(H, W)`` for the decoded fields.

    Returns
    -------
    torch.Tensor
        Analysis ocean state fields, shape ``(N, C, H, W)``.
    """
    model.eval()
    with torch.no_grad():
        # Canonicalize signs before decoding.
        z_corrected = model.sign_canonicalizer.canonicalize(
            z_ensemble, reference_z=z_background
        )
        x_analysis = model.decode(
            z_corrected, output_shape=output_shape
        )
    return x_analysis
