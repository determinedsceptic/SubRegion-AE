"""
SubRegion-AE: Deep Convolutional AutoEncoder with sign-stabilized latent space
for ocean data assimilation in the Kuroshio Extension region.

The primary contribution of this package is a structural solution to the
*sign-flip* instability that arises when a conventional DCAE is used inside an
ensemble / variational data assimilation (DA) loop.

Root cause
----------
A symmetric decoder  D(z) ≈ D(−z)  creates an equivalence class {z, −z} in the
latent space.  Different ensemble members or different DA iterations can settle
in opposite "half-spaces", making ensemble statistics (mean, covariance)
meaningless and producing spurious features in the decoded physical field.

Solution
--------
Three complementary mechanisms are combined:

1. **Asymmetric decoder activations** (ELU):  structurally enforce D(z) ≠ D(−z),
   so the reconstruction loss itself penalises sign flips.

2. **SignCanonicalizer** (EMA reference + per-dimension sign correction):
   applied inside the encoder forward pass to guarantee that all latent codes
   from the same model share the same sign convention.

3. **SignAlignmentLoss**: a differentiable soft penalty that steers gradient
   descent to keep encoded latent dimensions aligned with the running EMA
   reference, reducing how often hard sign corrections are needed.

Together these mechanisms let the DCAE retain its high compression ratio and
low reconstruction error while eliminating latent-space sign-flip instability.
"""

from subregion_ae.models.dcae import DCAE
from subregion_ae.models.sign_correction import SignCanonicalizer
from subregion_ae.losses.sign_regularization import SignAlignmentLoss

__all__ = ["DCAE", "SignCanonicalizer", "SignAlignmentLoss"]
