"""
Tests for SignAlignmentLoss.

Invariants verified
-------------------
1. Loss is a non-negative scalar.
2. Loss is zero before the EMA reference is initialized.
3. Loss is zero when all latent codes are perfectly aligned with the reference.
4. Loss is positive when codes are misaligned (pointing away from reference).
5. Gradient flows from the loss to the latent codes (and therefore to encoder params).
6. The ``weight`` parameter scales the loss correctly.
"""

import pytest
import torch

from subregion_ae.models.sign_correction import SignCanonicalizer
from subregion_ae.losses.sign_regularization import SignAlignmentLoss


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DIM = 16


@pytest.fixture
def sc() -> SignCanonicalizer:
    sc = SignCanonicalizer(latent_dim=DIM, ema_momentum=0.1)
    # Seed a positive reference.
    with torch.no_grad():
        sc.reference.copy_(torch.ones(DIM))
        sc._initialized.fill_(True)
    return sc


@pytest.fixture
def loss_fn(sc: SignCanonicalizer) -> SignAlignmentLoss:
    return SignAlignmentLoss(sign_canonicalizer=sc, weight=1.0)


# ---------------------------------------------------------------------------
# Basic properties
# ---------------------------------------------------------------------------

class TestBasicProperties:
    def test_loss_is_non_negative(self, loss_fn: SignAlignmentLoss) -> None:
        z = torch.randn(8, DIM)
        loss = loss_fn(z)
        assert loss.item() >= 0.0

    def test_loss_is_scalar(self, loss_fn: SignAlignmentLoss) -> None:
        z = torch.randn(8, DIM)
        loss = loss_fn(z)
        assert loss.shape == torch.Size([])

    def test_zero_before_init(self) -> None:
        """Loss must return 0 tensor when EMA is not yet initialized."""
        sc = SignCanonicalizer(latent_dim=DIM)
        loss_fn = SignAlignmentLoss(sc, weight=1.0)
        z = torch.randn(4, DIM)
        loss = loss_fn(z)
        assert loss.item() == 0.0


# ---------------------------------------------------------------------------
# Alignment sensitivity
# ---------------------------------------------------------------------------

class TestAlignmentSensitivity:
    def test_zero_loss_when_perfectly_aligned(self, loss_fn: SignAlignmentLoss) -> None:
        """Positive latent codes aligned with positive reference → loss = 0."""
        z = torch.abs(torch.randn(8, DIM)) + 0.1  # strictly positive
        loss = loss_fn(z)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_positive_loss_when_misaligned(self, loss_fn: SignAlignmentLoss) -> None:
        """Negative latent codes pointing against positive reference → loss > 0."""
        z = -torch.abs(torch.randn(8, DIM)) - 0.1  # strictly negative
        loss = loss_fn(z)
        assert loss.item() > 0.0

    def test_loss_increases_with_misalignment(self, loss_fn: SignAlignmentLoss) -> None:
        """Larger misalignment → larger loss."""
        z_small = -torch.ones(8, DIM) * 0.1
        z_large = -torch.ones(8, DIM) * 5.0
        assert loss_fn(z_large).item() > loss_fn(z_small).item()


# ---------------------------------------------------------------------------
# Weight parameter
# ---------------------------------------------------------------------------

class TestWeightScaling:
    def test_weight_scales_loss(self, sc: SignCanonicalizer) -> None:
        """Loss should be proportional to the weight parameter."""
        z = -torch.abs(torch.randn(8, DIM)) - 0.5
        loss_1 = SignAlignmentLoss(sc, weight=1.0)(z).item()
        loss_2 = SignAlignmentLoss(sc, weight=2.0)(z).item()
        assert loss_2 == pytest.approx(2.0 * loss_1, rel=1e-4)

    def test_zero_weight_returns_zero(self, sc: SignCanonicalizer) -> None:
        z = torch.randn(8, DIM)
        loss = SignAlignmentLoss(sc, weight=0.0)(z)
        assert loss.item() == pytest.approx(0.0, abs=1e-8)


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------

class TestGradientFlow:
    def test_gradient_flows_to_z(self, loss_fn: SignAlignmentLoss) -> None:
        """Gradient should reach the latent codes for misaligned inputs."""
        z = torch.randn(8, DIM, requires_grad=True)
        # Force misalignment: some dims negative against positive reference.
        z_neg = z - 2.0  # shift to make most values negative
        loss = loss_fn(z_neg)
        if loss.item() > 0:
            loss.backward()
            assert z.grad is not None
            assert z.grad.abs().sum() > 0
