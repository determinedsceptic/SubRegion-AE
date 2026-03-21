"""
Tests for SignCanonicalizer.

Key invariants verified
-----------------------
1. After canonicalization, every latent dimension has the same sign as the
   reference (alignment property).
2. The magnitude of the latent codes is unchanged (‖z_corrected‖ = ‖z‖).
3. The EMA reference is updated during training forward passes but NOT during
   eval forward passes.
4. A custom ``reference_z`` overrides the internal EMA reference.
5. ``canonicalize()`` never modifies the EMA reference even when in training
   mode.
6. The module serializes / deserializes correctly (state-dict round-trip).
"""

import pytest
import torch

from subregion_ae.models.sign_correction import SignCanonicalizer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def dim() -> int:
    return 32


@pytest.fixture
def canonicalizer(dim: int) -> SignCanonicalizer:
    return SignCanonicalizer(latent_dim=dim, ema_momentum=0.1)


# ---------------------------------------------------------------------------
# Alignment property
# ---------------------------------------------------------------------------

class TestSignAlignment:
    """Verify that the output is always aligned with the reference."""

    def test_aligned_with_ema_reference(self, canonicalizer: SignCanonicalizer, dim: int) -> None:
        """After a training forward pass, output signs match EMA reference."""
        canonicalizer.train()
        z = torch.randn(8, dim)
        z_out = canonicalizer(z)

        ref = canonicalizer.reference
        # For each dimension where ref != 0, sign(z_out) == sign(ref).
        nonzero_mask = ref.abs() > 1e-9
        if nonzero_mask.any():
            assert (torch.sign(z_out[:, nonzero_mask]) == torch.sign(ref[nonzero_mask].unsqueeze(0))).all()

    def test_aligned_with_custom_reference(self, canonicalizer: SignCanonicalizer, dim: int) -> None:
        """When reference_z is provided, output aligns with that reference."""
        canonicalizer.eval()
        z = torch.randn(4, dim)
        ref_z = torch.randn(1, dim)  # arbitrary fixed reference

        z_out = canonicalizer(z, reference_z=ref_z)

        # Each output element should agree in sign with the corresponding reference element.
        nonzero_mask = ref_z.squeeze(0).abs() > 1e-9
        if nonzero_mask.any():
            assert (
                torch.sign(z_out[:, nonzero_mask])
                == torch.sign(ref_z[:, nonzero_mask])
            ).all()

    def test_magnitude_preserved(self, canonicalizer: SignCanonicalizer, dim: int) -> None:
        """Sign correction must not change the magnitude of latent codes."""
        canonicalizer.eval()
        # Seed reference first.
        with torch.no_grad():
            canonicalizer.reference.copy_(torch.randn(dim))
            canonicalizer._initialized.fill_(True)

        z = torch.randn(16, dim)
        z_out = canonicalizer(z)
        torch.testing.assert_close(z.abs(), z_out.abs())

    def test_already_aligned_unchanged(self, canonicalizer: SignCanonicalizer, dim: int) -> None:
        """Codes already aligned with the reference should be returned as-is."""
        canonicalizer.eval()
        ref = torch.ones(dim)
        with torch.no_grad():
            canonicalizer.reference.copy_(ref)
            canonicalizer._initialized.fill_(True)

        # All positive → already aligned with positive reference.
        z = torch.abs(torch.randn(8, dim))
        z_out = canonicalizer(z)
        torch.testing.assert_close(z, z_out)

    def test_opposite_sign_flipped(self, canonicalizer: SignCanonicalizer, dim: int) -> None:
        """Codes opposite to the reference should be negated element-wise."""
        canonicalizer.eval()
        ref = torch.ones(dim)
        with torch.no_grad():
            canonicalizer.reference.copy_(ref)
            canonicalizer._initialized.fill_(True)

        # All negative → should all be flipped to positive.
        z = -torch.abs(torch.randn(8, dim)) - 0.1  # strictly negative
        z_out = canonicalizer(z)
        assert (z_out > 0).all()
        torch.testing.assert_close(z.abs(), z_out.abs())


# ---------------------------------------------------------------------------
# EMA reference update behaviour
# ---------------------------------------------------------------------------

class TestEMAReference:
    def test_reference_updated_in_train_mode(self, canonicalizer: SignCanonicalizer, dim: int) -> None:
        """EMA reference is updated when called in training mode."""
        canonicalizer.train()
        z = torch.ones(4, dim) * 3.0
        canonicalizer(z)
        assert canonicalizer._initialized.item()
        # After first call the reference should be close to z's mean (3.0 per dim).
        assert (canonicalizer.reference > 0).all()

    def test_reference_not_updated_in_eval_mode(self, canonicalizer: SignCanonicalizer, dim: int) -> None:
        """EMA reference must NOT change when called in eval mode."""
        canonicalizer.eval()
        with torch.no_grad():
            canonicalizer.reference.copy_(torch.ones(dim))
            canonicalizer._initialized.fill_(True)

        ref_before = canonicalizer.reference.clone()
        z = torch.randn(8, dim)
        canonicalizer(z)
        torch.testing.assert_close(canonicalizer.reference, ref_before)

    def test_canonicalize_does_not_update_reference(self, canonicalizer: SignCanonicalizer, dim: int) -> None:
        """``canonicalize()`` must not modify the EMA even in train mode."""
        canonicalizer.train()
        with torch.no_grad():
            canonicalizer.reference.copy_(torch.ones(dim))
            canonicalizer._initialized.fill_(True)

        ref_before = canonicalizer.reference.clone()
        z = -torch.abs(torch.randn(8, dim))  # all negative
        canonicalizer.canonicalize(z)
        torch.testing.assert_close(canonicalizer.reference, ref_before)

    def test_ema_momentum(self) -> None:
        """EMA reference should blend with new batch mean at the set momentum."""
        dim = 8
        sc = SignCanonicalizer(latent_dim=dim, ema_momentum=0.5)
        sc.train()

        # First call initialises the reference to z1_mean.
        z1 = torch.ones(4, dim) * 2.0
        sc(z1)
        torch.testing.assert_close(sc.reference, torch.ones(dim) * 2.0)

        # Second call should blend: ref = 0.5 * old + 0.5 * new_mean = 0.5*2 + 0.5*4 = 3.0
        z2 = torch.ones(4, dim) * 4.0
        sc(z2)
        torch.testing.assert_close(sc.reference, torch.ones(dim) * 3.0, atol=1e-5, rtol=1e-5)

    def test_passthrough_before_init(self, canonicalizer: SignCanonicalizer, dim: int) -> None:
        """Before EMA is initialized, forward pass returns z unchanged."""
        canonicalizer.eval()
        z = torch.randn(4, dim)
        z_out = canonicalizer(z)
        torch.testing.assert_close(z, z_out)


# ---------------------------------------------------------------------------
# Custom reference_z shapes
# ---------------------------------------------------------------------------

class TestCustomReference:
    def test_broadcast_single_reference(self, canonicalizer: SignCanonicalizer, dim: int) -> None:
        """A single reference row (1, D) should broadcast across the batch."""
        canonicalizer.eval()
        ref_z = torch.ones(1, dim)
        z = torch.randn(6, dim)
        z_out = canonicalizer(z, reference_z=ref_z)
        # All outputs should be positive (since ref > 0).
        # For zero z elements sign correction is +1, so output is 0 not negative.
        assert (z_out >= 0).all()

    def test_per_sample_reference(self, canonicalizer: SignCanonicalizer, dim: int) -> None:
        """A per-sample reference (N, D) aligns each sample independently."""
        canonicalizer.eval()
        n = 5
        ref_z = torch.randn(n, dim)
        z = torch.randn(n, dim)
        z_out = canonicalizer(z, reference_z=ref_z)

        nonzero_mask = ref_z.abs() > 1e-9
        # sign(z_out) == sign(ref_z) wherever ref is nonzero.
        assert (torch.sign(z_out[nonzero_mask]) == torch.sign(ref_z[nonzero_mask])).all()


# ---------------------------------------------------------------------------
# State-dict round-trip
# ---------------------------------------------------------------------------

class TestStateDictRoundTrip:
    def test_save_load(self, canonicalizer: SignCanonicalizer, dim: int) -> None:
        """Serialized reference is restored correctly."""
        canonicalizer.train()
        z = torch.randn(8, dim)
        canonicalizer(z)

        state = canonicalizer.state_dict()
        sc2 = SignCanonicalizer(latent_dim=dim)
        sc2.load_state_dict(state)

        torch.testing.assert_close(sc2.reference, canonicalizer.reference)
        assert sc2._initialized.item() == canonicalizer._initialized.item()
