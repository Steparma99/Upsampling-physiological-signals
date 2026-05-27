"""
test_dimensions.py — Synthetic forward-pass shape verification for every module.

All models are instantiated from the real configs/default.yaml and exercised
with B=2 random tensors.  No real ECG data is required.

Run:
    pytest tests/test_dimensions.py -v
"""

from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from src.data.preprocessing import L_HR, L_LR
from src.losses.adversarial import FeatureMatchingLoss, HingeLoss
from src.losses.contrastive import NTXentLoss
from src.losses.mae_loss import MAELoss
from src.losses.morphological import MorphologicalLoss
from src.losses.reconstruction import ReconstructionLoss
from src.models.decoder import UNetDecoder
from src.models.discriminator import PatchGANDiscriminator
from src.models.encoder import ECGEncoder
from src.models.mae_decoder import MAEDecoder
from src.models.projection_head import ProjectionHead
from src.utils.metrics import (
    compute_prd,
    compute_rmse,
    compute_snr,
    compute_ssim,
    evaluate_batch,
)

# ── Module-level constants (loaded once) ──────────────────────────────────────

_CFG_PATH = Path(__file__).parents[1] / "configs" / "default.yaml"
CFG = OmegaConf.load(_CFG_PATH)

B         = 2                                          # batch size for all tests
P         = L_LR // CFG.model.encoder.patch_len        # 1000 // 50 = 20
D         = CFG.model.d_model                          # 256
F_SIG     = CFG.model.f_signal_mlp.input_dim           # 12
F_META    = CFG.model.f_meta_mlp.input_dim             # 7
N_SKIPS   = CFG.model.encoder.n_residual_blocks        # 4
PATCH_LEN = CFG.model.encoder.patch_len                # 50


# ── Encoder ───────────────────────────────────────────────────────────────────

class TestECGEncoder:

    @pytest.fixture(scope="class")
    def encoder(self):
        m = ECGEncoder(CFG)
        m.eval()
        return m

    def _fwd(self, encoder):
        X_in   = torch.randn(B, 2, L_LR)
        f_sig  = torch.randn(B, F_SIG)
        f_meta = torch.randn(B, F_META)
        with torch.no_grad():
            return encoder(X_in, f_sig, f_meta)

    def test_output_shapes(self, encoder):
        z_seq, z_cls, z_patches, skips = self._fwd(encoder)

        assert z_seq.shape     == (B, P + 1, D),   f"z_seq:     {z_seq.shape}"
        assert z_cls.shape     == (B, D),           f"z_cls:     {z_cls.shape}"
        assert z_patches.shape == (B, P, D),        f"z_patches: {z_patches.shape}"
        assert len(skips) == N_SKIPS,               f"len(skips)={len(skips)}"
        for i, s in enumerate(skips):
            assert s.shape == (B, D, L_LR),         f"skips[{i}]: {s.shape}"

    def test_z_cls_is_position_zero(self, encoder):
        """Off-by-one guard: z_cls must be z_seq[:, 0, :], not position 1."""
        z_seq, z_cls, _, _ = self._fwd(encoder)
        assert torch.allclose(z_cls, z_seq[:, 0, :]), \
            "z_cls must equal z_seq[:, 0, :]"

    def test_z_patches_excludes_cls(self, encoder):
        """z_patches must be z_seq[:, 1:, :] — P tokens, not P+1."""
        z_seq, _, z_patches, _ = self._fwd(encoder)
        assert torch.allclose(z_patches, z_seq[:, 1:, :]), \
            "z_patches must equal z_seq[:, 1:, :]"


# ── Projection head ───────────────────────────────────────────────────────────

class TestProjectionHead:

    def test_output_shape(self):
        head     = ProjectionHead(CFG)
        proj_dim = CFG.phase_a.simclr.projection_output_dim
        out = head(torch.randn(B, D))
        assert out.shape == (B, proj_dim), f"proj head: {out.shape}"

    def test_no_l2_norm_in_head(self):
        """Scaling input 100× must scale output — L2 norm must NOT be applied here.
        (CLAUDE.md constraint #3: normalisation lives inside NTXentLoss.)
        """
        head = ProjectionHead(CFG)
        head.eval()
        z = torch.randn(B, D)
        with torch.no_grad():
            norm_base   = head(z).norm(dim=-1).mean()
            norm_scaled = head(z * 100).norm(dim=-1).mean()
        assert norm_scaled > norm_base * 5, \
            "Output norm must grow with input scale (no L2 normalisation in head)"


# ── MAE decoder ───────────────────────────────────────────────────────────────

class TestMAEDecoder:

    @pytest.fixture(scope="class")
    def mae(self):
        m = MAEDecoder(CFG)
        m.eval()
        return m

    def test_output_shape(self, mae):
        z_seq = torch.randn(B, P + 1, D)
        with torch.no_grad():
            out = mae(z_seq)
        assert out.shape == (B, P, PATCH_LEN), f"MAEDecoder: {out.shape}"

    def test_prepare_masked_preserves_unmasked(self, mae):
        """Unmasked CNN patches must be returned unchanged."""
        patches = torch.randn(B, P, D)
        mask    = torch.zeros(B, P, dtype=torch.bool)
        mask[:, ::2] = True  # mask every other patch

        masked = mae.prepare_masked_input(patches, mask)

        assert masked.shape == patches.shape
        unmasked = ~mask
        assert torch.allclose(masked[unmasked], patches[unmasked]), \
            "prepare_masked_input must not alter unmasked positions"

    def test_prepare_masked_replaces_masked(self, mae):
        """Masked positions must be replaced by the (single) mask token value."""
        patches = torch.zeros(B, P, D)
        mask    = torch.ones(B, P, dtype=torch.bool)

        masked = mae.prepare_masked_input(patches, mask)

        # All masked positions replaced by mask_token — check they are uniform
        expected_val = mae.mask_token.squeeze()
        for b in range(B):
            for p in range(P):
                assert torch.allclose(masked[b, p], expected_val.detach()), \
                    f"Position ({b},{p}) not replaced by mask_token"


# ── U-Net decoder ─────────────────────────────────────────────────────────────

class TestUNetDecoder:

    @pytest.fixture(scope="class")
    def decoder(self):
        m = UNetDecoder(CFG)
        m.eval()
        return m

    def _skips(self):
        return [torch.randn(B, D, L_LR) for _ in range(N_SKIPS)]

    def test_output_shape(self, decoder):
        z_patches = torch.randn(B, P, D)
        z_cls     = torch.randn(B, D)
        with torch.no_grad():
            out = decoder(z_patches, z_cls, self._skips())
        assert out.shape == (B, L_HR), f"UNetDecoder: {out.shape}"

    def test_output_is_squeezed_2d(self, decoder):
        """Head must squeeze the channel dim — output must be (B, L_HR), not (B, 1, L_HR)."""
        z_patches = torch.randn(B, P, D)
        z_cls     = torch.randn(B, D)
        with torch.no_grad():
            out = decoder(z_patches, z_cls, self._skips())
        assert out.dim() == 2, f"Expected 2D output, got {out.dim()}D"


# ── PatchGAN discriminator ────────────────────────────────────────────────────

class TestPatchGANDiscriminator:

    @pytest.fixture(scope="class")
    def disc(self):
        m = PatchGANDiscriminator(CFG)
        m.eval()
        return m

    def test_output_shapes(self, disc):
        x = torch.randn(B, L_HR)
        with torch.no_grad():
            logits, feats = disc(x)

        assert logits.shape     == (B, 1, 78),      f"logits:   {logits.shape}"
        assert len(feats)       == 3,               f"n_feats:  {len(feats)}"
        assert feats[0].shape   == (B, 64,  1250),  f"D1: {feats[0].shape}"
        assert feats[1].shape   == (B, 128,  312),  f"D2: {feats[1].shape}"
        assert feats[2].shape   == (B, 256,   78),  f"D3: {feats[2].shape}"

    def test_accepts_3d_input(self, disc):
        """(B, 1, L_HR) input must produce the same output shape as (B, L_HR)."""
        with torch.no_grad():
            logits, _ = disc(torch.randn(B, 1, L_HR))
        assert logits.shape == (B, 1, 78)

    def test_no_sigmoid_on_output(self, disc):
        """Raw logits must not be clamped to [0,1] — hinge loss expects unbounded scores."""
        x = torch.randn(B, L_HR)
        with torch.no_grad():
            logits, _ = disc(x)
        # Sigmoid clips everything to (0,1); raw scores from a random network will
        # almost certainly have some values outside [0,1].
        assert (logits > 1.0).any() or (logits < 0.0).any(), \
            "Logits appear to be sigmoid-bounded — discriminator must NOT apply sigmoid"


# ── Full pipeline ─────────────────────────────────────────────────────────────

class TestFullPipeline:

    def _build_models(self):
        enc  = ECGEncoder(CFG).eval()
        dec  = UNetDecoder(CFG).eval()
        disc = PatchGANDiscriminator(CFG).eval()
        return enc, dec, disc

    def _inputs(self):
        return (
            torch.randn(B, 2, L_LR),  # X_in
            torch.randn(B, F_SIG),    # f_signal
            torch.randn(B, F_META),   # f_meta
        )

    def test_encoder_to_decoder(self):
        enc, dec, _ = self._build_models()
        X_in, f_sig, f_meta = self._inputs()
        with torch.no_grad():
            _, z_cls, z_patches, skips = enc(X_in, f_sig, f_meta)
            x_hat = dec(z_patches, z_cls, skips)
        assert x_hat.shape == (B, L_HR)

    def test_encoder_decoder_discriminator(self):
        enc, dec, disc = self._build_models()
        X_in, f_sig, f_meta = self._inputs()
        with torch.no_grad():
            _, z_cls, z_patches, skips = enc(X_in, f_sig, f_meta)
            x_hat = dec(z_patches, z_cls, skips)
            logits, feats = disc(x_hat)
        assert logits.shape == (B, 1, 78)
        assert len(feats)   == 3


# ── Loss functions ────────────────────────────────────────────────────────────

class TestLossFunctions:

    def test_ntxent_scalar_nonneg(self):
        loss_fn = NTXentLoss(CFG.phase_a.simclr.temperature)
        proj_dim = CFG.phase_a.simclr.projection_output_dim
        h = torch.randn(B, proj_dim)
        loss = loss_fn(h, h.clone() + 0.01)
        assert loss.shape == ()
        assert loss.item() >= 0.0

    def test_mae_loss_returns_three_scalars(self):
        loss_fn = MAELoss(CFG.phase_a.mae.fft_loss_weight)
        pred   = torch.randn(B, P, PATCH_LEN)
        target = torch.randn(B, P, PATCH_LEN)
        mask   = torch.zeros(B, P, dtype=torch.bool)
        mask[:, : P // 2] = True
        total, rec, fft = loss_fn(pred, target, mask)
        for name, t in [("total", total), ("rec", rec), ("fft", fft)]:
            assert t.shape == (), f"MAELoss.{name} not scalar: {t.shape}"

    def test_reconstruction_loss_returns_three_scalars(self):
        loss_fn = ReconstructionLoss(CFG)
        pred   = torch.randn(B, L_HR)
        target = torch.randn(B, L_HR)
        l1, fft, stft = loss_fn(pred, target)
        for name, t in [("l1", l1), ("fft", fft), ("stft", stft)]:
            assert t.shape == (), f"ReconstructionLoss.{name}: {t.shape}"

    def test_morphological_loss_none_is_differentiable_zero(self):
        """None annotations → scalar 0 that still has a gradient path."""
        loss_fn = MorphologicalLoss(CFG)
        pred    = torch.randn(B, L_HR, requires_grad=True)
        target  = torch.randn(B, L_HR)
        loss = loss_fn(pred, target, annotations=None)
        assert loss.shape == ()
        assert loss.item() == 0.0
        loss.backward()
        assert pred.grad is not None, "Gradient must flow through zero morph loss"

    def test_morphological_loss_with_annotations(self):
        loss_fn     = MorphologicalLoss(CFG)
        pred        = torch.randn(B, L_HR, requires_grad=True)
        target      = torch.randn(B, L_HR)
        K           = 5
        annotations = torch.randint(200, L_HR - 200, (B, K, 2))
        annotations[:, :, 1] = 25
        loss = loss_fn(pred, target, annotations)
        assert loss.shape == ()
        loss.backward()
        assert pred.grad is not None

    def test_hinge_discriminator_loss_nonneg(self):
        hinge       = HingeLoss()
        logits_real = torch.randn(B, 1, 78)
        logits_fake = torch.randn(B, 1, 78)
        loss = hinge.discriminator_loss(logits_real, logits_fake)
        assert loss.shape == ()
        assert loss.item() >= 0.0, "Hinge discriminator loss must be ≥ 0"

    def test_hinge_generator_loss_scalar(self):
        hinge  = HingeLoss()
        logits = torch.randn(B, 1, 78)
        loss   = hinge.generator_loss(logits)
        assert loss.shape == ()

    def test_feature_matching_identical_inputs_zero(self):
        """L_FM = 0 when generated and real features are identical."""
        fm = FeatureMatchingLoss(n_layers=3)
        feats = [torch.randn(B, 64, 1250), torch.randn(B, 128, 312), torch.randn(B, 256, 78)]
        loss  = fm(feats, [f.clone() for f in feats])
        assert loss.shape == ()
        assert loss.item() == pytest.approx(0.0, abs=1e-5)


# ── Metrics ───────────────────────────────────────────────────────────────────

class TestMetrics:

    def test_perfect_reconstruction_prd_rmse_zero(self):
        x = torch.randn(B, L_HR)
        assert compute_prd(x, x).item()  == pytest.approx(0.0, abs=1e-4)
        assert compute_rmse(x, x).item() == pytest.approx(0.0, abs=1e-4)

    def test_perfect_reconstruction_ssim_one(self):
        x = torch.randn(B, L_HR)
        assert compute_ssim(x, x).item() == pytest.approx(1.0, abs=1e-3)

    def test_snr_finite_for_typical_inputs(self):
        pred   = torch.randn(B, L_HR)
        target = torch.randn(B, L_HR)
        assert torch.isfinite(compute_snr(pred, target))

    def test_prd_nonneg(self):
        pred   = torch.randn(B, L_HR)
        target = torch.randn(B, L_HR)
        assert compute_prd(pred, target).item() >= 0.0

    def test_evaluate_batch_returns_required_keys(self):
        pred   = torch.randn(B, L_HR)
        target = torch.randn(B, L_HR)
        result = evaluate_batch(pred, target)
        assert set(result.keys()) == {"snr_db", "prd", "rmse", "ssim"}

    def test_evaluate_batch_rmse_mv_with_denorm(self):
        """When mu_w/sigma_w are identity (0/1), rmse_mv must equal rmse."""
        pred    = torch.randn(B, L_HR)
        target  = torch.randn(B, L_HR)
        mu_w    = torch.zeros(B)
        sigma_w = torch.ones(B)
        result = evaluate_batch(pred, target, mu_w=mu_w, sigma_w=sigma_w)
        assert "rmse_mv" in result
        assert result["rmse"] == pytest.approx(result["rmse_mv"], abs=1e-5)

    def test_ssim_window_must_be_odd(self):
        x = torch.randn(B, L_HR)
        with pytest.raises(ValueError, match="odd"):
            compute_ssim(x, x, window_size=64)
