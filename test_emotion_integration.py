"""
test_emotion_integration.py
---------------------------
Automated unit + integration tests for the MEMO→DITTO emotion module integration.

Run from the DITTO root:
    cd "e:\\LAB\\code\\Talking Head\\DITTO"
    python test_emotion_integration.py

Tests:
  1. EmotionProjection shape and no-NaN check
  2. MotionDecoder forward with emotion_embed (shape + no-NaN)
  3. MotionDecoder forward WITHOUT emotion_embed (backward compat, identical interface)
  4. MotionDiffusion p_losses with emotion_embed (training path)
  5. MotionDiffusion ddim_sample with emotion_embed (inference path)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "MotionDiT"))

import torch
import traceback


def test_emotion_projection():
    """Test 1: EmotionProjection shape and NaN check."""
    print("\n[Test 1] EmotionProjection shape check...")
    from src.models.modules.emotion_proj import EmotionProjection

    emotion_feat_dim = 1024
    latent_dim = 512
    num_tokens = 4
    B, L = 2, 80

    proj = EmotionProjection(
        emotion_feat_dim=emotion_feat_dim,
        latent_dim=latent_dim,
        num_emotion_tokens=num_tokens,
    )
    x = torch.randn(B, L, emotion_feat_dim)
    out = proj(x)

    assert out.shape == (B, num_tokens, latent_dim), \
        f"Expected ({B}, {num_tokens}, {latent_dim}), got {out.shape}"
    assert not torch.isnan(out).any(), "NaN in EmotionProjection output!"
    assert not torch.isinf(out).any(), "Inf in EmotionProjection output!"
    print(f"  PASS  output shape: {tuple(out.shape)}")


def test_motion_decoder_with_emotion():
    """Test 2: MotionDecoder forward with emotion_embed."""
    print("\n[Test 2] MotionDecoder.forward with emotion_embed...")
    from src.models.modules.model import MotionDecoder

    B, L, nfeats = 2, 80, 265
    audio_dim = 1024
    emotion_feat_dim = 1024
    num_emotion_tokens = 4

    decoder = MotionDecoder(
        nfeats=nfeats,
        seq_len=L,
        latent_dim=512,
        ff_size=1024,
        num_layers=2,   # small for speed
        num_heads=8,
        dropout=0.1,
        cond_feature_dim=audio_dim,
        emotion_feat_dim=emotion_feat_dim,
        num_emotion_tokens=num_emotion_tokens,
    )

    x = torch.randn(B, L, nfeats)
    cond_frame = torch.randn(B, nfeats)
    cond_embed = torch.randn(B, L, audio_dim)
    times = torch.randint(0, 1000, (B,))
    emotion_embed = torch.randn(B, L, emotion_feat_dim)

    with torch.no_grad():
        out = decoder(x, cond_frame, cond_embed, times, emotion_embed=emotion_embed)

    assert out.shape == (B, L, nfeats), \
        f"Expected ({B}, {L}, {nfeats}), got {out.shape}"
    assert not torch.isnan(out).any(), "NaN in decoder output!"
    print(f"  PASS  output shape: {tuple(out.shape)}")


def test_motion_decoder_backward_compat():
    """Test 3: MotionDecoder forward WITHOUT emotion_embed (backward compat)."""
    print("\n[Test 3] MotionDecoder.forward WITHOUT emotion_embed (backward compat)...")
    from src.models.modules.model import MotionDecoder

    B, L, nfeats = 2, 80, 265
    audio_dim = 1024

    decoder = MotionDecoder(
        nfeats=nfeats,
        seq_len=L,
        latent_dim=512,
        ff_size=1024,
        num_layers=2,
        num_heads=8,
        dropout=0.1,
        cond_feature_dim=audio_dim,
        emotion_feat_dim=0,   # disabled
    )

    x = torch.randn(B, L, nfeats)
    cond_frame = torch.randn(B, nfeats)
    cond_embed = torch.randn(B, L, audio_dim)
    times = torch.randint(0, 1000, (B,))

    with torch.no_grad():
        out = decoder(x, cond_frame, cond_embed, times)   # no emotion_embed

    assert out.shape == (B, L, nfeats), \
        f"Expected ({B}, {L}, {nfeats}), got {out.shape}"
    assert not torch.isnan(out).any(), "NaN in decoder output!"
    print(f"  PASS  output shape: {tuple(out.shape)}")


def test_diffusion_p_losses_with_emotion():
    """Test 4: MotionDiffusion p_losses with emotion_embed (training path)."""
    print("\n[Test 4] MotionDiffusion.p_losses with emotion_embed...")
    from src.models.modules.model import MotionDecoder
    from src.models.modules.diffusion import MotionDiffusion

    B, L, nfeats = 2, 80, 265
    audio_dim = 1024
    emotion_feat_dim = 1024

    decoder = MotionDecoder(
        nfeats=nfeats,
        seq_len=L,
        latent_dim=512,
        ff_size=1024,
        num_layers=2,
        num_heads=8,
        dropout=0.1,
        cond_feature_dim=audio_dim,
        emotion_feat_dim=emotion_feat_dim,
        num_emotion_tokens=4,
    )

    diffusion = MotionDiffusion(
        model=decoder,
        horizon=L,
        repr_dim=nfeats,
        n_timestep=100,   # small for speed
        schedule="cosine",
        loss_type="l2",
        predict_epsilon=False,
        guidance_weight=2,
        part_w_dict={"all": [0, -1, 1]},
    )

    x = torch.randn(B, L, nfeats)
    cond_frame = torch.randn(B, nfeats)
    cond = torch.randn(B, L, audio_dim)
    emotion_embed = torch.randn(B, L, emotion_feat_dim)
    t = torch.randint(0, 100, (B,))

    total_loss, loss_dict = diffusion.p_losses(x, cond_frame, cond, t,
                                               emotion_embed=emotion_embed)
    assert not torch.isnan(total_loss), "NaN in diffusion loss!"
    print(f"  PASS  total_loss={total_loss.item():.4f}, keys={list(loss_dict.keys())}")


def test_ddim_sample_with_emotion():
    """Test 5: MotionDiffusion.ddim_sample with emotion_embed (inference path)."""
    print("\n[Test 5] MotionDiffusion.ddim_sample with emotion_embed...")
    from src.models.modules.model import MotionDecoder
    from src.models.modules.diffusion import MotionDiffusion

    B, L, nfeats = 1, 80, 265
    audio_dim = 1024
    emotion_feat_dim = 1024

    decoder = MotionDecoder(
        nfeats=nfeats,
        seq_len=L,
        latent_dim=512,
        ff_size=1024,
        num_layers=2,
        num_heads=8,
        dropout=0.1,
        cond_feature_dim=audio_dim,
        emotion_feat_dim=emotion_feat_dim,
        num_emotion_tokens=4,
    )

    diffusion = MotionDiffusion(
        model=decoder,
        horizon=L,
        repr_dim=nfeats,
        n_timestep=10,   # very small for speed
        schedule="cosine",
        loss_type="l2",
        predict_epsilon=False,
        guidance_weight=2,
        part_w_dict={"all": [0, -1, 1]},
    )

    cond_frame = torch.randn(B, nfeats)
    cond = torch.randn(B, L, audio_dim)
    emotion_embed = torch.randn(B, L, emotion_feat_dim)

    with torch.no_grad():
        samples = diffusion.ddim_sample(
            shape=(B, L, nfeats),
            cond_frame=cond_frame,
            cond=cond,
            emotion_embed=emotion_embed,
        )

    assert samples.shape == (B, L, nfeats), \
        f"Expected ({B}, {L}, {nfeats}), got {samples.shape}"
    assert not torch.isnan(samples).any(), "NaN in DDIM samples!"
    print(f"  PASS  samples shape: {tuple(samples.shape)}")


if __name__ == "__main__":
    tests = [
        test_emotion_projection,
        test_motion_decoder_with_emotion,
        test_motion_decoder_backward_compat,
        test_diffusion_p_losses_with_emotion,
        test_ddim_sample_with_emotion,
    ]

    passed, failed = 0, 0
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  FAIL  {test_fn.__name__}: {e}")
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    if failed == 0:
        print("ALL TESTS PASSED ✓")
    else:
        sys.exit(1)
