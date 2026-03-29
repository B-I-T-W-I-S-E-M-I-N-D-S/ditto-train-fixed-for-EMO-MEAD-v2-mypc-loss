"""
test_emotion_losses.py
----------------------
Unit tests for the new emotion auxiliary loss functions.

Run from the project root:
    cd "e:\\LAB\\code\\ditto-v2-loss"
    python test_emotion_losses.py

Tests:
  1. EmotionClassifierHead – shape, NaN, forward_features
  2. emotion_classification_loss – with valid labels, with no valid labels (all -1)
  3. perceptual_emotion_consistency_loss – MSE feature alignment
  4. contrastive_emotion_loss – multi-class, single-class (should return 0)
  5. temporal_emotion_consistency_loss – with gt and without gt
  6. compute_all_emotion_losses – combined call, correct keys
  7. MotionDiffusion.p_losses with use_emotion_losses=True – end-to-end
  8. MotionDiffusion.p_losses with use_emotion_losses=False – no regression
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "MotionDiT"))

import torch
import traceback

# ─── constants mirrored from emotion_losses.py ──────────────────────────────
EXP_DIM = 63
NUM_CLASSES = 8
MOTION_DIM = 265
B, L = 4, 80


def _make_motion(batch=B, seq=L, dim=MOTION_DIM):
    return torch.randn(batch, seq, dim)


def _make_labels(batch=B, valid=True):
    if valid:
        return torch.randint(0, NUM_CLASSES, (batch,))
    else:
        return torch.full((batch,), -1, dtype=torch.long)


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: EmotionClassifierHead
# ─────────────────────────────────────────────────────────────────────────────

def test_emotion_classifier_head():
    print("\n[Test 1] EmotionClassifierHead shape and NaN check...")
    from src.models.modules.emotion_losses import EmotionClassifierHead

    head = EmotionClassifierHead(exp_dim=EXP_DIM, hidden_dim=128, num_classes=NUM_CLASSES)
    x = torch.randn(B, EXP_DIM)

    # Test forward_features
    feat = head.forward_features(x)
    assert feat.shape == (B, 128), f"Expected ({B}, 128), got {feat.shape}"
    assert not torch.isnan(feat).any(), "NaN in forward_features output"

    # Test full forward
    logits, feat2 = head(x)
    assert logits.shape == (B, NUM_CLASSES), f"Expected ({B}, {NUM_CLASSES}), got {logits.shape}"
    assert feat2.shape == feat.shape, "feature shapes don't match"
    assert not torch.isnan(logits).any(), "NaN in logits"

    print(f"  PASS  logits={tuple(logits.shape)}, feat={tuple(feat.shape)}")


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: emotion_classification_loss
# ─────────────────────────────────────────────────────────────────────────────

def test_emotion_classification_loss():
    print("\n[Test 2] emotion_classification_loss...")
    from src.models.modules.emotion_losses import EmotionClassifierHead, emotion_classification_loss

    head = EmotionClassifierHead()
    pred = _make_motion()

    # Case A: valid labels → should return meaningful scalar
    labels = _make_labels(valid=True)
    loss = emotion_classification_loss(head, pred, labels)
    assert loss.shape == (), f"Expected scalar, got shape {loss.shape}"
    assert not torch.isnan(loss), "NaN in classification loss"
    assert loss.item() > 0, "Expected positive loss for untrained classifier on real labels"
    print(f"  PASS (valid labels): loss = {loss.item():.4f}")

    # Case B: all -1 labels → should return exactly 0.0
    labels_none = _make_labels(valid=False)
    loss_none = emotion_classification_loss(head, pred, labels_none)
    assert loss_none.item() == 0.0, f"Expected 0 for all-invalid labels, got {loss_none.item()}"
    print(f"  PASS (all -1 labels): loss = {loss_none.item():.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: perceptual_emotion_consistency_loss
# ─────────────────────────────────────────────────────────────────────────────

def test_perceptual_emotion_consistency_loss():
    print("\n[Test 3] perceptual_emotion_consistency_loss...")
    from src.models.modules.emotion_losses import EmotionClassifierHead, perceptual_emotion_consistency_loss

    head = EmotionClassifierHead()
    pred = _make_motion()
    gt   = _make_motion()

    loss = perceptual_emotion_consistency_loss(head, pred, gt)
    assert loss.shape == (), f"Expected scalar, got shape {loss.shape}"
    assert not torch.isnan(loss), "NaN in perceptual loss"
    assert loss.item() >= 0, "Expected non-negative loss"

    # Self-consistency: if pred == gt, loss must be exactly 0.
    # The fix in emotion_losses.py uses eval mode (dropout off) so both calls
    # to forward_features produce identical outputs → MSE == 0.
    loss_self = perceptual_emotion_consistency_loss(head, pred, pred)
    assert loss_self.item() < 1e-6, (
        f"Expected ~0 self-loss (eval mode disables dropout), got {loss_self.item():.6f}. "
        "This indicates the classifier is NOT switching to eval mode correctly."
    )
    print(f"  PASS  loss={loss.item():.4f}, self_loss={loss_self.item():.2e}")


# ─────────────────────────────────────────────────────────────────────────────
# Test 4: contrastive_emotion_loss
# ─────────────────────────────────────────────────────────────────────────────

def test_contrastive_emotion_loss():
    print("\n[Test 4] contrastive_emotion_loss...")
    from src.models.modules.emotion_losses import EmotionClassifierHead, contrastive_emotion_loss

    head = EmotionClassifierHead()
    pred = _make_motion()

    # Case A: diverse labels → should return positive loss
    labels = torch.tensor([0, 1, 0, 2])  # 3 classes, 2 positives for class 0
    loss = contrastive_emotion_loss(head, pred, labels)
    assert loss.shape == (), f"Expected scalar, got shape {loss.shape}"
    assert not torch.isnan(loss), "NaN in contrastive loss"
    print(f"  PASS (multi-class): loss = {loss.item():.4f}")

    # Case B: single class → should return 0.0
    labels_mono = torch.zeros(B, dtype=torch.long)
    loss_mono = contrastive_emotion_loss(head, pred, labels_mono)
    assert loss_mono.item() == 0.0, f"Expected 0 for single-class batch, got {loss_mono.item()}"
    print(f"  PASS (single-class): loss = {loss_mono.item():.4f}")

    # Case C: all -1 → should return 0.0
    labels_none = _make_labels(valid=False)
    loss_none = contrastive_emotion_loss(head, pred, labels_none)
    assert loss_none.item() == 0.0, f"Expected 0 for no-label batch, got {loss_none.item()}"
    print(f"  PASS (all -1): loss = {loss_none.item():.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Test 5: temporal_emotion_consistency_loss
# ─────────────────────────────────────────────────────────────────────────────

def test_temporal_emotion_consistency_loss():
    print("\n[Test 5] temporal_emotion_consistency_loss...")
    from src.models.modules.emotion_losses import temporal_emotion_consistency_loss

    pred = _make_motion()
    gt   = _make_motion()

    # With gt reference
    loss_with_gt = temporal_emotion_consistency_loss(pred, gt)
    assert loss_with_gt.shape == (), f"Expected scalar, got shape {loss_with_gt.shape}"
    assert not torch.isnan(loss_with_gt), "NaN in temporal loss (with gt)"
    assert loss_with_gt.item() >= 0

    # Without gt (smoothness regulariser)
    loss_smooth = temporal_emotion_consistency_loss(pred, None)
    assert loss_smooth.shape == (), f"Expected scalar, got shape {loss_smooth.shape}"
    assert not torch.isnan(loss_smooth), "NaN in temporal loss (no gt)"
    assert loss_smooth.item() >= 0

    # Constant motion should have near-zero velocity
    pred_const = torch.zeros(B, L, MOTION_DIM)
    loss_const = temporal_emotion_consistency_loss(pred_const, None)
    assert loss_const.item() < 1e-8, f"Expected ~0 for constant motion, got {loss_const.item()}"

    print(f"  PASS  loss_with_gt={loss_with_gt.item():.4f}, loss_smooth={loss_smooth.item():.4f}, loss_const={loss_const.item():.2e}")


# ─────────────────────────────────────────────────────────────────────────────
# Test 6: compute_all_emotion_losses
# ─────────────────────────────────────────────────────────────────────────────

def test_compute_all_emotion_losses():
    print("\n[Test 6] compute_all_emotion_losses (combined)...")
    from src.models.modules.emotion_losses import EmotionClassifierHead, compute_all_emotion_losses

    head = EmotionClassifierHead()
    pred = _make_motion()
    gt   = _make_motion()
    labels = _make_labels(valid=True)

    total, d = compute_all_emotion_losses(head, pred, gt, labels)

    assert total.shape == (), f"Expected scalar total, got {total.shape}"
    assert not torch.isnan(total), "NaN in total emotion loss"

    expected_keys = {"emo_cls", "emo_perc", "emo_contrast", "emo_temporal"}
    assert expected_keys == set(d.keys()), f"Unexpected keys: {set(d.keys())}"

    for k, v in d.items():
        assert not torch.isnan(v), f"NaN in loss component {k}"

    print(f"  PASS  total={total.item():.4f} | " +
          " | ".join(f"{k}={v.item():.4f}" for k, v in d.items()))


# ─────────────────────────────────────────────────────────────────────────────
# Test 7: End-to-end MotionDiffusion.p_losses with use_emotion_losses=True
# ─────────────────────────────────────────────────────────────────────────────

def test_diffusion_p_losses_with_emotion_losses():
    print("\n[Test 7] MotionDiffusion.p_losses with use_emotion_losses=True...")
    from src.models.modules.model import MotionDecoder
    from src.models.modules.diffusion import MotionDiffusion

    decoder = MotionDecoder(
        nfeats=MOTION_DIM, seq_len=L, latent_dim=512, ff_size=1024,
        num_layers=2, num_heads=8, dropout=0.1, cond_feature_dim=1024,
    )
    diffusion = MotionDiffusion(
        model=decoder, horizon=L, repr_dim=MOTION_DIM,
        n_timestep=100, schedule="cosine", loss_type="l2",
        predict_epsilon=False, guidance_weight=2,
        part_w_dict={"all": [0, -1, 1]},
        use_emotion_losses=True,
        lambda_emo_cls=0.1, lambda_emo_perc=0.05,
        lambda_emo_contrast=0.05, lambda_emo_temporal=0.02,
    )

    x = _make_motion()
    cond_frame = torch.randn(B, MOTION_DIM)
    cond = torch.randn(B, L, 1024)
    labels = _make_labels(valid=True)
    t = torch.randint(0, 100, (B,))

    total, loss_dict = diffusion.p_losses(x, cond_frame, cond, t,
                                          emotion_labels=labels)
    assert not torch.isnan(total), "NaN in p_losses total"
    assert "emo_cls" in loss_dict, "Missing emo_cls in loss_dict"
    assert "emo_perc" in loss_dict, "Missing emo_perc in loss_dict"
    assert "emo_contrast" in loss_dict, "Missing emo_contrast in loss_dict"
    assert "emo_temporal" in loss_dict, "Missing emo_temporal in loss_dict"

    print(f"  PASS  total={total.item():.4f}, emotion_keys={[k for k in loss_dict if k.startswith('emo')]}")


# ─────────────────────────────────────────────────────────────────────────────
# Test 8: MotionDiffusion.p_losses with use_emotion_losses=False (no regression)
# ─────────────────────────────────────────────────────────────────────────────

def test_diffusion_p_losses_baseline_no_regression():
    print("\n[Test 8] MotionDiffusion.p_losses with use_emotion_losses=False (baseline unchanged)...")
    from src.models.modules.model import MotionDecoder
    from src.models.modules.diffusion import MotionDiffusion

    decoder = MotionDecoder(
        nfeats=MOTION_DIM, seq_len=L, latent_dim=512, ff_size=1024,
        num_layers=2, num_heads=8, dropout=0.1, cond_feature_dim=1024,
    )
    diffusion = MotionDiffusion(
        model=decoder, horizon=L, repr_dim=MOTION_DIM,
        n_timestep=100, schedule="cosine", loss_type="l2",
        predict_epsilon=False, guidance_weight=2,
        part_w_dict={"all": [0, -1, 1]},
        use_emotion_losses=False,   # <── disabled
    )

    x = _make_motion()
    cond_frame = torch.randn(B, MOTION_DIM)
    cond = torch.randn(B, L, 1024)
    t = torch.randint(0, 100, (B,))

    total, loss_dict = diffusion.p_losses(x, cond_frame, cond, t)
    assert not torch.isnan(total), "NaN in baseline p_losses"
    assert "emo_cls" not in loss_dict, "Unexpected emo_cls in baseline loss_dict"

    print(f"  PASS  total={total.item():.4f}, keys={list(loss_dict.keys())}")


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_emotion_classifier_head,
        test_emotion_classification_loss,
        test_perceptual_emotion_consistency_loss,
        test_contrastive_emotion_loss,
        test_temporal_emotion_consistency_loss,
        test_compute_all_emotion_losses,
        test_diffusion_p_losses_with_emotion_losses,
        test_diffusion_p_losses_baseline_no_regression,
    ]

    passed, failed = 0, 0
    for fn in tests:
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"  FAIL  {fn.__name__}: {e}")
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*55}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    if failed == 0:
        print("ALL TESTS PASSED ✓")
    else:
        sys.exit(1)
