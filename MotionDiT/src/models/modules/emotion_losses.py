"""
emotion_losses.py
-----------------
Auxiliary emotion loss functions for DITTO's MotionDecoder training.

These losses operate on the **expression slice** of the predicted motion
(dims 202–265, i.e. the 63-dim ``exp`` block), which directly encodes
facial deformations — the part most responsible for emotional expressiveness.

Four losses are provided:

1. ``EmotionClassifierHead`` + ``emotion_classification_loss``
   - A small 3-layer MLP trained jointly that maps exp features → emotion
     logits (8 MEAD classes).  Cross-entropy forces the expression slice to
     be discriminative per emotion class.
   - Requires ``label >= 0`` (MEAD data); silently returns 0 otherwise.
   - λ recommended: 0.10

2. ``perceptual_emotion_consistency_loss``
   - MSE between the classifier's intermediate (hidden) representation of
     *predicted* expression vs. *ground-truth* expression.  Feature-level
     alignment beyond the raw L2 denoising objective.
   - λ recommended: 0.05

3. ``contrastive_emotion_loss``
   - NT-Xent / InfoNCE contrastive loss over within-batch expression
     embeddings.  Positives = same emotion label; negatives = different.
   - Requires ≥ 2 distinct emotion classes in the batch.
   - λ recommended: 0.05

4. ``temporal_emotion_consistency_loss``
   - L2 on the per-frame velocity of expression features, penalising abrupt
     frame-to-frame jumps. Promotes temporally smooth emotion transitions.
   - λ recommended: 0.02

Usage (inside ``MotionDiffusion.p_losses``):
    from .emotion_losses import (
        EmotionClassifierHead,
        emotion_classification_loss,
        perceptual_emotion_consistency_loss,
        contrastive_emotion_loss,
        temporal_emotion_consistency_loss,
    )
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# Indices into the 265-dim motion vector (same as diffusion.py part_w_dict)
EXP_START = 202
EXP_END   = 265   # exclusive
EXP_DIM   = EXP_END - EXP_START    # 63

NUM_EMOTION_CLASSES = 8  # MEAD: Angry, Contempt, Disgust, Fear, Happy, Neutral, Sad, Surprised


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Emotion Classifier Head  (jointly trained auxiliary MLP)
# ─────────────────────────────────────────────────────────────────────────────

class EmotionClassifierHead(nn.Module):
    """
    Lightweight MLP: expression embedding → emotion class logits.

    Input:  [B, EXP_DIM]   (pooled expression slice from a motion clip)
    Output: [B, num_classes]  (raw logits, use CrossEntropyLoss)

    Also exposes an intermediate hidden representation used by the
    perceptual consistency loss.

    Args:
        exp_dim      (int): Dimensionality of the expression features (default 63).
        hidden_dim   (int): Width of each hidden layer.
        num_classes  (int): Number of emotion categories (default 8 for MEAD).
        dropout      (float): Dropout probability between layers.
    """

    def __init__(
        self,
        exp_dim: int = EXP_DIM,
        hidden_dim: int = 128,
        num_classes: int = NUM_EMOTION_CLASSES,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.exp_dim = exp_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        # Layer 1: project raw exp → hidden
        self.fc1 = nn.Linear(exp_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.act1 = nn.SiLU()
        self.drop1 = nn.Dropout(dropout)

        # Layer 2: refine (this is the "perceptual feature" layer)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.act2 = nn.SiLU()
        self.drop2 = nn.Dropout(dropout)

        # Output: logits
        self.fc_out = nn.Linear(hidden_dim, num_classes)

        # Initialise last layer near zero for stable early training
        nn.init.zeros_(self.fc_out.weight)
        nn.init.zeros_(self.fc_out.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns the intermediate hidden feature (used by perceptual loss).

        Args:
            x: [B, exp_dim]

        Returns:
            feat: [B, hidden_dim]
        """
        x = self.drop1(self.act1(self.bn1(self.fc1(x))))
        x = self.drop2(self.act2(self.bn2(self.fc2(x))))
        return x

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, exp_dim]

        Returns:
            logits: [B, num_classes]
            feat:   [B, hidden_dim]   (intermediate representation)
        """
        feat   = self.forward_features(x)
        logits = self.fc_out(feat)
        return logits, feat


# ─────────────────────────────────────────────────────────────────────────────
# Helper: extract & pool expression slice
# ─────────────────────────────────────────────────────────────────────────────

def _get_exp_pooled(motion_seq: torch.Tensor) -> torch.Tensor:
    """
    Extract expression slice from a motion sequence and mean-pool over time.

    Args:
        motion_seq: [B, L, 265]

    Returns:
        exp_pooled: [B, 63]
    """
    exp_seq = motion_seq[..., EXP_START:EXP_END]   # [B, L, 63]
    return exp_seq.mean(dim=1)                       # [B, 63]


def _get_exp_seq(motion_seq: torch.Tensor) -> torch.Tensor:
    """
    Extract expression slice preserving time dimension.

    Args:
        motion_seq: [B, L, 265]

    Returns:
        exp_seq: [B, L, 63]
    """
    return motion_seq[..., EXP_START:EXP_END]


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Loss 1: Emotion Classification Loss
# ─────────────────────────────────────────────────────────────────────────────

def emotion_classification_loss(
    classifier: EmotionClassifierHead,
    pred_motion: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """
    Cross-entropy between predicted emotion logits and ground-truth labels.

    Forces the expression features in ``pred_motion`` to be discriminative
    across emotion classes, directly supervising emotional content.

    Args:
        classifier:   Jointly-trained :class:`EmotionClassifierHead`.
        pred_motion:  [B, L, 265]  — denoised motion sequence.
        labels:       [B]          — integer emotion class (0–7); -1 = unknown.

    Returns:
        Scalar loss.  Returns 0.0 if no valid labels in batch.
    """
    valid_mask = labels >= 0
    if valid_mask.sum() == 0:
        return pred_motion.new_zeros(())    # 0, same device/dtype

    exp_pooled = _get_exp_pooled(pred_motion)   # [B, 63]
    logits, _ = classifier(exp_pooled)          # [B, 8]

    logits_v = logits[valid_mask]
    labels_v = labels[valid_mask].long()

    return F.cross_entropy(logits_v, labels_v)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Loss 2: Perceptual Emotion Consistency Loss
# ─────────────────────────────────────────────────────────────────────────────

def perceptual_emotion_consistency_loss(
    classifier: EmotionClassifierHead,
    pred_motion: torch.Tensor,
    gt_motion: torch.Tensor,
) -> torch.Tensor:
    """
    MSE between the classifier's hidden representation of predicted vs.
    ground-truth expression sequences.

    Unlike raw L2 on the expression coefficients, this feature-level MSE
    penalises *semantically* meaningful differences — the classifier latent
    space is trained to be emotion-discriminative, so errors there correspond
    to perceptually different emotional states.

    Both pred and gt features are extracted in **eval mode** (dropout off)
    so the similarity signal is deterministic and the loss is numerically
    stable.  Gradients still flow through pred_feat back into classifier
    parameters and the upstream denoising network.

    Args:
        classifier:  :class:`EmotionClassifierHead` (shared with cls loss).
        pred_motion: [B, L, 265]
        gt_motion:   [B, L, 265]

    Returns:
        Scalar MSE loss.
    """
    pred_exp = _get_exp_pooled(pred_motion)   # [B, 63]
    gt_exp   = _get_exp_pooled(gt_motion)     # [B, 63]

    # Switch to eval so dropout is disabled → deterministic features.
    # Restore training state afterwards to not affect other loss terms.
    was_training = classifier.training
    classifier.eval()
    try:
        pred_feat = classifier.forward_features(pred_exp)       # [B, hidden]
        gt_feat   = classifier.forward_features(gt_exp.detach()) # [B, hidden], no grad
    finally:
        if was_training:
            classifier.train()

    return F.mse_loss(pred_feat, gt_feat.detach())


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Loss 3: Contrastive Emotion Loss (NT-Xent / InfoNCE)
# ─────────────────────────────────────────────────────────────────────────────

def contrastive_emotion_loss(
    classifier: EmotionClassifierHead,
    pred_motion: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    NT-Xent contrastive loss over within-batch expression embeddings.

    Positives: pairs sharing the same emotion label.
    Negatives: all other pairs.

    This enforces that the learned expression representation forms tight,
    well-separated clusters per emotion class.

    Args:
        classifier:   :class:`EmotionClassifierHead`.
        pred_motion:  [B, L, 265].
        labels:       [B]  integer emotion labels (-1 = ignore).
        temperature:  InfoNCE temperature (default 0.07).

    Returns:
        Scalar loss.  Returns 0.0 if fewer than 2 valid samples or only
        one emotion class in the batch (no positives to contrast).
    """
    valid_mask = labels >= 0
    n_valid = valid_mask.sum().item()
    if n_valid < 2:
        return pred_motion.new_zeros(())

    exp_pooled = _get_exp_pooled(pred_motion)       # [B, 63]
    _, feats = classifier(exp_pooled)               # [B, hidden]

    feats_v  = feats[valid_mask]                    # [n, hidden]
    labels_v = labels[valid_mask].long()            # [n]

    # Check we have at least 2 distinct classes
    unique_labels = labels_v.unique()
    if unique_labels.numel() < 2:
        return pred_motion.new_zeros(())

    # L2-normalise embeddings
    feats_norm = F.normalize(feats_v, dim=-1)       # [n, hidden]

    # Similarity matrix
    sim = torch.mm(feats_norm, feats_norm.t()) / temperature   # [n, n]

    # Positive mask: same label, different sample
    n = feats_norm.shape[0]
    label_eq = labels_v.unsqueeze(0) == labels_v.unsqueeze(1)  # [n, n] bool
    eye_mask  = torch.eye(n, dtype=torch.bool, device=feats_v.device)
    pos_mask  = label_eq & ~eye_mask                # [n, n]

    # For numerical stability
    sim = sim - sim.max(dim=1, keepdim=True).values.detach()

    # InfoNCE: for each anchor, sum over positives in numerator
    exp_sim = torch.exp(sim)
    # Exclude self-similarity from denominator
    exp_sim_no_self = exp_sim * (~eye_mask).float()

    # Sum over all positive pairs
    pos_sum = (exp_sim * pos_mask.float()).sum(dim=1)    # [n]
    all_sum = exp_sim_no_self.sum(dim=1)                 # [n]

    # Avoid log(0) — samples with no positives contribute 0
    has_positive = pos_mask.any(dim=1)
    loss_per_sample = torch.where(
        has_positive,
        -torch.log(pos_sum.clamp(min=1e-8) / all_sum.clamp(min=1e-8)),
        torch.zeros_like(pos_sum),
    )

    return loss_per_sample.mean()


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Loss 4: Temporal Emotion Consistency Loss
# ─────────────────────────────────────────────────────────────────────────────

def temporal_emotion_consistency_loss(
    pred_motion: torch.Tensor,
    gt_motion: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Penalises abrupt frame-to-frame changes in the expression features.

    Two modes:
    - ``gt_motion`` provided → L2 between *pred* velocity and *gt* velocity.
      Encourages the model to reproduce the temporal dynamics of the target.
    - ``gt_motion=None``    → L2 of the *pred* velocity against zero (i.e.
      smoothness regulariser).

    Args:
        pred_motion: [B, L, 265]
        gt_motion:   [B, L, 265]  (optional)

    Returns:
        Scalar MSE loss.
    """
    pred_exp = _get_exp_seq(pred_motion)     # [B, L, 63]
    pred_vel = pred_exp[:, 1:] - pred_exp[:, :-1]   # [B, L-1, 63]

    if gt_motion is not None:
        gt_exp = _get_exp_seq(gt_motion)     # [B, L, 63]
        gt_vel = gt_exp[:, 1:] - gt_exp[:, :-1]     # [B, L-1, 63]
        return F.mse_loss(pred_vel, gt_vel.detach())
    else:
        # Smoothness regulariser: penalise large velocity
        return pred_vel.pow(2).mean()


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Convenience: compute all emotion losses in one call
# ─────────────────────────────────────────────────────────────────────────────

def compute_all_emotion_losses(
    classifier: EmotionClassifierHead,
    pred_motion: torch.Tensor,
    gt_motion: torch.Tensor,
    labels: torch.Tensor,
    lambda_cls: float = 0.10,
    lambda_perc: float = 0.05,
    lambda_contrast: float = 0.05,
    lambda_temporal: float = 0.02,
    temperature: float = 0.07,
) -> Tuple[torch.Tensor, dict]:
    """
    Compute all four emotion auxiliary losses and return their weighted sum.

    Args:
        classifier:       :class:`EmotionClassifierHead`.
        pred_motion:      [B, L, 265]  — denoised motion.
        gt_motion:        [B, L, 265]  — ground-truth motion (x_start / target).
        labels:           [B]          — integer emotion labels (-1 = unknown).
        lambda_cls:       Weight for classification loss.
        lambda_perc:      Weight for perceptual consistency loss.
        lambda_contrast:  Weight for contrastive loss.
        lambda_temporal:  Weight for temporal consistency loss.
        temperature:      InfoNCE temperature.

    Returns:
        total_emo_loss: Scalar weighted sum of all emotion losses.
        emo_loss_dict:  Dict with individual scalar loss values (for logging).
    """
    l_cls  = emotion_classification_loss(classifier, pred_motion, labels)
    l_perc = perceptual_emotion_consistency_loss(classifier, pred_motion, gt_motion)
    l_ct   = contrastive_emotion_loss(classifier, pred_motion, labels, temperature)
    l_temp = temporal_emotion_consistency_loss(pred_motion, gt_motion)

    total = (
        lambda_cls      * l_cls
        + lambda_perc   * l_perc
        + lambda_contrast * l_ct
        + lambda_temporal * l_temp
    )

    emo_loss_dict = {
        "emo_cls":      l_cls,
        "emo_perc":     l_perc,
        "emo_contrast": l_ct,
        "emo_temporal": l_temp,
    }
    return total, emo_loss_dict
