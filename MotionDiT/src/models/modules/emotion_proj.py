"""
emotion_proj.py
---------------
EmotionProjection: projects Emotion2vec continuous features into a small
set of cross-attention tokens that are injected into DITTO's MotionDecoder.

Input:  [B, L, emotion_feat_dim]  (emotion_feat_dim=1024 for Emotion2vec)
Output: [B, num_emotion_tokens, latent_dim]

These emotion tokens are concatenated alongside audio tokens in the
MotionDecoder cross-attention context, allowing the diffusion model to
modulate motion generation based on audio-derived emotional content.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class EmotionProjection(nn.Module):
    """
    Projects per-frame Emotion2vec features into a compact set of context
    tokens to be fused into MotionDecoder's cross-attention.

    Args:
        emotion_feat_dim (int): Dimensionality of the Emotion2vec features.
            Default: 1024 (Emotion2vec-plus-large output dim).
        latent_dim (int): Transformer latent dimension (matches MotionDecoder).
        num_emotion_tokens (int): How many summary tokens to produce.
            Fewer → less capacity but faster; default 4 is a good trade-off.
        dropout (float): Dropout applied after projection.
    """

    def __init__(
        self,
        emotion_feat_dim: int = 1024,
        latent_dim: int = 512,
        num_emotion_tokens: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.emotion_feat_dim = emotion_feat_dim
        self.latent_dim = latent_dim
        self.num_emotion_tokens = num_emotion_tokens

        # ── Frame-level projection: [B, L, D_emo] → [B, L, latent_dim] ──
        self.frame_proj = nn.Sequential(
            nn.Linear(emotion_feat_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        # ── Temporal pooling: L frames → num_emotion_tokens  ──────────────
        # Learned weight vector per token over the L time steps.
        # We pool with a small MLP that maps [B, latent_dim] → [B, num_tokens, latent]
        self.token_proj = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * num_emotion_tokens),
        )

        self.norm_out = nn.LayerNorm(latent_dim)

    def forward(self, emotion_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
            emotion_embed: [B, L, emotion_feat_dim]  — per-frame Emotion2vec features

        Returns:
            emotion_tokens: [B, num_emotion_tokens, latent_dim]
        """
        # Step 1: project each frame independently
        x = self.frame_proj(emotion_embed)          # [B, L, latent_dim]

        # Step 2: mean-pool across time → single clip-level embedding
        x_pooled = x.mean(dim=1)                    # [B, latent_dim]

        # Step 3: fan out to num_emotion_tokens
        tokens = self.token_proj(x_pooled)           # [B, latent_dim * num_tokens]
        B = tokens.shape[0]
        tokens = tokens.view(B, self.num_emotion_tokens, self.latent_dim)
        tokens = self.norm_out(tokens)              # [B, num_tokens, latent_dim]

        return tokens
