"""
COLLISION Parallel Discrete Diffusion & Non-Autoregressive Transformer Decoder (Pillar 1).

Key Innovations:
- Bidirectional Masked Token Reconstruction: Predicts all sequence tokens simultaneously.
- Confidence-Calibrated Latent Heads: Computes per-token confidence scores for selective unmasking.
- Block-Jacobi Refinement Layer: Enables multi-token parallel updates in O(K) iterations rather than O(N) sequential steps.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any


class ParallelDiffusionAttention(nn.Module):
    """
    Bidirectional Multi-Head Attention without causal masking, allowing full
    cross-context diffusion of information across all tokens simultaneously.
    """
    def __init__(self, d_model: int, n_head: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_head == 0, "d_model must be divisible by n_head"
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, S, C = x.shape
        qkv = self.qkv_proj(x).view(B, S, 3, self.n_head, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if mask is not None:
            # mask shape: (B, 1, 1, S) or (B, 1, S, S)
            attn_weights = attn_weights.masked_fill(mask == 0, float("-inf"))

        attn_probs = F.softmax(attn_weights, dim=-1)
        attn_probs = self.dropout(attn_probs)
        out = torch.matmul(attn_probs, v)  # (B, n_head, S, head_dim)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, S, C)
        return self.o_proj(out)


class ParallelFeedForward(nn.Module):
    """SwiGLU-style high-efficiency parallel feed-forward network."""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class ParallelDiffusionBlock(nn.Module):
    """Single Non-Autoregressive Transformer Layer with bidirectional message passing."""
    def __init__(self, d_model: int, n_head: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = ParallelDiffusionAttention(d_model, n_head, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = ParallelFeedForward(d_model, d_ff, dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), mask=mask)
        x = x + self.ffn(self.ln2(x))
        return x


class CollisionParallelDecoder(nn.Module):
    """
    COLLISION Non-Autoregressive & Discrete Diffusion Decoder.
    Simultaneously refines a sequence of masked/noisy tokens into full sentences/code.
    """
    def __init__(
        self,
        vocab_size: int = 32000,
        d_model: int = 512,
        n_layer: int = 6,
        n_head: int = 8,
        d_ff: int = 2048,
        max_seq_len: int = 512,
        mask_token_id: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.mask_token_id = mask_token_id

        self.tok_embeddings = nn.Embedding(vocab_size, d_model)
        self.pos_embeddings = nn.Embedding(max_seq_len, d_model)
        self.step_embedding = nn.Embedding(32, d_model)  # Diffusion refinement step embedding

        self.blocks = nn.ModuleList([
            ParallelDiffusionBlock(d_model, n_head, d_ff, dropout)
            for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Confidence Calibration Head: Predicts certainty of each predicted token
        self.confidence_head = nn.Linear(d_model, 1)

        # Weight tying
        self.lm_head.weight = self.tok_embeddings.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        step: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning (logits, confidence_scores).
        Args:
            input_ids: (B, S) token ids (may contain [MASK] tokens)
            step: (B,) current refinement step index (0 to max_steps-1)
            attention_mask: (B, S) padding / valid mask
        """
        B, S = input_ids.shape
        positions = torch.arange(S, device=input_ids.device).unsqueeze(0).expand(B, S)

        x = self.tok_embeddings(input_ids) + self.pos_embeddings(positions)
        if step is not None:
            step_clamped = torch.clamp(step, 0, 31)
            step_emb = self.step_embedding(step_clamped).unsqueeze(1)  # (B, 1, d_model)
            x = x + step_emb

        attn_mask = None
        if attention_mask is not None:
            attn_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, S)

        for block in self.blocks:
            x = block(x, mask=attn_mask)

        hidden = self.ln_f(x)
        logits = self.lm_head(hidden)  # (B, S, vocab_size)
        confidence = torch.sigmoid(self.confidence_head(hidden)).squeeze(-1)  # (B, S)

        return logits, confidence
