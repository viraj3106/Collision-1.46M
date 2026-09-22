"""
COLLISION Cognitive Neuro-Symbolic & Hegelian Dialectic Core (Pillar 2).

Architecture:
- System 1 (Fast Intuitive Heuristic Path): Low-latency direct answer projection.
- System 2 (Dialectic Multi-Perspective Processing):
    1. Thesis Engine: Projects the strongest constructive proposition.
    2. Antithesis Engine: Stresses counter-arguments, failure edge-cases, and constraint violations.
    3. Synthesis Engine: Dynamically unifies Thesis and Antithesis through cross-dialectic attention.
- Dynamic Cognitive Router: Automatically balances System 1 vs System 2 based on prompt complexity.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any


class DialecticCrossAttention(nn.Module):
    """Cross-attention mechanism that resolves conflict between Thesis and Antithesis representations."""
    def __init__(self, d_model: int, n_head: int = 8, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, thesis: torch.Tensor, antithesis: torch.Tensor) -> torch.Tensor:
        B, S, C = thesis.shape
        q = self.q_proj(thesis).view(B, S, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(antithesis).view(B, S, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(antithesis).view(B, S, self.n_head, self.head_dim).transpose(1, 2)

        attn = (torch.matmul(q, k.transpose(-2, -1)) * self.scale)
        attn_probs = self.dropout(F.softmax(attn, dim=-1))
        out = torch.matmul(attn_probs, v).transpose(1, 2).contiguous().view(B, S, C)
        return self.o_proj(out)


class HegelianDialecticCore(nn.Module):
    """
    Hegelian Dialectic Processing Module.
    Takes latent representations and passes them through Thesis, Antithesis, and Synthesis pathways.
    """
    def __init__(self, d_model: int, d_ff: int = 2048, n_head: int = 8, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model

        # Thesis Pathway (Constructive reasoning)
        self.thesis_proj = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.LayerNorm(d_model)
        )

        # Antithesis Pathway (Adversarial stress-testing & critique)
        self.antithesis_proj = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.LayerNorm(d_model)
        )

        # Synthesis Synthesis Layer: Cross-attention fusion + gating
        self.cross_attn = DialecticCrossAttention(d_model, n_head, dropout)
        self.synthesis_gate = nn.Linear(d_model * 2, d_model)
        self.ln_synthesis = nn.LayerNorm(d_model)

    def forward(self, h: torch.Tensor) -> Dict[str, torch.Tensor]:
        thesis = self.thesis_proj(h)
        antithesis = self.antithesis_proj(h)

        # Cross-critique
        critique = self.cross_attn(thesis, antithesis)

        # Gated Synthesis
        combined = torch.cat([thesis, critique], dim=-1)
        synthesis = self.ln_synthesis(h + self.synthesis_gate(combined))

        return {
            "thesis": thesis,
            "antithesis": antithesis,
            "synthesis": synthesis
        }


class DualProcessCognitiveEngine(nn.Module):
    """
    Unified System 1 (Intuitive) & System 2 (Dialectic Deliberation) Engine.
    Routes queries dynamically based on predicted reasoning complexity.
    """
    def __init__(self, d_model: int, vocab_size: int = 32000, d_ff: int = 2048):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        # Complexity Estimator / Meta-Cognitive Router
        self.router = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        # System 1: Fast Heuristic Head
        self.system1_head = nn.Linear(d_model, vocab_size, bias=False)

        # System 2: Deep Dialectic Engine
        self.system2_dialectic = HegelianDialecticCore(d_model, d_ff)
        self.system2_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, h: torch.Tensor, force_system: Optional[int] = None) -> Dict[str, Any]:
        """
        Args:
            h: (B, S, d_model) hidden representations
            force_system: Optional override (1 for System 1, 2 for System 2)
        """
        # Meta-cognitive routing score: 0.0 (Simple) -> 1.0 (Complex reasoning required)
        complexity_score = self.router(h.mean(dim=1))  # (B, 1)

        sys1_logits = self.system1_head(h)

        if force_system == 1:
            return {
                "logits": sys1_logits,
                "complexity": complexity_score,
                "system_used": "System 1 (Heuristic Fast Path)"
            }

        dialectic_out = self.system2_dialectic(h)
        sys2_logits = self.system2_head(dialectic_out["synthesis"])

        if force_system == 2:
            return {
                "logits": sys2_logits,
                "complexity": complexity_score,
                "system_used": "System 2 (Hegelian Dialectic Core)",
                "dialectic_tensors": dialectic_out
            }

        # Dynamic soft blend based on routing confidence
        w = complexity_score.unsqueeze(1)  # (B, 1, 1)
        blended_logits = (1.0 - w) * sys1_logits + w * sys2_logits

        return {
            "logits": blended_logits,
            "complexity": complexity_score,
            "system_used": "Dynamic Dual-Process Blend",
            "dialectic_tensors": dialectic_out
        }
