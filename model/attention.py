import math
import torch
import torch.nn as nn

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_head == 0, "Embedding dimension must be divisible by head count"
        
        self.d_model = d_model
        self.n_head = n_head
        self.d_k = d_model // n_head
        
        # Key, Query, Value projections in a single batch
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        
        # Dropout layers
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        
        # Causal mask registration (lower triangular matrix)
        self.register_buffer(
            "bias", 
            torch.tril(torch.ones(max_seq_len, max_seq_len))
            .view(1, 1, max_seq_len, max_seq_len)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size() # Batch size, sequence length, embedding dim
        
        # Project and split to Q, K, V
        q, k, v = self.qkv_proj(x).split(self.d_model, dim=2)
        
        # Reshape to (B, n_head, T, d_k)
        q = q.view(B, T, self.n_head, self.d_k).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.d_k).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_k).transpose(1, 2)
        
        # Compute raw attention scores: (B, n_head, T, T)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply causal mask (up to current sequence length T)
        scores = scores.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        
        # Softmax & dropout
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Weighted sum of values: (B, n_head, T, d_k)
        out = torch.matmul(attn_weights, v)
        
        # Concat heads and project back: (B, T, C)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.o_proj(out)
        out = self.resid_dropout(out)
        
        return out
