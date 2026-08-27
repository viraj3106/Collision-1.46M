import torch
import torch.nn as nn
from model.config import ModelConfig
from model.embeddings import TokenEmbedding, PositionalEmbedding
from model.blocks import TransformerBlock

class CollisionTransformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Embedding layer
        self.token_emb = TokenEmbedding(config.vocab_size, config.d_model)
        self.pos_emb = PositionalEmbedding(config.max_seq_len, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=config.d_model,
                n_head=config.n_head,
                max_seq_len=config.max_seq_len,
                d_ff=config.d_ff,
                dropout=config.dropout
            ) for _ in range(config.n_layer)
        ])
        
        # Final LayerNorm
        self.ln_f = nn.LayerNorm(config.d_model)
        
        # Language-model head (predicts next token)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=True)
        
        # Tie embeddings if requested
        if config.tie_embeddings:
            self.lm_head.weight = self.token_emb.emb.weight
            
        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.emb.weight if hasattr(module, 'emb') else module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        # idx shape: (batch_size, seq_len)
        device = idx.device
        b, t = idx.size()
        
        assert t <= self.config.max_seq_len, f"Cannot forward sequence of length {t}, max sequence length is {self.config.max_seq_len}"
        
        # Compute embeddings
        x = self.token_emb(idx) + self.pos_emb(idx)
        x = self.drop(x)
        
        # Pass through blocks
        for block in self.blocks:
            x = block(x)
            
        x = self.ln_f(x)
        
        # Project to vocabulary
        logits = self.lm_head(x) # (B, T, vocab_size)
        
        loss = None
        if targets is not None:
            # Flatten logits and targets to compute cross entropy loss
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), 
                targets.view(-1),
                ignore_index=-1 # Standard ignore index for padding
            )
            
        return logits, loss

    def get_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
