import yaml

class ModelConfig:
    def __init__(
        self,
        vocab_size: int = 8000,
        max_seq_len: int = 256,
        d_model: int = 128,
        n_layer: int = 3,
        n_head: int = 4,
        d_ff: int = 256,
        dropout: float = 0.1,
        tie_embeddings: bool = True
    ):
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.n_layer = n_layer
        self.n_head = n_head
        self.d_ff = d_ff
        self.dropout = dropout
        self.tie_embeddings = tie_embeddings
        self.validate()

    def validate(self):
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {self.vocab_size}")
        if self.max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {self.max_seq_len}")
        if self.d_model <= 0:
            raise ValueError(f"d_model must be positive, got {self.d_model}")
        if self.n_layer <= 0:
            raise ValueError(f"n_layer must be positive, got {self.n_layer}")
        if self.n_head <= 0:
            raise ValueError(f"n_head must be positive, got {self.n_head}")
        if self.d_ff <= 0:
            raise ValueError(f"d_ff must be positive, got {self.d_ff}")
        if self.d_model % self.n_head != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by n_head ({self.n_head})")
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError(f"dropout must be in range [0, 1), got {self.dropout}")


    @classmethod
    def from_yaml(cls, path: str) -> "ModelConfig":
        with open(path, "r") as f:
            cfg_dict = yaml.safe_load(f)
        model_cfg = cfg_dict.get("model", {})
        return cls(**model_cfg)

    def calculate_parameter_count(self) -> int:
        # Programmatic parameter count estimation matching the model classes:
        # 1. Embeddings
        token_emb = self.vocab_size * self.d_model
        pos_emb = self.max_seq_len * self.d_model
        
        # 2. Transformer layers (n_layer blocks)
        # Each block has:
        # - Attention:
        #   - QKV projection: 3 * d_model * d_model (weight) + 3 * d_model (bias)
        #   - O projection: d_model * d_model (weight) + d_model (bias)
        # - MLP:
        #   - FC1: d_model * d_ff (weight) + d_ff (bias)
        #   - FC2: d_ff * d_model (weight) + d_model (bias)
        # - LayerNorms: 2 * d_model * 2 (each has weight and bias of shape d_model)
        attn_params = (4 * self.d_model * self.d_model) + (4 * self.d_model)
        mlp_params = (2 * self.d_model * self.d_ff) + (self.d_ff + self.d_model)
        ln_params = 4 * self.d_model
        
        block_params = attn_params + mlp_params + ln_params
        total_blocks = self.n_layer * block_params
        
        # 3. Final LayerNorm: 2 * d_model
        final_ln = 2 * self.d_model
        
        # 4. Language-model head
        if self.tie_embeddings:
            # Only the bias if used (which we will define as having bias of vocab_size)
            lm_head = self.vocab_size
        else:
            lm_head = (self.vocab_size * self.d_model) + self.vocab_size
            
        total = token_emb + pos_emb + total_blocks + final_ln + lm_head
        return total
