"""
COLLISION Configuration for Hugging Face Transformers Integration.
Allows AutoConfig.from_pretrained("collision-10M/Collision-1B", trust_remote_code=True).
"""

try:
    from transformers.configuration_utils import PretrainedConfig
except ImportError:
    class PretrainedConfig:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

class CollisionConfig(PretrainedConfig):
    model_type = "collision"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 32000,
        max_seq_len: int = 1024,
        max_position_embeddings: int = 1024,
        d_model: int = 2048,
        hidden_size: int = 2048,
        n_layer: int = 24,
        num_hidden_layers: int = 24,
        n_head: int = 16,
        num_attention_heads: int = 16,
        d_ff: int = 5376,
        intermediate_size: int = 5376,
        dropout: float = 0.1,
        tie_embeddings: bool = True,
        tie_word_embeddings: bool = True,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        pad_token_id: int = 0,
        unk_token_id: int = 0,
        initializer_range: float = 0.02,
        layer_norm_epsilon: float = 1e-5,
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.max_position_embeddings = max_position_embeddings or max_seq_len
        self.d_model = d_model or hidden_size
        self.hidden_size = self.d_model
        self.n_layer = n_layer or num_hidden_layers
        self.num_hidden_layers = self.n_layer
        self.n_head = n_head or num_attention_heads
        self.num_attention_heads = self.n_head
        self.d_ff = d_ff or intermediate_size
        self.intermediate_size = self.d_ff
        self.dropout = dropout
        self.tie_embeddings = tie_embeddings
        self.tie_word_embeddings = tie_word_embeddings
        self.initializer_range = initializer_range
        self.layer_norm_epsilon = layer_norm_epsilon

        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            unk_token_id=unk_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs
        )
