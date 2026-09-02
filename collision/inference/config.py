import os
import json

class InferenceConfig:
    def __init__(self, config_path=None):
        if config_path is None:
            # Default location
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            config_path = os.path.join(project_root, "models", "collision-10m", "config.json")
            
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found at {config_path}")
            
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        self.vocab_size = data["vocab_size"]
        self.max_seq_len = data["max_seq_len"]
        self.d_model = data["d_model"]
        self.n_layer = data["n_layer"]
        self.n_head = data["n_head"]
        self.d_ff = data["d_ff"]
        self.dropout = data.get("dropout", 0.0)
        self.tie_embeddings = data.get("tie_embeddings", True)
        self.normalization = data.get("normalization", "layer_norm")
        self.positional_encoding = data.get("positional_encoding", "absolute_learned")
