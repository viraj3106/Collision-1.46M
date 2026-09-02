import os
import sys
import time
import torch
import torch.nn as nn

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model.config import ModelConfig
from model.transformer import CollisionTransformer
from collision.inference.config import InferenceConfig
from collision.inference.tokenizer import CollisionTokenizer
from collision.inference.generation import top_k_top_p_filtering

class CollisionInferenceEngine:
    def __init__(self, model_dir=None):
        if model_dir is None:
            model_dir = os.path.join(PROJECT_ROOT, "models", "collision-10m")
            
        self.model_dir = model_dir
        self.device = torch.device("cpu")
        
        # Load configs
        self.config = InferenceConfig(os.path.join(model_dir, "config.json"))
        
        # Build model structure
        model_cfg_dict = {
            "vocab_size": self.config.vocab_size,
            "max_seq_len": self.config.max_seq_len,
            "d_model": self.config.d_model,
            "n_layer": self.config.n_layer,
            "n_head": self.config.n_head,
            "d_ff": self.config.d_ff,
            "dropout": self.config.dropout,
            "tie_embeddings": self.config.tie_embeddings
        }
        self.model_cfg = ModelConfig(**model_cfg_dict)
        self.model = CollisionTransformer(self.model_cfg).to(self.device)
        
        # Load state dict
        model_pt_path = os.path.join(model_dir, "model.pt")
        if not os.path.exists(model_pt_path):
            raise FileNotFoundError(f"Checkpoint file not found at {model_pt_path}")
            
        print(f"Loading weights from {model_pt_path}...")
        checkpoint = torch.load(model_pt_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        
        # Disable gradients globally for model parameters
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Load tokenizer
        self.tokenizer = CollisionTokenizer(os.path.join(model_dir, "tokenizer"))
        
        # Warmup execution
        self.warmup()

    def warmup(self):
        print("Performing warmup inference...")
        t0 = time.time()
        # Single short warmup pass
        self.generate("Warmup query", max_tokens=5, temp=0.7, top_k=50, top_p=0.9)
        elapsed = time.time() - t0
        print(f"Warmup completed in {elapsed:.3f} seconds.")

    def generate(self, prompt: str, max_tokens=100, temp=0.7, top_k=50, top_p=0.9):
        # Validation controls
        if temp <= 0.0:
            raise ValueError("temperature must be greater than 0")
        if top_k < 0:
            raise ValueError("top_k must be non-negative")
        if not (0.0 < top_p <= 1.0):
            raise ValueError("top_p must be greater than 0 and less than or equal to 1")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
            
        # Safe upper limits
        max_tokens = min(max_tokens, 256)
        
        # Clean prompt empty check
        if not prompt or not prompt.strip():
            raise ValueError("prompt must not be empty")
            
        ids = self.tokenizer.encode(prompt, bos=True)
        
        # Oversized prompt check
        if len(ids) >= self.model_cfg.max_seq_len:
            raise ValueError(f"Prompt length ({len(ids)} tokens) exceeds context limit of {self.model_cfg.max_seq_len} tokens")

        x = torch.tensor([ids], dtype=torch.long, device=self.device)
        prompt_len = len(ids)
        tokens_generated = 0
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(max_tokens):
                # Context window cropping
                x_cond = x if x.size(1) <= self.model_cfg.max_seq_len else x[:, -self.model_cfg.max_seq_len:]
                logits, _ = self.model(x_cond)
                next_token_logits = logits[0, -1, :] / temp
                
                filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
                probs = torch.softmax(filtered_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                x = torch.cat((x, next_token.unsqueeze(0)), dim=1)
                tokens_generated += 1
                
                if next_token.item() == self.tokenizer.special_tokens.get("[EOS]", 259):
                    break
                    
        elapsed = time.time() - start_time
        latency_ms = elapsed * 1000
        tok_per_sec = tokens_generated / max(0.0001, elapsed)
        
        generated_ids = x[0][prompt_len:].tolist()
        generated_text = self.tokenizer.decode(generated_ids)
        
        return {
            "text": generated_text,
            "prompt_tokens": prompt_len,
            "completion_tokens": tokens_generated,
            "total_tokens": prompt_len + tokens_generated,
            "latency_ms": latency_ms,
            "tokens_per_second": tok_per_sec
        }
