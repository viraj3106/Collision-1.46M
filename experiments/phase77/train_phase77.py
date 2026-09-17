import os
import sys
import time
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any, List

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.phase76.training.train_phase76 import compute_hybrid_loss
from experiments.phase73.train_phase73 import verify_immutability, create_masked_batch
from model.config import ModelConfig
from model.transformer import CollisionTransformer
from data.tokenize import BPETokenizer

PROD_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "collision-10m", "model.pt")

MODEL_SPECS = {
    "COLLISION-10M": {
        "d_model": 384, "n_layer": 6, "n_head": 8, "d_ff": 768,
        "checkpoint": os.path.join(PROJECT_ROOT, "experiments", "phase77", "checkpoints", "collision_10m_control.pt")
    },
    "J77-25M": {
        "d_model": 512, "n_layer": 10, "n_head": 8, "d_ff": 1024,
        "checkpoint": os.path.join(PROJECT_ROOT, "experiments", "phase77", "checkpoints", "collision_25m_j77a.pt")
    },
    "J77-35M": {
        "d_model": 640, "n_layer": 9, "n_head": 10, "d_ff": 1280,
        "checkpoint": os.path.join(PROJECT_ROOT, "experiments", "phase77", "checkpoints", "collision_35m_j77b.pt")
    },
    "J77-50M": {
        "d_model": 768, "n_layer": 9, "n_head": 12, "d_ff": 1536,
        "checkpoint": os.path.join(PROJECT_ROOT, "experiments", "phase77", "checkpoints", "collision_50m_j77c.pt")
    }
}

def train_phase77_model(
    model_key: str,
    tokenizer: BPETokenizer,
    steps: int = 20,
    alpha: float = 0.10,
    beta: float = 0.90
) -> Dict[str, Any]:
    verify_immutability()
    device = torch.device("cpu")
    
    spec = MODEL_SPECS[model_key]
    save_path = spec["checkpoint"]
    
    cfg = ModelConfig(
        vocab_size=tokenizer.vocab_size if hasattr(tokenizer, "vocab_size") else 8000,
        max_seq_len=256,
        d_model=spec["d_model"],
        n_layer=spec["n_layer"],
        n_head=spec["n_head"],
        d_ff=spec["d_ff"],
        dropout=0.1,
        tie_embeddings=True
    )
    
    model = CollisionTransformer(cfg).to(device)
    exact_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # If control model 10M, load weights from prod model
    if model_key == "COLLISION-10M" and os.path.exists(PROD_MODEL_PATH):
        ckpt = torch.load(PROD_MODEL_PATH, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt, strict=False)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-5, weight_decay=0.01)
    
    sample_pairs = [
        ("User: My name is Alex.", "Assistant: Nice to meet you Alex!"),
        ("User: What is Python?", "Assistant: Python is a clean programming language."),
        ("User: Explain gravity.", "Assistant: Gravity pulls objects together."),
        ("User: List 3 primary colors.", "Assistant: 1. Red 2. Blue 3. Yellow.")
    ]
    
    model.train()
    history = []
    
    start_time = time.time()
    for step in range(steps):
        p, r = sample_pairs[step % len(sample_pairs)]
        batch = create_masked_batch(p, r, tokenizer)
        
        x = torch.tensor([batch["input_ids"]], dtype=torch.long, device=device)
        y = torch.tensor([batch["target_ids"]], dtype=torch.long, device=device)
        mask = torch.tensor([batch["loss_mask"]], dtype=torch.float, device=device)
        
        optimizer.zero_grad()
        loss, c_loss, r_loss = compute_hybrid_loss(model, x, y, mask, alpha=alpha, beta=beta)
        loss.backward()
        optimizer.step()
        
        history.append({
            "step": step + 1,
            "loss": round(loss.item(), 4),
            "context_loss": round(c_loss.item(), 4),
            "response_loss": round(r_loss.item(), 4)
        })
        
    elapsed_time = time.time() - start_time
    final_loss = history[-1]["loss"]
    ppl = math.exp(min(final_loss, 20.0))
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": cfg,
        "candidate": model_key,
        "exact_params": exact_params,
        "alpha": alpha,
        "beta": beta,
        "final_loss": final_loss,
        "ppl": ppl,
        "elapsed_time": elapsed_time
    }, save_path)
    
    print(f"[{model_key}] Params: {exact_params:,} | Final Loss: {final_loss:.4f} | PPL: {ppl:.2f} | Time: {elapsed_time:.2f}s", flush=True)
    
    return {
        "model_key": model_key,
        "exact_params": exact_params,
        "d_model": spec["d_model"],
        "n_layer": spec["n_layer"],
        "n_head": spec["n_head"],
        "d_ff": spec["d_ff"],
        "final_loss": final_loss,
        "ppl": round(ppl, 2),
        "elapsed_time_s": round(elapsed_time, 2),
        "save_path": save_path,
        "history": history
    }

def train_all_phase77_models(tokenizer: BPETokenizer) -> Dict[str, Dict[str, Any]]:
    print("=" * 60)
    print("  TRAINING PHASE 77 SCALING CANDIDATES")
    print("=" * 60)
    results = {}
    for key in ["COLLISION-10M", "J77-25M", "J77-35M", "J77-50M"]:
        results[key] = train_phase77_model(key, tokenizer, steps=20)
    return results

if __name__ == "__main__":
    tok = BPETokenizer()
    tok.load(os.path.join(PROJECT_ROOT, "artifacts", "tokenizer"))
    res = train_all_phase77_models(tok)
    print("Training finished successfully.")
