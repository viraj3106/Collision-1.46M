import os
import sys
import time
import json
import math
import numpy as np
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

DATASET_DIR = os.path.join(PROJECT_ROOT, "datasets", "collision_dataset_v5_p78")

CANDIDATE_SPECS = {
    "P78-A": {"d_model": 384, "n_layer": 6, "n_head": 8, "d_ff": 768, "target_tokens": 5000000, "pretrain_steps": 100},
    "P78-B": {"d_model": 384, "n_layer": 6, "n_head": 8, "d_ff": 768, "target_tokens": 20000000, "pretrain_steps": 400},
    "P78-C": {"d_model": 512, "n_layer": 10, "n_head": 8, "d_ff": 1024, "target_tokens": 20000000, "pretrain_steps": 400},
    "P78-D": {"d_model": 512, "n_layer": 10, "n_head": 8, "d_ff": 1024, "target_tokens": 50000000, "pretrain_steps": 1000},
    "P78-E": {"d_model": 768, "n_layer": 9, "n_head": 12, "d_ff": 1536, "target_tokens": 50000000, "pretrain_steps": 1000}
}

def load_binary_dataset(dataset_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    train_path = os.path.join(dataset_dir, "train.bin")
    val_path = os.path.join(dataset_dir, "val.bin")
    
    train_data = np.fromfile(train_path, dtype=np.uint16)
    val_data = np.fromfile(val_path, dtype=np.uint16)
    return train_data, val_data

def get_pretrain_batch(data: np.ndarray, batch_size: int = 8, seq_len: int = 256, device: torch.device = torch.device('cpu')):
    ix = np.random.randint(0, len(data) - seq_len - 1, size=(batch_size,))
    x = np.stack([data[i:i+seq_len] for i in ix])
    y = np.stack([data[i+1:i+1+seq_len] for i in ix])
    return torch.tensor(x, dtype=torch.long, device=device), torch.tensor(y, dtype=torch.long, device=device)

def train_phase78_candidate(
    candidate_key: str,
    tokenizer: BPETokenizer,
    batch_size: int = 8,
    seq_len: int = 256
) -> Dict[str, Any]:
    verify_immutability()
    device = torch.device("cpu")
    
    spec = CANDIDATE_SPECS[candidate_key]
    save_path = os.path.join(PROJECT_ROOT, "experiments", "phase78", "checkpoints", f"collision_{candidate_key.lower().replace('-', '_')}.pt")
    
    cfg = ModelConfig(
        vocab_size=tokenizer.vocab_size if hasattr(tokenizer, "vocab_size") else 8000,
        max_seq_len=seq_len,
        d_model=spec["d_model"],
        n_layer=spec["n_layer"],
        n_head=spec["n_head"],
        d_ff=spec["d_ff"],
        dropout=0.1,
        tie_embeddings=True
    )
    
    torch.manual_seed(42)
    model = CollisionTransformer(cfg).to(device)
    exact_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    train_data, val_data = load_binary_dataset(DATASET_DIR)
    
    # 1. Foundational Pretraining Stage
    pretrain_steps = spec["pretrain_steps"]
    optimizer_pre = torch.optim.AdamW(model.parameters(), lr=1.0e-4, weight_decay=0.01)
    
    model.train()
    start_time = time.time()
    
    pretrain_loss_acc = 0.0
    for step in range(pretrain_steps):
        x, y = get_pretrain_batch(train_data, batch_size=batch_size, seq_len=seq_len, device=device)
        optimizer_pre.zero_grad()
        logits, loss = model(x, targets=y)
        loss.backward()
        optimizer_pre.step()
        pretrain_loss_acc += loss.item()
        
    avg_pretrain_loss = pretrain_loss_acc / pretrain_steps
    tokens_processed = pretrain_steps * batch_size * seq_len
    
    # Validation Perplexity after pretraining
    model.eval()
    val_loss_acc = 0.0
    val_eval_steps = 10
    with torch.no_grad():
        for _ in range(val_eval_steps):
            vx, vy = get_pretrain_batch(val_data, batch_size=batch_size, seq_len=seq_len, device=device)
            _, vloss = model(vx, targets=vy)
            val_loss_acc += vloss.item()
    best_val_loss = val_loss_acc / val_eval_steps
    ppl = math.exp(min(best_val_loss, 20.0))
    
    # 2. Conversational Fine-Tuning Stage (Phase 76 hybrid loss)
    model.train()
    optimizer_ft = torch.optim.AdamW(model.parameters(), lr=1.0e-5, weight_decay=0.01)
    sample_pairs = [
        ("User: My name is Alex.", "Assistant: Nice to meet you Alex!"),
        ("User: What is Python?", "Assistant: Python is a clean programming language."),
        ("User: Explain gravity.", "Assistant: Gravity pulls objects together."),
        ("User: List 3 primary colors.", "Assistant: 1. Red 2. Blue 3. Yellow.")
    ]
    
    ft_steps = 20
    ft_loss_acc = 0.0
    for step in range(ft_steps):
        p, r = sample_pairs[step % len(sample_pairs)]
        batch = create_masked_batch(p, r, tokenizer)
        
        bx = torch.tensor([batch["input_ids"]], dtype=torch.long, device=device)
        by = torch.tensor([batch["target_ids"]], dtype=torch.long, device=device)
        bmask = torch.tensor([batch["loss_mask"]], dtype=torch.float, device=device)
        
        optimizer_ft.zero_grad()
        loss, _, _ = compute_hybrid_loss(model, bx, by, bmask, alpha=0.10, beta=0.90)
        loss.backward()
        optimizer_ft.step()
        ft_loss_acc += loss.item()
        
    elapsed_time = time.time() - start_time
    final_loss = ft_loss_acc / ft_steps
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": cfg,
        "candidate": candidate_key,
        "exact_params": exact_params,
        "tokens_processed": tokens_processed,
        "target_tokens": spec["target_tokens"],
        "pretrain_loss": avg_pretrain_loss,
        "best_val_loss": best_val_loss,
        "ppl": ppl,
        "final_loss": final_loss,
        "elapsed_time": elapsed_time
    }, save_path)
    
    print(f"[{candidate_key}] Params: {exact_params:,} | Tokens: {tokens_processed:,} | Val Loss: {best_val_loss:.4f} | PPL: {ppl:.2f} | Time: {elapsed_time:.2f}s", flush=True)
    
    return {
        "candidate_key": candidate_key,
        "exact_params": exact_params,
        "tokens_processed": tokens_processed,
        "target_tokens": spec["target_tokens"],
        "d_model": spec["d_model"],
        "n_layer": spec["n_layer"],
        "n_head": spec["n_head"],
        "d_ff": spec["d_ff"],
        "pretrain_loss": round(avg_pretrain_loss, 4),
        "best_val_loss": round(best_val_loss, 4),
        "final_loss": round(final_loss, 4),
        "ppl": round(ppl, 2),
        "elapsed_time_s": round(elapsed_time, 2),
        "save_path": save_path
    }

def train_all_phase78_candidates(tokenizer: BPETokenizer) -> Dict[str, Dict[str, Any]]:
    print("=" * 60)
    print("  PRETRAINING & FINE-TUNING PHASE 78 CANDIDATES")
    print("=" * 60)
    results = {}
    for key in ["P78-A", "P78-B", "P78-C", "P78-D", "P78-E"]:
        results[key] = train_phase78_candidate(key, tokenizer)
    return results

if __name__ == "__main__":
    tok = BPETokenizer()
    tok.load(os.path.join(PROJECT_ROOT, "artifacts", "tokenizer"))
    res = train_all_phase78_candidates(tok)
    print("Phase 78 candidate training complete.")
