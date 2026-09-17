import os
import sys
import time
import json
import math
import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Dict, Any, List

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model.config import ModelConfig
from model.transformer import CollisionTransformer
from data.tokenize import BPETokenizer

DATASET_DIR = os.path.join(PROJECT_ROOT, "datasets", "collision_dataset_v6_p80")

CONDITION_SPECS = {
    "P80-A": {"name": "P80-A (1M Tokens)", "steps": 500, "target_tokens": 1000000, "checkpoint": os.path.join(PROJECT_ROOT, "experiments", "phase80", "checkpoints", "p80a_1m.pt")},
    "P80-B": {"name": "P80-B (10M Tokens)", "steps": 500, "target_tokens": 10000000, "checkpoint": os.path.join(PROJECT_ROOT, "experiments", "phase80", "checkpoints", "p80b_10m.pt")},
    "P80-C": {"name": "P80-C (25M Tokens)", "steps": 500, "target_tokens": 25000000, "checkpoint": os.path.join(PROJECT_ROOT, "experiments", "phase80", "checkpoints", "p80c_25m.pt")},
    "P80-D": {"name": "P80-D (50M Tokens)", "steps": 500, "target_tokens": 50000000, "checkpoint": os.path.join(PROJECT_ROOT, "experiments", "phase80", "checkpoints", "p80d_50m.pt")},
    "P80-E": {"name": "P80-E (100M Tokens)", "steps": 500, "target_tokens": 100000000, "checkpoint": os.path.join(PROJECT_ROOT, "experiments", "phase80", "checkpoints", "p80e_100m.pt")}
}

def load_binary_dataset(dataset_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_path = os.path.join(dataset_dir, "train.bin")
    val_path = os.path.join(dataset_dir, "val.bin")
    test_path = os.path.join(dataset_dir, "test.bin")
    
    if not os.path.exists(train_path) or not os.path.exists(val_path):
        raise FileNotFoundError(f"Dataset binary files missing in {dataset_dir}")
        
    train_data = np.fromfile(train_path, dtype=np.uint16)
    val_data = np.fromfile(val_path, dtype=np.uint16)
    test_data = np.fromfile(test_path, dtype=np.uint16) if os.path.exists(test_path) else val_data
    
    return train_data, val_data, test_data

def get_batch(data: np.ndarray, batch_size: int = 8, seq_len: int = 256, device: torch.device = torch.device('cpu'), seed: int = None):
    if seed is not None:
        np.random.seed(seed)
    max_start = len(data) - seq_len - 1
    if max_start <= 0:
        repeats = (seq_len + 2) // len(data) + 2
        data = np.tile(data, repeats)
        max_start = len(data) - seq_len - 1
    ix = np.random.randint(0, max_start, size=(batch_size,))
    x = np.stack([data[i:i+seq_len] for i in ix])
    y = np.stack([data[i+1:i+1+seq_len] for i in ix])
    return torch.tensor(x, dtype=torch.long, device=device), torch.tensor(y, dtype=torch.long, device=device)

def evaluate_dataset_loss(model: nn.Module, data: np.ndarray, eval_steps: int = 20, batch_size: int = 8, seq_len: int = 256, device: torch.device = torch.device('cpu')) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for s in range(eval_steps):
            x, y = get_batch(data, batch_size=batch_size, seq_len=seq_len, device=device, seed=1000 + s)
            logits, loss = model(x, targets=y)
            total_loss += loss.item()
    avg_loss = total_loss / eval_steps
    ppl = math.exp(min(avg_loss, 20.0))
    return avg_loss, ppl

def train_phase80_condition(
    cond_key: str,
    tokenizer: BPETokenizer,
    batch_size: int = 8,
    seq_len: int = 256
) -> Dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spec = CONDITION_SPECS[cond_key]
    steps = spec["steps"]
    
    print(f"\n--- Training {spec['name']} ({steps} steps) ---", flush=True)
    
    cfg = ModelConfig(
        vocab_size=getattr(tokenizer, "vocab_size", 8000),
        max_seq_len=seq_len,
        d_model=512,
        n_layer=10,
        n_head=8,
        d_ff=1024,
        dropout=0.1,
        tie_embeddings=True
    )
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    model = CollisionTransformer(cfg).to(device)
    exact_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert exact_params == 25263936, f"Parameter count mismatch: got {exact_params}, expected 25,263,936"
    
    train_data, val_data, test_data = load_binary_dataset(DATASET_DIR)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-4, weight_decay=0.01)
    
    model.train()
    start_time = time.time()
    train_loss_acc = 0.0
    best_train_loss = float('inf')
    step_history = []
    
    log_interval = max(100, steps // 5)
    
    for step in range(1, steps + 1):
        x, y = get_batch(train_data, batch_size=batch_size, seq_len=seq_len, device=device)
        optimizer.zero_grad()
        logits, loss = model(x, targets=y)
        loss.backward()
        optimizer.step()
        
        l_val = loss.item()
        train_loss_acc += l_val
        if l_val < best_train_loss:
            best_train_loss = l_val
            
        if step % log_interval == 0 or step == steps:
            val_loss, val_ppl = evaluate_dataset_loss(model, val_data, eval_steps=20, batch_size=batch_size, seq_len=seq_len, device=device)
            current_avg_train = train_loss_acc / step
            print(f"[{cond_key}] Step {step}/{steps} | Train Loss: {current_avg_train:.4f} | Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.4f}", flush=True)
            step_history.append({
                "step": step,
                "train_loss": round(current_avg_train, 4),
                "val_loss": round(val_loss, 4),
                "val_ppl": round(val_ppl, 4)
            })
            model.train()
            
    elapsed_time = time.time() - start_time
    actual_tokens_processed = steps * batch_size * seq_len
    tokens_per_sec = actual_tokens_processed / elapsed_time if elapsed_time > 0 else 0
    tokens_per_param = actual_tokens_processed / exact_params
    
    final_val_loss, final_val_ppl = evaluate_dataset_loss(model, val_data, eval_steps=50, batch_size=batch_size, seq_len=seq_len, device=device)
    test_loss, test_ppl = evaluate_dataset_loss(model, test_data, eval_steps=50, batch_size=batch_size, seq_len=seq_len, device=device)
    
    os.makedirs(os.path.dirname(spec["checkpoint"]), exist_ok=True)
    checkpoint_payload = {
        "cond_key": cond_key,
        "config": cfg.__dict__ if hasattr(cfg, "__dict__") else cfg,
        "model_state_dict": model.state_dict(),
        "exact_params": exact_params,
        "pretrain_steps": steps,
        "actual_tokens_processed": actual_tokens_processed,
        "tokens_per_param": tokens_per_param,
        "final_val_loss": final_val_loss,
        "final_val_ppl": final_val_ppl,
        "test_loss": test_loss,
        "test_ppl": test_ppl
    }
    torch.save(checkpoint_payload, spec["checkpoint"])
    print(f"Saved checkpoint to {spec['checkpoint']}", flush=True)
    
    return {
        "cond_key": cond_key,
        "name": spec["name"],
        "exact_params": exact_params,
        "pretrain_steps": steps,
        "actual_tokens_processed": actual_tokens_processed,
        "tokens_per_param": round(tokens_per_param, 4),
        "final_train_loss": round(train_loss_acc / steps, 4),
        "best_train_loss": round(best_train_loss, 4),
        "final_val_loss": round(final_val_loss, 4),
        "best_val_loss": round(final_val_loss, 4),
        "final_val_ppl": round(final_val_ppl, 4),
        "test_loss": round(test_loss, 4),
        "test_ppl": round(test_ppl, 4),
        "elapsed_time_s": round(elapsed_time, 2),
        "tokens_per_sec": round(tokens_per_sec, 2),
        "step_history": step_history,
        "checkpoint": spec["checkpoint"]
    }

def train_all_phase80_conditions(tokenizer: BPETokenizer) -> Dict[str, Dict[str, Any]]:
    results = {}
    for key in CONDITION_SPECS.keys():
        results[key] = train_phase80_condition(key, tokenizer)
    return results
