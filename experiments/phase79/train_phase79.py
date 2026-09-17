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

from model.config import ModelConfig
from model.transformer import CollisionTransformer
from data.tokenize import BPETokenizer

DATASET_DIR = os.path.join(PROJECT_ROOT, "datasets", "collision_dataset_v5_p78")

MODEL_SPECS = {
    "P79-A": {
        "name": "P79-A (10M Control)",
        "model_scale": "10M",
        "d_model": 384,
        "n_layer": 6,
        "n_head": 8,
        "d_ff": 768,
        "exact_params": 10282304,
        "pretrain_steps": 500,
        "checkpoint": os.path.join(PROJECT_ROOT, "experiments", "phase79", "checkpoints", "collision_p79a.pt")
    },
    "P79-B": {
        "name": "P79-B (25M Candidate)",
        "model_scale": "25M",
        "d_model": 512,
        "n_layer": 10,
        "n_head": 8,
        "d_ff": 1024,
        "exact_params": 25263936,
        "pretrain_steps": 500,
        "checkpoint": os.path.join(PROJECT_ROOT, "experiments", "phase79", "checkpoints", "collision_p79b.pt")
    }
}

def load_binary_dataset(dataset_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_path = os.path.join(dataset_dir, "train.bin")
    val_path = os.path.join(dataset_dir, "val.bin")
    
    if not os.path.exists(train_path) or not os.path.exists(val_path):
        raise FileNotFoundError(f"Dataset binary files missing in {dataset_dir}")
        
    train_data = np.fromfile(train_path, dtype=np.uint16)
    val_data = np.fromfile(val_path, dtype=np.uint16)
    
    if len(val_data) < 600:
        actual_val = val_data
        test_data = val_data
    else:
        val_split_point = int(len(val_data) * 0.8)
        actual_val = val_data[:val_split_point]
        test_data = val_data[val_split_point:]
        
    return train_data, actual_val, test_data

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

def train_phase79_candidate(
    model_key: str,
    tokenizer: BPETokenizer,
    batch_size: int = 8,
    seq_len: int = 256,
    pretrain_steps: int = 500
) -> Dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spec = MODEL_SPECS[model_key]
    
    print(f"\n--- Training {spec['name']} ---", flush=True)
    
    cfg = ModelConfig(
        vocab_size=getattr(tokenizer, "vocab_size", 8000),
        max_seq_len=seq_len,
        d_model=spec["d_model"],
        n_layer=spec["n_layer"],
        n_head=spec["n_head"],
        d_ff=spec["d_ff"],
        dropout=0.1,
        tie_embeddings=True
    )
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    model = CollisionTransformer(cfg).to(device)
    exact_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    expected_params = spec["exact_params"]
    assert exact_params == expected_params, f"Parameter count mismatch for {model_key}: expected {expected_params}, got {exact_params}"
    
    train_data, val_data, test_data = load_binary_dataset(DATASET_DIR)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-4, weight_decay=0.01)
    
    model.train()
    start_time = time.time()
    train_loss_acc = 0.0
    
    step_history = []
    
    for step in range(1, pretrain_steps + 1):
        x, y = get_batch(train_data, batch_size=batch_size, seq_len=seq_len, device=device)
        optimizer.zero_grad()
        logits, loss = model(x, targets=y)
        loss.backward()
        optimizer.step()
        
        train_loss_acc += loss.item()
        
        if step % 100 == 0 or step == pretrain_steps:
            val_loss, val_ppl = evaluate_dataset_loss(model, val_data, eval_steps=20, batch_size=batch_size, seq_len=seq_len, device=device)
            current_avg_train = train_loss_acc / step
            print(f"[{model_key}] Step {step}/{pretrain_steps} | Train Loss: {current_avg_train:.4f} | Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.4f}", flush=True)
            step_history.append({
                "step": step,
                "train_loss": round(current_avg_train, 4),
                "val_loss": round(val_loss, 4),
                "val_ppl": round(val_ppl, 4)
            })
            model.train()
            
    elapsed_time = time.time() - start_time
    tokens_processed = pretrain_steps * batch_size * seq_len
    tokens_per_sec = tokens_processed / elapsed_time if elapsed_time > 0 else 0
    tokens_per_param = tokens_processed / exact_params
    
    # Final Evaluations
    final_val_loss, final_val_ppl = evaluate_dataset_loss(model, val_data, eval_steps=50, batch_size=batch_size, seq_len=seq_len, device=device)
    test_loss, test_ppl = evaluate_dataset_loss(model, test_data, eval_steps=50, batch_size=batch_size, seq_len=seq_len, device=device)
    
    # Checkpoint saving
    os.makedirs(os.path.dirname(spec["checkpoint"]), exist_ok=True)
    checkpoint_payload = {
        "model_key": model_key,
        "config": cfg.__dict__ if hasattr(cfg, "__dict__") else cfg,
        "model_state_dict": model.state_dict(),
        "exact_params": exact_params,
        "pretrain_steps": pretrain_steps,
        "tokens_processed": tokens_processed,
        "final_val_loss": final_val_loss,
        "final_val_ppl": final_val_ppl,
        "test_loss": test_loss,
        "test_ppl": test_ppl
    }
    torch.save(checkpoint_payload, spec["checkpoint"])
    print(f"Saved checkpoint to {spec['checkpoint']}", flush=True)
    
    return {
        "model_key": model_key,
        "name": spec["name"],
        "exact_params": exact_params,
        "pretrain_steps": pretrain_steps,
        "tokens_processed": tokens_processed,
        "tokens_per_param": round(tokens_per_param, 4),
        "final_train_loss": round(train_loss_acc / pretrain_steps, 4),
        "final_val_loss": round(final_val_loss, 4),
        "final_val_ppl": round(final_val_ppl, 4),
        "test_loss": round(test_loss, 4),
        "test_ppl": round(test_ppl, 4),
        "elapsed_time_s": round(elapsed_time, 2),
        "tokens_per_sec": round(tokens_per_sec, 2),
        "step_history": step_history,
        "checkpoint": spec["checkpoint"]
    }

def train_all_phase79_models(tokenizer: BPETokenizer) -> Dict[str, Dict[str, Any]]:
    results = {}
    for key, spec in MODEL_SPECS.items():
        pretrain_steps = spec.get("pretrain_steps", 500)
        results[key] = train_phase79_candidate(key, tokenizer, pretrain_steps=pretrain_steps)
    return results
