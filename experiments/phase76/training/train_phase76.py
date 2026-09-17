import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any, List

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.phase73.train_phase73 import verify_immutability, create_masked_batch
from model.config import ModelConfig
from model.transformer import CollisionTransformer
from data.tokenize import BPETokenizer

PROD_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "collision-10m", "model.pt")

def compute_hybrid_loss(
    model: CollisionTransformer,
    x: torch.Tensor,
    targets: torch.Tensor,
    loss_mask: torch.Tensor,
    alpha: float = 0.10,
    beta: float = 0.90
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    out = model(x)
    logits = out[0] if isinstance(out, tuple) else out
    
    vocab_size = logits.size(-1)
    logits_flat = logits.view(-1, vocab_size)
    targets_flat = targets.view(-1)
    mask_flat = loss_mask.view(-1)
    
    unmasked_loss = F.cross_entropy(logits_flat, targets_flat, reduction='none')
    
    # Context tokens mask (1 - loss_mask) vs Response tokens mask (loss_mask)
    context_weight = (1.0 - mask_flat) * alpha
    response_weight = mask_flat * beta
    total_weights = context_weight + response_weight
    
    weighted_loss = (unmasked_loss * total_weights).sum() / (total_weights.sum() + 1e-8)
    context_loss = (unmasked_loss * (1.0 - mask_flat)).sum() / ((1.0 - mask_flat).sum() + 1e-8)
    response_loss = (unmasked_loss * mask_flat).sum() / (mask_flat.sum() + 1e-8)
    
    return weighted_loss, context_loss, response_loss

def train_phase76_candidate(
    candidate_name: str,
    alpha: float,
    beta: float,
    tokenizer: BPETokenizer,
    save_path: str,
    steps: int = 20
) -> Dict[str, Any]:
    verify_immutability()
    
    device = torch.device("cpu")
    ckpt = torch.load(PROD_MODEL_PATH, map_location=device, weights_only=False)
    cfg_dict = ckpt.get("config", {})
    cfg = cfg_dict if isinstance(cfg_dict, ModelConfig) else ModelConfig(**cfg_dict)
    
    model = CollisionTransformer(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt, strict=False)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-5)
    
    # Controlled synthetic conversation data batching
    sample_pairs = [
        ("User: My name is Alex.", "Assistant: Nice to meet you Alex!"),
        ("User: What is Python?", "Assistant: Python is a clean programming language."),
        ("User: Explain gravity.", "Assistant: Gravity pulls objects together."),
        ("User: List 3 primary colors.", "Assistant: 1. Red 2. Blue 3. Yellow.")
    ]
    
    model.train()
    history = []
    
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
        
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": cfg,
        "candidate": candidate_name,
        "alpha": alpha,
        "beta": beta
    }, save_path)
    
    print(f"Saved {candidate_name} checkpoint to {save_path}", flush=True)
    return {"candidate": candidate_name, "final_loss": history[-1]["loss"], "history": history}
