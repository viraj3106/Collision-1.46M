import os
import sys
import json
import copy
import hashlib
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import random
from typing import List, Dict, Any, Tuple
from model.config import ModelConfig
from model.transformer import CollisionTransformer
from data.tokenize import BPETokenizer

PROD_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "collision-10m", "model.pt")
EXPECTED_SHA256 = "d256d46d962d6416fe22d2cfe80b13df0574279fb980d7d8576c2bdcf3775b97"

def get_sha256(path: str) -> str:
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha.update(chunk)
    return sha.hexdigest()

def verify_immutability():
    if not os.path.exists(PROD_MODEL_PATH):
        raise FileNotFoundError(f"Production model missing at {PROD_MODEL_PATH}")
    prod_sha = get_sha256(PROD_MODEL_PATH)
    if prod_sha.lower() != EXPECTED_SHA256.lower():
        raise ValueError(f"CRITICAL: Production SHA256 mismatch! Expected {EXPECTED_SHA256}, got {prod_sha}")
    return prod_sha

def create_masked_batch(prompt: str, response: str, tokenizer: BPETokenizer, max_seq_len: int = 256):
    prompt_ids = tokenizer.encode(prompt, bos=True, eos=False)
    resp_ids = tokenizer.encode(response, bos=False, eos=True)
    
    full_ids = prompt_ids + resp_ids
    if len(full_ids) > max_seq_len:
        full_ids = full_ids[:max_seq_len]
        
    input_ids = full_ids[:-1]
    target_ids = full_ids[1:]
    
    prompt_len = max(0, len(prompt_ids) - 1)
    loss_mask = [0] * len(input_ids)
    for i in range(prompt_len, len(loss_mask)):
        loss_mask[i] = 1
        
    return {
        "input_ids": input_ids,
        "target_ids": target_ids,
        "loss_mask": loss_mask,
        "prompt_len": prompt_len,
        "raw_prompt": prompt,
        "raw_response": response
    }

def compute_sft_loss(candidate_model: CollisionTransformer, base_model: CollisionTransformer,
                     x: torch.Tensor, targets: torch.Tensor, loss_mask: torch.Tensor,
                     kl_weight: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Candidate forward pass
    cand_logits, _ = candidate_model(x)
    
    # Flat loss calculation
    vocab_size = cand_logits.size(-1)
    cand_flat = cand_logits.view(-1, vocab_size)
    targets_flat = targets.view(-1)
    mask_flat = loss_mask.view(-1)
    
    unmasked_loss = F.cross_entropy(cand_flat, targets_flat, reduction='none')
    masked_loss = (unmasked_loss * mask_flat).sum() / (mask_flat.sum() + 1e-8)
    
    kl_loss = torch.tensor(0.0, device=x.device)
    if kl_weight > 0.0 and base_model is not None:
        with torch.no_grad():
            base_logits, _ = base_model(x)
            
        p_base = F.softmax(base_logits, dim=-1)
        log_p_cand = F.log_softmax(cand_logits, dim=-1)
        
        # KL divergence: sum p_base * (log p_base - log p_cand)
        kl_pos = F.kl_div(log_p_cand, p_base, reduction='none').sum(dim=-1)
        kl_loss = (kl_pos.view(-1) * mask_flat).sum() / (mask_flat.sum() + 1e-8)
        
    total_loss = masked_loss + kl_weight * kl_loss
    return total_loss, masked_loss, kl_loss

def train_candidate(candidate_id: str, config: dict, data_records: list, tokenizer: BPETokenizer,
                    base_ckpt_path: str, save_path: str) -> dict:
    verify_immutability()
    
    ckpt = torch.load(base_ckpt_path, map_location="cpu", weights_only=False)
    cfg_dict = ckpt.get("config", {})
    cfg = cfg_dict if isinstance(cfg_dict, ModelConfig) else ModelConfig(**cfg_dict)
    
    cand_model = CollisionTransformer(cfg)
    cand_model.load_state_dict(ckpt["model_state_dict"])
    cand_model.train()
    
    base_model = None
    kl_weight = config.get("kl_weight", 0.0)
    if kl_weight > 0.0:
        base_model = CollisionTransformer(cfg)
        base_model.load_state_dict(ckpt["model_state_dict"])
        base_model.eval()
        for p in base_model.parameters():
            p.requires_grad = False

    optimizer = torch.optim.AdamW(cand_model.parameters(), lr=config.get("learning_rate", 1e-5))
    
    history = []
    step = 0
    total_records = len(data_records)
    batch_size = config.get("batch_size", 8)
    epochs = config.get("epochs", 1)
    
    print(f"\nTraining {candidate_id} for {epochs} epoch(s) over {total_records} records...", flush=True)
    
    for epoch in range(epochs):
        random.shuffle(data_records)
        for i in range(0, total_records, batch_size):
            batch_items = data_records[i:i+batch_size]
            
            optimizer.zero_grad()
            batch_loss = torch.tensor(0.0)
            batch_ce = torch.tensor(0.0)
            batch_kl = torch.tensor(0.0)
            
            for item in batch_items:
                prompt = item.get("prompt", "")
                response = item.get("response", "")
                m_data = create_masked_batch(prompt, response, tokenizer, cfg.max_seq_len)
                
                if len(m_data["input_ids"]) == 0:
                    continue
                    
                x = torch.tensor([m_data["input_ids"]], dtype=torch.long)
                t = torch.tensor([m_data["target_ids"]], dtype=torch.long)
                m = torch.tensor([m_data["loss_mask"]], dtype=torch.float32)
                
                tot_l, ce_l, kl_l = compute_sft_loss(cand_model, base_model, x, t, m, kl_weight)
                
                # Check for NaN / Inf
                if torch.isnan(tot_l) or torch.isinf(tot_l):
                    raise ValueError(f"CRITICAL: Exploding/NaN loss encountered at step {step}!")
                    
                tot_l.backward()
                batch_loss += tot_l.item()
                batch_ce += ce_l.item()
                batch_kl += kl_l.item()
                
            torch.nn.utils.clip_grad_norm_(cand_model.parameters(), config.get("max_grad_norm", 1.0))
            optimizer.step()
            step += 1
            
            if step % 20 == 0 or i + batch_size >= total_records:
                avg_l = float(batch_loss / len(batch_items))
                avg_ce = float(batch_ce / len(batch_items))
                avg_kl = float(batch_kl / len(batch_items))
                history.append({
                    "step": step,
                    "total_loss": round(avg_l, 4),
                    "ce_loss": round(avg_ce, 4),
                    "kl_loss": round(avg_kl, 4)
                })
                print(f"[{candidate_id}] Step {step} | Loss: {avg_l:.4f} | CE: {avg_ce:.4f} | KL: {avg_kl:.4f}", flush=True)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    save_ckpt = {
        "model_state_dict": cand_model.state_dict(),
        "config": cfg,
        "candidate_id": candidate_id,
        "training_history": history
    }
    torch.save(save_ckpt, save_path)
    print(f"Saved candidate {candidate_id} checkpoint to {save_path}", flush=True)
    return {"candidate_id": candidate_id, "history": history}

