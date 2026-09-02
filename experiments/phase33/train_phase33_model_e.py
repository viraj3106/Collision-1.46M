import os
import sys
import time
import json
import torch
import random

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model.config import ModelConfig
from model.transformer import CollisionTransformer
from data.tokenize import BPETokenizer

EXP_DIR = os.path.join(PROJECT_ROOT, "experiments", "phase33")
CP_DIR = os.path.join(PROJECT_ROOT, "checkpoints", "phase33")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "collision-10m")
TOKENIZER_DIR = os.path.join(PROJECT_ROOT, "artifacts", "tokenizer")
AUG_V2_DIR = os.path.join(PROJECT_ROOT, "datasets", "collision_augmented_v2")
os.makedirs(CP_DIR, exist_ok=True)

EXPECTED_PARAMS = 10282304

def encode_pair(tokenizer, prompt, resp, max_seq_len=256):
    p_ids = tokenizer.encode(prompt, bos=True, eos=False)
    r_ids = tokenizer.encode(resp, bos=False, eos=True)
    comb = p_ids + r_ids
    if len(comb) > max_seq_len:
        comb = comb[:max_seq_len]
    return comb

def main():
    print("================================================================")
    print("  PHASE 33: TRAINING MODEL E (COLLISION-10M + AUGMENTED V2)     ")
    print("================================================================")

    model_pt = os.path.join(MODEL_DIR, "model.pt")
    ck_base = torch.load(model_pt, map_location="cpu")
    model_cfg = ModelConfig(**ck_base["config"])

    model = CollisionTransformer(model_cfg)
    model.load_state_dict(ck_base["model_state_dict"])
    
    params = sum(p.numel() for p in model.parameters())
    print(f"Loaded Baseline Model A: {params:,} parameters.")
    if params != EXPECTED_PARAMS:
        raise ValueError(f"Param mismatch: expected {EXPECTED_PARAMS}, got {params}")

    tokenizer = BPETokenizer()
    tokenizer.load(TOKENIZER_DIR)

    train_file = os.path.join(AUG_V2_DIR, "train.jsonl")
    train_records = []
    with open(train_file, "r", encoding="utf-8") as f:
        for l in f:
            if l.strip():
                train_records.append(json.loads(l))

    print(f"Loaded Augmented V2 Train Split: {len(train_records)} records.")

    # Controlled training hyperparameters
    lr = 1e-4
    steps = 60
    batch_size = 24
    seed = 42

    random.seed(seed)
    torch.manual_seed(seed)

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    print(f"Starting training run for {steps} steps (batch size={batch_size}, lr={lr})...")
    t0 = time.time()

    for step in range(1, steps + 1):
        optimizer.zero_grad()
        batch_sample = random.sample(train_records, min(batch_size, len(train_records)))
        step_loss = 0.0
        
        for rec in batch_sample:
            tokens = encode_pair(tokenizer, rec.get("instruction", rec.get("prompt", "")), rec.get("response", ""))
            if len(tokens) < 2:
                continue
            x = torch.tensor([tokens[:-1]], dtype=torch.long)
            y = torch.tensor([tokens[1:]], dtype=torch.long)
            _, loss = model(x, y)
            loss.backward()
            step_loss += loss.item()
            
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if step % 10 == 0 or step == steps:
            avg_loss = step_loss / max(1, len(batch_sample))
            print(f"  Step {step:03d} / {steps} | Loss: {avg_loss:.4f}")

    elapsed = time.time() - t0
    print(f"Training completed in {elapsed:.2f} seconds.")

    # Save Candidate Model E Checkpoint
    cp_dest = os.path.join(CP_DIR, "collision_10m_production_candidate_v2.pt")
    torch.save({"model_state_dict": model.state_dict(), "config": model_cfg.__dict__}, cp_dest)
    print(f"Saved Candidate Model E checkpoint to: {cp_dest}")

    # Save Training Config
    train_cfg_dict = {
        "candidate_model": "Model E (COLLISION-10M + Augmented V2)",
        "base_checkpoint": "models/collision-10m/model.pt",
        "output_checkpoint": cp_dest,
        "parameter_count": params,
        "seed": seed,
        "learning_rate": lr,
        "steps": steps,
        "batch_size": batch_size,
        "train_records_count": len(train_records),
        "training_time_seconds": round(elapsed, 2)
    }

    cfg_out = os.path.join(EXP_DIR, "training_config.json")
    with open(cfg_out, "w", encoding="utf-8") as f:
        json.dump(train_cfg_dict, f, indent=2)

    print(f"Saved training config to: {cfg_out}\n")

if __name__ == "__main__":
    main()
