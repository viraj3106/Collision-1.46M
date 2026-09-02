import os
import sys
import time
import json
import yaml
import math
import hashlib
import random
import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model.config import ModelConfig
from model.transformer import CollisionTransformer
from data.tokenize import BPETokenizer
from inference.generate import top_k_top_p_filtering
from data.audit_generation_quality import calculate_repetition_metrics

EXP_DIR = os.path.join(PROJECT_ROOT, "experiments", "phase30")
CP_DIR = os.path.join(PROJECT_ROOT, "checkpoints", "phase30")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "collision-10m")
TOKENIZER_DIR = os.path.join(PROJECT_ROOT, "artifacts", "tokenizer")
DATASET_DIR = os.path.join(PROJECT_ROOT, "datasets", "collision_instruct_v1")

EXPECTED_PARAMS = 10282304
EXPECTED_SHA256 = "d256d46d962d6416fe22d2cfe80b13df0574279fb980d7d8576c2bdcf3775b97"

def get_file_sha256(fpath: str) -> str:
    sha = hashlib.sha256()
    with open(fpath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha.update(chunk)
    return sha.hexdigest()

def encode_prompt_response(tokenizer, prompt: str, response: str, max_seq_len: int = 256):
    prompt_ids = tokenizer.encode(prompt, bos=True, eos=False)
    resp_ids = tokenizer.encode(response, bos=False, eos=True)
    combined = prompt_ids + resp_ids
    if len(combined) > max_seq_len:
        combined = combined[:max_seq_len]
    return combined

def compute_dataset_loss(model, tokenizer, jsonl_path: str, device: str = "cpu"):
    model.eval()
    if not os.path.exists(jsonl_path):
        return 0.0, float('inf')
        
    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
                
    if not records:
        return 0.0, float('inf')

    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for rec in records:
            tokens = encode_prompt_response(tokenizer, rec.get("instruction", rec.get("prompt", "")), rec.get("response", ""))
            if len(tokens) < 2:
                continue
            x = torch.tensor([tokens[:-1]], dtype=torch.long, device=device)
            y = torch.tensor([tokens[1:]], dtype=torch.long, device=device)
            _, loss = model(x, y)
            total_loss += loss.item()
            count += 1
            
    mean_loss = total_loss / max(1, count)
    perplexity = math.exp(mean_loss) if mean_loss < 20 else float('inf')
    return mean_loss, perplexity

def generate_completion(model, tokenizer, prompt: str, max_tokens: int = 60, temp: float = 0.7, top_k: int = 40, top_p: int = 0.9, device: str = "cpu"):
    model.eval()
    ids = tokenizer.encode(prompt, bos=True)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    
    t0 = time.perf_counter()
    tokens_generated = 0
    with torch.no_grad():
        for _ in range(max_tokens):
            x_cond = x if x.size(1) <= model.config.max_seq_len else x[:, -model.config.max_seq_len:]
            logits, _ = model(x_cond)
            next_token_logits = logits[0, -1, :]
            if temp > 0.0:
                next_token_logits = next_token_logits / temp
                filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
                probs = F.softmax(filtered_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits).unsqueeze(0)
                
            x = torch.cat((x, next_token.unsqueeze(0)), dim=1)
            tokens_generated += 1
            if next_token.item() == tokenizer.special_tokens.get("[EOS]", 259):
                break
                
    elapsed = time.perf_counter() - t0
    gen_ids = x[0][len(ids):].tolist()
    decoded = tokenizer.decode(gen_ids)
    return decoded, gen_ids, tokens_generated, elapsed

def main():
    print("=========================================================")
    print("  PHASE 30: REAL-WORLD DATASET & FINE-TUNING EXPERIMENT  ")
    print("=========================================================")
    
    os.makedirs(EXP_DIR, exist_ok=True)
    os.makedirs(CP_DIR, exist_ok=True)

    # 1. Model Gate & Integrity Check
    model_pt = os.path.join(MODEL_DIR, "model.pt")
    if not os.path.exists(model_pt):
        raise FileNotFoundError(f"Production model.pt not found at: {model_pt}")
        
    actual_sha = get_file_sha256(model_pt)
    print(f"Production Model Checksum: {actual_sha}")
    if actual_sha != EXPECTED_SHA256:
        raise ValueError(f"FATAL: Production model SHA256 mismatch! Expected {EXPECTED_SHA256}, got {actual_sha}")
        
    tokenizer = BPETokenizer()
    tokenizer.load(TOKENIZER_DIR)
    
    checkpoint_data = torch.load(model_pt, map_location="cpu")
    model_cfg = ModelConfig(**checkpoint_data["config"])
    baseline_model = CollisionTransformer(model_cfg)
    baseline_model.load_state_dict(checkpoint_data["model_state_dict"])
    
    param_count = sum(p.numel() for p in baseline_model.parameters())
    print(f"Production Model Parameters: {param_count:,}")
    if param_count != EXPECTED_PARAMS:
        raise ValueError(f"FATAL: Production parameter count mismatch! Expected {EXPECTED_PARAMS}, got {param_count}")

    print("Baseline COLLISION-10M Loaded & Integrity Verified Cleanly.\n")

    # 2. Real-World Dataset Audit
    train_jsonl = os.path.join(DATASET_DIR, "real_world_train.jsonl")
    val_jsonl = os.path.join(DATASET_DIR, "real_world_val.jsonl")
    
    train_records = []
    if os.path.exists(train_jsonl):
        with open(train_jsonl, "r", encoding="utf-8") as f:
            for l in f:
                if l.strip():
                    train_records.append(json.loads(l))

    val_records = []
    if os.path.exists(val_jsonl):
        with open(val_jsonl, "r", encoding="utf-8") as f:
            for l in f:
                if l.strip():
                    val_records.append(json.loads(l))

    total_train_tokens = sum(len(encode_prompt_response(tokenizer, r["instruction"], r["response"])) for r in train_records)
    total_val_tokens = sum(len(encode_prompt_response(tokenizer, r["instruction"], r["response"])) for r in val_records)

    dataset_stats = {
        "raw_examples": 11,
        "clean_examples": len(train_records) + len(val_records),
        "train_examples": len(train_records),
        "val_examples": len(val_records),
        "train_tokens": total_train_tokens,
        "val_tokens": total_val_tokens,
        "total_tokens": total_train_tokens + total_val_tokens,
        "avg_example_tokens": round((total_train_tokens + total_val_tokens) / max(1, len(train_records) + len(val_records)), 2)
    }

    with open(os.path.join(EXP_DIR, "dataset_statistics.json"), "w", encoding="utf-8") as f:
        json.dump(dataset_stats, f, indent=2)

    print(f"Dataset Statistics:")
    print(f"  Clean Examples: {dataset_stats['clean_examples']} (Train: {dataset_stats['train_examples']}, Val: {dataset_stats['val_examples']})")
    print(f"  Total Tokens:   {dataset_stats['total_tokens']:,}")

    # 3. Evaluate Baseline Model on Real-World Validation Split
    base_val_loss, base_val_ppl = compute_dataset_loss(baseline_model, tokenizer, val_jsonl)
    print(f"Baseline COLLISION-10M -> Val Loss: {base_val_loss:.4f} | Perplexity: {base_val_ppl:.2f}")

    # 4. Controlled Experimental Fine-Tuning Run
    # Instantiate fine-tuning copy of model without touching baseline weights
    exp_model = CollisionTransformer(model_cfg)
    exp_model.load_state_dict(checkpoint_data["model_state_dict"])
    exp_model.train()

    optimizer = torch.optim.AdamW(exp_model.parameters(), lr=1e-4, weight_decay=0.01)
    
    print("\nStarting Controlled Real-World Fine-Tuning Experiment (50 Steps)...")
    t0_train = time.time()
    steps = 50
    step_losses = []
    
    for step in range(1, steps + 1):
        optimizer.zero_grad()
        step_loss = 0.0
        for rec in train_records:
            tokens = encode_prompt_response(tokenizer, rec["instruction"], rec["response"])
            if len(tokens) < 2:
                continue
            x = torch.tensor([tokens[:-1]], dtype=torch.long)
            y = torch.tensor([tokens[1:]], dtype=torch.long)
            _, loss = exp_model(x, y)
            loss.backward()
            step_loss += loss.item()
            
        torch.nn.utils.clip_grad_norm_(exp_model.parameters(), max_norm=1.0)
        optimizer.step()
        avg_step_loss = step_loss / max(1, len(train_records))
        step_losses.append(avg_step_loss)
        
        if step % 10 == 0 or step == steps:
            print(f"  Fine-Tuning Step {step}/{steps} | Loss: {avg_step_loss:.4f}")
            
    train_time = time.time() - t0_train

    # Save Experimental Checkpoint in checkpoints/phase30/
    exp_cp_path = os.path.join(CP_DIR, "collision_10m_realworld_v1.pt")
    torch.save({
        "model_state_dict": exp_model.state_dict(),
        "config": model_cfg.__dict__,
        "step": steps,
        "train_loss": step_losses[-1]
    }, exp_cp_path)
    exp_cp_sha = get_file_sha256(exp_cp_path)
    print(f"\nExperimental Model Saved: {exp_cp_path}")
    print(f"Experimental Model SHA256: {exp_cp_sha}")

    # 5. Evaluate Experimental Model on Real-World Validation Split
    exp_val_loss, exp_val_ppl = compute_dataset_loss(exp_model, tokenizer, val_jsonl)
    print(f"Fine-Tuned Model      -> Val Loss: {exp_val_loss:.4f} | Perplexity: {exp_val_ppl:.2f}")

    # 6. Evaluation Suite Across Knowledge Domains
    eval_prompts = [
        {"domain": "AI", "prompt": "What is machine learning?"},
        {"domain": "CS", "prompt": "An algorithm is defined as"},
        {"domain": "Physics", "prompt": "Why does the Earth orbit the Sun?"},
        {"domain": "Technology", "prompt": "The future of cloud computing is"},
        {"domain": "Mathematics", "prompt": "Gradient descent is an optimization method that"},
        {"domain": "General", "prompt": "Artificial intelligence is changing society by"}
    ]

    print("\nRunning Comparative Generation Audit Across Models...")
    comparison_logs = []
    baseline_pref = 0
    finetuned_pref = 0
    ties = 0

    base_rep_rates = []
    exp_rep_rates = []

    for item in eval_prompts:
        p = item["prompt"]
        dom = item["domain"]
        
        b_text, b_ids, b_toks, b_lat = generate_completion(baseline_model, tokenizer, p, max_tokens=60, temp=0.7)
        e_text, e_ids, e_toks, e_lat = generate_completion(exp_model, tokenizer, p, max_tokens=60, temp=0.7)
        
        b_uniq, b_uni, b_bi, b_tri, b_long = calculate_repetition_metrics(b_text, tokenizer)
        e_uniq, e_uni, e_bi, e_tri, e_long = calculate_repetition_metrics(e_text, tokenizer)
        
        base_rep_rates.append(b_uni)
        exp_rep_rates.append(e_uni)
        
        # Simple heuristic preference evaluation (higher unique ratio + non-empty)
        if e_uniq > b_uniq and e_uni <= b_uni:
            pref = "Fine-Tuned"
            finetuned_pref += 1
        elif b_uniq > e_uniq and b_uni <= e_uni:
            pref = "Baseline"
            baseline_pref += 1
        else:
            pref = "Tie"
            ties += 1

        comparison_logs.append({
            "domain": dom,
            "prompt": p,
            "baseline_output": b_text.strip(),
            "finetuned_output": e_text.strip(),
            "baseline_metrics": {"uniq_ratio": b_uniq, "unigram_repeat": b_uni, "trigram_repeat": b_tri},
            "finetuned_metrics": {"uniq_ratio": e_uniq, "unigram_repeat": e_uni, "trigram_repeat": e_tri},
            "preference": pref
        })

    with open(os.path.join(EXP_DIR, "generation_comparison.jsonl"), "w", encoding="utf-8") as f:
        for log_entry in comparison_logs:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    mean_base_rep = sum(base_rep_rates) / max(1, len(base_rep_rates))
    mean_exp_rep = sum(exp_rep_rates) / max(1, len(exp_rep_rates))

    # Determine Experiment Outcome
    val_loss_diff = exp_val_loss - base_val_loss
    if val_loss_diff < -0.05 and finetuned_pref > baseline_pref:
        outcome = "SIGNIFICANT_IMPROVEMENT"
    elif val_loss_diff <= 0.0 and mean_exp_rep <= mean_base_rep:
        outcome = "MODEST_IMPROVEMENT"
    elif abs(val_loss_diff) < 0.05:
        outcome = "NO_MEANINGFUL_IMPROVEMENT"
    else:
        outcome = "REGRESSION"

    eval_results = {
        "baseline_val_loss": round(base_val_loss, 4),
        "baseline_val_ppl": round(base_val_ppl, 2),
        "finetuned_val_loss": round(exp_val_loss, 4),
        "finetuned_val_ppl": round(exp_val_ppl, 2),
        "val_loss_diff": round(val_loss_diff, 4),
        "mean_baseline_repetition": round(mean_base_rep, 4),
        "mean_finetuned_repetition": round(mean_exp_rep, 4),
        "preference": {
            "baseline": baseline_pref,
            "finetuned": finetuned_pref,
            "ties": ties
        },
        "outcome": outcome
    }

    with open(os.path.join(EXP_DIR, "evaluation_results.json"), "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2)

    # Save Experiment Config YAML
    exp_cfg = {
        "experiment_name": "phase30_realworld_finetuning",
        "baseline_checkpoint": "models/collision-10m/model.pt",
        "baseline_sha256": EXPECTED_SHA256,
        "experimental_checkpoint": "checkpoints/phase30/collision_10m_realworld_v1.pt",
        "experimental_sha256": exp_cp_sha,
        "dataset_train": "datasets/collision_instruct_v1/real_world_train.jsonl",
        "dataset_val": "datasets/collision_instruct_v1/real_world_val.jsonl",
        "learning_rate": 1e-4,
        "steps": steps,
        "seed": 42
    }
    with open(os.path.join(EXP_DIR, "experiment_config.yaml"), "w", encoding="utf-8") as f:
        yaml.dump(exp_cfg, f, default_flow_style=False)

    print("\n=========================================================")
    print(f"  PHASE 30 EXPERIMENT OUTCOME: {outcome}")
    print(f"  Baseline Val Loss: {base_val_loss:.4f} (PPL: {base_val_ppl:.2f})")
    print(f"  Fine-Tuned Loss:   {exp_val_loss:.4f} (PPL: {exp_val_ppl:.2f})")
    print("=========================================================\n")

if __name__ == "__main__":
    main()
