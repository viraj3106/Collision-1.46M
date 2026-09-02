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

EXP_DIR = os.path.join(PROJECT_ROOT, "experiments", "phase31")
CP_DIR = os.path.join(PROJECT_ROOT, "checkpoints", "phase31")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "collision-10m")
TOKENIZER_DIR = os.path.join(PROJECT_ROOT, "artifacts", "tokenizer")
AUG_DIR = os.path.join(PROJECT_ROOT, "datasets", "collision_augmented_v1")

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

def train_fine_tuned_model(base_state_dict, model_cfg, tokenizer, dataset_records, steps: int = 40, lr: float = 1e-4):
    model = CollisionTransformer(model_cfg)
    model.load_state_dict(base_state_dict)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    for step in range(1, steps + 1):
        optimizer.zero_grad()
        step_loss = 0.0
        # Mini-batch over dataset_records
        sample_batch = dataset_records if len(dataset_records) <= 30 else random.sample(dataset_records, 30)
        for rec in sample_batch:
            tokens = encode_prompt_response(tokenizer, rec.get("instruction", rec.get("prompt", "")), rec["response"])
            if len(tokens) < 2:
                continue
            x = torch.tensor([tokens[:-1]], dtype=torch.long)
            y = torch.tensor([tokens[1:]], dtype=torch.long)
            _, loss = model(x, y)
            loss.backward()
            step_loss += loss.item()
            
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
    model.eval()
    return model

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
    print("  PHASE 31: DATA ABLATION STUDY & AUGMENTATION PIPELINE  ")
    print("=========================================================")
    
    os.makedirs(EXP_DIR, exist_ok=True)
    os.makedirs(CP_DIR, exist_ok=True)

    # 1. Model Gate & Integrity Verification
    model_pt = os.path.join(MODEL_DIR, "model.pt")
    if not os.path.exists(model_pt):
        raise FileNotFoundError(f"Production model.pt not found at: {model_pt}")
        
    actual_sha = get_file_sha256(model_pt)
    print(f"Production Model Checksum: {actual_sha}")
    if actual_sha != EXPECTED_SHA256:
        raise ValueError(f"FATAL: Production model SHA256 mismatch! Expected {EXPECTED_SHA256}, got {actual_sha}")

    checkpoint_data = torch.load(model_pt, map_location="cpu")
    model_cfg = ModelConfig(**checkpoint_data["config"])
    baseline_model = CollisionTransformer(model_cfg)
    baseline_model.load_state_dict(checkpoint_data["model_state_dict"])
    
    param_count = sum(p.numel() for p in baseline_model.parameters())
    print(f"Production Model Parameters: {param_count:,}")
    if param_count != EXPECTED_PARAMS:
        raise ValueError(f"FATAL: Production parameter count mismatch! Expected {EXPECTED_PARAMS}, got {param_count}")

    print("Baseline COLLISION-10M Loaded & Integrity Verified Cleanly.\n")

    tokenizer = BPETokenizer()
    tokenizer.load(TOKENIZER_DIR)

    # 2. Load Datasets for Ablations
    rw_file = os.path.join(PROJECT_ROOT, "data", "real_world", "cleaned", "collision_real_world_v2.jsonl")
    syn_file = os.path.join(PROJECT_ROOT, "datasets", "collision_instruct_v1", "collision_synthetic_v1.jsonl")
    aug_train_file = os.path.join(AUG_DIR, "train.jsonl")
    aug_val_file = os.path.join(AUG_DIR, "val.jsonl")

    def load_jsonl(fpath):
        recs = []
        if os.path.exists(fpath):
            with open(fpath, "r", encoding="utf-8") as f:
                for l in f:
                    if l.strip():
                        recs.append(json.loads(l))
        return recs

    rw_recs = load_jsonl(rw_file)
    syn_recs = load_jsonl(syn_file)
    aug_train_recs = load_jsonl(aug_train_file)
    val_recs = load_jsonl(aug_val_file)

    print(f"Dataset Overview:")
    print(f"  Real-World v2: {len(rw_recs)} records")
    print(f"  Synthetic v1:  {len(syn_recs)} records")
    print(f"  Augmented Train: {len(aug_train_recs)} records")
    print(f"  Augmented Val:   {len(val_recs)} records\n")

    # 3. Train Experimental Ablation Checkpoints
    print("Executing Ablation B (Real-World Only)...")
    rw_model = train_fine_tuned_model(checkpoint_data["model_state_dict"], model_cfg, tokenizer, rw_recs, steps=30, lr=1e-4)
    cp_rw = os.path.join(CP_DIR, "collision_10m_realworld_only.pt")
    torch.save({"model_state_dict": rw_model.state_dict(), "config": model_cfg.__dict__}, cp_rw)

    print("Executing Ablation C (Synthetic Only)...")
    syn_model = train_fine_tuned_model(checkpoint_data["model_state_dict"], model_cfg, tokenizer, syn_recs, steps=30, lr=1e-4)
    cp_syn = os.path.join(CP_DIR, "collision_10m_synthetic_only.pt")
    torch.save({"model_state_dict": syn_model.state_dict(), "config": model_cfg.__dict__}, cp_syn)

    print("Executing Ablation D (Real-World + Synthetic Augmented)...")
    aug_model = train_fine_tuned_model(checkpoint_data["model_state_dict"], model_cfg, tokenizer, aug_train_recs, steps=40, lr=1e-4)
    cp_aug = os.path.join(CP_DIR, "collision_10m_augmented_v1.pt")
    torch.save({"model_state_dict": aug_model.state_dict(), "config": model_cfg.__dict__}, cp_aug)

    print("All Ablation Checkpoints Trained & Saved in checkpoints/phase31/.\n")

    # 4. Evaluate Loss & Perplexity across all 4 Ablation Configurations
    models = {
        "A_Baseline": baseline_model,
        "B_RealWorld_Only": rw_model,
        "C_Synthetic_Only": syn_model,
        "D_Augmented_v1": aug_model
    }

    loss_ppl_results = {}
    for name, m in models.items():
        loss, ppl = compute_dataset_loss(m, tokenizer, aug_val_file)
        loss_ppl_results[name] = {"val_loss": round(loss, 4), "perplexity": round(ppl, 2)}
        print(f"Model {name:<20} -> Val Loss: {loss:.4f} | Perplexity: {ppl:.2f}")

    # 5. Expanded Fixed Evaluation Suite (10 Target Prompts)
    fixed_eval_prompts = [
        {"domain": "AI", "prompt": "What is machine learning?"},
        {"domain": "CS", "prompt": "An algorithm is defined as"},
        {"domain": "Physics", "prompt": "Why does the Earth orbit the Sun?"},
        {"domain": "Technology", "prompt": "The future of cloud computing is"},
        {"domain": "Mathematics", "prompt": "Gradient descent is an optimization method that"},
        {"domain": "Space", "prompt": "A black hole is a region of spacetime where"},
        {"domain": "QA", "prompt": "Explain how a transformer model works."},
        {"domain": "General", "prompt": "Artificial intelligence is changing society by"},
        {"domain": "Explanation", "prompt": "Explain what photosynethesis is."},
        {"domain": "Completion", "prompt": "Computer science is"}
    ]

    print("\nRunning Fixed 10-Prompt Comparative Evaluation Suite...")
    eval_comparison = []

    model_rep_scores = {name: [] for name in models.keys()}
    model_pref_wins = {name: 0 for name in models.keys()}

    for item in fixed_eval_prompts:
        p = item["prompt"]
        dom = item["domain"]
        entry = {"domain": dom, "prompt": p, "outputs": {}, "metrics": {}}

        best_score = -1.0
        best_model_name = None

        for name, m in models.items():
            text, ids, toks, lat = generate_completion(m, tokenizer, p, max_tokens=60, temp=0.7)
            uniq_r, uni_r, bi_r, tri_r, longest = calculate_repetition_metrics(text, tokenizer)
            
            model_rep_scores[name].append(uni_r)
            entry["outputs"][name] = text.strip()
            entry["metrics"][name] = {
                "uniq_ratio": round(uniq_r, 4),
                "unigram_repeat": round(uni_r, 4),
                "trigram_repeat": round(tri_r, 4)
            }

            # Heuristic preference scoring (unique ratio balancing non-repetition)
            score = uniq_r * (1.0 - uni_r)
            if score > best_score:
                best_score = score
                best_model_name = name

        if best_model_name:
            model_pref_wins[best_model_name] += 1
            entry["preference_winner"] = best_model_name

        eval_comparison.append(entry)

    with open(os.path.join(EXP_DIR, "generation_comparison.jsonl"), "w", encoding="utf-8") as f:
        for entry in eval_comparison:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # 6. Summary Metrics & Decision Classification
    avg_reps = {name: round(sum(scores) / max(1, len(scores)), 4) for name, scores in model_rep_scores.items()}
    
    # Classification logic: compare D (Augmented_v1) against A (Baseline)
    aug_ppl = loss_ppl_results["D_Augmented_v1"]["perplexity"]
    base_ppl = loss_ppl_results["A_Baseline"]["perplexity"]
    aug_rep = avg_reps["D_Augmented_v1"]
    base_rep = avg_reps["A_Baseline"]

    if aug_ppl < base_ppl and aug_rep <= base_rep and model_pref_wins["D_Augmented_v1"] > model_pref_wins["A_Baseline"]:
        classification = "SIGNIFICANT_IMPROVEMENT"
    elif aug_ppl < base_ppl and aug_rep <= base_rep + 0.05:
        classification = "MODEST_IMPROVEMENT"
    elif abs(aug_ppl - base_ppl) < 1.0:
        classification = "NO_MEANINGFUL_IMPROVEMENT"
    else:
        classification = "REGRESSION"

    ablation_summary = {
        "loss_and_perplexity": loss_ppl_results,
        "average_unigram_repetition": avg_reps,
        "blind_preference_wins": model_pref_wins,
        "result_classification": classification
    }

    with open(os.path.join(EXP_DIR, "ablation_evaluation_results.json"), "w", encoding="utf-8") as f:
        json.dump(ablation_summary, f, indent=2)

    print("\n=========================================================")
    print(f"  PHASE 31 DATA ABLATION RESULT: {classification}")
    print(f"  Baseline (A)       -> Loss: {loss_ppl_results['A_Baseline']['val_loss']} | PPL: {loss_ppl_results['A_Baseline']['perplexity']} | Rep: {avg_reps['A_Baseline']}")
    print(f"  RealWorld Only (B) -> Loss: {loss_ppl_results['B_RealWorld_Only']['val_loss']} | PPL: {loss_ppl_results['B_RealWorld_Only']['perplexity']} | Rep: {avg_reps['B_RealWorld_Only']}")
    print(f"  Synthetic Only (C) -> Loss: {loss_ppl_results['C_Synthetic_Only']['val_loss']} | PPL: {loss_ppl_results['C_Synthetic_Only']['perplexity']} | Rep: {avg_reps['C_Synthetic_Only']}")
    print(f"  Augmented v1 (D)   -> Loss: {loss_ppl_results['D_Augmented_v1']['val_loss']} | PPL: {loss_ppl_results['D_Augmented_v1']['perplexity']} | Rep: {avg_reps['D_Augmented_v1']}")
    print("=========================================================\n")

if __name__ == "__main__":
    main()
