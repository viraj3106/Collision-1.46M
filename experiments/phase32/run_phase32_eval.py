import os
import sys
import time
import json
import math
import hashlib
import random
import torch
import torch.nn.functional as F
from collections import Counter

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model.config import ModelConfig
from model.transformer import CollisionTransformer
from data.tokenize import BPETokenizer
from inference.generate import top_k_top_p_filtering
from data.audit_generation_quality import calculate_repetition_metrics

EXP_DIR = os.path.join(PROJECT_ROOT, "experiments", "phase32")
EVAL_DIR = os.path.join(EXP_DIR, "evaluation_v1")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "collision-10m")
TOKENIZER_DIR = os.path.join(PROJECT_ROOT, "artifacts", "tokenizer")
AUG_DIR = os.path.join(PROJECT_ROOT, "datasets", "collision_augmented_v1")

EXPECTED_PARAMS = 10282304
EXPECTED_SHA256 = "d256d46d962d6416fe22d2cfe80b13df0574279fb980d7d8576c2bdcf3775b97"

MODEL_PATHS = {
    "Model_A_Baseline": os.path.join(MODEL_DIR, "model.pt"),
    "Model_B_RealWorld": os.path.join(PROJECT_ROOT, "checkpoints", "phase31", "collision_10m_realworld_only.pt"),
    "Model_C_Synthetic": os.path.join(PROJECT_ROOT, "checkpoints", "phase31", "collision_10m_synthetic_only.pt"),
    "Model_D_Augmented_v1": os.path.join(PROJECT_ROOT, "checkpoints", "phase31", "collision_10m_augmented_v1.pt")
}

def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_sha256(path):
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha.update(chunk)
    return sha.hexdigest()

def encode_pair(tokenizer, prompt, resp="", max_seq_len=256):
    p_ids = tokenizer.encode(prompt, bos=True, eos=False)
    r_ids = tokenizer.encode(resp, bos=False, eos=True) if resp else []
    comb = p_ids + r_ids
    if len(comb) > max_seq_len:
        comb = comb[:max_seq_len]
    return comb

def compute_split_loss(model, tokenizer, jsonl_path, device="cpu"):
    model.eval()
    if not os.path.exists(jsonl_path):
        return 0.0, float('inf')
    recs = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                recs.append(json.loads(line))
    if not recs:
        return 0.0, float('inf')
    
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for rec in recs:
            tokens = encode_pair(tokenizer, rec.get("instruction", rec.get("prompt", "")), rec.get("response", ""))
            if len(tokens) < 2:
                continue
            x = torch.tensor([tokens[:-1]], dtype=torch.long, device=device)
            y = torch.tensor([tokens[1:]], dtype=torch.long, device=device)
            _, loss = model(x, y)
            total_loss += loss.item()
            count += 1
    mean_loss = total_loss / max(1, count)
    ppl = math.exp(mean_loss) if mean_loss < 20 else float('inf')
    return mean_loss, ppl

def generate_locked(model, tokenizer, prompt, max_tokens=60, temp=0.7, top_k=40, top_p=0.9, seed=42, device="cpu"):
    set_seed(seed)
    model.eval()
    ids = tokenizer.encode(prompt, bos=True)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    
    t0 = time.perf_counter()
    tokens_gen = 0
    with torch.no_grad():
        for _ in range(max_tokens):
            x_cond = x if x.size(1) <= model.config.max_seq_len else x[:, -model.config.max_seq_len:]
            logits, _ = model(x_cond)
            next_logits = logits[0, -1, :]
            if temp > 0.0:
                next_logits = next_logits / temp
                filt_logits = top_k_top_p_filtering(next_logits, top_k=top_k, top_p=top_p)
                probs = F.softmax(filt_logits, dim=-1)
                next_tok = torch.multinomial(probs, num_samples=1)
            else:
                next_tok = torch.argmax(next_logits).unsqueeze(0)
            
            x = torch.cat((x, next_tok.unsqueeze(0)), dim=1)
            tokens_gen += 1
            if next_tok.item() == tokenizer.special_tokens.get("[EOS]", 259):
                break

    elapsed = time.perf_counter() - t0
    gen_ids = x[0][len(ids):].tolist()
    decoded = tokenizer.decode(gen_ids)
    return decoded.strip(), gen_ids, tokens_gen, elapsed

def evaluate_quality_heuristic(text, tokenizer, prompt):
    words = text.split()
    if not words:
        return {"coherence": 0.0, "relevance": 0.0, "completeness": 0.0, "repetition": 1.0, "stability": 0.0, "instruction": 0.0, "overall": 0.0, "unigram_repeat": 0.0, "trigram_repeat": 0.0, "unique_ratio": 0.0}
    
    uniq_r, uni_r, bi_r, tri_r, longest = calculate_repetition_metrics(text, tokenizer)
    
    # Coherence: balanced length and non-repeating structure
    rep_penalty = min(1.0, uni_r * 2.0 + tri_r * 3.0)
    coherence = max(0.0, 1.0 - rep_penalty)
    
    # Relevance: prompt keyword overlap or non-empty continuation
    p_words = set(prompt.lower().split())
    t_words = set(text.lower().split())
    overlap = len(p_words.intersection(t_words))
    relevance = min(1.0, 0.5 + 0.1 * overlap)
    
    # Completeness: terminates with sentence-ending punct or EOS
    ends_punct = text.endswith(('.', '!', '?', '"', '\n')) or len(words) < 55
    completeness = 1.0 if ends_punct else 0.6
    
    # Repetition score (higher is better: 1 = no rep, 0 = severe rep)
    repetition = max(0.0, 1.0 - uni_r)
    
    # Stability: penalty if longest repeated phrase > 4 words
    stability = max(0.0, 1.0 - (longest / 10.0))
    
    # Instruction following heuristic
    instruction = 0.8 if len(text) > 10 and coherence > 0.4 else 0.3
    
    overall = (coherence * 0.25) + (relevance * 0.20) + (completeness * 0.15) + (repetition * 0.20) + (stability * 0.10) + (instruction * 0.10)
    
    return {
        "coherence": round(coherence, 4),
        "relevance": round(relevance, 4),
        "completeness": round(completeness, 4),
        "repetition": round(repetition, 4),
        "stability": round(stability, 4),
        "instruction": round(instruction, 4),
        "overall": round(overall, 4),
        "unigram_repeat": round(uni_r, 4),
        "trigram_repeat": round(tri_r, 4),
        "unique_ratio": round(uniq_r, 4)
    }

def main():
    print("================================================================")
    print("  PHASE 32: PRODUCTION CANDIDATE EVALUATION PIPELINE RUNNER     ")
    print("================================================================")

    # Verification of Baseline Checkpoint
    base_path = MODEL_PATHS["Model_A_Baseline"]
    base_sha = get_sha256(base_path)
    print(f"Production Baseline Checkpoint SHA256: {base_sha}")
    if base_sha != EXPECTED_SHA256:
        raise ValueError(f"FATAL: Production baseline SHA256 mismatch! Got {base_sha}, expected {EXPECTED_SHA256}")

    tokenizer = BPETokenizer()
    tokenizer.load(TOKENIZER_DIR)

    # Load All 4 Models
    loaded_models = {}
    for name, path in MODEL_PATHS.items():
        ck = torch.load(path, map_location="cpu")
        cfg = ModelConfig(**ck["config"])
        m = CollisionTransformer(cfg)
        m.load_state_dict(ck["model_state_dict"])
        m.eval()
        p_count = sum(p.numel() for p in m.parameters())
        print(f"Loaded {name}: {p_count:,} params from {path}")
        if p_count != EXPECTED_PARAMS:
            raise ValueError(f"Param count mismatch for {name}: {p_count}")
        loaded_models[name] = m

    # 1. Dataset Split Loss & Perplexity (Train, Val, Test on collision_augmented_v1)
    print("\n--- QUANTITATIVE SPLIT EVALUATION (Train, Val, Test) ---")
    splits = {
        "train": os.path.join(AUG_DIR, "train.jsonl"),
        "val": os.path.join(AUG_DIR, "val.jsonl"),
        "test": os.path.join(AUG_DIR, "test.jsonl")
    }

    split_results = {}
    for m_name, m in loaded_models.items():
        split_results[m_name] = {}
        for s_name, s_path in splits.items():
            l, ppl = compute_split_loss(m, tokenizer, s_path)
            split_results[m_name][s_name] = {"loss": round(l, 4), "ppl": round(ppl, 2)}
            print(f"  {m_name:<20} [{s_name:<5}] -> Loss: {l:.4f} | PPL: {ppl:.2f}")

    # 2. Evaluation Suite Loading
    eval_suite_path = os.path.join(EVAL_DIR, "eval_suite_v1.json")
    rw_suite_path = os.path.join(EVAL_DIR, "realworld_eval_v1.json")
    
    with open(eval_suite_path, "r", encoding="utf-8") as f:
        core_eval_prompts = json.load(f)

    with open(rw_suite_path, "r", encoding="utf-8") as f:
        rw_eval_prompts = json.load(f)

    # 3. Fixed Prompt Evaluation & Pairwise Comparisons
    print(f"\n--- EVALUATING {len(core_eval_prompts)} CORE BENCHMARK PROMPTS ---")
    
    per_model_eval_metrics = {m: {"coherence": [], "relevance": [], "completeness": [], "repetition": [], "stability": [], "instruction": [], "overall": [], "unigram_rep": [], "trigram_rep": []} for m in loaded_models.keys()}
    per_category_metrics = {}
    
    detailed_eval_records = []
    
    blind_pairs = {
        "A_vs_D": {"A_wins": 0, "D_wins": 0, "ties": 0},
        "B_vs_D": {"B_wins": 0, "D_wins": 0, "ties": 0},
        "C_vs_D": {"C_wins": 0, "D_wins": 0, "ties": 0}
    }

    domain_regression_tracker = {}

    for item in core_eval_prompts:
        pid = item["id"]
        cat = item["category"]
        prompt = item["prompt"]

        if cat not in per_category_metrics:
            per_category_metrics[cat] = {m: [] for m in loaded_models.keys()}
        if cat not in domain_regression_tracker:
            domain_regression_tracker[cat] = {"Model_A_scores": [], "Model_D_scores": []}

        record = {"id": pid, "category": cat, "prompt": prompt, "generations": {}, "quality_metrics": {}}

        # Generate & score for all 4 models
        for m_name, m in loaded_models.items():
            gen_text, gen_ids, tok_cnt, lat = generate_locked(m, tokenizer, prompt, max_tokens=60, temp=0.7, seed=42)
            q_metrics = evaluate_quality_heuristic(gen_text, tokenizer, prompt)
            
            record["generations"][m_name] = gen_text
            record["quality_metrics"][m_name] = q_metrics

            # Aggregate
            for k in ["coherence", "relevance", "completeness", "repetition", "stability", "instruction", "overall"]:
                per_model_eval_metrics[m_name][k].append(q_metrics[k])
            per_model_eval_metrics[m_name]["unigram_rep"].append(q_metrics["unigram_repeat"])
            per_model_eval_metrics[m_name]["trigram_rep"].append(q_metrics["trigram_repeat"])

            per_category_metrics[cat][m_name].append(q_metrics["overall"])

            if m_name == "Model_A_Baseline":
                domain_regression_tracker[cat]["Model_A_scores"].append(q_metrics["overall"])
            elif m_name == "Model_D_Augmented_v1":
                domain_regression_tracker[cat]["Model_D_scores"].append(q_metrics["overall"])

        # Pairwise Blind Evaluation (Masked identities)
        score_A = record["quality_metrics"]["Model_A_Baseline"]["overall"]
        score_B = record["quality_metrics"]["Model_B_RealWorld"]["overall"]
        score_C = record["quality_metrics"]["Model_C_Synthetic"]["overall"]
        score_D = record["quality_metrics"]["Model_D_Augmented_v1"]["overall"]

        # A vs D
        if abs(score_A - score_D) < 0.05:
            blind_pairs["A_vs_D"]["ties"] += 1
        elif score_D > score_A:
            blind_pairs["A_vs_D"]["D_wins"] += 1
        else:
            blind_pairs["A_vs_D"]["A_wins"] += 1

        # B vs D
        if abs(score_B - score_D) < 0.05:
            blind_pairs["B_vs_D"]["ties"] += 1
        elif score_D > score_B:
            blind_pairs["B_vs_D"]["D_wins"] += 1
        else:
            blind_pairs["B_vs_D"]["B_wins"] += 1

        # C vs D
        if abs(score_C - score_D) < 0.05:
            blind_pairs["C_vs_D"]["ties"] += 1
        elif score_D > score_C:
            blind_pairs["C_vs_D"]["D_wins"] += 1
        else:
            blind_pairs["C_vs_D"]["C_wins"] += 1

        detailed_eval_records.append(record)

    # 4. Real-World Telemetry Generalization Benchmark
    print(f"\n--- EVALUATING {len(rw_eval_prompts)} REAL-WORLD TELEMETRY PROMPTS ---")
    rw_results = []
    rw_scores = {m: [] for m in loaded_models.keys()}
    for item in rw_eval_prompts:
        prompt = item["prompt"]
        cat = item["category"]
        rw_rec = {"prompt": prompt, "category": cat, "outputs": {}}
        for m_name, m in loaded_models.items():
            gen_text, _, _, _ = generate_locked(m, tokenizer, prompt, max_tokens=60, temp=0.7, seed=42)
            q_metrics = evaluate_quality_heuristic(gen_text, tokenizer, prompt)
            rw_rec["outputs"][m_name] = {"text": gen_text, "score": q_metrics["overall"]}
            rw_scores[m_name].append(q_metrics["overall"])
        rw_results.append(rw_rec)

    # Aggregate Overall Model Benchmark Summaries
    model_summary = {}
    for m_name in loaded_models.keys():
        model_summary[m_name] = {
            "mean_coherence": round(sum(per_model_eval_metrics[m_name]["coherence"]) / len(core_eval_prompts), 4),
            "mean_relevance": round(sum(per_model_eval_metrics[m_name]["relevance"]) / len(core_eval_prompts), 4),
            "mean_completeness": round(sum(per_model_eval_metrics[m_name]["completeness"]) / len(core_eval_prompts), 4),
            "mean_repetition_score": round(sum(per_model_eval_metrics[m_name]["repetition"]) / len(core_eval_prompts), 4),
            "mean_unigram_repeat": round(sum(per_model_eval_metrics[m_name]["unigram_rep"]) / len(core_eval_prompts), 4),
            "mean_trigram_repeat": round(sum(per_model_eval_metrics[m_name]["trigram_rep"]) / len(core_eval_prompts), 4),
            "mean_stability": round(sum(per_model_eval_metrics[m_name]["stability"]) / len(core_eval_prompts), 4),
            "mean_instruction_following": round(sum(per_model_eval_metrics[m_name]["instruction"]) / len(core_eval_prompts), 4),
            "overall_quality_score": round(sum(per_model_eval_metrics[m_name]["overall"]) / len(core_eval_prompts), 4),
            "realworld_generalization_score": round(sum(rw_scores[m_name]) / max(1, len(rw_scores[m_name])), 4)
        }

    # Aggregate Domain Regression Scores
    domain_summary = {}
    for cat, tracker in domain_regression_tracker.items():
        avg_A = sum(tracker["Model_A_scores"]) / max(1, len(tracker["Model_A_scores"]))
        avg_D = sum(tracker["Model_D_scores"]) / max(1, len(tracker["Model_D_scores"]))
        diff = avg_D - avg_A
        if diff > 0.05:
            status = "IMPROVED"
        elif diff < -0.05:
            status = "REGRESSED"
        else:
            status = "UNCHANGED"
        domain_summary[cat] = {
            "Model_A_mean": round(avg_A, 4),
            "Model_D_mean": round(avg_D, 4),
            "diff": round(diff, 4),
            "status": status
        }

    # Overfitting & Generalization Assessment
    train_loss_D = split_results["Model_D_Augmented_v1"]["train"]["loss"]
    val_loss_D = split_results["Model_D_Augmented_v1"]["val"]["loss"]
    test_loss_D = split_results["Model_D_Augmented_v1"]["test"]["loss"]
    overfit_gap = abs(val_loss_D - train_loss_D)
    
    is_overfit = overfit_gap > 1.5 or (train_loss_D < 0.5 and val_loss_D > 2.0)
    overfitting_status = "FAIL" if is_overfit else "PASS"

    # Export Results
    final_output = {
        "production_baseline_checksum": base_sha,
        "split_performance": split_results,
        "model_benchmarks": model_summary,
        "blind_preference_pairs": blind_pairs,
        "domain_regression_analysis": domain_summary,
        "overfitting_analysis": {
            "train_loss": train_loss_D,
            "val_loss": val_loss_D,
            "test_loss": test_loss_D,
            "overfitting_status": overfitting_status,
            "gap": round(overfit_gap, 4)
        },
        "realworld_eval_records": rw_results,
        "detailed_records": detailed_eval_records
    }

    out_json = os.path.join(EXP_DIR, "evaluation_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2)

    print("\n================================================================")
    print("  PHASE 32 EVALUATION SUMMARY RESULTS                           ")
    print("================================================================")
    for m_name, res in model_summary.items():
        val_l = split_results[m_name]["val"]["loss"]
        val_p = split_results[m_name]["val"]["ppl"]
        print(f"{m_name:<22} | Val Loss: {val_l:.4f} | PPL: {val_p:>6.2f} | Overall Quality: {res['overall_quality_score']:.4f} | Unigram Rep: {res['mean_unigram_repeat']:.4f} | RW Score: {res['realworld_generalization_score']:.4f}")
    
    print("\nBlind Pairwise Wins:")
    for pair, wins in blind_pairs.items():
        print(f"  {pair}: {wins}")

    print(f"\nOverfitting Status: {overfitting_status}")
    print(f"Results written to: {out_json}\n")

if __name__ == "__main__":
    main()
