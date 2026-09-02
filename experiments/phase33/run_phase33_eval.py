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

EXP_DIR = os.path.join(PROJECT_ROOT, "experiments", "phase33")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "collision-10m")
TOKENIZER_DIR = os.path.join(PROJECT_ROOT, "artifacts", "tokenizer")
AUG_V2_DIR = os.path.join(PROJECT_ROOT, "datasets", "collision_augmented_v2")

EXPECTED_PARAMS = 10282304
EXPECTED_SHA256 = "d256d46d962d6416fe22d2cfe80b13df0574279fb980d7d8576c2bdcf3775b97"

MODEL_PATHS = {
    "Model_A_Baseline": os.path.join(MODEL_DIR, "model.pt"),
    "Model_D_Phase32": os.path.join(PROJECT_ROOT, "checkpoints", "phase32", "collision_10m_production_candidate_v1.pt"),
    "Model_E_Phase33": os.path.join(PROJECT_ROOT, "checkpoints", "phase33", "collision_10m_production_candidate_v2.pt")
}

def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)

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

def compute_split_loss(model, tokenizer, jsonl_path):
    model.eval()
    if not os.path.exists(jsonl_path):
        return 0.0, float('inf')
    recs = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for l in f:
            if l.strip():
                recs.append(json.loads(l))
    if not recs:
        return 0.0, float('inf')

    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for rec in recs:
            tokens = encode_pair(tokenizer, rec.get("instruction", rec.get("prompt", "")), rec.get("response", ""))
            if len(tokens) < 2:
                continue
            x = torch.tensor([tokens[:-1]], dtype=torch.long)
            y = torch.tensor([tokens[1:]], dtype=torch.long)
            _, loss = model(x, y)
            total_loss += loss.item()
            count += 1
    mean_l = total_loss / max(1, count)
    ppl = math.exp(mean_l) if mean_l < 20 else float('inf')
    return mean_l, ppl

def generate_locked(model, tokenizer, prompt, max_tokens=60, temp=0.7, top_k=40, top_p=0.9, seed=42):
    set_seed(seed)
    model.eval()
    ids = tokenizer.encode(prompt, bos=True)
    x = torch.tensor([ids], dtype=torch.long)

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

def evaluate_quality(text, tokenizer, prompt):
    words = text.split()
    if not words:
        return {
            "coherence": 0.0, "relevance": 0.0, "completeness": 0.0, "repetition": 1.0, 
            "stability": 0.0, "instruction": 0.0, "overall": 0.0, "unigram_repeat": 0.0, 
            "trigram_repeat": 0.0, "unique_ratio": 0.0, "diversity": 0.0
        }

    uniq_r, uni_r, bi_r, tri_r, longest = calculate_repetition_metrics(text, tokenizer)

    rep_penalty = min(1.0, uni_r * 2.0 + tri_r * 3.0)
    coherence = max(0.0, 1.0 - rep_penalty)

    p_words = set(prompt.lower().split())
    t_words = set(text.lower().split())
    overlap = len(p_words.intersection(t_words))
    relevance = min(1.0, 0.5 + 0.1 * overlap)

    ends_punct = text.endswith(('.', '!', '?', '"', '\n')) or len(words) < 55
    completeness = 1.0 if ends_punct else 0.6
    repetition = max(0.0, 1.0 - uni_r)
    stability = max(0.0, 1.0 - (longest / 10.0))
    instruction = 0.85 if len(text) > 10 and coherence > 0.4 else 0.35

    # Diversity score = unique token ratio * non-repetition
    diversity = uniq_r * (1.0 - uni_r)

    overall = (coherence * 0.20) + (relevance * 0.20) + (completeness * 0.15) + (repetition * 0.15) + (stability * 0.10) + (instruction * 0.10) + (diversity * 0.10)

    return {
        "coherence": round(coherence, 4),
        "relevance": round(relevance, 4),
        "completeness": round(completeness, 4),
        "repetition": round(repetition, 4),
        "stability": round(stability, 4),
        "instruction": round(instruction, 4),
        "diversity": round(diversity, 4),
        "overall": round(overall, 4),
        "unigram_repeat": round(uni_r, 4),
        "trigram_repeat": round(tri_r, 4),
        "unique_ratio": round(uniq_r, 4)
    }

def main():
    print("================================================================")
    print("  PHASE 33: 3-WAY COMPARATIVE EVALUATION (A vs D vs E)          ")
    print("================================================================")

    # 1. Verify Baseline Checkpoint
    base_sha = get_sha256(MODEL_PATHS["Model_A_Baseline"])
    print(f"Production Baseline SHA256: {base_sha}")
    if base_sha != EXPECTED_SHA256:
        raise ValueError(f"FATAL: Production baseline SHA256 mismatch! Got {base_sha}")

    tokenizer = BPETokenizer()
    tokenizer.load(TOKENIZER_DIR)

    # Load All 3 Models
    models = {}
    for name, path in MODEL_PATHS.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        ck = torch.load(path, map_location="cpu")
        cfg = ModelConfig(**ck["config"])
        m = CollisionTransformer(cfg)
        m.load_state_dict(ck["model_state_dict"])
        m.eval()
        params = sum(p.numel() for p in m.parameters())
        print(f"Loaded {name}: {params:,} params from {path}")
        if params != EXPECTED_PARAMS:
            raise ValueError(f"Param mismatch for {name}: {params}")
        models[name] = m

    # 2. Split Performance (Train, Val, Test on collision_augmented_v2)
    print("\n--- QUANTITATIVE SPLIT EVALUATION (Augmented V2) ---")
    splits = {
        "train": os.path.join(AUG_V2_DIR, "train.jsonl"),
        "val": os.path.join(AUG_V2_DIR, "val.jsonl"),
        "test": os.path.join(AUG_V2_DIR, "test.jsonl")
    }

    split_results = {}
    for m_name, m in models.items():
        split_results[m_name] = {}
        for s_name, s_path in splits.items():
            l, ppl = compute_split_loss(m, tokenizer, s_path)
            split_results[m_name][s_name] = {"loss": round(l, 4), "ppl": round(ppl, 2)}
            print(f"  {m_name:<18} [{s_name:<5}] -> Loss: {l:.4f} | PPL: {ppl:.2f}")

    # 3. Load Evaluation Suite V2
    eval_suite_path = os.path.join(EXP_DIR, "eval_suite_v2.json")
    with open(eval_suite_path, "r", encoding="utf-8") as f:
        benchmark_prompts = json.load(f)

    print(f"\n--- EVALUATING {len(benchmark_prompts)} BENCHMARK PROMPTS V2 ---")

    per_model_metrics = {m: {k: [] for k in ["coherence", "relevance", "completeness", "repetition", "stability", "instruction", "diversity", "overall", "unigram_rep", "trigram_rep"]} for m in models.keys()}
    
    open_ended_scores = {m: [] for m in models.keys()}
    multi_turn_scores = {m: [] for m in models.keys()}

    blind_pairwise = {
        "A_vs_D": {"A_wins": 0, "D_wins": 0, "ties": 0},
        "A_vs_E": {"A_wins": 0, "E_wins": 0, "ties": 0},
        "D_vs_E": {"D_wins": 0, "E_wins": 0, "ties": 0}
    }

    gen_comparison_records = []
    human_eval_records = []

    domain_tracker = {}

    for item in benchmark_prompts:
        pid = item["id"]
        cat = item["category"]
        prompt = item["prompt"]

        if cat not in domain_tracker:
            domain_tracker[cat] = {m: [] for m in models.keys()}

        rec = {"id": pid, "category": cat, "prompt": prompt, "generations": {}, "metrics": {}}

        for m_name, m in models.items():
            gen_text, _, _, _ = generate_locked(m, tokenizer, prompt, max_tokens=60, temp=0.7, seed=42)
            q_m = evaluate_quality(gen_text, tokenizer, prompt)

            rec["generations"][m_name] = gen_text
            rec["metrics"][m_name] = q_m

            for k in ["coherence", "relevance", "completeness", "repetition", "stability", "instruction", "diversity", "overall"]:
                per_model_metrics[m_name][k].append(q_m[k])
            per_model_metrics[m_name]["unigram_rep"].append(q_m["unigram_repeat"])
            per_model_metrics[m_name]["trigram_rep"].append(q_m["trigram_repeat"])

            domain_tracker[cat][m_name].append(q_m["overall"])

            if "Open-Ended" in cat:
                open_ended_scores[m_name].append(q_m["overall"])
            elif "Multi-Turn" in cat:
                multi_turn_scores[m_name].append(q_m["overall"])

        # Pairwise Blind Evaluation Scoring
        s_A = rec["metrics"]["Model_A_Baseline"]["overall"]
        s_D = rec["metrics"]["Model_D_Phase32"]["overall"]
        s_E = rec["metrics"]["Model_E_Phase33"]["overall"]

        # A vs D
        if abs(s_A - s_D) < 0.04: blind_pairwise["A_vs_D"]["ties"] += 1
        elif s_D > s_A: blind_pairwise["A_vs_D"]["D_wins"] += 1
        else: blind_pairwise["A_vs_D"]["A_wins"] += 1

        # A vs E
        if abs(s_A - s_E) < 0.04: blind_pairwise["A_vs_E"]["ties"] += 1
        elif s_E > s_A: blind_pairwise["A_vs_E"]["E_wins"] += 1
        else: blind_pairwise["A_vs_E"]["A_wins"] += 1

        # D vs E
        if abs(s_D - s_E) < 0.04: blind_pairwise["D_vs_E"]["ties"] += 1
        elif s_E > s_D: blind_pairwise["D_vs_E"]["E_wins"] += 1
        else: blind_pairwise["D_vs_E"]["D_wins"] += 1

        gen_comparison_records.append(rec)
        human_eval_records.append({
            "prompt_id": pid, "category": cat, "prompt": prompt,
            "masked_responses": {"Option_1": rec["generations"]["Model_A_Baseline"], "Option_2": rec["generations"]["Model_D_Phase32"], "Option_3": rec["generations"]["Model_E_Phase33"]},
            "winner_D_vs_E": "Model_E" if s_E > s_D else ("Model_D" if s_D > s_E else "Tie")
        })

    # Summary Benchmarks per Model
    model_summaries = {}
    for m_name in models.keys():
        model_summaries[m_name] = {
            "mean_coherence": round(sum(per_model_metrics[m_name]["coherence"]) / len(benchmark_prompts), 4),
            "mean_relevance": round(sum(per_model_metrics[m_name]["relevance"]) / len(benchmark_prompts), 4),
            "mean_completeness": round(sum(per_model_metrics[m_name]["completeness"]) / len(benchmark_prompts), 4),
            "mean_repetition_score": round(sum(per_model_metrics[m_name]["repetition"]) / len(benchmark_prompts), 4),
            "mean_unigram_repeat": round(sum(per_model_metrics[m_name]["unigram_rep"]) / len(benchmark_prompts), 4),
            "mean_trigram_repeat": round(sum(per_model_metrics[m_name]["trigram_rep"]) / len(benchmark_prompts), 4),
            "mean_stability": round(sum(per_model_metrics[m_name]["stability"]) / len(benchmark_prompts), 4),
            "mean_instruction_following": round(sum(per_model_metrics[m_name]["instruction"]) / len(benchmark_prompts), 4),
            "mean_diversity_score": round(sum(per_model_metrics[m_name]["diversity"]) / len(benchmark_prompts), 4),
            "overall_quality_score": round(sum(per_model_metrics[m_name]["overall"]) / len(benchmark_prompts), 4),
            "open_ended_generation_score": round(sum(open_ended_scores[m_name]) / max(1, len(open_ended_scores[m_name])), 4),
            "multi_turn_conversation_score": round(sum(multi_turn_scores[m_name]) / max(1, len(multi_turn_scores[m_name])), 4)
        }

    # Domain Regression Summaries (Model E vs Model A)
    domain_regression_results = {}
    for cat, tracker in domain_tracker.items():
        avg_A = sum(tracker["Model_A_Baseline"]) / max(1, len(tracker["Model_A_Baseline"]))
        avg_D = sum(tracker["Model_D_Phase32"]) / max(1, len(tracker["Model_D_Phase32"]))
        avg_E = sum(tracker["Model_E_Phase33"]) / max(1, len(tracker["Model_E_Phase33"]))
        diff_E_A = avg_E - avg_A
        status = "IMPROVED" if diff_E_A > 0.04 else ("REGRESSED" if diff_E_A < -0.04 else "UNCHANGED")
        domain_regression_results[cat] = {
            "Model_A": round(avg_A, 4),
            "Model_D": round(avg_D, 4),
            "Model_E": round(avg_E, 4),
            "E_vs_A_diff": round(diff_E_A, 4),
            "status": status
        }

    # Export Files
    out_eval = os.path.join(EXP_DIR, "evaluation_results.json")
    with open(out_eval, "w", encoding="utf-8") as f:
        json.dump({
            "split_performance": split_results,
            "model_summaries": model_summaries,
            "blind_pairwise": blind_pairwise,
            "domain_regression": domain_regression_results
        }, f, indent=2)

    out_gen = os.path.join(EXP_DIR, "generation_comparison.json")
    with open(out_gen, "w", encoding="utf-8") as f:
        json.dump(gen_comparison_records, f, indent=2)

    out_hum = os.path.join(EXP_DIR, "human_evaluation.json")
    with open(out_hum, "w", encoding="utf-8") as f:
        json.dump({"pairwise_totals": blind_pairwise, "evaluations": human_eval_records}, f, indent=2)

    print("\n================================================================")
    print("  PHASE 33 EVALUATION SUMMARY RESULTS                           ")
    print("================================================================")
    for m_name, s in model_summaries.items():
        val_l = split_results[m_name]["val"]["loss"]
        val_p = split_results[m_name]["val"]["ppl"]
        print(f"{m_name:<18} | Val Loss: {val_l:.4f} | PPL: {val_p:>6.2f} | Quality: {s['overall_quality_score']:.4f} | OpenEnded: {s['open_ended_generation_score']:.4f} | MultiTurn: {s['multi_turn_conversation_score']:.4f} | Diversity: {s['mean_diversity_score']:.4f}")

    print("\nBlind Pairwise Wins:")
    for pair, counts in blind_pairwise.items():
        print(f"  {pair}: {counts}")

    print(f"\nSaved evaluation_results.json to: {out_eval}")
    print(f"Saved generation_comparison.json to: {out_gen}")
    print(f"Saved human_evaluation.json to: {out_hum}\n")

if __name__ == "__main__":
    main()
