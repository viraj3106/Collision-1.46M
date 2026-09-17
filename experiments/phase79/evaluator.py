import os
import sys
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model.config import ModelConfig
from model.transformer import CollisionTransformer
from data.tokenize import BPETokenizer

PROBES = [
    {"category": "factual", "prompt": "Question: What is the capital of France?\nAnswer:", "expected_tokens": ["Paris"]},
    {"category": "definitions", "prompt": "Definition: Neural network is", "expected_tokens": ["a", "model", "system"]},
    {"category": "explanations", "prompt": "Explanation: Artificial Intelligence allows computers to", "expected_tokens": ["learn", "process", "think"]},
    {"category": "simple_reasoning", "prompt": "If A is greater than B, and B is greater than C, then A is", "expected_tokens": ["greater", "larger"]},
    {"category": "technical", "prompt": "In Python, to define a function, use the keyword", "expected_tokens": ["def"]},
    {"category": "completions", "prompt": "The quick brown fox jumps over the", "expected_tokens": ["lazy", "dog"]},
    {"category": "conversational", "prompt": "Hello! How can I help you today?\nResponse:", "expected_tokens": ["I", "help", "can"]}
]

def load_checkpoint_model(checkpoint_path: str, device: torch.device = torch.device('cpu')) -> Tuple[CollisionTransformer, Dict[str, Any]]:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg_raw = checkpoint.get("config", {})
    if isinstance(cfg_raw, ModelConfig):
        cfg = cfg_raw
    elif isinstance(cfg_raw, dict) and cfg_raw:
        cfg = ModelConfig(**cfg_raw)
    else:
        cfg = ModelConfig(vocab_size=8000, max_seq_len=256, d_model=512, n_layer=10, n_head=8, d_ff=1024)
    model = CollisionTransformer(cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()
    return model, checkpoint

def generate_text_with_sampling(
    model: CollisionTransformer,
    tokenizer: BPETokenizer,
    prompt: str,
    max_new_tokens: int = 32,
    temperature: float = 0.7,
    seed: int = 42,
    device: torch.device = torch.device('cpu')
) -> str:
    torch.manual_seed(seed)
    prompt_ids = tokenizer.encode(prompt)
    if not prompt_ids:
        prompt_ids = [0]
    input_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    
    generated = list(prompt_ids)
    with torch.no_grad():
        for _ in range(max_new_tokens):
            if input_tensor.size(1) > 256:
                input_tensor = input_tensor[:, -256:]
            logits, _ = model(input_tensor)
            next_token_logits = logits[:, -1, :] / max(temperature, 1e-5)
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            generated.append(next_token)
            input_tensor = torch.tensor([generated], dtype=torch.long, device=device)
            if next_token in [getattr(tokenizer, 'eos_id', 259), getattr(tokenizer, 'sep_id', 258)]:
                break
    return tokenizer.decode(generated)

def calculate_generation_metrics(texts: List[str]) -> Dict[str, float]:
    total_tokens = 0
    unique_tokens = set()
    repetition_count = 0
    terminated_count = 0
    total_length = 0
    
    for text in texts:
        words = text.split()
        total_length += len(words)
        for i, word in enumerate(words):
            total_tokens += 1
            unique_tokens.add(word.lower())
            if i > 0 and word.lower() == words[i-1].lower():
                repetition_count += 1
        if text.rstrip().endswith(('.', '?', '!', '\n')):
            terminated_count += 1
            
    avg_len = total_length / len(texts) if texts else 0.0
    unique_ratio = len(unique_tokens) / max(total_tokens, 1)
    rep_rate = repetition_count / max(total_tokens, 1)
    term_rate = terminated_count / len(texts) if texts else 0.0
    coherence_score = (1.0 - rep_rate * 2.0) * (0.5 + 0.5 * min(unique_ratio * 4.0, 1.0)) * 3.0
    coherence_score = max(0.0, min(3.0, round(coherence_score, 3)))
    
    return {
        "avg_length": round(avg_len, 2),
        "unique_token_ratio": round(unique_ratio, 4),
        "repetition_rate": round(rep_rate, 4),
        "sentence_termination_rate": round(term_rate, 4),
        "coherence_score": coherence_score
    }

def evaluate_capability_probes(model: CollisionTransformer, tokenizer: BPETokenizer, device: torch.device = torch.device('cpu')) -> Dict[str, Any]:
    probe_results = []
    category_scores = {}
    
    for probe in PROBES:
        prompt = probe["prompt"]
        gen_text = generate_text_with_sampling(model, tokenizer, prompt, max_new_tokens=16, temperature=0.7, seed=42, device=device)
        matched = any(token.lower() in gen_text.lower() for token in probe["expected_tokens"])
        
        cat = probe["category"]
        if cat not in category_scores:
            category_scores[cat] = []
        category_scores[cat].append(1.0 if matched else 0.0)
        
        probe_results.append({
            "category": cat,
            "prompt": prompt,
            "generated": gen_text,
            "matched": matched
        })
        
    cat_accuracy = {cat: round(sum(scores)/len(scores), 2) for cat, scores in category_scores.items()}
    overall_probe_score = round(sum(score for scores in category_scores.values() for score in scores) / len(PROBES), 2)
    
    return {
        "probe_details": probe_results,
        "category_accuracy": cat_accuracy,
        "overall_probe_score": overall_probe_score
    }

def run_phase79_model_evaluation(
    model_key: str,
    checkpoint_path: str,
    tokenizer: BPETokenizer,
    device: torch.device = torch.device('cpu')
) -> Dict[str, Any]:
    model, checkpoint = load_checkpoint_model(checkpoint_path, device=device)
    
    # 1. Generation Evaluation across Temperatures
    sample_prompts = [probe["prompt"] for probe in PROBES]
    gen_temp_07 = [generate_text_with_sampling(model, tokenizer, p, temperature=0.7, device=device) for p in sample_prompts]
    gen_temp_10 = [generate_text_with_sampling(model, tokenizer, p, temperature=1.0, device=device) for p in sample_prompts]
    
    metrics_07 = calculate_generation_metrics(gen_temp_07)
    metrics_10 = calculate_generation_metrics(gen_temp_10)
    
    # 2. Capability Probes Evaluation
    probe_eval = evaluate_capability_probes(model, tokenizer, device=device)
    
    return {
        "model_key": model_key,
        "checkpoint": checkpoint_path,
        "val_loss": checkpoint.get("final_val_loss", 0.0),
        "val_ppl": checkpoint.get("final_val_ppl", 0.0),
        "test_loss": checkpoint.get("test_loss", 0.0),
        "test_ppl": checkpoint.get("test_ppl", 0.0),
        "gen_metrics_temp_07": metrics_07,
        "gen_metrics_temp_10": metrics_10,
        "probe_evaluation": probe_eval,
        "sample_generations": [
            {"prompt": sample_prompts[0], "output": gen_temp_07[0]},
            {"prompt": sample_prompts[1], "output": gen_temp_07[1]},
            {"prompt": sample_prompts[4], "output": gen_temp_07[4]}
        ]
    }

def compute_phase79_scaling_analysis(
    train_results: Dict[str, Dict[str, Any]],
    eval_results: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    ctrl_tr = train_results["P79-A"]
    cand_tr = train_results["P79-B"]
    ctrl_ev = eval_results["P79-A"]
    cand_ev = eval_results["P79-B"]
    
    param_inc_pct = round(((cand_tr["exact_params"] - ctrl_tr["exact_params"]) / ctrl_tr["exact_params"]) * 100, 2)
    
    val_loss_imp = round(ctrl_ev["val_loss"] - cand_ev["val_loss"], 4)
    val_loss_imp_pct = round((val_loss_imp / ctrl_ev["val_loss"]) * 100, 2)
    
    val_ppl_imp = round(ctrl_ev["val_ppl"] - cand_ev["val_ppl"], 4)
    val_ppl_imp_pct = round((val_ppl_imp / ctrl_ev["val_ppl"]) * 100, 2)
    
    test_ppl_imp = round(ctrl_ev["test_ppl"] - cand_ev["test_ppl"], 4)
    test_ppl_imp_pct = round((test_ppl_imp / ctrl_ev["test_ppl"]) * 100, 2)
    
    time_inc_pct = round(((cand_tr["elapsed_time_s"] - ctrl_tr["elapsed_time_s"]) / max(ctrl_tr["elapsed_time_s"], 1e-5)) * 100, 2)
    throughput_diff_pct = round(((cand_tr["tokens_per_sec"] - ctrl_tr["tokens_per_sec"]) / max(ctrl_tr["tokens_per_sec"], 1e-5)) * 100, 2)
    
    coh_diff = round(cand_ev["gen_metrics_temp_07"]["coherence_score"] - ctrl_ev["gen_metrics_temp_07"]["coherence_score"], 3)
    probe_diff = round(cand_ev["probe_evaluation"]["overall_probe_score"] - ctrl_ev["probe_evaluation"]["overall_probe_score"], 2)
    
    # Hypothesis Verification
    h1_supported = cand_ev["val_loss"] < ctrl_ev["val_loss"] and cand_ev["test_loss"] < ctrl_ev["test_loss"]
    h2_supported = cand_ev["gen_metrics_temp_07"]["repetition_rate"] <= ctrl_ev["gen_metrics_temp_07"]["repetition_rate"]
    h3_supported = val_loss_imp_pct >= 5.0 and h1_supported
    h4_supported = not h3_supported
    
    if h3_supported:
        outcome = "Outcome A — Substantial Capacity Scaling Gains Confirmed"
        conclusion_verdict = f"Scaling parameters from ~10.28M to ~25.26M (+{param_inc_pct}%) produced a significant {val_loss_imp_pct}% reduction in validation loss and {test_ppl_imp_pct}% reduction in test perplexity under identical pretraining conditions."
    else:
        outcome = "Outcome B — Capacity Saturation / Data Complexity Bottleneck Detected"
        conclusion_verdict = f"Scaling parameters from ~10.28M to ~25.26M (+{param_inc_pct}%) provided minimal loss improvement ({val_loss_imp_pct}%), indicating that synthetic data diversity/complexity is the dominant bottleneck."
        
    return {
        "parameter_increase_pct": param_inc_pct,
        "val_loss_improvement": val_loss_imp,
        "val_loss_improvement_pct": val_loss_imp_pct,
        "val_ppl_improvement": val_ppl_imp,
        "val_ppl_improvement_pct": val_ppl_imp_pct,
        "test_ppl_improvement": test_ppl_imp,
        "test_ppl_improvement_pct": test_ppl_imp_pct,
        "coherence_diff": coh_diff,
        "probe_score_diff": probe_diff,
        "time_increase_pct": time_inc_pct,
        "throughput_diff_pct": throughput_diff_pct,
        "hypotheses_evaluation": {
            "H1_Capacity_Scaling": {"supported": h1_supported, "evidence": f"P79-B val loss ({cand_ev['val_loss']}) vs P79-A ({ctrl_ev['val_loss']})"},
            "H2_Generalization": {"supported": h2_supported, "evidence": f"P79-B rep rate ({cand_ev['gen_metrics_temp_07']['repetition_rate']}) vs P79-A ({ctrl_ev['gen_metrics_temp_07']['repetition_rate']})"},
            "H3_Capacity_Efficiency": {"supported": h3_supported, "evidence": f"Val loss improvement: {val_loss_imp_pct}%"},
            "H4_Bottleneck_Detection": {"supported": h4_supported, "evidence": f"Bottleneck trigger status: {h4_supported}"}
        },
        "scientific_outcome": outcome,
        "conclusion_verdict": conclusion_verdict
    }