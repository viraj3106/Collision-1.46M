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
        "unique_ratio": round(unique_ratio, 4),
        "repetition_rate": round(rep_rate, 4),
        "termination_rate": round(term_rate, 4),
        "coherence_score": coherence_score
    }

def run_probe_evaluation(
    model: CollisionTransformer,
    tokenizer: BPETokenizer,
    device: torch.device = torch.device('cpu')
) -> Dict[str, Any]:
    cat_correct = {p["category"]: 0 for p in PROBES}
    cat_total = {p["category"]: 0 for p in PROBES}
    
    for probe in PROBES:
        cat = probe["category"]
        prompt = probe["prompt"]
        expected = [e.lower() for e in probe["expected_tokens"]]
        
        gen_text = generate_text_with_sampling(model, tokenizer, prompt, max_new_tokens=16, temperature=0.3, device=device)
        cat_total[cat] += 1
        
        matched = any(exp in gen_text.lower() for exp in expected)
        if matched:
            cat_correct[cat] += 1
            
    cat_acc = {cat: round(cat_correct[cat] / max(cat_total[cat], 1), 4) for cat in cat_total}
    overall_score = round(sum(cat_correct.values()) / max(sum(cat_total.values()), 1), 4)
    
    return {
        "category_accuracy": cat_acc,
        "overall_probe_score": overall_score
    }

def run_phase80_model_evaluation(
    cond_key: str,
    checkpoint_path: str,
    tokenizer: BPETokenizer,
    device: torch.device = torch.device('cpu')
) -> Dict[str, Any]:
    model, chk = load_checkpoint_model(checkpoint_path, device=device)
    
    # 1. Sample generation across prompts
    test_prompts = [p["prompt"] for p in PROBES]
    gen_texts = [generate_text_with_sampling(model, tokenizer, p, max_new_tokens=32, temperature=0.7, device=device) for p in test_prompts]
    gen_metrics = calculate_generation_metrics(gen_texts)
    
    # 2. Probe accuracy evaluation
    probe_eval = run_probe_evaluation(model, tokenizer, device=device)
    
    return {
        "cond_key": cond_key,
        "val_loss": round(chk.get("final_val_loss", 0.0), 4),
        "val_ppl": round(chk.get("final_val_ppl", 1.0), 4),
        "test_loss": round(chk.get("test_loss", 0.0), 4),
        "test_ppl": round(chk.get("test_ppl", 1.0), 4),
        "gen_metrics": gen_metrics,
        "probe_evaluation": probe_eval
    }

def compute_phase80_scaling_analysis(
    train_results: Dict[str, Dict[str, Any]],
    eval_results: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    keys = list(train_results.keys())
    p80a_val = eval_results["P80-A"]["val_loss"]
    p80e_val = eval_results["P80-E"]["val_loss"]
    
    abs_imp = round(p80a_val - p80e_val, 4)
    pct_imp = round(((p80a_val - p80e_val) / max(p80a_val, 1e-5)) * 100.0, 2)
    
    # Determine Outcome
    # Outcome A: Strong data scaling (>15% improvement from 1M to 100M)
    # Outcome B: Diminishing returns (improves then flattens)
    # Outcome C: Early saturation (<3% improvement)
    # Outcome D: Instability
    if pct_imp > 15.0:
        outcome = "Outcome A — Strong Data Scaling Confirmed"
        verdict = "COLLISION-25M is primarily data-constrained; scaling token budget yields strong continuous loss & quality improvements."
    elif pct_imp > 5.0:
        outcome = "Outcome B — Diminishing Returns"
        verdict = "COLLISION-25M improves with initial token scaling but exhibits diminishing marginal returns."
    elif pct_imp >= 0:
        outcome = "Outcome C — Early Saturation"
        verdict = "COLLISION-25M receives little benefit from token scaling beyond low token budgets."
    else:
        outcome = "Outcome D — Training Instability"
        verdict = "Higher token budgets caused training instability or loss degradation."

    return {
        "val_loss_improvement_abs": abs_imp,
        "val_loss_improvement_pct": pct_imp,
        "p80a_val_loss": p80a_val,
        "p80e_val_loss": p80e_val,
        "scientific_outcome": outcome,
        "conclusion_verdict": verdict
    }
