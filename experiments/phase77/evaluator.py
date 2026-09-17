import os
import sys
import json
import torch
from typing import Dict, Any, List, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.phase75.evaluation.evaluator import evaluate_models, PROD_MODEL_PATH, GOLD_SET_PATH, TOKENIZER_DIR
from data.tokenize import BPETokenizer

PHASE77_CHECKPOINTS = {
    "COLLISION-10M": os.path.join(PROJECT_ROOT, "experiments", "phase77", "checkpoints", "collision_10m_control.pt") if os.path.exists(os.path.join(PROJECT_ROOT, "experiments", "phase77", "checkpoints", "collision_10m_control.pt")) else PROD_MODEL_PATH,
    "J77-25M": os.path.join(PROJECT_ROOT, "experiments", "phase77", "checkpoints", "collision_25m_j77a.pt"),
    "J77-35M": os.path.join(PROJECT_ROOT, "experiments", "phase77", "checkpoints", "collision_35m_j77b.pt"),
    "J77-50M": os.path.join(PROJECT_ROOT, "experiments", "phase77", "checkpoints", "collision_50m_j77c.pt")
}

def run_phase77_full_evaluation(
    gold_records: List[Dict[str, Any]],
    tokenizer: BPETokenizer,
    model_paths: Dict[str, str] = None
) -> Dict[str, Any]:
    if model_paths is None:
        model_paths = PHASE77_CHECKPOINTS
        
    print("=" * 60)
    print("  RUNNING PHASE 77 CONVERSATIONAL EVALUATION")
    print("=" * 60)
    
    # Evaluate using phase75 evaluator pipeline
    raw_eval_results = evaluate_models(gold_records, model_paths, tokenizer)
    return raw_eval_results

def compute_scaling_metrics(
    eval_results: Dict[str, Any],
    train_results: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """Computes parameter efficiency, quality gain over 10M control,
    compute efficiency, and classifies scientific outcome.
    """
    control_key = "COLLISION-10M"
    ctrl_eval = eval_results.get(control_key, {}).get("aggregate_metrics", {})
    ctrl_quality = ctrl_eval.get("avg_response_quality", 3.1)
    ctrl_coherence = ctrl_eval.get("avg_coherence", 2.36)
    ctrl_params = train_results.get(control_key, {}).get("exact_params", 10282304)
    ctrl_time = train_results.get(control_key, {}).get("elapsed_time_s", 1.0)
    
    scaling_analysis = {}
    
    for candidate in ["COLLISION-10M", "J77-25M", "J77-35M", "J77-50M"]:
        cand_eval = eval_results.get(candidate, {}).get("aggregate_metrics", {})
        cand_failures = eval_results.get(candidate, {}).get("failure_distribution", {})
        cand_train = train_results.get(candidate, {})
        
        quality = cand_eval.get("avg_response_quality", 0.0)
        coherence = cand_eval.get("avg_coherence", 0.0)
        params = cand_train.get("exact_params", 10282304)
        time_s = cand_train.get("elapsed_time_s", 1.0)
        val_loss = cand_train.get("final_loss", 0.0)
        ppl = cand_train.get("ppl", 0.0)
        
        param_delta_m = (params - ctrl_params) / 1e6
        quality_gain = quality - ctrl_quality
        coherence_gain = coherence - ctrl_coherence
        
        param_efficiency = (coherence_gain / param_delta_m) if param_delta_m > 0 else 0.0
        compute_efficiency = (coherence_gain / (time_s - ctrl_time)) if (time_s - ctrl_time) > 0 else 0.0
        
        scaling_analysis[candidate] = {
            "exact_params": params,
            "params_m": round(params / 1e6, 2),
            "d_model": cand_train.get("d_model", 384),
            "n_layer": cand_train.get("n_layer", 6),
            "n_head": cand_train.get("n_head", 8),
            "d_ff": cand_train.get("d_ff", 768),
            "val_loss": val_loss,
            "ppl": ppl,
            "quality": round(quality, 3),
            "coherence": round(coherence, 3),
            "quality_gain": round(quality_gain, 3),
            "coherence_gain": round(coherence_gain, 3),
            "param_efficiency": round(param_efficiency, 4),
            "compute_efficiency": round(compute_efficiency, 4),
            "premature_termination": cand_failures.get("PREMATURE_TERMINATION", 0),
            "fragmentation": cand_failures.get("FRAGMENTATION", 0),
            "irrelevance": cand_failures.get("IRRELEVANCE", 0),
            "repetition": cand_failures.get("REPETITION", 0),
            "train_time_s": time_s
        }
        
    # Scientific outcome classification:
    # Outcome A: Capacity confirmed - 50M/35M show substantial coherence improvement (>0.2) over 10M
    # Outcome B: Capacity partially confirmed - moderate improvement (0.05-0.2)
    # Outcome C: Capacity not confirmed - negligible/no improvement (<0.05)
    coh_50m = scaling_analysis.get("J77-50M", {}).get("coherence", 0.0)
    coh_10m = scaling_analysis.get("COLLISION-10M", {}).get("coherence", 0.0)
    gain_50m = coh_50m - coh_10m
    
    if gain_50m >= 0.20:
        outcome = "Outcome A — Capacity Confirmed"
        hypothesis_supported = True
        verdict_summary = f"Increasing model parameter capacity to ~50M substantially improved multi-turn coherence (+{gain_50m:.2f}). Capacity is confirmed as the primary bottleneck."
    elif gain_50m >= 0.05:
        outcome = "Outcome B — Capacity Partially Confirmed"
        hypothesis_supported = True
        verdict_summary = f"Increasing model parameter capacity improved multi-turn coherence (+{gain_50m:.2f}), though persistent structural degradation remains."
    else:
        outcome = "Outcome C — Capacity Not Confirmed"
        hypothesis_supported = False
        verdict_summary = f"Increasing parameters provided negligible conversational improvement (+{gain_50m:.2f}). Evidence does NOT support parameter capacity as the dominant bottleneck."
        
    return {
        "scaling_table": scaling_analysis,
        "scientific_outcome": outcome,
        "hypothesis_supported": hypothesis_supported,
        "verdict_summary": verdict_summary
    }

if __name__ == "__main__":
    tok = BPETokenizer()
    tok.load(TOKENIZER_DIR)
    with open(GOLD_SET_PATH, "r", encoding="utf-8") as f:
        gold = [json.loads(line) for line in f if line.strip()]
    eval_res = run_phase77_full_evaluation(gold, tok)
    print("Phase 77 evaluation complete.")
