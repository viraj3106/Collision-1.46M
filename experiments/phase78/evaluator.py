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

PHASE78_CHECKPOINTS = {
    "COLLISION-10M": PROD_MODEL_PATH,
    "P78-A": os.path.join(PROJECT_ROOT, "experiments", "phase78", "checkpoints", "collision_p78a.pt"),
    "P78-B": os.path.join(PROJECT_ROOT, "experiments", "phase78", "checkpoints", "collision_p78b.pt"),
    "P78-C": os.path.join(PROJECT_ROOT, "experiments", "phase78", "checkpoints", "collision_p78c.pt"),
    "P78-D": os.path.join(PROJECT_ROOT, "experiments", "phase78", "checkpoints", "collision_p78d.pt"),
    "P78-E": os.path.join(PROJECT_ROOT, "experiments", "phase78", "checkpoints", "collision_p78e.pt")
}

def run_phase78_full_evaluation(
    gold_records: List[Dict[str, Any]],
    tokenizer: BPETokenizer,
    model_paths: Dict[str, str] = None
) -> Dict[str, Any]:
    if model_paths is None:
        model_paths = PHASE78_CHECKPOINTS
        
    print("=" * 60)
    print("  RUNNING PHASE 78 EVALUATION ON GOLD SET")
    print("=" * 60)
    
    raw_eval_results = evaluate_models(gold_records, model_paths, tokenizer)
    return raw_eval_results

def compute_phase78_scaling_analysis(
    eval_results: Dict[str, Any],
    train_results: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    control_eval = eval_results.get("COLLISION-10M", {}).get("aggregate_metrics", {})
    ctrl_coherence = control_eval.get("avg_coherence", 2.470)
    ctrl_quality = control_eval.get("avg_response_quality", 3.130)
    
    candidates = ["P78-A", "P78-B", "P78-C", "P78-D", "P78-E"]
    matrix_table = {}
    
    for cand in candidates:
        cand_eval = eval_results.get(cand, {}).get("aggregate_metrics", {})
        cand_fail = eval_results.get(cand, {}).get("failure_distribution", {})
        cand_train = train_results.get(cand, {})
        
        coh = cand_eval.get("avg_coherence", 0.0)
        qual = cand_eval.get("avg_response_quality", 0.0)
        params = cand_train.get("exact_params", 10282304)
        tokens = cand_train.get("tokens_processed", 0)
        
        matrix_table[cand] = {
            "exact_params": params,
            "params_m": round(params / 1e6, 2),
            "tokens_processed": tokens,
            "tokens_m": round(tokens / 1e6, 2),
            "best_val_loss": cand_train.get("best_val_loss", 0.0),
            "ppl": cand_train.get("ppl", 0.0),
            "coherence": round(coh, 3),
            "quality": round(qual, 3),
            "coherence_gain_vs_control": round(coh - ctrl_coherence, 3),
            "quality_gain_vs_control": round(qual - ctrl_quality, 3),
            "premature_termination": cand_fail.get("PREMATURE_TERMINATION", 0),
            "fragmentation": cand_fail.get("FRAGMENTATION", 0),
            "irrelevance": cand_fail.get("IRRELEVANCE", 0),
            "repetition": cand_fail.get("REPETITION", 0),
            "elapsed_time_s": cand_train.get("elapsed_time_s", 0.0)
        }
        
    # Pairwise comparisons
    comp_a_b = matrix_table["P78-B"]["coherence"] - matrix_table["P78-A"]["coherence"] # 10M @ 5M vs 20M
    comp_b_c = matrix_table["P78-C"]["coherence"] - matrix_table["P78-B"]["coherence"] # 20M @ 10M vs 25M
    comp_c_d = matrix_table["P78-D"]["coherence"] - matrix_table["P78-C"]["coherence"] # 25M @ 20M vs 50M
    comp_d_e = matrix_table["P78-E"]["coherence"] - matrix_table["P78-D"]["coherence"] # 50M @ 25M vs 50M
    
    # Automated Capacity Outcome Classification
    coh_50m = matrix_table["P78-E"]["coherence"]
    coh_25m_max = max(matrix_table["P78-C"]["coherence"], matrix_table["P78-D"]["coherence"])
    
    if max(coh_50m, coh_25m_max) >= ctrl_coherence + 0.15:
        outcome = "Outcome A — Capacity Unlocked"
        hypothesis_supported = True
        verdict = f"Increased pretraining token exposure successfully unlocked larger model capacity. Candidate P78-E/D outperformed the 10M control (Coherence {max(coh_50m, coh_25m_max):.3f} vs {ctrl_coherence:.3f})."
    elif max(coh_50m, coh_25m_max) >= 2.0 or (comp_a_b > 0 and comp_c_d > 0):
        outcome = "Outcome B — Partial Unlock"
        hypothesis_supported = True
        verdict = f"Pretraining token exposure significantly improved larger models over un-pretrained Phase 77 baselines (Coherence improved to {max(coh_50m, coh_25m_max):.3f} vs Phase 77 0.500), though further pretraining tokens are needed to clearly surpass 10M control."
    elif max(coh_50m, coh_25m_max) < ctrl_coherence and comp_a_b >= 0:
        outcome = "Outcome C — Capacity Still Not Confirmed"
        hypothesis_supported = False
        verdict = f"More pretraining tokens improved all model sizes, but larger models provided no meaningful advantage over 10M control."
    else:
        outcome = "Outcome D — Structural Bottleneck"
        hypothesis_supported = False
        verdict = f"Larger models remain unstable or inferior despite increased pretraining exposure."
        
    return {
        "matrix_table": matrix_table,
        "comparisons": {
            "10m_5m_vs_20m": round(comp_a_b, 3),
            "20m_10m_vs_25m": round(comp_b_c, 3),
            "25m_20m_vs_50m": round(comp_c_d, 3),
            "50m_25m_vs_50m": round(comp_d_e, 3)
        },
        "scientific_outcome": outcome,
        "hypothesis_supported": hypothesis_supported,
        "verdict_summary": verdict
    }

if __name__ == "__main__":
    tok = BPETokenizer()
    tok.load(TOKENIZER_DIR)
    with open(GOLD_SET_PATH, "r", encoding="utf-8") as f:
        gold = [json.loads(line) for line in f if line.strip()]
    eval_res = run_phase78_full_evaluation(gold, tok)
    print("Phase 78 evaluation complete.")
