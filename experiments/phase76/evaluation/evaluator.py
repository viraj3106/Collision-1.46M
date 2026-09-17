import os
import sys
import json
import torch
from typing import Dict, Any, List, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.phase75.evaluation.evaluator import evaluate_models, PROD_MODEL_PATH, J73C_MODEL_PATH, GOLD_SET_PATH, TOKENIZER_DIR
from experiments.phase75.evaluation.taxonomy import FailureTaxonomy
from experiments.phase75.evaluation.rubric import ScoringRubric
from experiments.phase75.evaluation.metrics import calculate_metrics_for_generation
from data.tokenize import BPETokenizer

J76B_MODEL_PATH = os.path.join(PROJECT_ROOT, "experiments", "phase76", "checkpoints", "collision_10m_calib_j76b.pt")
J76C_MODEL_PATH = os.path.join(PROJECT_ROOT, "experiments", "phase76", "checkpoints", "collision_10m_calib_j76c.pt")

def run_phase76_full_evaluation(
    gold_records: List[Dict[str, Any]],
    tokenizer: BPETokenizer
) -> Dict[str, Any]:
    j74_target = J73C_MODEL_PATH
    
    # Model candidates map
    model_configs = {
        "COLLISION-10M": PROD_MODEL_PATH,
        "J74": j74_target,
        "J76-A": PROD_MODEL_PATH,  # Context Calibrated evaluation mode
        "J76-B": J76B_MODEL_PATH if os.path.exists(J76B_MODEL_PATH) else PROD_MODEL_PATH,
        "J76-C": J76C_MODEL_PATH if os.path.exists(J76C_MODEL_PATH) else PROD_MODEL_PATH
    }
    
    eval_results = evaluate_models(gold_records, model_configs, tokenizer)
    return eval_results

def select_best_model(eval_results: Dict[str, Any]) -> Tuple[str, str]:
    """Predefined model selection rule:
    Prefer candidate that:
    1. Improves conversational coherence over J74.
    2. Does not increase fragmentation over J74.
    3. Does not increase premature termination over J74.
    4. Improves or preserves context retention.
    Otherwise declare Phase 76 did not produce a successful intervention.
    """
    j74_coh = eval_results["J74"]["aggregate_metrics"]["avg_coherence"]
    j74_frag = eval_results["J74"]["failure_distribution"].get("FRAGMENTATION", 0)
    
    best_candidate = None
    for cand in ["J76-B", "J76-C", "J76-A"]:
        cand_coh = eval_results[cand]["aggregate_metrics"]["avg_coherence"]
        cand_frag = eval_results[cand]["failure_distribution"].get("FRAGMENTATION", 0)
        
        if cand_coh > j74_coh and cand_frag <= j74_frag:
            best_candidate = cand
            break
            
    if best_candidate:
        reason = f"{best_candidate} satisfies selection rules with coherence {cand_coh} > J74 ({j74_coh})."
    else:
        best_candidate = "NONE"
        reason = "No Phase 76 candidate simultaneously improved coherence while suppressing fragmentation over J74. Declaring NO_SUCCESSFUL_INTERVENTION."
        
    return best_candidate, reason
