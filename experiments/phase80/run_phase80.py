import os
import sys
import json
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.tokenize import BPETokenizer
from experiments.phase80.train_phase80 import train_all_phase80_conditions, CONDITION_SPECS
from experiments.phase80.evaluator import run_phase80_model_evaluation, compute_phase80_scaling_analysis
from experiments.phase80.generate_report import generate_phase80_report

TOKENIZER_DIR = os.path.join(PROJECT_ROOT, "artifacts", "tokenizer")
RESULTS_JSON_PATH = os.path.join(PROJECT_ROOT, "experiments", "phase80", "results", "phase80_results.json")
HISTORY_PATH = os.path.join(PROJECT_ROOT, "experiments", "experiments_history.jsonl")

def main():
    print("=" * 70)
    print("  PHASE 80 — 25M PRETRAINING DATA SCALING EXPERIMENT")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Tokenizer
    tokenizer = BPETokenizer()
    tokenizer.load(TOKENIZER_DIR)
    print(f"Loaded Tokenizer from {TOKENIZER_DIR} (Vocab size: {getattr(tokenizer, 'vocab_size', 8000)})")
    
    # 2. Train Conditions (P80-A 1M, P80-B 10M, P80-C 25M, P80-D 50M, P80-E 100M)
    train_results = train_all_phase80_conditions(tokenizer)
    
    # 3. Evaluate Models
    eval_results = {}
    for key in CONDITION_SPECS.keys():
        chk_path = train_results[key]["checkpoint"]
        print(f"Evaluating {key} from checkpoint {chk_path}...")
        eval_results[key] = run_phase80_model_evaluation(key, chk_path, tokenizer, device=device)
        
    # 4. Compute Comparative Scaling Analysis
    scaling_analysis = compute_phase80_scaling_analysis(train_results, eval_results)
    
    # 5. Save Results Payload JSON
    results_payload = {
        "train_results": train_results,
        "eval_results": eval_results,
        "scaling_analysis": scaling_analysis
    }
    os.makedirs(os.path.dirname(RESULTS_JSON_PATH), exist_ok=True)
    with open(RESULTS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)
    print(f"Saved Phase 80 results JSON to {RESULTS_JSON_PATH}")
    
    # 6. Generate Markdown Report
    generate_phase80_report(train_results, eval_results, scaling_analysis)
    
    # 7. Append to experiments_history.jsonl
    history_entry = {
        "phase": "phase80",
        "action": "25M_PRETRAINING_DATA_SCALING_EXPERIMENT",
        "scientific_outcome": scaling_analysis["scientific_outcome"],
        "conclusion_verdict": scaling_analysis["conclusion_verdict"],
        "p80a_val_loss": eval_results["P80-A"]["val_loss"],
        "p80e_val_loss": eval_results["P80-E"]["val_loss"],
        "val_loss_imp_pct": scaling_analysis["val_loss_improvement_pct"]
    }
    os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)
    with open(HISTORY_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(history_entry) + "\n")
    print(f"Updated experiments history log at {HISTORY_PATH}")
    
    print("=" * 70)
    print(f"  PHASE 80 COMPLETE — OUTCOME: {scaling_analysis['scientific_outcome']}")
    print("=" * 70)

if __name__ == "__main__":
    main()
