import os
import sys
import json
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.tokenize import BPETokenizer
from experiments.phase79.train_phase79 import train_all_phase79_models, MODEL_SPECS
from experiments.phase79.evaluator import run_phase79_model_evaluation, compute_phase79_scaling_analysis
from experiments.phase79.generate_report import generate_phase79_report

TOKENIZER_DIR = os.path.join(PROJECT_ROOT, "artifacts", "tokenizer")
RESULTS_JSON_PATH = os.path.join(PROJECT_ROOT, "experiments", "phase79", "results", "phase79_results.json")
HISTORY_PATH = os.path.join(PROJECT_ROOT, "experiments", "experiments_history.jsonl")

def main():
    print("=" * 70)
    print("  PHASE 79 — 10M -> 25M CAPACITY SCALING EXPERIMENT")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Tokenizer
    tokenizer = BPETokenizer()
    tokenizer.load(TOKENIZER_DIR)
    print(f"Loaded Tokenizer from {TOKENIZER_DIR} (Vocab size: {getattr(tokenizer, 'vocab_size', 8000)})")
    
    # 2. Train Models (P79-A 10M Control & P79-B 25M Candidate)
    train_results = train_all_phase79_models(tokenizer)
    
    # 3. Evaluate Models
    eval_results = {}
    for key in MODEL_SPECS.keys():
        chk_path = train_results[key]["checkpoint"]
        print(f"Evaluating {key} from checkpoint {chk_path}...")
        eval_results[key] = run_phase79_model_evaluation(key, chk_path, tokenizer, device=device)
        
    # 4. Compute Comparative Scaling Analysis
    scaling_analysis = compute_phase79_scaling_analysis(train_results, eval_results)
    
    # 5. Save Results Payload JSON
    results_payload = {
        "train_results": train_results,
        "eval_results": eval_results,
        "scaling_analysis": scaling_analysis
    }
    os.makedirs(os.path.dirname(RESULTS_JSON_PATH), exist_ok=True)
    with open(RESULTS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)
    print(f"Saved Phase 79 results JSON to {RESULTS_JSON_PATH}")
    
    # 6. Generate Markdown Report
    generate_phase79_report(train_results, eval_results, scaling_analysis)
    
    # 7. Append to experiments_history.jsonl
    history_entry = {
        "phase": "phase79",
        "action": "10M_TO_25M_CAPACITY_SCALING_EXPERIMENT",
        "scientific_outcome": scaling_analysis["scientific_outcome"],
        "conclusion_verdict": scaling_analysis["conclusion_verdict"],
        "p79a_params": train_results["P79-A"]["exact_params"],
        "p79b_params": train_results["P79-B"]["exact_params"],
        "p79a_val_loss": eval_results["P79-A"]["val_loss"],
        "p79b_val_loss": eval_results["P79-B"]["val_loss"],
        "val_loss_imp_pct": scaling_analysis["val_loss_improvement_pct"],
        "test_ppl_imp_pct": scaling_analysis["test_ppl_improvement_pct"]
    }
    os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)
    with open(HISTORY_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(history_entry) + "\n")
    print(f"Updated experiments history log at {HISTORY_PATH}")
    
    print("=" * 70)
    print(f"  PHASE 79 COMPLETE — OUTCOME: {scaling_analysis['scientific_outcome']}")
    print("=" * 70)

if __name__ == "__main__":
    main()
