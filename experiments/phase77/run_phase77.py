import os
import sys
import json
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.tokenize import BPETokenizer
from experiments.phase77.train_phase77 import train_all_phase77_models
from experiments.phase77.evaluator import run_phase77_full_evaluation, compute_scaling_metrics
from experiments.phase77.generate_report import generate_phase77_report

TOKENIZER_DIR = os.path.join(PROJECT_ROOT, "artifacts", "tokenizer")
GOLD_SET_PATH = os.path.join(PROJECT_ROOT, "experiments", "phase75", "data", "gold", "gold_eval_set.jsonl")
RESULTS_JSON_PATH = os.path.join(PROJECT_ROOT, "experiments", "phase77", "results", "phase77_results.json")
HISTORY_PATH = os.path.join(PROJECT_ROOT, "experiments", "experiments_history.jsonl")

def main():
    print("=" * 70)
    print("  PHASE 77 — PARAMETER CAPACITY EXPANSION EXPERIMENT")
    print("=" * 70)
    
    # 1. Load Tokenizer & Gold Dataset
    tokenizer = BPETokenizer()
    tokenizer.load(TOKENIZER_DIR)
    with open(GOLD_SET_PATH, "r", encoding="utf-8") as f:
        gold_records = [json.loads(line) for line in f if line.strip()]
    print(f"Loaded Tokenizer and {len(gold_records)} Gold Evaluation records.")
    
    # 2. Train/Fine-tune all candidates
    train_results = train_all_phase77_models(tokenizer)
    
    # 3. Evaluate all candidates
    model_paths = {k: v["save_path"] for k, v in train_results.items()}
    eval_results = run_phase77_full_evaluation(gold_records, tokenizer, model_paths)
    
    # 4. Compute Scaling Analysis
    scaling_analysis = compute_scaling_metrics(eval_results, train_results)
    
    # 5. Save Results JSON
    results_payload = {
        "train_results": train_results,
        "eval_results": eval_results,
        "scaling_analysis": scaling_analysis
    }
    os.makedirs(os.path.dirname(RESULTS_JSON_PATH), exist_ok=True)
    with open(RESULTS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)
    print(f"Saved results JSON to {RESULTS_JSON_PATH}")
    
    # 6. Generate Markdown Report
    generate_phase77_report(eval_results, train_results, scaling_analysis)
    
    # 7. Update experiments_history.jsonl
    history_entry = {
        "phase": "phase77",
        "action": "PARAMETER_CAPACITY_EXPANSION",
        "scientific_outcome": scaling_analysis["scientific_outcome"],
        "verdict_summary": scaling_analysis["verdict_summary"],
        "control_coherence": scaling_analysis["scaling_table"]["COLLISION-10M"]["coherence"],
        "j25m_coherence": scaling_analysis["scaling_table"]["J77-25M"]["coherence"],
        "j35m_coherence": scaling_analysis["scaling_table"]["J77-35M"]["coherence"],
        "j50m_coherence": scaling_analysis["scaling_table"]["J77-50M"]["coherence"],
        "hypothesis_supported": scaling_analysis["hypothesis_supported"]
    }
    with open(HISTORY_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(history_entry) + "\n")
    print(f"Updated experiments history log at {HISTORY_PATH}")
    
    print("=" * 70)
    print(f"  PHASE 77 COMPLETE — OUTCOME: {scaling_analysis['scientific_outcome']}")
    print("=" * 70)

if __name__ == "__main__":
    main()
