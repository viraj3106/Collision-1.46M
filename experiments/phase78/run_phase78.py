import os
import sys
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.tokenize import BPETokenizer
from experiments.phase78.build_dataset_p78 import build_phase78_dataset
from experiments.phase78.train_phase78 import train_all_phase78_candidates
from experiments.phase78.evaluator import run_phase78_full_evaluation, compute_phase78_scaling_analysis
from experiments.phase78.generate_report import generate_phase78_report

TOKENIZER_DIR = os.path.join(PROJECT_ROOT, "artifacts", "tokenizer")
GOLD_SET_PATH = os.path.join(PROJECT_ROOT, "experiments", "phase75", "data", "gold", "gold_eval_set.jsonl")
RESULTS_JSON_PATH = os.path.join(PROJECT_ROOT, "experiments", "phase78", "results", "phase78_results.json")
HISTORY_PATH = os.path.join(PROJECT_ROOT, "experiments", "experiments_history.jsonl")

def main():
    print("=" * 70)
    print("  PHASE 78 — PRETRAINING TOKEN BUDGET EXPANSION EXPERIMENT")
    print("=" * 70)
    
    # 1. Load Tokenizer & Gold Dataset
    tokenizer = BPETokenizer()
    tokenizer.load(TOKENIZER_DIR)
    with open(GOLD_SET_PATH, "r", encoding="utf-8") as f:
        gold_records = [json.loads(line) for line in f if line.strip()]
    print(f"Loaded Tokenizer and {len(gold_records)} Gold Evaluation records.")
    
    # 2. Build Pretraining Dataset
    audit_data = build_phase78_dataset(target_documents=5000)
    
    # 3. Pretrain & Fine-tune all Phase 78 candidates
    train_results = train_all_phase78_candidates(tokenizer)
    
    # 4. Evaluate all candidates on Gold Set
    eval_results = run_phase78_full_evaluation(gold_records, tokenizer)
    
    # 5. Compute Comparative Scaling Analysis
    scaling_analysis = compute_phase78_scaling_analysis(eval_results, train_results)
    
    # 6. Save Results JSON
    results_payload = {
        "dataset_audit": audit_data,
        "train_results": train_results,
        "eval_results": eval_results,
        "scaling_analysis": scaling_analysis
    }
    os.makedirs(os.path.dirname(RESULTS_JSON_PATH), exist_ok=True)
    with open(RESULTS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)
    print(f"Saved Phase 78 results JSON to {RESULTS_JSON_PATH}")
    
    # 7. Generate Markdown Report
    generate_phase78_report(eval_results, train_results, scaling_analysis)
    
    # 8. Append to experiments_history.jsonl
    history_entry = {
        "phase": "phase78",
        "action": "PRETRAINING_TOKEN_BUDGET_EXPANSION",
        "scientific_outcome": scaling_analysis["scientific_outcome"],
        "verdict_summary": scaling_analysis["verdict_summary"],
        "control_coherence": eval_results.get("COLLISION-10M", {}).get("aggregate_metrics", {}).get("avg_coherence", 2.47),
        "p78a_coherence": scaling_analysis["matrix_table"]["P78-A"]["coherence"],
        "p78b_coherence": scaling_analysis["matrix_table"]["P78-B"]["coherence"],
        "p78c_coherence": scaling_analysis["matrix_table"]["P78-C"]["coherence"],
        "p78d_coherence": scaling_analysis["matrix_table"]["P78-D"]["coherence"],
        "p78e_coherence": scaling_analysis["matrix_table"]["P78-E"]["coherence"],
        "hypothesis_supported": scaling_analysis["hypothesis_supported"]
    }
    with open(HISTORY_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(history_entry) + "\n")
    print(f"Updated experiments history log at {HISTORY_PATH}")
    
    print("=" * 70)
    print(f"  PHASE 78 COMPLETE — OUTCOME: {scaling_analysis['scientific_outcome']}")
    print("=" * 70)

if __name__ == "__main__":
    main()
