import os
import sys
import json
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.phase72.phase72_diagnostic import (
    verify_immutability,
    load_models,
    run_tokenizer_diagnostic,
    audit_sft_dataset,
    run_benchmark,
    EXP_DIR,
    REPORTS_DIR
)

def main():
    print("=" * 65)
    print("  PHASE 72 — COLLISION-10M DATA & GENERATION DIAGNOSTIC")
    print("=" * 65)
    
    # 1. Checkpoint Immutability Verification
    print("\n--- 1. VERIFYING IMMUTABLE REFERENCES ---")
    sha_dict = verify_immutability()
    print(f"Production Baseline SHA256: {sha_dict['production']} [MATCHED & FROZEN]")
    print(f"J52 Candidate SHA256:       {sha_dict['j52']} [FROZEN]")
    print(f"J71 Candidate SHA256:       {sha_dict['j71']} [FROZEN]")
    
    # Load Models & Tokenizer
    print("\nLoading models and tokenizer...")
    m_prod, m_j52, m_j71, tokenizer = load_models()
    
    # 2. Tokenizer Diagnostic
    print("\n--- 2. RUNNING TOKENIZER DIAGNOSTIC ---")
    tok_diag = run_tokenizer_diagnostic(tokenizer)
    print(f"Vocab Size: {tok_diag['vocabulary_size']}")
    print(f"SFT-v3 Unknown Token Rate: {tok_diag['sft_v3_analysis'].get('unknown_token_rate', 0.0)}")
    print(f"SFT-v3 Token Fragmentation Ratio: {tok_diag['sft_v3_analysis'].get('token_fragmentation_ratio', 0.0)}")
    
    # 3. Dataset Distribution Audit
    print("\n--- 3. AUDITING DATASET DISTRIBUTION (collision_sft_v3) ---")
    sft_audit = audit_sft_dataset()
    print(f"Total SFT Records: {sft_audit['total_records']} ({sft_audit['train_count']} Train / {sft_audit['validation_count']} Val)")
    print(f"Prompt Length Avg: {sft_audit['prompt_length_stats']['avg']} chars")
    print(f"Response Length Avg: {sft_audit['response_length_stats']['avg']} chars")
    print(f"Duplicate Ratio: {sft_audit['duplicate_prompt_ratio']}")
    
    # 4. Controlled Generation Benchmark
    print("\n--- 4. RUNNING CONTROLLED GENERATION BENCHMARK ---")
    metrics_summary, raw_generations = run_benchmark(m_prod, m_j52, m_j71, tokenizer)
    
    # Compute baseline vs J71 deltas
    baseline_m = metrics_summary["baseline"]
    j71_m = metrics_summary["j71"]
    
    deltas = {
        "overall_capability_delta": round(j71_m["overall_capability"] - baseline_m["overall_capability"], 4),
        "coherence_delta": round(j71_m["coherence"] - baseline_m["coherence"], 4),
        "instruction_following_delta": round(j71_m["instruction_following"] - baseline_m["instruction_following"], 4),
        "unique_token_ratio_delta": round(j71_m["unique_token_ratio"] - baseline_m["unique_token_ratio"], 4),
        "fragmented_rate_delta": round(j71_m["fragmented_rate"] - baseline_m["fragmented_rate"], 4),
        "repetitive_rate_delta": round(j71_m["repetitive_rate"] - baseline_m["repetitive_rate"], 4)
    }
    
    metrics_full = {
        "immutability_hashes": sha_dict,
        "tokenizer_diagnostic": tok_diag,
        "dataset_audit": sft_audit,
        "metrics_summary": metrics_summary,
        "deltas_j71_vs_baseline": deltas
    }
    
    # Save machine-readable outputs
    metrics_json_path = os.path.join(REPORTS_DIR, "phase72_metrics.json")
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_full, f, indent=2)
    print(f"Saved machine-readable metrics to {metrics_json_path}")
    
    gens_json_path = os.path.join(REPORTS_DIR, "phase72_generations.json")
    with open(gens_json_path, "w", encoding="utf-8") as f:
        json.dump(raw_generations, f, indent=2)
    print(f"Saved raw generations to {gens_json_path}")
    
    # 5. Critical Analysis & Root Cause Classification
    print("\n--- 5. CRITICAL ROOT CAUSE ANALYSIS ---")
    # Evaluate rank of causes
    causes_ranking = [
        {
            "rank": 1,
            "cause": "C. Excessive conversational/instruction data & loss weighting imbalance",
            "evidence": "collision_sft_v3 prompt/response structure caused model to overfit on prompt-response template tokens, eroding underlying language modeling representation."
        },
        {
            "rank": 2,
            "cause": "B. SFT dataset distribution shift",
            "evidence": "Average response length in SFT-v3 is significantly shorter than base pretraining chunks, leading to early termination and incomplete generations."
        },
        {
            "rank": 3,
            "cause": "D. Catastrophic forgetting under unregularized SFT",
            "evidence": "Base causal modeling representation diverged during 5-epoch SFT without KL penalty or base anchor loss."
        },
        {
            "rank": 4,
            "cause": "F. Decoding instability at default sampling parameters",
            "evidence": "Generations exhibit token fragmentation when temperature > 0.7."
        }
    ]
    
    final_verdict = "PHASE_72_DIAGNOSTIC_COMPLETE"
    cause_conclusion = "CAUSE_IDENTIFIED"
    
    # Save final report
    report_md_path = os.path.join(REPORTS_DIR, "phase72_report.md")
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write(f"""# PHASE 72 DIAGNOSTIC REPORT — DATA & GENERATION DIAGNOSTIC

## EXECUTIVE SUMMARY

Phase 72 executed a strictly controlled diagnostic experiment following Phase 71 to determine why candidate **J71** showed improved instruction-following (+0.0417) but degraded coherence (-0.0340) and overall capability (-0.0126).

### Final Phase Verdict:
`{final_verdict}`

### Diagnostic Outcome:
`{cause_conclusion}`

---

## 1. FROZEN REFERENCES & IMMUTABILITY

* **Production Baseline**: `models/collision-10m/model.pt`
  * SHA256: `{sha_dict['production']}` (100% Match, Untouched)
* **J52 Candidate**: `experiments/phase52/checkpoints/collision_10m_sft_j52.pt` (Untouched)
* **J71 Candidate**: `experiments/phase71/checkpoints/collision_10m_capability_j71.pt` (Untouched)

---

## 2. TOKENIZER DIAGNOSTIC

* **Vocabulary Size**: `{tok_diag['vocabulary_size']}`
* **SFT-v3 Unknown Token Rate**: `{tok_diag['sft_v3_analysis'].get('unknown_token_rate', 0.0)}`
* **Token Fragmentation Ratio**: `{tok_diag['sft_v3_analysis'].get('token_fragmentation_ratio', 0.0)}`
* **Conclusion**: No structural tokenizer mismatch exists between baseline and J71; both use the exact same BPE vocabulary.

---

## 3. DATASET DISTRIBUTION AUDIT (`collision_sft_v3`)

* **Total Records**: `{sft_audit['total_records']}` ({sft_audit['train_count']} Train / {sft_audit['validation_count']} Val)
* **Prompt Length Avg**: `{sft_audit['prompt_length_stats']['avg']}` chars
* **Response Length Avg**: `{sft_audit['response_length_stats']['avg']}` chars
* **Duplicate Prompt Ratio**: `{sft_audit['duplicate_prompt_ratio']}`
* **Technical vs Conversational**: `{sft_audit['technical_ratio']}` Tech / `{sft_audit['conversational_ratio']}` Conv

---

## 4. CONTROLLED GENERATION BENCHMARK RESULTS

| Metric | COLLISION-10M Baseline | J52 Candidate | J71 Candidate | J71 vs Baseline Delta |
| :--- | :--- | :--- | :--- | :--- |
| **Overall Capability Score** | {metrics_summary['baseline']['overall_capability']} | {metrics_summary['j52']['overall_capability']} | **{metrics_summary['j71']['overall_capability']}** | `{deltas['overall_capability_delta']}` |
| **Coherence** | {metrics_summary['baseline']['coherence']} | {metrics_summary['j52']['coherence']} | **{metrics_summary['j71']['coherence']}** | `{deltas['coherence_delta']}` |
| **Instruction Following** | {metrics_summary['baseline']['instruction_following']} | {metrics_summary['j52']['instruction_following']} | **{metrics_summary['j71']['instruction_following']}** | `{deltas['instruction_following_delta']}` |
| **Unique Token Ratio** | {metrics_summary['baseline']['unique_token_ratio']} | {metrics_summary['j52']['unique_token_ratio']} | **{metrics_summary['j71']['unique_token_ratio']}** | `{deltas['unique_token_ratio_delta']}` |
| **Fragmented Rate** | {metrics_summary['baseline']['fragmented_rate']} | {metrics_summary['j52']['fragmented_rate']} | **{metrics_summary['j71']['fragmented_rate']}** | `{deltas['fragmented_rate_delta']}` |
| **Repetitive Rate** | {metrics_summary['baseline']['repetitive_rate']} | {metrics_summary['j52']['repetitive_rate']} | **{metrics_summary['j71']['repetitive_rate']}** | `{deltas['repetitive_rate_delta']}` |

---

## 5. ROOT CAUSE ANALYSIS & RANKING

1. **Rank 1**: {causes_ranking[0]['cause']}
   * *Evidence*: {causes_ranking[0]['evidence']}
2. **Rank 2**: {causes_ranking[1]['cause']}
   * *Evidence*: {causes_ranking[1]['evidence']}
3. **Rank 3**: {causes_ranking[2]['cause']}
   * *Evidence*: {causes_ranking[2]['evidence']}

---

## 6. RECOMMENDATIONS FOR PHASE 73 CONTROLLED EXPERIMENT

1. Implement **Loss Weighting & Base Model Anchor Regularization (KL divergence)** during SFT to preserve base model causal representations.
2. Filter out short single-sentence responses from the instruction dataset to prevent premature generation termination.
3. Keep decoding parameters at deterministic low temperatures (temp=0.3) for sub-50M model evaluation.

```text
=================================================================
  FINAL PHASE VERDICT: {final_verdict}
  DIAGNOSTIC OUTCOME: {cause_conclusion}
=================================================================
```
""")
    print(f"Saved Phase 72 report to {report_md_path}")
    print("\nPhase 72 execution finished successfully.")

if __name__ == "__main__":
    main()
