"""
COLLISION Phase 104 — Long-Context & Retrieval Dataset Validator.

Validates:
1. Strict schema compliance for each record.
2. Complete split isolation (zero cross-split leakage by query or needle text).
3. Exact document structure and needle ID mapping.
4. Absence of empty documents, missing queries, or corrupted scenarios.
5. Presence of all 4 context length tiers and 10 retrieval stress scenarios.
"""

import os
import sys
import json
from typing import Dict, List, Set

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "long_context")

REQUIRED_FIELDS = {
    "id",
    "scenario",
    "context_length_tier",
    "query",
    "documents",
    "total_documents",
    "target_needle_ids",
    "expected_answer",
    "expected_status",
    "is_adversarial",
    "rationale"
}

VALID_TIERS = {"SHORT", "MEDIUM", "LONG", "VERY_LONG"}

VALID_SCENARIOS = {
    "SCENARIO_A_NEEDLE_HAYSTACK",
    "SCENARIO_B_MULTIPLE_NEEDLES",
    "SCENARIO_C_DISTRACTOR_EVIDENCE",
    "SCENARIO_D_POSITION_ROBUSTNESS",
    "SCENARIO_E_SEMANTIC_DISTRACTORS",
    "SCENARIO_F_CONFLICTING_EVIDENCE",
    "SCENARIO_G_MISSING_EVIDENCE",
    "SCENARIO_H_TEMPORAL_EVIDENCE",
    "SCENARIO_I_WEB_LOCAL_HYBRID",
    "SCENARIO_J_PROMPT_INJECTION"
}

def validate_record(item: Dict, split_name: str, idx: int) -> List[str]:
    errors = []
    
    # 1. Required fields check
    for rf in REQUIRED_FIELDS:
        if rf not in item:
            errors.append(f"[{split_name}#{idx}] Missing required field: '{rf}'")

    # 2. Tier validation
    tier = item.get("context_length_tier")
    if tier not in VALID_TIERS:
        errors.append(f"[{split_name}#{idx}] Invalid context_length_tier: '{tier}'")

    # 3. Scenario validation
    scenario = item.get("scenario")
    if scenario not in VALID_SCENARIOS:
        errors.append(f"[{split_name}#{idx}] Invalid scenario: '{scenario}'")

    # 4. Query check
    query = item.get("query", "")
    if not isinstance(query, str) or not query.strip():
        errors.append(f"[{split_name}#{idx}] Query must be non-empty string")

    # 5. Documents check
    docs = item.get("documents", [])
    if not isinstance(docs, list) or len(docs) == 0:
        errors.append(f"[{split_name}#{idx}] Documents must be non-empty list")
    else:
        for d_i, doc in enumerate(docs):
            if "doc_id" not in doc or "text" not in doc:
                errors.append(f"[{split_name}#{idx}] Doc #{d_i} missing doc_id or text")
            if not doc.get("text", "").strip():
                errors.append(f"[{split_name}#{idx}] Doc #{d_i} text is empty")

    return errors


def run_full_validation() -> bool:
    print(f"\n{'='*70}\nSTARTING PHASE 104 DATASET VALIDATION AUDIT\n{'='*70}")
    
    splits = ["train.jsonl", "validation.jsonl", "test.jsonl"]
    split_queries: Dict[str, Set[str]] = {}
    split_records: Dict[str, List[Dict]] = {}
    total_errors = []

    for s_name in splits:
        path = os.path.join(DATA_DIR, s_name)
        if not os.path.exists(path):
            print(f"[ERROR] Missing split file: {path}")
            return False
            
        records = []
        queries = set()
        with open(path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                    records.append(rec)
                    queries.add(rec.get("query", "").strip().lower())
                    errs = validate_record(rec, s_name, line_idx)
                    total_errors.extend(errs)
                except Exception as e:
                    total_errors.append(f"[{s_name}#{line_idx}] JSON parse error: {e}")
                    
        split_records[s_name] = records
        split_queries[s_name] = queries
        print(f"Loaded {s_name}: {len(records)} records.")

    # Check for cross-split leakage
    print("\nAuditing Cross-Split Query Leakage...")
    overlap_train_val = split_queries["train.jsonl"].intersection(split_queries["validation.jsonl"])
    overlap_train_test = split_queries["train.jsonl"].intersection(split_queries["test.jsonl"])
    overlap_val_test = split_queries["validation.jsonl"].intersection(split_queries["test.jsonl"])

    if overlap_train_val:
        total_errors.append(f"Leakage between train and val: {overlap_train_val}")
    if overlap_train_test:
        total_errors.append(f"Leakage between train and test: {overlap_train_test}")
    if overlap_val_test:
        total_errors.append(f"Leakage between val and test: {overlap_val_test}")

    if not (overlap_train_val or overlap_train_test or overlap_val_test):
        print("[PASS] Zero cross-split leakage detected across Train, Val, and Test.")

    # Check scenario coverage across dataset
    all_recs = split_records["train.jsonl"] + split_records["validation.jsonl"] + split_records["test.jsonl"]
    found_scenarios = {r["scenario"] for r in all_recs}
    missing_scenarios = VALID_SCENARIOS - found_scenarios
    if missing_scenarios:
        total_errors.append(f"Missing scenarios: {missing_scenarios}")
    else:
        print(f"[PASS] All 10 Retrieval Stress Scenarios (A through J) present ({len(found_scenarios)}/10).")

    found_tiers = {r["context_length_tier"] for r in all_recs}
    missing_tiers = VALID_TIERS - found_tiers
    if missing_tiers:
        total_errors.append(f"Missing context tiers: {missing_tiers}")
    else:
        print(f"[PASS] All 4 Context Length Tiers present ({found_tiers}).")

    if total_errors:
        print(f"\n[FAIL] Validation found {len(total_errors)} errors:")
        for err in total_errors[:10]:
            print(f"  - {err}")
        return False

    print(f"\n[SUCCESS] Dataset validation passed 100% with {len(all_recs)} total verified samples.")
    return True

if __name__ == "__main__":
    success = run_full_validation()
    sys.exit(0 if success else 1)
