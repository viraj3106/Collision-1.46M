import os
import sys
import json
import re
import math
import statistics
import random
from collections import Counter

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(PROJECT_ROOT, "data", "preferences", "preference_dataset_v3.jsonl")
TRAIN_PATH = os.path.join(PROJECT_ROOT, "data", "preferences", "preference_dataset_v3_train.jsonl")
VAL_PATH = os.path.join(PROJECT_ROOT, "data", "preferences", "preference_dataset_v3_val.jsonl")
EXP_DIR = os.path.join(PROJECT_ROOT, "experiments", "phase41")

os.makedirs(EXP_DIR, exist_ok=True)

PII_PATTERNS = [
    (r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", "email_address"),
    (r"(?i)\b(?:AKIA|ASIA)[0-9A-Z]{16}\b", "aws_access_key"),
    (r"(?i)\b[0-9]{3}-[0-9]{2}-[0-9]{4}\b", "ssn"),
    (r"(?i)bearer\s+[a-zA-Z0-9_\-\.]{20,}", "bearer_token"),
    (r"(?i)password\s*=\s*['\"][^'\"]+['\"]", "hardcoded_password")
]

def audit_and_split():
    print(f"Auditing preference dataset v3 at {DATASET_PATH}...", flush=True)

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset file missing: {DATASET_PATH}")

    records = []
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line.strip()))

    total_pairs = len(records)
    print(f"Total Pairs Loaded: {total_pairs}", flush=True)

    prompts = [r["prompt"] for r in records]
    chosen = [r["chosen"] for r in records]
    rejected = [r["rejected"] for r in records]
    categories = [r["category"] for r in records]
    difficulties = [r["difficulty"] for r in records]

    unique_prompts = set(prompts)
    unique_chosen = set(chosen)
    unique_rejected = set(rejected)

    prompt_dup_count = total_pairs - len(unique_prompts)
    chosen_dup_count = total_pairs - len(unique_chosen)
    rejected_dup_count = total_pairs - len(unique_rejected)

    unique_prompt_ratio = (len(unique_prompts) / total_pairs) * 100.0
    exact_duplicate_rate = (prompt_dup_count / total_pairs) * 100.0

    cat_counts = Counter(categories)
    max_cat_pct = max(count / total_pairs * 100.0 for count in cat_counts.values())

    diff_counts = Counter(difficulties)

    prompt_word_lens = [len(p.split()) for p in prompts]
    chosen_word_lens = [len(c.split()) for c in chosen]
    rejected_word_lens = [len(r.split()) for r in rejected]

    pii_violations = []
    for idx, r in enumerate(records):
        text_full = f"{r['prompt']} {r['chosen']} {r['rejected']}"
        for pat, pat_name in PII_PATTERNS:
            if re.search(pat, text_full):
                pii_violations.append((idx, r["id"], pat_name))

    zero_pii_leaks = (len(pii_violations) == 0)

    # Check Quality Gates:
    # 1. unique_prompt_ratio >= 95%
    # 2. exact_duplicate_rate <= 1%
    # 3. max_cat_pct <= 20%
    # 4. zero_pii_leaks == True
    gate_prompt_ratio_ok = (unique_prompt_ratio >= 95.0)
    gate_dup_rate_ok = (exact_duplicate_rate <= 1.0)
    gate_cat_dist_ok = (max_cat_pct <= 20.0)
    gate_pii_ok = zero_pii_leaks

    all_gates_pass = gate_prompt_ratio_ok and gate_dup_rate_ok and gate_cat_dist_ok and gate_pii_ok

    readiness_status = "PHASE_41_DATASET_READY" if all_gates_pass else "PHASE_41_DATASET_NOT_READY"

    audit_result = {
        "dataset_name": "preference_dataset_v3.jsonl",
        "total_pairs": total_pairs,
        "unique_prompts": len(unique_prompts),
        "unique_chosen": len(unique_chosen),
        "unique_rejected": len(unique_rejected),
        "unique_prompt_ratio_pct": round(unique_prompt_ratio, 2),
        "exact_duplicate_rate_pct": round(exact_duplicate_rate, 2),
        "category_counts": dict(cat_counts),
        "max_category_pct": round(max_cat_pct, 2),
        "difficulty_counts": dict(diff_counts),
        "length_stats": {
            "prompt_words_mean": round(statistics.mean(prompt_word_lens), 2),
            "chosen_words_mean": round(statistics.mean(chosen_word_lens), 2),
            "rejected_words_mean": round(statistics.mean(rejected_word_lens), 2)
        },
        "pii_violations": pii_violations,
        "zero_pii_leaks": zero_pii_leaks,
        "quality_gates": {
            "unique_prompt_ratio_ge_95": gate_prompt_ratio_ok,
            "exact_duplicate_rate_le_1": gate_dup_rate_ok,
            "no_category_gt_20": gate_cat_dist_ok,
            "zero_pii_leaks": gate_pii_ok,
            "all_gates_pass": all_gates_pass
        },
        "readiness_status": readiness_status
    }

    out_audit = os.path.join(EXP_DIR, "dataset_audit_v3.json")
    with open(out_audit, "w", encoding="utf-8") as f:
        json.dump(audit_result, f, indent=2)

    print(f"Audit results saved to {out_audit}")
    print(f"  Unique Prompt Ratio: {unique_prompt_ratio:.2f}% (Target: >=95%)")
    print(f"  Duplicate Rate: {exact_duplicate_rate:.2f}% (Target: <=1%)")
    print(f"  Max Category Pct: {max_cat_pct:.2f}% (Target: <=20%)")
    print(f"  Zero PII Leaks: {zero_pii_leaks}")
    print(f"  Readiness Status: {readiness_status}")

    # Deterministic 90/10 Train/Val Split by record index seed
    print("\nCreating deterministic 90% train / 10% validation split...", flush=True)
    random.seed(42)

    cat_groups = {}
    for r in records:
        cat = r["category"]
        if cat not in cat_groups:
            cat_groups[cat] = []
        cat_groups[cat].append(r)

    train_records = []
    val_records = []

    for cat, items in cat_groups.items():
        random.shuffle(items)
        val_count = max(1, int(len(items) * 0.10))
        val_records.extend(items[:val_count])
        train_records.extend(items[val_count:])

    with open(TRAIN_PATH, "w", encoding="utf-8") as f:
        for r in train_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(VAL_PATH, "w", encoding="utf-8") as f:
        for r in val_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Saved Train Set ({len(train_records)} pairs) to: {TRAIN_PATH}")
    print(f"Saved Val Set ({len(val_records)} pairs) to: {VAL_PATH}")

    return audit_result, train_records, val_records

if __name__ == "__main__":
    audit_and_split()
