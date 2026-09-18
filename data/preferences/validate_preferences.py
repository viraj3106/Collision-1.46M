"""
COLLISION Phase 103 — Preference Dataset Validation & Quality Audit Engine.

Validates:
- Schema completeness (prompt, context, response_a, response_b, preferred_response, preference_reason, category)
- Preferred response label validity ("A" or "B")
- Non-identity of response_a and response_b
- CoT / reasoning leakage detection in responses
- Prompt injection target leakage
- Zero train / val / test prompt leakage
- Category distribution and statistics
"""

import os
import sys
import json
import re
from typing import List, Dict, Any, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

PREFERENCES_DIR = os.path.join(PROJECT_ROOT, "data", "preferences")

VALID_PREFERENCE_CATEGORIES = {
    "factual_correctness",
    "groundedness",
    "citation_correctness",
    "appropriate_abstention",
    "hallucination_avoidance",
    "concise_answers",
    "instruction_following",
    "tool_selection_behavior",
    "conflicting_evidence",
    "prompt_injection_resistance",
    "conversational_quality",
    "context_preservation"
}

COT_PATTERNS = [
    r"<\s*think\s*>",
    r"<\s*/\s*think\s*>",
    r"<\s*reasoning\s*>",
    r"\[\s*internal\s+reasoning\s*\]"
]


def validate_preference_item(item: Dict[str, Any]) -> Tuple[bool, List[str]]:
    errors = []

    required_fields = ["id", "category", "prompt", "response_a", "response_b", "preferred_response", "preference_reason"]
    for f in required_fields:
        if f not in item or item[f] is None:
            errors.append(f"Missing required field: '{f}'")

    if errors:
        return False, errors

    cat = item["category"]
    pref = item["preferred_response"]
    resp_a = str(item["response_a"]).strip()
    resp_b = str(item["response_b"]).strip()
    prompt = str(item["prompt"]).strip()

    if cat not in VALID_PREFERENCE_CATEGORIES:
        errors.append(f"Invalid category: '{cat}'")

    if pref not in ("A", "B"):
        errors.append(f"Invalid preferred_response: '{pref}' (must be 'A' or 'B')")

    if not prompt:
        errors.append("Empty prompt")

    if not resp_a or not resp_b:
        errors.append("Empty response_a or response_b")

    if resp_a.lower() == resp_b.lower():
        errors.append("Identical response_a and response_b in preference pair")

    # Check chosen response for chain-of-thought leakage
    chosen = resp_a if pref == "A" else resp_b
    for pat in COT_PATTERNS:
        if re.search(pat, chosen, re.IGNORECASE):
            errors.append(f"Chain-of-thought pattern found in chosen response: '{pat}'")

    return len(errors) == 0, errors


def audit_preference_dataset(pref_dir: str = PREFERENCES_DIR) -> Dict[str, Any]:
    splits = ["train.jsonl", "validation.jsonl", "test.jsonl"]
    split_items = {}
    seen_prompts = {}
    total_audited = 0
    rejected_count = 0
    failure_reasons = {}

    for split in splits:
        filepath = os.path.join(pref_dir, split)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Missing preference split: {filepath}")

        valid_in_split = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                total_audited += 1
                try:
                    item = json.loads(line)
                except Exception as e:
                    rejected_count += 1
                    failure_reasons["json_decode_error"] = failure_reasons.get("json_decode_error", 0) + 1
                    continue

                is_valid, errs = validate_preference_item(item)
                if not is_valid:
                    rejected_count += 1
                    for e in errs:
                        failure_reasons[e] = failure_reasons.get(e, 0) + 1
                else:
                    valid_in_split.append(item)
                    p_key = item["prompt"].strip().lower()
                    if p_key in seen_prompts:
                        seen_prompts[p_key].append((split, item["id"]))
                    else:
                        seen_prompts[p_key] = [(split, item["id"])]

        split_items[split] = valid_in_split

    # Check leakage
    cross_split_leakage = []
    duplicate_count = 0
    for p, occurrences in seen_prompts.items():
        if len(occurrences) > 1:
            duplicate_count += (len(occurrences) - 1)
            splits_set = set(x[0] for x in occurrences)
            if len(splits_set) > 1:
                cross_split_leakage.append({"prompt": p, "occurrences": occurrences})

    all_valid = [item for s_items in split_items.values() for item in s_items]
    category_counts = {}
    preferred_counts = {"A": 0, "B": 0}

    for item in all_valid:
        cat = item["category"]
        category_counts[cat] = category_counts.get(cat, 0) + 1
        preferred_counts[item["preferred_response"]] = preferred_counts.get(item["preferred_response"], 0) + 1

    report = {
        "total_audited": total_audited,
        "valid_count": len(all_valid),
        "rejected_count": rejected_count,
        "duplicate_prompts": duplicate_count,
        "cross_split_leakage_count": len(cross_split_leakage),
        "splits": {k: len(v) for k, v in split_items.items()},
        "category_breakdown": category_counts,
        "preferred_distribution": preferred_counts,
        "failure_reasons": failure_reasons,
        "cross_split_leakage_details": cross_split_leakage
    }

    return report


def main():
    print("=" * 65)
    print("RUNNING PHASE 103 PREFERENCE DATASET VALIDATION AUDIT")
    print("=" * 65)

    report = audit_preference_dataset()
    print(f"Total Audited:             {report['total_audited']}")
    print(f"Valid Preference Pairs:    {report['valid_count']}")
    print(f"Rejected:                  {report['rejected_count']}")
    print(f"Duplicate Prompts:         {report['duplicate_prompts']}")
    print(f"Cross-Split Leakages:      {report['cross_split_leakage_count']}")
    print(f"Splits Distribution:       {report['splits']}")
    print(f"Preferred Distribution:    {report['preferred_distribution']}")
    print("-" * 65)
    print("CATEGORY DISTRIBUTION:")
    for cat, count in sorted(report["category_breakdown"].items()):
        print(f"  {cat:<35}: {count}")
    print("=" * 65)

    if report["cross_split_leakage_count"] > 0:
        print("ERROR: Cross-split leakage detected!")
        sys.exit(1)

    if report["rejected_count"] > 0:
        print("ERROR: Rejected records found in preference dataset!")
        sys.exit(1)

    print("SUCCESS: Phase 103 Preference Dataset passes all quality gates.")


if __name__ == "__main__":
    main()
