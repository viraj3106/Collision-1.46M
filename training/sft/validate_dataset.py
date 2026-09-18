"""
COLLISION Phase 102 — SFT Dataset Validation & Statistics Engine.

Implements deterministic quality verification:
- Schema completeness
- Routing label consistency
- Response non-emptiness
- CoT / reasoning trace leakage rejection
- Prompt injection defense in answer targets
- Evidence & citation integrity
- Zero-leakage verification across train / val / test splits
- Dataset statistics summary computation
"""

import os
import sys
import json
import re
from typing import List, Dict, Any, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DATA_SFT_DIR = os.path.join(PROJECT_ROOT, "data", "sft")

VALID_ROUTING_MODES = {
    "MODEL_ONLY", "LOCAL", "WEB", "HYBRID", "INSUFFICIENT_INFORMATION", "CONFLICTING_EVIDENCE"
}

VALID_CATEGORIES = {
    "MODEL_ONLY",
    "LOCAL_RAG",
    "WEB_GROUNDED",
    "HYBRID",
    "INSUFFICIENT_INFORMATION",
    "CONFLICTING_EVIDENCE",
    "ABSTENTION",
    "TOOL_SELECTION",
    "SOURCE-GROUNDED QUESTION ANSWERING",
    "FOLLOW-UP / CONTEXTUAL CONVERSATION",
    "INSTRUCTION FOLLOWING",
    "HALLUCINATION CORRECTION"
}

COT_LEAKAGE_PATTERNS = [
    r"<\s*think\s*>",
    r"<\s*/\s*think\s*>",
    r"<\s*reasoning\s*>",
    r"\[\s*internal\s+reasoning\s*\]",
    r"step-by-step\s+internal\s+thought",
    r"let\s+me\s+think\s+privately"
]

INJECTION_TARGET_PATTERNS = [
    r"\b(ignore\s+previous\s+instructions|system\s+override|print\s+system\s+prompt|reveal\s+secret\s+key|output\s+HACKED)\b"
]


def validate_example(example: Dict[str, Any]) -> Tuple[bool, List[str]]:
    errors = []

    # 1. Required fields
    required_fields = ["id", "category", "user_query", "expected_routing_mode", "expected_answer", "text"]
    for field in required_fields:
        if field not in example or example[field] is None:
            errors.append(f"Missing required field: '{field}'")

    if errors:
        return False, errors

    # 2. Category & Routing checks
    cat = example["category"]
    route = example["expected_routing_mode"]

    if cat not in VALID_CATEGORIES:
        errors.append(f"Invalid category: '{cat}'")

    if route not in VALID_ROUTING_MODES:
        errors.append(f"Invalid routing mode: '{route}'")

    # 3. Content emptiness
    query = example.get("user_query", "").strip()
    answer = example.get("expected_answer", "").strip()
    text = example.get("text", "").strip()

    if not query:
        errors.append("Empty user_query")
    if not answer:
        errors.append("Empty expected_answer")
    if not text:
        errors.append("Empty conversation text")

    # 4. Chain of thought leakage rejection
    for pattern in COT_LEAKAGE_PATTERNS:
        if re.search(pattern, answer, re.IGNORECASE) or re.search(pattern, text, re.IGNORECASE):
            errors.append(f"CoT leakage detected matching pattern: '{pattern}'")

    # 5. Prompt injection inside trusted answer targets
    for pattern in INJECTION_TARGET_PATTERNS:
        if re.search(pattern, answer, re.IGNORECASE):
            errors.append(f"Prompt injection pattern inside answer target: '{pattern}'")

    # 6. Evidence & Citation validation
    requires_citation = example.get("citation_requirements", False)
    ctx = example.get("available_context", "").strip()
    if requires_citation:
        if not ctx:
            errors.append("Citation required but available_context is empty")
        # Check if citations or bracketed sources exist in the answer
        if not re.search(r"\[.+?\]", answer):
            errors.append("Citation required but no source bracket '[...]' found in answer")

    # 7. Conversational structure
    if "USER:\n" not in text or "\nASSISTANT:\n" not in text:
        errors.append("Malformed conversational structure (must contain 'USER:\\n' and '\\nASSISTANT:\\n')")

    return len(errors) == 0, errors


def audit_splits(sft_dir: str = DATA_SFT_DIR) -> Dict[str, Any]:
    splits = ["train.jsonl", "validation.jsonl", "test.jsonl"]
    split_data = {}
    seen_queries = {}
    total_examples = 0
    rejected_examples = 0
    failure_categories = {}

    for split in splits:
        filepath = os.path.join(sft_dir, split)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Required split file missing: {filepath}")

        items = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                total_examples += 1
                try:
                    item = json.loads(line)
                except Exception as e:
                    rejected_examples += 1
                    failure_categories["json_decode_error"] = failure_categories.get("json_decode_error", 0) + 1
                    continue

                is_valid, errs = validate_example(item)
                if not is_valid:
                    rejected_examples += 1
                    for e in errs:
                        failure_categories[e] = failure_categories.get(e, 0) + 1
                else:
                    items.append(item)
                    q_key = item["user_query"].strip().lower()
                    if q_key in seen_queries:
                        seen_queries[q_key].append((split, item["id"]))
                    else:
                        seen_queries[q_key] = [(split, item["id"])]

        split_data[split] = items

    # Check for duplicate queries and leakage between splits
    cross_split_leakage = []
    duplicate_count = 0
    for q, occurrences in seen_queries.items():
        if len(occurrences) > 1:
            duplicate_count += (len(occurrences) - 1)
            splits_involved = set(x[0] for x in occurrences)
            if len(splits_involved) > 1:
                cross_split_leakage.append({"query": q, "occurrences": occurrences})

    # Aggregate stats
    all_valid_items = [item for split_items in split_data.values() for item in split_items]
    category_counts = {}
    routing_counts = {}
    prompt_lengths = []
    response_lengths = []

    for item in all_valid_items:
        cat = item["category"]
        category_counts[cat] = category_counts.get(cat, 0) + 1
        route = item["expected_routing_mode"]
        routing_counts[route] = routing_counts.get(route, 0) + 1

        prompt_lengths.append(len(item.get("prompt", item["user_query"])))
        response_lengths.append(len(item.get("response", item["expected_answer"])))

    avg_prompt_len = sum(prompt_lengths) / max(1, len(prompt_lengths))
    avg_response_len = sum(response_lengths) / max(1, len(response_lengths))

    report = {
        "total_examples_audited": total_examples,
        "valid_examples": len(all_valid_items),
        "rejected_examples": rejected_examples,
        "duplicate_queries": duplicate_count,
        "cross_split_leakage_count": len(cross_split_leakage),
        "splits": {k: len(v) for k, v in split_data.items()},
        "category_breakdown": category_counts,
        "routing_mode_breakdown": routing_counts,
        "avg_prompt_length_chars": round(avg_prompt_len, 2),
        "avg_response_length_chars": round(avg_response_len, 2),
        "failure_categories": failure_categories,
        "cross_split_leakage_details": cross_split_leakage
    }

    return report


def main():
    print("=" * 60)
    print("RUNNING SFT DATASET VALIDATION & INTEGRITY AUDIT")
    print("=" * 60)

    report = audit_splits()
    print(f"Total Examples Audited:       {report['total_examples_audited']}")
    print(f"Valid Examples:              {report['valid_examples']}")
    print(f"Rejected Examples:           {report['rejected_examples']}")
    print(f"Duplicate Queries:           {report['duplicate_queries']}")
    print(f"Cross-Split Leakages:        {report['cross_split_leakage_count']}")
    print(f"Splits Distribution:         {report['splits']}")
    print(f"Avg Prompt Length:           {report['avg_prompt_length_chars']} chars")
    print(f"Avg Response Length:         {report['avg_response_length_chars']} chars")
    print("-" * 60)
    print("CATEGORY BREAKDOWN:")
    for cat, count in sorted(report["category_breakdown"].items()):
        print(f"  {cat:<35}: {count}")
    print("-" * 60)
    print("ROUTING MODE BREAKDOWN:")
    for r, count in sorted(report["routing_mode_breakdown"].items()):
        print(f"  {r:<30}: {count}")
    print("=" * 60)

    if report["cross_split_leakage_count"] > 0:
        print("ERROR: Cross-split leakage detected!")
        sys.exit(1)

    if report["rejected_examples"] > 0:
        print("ERROR: Rejected examples detected during validation!")
        sys.exit(1)

    print("SUCCESS: All SFT dataset quality and integrity gates passed.")


if __name__ == "__main__":
    main()
