import os
import sys
import json
import difflib
from collections import Counter

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

EXP_DIR = os.path.join(PROJECT_ROOT, "experiments", "phase33")
V1_PATH = os.path.join(PROJECT_ROOT, "datasets", "collision_instruct_v1", "collision_synthetic_v1.jsonl")
V2_PATH = os.path.join(PROJECT_ROOT, "data", "collision_synthetic_v2.jsonl")

def analyze_dataset(file_path):
    if not os.path.exists(file_path):
        return None

    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        for l in f:
            if l.strip():
                records.append(json.loads(l))

    total = len(records)
    if total == 0:
        return None

    instructions = [r.get("instruction", r.get("prompt", "")) for r in records]
    responses = [r.get("response", "") for r in records]

    # 1. Exact & Unique Counts
    unique_inst = len(set(instructions))
    unique_resp = len(set(responses))
    unique_resp_ratio = unique_resp / total

    # 2. Prefix Diversity (3, 5, 10 words)
    prefix_3 = len(set(" ".join(r.split()[:3]).lower() for r in responses if len(r.split()) >= 3))
    prefix_5 = len(set(" ".join(r.split()[:5]).lower() for r in responses if len(r.split()) >= 5))
    prefix_10 = len(set(" ".join(r.split()[:10]).lower() for r in responses if len(r.split()) >= 10))

    # 3. Word Token & N-Gram Diversity
    all_words = " ".join(responses).lower().split()
    unique_words = set(all_words)
    ttr = len(unique_words) / max(1, len(all_words))

    bigrams = [f"{all_words[i]} {all_words[i+1]}" for i in range(len(all_words)-1)]
    trigrams = [f"{all_words[i]} {all_words[i+1]} {all_words[i+2]}" for i in range(len(all_words)-2)]
    fourgrams = [f"{all_words[i]} {all_words[i+1]} {all_words[i+2]} {all_words[i+3]}" for i in range(len(all_words)-3)]

    unique_bigrams = len(set(bigrams))
    unique_trigrams = len(set(trigrams))

    rep_trigram_rate = 1.0 - (len(set(trigrams)) / max(1, len(trigrams)))
    rep_4gram_rate = 1.0 - (len(set(fourgrams)) / max(1, len(fourgrams)))

    # 4. Length Distribution
    word_lengths = [len(r.split()) for r in responses]
    word_lengths.sort()
    avg_len = sum(word_lengths) / total
    median_len = word_lengths[total // 2]
    min_len = word_lengths[0]
    max_len = word_lengths[-1]

    # 5. Duplicates and Near Duplicates
    exact_duplicates = total - unique_resp
    near_duplicates = 0
    # Sample check for near duplicates if size reasonable
    sample_resps = responses[:150]
    for i in range(len(sample_resps)):
        for j in range(i+1, len(sample_resps)):
            if sample_resps[i] != sample_resps[j]:
                sim = difflib.SequenceMatcher(None, sample_resps[i].lower(), sample_resps[j].lower()).ratio()
                if sim > 0.88:
                    near_duplicates += 1

    # 6. Categories & Domains
    cat_counts = dict(Counter(r.get("category", "unknown") for r in records))
    dom_counts = dict(Counter(r.get("domain", "unknown") for r in records))

    return {
        "file_path": file_path,
        "total_records": total,
        "unique_instructions": unique_inst,
        "unique_responses": unique_resp,
        "unique_response_ratio": round(unique_resp_ratio, 4),
        "prefix_diversity_3_words": prefix_3,
        "prefix_diversity_5_words": prefix_5,
        "prefix_diversity_10_words": prefix_10,
        "type_token_ratio": round(ttr, 4),
        "total_words": len(all_words),
        "unique_words": len(unique_words),
        "unique_bigrams": unique_bigrams,
        "unique_trigrams": unique_trigrams,
        "repeated_trigram_rate": round(rep_trigram_rate, 4),
        "repeated_4gram_rate": round(rep_4gram_rate, 4),
        "avg_response_length_words": round(avg_len, 2),
        "median_response_length_words": median_len,
        "min_response_length_words": min_len,
        "max_response_length_words": max_len,
        "exact_duplicates": exact_duplicates,
        "near_duplicates_sample": near_duplicates,
        "categories": cat_counts,
        "domains": dom_counts
    }

def main():
    print("================================================================")
    print("  PHASE 33: DATASET DIVERSITY & QUALITY AUDIT (V1 vs V2)       ")
    print("================================================================")

    audit_v1 = analyze_dataset(V1_PATH)
    audit_v2 = analyze_dataset(V2_PATH)

    if audit_v1 is None or audit_v2 is None:
        raise FileNotFoundError("Error loading synthetic V1 or V2 datasets.")

    comparison_summary = {
        "dataset_v1": audit_v1,
        "dataset_v2": audit_v2,
        "improvements": {
            "record_increase": audit_v2["total_records"] - audit_v1["total_records"],
            "unique_response_increase": audit_v2["unique_responses"] - audit_v1["unique_responses"],
            "response_ratio_change": round(audit_v2["unique_response_ratio"] - audit_v1["unique_response_ratio"], 4),
            "prefix_3_increase": audit_v2["prefix_diversity_3_words"] - audit_v1["prefix_diversity_3_words"],
            "unique_vocabulary_increase": audit_v2["unique_words"] - audit_v1["unique_words"],
            "unique_trigrams_increase": audit_v2["unique_trigrams"] - audit_v1["unique_trigrams"]
        }
    }

    # Save diversity audit
    div_out = os.path.join(EXP_DIR, "audit_dataset_diversity.json")
    with open(div_out, "w", encoding="utf-8") as f:
        json.dump(comparison_summary, f, indent=2)

    # Save data quality report
    q_report = {
        "dataset": "collision_synthetic_v2.jsonl",
        "total_records": audit_v2["total_records"],
        "unique_response_count": audit_v2["unique_responses"],
        "unique_response_ratio": audit_v2["unique_response_ratio"],
        "target_passed": audit_v2["unique_responses"] >= 400 and audit_v2["unique_response_ratio"] >= 0.75,
        "exact_duplicates": audit_v2["exact_duplicates"],
        "domain_count": len(audit_v2["domains"]),
        "category_count": len(audit_v2["categories"])
    }

    q_out = os.path.join(EXP_DIR, "data_quality_report.json")
    with open(q_out, "w", encoding="utf-8") as f:
        json.dump(q_report, f, indent=2)

    print(f"Dataset V1 -> Records: {audit_v1['total_records']}, Unique Responses: {audit_v1['unique_responses']}, Ratio: {audit_v1['unique_response_ratio']}, Vocab: {audit_v1['unique_words']}")
    print(f"Dataset V2 -> Records: {audit_v2['total_records']}, Unique Responses: {audit_v2['unique_responses']}, Ratio: {audit_v2['unique_response_ratio']}, Vocab: {audit_v2['unique_words']}")
    print(f"Audit Target Passed: {q_report['target_passed']}")
    print(f"Saved diversity audit to: {div_out}")
    print(f"Saved data quality report to: {q_out}\n")

if __name__ == "__main__":
    main()
