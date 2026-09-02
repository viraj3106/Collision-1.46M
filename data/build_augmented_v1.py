import os
import sys
import json
import random
import hashlib

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.tokenize import BPETokenizer

DATASET_DIR = os.path.join(PROJECT_ROOT, "datasets", "collision_instruct_v1")
AUG_DIR = os.path.join(PROJECT_ROOT, "datasets", "collision_augmented_v1")
EXP_DIR = os.path.join(PROJECT_ROOT, "experiments", "phase31")
TOKENIZER_DIR = os.path.join(PROJECT_ROOT, "artifacts", "tokenizer")

def build_augmented_dataset_v1(seed: int = 42, val_ratio: float = 0.15, test_ratio: float = 0.15):
    os.makedirs(AUG_DIR, exist_ok=True)
    os.makedirs(EXP_DIR, exist_ok=True)

    tokenizer = BPETokenizer()
    tokenizer.load(TOKENIZER_DIR)

    # 1. Load Real-World Dataset v2
    rw_file = os.path.join(PROJECT_ROOT, "data", "real_world", "cleaned", "collision_real_world_v2.jsonl")
    rw_records = []
    if os.path.exists(rw_file):
        with open(rw_file, "r", encoding="utf-8") as f:
            for l in f:
                if l.strip():
                    item = json.loads(l)
                    item["source"] = "real_world"
                    rw_records.append(item)

    # 2. Load Synthetic Dataset v1
    syn_file = os.path.join(DATASET_DIR, "collision_synthetic_v1.jsonl")
    syn_records = []
    if os.path.exists(syn_file):
        with open(syn_file, "r", encoding="utf-8") as f:
            for l in f:
                if l.strip():
                    item = json.loads(l)
                    item["source"] = "synthetic"
                    syn_records.append(item)

    # Calculate token counts
    def count_tokens(recs):
        return sum(len(tokenizer.encode(r.get("instruction", r.get("prompt", "")), bos=True)) + len(tokenizer.encode(r["response"], eos=True)) for r in recs)

    rw_tokens = count_tokens(rw_records)
    syn_tokens = count_tokens(syn_records)

    # 3. Combine with explicit provenance retention
    combined = rw_records + syn_records
    total_tokens = rw_tokens + syn_tokens

    # Deterministic shuffling
    random.seed(seed)
    shuffled = list(combined)
    random.shuffle(shuffled)

    n_val = max(1, int(len(shuffled) * val_ratio))
    n_test = max(1, int(len(shuffled) * test_ratio))
    n_train = len(shuffled) - n_val - n_test

    val_records = shuffled[:n_val]
    test_records = shuffled[n_val : n_val + n_test]
    train_records = shuffled[n_val + n_test :]

    train_file = os.path.join(AUG_DIR, "train.jsonl")
    val_file = os.path.join(AUG_DIR, "val.jsonl")
    test_file = os.path.join(AUG_DIR, "test.jsonl")

    def write_jsonl(filepath, records):
        with open(filepath, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    write_jsonl(train_file, train_records)
    write_jsonl(val_file, val_records)
    write_jsonl(test_file, test_records)

    stats = {
        "dataset_name": "collision_augmented_v1",
        "total_records": len(combined),
        "real_world_records": len(rw_records),
        "synthetic_records": len(syn_records),
        "real_world_token_count": rw_tokens,
        "synthetic_token_count": syn_tokens,
        "total_token_count": total_tokens,
        "real_world_token_percentage": round((rw_tokens / max(1, total_tokens)) * 100, 2),
        "synthetic_token_percentage": round((syn_tokens / max(1, total_tokens)) * 100, 2),
        "splits": {
            "train_records": len(train_records),
            "train_tokens": count_tokens(train_records),
            "val_records": len(val_records),
            "val_tokens": count_tokens(val_records),
            "test_records": len(test_records),
            "test_tokens": count_tokens(test_records)
        },
        "seed": seed
    }

    stats_file = os.path.join(EXP_DIR, "augmented_dataset_statistics.json")
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"Combined Augmented Dataset v1 Built at datasets/collision_augmented_v1/:")
    print(f"  Total Examples: {len(combined):,} (Real: {len(rw_records)}, Synthetic: {len(syn_records)})")
    print(f"  Total Tokens:   {total_tokens:,} (Real: {rw_tokens:,}, Synthetic: {syn_tokens:,})")
    print(f"  Splits -> Train: {len(train_records)}, Val: {len(val_records)}, Test: {len(test_records)}")
    return stats

if __name__ == "__main__":
    build_augmented_dataset_v1()
