import os
import sys
import json
import argparse

# Resolve project root path and insert into Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.clean_real_world import process_data_pipeline

import random

def convert_real_world_to_collision_dataset(
    cleaned_jsonl_path: str = os.path.join("data", "real_world", "cleaned", "real_world_cleaned.jsonl"),
    output_jsonl_path: str = os.path.join("datasets", "collision_instruct_v1", "real_world_formatted.jsonl"),
    seed: int = 42,
    val_ratio: float = 0.2
):
    """
    Converts cleaned feedback records into COLLISION training format (instruction / response dicts)
    and generates deterministic train/validation splits (real_world_train.jsonl & real_world_val.jsonl).
    """
    if not os.path.exists(cleaned_jsonl_path):
        print(f"Cleaned dataset not found at {cleaned_jsonl_path}. Executing data pipeline first...")
        process_data_pipeline()

    if not os.path.exists(cleaned_jsonl_path):
        print(f"Error: {cleaned_jsonl_path} still missing. No data available.")
        return 0

    out_dir = os.path.dirname(output_jsonl_path)
    os.makedirs(out_dir, exist_ok=True)

    formatted_records = []
    with open(cleaned_jsonl_path, "r", encoding="utf-8") as infile:
        for line in infile:
            if not line.strip():
                continue
            rec = json.loads(line)
            training_example = {
                "instruction": rec["prompt"],
                "response": rec["response"],
                "category": rec.get("category", "real_world_feedback"),
                "source": "user_feedback"
            }
            formatted_records.append(training_example)

    # Write formatted output
    with open(output_jsonl_path, "w", encoding="utf-8") as outfile:
        for item in formatted_records:
            outfile.write(json.dumps(item, ensure_ascii=False) + "\n")

    # Deterministic train/val split
    random.seed(seed)
    shuffled = list(formatted_records)
    random.shuffle(shuffled)

    val_size = max(1, int(len(shuffled) * val_ratio)) if len(shuffled) > 1 else 0
    val_records = shuffled[:val_size]
    train_records = shuffled[val_size:]

    train_path = os.path.join(out_dir, "real_world_train.jsonl")
    val_path = os.path.join(out_dir, "real_world_val.jsonl")

    with open(train_path, "w", encoding="utf-8") as f_tr:
        for item in train_records:
            f_tr.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(val_path, "w", encoding="utf-8") as f_val:
        for item in val_records:
            f_val.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Successfully converted {len(formatted_records)} real-world feedback records into COLLISION dataset format at: {output_jsonl_path}")
    print(f"  Train set ({len(train_records)} examples) -> {train_path}")
    print(f"  Validation set ({len(val_records)} examples) -> {val_path}")
    return len(formatted_records)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert approved real-world feedback into COLLISION dataset format.")
    parser.add_argument("--input-file", type=str, default=os.path.join("data", "real_world", "cleaned", "real_world_cleaned.jsonl"))
    parser.add_argument("--output-file", type=str, default=os.path.join("datasets", "collision_instruct_v1", "real_world_formatted.jsonl"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    convert_real_world_to_collision_dataset(args.input_file, args.output_file, seed=args.seed)

