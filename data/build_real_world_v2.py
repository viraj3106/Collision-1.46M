import os
import sys
import json
import sqlite3

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.clean_real_world import process_data_pipeline, validate_and_clean_records
from data.tokenize import BPETokenizer

RAW_DIR = os.path.join(PROJECT_ROOT, "data", "real_world", "raw")
CLEANED_DIR = os.path.join(PROJECT_ROOT, "data", "real_world", "cleaned")
OUT_DIR = os.path.join(PROJECT_ROOT, "datasets", "collision_instruct_v1")
TOKENIZER_DIR = os.path.join(PROJECT_ROOT, "artifacts", "tokenizer")

def build_real_world_v2():
    os.makedirs(CLEANED_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    
    tokenizer = BPETokenizer()
    tokenizer.load(TOKENIZER_DIR)

    # 1. Gather all raw feedback records
    raw_records = []
    
    # Read DB feedback table
    db_path = os.environ.get("COLLISION_DB_PATH", os.path.join(PROJECT_ROOT, "collision_api.db"))
    if os.path.exists(db_path):
        try:
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM feedback")
            for row in cursor.fetchall():
                raw_records.append(dict(row))
            conn.close()
        except Exception as e:
            print(f"DB feedback query note: {e}")

    # Read raw JSON/JSONL files in data/real_world/raw/
    if os.path.exists(RAW_DIR):
        for f in os.listdir(RAW_DIR):
            if f.endswith(".json") or f.endswith(".jsonl"):
                fpath = os.path.join(RAW_DIR, f)
                with open(fpath, "r", encoding="utf-8") as file:
                    if f.endswith(".jsonl"):
                        for line in file:
                            if line.strip():
                                try:
                                    raw_records.append(json.loads(line))
                                except Exception:
                                    pass
                    else:
                        try:
                            data = json.load(file)
                            if isinstance(data, list):
                                raw_records.extend(data)
                            else:
                                raw_records.append(data)
                        except Exception:
                            pass

    # 2. Run data pipeline cleaning & quality validation
    cleaned_records, rejected_records = validate_and_clean_records(raw_records)

    # 3. Format with provenance source: real_world
    formatted_v2 = []
    for item in cleaned_records:
        formatted_v2.append({
            "instruction": item["prompt"],
            "response": item["response"],
            "category": item.get("category", "real_world_feedback"),
            "source": "real_world",
            "user_id": item.get("user_id", "anonymous"),
            "rating": item.get("rating", "thumbs_up")
        })

    # Save to data/real_world/cleaned/collision_real_world_v2.jsonl
    cleaned_v2_path = os.path.join(CLEANED_DIR, "collision_real_world_v2.jsonl")
    with open(cleaned_v2_path, "w", encoding="utf-8") as f:
        for item in formatted_v2:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # Save to datasets/collision_instruct_v1/real_world_v2_formatted.jsonl
    formatted_v2_path = os.path.join(OUT_DIR, "real_world_v2_formatted.jsonl")
    with open(formatted_v2_path, "w", encoding="utf-8") as f:
        for item in formatted_v2:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    total_tokens = sum(len(tokenizer.encode(r["instruction"], bos=True)) + len(tokenizer.encode(r["response"], eos=True)) for r in formatted_v2)

    stats = {
        "raw_records_evaluated": len(raw_records),
        "clean_v2_records": len(formatted_v2),
        "rejected_v2_records": len(rejected_records),
        "total_real_world_tokens": total_tokens,
        "provenance_label": "source=real_world",
        "cleaned_file": cleaned_v2_path
    }

    print(f"Real-World Dataset v2 Built:")
    print(f"  Evaluated: {len(raw_records)} -> Cleaned: {len(formatted_v2)} (Tokens: {total_tokens:,})")
    return stats

if __name__ == "__main__":
    build_real_world_v2()
