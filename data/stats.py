import os
import json
import re

def get_latest_version_dir(datasets_dir="datasets"):
    if not os.path.exists(datasets_dir):
        return None
    versions = []
    for d in os.listdir(datasets_dir):
        match = re.match(r'collision_dataset_v(\d+)', d)
        if match:
            versions.append((int(match.group(1)), d))
    if not versions:
        return None
    # Sort by version number
    latest_name = sorted(versions, key=lambda x: x[0])[-1][1]
    return os.path.join(datasets_dir, latest_name)

def main():
    latest_dir = get_latest_version_dir()
    if not latest_dir:
        print("No prepared dataset versions found. Please run 'python -m data.prepare' first.")
        return

    meta_path = os.path.join(latest_dir, "metadata.json")
    if not os.path.exists(meta_path):
        print(f"Error: metadata.json missing in {latest_dir}")
        return

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    # Calculate average document length
    raw_path = os.path.join("data", "raw")
    txt_files = meta.get("source_files", [])
    lengths = []
    for f_name in txt_files:
        f_path = os.path.join(raw_path, f_name)
        if os.path.exists(f_path):
            lengths.append(os.path.getsize(f_path))
    avg_len = sum(lengths) / len(lengths) if lengths else 0

    print("## COLLISION DATASET\n")
    print(f"Dataset Version:     {meta.get('dataset_version', 'N/A')}")
    print(f"Files:               {', '.join(txt_files)}")
    print(f"Characters:          {meta.get('cleaned_characters', 0):,}")
    print(f"Estimated tokens:    {meta.get('token_count', 0):,}")
    print(f"Training tokens:     {meta.get('train_tokens', 0):,}")
    print(f"Validation tokens:   {meta.get('validation_tokens', 0):,}")
    print(f"Vocabulary:          {meta.get('vocabulary_size', 0)}")
    print(f"Average doc length:  {avg_len:,.1f} bytes")
    print()

    # Warn about dataset size
    token_count = meta.get("token_count", 0)
    if token_count < 100_000:
        print("WARNING: Dataset is suitable only for testing.")
    elif 100_000 <= token_count < 1_000_000:
        print("Dataset classification: Small training dataset.")
    elif 1_000_000 <= token_count < 10_000_000:
        print("Dataset classification: Useful small-model training dataset.")
    else:
        print("Dataset classification: Good for extended experiments.")

if __name__ == "__main__":
    main()
