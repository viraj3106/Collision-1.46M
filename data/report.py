import os
import json

from data.stats import get_latest_version_dir

def evaluate_health(meta):
    token_count = meta.get("token_count", 0)
    vocab_size = meta.get("vocabulary_size", 0)
    dup_count = meta.get("duplicate_count", 0)
    num_docs = meta.get("number_of_documents", 1)
    
    dup_rate = dup_count / max(1, num_docs + dup_count)
    warnings = meta.get("quality_warnings", {})
    
    score = "HIGH"
    reasons = []
    
    # 1. Size evaluation
    if token_count < 100_000:
        score = "LOW"
        reasons.append("Dataset token count is extremely small (< 100K tokens).")
    elif token_count < 1_000_000:
        score = "MEDIUM"
        reasons.append("Dataset size is small (< 1M tokens) for training a reliable Transformer model.")
        
    # 2. Vocabulary evaluation
    if vocab_size < 300:
        score = "LOW"
        reasons.append(f"Vocabulary size is very restricted ({vocab_size} tokens), suggesting repetitive text patterns.")
    elif vocab_size < 1000 and score == "HIGH":
        score = "MEDIUM"
        reasons.append(f"Moderate vocabulary diversity ({vocab_size} tokens).")
        
    # 3. Cleanliness/Deduplication evaluation
    if dup_rate > 0.30:
        score = "LOW"
        reasons.append(f"Extremely high rate of duplicate documents ({dup_rate:.1%}).")
    elif dup_rate > 0.10 and score == "HIGH":
        score = "MEDIUM"
        reasons.append(f"Moderate duplicate rate ({dup_rate:.1%}).")
        
    # 4. Quantity of warnings
    if len(warnings) > (num_docs * 0.2) and score == "HIGH":
        score = "MEDIUM"
        reasons.append("More than 20% of documents have quality warnings (e.g. extremely short or symbols-heavy).")
        
    if not reasons:
        reasons.append("Large token size, healthy vocabulary diversity, and low duplication rates.")
        
    return score, reasons

def main():
    latest_dir = get_latest_version_dir()
    if not latest_dir:
        print("No prepared dataset versions found. Please run 'python -m data.build' first.")
        return

    meta_path = os.path.join(latest_dir, "metadata.json")
    if not os.path.exists(meta_path):
        print(f"Error: metadata.json missing in {latest_dir}")
        return

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    token_count = meta.get("token_count", 0)
    classification = ""
    if token_count < 100_000:
        classification = "TESTING ONLY"
    elif 100_000 <= token_count < 1_000_000:
        classification = "SMALL"
    elif 1_000_000 <= token_count < 5_000_000:
        classification = "GOOD FOR FIRST EXPERIMENT"
    elif 5_000_000 <= token_count < 10_000_000:
        classification = "STRONG SMALL-MODEL DATASET"
    else:
        classification = "EXTENDED EXPERIMENT"

    health_score, justifications = evaluate_health(meta)

    # Compile warnings list
    q_warnings = meta.get("quality_warnings", {})
    warnings_list = []
    for doc, warns in q_warnings.items():
        for w in warns:
            warnings_list.append(f"[{doc}] {w}")

    print("================================")
    print("COLLISION DATASET REPORT")
    print("========================\n")
    print(f"Dataset:            {meta.get('dataset_version', 'N/A')}")
    print(f"Documents:          {meta.get('number_of_documents', 0):,}")
    print(f"Characters:         {meta.get('character_count', 0):,}")
    print(f"Total Tokens:       {token_count:,}")
    print(f"Training Tokens:    {meta.get('train_tokens', 0):,}")
    print(f"Validation Tokens:  {meta.get('validation_tokens', 0):,}")
    print(f"Vocabulary:         {meta.get('vocabulary_size', 0)}")
    print(f"Duplicate Documents:{meta.get('duplicate_count', 0):,}")
    print(f"Quality Warnings:   {len(warnings_list)}")
    print()
    print("================================")
    print(f"CLASSIFICATION:     {classification}")
    print(f"HEALTH SCORE:       {health_score}")
    print("================================")
    print("Health Justifications:")
    for j in justifications:
        print(f"  - {j}")
        
    if warnings_list:
        print("\nQuality Warnings Log (Top 10):")
        for log in warnings_list[:10]:
            print(f"  - {log}")

if __name__ == "__main__":
    main()
