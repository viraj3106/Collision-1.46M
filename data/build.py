import os
import json
import re
import argparse
from datetime import datetime
import numpy as np

from data.prepare import clean_and_normalize, get_next_version
from data.tokenize import BPETokenizer

def check_quality(filename, text):
    warnings = []
    # 1. Very short documents
    if len(text.strip()) < 50:
        warnings.append("Extremely short document (less than 50 characters).")
        
    # 2. Mostly whitespace
    whitespace_ratio = len(re.findall(r'\s', text)) / max(1, len(text))
    if whitespace_ratio > 0.5:
        warnings.append(f"Mostly whitespace (whitespace ratio: {whitespace_ratio:.1%}).")
        
    # 3. Mostly symbols
    symbols = re.findall(r'[^a-zA-Z0-9\s]', text)
    symbol_ratio = len(symbols) / max(1, len(text))
    if symbol_ratio > 0.4:
        warnings.append(f"Mostly symbols/punctuation (symbol ratio: {symbol_ratio:.1%}).")
        
    # 4. Extremely repetitive text (check 3-gram frequencies)
    words = re.findall(r'\w+', text.lower())
    if len(words) >= 10:
        trigrams = list(zip(words[:-2], words[1:-1], words[2:]))
        if trigrams:
            freq = {}
            for tg in trigrams:
                freq[tg] = freq.get(tg, 0) + 1
            max_freq = max(freq.values())
            ratio = max_freq / len(trigrams)
            if ratio > 0.2:
                warnings.append(f"Highly repetitive phrasing detected (top trigram frequency ratio: {ratio:.1%}).")

    return warnings

def main():
    parser = argparse.ArgumentParser(description="COLLISION Dataset Builder")
    parser.add_argument("--raw-dir", type=str, default="data/raw", help="Path to raw data directory")
    parser.add_argument("--datasets-dir", type=str, default="datasets", help="Path to datasets directory")
    parser.add_argument("--save-dir", type=str, default="artifacts/tokenizer", help="Tokenizer save directory")
    parser.add_argument("--vocab-size", type=int, default=8000, help="Target vocabulary size")
    parser.add_argument("--seed", type=int, default=42, help="Seed for deterministic data splitting")
    args = parser.parse_args()

    # Discover files
    if not os.path.exists(args.raw_dir):
        print(f"Error: Raw directory not found at {args.raw_dir}")
        return

    txt_files = [f for f in os.listdir(args.raw_dir) if f.endswith(".txt")]
    if not txt_files:
        print(f"No .txt documents discovered in {args.raw_dir}.")
        return

    print(f"--- Discovering raw documents ---")
    documents = []
    seen_hashes = set()
    duplicate_count = 0
    empty_file_count = 0
    total_characters = 0
    quality_warnings = {}

    for f_name in txt_files:
        f_path = os.path.join(args.raw_dir, f_name)
        with open(f_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
            
        if not content.strip():
            empty_file_count += 1
            quality_warnings[f_name] = ["Empty document file."]
            continue

        total_characters += len(content)
        cleaned = clean_and_normalize(content)
        
        # Check suspicious patterns
        warnings = check_quality(f_name, cleaned)
        if warnings:
            quality_warnings[f_name] = warnings

        # Deduplication
        doc_hash = hash(cleaned)
        if doc_hash in seen_hashes:
            duplicate_count += 1
            continue
            
        seen_hashes.add(doc_hash)
        documents.append({
            "name": f_name,
            "text": cleaned
        })

    print(f"Found {len(txt_files)} file(s).")
    print(f"Empty files: {empty_file_count} | Duplicates removed: {duplicate_count}")
    print(f"Unique documents remaining: {len(documents)}")

    if not documents:
        print("Error: No unique content remaining to tokenize.")
        return

    # Train/Reload Tokenizer
    print("\n--- Tokenizer Training & Setup ---")
    combined_clean_text = "\n".join([doc["text"] for doc in documents])
    tokenizer = BPETokenizer()
    training_text_subset = combined_clean_text[:20000]
    stats = tokenizer.train(training_text_subset, args.vocab_size)
    tokenizer.save(args.save_dir)

    
    with open(os.path.join(args.save_dir, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
        
    print(f"Tokenizer vocabulary size: {len(tokenizer.inverse_vocab)}")

    # Verificationencode/decode
    test_str = "The future of artificial intelligence."
    enc_test = tokenizer.encode(test_str, bos=True, eos=True)
    dec_test = tokenizer.decode(enc_test)
    is_valid = dec_test.strip() == test_str.strip() or test_str.replace(" ", "") in dec_test.replace(" ", "")
    print(f"Tokenizer Verification | Input: '{test_str}' | Output: '{dec_test.strip()}' | Reconstructed matches: {is_valid}")

    # Deterministic Split
    print("\n--- Deterministic Dataset Splitting (90/10) ---")
    # Set seed locally for split
    rng = np.random.default_rng(args.seed)
    indices = np.arange(len(documents))
    rng.shuffle(indices)
    
    split_idx = int(0.9 * len(documents))
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    # If single document, allocate at least one to val for safety
    if len(documents) == 1:
        # Fallback to token level split
        print("Single document corpus detected. Performing split at token level instead of document level.")
        token_ids = tokenizer.encode(combined_clean_text, bos=True, eos=True)
        split_tok_idx = int(0.9 * len(token_ids))
        train_tokens = token_ids[:split_tok_idx]
        val_tokens = token_ids[split_tok_idx:]
    else:
        train_tokens = []
        for idx in train_indices:
            train_tokens.extend(tokenizer.encode(documents[idx]["text"], bos=True, eos=True))
            
        val_tokens = []
        for idx in val_indices:
            val_tokens.extend(tokenizer.encode(documents[idx]["text"], bos=True, eos=True))

    total_tokens = len(train_tokens) + len(val_tokens)

    # Determine Version Directory
    next_ver = get_next_version(args.datasets_dir)
    version_name = f"collision_dataset_v{next_ver}"
    version_dir = os.path.join(args.datasets_dir, version_name)
    os.makedirs(version_dir, exist_ok=True)

    # Save to Version Dir
    v_train_path = os.path.join(version_dir, "train.bin")
    v_val_path = os.path.join(version_dir, "val.bin")
    np.array(train_tokens, dtype=np.uint16).tofile(v_train_path)
    np.array(val_tokens, dtype=np.uint16).tofile(v_val_path)

    # Copy to global processed dir
    g_train_path = os.path.join("data", "processed", "train.bin")
    g_val_path = os.path.join("data", "processed", "val.bin")
    os.makedirs(os.path.dirname(g_train_path), exist_ok=True)
    np.array(train_tokens, dtype=np.uint16).tofile(g_train_path)
    np.array(val_tokens, dtype=np.uint16).tofile(g_val_path)

    # Print Quality warning summary
    if quality_warnings:
        print("\n--- DATASET QUALITY WARNINGS (Needs Review) ---")
        for doc_name, warns in quality_warnings.items():
            print(f"  [{doc_name}]:")
            for w in warns:
                print(f"    - {w}")

    # Generate metadata.json
    doc_lens = [len(doc["text"]) for doc in documents]
    meta = {
        "dataset_version": version_name,
        "creation_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "source_files": txt_files,
        "number_of_files": len(txt_files),
        "number_of_documents": len(documents),
        "character_count": total_characters,
        "token_count": total_tokens,
        "train_tokens": len(train_tokens),
        "validation_tokens": len(val_tokens),
        "tokenizer_version": "1.0-BPETokenizer",
        "vocabulary_size": len(tokenizer.inverse_vocab),
        "avg_document_length_chars": float(np.mean(doc_lens)) if doc_lens else 0.0,
        "min_document_length_chars": int(np.min(doc_lens)) if doc_lens else 0,
        "max_document_length_chars": int(np.max(doc_lens)) if doc_lens else 0,
        "duplicate_count": duplicate_count,
        "empty_file_count": empty_file_count,
        "preprocessing_configuration": {
            "whitespace_normalized": True,
            "duplicate_filtering": True,
            "split_ratio": "90/10",
            "seed": args.seed
        },
        "quality_warnings": quality_warnings
    }
    
    meta_path = os.path.join(version_dir, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved dataset version metadata to {meta_path}")
    print(f"Dataset Build completed successfully. Token count: {total_tokens:,}")

if __name__ == "__main__":
    main()
