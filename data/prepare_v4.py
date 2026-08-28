import os
import json
import re
import numpy as np
import random
from datetime import datetime
from data.tokenize import BPETokenizer

def clean_and_normalize(text):
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n+', '\n', text)
    return text.strip()

def build_v4():
    raw_dir = "data/raw"
    datasets_dir = "datasets"
    tokenizer_dir = "artifacts/tokenizer"
    
    # 1. Discover files
    txt_files = sorted([f for f in os.listdir(raw_dir) if f.endswith(".txt")])
    print(f"Discovered source files for v4: {txt_files}")
    
    train_docs = []
    val_docs = []
    
    total_raw_chars = 0
    total_cleaned_chars = 0
    
    # Track details for metadata
    subject_stats = {}
    
    for f_name in txt_files:
        f_path = os.path.join(raw_dir, f_name)
        subject = f_name.replace(".txt", "")
        
        with open(f_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
            total_raw_chars += len(content)
            
        # Split by paragraph
        paras = [p.strip() for p in content.split("\n") if p.strip()]
        
        cleaned_paras = []
        seen_paras = set()
        
        for p in paras:
            # Strip off Document Section prefix
            p_clean = re.sub(r'^Document Section \d+:\s*', '', p)
            p_clean = clean_and_normalize(p_clean)
            if not p_clean:
                continue
                
            if p_clean not in seen_paras:
                seen_paras.add(p_clean)
                cleaned_paras.append(p_clean)
        
        # Deterministic split per subject
        # Sort to ensure absolute determinism across environments
        cleaned_paras.sort()
        
        # Seeded shuffle
        rng = random.Random(42)
        rng.shuffle(cleaned_paras)
        
        split_idx = int(0.9 * len(cleaned_paras))
        subj_train = cleaned_paras[:split_idx]
        subj_val = cleaned_paras[split_idx:]
        
        train_docs.extend(subj_train)
        val_docs.extend(subj_val)
        
        subject_stats[subject] = {
            "total_unique_paragraphs": len(cleaned_paras),
            "train_paragraphs": len(subj_train),
            "val_paragraphs": len(subj_val)
        }
        
        print(f"Subject {subject}: unique={len(cleaned_paras)}, train={len(subj_train)}, val={len(subj_val)}")

    # Shuffle train and validation sets to mix subjects during training
    rng_train = random.Random(1337)
    rng_train.shuffle(train_docs)
    
    rng_val = random.Random(1337)
    rng_val.shuffle(val_docs)

    # Join documents with double newline to preserve boundaries
    train_text = "\n\n".join(train_docs)
    val_text = "\n\n".join(val_docs)
    
    total_cleaned_chars = len(train_text) + len(val_text)
    
    # Create dataset v4 directory
    version_dir = os.path.join(datasets_dir, "collision_dataset_v4")
    os.makedirs(version_dir, exist_ok=True)
    
    # Save the cleaned text splits for reference/debugging
    with open(os.path.join(version_dir, "train_cleaned.txt"), "w", encoding="utf-8") as f:
        f.write(train_text)
    with open(os.path.join(version_dir, "val_cleaned.txt"), "w", encoding="utf-8") as f:
        f.write(val_text)

    # 2. Tokenize and write binaries
    tokenizer = BPETokenizer()
    tokenizer.load(tokenizer_dir)
    
    print("Tokenizing train split...")
    train_ids = tokenizer.encode(train_text, bos=True, eos=True)
    print("Tokenizing val split...")
    val_ids = tokenizer.encode(val_text, bos=True, eos=True)
    
    train_bin_path = os.path.join(version_dir, "train.bin")
    val_bin_path = os.path.join(version_dir, "val.bin")
    
    np.array(train_ids, dtype=np.uint16).tofile(train_bin_path)
    np.array(val_ids, dtype=np.uint16).tofile(val_bin_path)
    
    # Also save to data/processed for training scripts that use it
    processed_dir = "data/processed"
    os.makedirs(processed_dir, exist_ok=True)
    np.array(train_ids, dtype=np.uint16).tofile(os.path.join(processed_dir, "train.bin"))
    np.array(val_ids, dtype=np.uint16).tofile(os.path.join(processed_dir, "val.bin"))
    
    # Write metadata
    meta = {
        "dataset_version": "collision_dataset_v4",
        "creation_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "source_files": txt_files,
        "raw_characters": total_raw_chars,
        "cleaned_characters": total_cleaned_chars,
        "token_count": len(train_ids) + len(val_ids),
        "train_tokens": len(train_ids),
        "validation_tokens": len(val_ids),
        "tokenizer_version": "1.0-BPETokenizer",
        "vocabulary_size": len(tokenizer.inverse_vocab),
        "preprocessing_configuration": {
            "whitespace_normalized": True,
            "duplicate_filtering": True,
            "strip_document_prefixes": True,
            "split_ratio": "90/10",
            "seed": 42
        },
        "subject_distribution": subject_stats
    }
    
    with open(os.path.join(version_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
        
    print("collision_dataset_v4 built successfully!")
    print(f"Train tokens: {len(train_ids):,}")
    print(f"Val tokens: {len(val_ids):,}")
    print(f"Total tokens: {len(train_ids) + len(val_ids):,}")

if __name__ == "__main__":
    build_v4()
