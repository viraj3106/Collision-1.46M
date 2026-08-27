import os
import json
import re
import argparse
from datetime import datetime

def generate_sample_text():
    text_blocks = [
        "COLLISION-1M is a small decoder-only Transformer language model.",
        "We are training COLLISION-1M from scratch on a computer CPU.",
        "A language model learns to predict the next token in a sequence.",
        "The model uses causal self-attention to read previous tokens.",
        "Each token is represented by an embedding vector of numbers.",
        "Attention heads compute relationships between words and sentences.",
        "The neural network updates its weights using gradient descent and backpropagation.",
        "AdamW is the optimizer used to train the language model weights.",
        "A checkpoint saves the model weights and optimizer state so we can resume training.",
        "Streamlit displays the COLLISION LAB dashboard for training progress.",
        "The physics of a collision involves conservation of momentum and energy.",
        "An elastic collision conserves both kinetic energy and momentum.",
        "An inelastic collision converts kinetic energy into other forms of energy like heat.",
        "When particles collide, they exchange energy and momentum.",
        "The collider accelerates particles to near the speed of light.",
        "High-energy collisions help physicists discover new particles and forces.",
        "In deep learning, we stack multiple layers of attention and feed-forward networks.",
        "Positional embeddings allow the model to understand the order of tokens in a sequence.",
        "A tokenizer splits text into integers called token IDs.",
        "Evaluating the model gives us the training loss, validation loss, and perplexity.",
        "Perplexity measures how well the model predicts sample text.",
        "We can scale the model: COLLISION-1M to COLLISION-10M and then COLLISION-50M.",
        "The future of learning is built on feedback loops and model alignment."
    ]
    replicated_text = "\n".join(text_blocks * 50)
    return replicated_text

def clean_and_normalize(text):
    # Removes obviously corrupted content / non-printable control chars, preserves printable ascii and common spacing
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    # Normalizes spaces
    text = re.sub(r'[ \t]+', ' ', text)
    # Normalizes double newlines to single newlines or standard spacing
    text = re.sub(r'\n+', '\n', text)
    return text.strip()

def get_next_version(datasets_dir):
    os.makedirs(datasets_dir, exist_ok=True)
    versions = []
    for d in os.listdir(datasets_dir):
        match = re.match(r'collision_dataset_v(\d+)', d)
        if match:
            versions.append(int(match.group(1)))
    next_ver = max(versions) + 1 if versions else 1
    return next_ver

def main():
    parser = argparse.ArgumentParser(description="Prepare text datasets and generate versioned split outputs")
    parser.add_argument("--raw-dir", type=str, default="data/raw", help="Path to raw data directory")
    parser.add_argument("--datasets-dir", type=str, default="datasets", help="Path to datasets directory")
    args = parser.parse_args()

    os.makedirs(args.raw_dir, exist_ok=True)
    os.makedirs(args.datasets_dir, exist_ok=True)

    # 1. Ensure sample exists if empty
    sample_path = os.path.join(args.raw_dir, "sample.txt")
    if not os.path.exists(sample_path):
        with open(sample_path, "w", encoding="utf-8") as f:
            f.write(generate_sample_text())
        print(f"Created initial sample raw text at {sample_path}")

    # 2. Automatically discover all .txt files
    txt_files = [f for f in os.listdir(args.raw_dir) if f.endswith(".txt")]
    print(f"Discovered {len(txt_files)} file(s) in {args.raw_dir}: {txt_files}")

    documents = []
    seen_hashes = set()
    total_raw_chars = 0

    for f_name in txt_files:
        f_path = os.path.join(args.raw_dir, f_name)
        with open(f_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
            total_raw_chars += len(content)
            
            # Simple line-by-line deduplication/cleaning to remove redundant blocks
            lines = content.split("\n")
            cleaned_lines = []
            for line in lines:
                cleaned = clean_and_normalize(line)
                if not cleaned:
                    continue
                # Remove exact duplicate lines/documents to clean corpus
                line_hash = hash(cleaned)
                if line_hash not in seen_hashes:
                    seen_hashes.add(line_hash)
                    cleaned_lines.append(cleaned)
                    
            if cleaned_lines:
                documents.append("\n".join(cleaned_lines))

    # Join cleaned docs
    combined_text = "\n".join(documents)
    total_cleaned_chars = len(combined_text)

    # 3. Create auto-incrementing dataset version folder
    next_ver = get_next_version(args.datasets_dir)
    version_name = f"collision_dataset_v{next_ver}"
    version_dir = os.path.join(args.datasets_dir, version_name)
    os.makedirs(version_dir, exist_ok=True)

    # Write the cleaned text temporarily to be tokenized later
    cleaned_txt_path = os.path.join(version_dir, "cleaned.txt")
    with open(cleaned_txt_path, "w", encoding="utf-8") as f:
        f.write(combined_text)

    # Write initial metadata (will be fully completed after tokenization)
    meta = {
        "dataset_version": version_name,
        "source_files": txt_files,
        "raw_characters": total_raw_chars,
        "cleaned_characters": total_cleaned_chars,
        "creation_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "token_count": 0,
        "vocabulary_size": 0,
        "train_tokens": 0,
        "validation_tokens": 0
    }
    
    meta_path = os.path.join(version_dir, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Created versioned dataset directory at {version_dir}")
    print(f"Characters cleaned and written: {total_cleaned_chars} (raw: {total_raw_chars})")

if __name__ == "__main__":
    main()
