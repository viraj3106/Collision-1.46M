import os
import json
import numpy as np
import re
from collections import Counter
from data.tokenize import BPETokenizer

def analyze_v3():
    datasets_dir = "datasets/collision_dataset_v3"
    raw_dir = "data/raw"
    tokenizer_dir = "artifacts/tokenizer"

    # 1. Load Tokenizer
    tokenizer = BPETokenizer()
    tokenizer.load(tokenizer_dir)

    # 2. Load Binaries
    train_ids = np.fromfile(os.path.join(datasets_dir, "train.bin"), dtype=np.uint16)
    val_ids = np.fromfile(os.path.join(datasets_dir, "val.bin"), dtype=np.uint16)
    total_tokens = len(train_ids) + len(val_ids)

    # 3. Read raw files and split into document sections
    raw_files = [f for f in os.listdir(raw_dir) if f.endswith(".txt")]
    
    sections = []
    subject_map = {}
    
    for f_name in raw_files:
        f_path = os.path.join(raw_dir, f_name)
        subject = f_name.replace(".txt", "")
        with open(f_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        
        # Split by paragraph
        paras = [p.strip() for p in content.split("\n") if p.strip()]
        for p in paras:
            sections.append({
                "text": p,
                "subject": subject,
                "file": f_name
            })
            subject_map[subject] = subject_map.get(subject, 0) + 1

    # Analyze document details
    num_docs = len(sections)
    doc_char_lens = [len(s["text"]) for s in sections]
    
    # Tokenize each section to get token count per document
    doc_token_lens = []
    for s in sections:
        ids = tokenizer.encode(s["text"])
        doc_token_lens.append(len(ids))
        s["tokens"] = ids
        s["token_count"] = len(ids)

    # Repetition/Duplicate Rate
    # A document is a duplicate if its text matches another document exactly (or ignoring 'Document Section XXX:' prefix)
    cleaned_texts = []
    for s in sections:
        t = s["text"]
        # Strip off Document Section prefix
        t_clean = re.sub(r'^Document Section \d+:\s*', '', t)
        # For sample.txt, check its structure
        cleaned_texts.append(t_clean)
        
    unique_texts = set(cleaned_texts)
    duplicate_count = len(cleaned_texts) - len(unique_texts)
    duplicate_rate = (duplicate_count / num_docs) * 100 if num_docs > 0 else 0

    # Let's count repeated sentences across the entire corpus
    all_sentences = []
    for text in cleaned_texts:
        # Split into sentences
        sents = re.split(r'(?<=[.!?])\s+', text)
        all_sentences.extend([s.strip() for s in sents if s.strip()])
    
    sent_counts = Counter(all_sentences)
    total_sents = len(all_sentences)
    unique_sents = len(sent_counts)
    repeated_sents_count = sum(c for s, c in sent_counts.items() if c > 1)
    repetition_rate = (repeated_sents_count / total_sents) * 100 if total_sents > 0 else 0

    # Train/Validation Overlap
    # Since train.bin and val.bin are contiguous splits of combined_text, let's reconstruct train and val text
    train_text = tokenizer.decode(train_ids.tolist())
    val_text = tokenizer.decode(val_ids.tolist())

    # We can check how many sentences in val_text also exist in train_text
    val_sentences = re.split(r'(?<=[.!?])\s+', val_text)
    val_sentences = [s.strip() for s in val_sentences if s.strip()]
    
    overlap_count = 0
    for s in val_sentences:
        if s in train_text:
            overlap_count += 1
    val_leakage_rate = (overlap_count / len(val_sentences)) * 100 if val_sentences else 0

    # Subject distribution in training vs validation
    # Let's map token indices to subjects by reconstructing
    # The prepare script combined them in the order: discovered files.
    # Let's see which files represent what percentage of train vs val.
    # Since the split is at 90% mark:
    split_idx = len(train_ids)
    
    # We can reconstruct the combined text and see the character offsets
    combined_tokens_decoded = tokenizer.decode(train_ids.tolist() + val_ids.tolist())
    # But wait, we can also decode train and val separately.
    # Let's check which subject-specific phrases occur in train vs val.
    # We can count subject-specific keywords or sentences
    subject_in_train = {}
    subject_in_val = {}
    
    for subj, sents in [
        ("physics", ["Classical mechanics", "Newton", "thermodynamics", "entropy", "Maxwell"]),
        ("computer_science", ["algorithm", "data structures", "binary search", "Big O", "polymorphism"]),
        ("artificial_intelligence", ["machine learning", "supervised learning", "neural networks", "Transformers", "gradient descent"]),
        ("astronomy", ["solar system", "planets orbit", "supernovae", "black holes", "Milky Way"]),
        ("philosophy", ["epistemology", "metaphysics", "utilitarianism", "existentialism", "Socrates"]),
        ("sample", ["COLLISION-1M is a small decoder-only", "We are training COLLISION-1M", "Streamlit displays the COLLISION LAB"])
    ]:
        subject_in_train[subj] = sum(1 for keyword in sents if keyword.lower() in train_text.lower())
        subject_in_val[subj] = sum(1 for keyword in sents if keyword.lower() in val_text.lower())

    # Formatted Audit report
    audit_report = f"""# Dataset Quality Audit: collision_dataset_v3

## General Statistics
* **Number of Documents (sections/paragraphs)**: {num_docs}
* **Total Tokens in Dataset**: {total_tokens:,}
* **Training Tokens**: {len(train_ids):,}
* **Validation Tokens**: {len(val_ids):,}
* **Split Ratio**: {len(train_ids)/total_tokens*100:.1f}% Train / {len(val_ids)/total_tokens*100:.1f}% Val

## Document Lengths (Characters)
* **Average Document Length**: {np.mean(doc_char_lens):.1f} chars
* **Shortest Document**: {np.min(doc_char_lens)} chars
* **Longest Document**: {np.max(doc_char_lens)} chars

## Document Lengths (Tokens)
* **Average Document Length**: {np.mean(doc_token_lens):.1f} tokens
* **Shortest Document**: {np.min(doc_token_lens)} tokens
* **Longest Document**: {np.max(doc_token_lens)} tokens

## Token and Character Distributions
* **Token count standard deviation**: {np.std(doc_token_lens):.2f}
* **Character count standard deviation**: {np.std(doc_char_lens):.2f}

## Redundancy & Repetition Analysis
* **Exact Duplicate Paragraphs/Sections**: {duplicate_count} ({duplicate_rate:.2f}%)
* **Sentence Repetition Rate**: {repetition_rate:.2f}% ({repeated_sents_count} of {total_sents} sentences are duplicates)
* **Train/Validation Sentence Leakage**: {val_leakage_rate:.2f}% ({overlap_count} of {len(val_sentences)} validation sentences also appear in the training set)

## Subject Representation (Keyword Occurrence)
* **Physics**: Train: {subject_in_train['physics']}/5 keywords | Val: {subject_in_val['physics']}/5 keywords
* **Computer Science**: Train: {subject_in_train['computer_science']}/5 keywords | Val: {subject_in_val['computer_science']}/5 keywords
* **Artificial Intelligence**: Train: {subject_in_train['artificial_intelligence']}/5 keywords | Val: {subject_in_val['artificial_intelligence']}/5 keywords
* **Astronomy**: Train: {subject_in_train['astronomy']}/5 keywords | Val: {subject_in_val['astronomy']}/5 keywords
* **Philosophy**: Train: {subject_in_train['philosophy']}/5 keywords | Val: {subject_in_val['philosophy']}/5 keywords
* **Sample/Repl Text**: Train: {subject_in_train['sample']}/3 keywords | Val: {subject_in_val['sample']}/3 keywords

## Validation Representativeness Analysis
Since the validation set was created by splitting the end 10% of the token stream, it is heavily biased toward the files parsed last (e.g., `sample.txt` and `physics.txt`).
Indeed, looking at keyword occurrence, subjects like computer science, AI, astronomy, and philosophy are completely missing or severely underrepresented in the validation set, while `sample.txt` is heavily overrepresented.
Furthermore, because `sample.txt` and the subject files are generated by repeating a small, closed list of sentences multiple times, the validation set suffers from a **{val_leakage_rate:.2f}% sentence leakage rate** (almost every sentence in the validation set has already been seen in the training set). This leads to artificially low validation loss and high perplexity metrics that do not reflect true generalization.
"""
    
    os.makedirs("experiments/phase6", exist_ok=True)
    with open("experiments/phase6/dataset_audit.md", "w", encoding="utf-8") as f:
        f.write(audit_report)
    print("Audit report written to experiments/phase6/dataset_audit.md")

if __name__ == "__main__":
    analyze_v3()
