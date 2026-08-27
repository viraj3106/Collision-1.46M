import os
import json
import argparse
import re
import numpy as np
from typing import List, Dict, Tuple
from datetime import datetime
from data.stats import get_latest_version_dir

class BPETokenizer:
    def __init__(self):
        # Define special tokens
        self.special_tokens = {
            "[PAD]": 256,
            "[UNK]": 257,
            "[BOS]": 258,
            "[EOS]": 259
        }
        self.inv_special_tokens = {v: k for k, v in self.special_tokens.items()}
        
        # Initialize vocab
        self.vocab = {}
        self.inverse_vocab = {}
        self.merges = {}
        self.reset_vocab()

    def reset_vocab(self):
        self.vocab = {bytes([i]): i for i in range(256)}
        for token, idx in self.special_tokens.items():
            self.vocab[token.encode('utf-8')] = idx
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        self.merges = {}

    def get_stats(self, words: List[List[int]]) -> Dict[Tuple[int, int], int]:
        stats = {}
        for word in words:
            for pair in zip(word[:-1], word[1:]):
                stats[pair] = stats.get(pair, 0) + 1
        return stats

    def merge_word(self, word: List[int], pair: Tuple[int, int], idx: int) -> List[int]:
        new_word = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == pair[0] and word[i+1] == pair[1]:
                new_word.append(idx)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        return new_word

    def train(self, text: str, vocab_size: int, verbose: bool = True) -> Dict:
        self.reset_vocab()
        if vocab_size <= 260:
            raise ValueError("vocab_size must be greater than 260 to accommodate base bytes and special tokens.")

        # Pre-tokenization
        word_chunks = re.findall(r'\s+|\w+|[^\w\s]', text)
        words = [list(chunk.encode('utf-8')) for chunk in word_chunks if chunk]
        
        num_merges = vocab_size - 260
        stats_history = []
        start_time = datetime.now()

        if verbose:
            print(f"Training tokenizer on text split into {len(words)} word chunks...")
            print(f"Target vocab size: {vocab_size} (number of merges to perform: {num_merges})")

        for i in range(num_merges):
            stats = self.get_stats(words)
            if not stats:
                break
            best_pair = max(stats, key=stats.get)
            if stats[best_pair] < 5:
                if verbose:
                    print("No more frequent pairs found (threshold < 5). Stopping early.")
                break

                
            new_idx = 260 + i
            self.merges[best_pair] = new_idx
            
            p0_val = self.inverse_vocab[best_pair[0]]
            p1_val = self.inverse_vocab[best_pair[1]]
            self.vocab[p0_val + p1_val] = new_idx
            self.inverse_vocab[new_idx] = p0_val + p1_val
            
            words = [self.merge_word(w, best_pair, new_idx) for w in words]
            
            if (i + 1) % 100 == 0:
                stats_history.append({
                    "merge_step": i + 1,
                    "vocab_size": len(self.inverse_vocab),
                    "best_pair": f"{best_pair[0]},{best_pair[1]}",
                    "frequency": stats[best_pair]
                })
                if verbose:
                    print(f"Merge {i+1}/{num_merges} completed. Vocab size: {len(self.inverse_vocab)}")

        training_duration = (datetime.now() - start_time).total_seconds()
        
        return {
            "training_duration_seconds": training_duration,
            "final_vocab_size": len(self.inverse_vocab),
            "total_merges_performed": len(self.merges),
            "merge_step_history": stats_history
        }

    def save(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        json_merges = {f"{k[0]},{k[1]}": v for k, v in self.merges.items()}
        json_vocab = {k.hex() if isinstance(k, bytes) else k.decode('utf-8'): v for k, v in self.vocab.items()}
        
        with open(os.path.join(save_dir, "vocab.json"), "w", encoding="utf-8") as f:
            json.dump(json_vocab, f, indent=2)
        with open(os.path.join(save_dir, "merges.json"), "w", encoding="utf-8") as f:
            json.dump(json_merges, f, indent=2)
            
        # Write custom config.json
        config = {
            "vocab_size": len(self.inverse_vocab),
            "special_tokens": self.special_tokens,
            "pretokenizer_pattern": r'\s+|\w+|[^\w\s]'
        }
        with open(os.path.join(save_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

    def load(self, save_dir: str):
        vocab_path = os.path.join(save_dir, "vocab.json")
        merges_path = os.path.join(save_dir, "merges.json")
        if not os.path.exists(vocab_path) or not os.path.exists(merges_path):
            raise FileNotFoundError(f"Vocab or merges file not found in {save_dir}")
            
        with open(vocab_path, "r", encoding="utf-8") as f:
            json_vocab = json.load(f)
        with open(merges_path, "r", encoding="utf-8") as f:
            json_merges = json.load(f)
            
        self.vocab = {}
        for k, v in json_vocab.items():
            if k in self.special_tokens:
                self.vocab[k.encode('utf-8')] = v
            else:
                self.vocab[bytes.fromhex(k)] = v
                
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        
        self.merges = {}
        for k, v in json_merges.items():
            p0, p1 = map(int, k.split(","))
            self.merges[(p0, p1)] = v

    def encode(self, text: str, bos: bool = False, eos: bool = False) -> List[int]:
        if not text:
            return []
        word_chunks = re.findall(r'\s+|\w+|[^\w\s]', text)
        res = []
        if bos:
            res.append(self.special_tokens["[BOS]"])
            
        for chunk in word_chunks:
            if not chunk:
                continue
            tokens = list(chunk.encode('utf-8'))
            for (p0, p1), new_idx in self.merges.items():
                tokens = self.merge_word(tokens, (p0, p1), new_idx)
            res.extend(tokens)
            
        if eos:
            res.append(self.special_tokens["[EOS]"])
        return res

    def decode(self, ids: List[int]) -> str:
        byte_parts = []
        for idx in ids:
            if idx in self.inv_special_tokens:
                continue
            elif idx in self.inverse_vocab:
                val = self.inverse_vocab[idx]
                if isinstance(val, bytes):
                    byte_parts.append(val)
                else:
                    byte_parts.append(val.encode('utf-8'))
        merged_bytes = b"".join(byte_parts)
        return merged_bytes.decode('utf-8', errors='replace')

def main():
    parser = argparse.ArgumentParser(description="Train and run BPE Tokenizer")
    parser.add_argument("--train", action="store_true", help="Train tokenizer from the latest prepared dataset version")
    parser.add_argument("--processed-dir", type=str, default="data/processed", help="Path to processed data folder")
    parser.add_argument("--save-dir", type=str, default="artifacts/tokenizer", help="Directory to save tokenizer")
    parser.add_argument("--vocab-size", type=int, default=8000, help="Target vocabulary size")
    args = parser.parse_args()

    latest_dir = get_latest_version_dir()
    if not latest_dir:
        print("No prepared dataset versions found. Please run 'python -m data.prepare' first.")
        return

    cleaned_txt_path = os.path.join(latest_dir, "cleaned.txt")
    if not os.path.exists(cleaned_txt_path):
        print(f"Error: cleaned.txt missing in {latest_dir}")
        return

    with open(cleaned_txt_path, "r", encoding="utf-8") as f:
        full_text = f.read()

    tokenizer = BPETokenizer()

    if args.train:
        # Train and save stats
        stats = tokenizer.train(full_text, args.vocab_size)
        tokenizer.save(args.save_dir)
        
        # Store training stats
        with open(os.path.join(args.save_dir, "stats.json"), "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
            
        print("Tokenizer trained successfully.")
    else:
        # Just load existing
        try:
            tokenizer.load(args.save_dir)
            print("Tokenizer loaded successfully.")
        except Exception:
            print("Tokenizer files not found. Automatically training new tokenizer...")
            stats = tokenizer.train(full_text, args.vocab_size)
            tokenizer.save(args.save_dir)
            with open(os.path.join(args.save_dir, "stats.json"), "w", encoding="utf-8") as f:
                json.dump(stats, f, indent=2)

    # Tokenize dataset
    print("Tokenizing dataset and creating train/validation splits...")
    token_ids = tokenizer.encode(full_text, bos=True, eos=True)
    total_tokens = len(token_ids)

    # Split train/val
    if total_tokens < 10:
        raise ValueError(f"Total tokens generated ({total_tokens}) is too small to split.")

    split_idx = int(0.9 * total_tokens)
    train_ids = token_ids[:split_idx]
    val_ids = token_ids[split_idx:]

    # Save to BOTH dataset version dir and global processed dir
    os.makedirs(args.processed_dir, exist_ok=True)
    
    # Save to latest version dir
    v_train_path = os.path.join(latest_dir, "train.bin")
    v_val_path = os.path.join(latest_dir, "val.bin")
    np.array(train_ids, dtype=np.uint16).tofile(v_train_path)
    np.array(val_ids, dtype=np.uint16).tofile(v_val_path)

    # Copy/Save to global processed dir
    g_train_path = os.path.join(args.processed_dir, "train.bin")
    g_val_path = os.path.join(args.processed_dir, "val.bin")
    np.array(train_ids, dtype=np.uint16).tofile(g_train_path)
    np.array(val_ids, dtype=np.uint16).tofile(g_val_path)

    # Update metadata.json in latest version dir
    meta_path = os.path.join(latest_dir, "metadata.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    meta["token_count"] = total_tokens
    meta["vocabulary_size"] = len(tokenizer.inverse_vocab)
    meta["train_tokens"] = len(train_ids)
    meta["validation_tokens"] = len(val_ids)

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Vocabulary size: {len(tokenizer.inverse_vocab)}")
    print(f"Training tokens: {len(train_ids)}")
    print(f"Validation tokens: {len(val_ids)}")
    print(f"Saved token files to both {latest_dir} and {args.processed_dir}")

if __name__ == "__main__":
    main()
