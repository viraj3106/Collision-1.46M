import os
import json
import argparse
import re
import numpy as np
from typing import List, Dict, Tuple

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
        # Vocab starts with 256 byte values
        self.vocab = {bytes([i]): i for i in range(256)}
        # Add special tokens
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

    def train(self, text: str, vocab_size: int, verbose: bool = True):
        self.reset_vocab()
        if vocab_size <= 260:
            raise ValueError("vocab_size must be greater than 260 to accommodate base bytes and special tokens.")

        # Split text into word-like chunks (pre-tokenization)
        # Using a regex to split words, whitespace, and punctuation
        word_chunks = re.findall(r'\s+|\w+|[^\w\s]', text)
        words = [list(chunk.encode('utf-8')) for chunk in word_chunks if chunk]
        
        num_merges = vocab_size - 260
        
        if verbose:
            print(f"Training tokenizer on text split into {len(words)} word chunks...")
            print(f"Target vocab size: {vocab_size} (number of merges to perform: {num_merges})")

        for i in range(num_merges):
            stats = self.get_stats(words)
            if not stats:
                break
            # Find the most frequent pair
            best_pair = max(stats, key=stats.get)
            if stats[best_pair] < 2:
                if verbose:
                    print("No more frequent pairs found. Stopping early.")
                break
                
            new_idx = 260 + i
            self.merges[best_pair] = new_idx
            
            # Update vocabulary
            p0_val = self.inverse_vocab[best_pair[0]]
            p1_val = self.inverse_vocab[best_pair[1]]
            # Merge the bytes
            self.vocab[p0_val + p1_val] = new_idx
            self.inverse_vocab[new_idx] = p0_val + p1_val
            
            words = [self.merge_word(w, best_pair, new_idx) for w in words]
            
            if verbose and (i + 1) % 100 == 0:
                print(f"Merge {i+1}/{num_merges} completed. Vocab size: {len(self.inverse_vocab)}")

        if verbose:
            print(f"Training completed. Final vocab size: {len(self.inverse_vocab)}")

    def save(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        json_merges = {f"{k[0]},{k[1]}": v for k, v in self.merges.items()}
        json_vocab = {k.hex() if isinstance(k, bytes) else k.decode('utf-8'): v for k, v in self.vocab.items()}
        
        with open(os.path.join(save_dir, "vocab.json"), "w", encoding="utf-8") as f:
            json.dump(json_vocab, f, indent=2)
        with open(os.path.join(save_dir, "merges.json"), "w", encoding="utf-8") as f:
            json.dump(json_merges, f, indent=2)

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
            
        # Split text into word-like chunks (pre-tokenization)
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
            else:
                pass
        
        merged_bytes = b"".join(byte_parts)
        return merged_bytes.decode('utf-8', errors='replace')

def main():
    parser = argparse.ArgumentParser(description="Train and run BPE Tokenizer")
    parser.add_argument("--raw-dir", type=str, default="data/raw", help="Path to raw texts")
    parser.add_argument("--processed-dir", type=str, default="data/processed", help="Path to processed data folder")
    parser.add_argument("--save-dir", type=str, default="artifacts/tokenizer", help="Directory to save tokenizer")
    parser.add_argument("--vocab-size", type=int, default=8000, help="Target vocabulary size")
    args = parser.parse_args()

    # Find raw training data
    txt_files = [f for f in os.listdir(args.raw_dir) if f.endswith(".txt")]
    if not txt_files:
        print(f"No .txt files found in {args.raw_dir}. Please run 'python -m data.prepare' first.")
        return

    # Combine all raw text
    full_text = ""
    for f_name in txt_files:
        with open(os.path.join(args.raw_dir, f_name), "r", encoding="utf-8") as f:
            full_text += f.read() + "\n"

    print(f"Loaded {len(txt_files)} text file(s). Total text length: {len(full_text)} characters.")

    # Train tokenizer
    tokenizer = BPETokenizer()
    tokenizer.train(full_text, args.vocab_size)
    tokenizer.save(args.save_dir)
    print(f"Tokenizer saved to {args.save_dir}")

    # Tokenize dataset
    print("Tokenizing dataset and creating train/validation splits...")
    token_ids = tokenizer.encode(full_text, bos=True, eos=True)
    print(f"Total tokens generated: {len(token_ids)}")

    # Split train/val
    if len(token_ids) < 10:
        raise ValueError(f"Total tokens generated ({len(token_ids)}) is too small to split into train/val datasets.")
        
    split_idx = int(0.9 * len(token_ids))
    train_ids = token_ids[:split_idx]
    val_ids = token_ids[split_idx:]
    
    if len(train_ids) == 0 or len(val_ids) == 0:
        raise ValueError(f"Split resulted in empty dataset. Train size: {len(train_ids)}, Val size: {len(val_ids)}")


    # Save to binary files
    os.makedirs(args.processed_dir, exist_ok=True)
    train_path = os.path.join(args.processed_dir, "train.bin")
    val_path = os.path.join(args.processed_dir, "val.bin")

    np.array(train_ids, dtype=np.uint16).tofile(train_path)
    np.array(val_ids, dtype=np.uint16).tofile(val_path)

    print(f"Saved {len(train_ids)} train tokens to {train_path}")
    print(f"Saved {len(val_ids)} val tokens to {val_path}")

if __name__ == "__main__":
    main()
