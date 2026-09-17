"""
COLLISION-10M — Standalone Single-File Inference Script.

Directly loads COLLISION-10M weights and executes fast CPU-native text generation
with zero external dependencies (pure PyTorch + standard library).

Usage:
    python generate.py --prompt "Artificial intelligence is"
    python generate.py --interactive
"""

import os
import sys
import json
import argparse
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# -----------------------------------------------------------------------------
# 1. Minimal Standalone Causal Transformer Architecture
# -----------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int, max_seq_len: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_head == 0
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

        # Causal mask buffer
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len)).view(1, 1, max_seq_len, max_seq_len)
        self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(y)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, d_ff: int, max_seq_len: int, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head, max_seq_len, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.GELU(),
            nn.Linear(d_ff, d_model, bias=False),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class Collision10M(nn.Module):
    def __init__(self, vocab_size: int = 8000, max_seq_len: int = 256, d_model: int = 384, n_layer: int = 6, n_head: int = 8, d_ff: int = 768):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.drop = nn.Dropout(0.0)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_head, d_ff, max_seq_len) for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying
        self.head.weight = self.token_emb.weight

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0)
        x = self.token_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.head(x)


# -----------------------------------------------------------------------------
# 2. Standalone Minimal Tokenizer
# -----------------------------------------------------------------------------

class StandaloneTokenizer:
    def __init__(self, tokenizer_dir: str):
        vocab_path = os.path.join(tokenizer_dir, "vocab.json")
        merges_path = os.path.join(tokenizer_dir, "merges.json")
        
        with open(vocab_path, "r", encoding="utf-8") as f:
            self.vocab = json.load(f)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        
        self.merges = {}
        if os.path.exists(merges_path):
            with open(merges_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 2:
                        self.merges[(parts[0], parts[1])] = len(self.merges)

        self.bos_token_id = self.vocab.get("<bos>", self.vocab.get("<s>", 1))
        self.eos_token_id = self.vocab.get("<eos>", self.vocab.get("</s>", 2))
        self.unk_token_id = self.vocab.get("<unk>", 0)

    def encode(self, text: str, bos: bool = True) -> list:
        # Byte-level / char-level fallback encoding
        tokens = []
        if bos:
            tokens.append(self.bos_token_id)

        # Simple greedy lookup
        words = text.split()
        for idx, w in enumerate(words):
            prefix = "Ġ" + w if idx > 0 else w
            if prefix in self.vocab:
                tokens.append(self.vocab[prefix])
            elif w in self.vocab:
                tokens.append(self.vocab[w])
            else:
                for ch in prefix:
                    tokens.append(self.vocab.get(ch, self.unk_token_id))
        return tokens

    def decode(self, token_ids: list) -> str:
        text = ""
        for t in token_ids:
            if t in (self.bos_token_id, self.eos_token_id):
                continue
            token_str = self.inv_vocab.get(t, "")
            token_str = token_str.replace("Ġ", " ")
            text += token_str
        return text.strip()


# -----------------------------------------------------------------------------
# 3. Generation Logic
# -----------------------------------------------------------------------------

def generate_text(model: nn.Module, tokenizer: StandaloneTokenizer, prompt: str, max_tokens: int = 50, temperature: float = 0.7, top_k: int = 50, top_p: float = 0.9, repetition_penalty: float = 1.15) -> str:
    device = next(model.parameters()).device
    tokens = tokenizer.encode(prompt, bos=True)
    input_ids = torch.tensor([tokens], dtype=torch.long, device=device)

    generated = list(tokens)

    for _ in range(max_tokens):
        if input_ids.size(1) >= model.max_seq_len:
            input_ids = input_ids[:, -model.max_seq_len:]

        with torch.no_grad():
            logits = model(input_ids)
            next_token_logits = logits[0, -1, :]

        # Apply repetition penalty
        for prev_token in set(generated):
            if next_token_logits[prev_token] > 0:
                next_token_logits[prev_token] /= repetition_penalty
            else:
                next_token_logits[prev_token] *= repetition_penalty

        if temperature <= 0.01:
            next_token = torch.argmax(next_token_logits).item()
        else:
            next_token_logits = next_token_logits / temperature
            # Top-K
            if top_k > 0:
                v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                next_token_logits[next_token_logits < v[-1]] = float("-inf")
            # Top-P
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[indices_to_remove] = float("-inf")

            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()

        if next_token == tokenizer.eos_token_id:
            break

        generated.append(next_token)
        input_ids = torch.tensor([generated], dtype=torch.long, device=device)

    # Decode only the generated completion
    completion_ids = generated[len(tokens):]
    return tokenizer.decode(completion_ids)


def load_model(weights_path: str, config_path: str, tokenizer_dir: str):
    with open(config_path, "r") as f:
        cfg = json.load(f)

    model = Collision10M(
        vocab_size=cfg.get("vocab_size", 8000),
        max_seq_len=cfg.get("max_seq_len", 256),
        d_model=cfg.get("d_model", 384),
        n_layer=cfg.get("n_layer", 6),
        n_head=cfg.get("n_head", 8),
        d_ff=cfg.get("d_ff", 768)
    )

    checkpoint = torch.load(weights_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    tokenizer = StandaloneTokenizer(tokenizer_dir)
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="COLLISION-10M Standalone Fast CPU Inference")
    parser.add_argument("--prompt", type=str, default="Artificial intelligence is", help="Prompt text")
    parser.add_argument("--max-tokens", type=int, default=50, help="Maximum generated tokens")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=50, help="Top-K token threshold")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-P nucleus threshold")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive CLI chat mode")
    args = parser.parse_args()

    weights_file = os.path.join(CURRENT_DIR, "model.pt")
    config_file = os.path.join(CURRENT_DIR, "config.json")
    tokenizer_folder = os.path.join(CURRENT_DIR, "tokenizer")

    if not os.path.exists(weights_file):
        print(f"Error: Weights file not found at {weights_file}")
        sys.exit(1)

    print("Loading COLLISION-10M (10.28M Parameters)...")
    model, tokenizer = load_model(weights_file, config_file, tokenizer_folder)
    print("Model ready on CPU!\n")

    if args.interactive:
        print("=== COLLISION-10M Interactive Session (type 'exit' to quit) ===")
        while True:
            try:
                user_p = input("\nUser: ").strip()
                if not user_p or user_p.lower() in ("exit", "quit"):
                    break
                out = generate_text(model, tokenizer, user_p, max_tokens=args.max_tokens, temperature=args.temperature, top_k=args.top_k, top_p=args.top_p)
                print(f"COLLISION: {out}")
            except (KeyboardInterrupt, EOFError):
                break
    else:
        print(f"Prompt: {args.prompt}")
        out = generate_text(model, tokenizer, args.prompt, max_tokens=args.max_tokens, temperature=args.temperature, top_k=args.top_k, top_p=args.top_p)
        print(f"COLLISION: {out}")

if __name__ == "__main__":
    main()
