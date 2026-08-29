import os
import sys
import json
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Resolve project root path and insert into Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model.config import ModelConfig
from model.transformer import CollisionTransformer
from data.tokenize import BPETokenizer

# Paths
BASE_CP_PATH = "checkpoints/phase12b/collision-3.38m-phase12b-best.pt"
M10_CP_PATH = "checkpoints/phase14/collision-10m-best.pt"
TOKENIZER_DIR = "artifacts/tokenizer"
TEST_BIN = "datasets/collision_dataset_v5_expanded/test.bin"
OUTPUT_FILE = "experiments/phase14/generation_comparison.txt"

class NonOverlappingTokenDataset(Dataset):
    def __init__(self, bin_path: str, seq_len: int):
        self.data = np.fromfile(bin_path, dtype=np.uint16)
        self.seq_len = seq_len

    def __len__(self):
        return max(0, (len(self.data) - 1) // self.seq_len)

    def __getitem__(self, idx):
        start = idx * self.seq_len
        x = torch.from_numpy(self.data[start : start + self.seq_len].astype(np.int64))
        y = torch.from_numpy(self.data[start + 1 : start + self.seq_len + 1].astype(np.int64))
        return x, y

def evaluate_test_split(model, bin_path, seq_len, device):
    dataset = NonOverlappingTokenDataset(bin_path, seq_len)
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    model.eval()
    total_loss = 0.0
    steps = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            total_loss += loss.item()
            steps += 1
    mean_loss = total_loss / max(1, steps)
    perplexity = np.exp(mean_loss) if mean_loss < 20 else float('inf')
    return mean_loss, perplexity

def generate_response(model, tokenizer, prompt, device, max_tokens=100, temp=0.7, top_k=50):
    model.eval()
    ids = tokenizer.encode(prompt, bos=True)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        for _ in range(max_tokens):
            x_cond = x if x.size(1) <= model.config.max_seq_len else x[:, -model.config.max_seq_len:]
            logits, _ = model(x_cond)
            next_token_logits = logits[0, -1, :] / temp
            if top_k > 0:
                v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                next_token_logits[next_token_logits < v[-1]] = -float('Inf')
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            x = torch.cat((x, next_token.unsqueeze(0)), dim=1)
            if next_token.item() == tokenizer.special_tokens.get("[EOS]", 259):
                break
    generated_ids = x[0].tolist()
    return generated_ids, tokenizer.decode(generated_ids[len(ids):])

def calculate_quality_metrics(token_ids, prompt_len, tokenizer):
    gen_ids = token_ids[prompt_len:]
    if len(gen_ids) == 0:
        return {
            "repetition_rate": 0.0,
            "unique_token_ratio": 0.0,
            "repeated_2grams": 0,
            "repeated_3grams": 0,
            "length": 0,
            "terminated": False
        }
    
    unique_tokens = set(gen_ids)
    unique_ratio = len(unique_tokens) / len(gen_ids)
    repetition_rate = 1.0 - unique_ratio
    
    n2_grams = list(zip(gen_ids[:-1], gen_ids[1:]))
    n2_repeats = len(n2_grams) - len(set(n2_grams))
    
    n3_grams = list(zip(gen_ids[:-2], gen_ids[1:-1], gen_ids[2:]))
    n3_repeats = len(n3_grams) - len(set(n3_grams))
    
    eos_id = tokenizer.special_tokens.get("[EOS]", 259)
    terminated = False
    if gen_ids[-1] == eos_id:
        terminated = True
    else:
        text = tokenizer.decode(gen_ids)
        if text.strip() and text.strip()[-1] in ['.', '!', '?']:
            terminated = True
            
    return {
        "repetition_rate": repetition_rate,
        "unique_token_ratio": unique_ratio,
        "repeated_2grams": n2_repeats,
        "repeated_3grams": n3_repeats,
        "length": len(gen_ids),
        "terminated": terminated
    }

def main():
    device = torch.device("cpu")
    print("Starting Phase 14: 3.38M vs 10M Capacity Comparison...")
    
    tokenizer = BPETokenizer()
    tokenizer.load(TOKENIZER_DIR)
    
    # Load 3.38M Model
    print("Loading 3.38M Base checkpoint...")
    base_cp = torch.load(BASE_CP_PATH, map_location=device)
    base_cfg = ModelConfig(**base_cp["config"])
    base_model = CollisionTransformer(base_cfg).to(device)
    base_model.load_state_dict(base_cp["model_state_dict"])
    
    # Load 10M Model
    print("Loading 10M Base checkpoint...")
    if not os.path.exists(M10_CP_PATH):
        print(f"Error: 10M checkpoint not found at {M10_CP_PATH}. Run training first.")
        return
    m10_cp = torch.load(M10_CP_PATH, map_location=device)
    m10_cfg = ModelConfig(**m10_cp["config"])
    m10_model = CollisionTransformer(m10_cfg).to(device)
    m10_model.load_state_dict(m10_cp["model_state_dict"])
    
    # 1. Evaluate on test split
    print("Evaluating 3.38M on test split...")
    base_test_loss, base_test_ppx = evaluate_test_split(base_model, TEST_BIN, base_cfg.max_seq_len, device)
    
    print("Evaluating 10M on test split...")
    m10_test_loss, m10_test_ppx = evaluate_test_split(m10_model, TEST_BIN, m10_cfg.max_seq_len, device)
    
    print(f"3.38M Test Loss: {base_test_loss:.4f} | Perplexity: {base_test_ppx:.2f}")
    print(f"10M Test Loss: {m10_test_loss:.4f} | Perplexity: {m10_test_ppx:.2f}")
    
    # 2. Prompts
    prompts = [
        "Artificial intelligence is",
        "To prevent overfitting, a model should",
        "Explain machine learning.",
        "What is gravity?",
        "Why do stars shine?",
        "Explain a computer algorithm.",
        # Unseen generalization checks
        "What is a black hole?",
        "Define a prime number.",
        "Compare a stack and a queue."
    ]
    
    base_metrics = []
    m10_metrics = []
    
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("COLLISION Phase 14 — Model Capacity Generation Comparison (3.38M vs 10M)\n")
        f.write("========================================================================\n\n")
        
        for idx, prompt in enumerate(prompts):
            print(f"Generating for prompt {idx+1}/{len(prompts)}...")
            f.write(f"PROMPT {idx+1}: {prompt}\n")
            f.write("-" * 50 + "\n")
            
            # 3.38M Base Generation
            b_ids, b_text = generate_response(base_model, tokenizer, prompt, device)
            b_m = calculate_quality_metrics(b_ids, len(tokenizer.encode(prompt, bos=True)), tokenizer)
            base_metrics.append(b_m)
            f.write("3.38M BASE:\n")
            f.write(f"  {b_text.strip()}\n")
            f.write(f"  [Repetition: {b_m['repetition_rate']:.1%} | Unique: {b_m['unique_token_ratio']:.1%} | Terminated: {b_m['terminated']}]\n\n")
            
            # 10M Base Generation
            m_ids, m_text = generate_response(m10_model, tokenizer, prompt, device)
            m_m = calculate_quality_metrics(m_ids, len(tokenizer.encode(prompt, bos=True)), tokenizer)
            m10_metrics.append(m_m)
            f.write("10M BASE:\n")
            f.write(f"  {m_text.strip()}\n")
            f.write(f"  [Repetition: {m_m['repetition_rate']:.1%} | Unique: {m_m['unique_token_ratio']:.1%} | Terminated: {m_m['terminated']}]\n")
            f.write("=" * 80 + "\n\n")
            
    # Print comparison
    def get_avg(metrics_list, key):
        return np.mean([m[key] for m in metrics_list])
        
    print("\n==================================================")
    print("PHASE 14 COMPLETED — METRICS SUMMARY")
    print("==================================================")
    print(f"| Metric | 3.38M Model | 10M Model |")
    print(f"|---|---|---|")
    print(f"| Test Loss | {base_test_loss:.4f} | {m10_test_loss:.4f} |")
    print(f"| Test Perplexity | {base_test_ppx:.2f} | {m10_test_ppx:.2f} |")
    print(f"| Avg Repetition Rate | {get_avg(base_metrics, 'repetition_rate'):.1%} | {get_avg(m10_metrics, 'repetition_rate'):.1%} |")
    print(f"| Avg Unique Token Ratio | {get_avg(base_metrics, 'unique_token_ratio'):.1%} | {get_avg(m10_metrics, 'unique_token_ratio'):.1%} |")
    print(f"| Avg Repeated 2-grams | {get_avg(base_metrics, 'repeated_2grams'):.1f} | {get_avg(m10_metrics, 'repeated_2grams'):.1f} |")
    print(f"| Avg Repeated 3-grams | {get_avg(base_metrics, 'repeated_3grams'):.1f} | {get_avg(m10_metrics, 'repeated_3grams'):.1f} |")
    print(f"| Sentence Termination Rate | {get_avg(base_metrics, 'terminated'):.1%} | {get_avg(m10_metrics, 'terminated'):.1%} |")
    print(f"| Avg Response Length | {get_avg(base_metrics, 'length'):.1f} tokens | {get_avg(m10_metrics, 'length'):.1f} tokens |")
    print("==================================================")
    
    # Save metadata summary
    summary_meta = {
        "base_test_loss": base_test_loss,
        "base_test_ppx": base_test_ppx,
        "m10_test_loss": m10_test_loss,
        "m10_test_ppx": m10_test_ppx,
        "base_rep_rate": float(get_avg(base_metrics, 'repetition_rate')),
        "m10_rep_rate": float(get_avg(m10_metrics, 'repetition_rate')),
        "base_termination": float(get_avg(base_metrics, 'terminated')),
        "m10_termination": float(get_avg(m10_metrics, 'terminated'))
    }
    with open("experiments/phase14/evaluation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_meta, f, indent=2)

if __name__ == "__main__":
    main()
