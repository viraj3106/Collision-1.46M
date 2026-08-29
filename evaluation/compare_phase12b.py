import os
import sys
import time
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

# Config/Paths
V4_CP_PATH = "checkpoints/scaling/collision_3m/collision-3m-best.pt"
V5_CP_PATH = "checkpoints/phase12b/collision-3.38m-phase12b-best.pt"
TOKENIZER_DIR = "artifacts/tokenizer"
V4_VAL_BIN = "datasets/collision_dataset_v4/val.bin"
V5_VAL_BIN = "datasets/collision_dataset_v5_expanded/val.bin"
V5_TEST_BIN = "datasets/collision_dataset_v5_expanded/test.bin"

class TokenDataset(Dataset):
    def __init__(self, bin_path: str, seq_len: int):
        if not os.path.exists(bin_path):
            raise FileNotFoundError(f"Binary token file not found at {bin_path}.")
        self.data = np.fromfile(bin_path, dtype=np.uint16)
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.data) - self.seq_len - 1)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.data[idx : idx + self.seq_len].astype(np.int64))
        y = torch.from_numpy(self.data[idx + 1 : idx + self.seq_len + 1].astype(np.int64))
        return x, y

def evaluate_model(model, bin_path, seq_len, device):
    if not os.path.exists(bin_path):
        return float('nan'), float('nan')
    dataset = TokenDataset(bin_path, seq_len)
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
            if steps >= 200:  # Cap at 200 steps for fast evaluation
                break
    mean_loss = total_loss / max(1, steps)
    perplexity = np.exp(mean_loss) if mean_loss < 20 else float('inf')
    return mean_loss, perplexity

def generate_sample(model, tokenizer, prompt, device, max_tokens=50, temp=0.8, top_k=50):
    model.eval()
    ids = tokenizer.encode(prompt, bos=True)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    
    start_time = time.time()
    tokens_generated = 0
    
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
            tokens_generated += 1
            if next_token.item() == tokenizer.special_tokens.get("[EOS]", 259):
                break
                
    elapsed = time.time() - start_time
    tok_per_sec = tokens_generated / max(0.0001, elapsed)
    return x[0].tolist(), tokenizer.decode(x[0].tolist()), tok_per_sec

def calculate_quality_metrics(token_ids, prompt_len, tokenizer):
    gen_ids = token_ids[prompt_len:]
    if len(gen_ids) == 0:
        return {
            "repetition_rate": 0.0,
            "unique_token_ratio": 0.0,
            "repeated_2grams": 0,
            "repeated_3grams": 0,
            "length": 0,
            "unk_frequency": 0.0,
            "terminated": False
        }
    
    unique_tokens = set(gen_ids)
    unique_ratio = len(unique_tokens) / len(gen_ids)
    repetition_rate = 1.0 - unique_ratio
    
    n2_grams = list(zip(gen_ids[:-1], gen_ids[1:]))
    n2_repeats = len(n2_grams) - len(set(n2_grams))
    
    n3_grams = list(zip(gen_ids[:-2], gen_ids[1:-1], gen_ids[2:]))
    n3_repeats = len(n3_grams) - len(set(n3_grams))
    
    unk_id = tokenizer.special_tokens.get("[UNK]", 257)
    unk_count = gen_ids.count(unk_id)
    unk_freq = unk_count / len(gen_ids)
    
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
        "unk_frequency": unk_freq,
        "terminated": terminated
    }

def main():
    device = torch.device("cpu")
    print("Initializing Comparison between COLLISION V4 (Previous Best) and V5 (Phase 12B)...")
    
    # Load tokenizer
    tokenizer = BPETokenizer()
    tokenizer.load(TOKENIZER_DIR)
    
    # Prompts
    prompts = [
        "Artificial intelligence is",
        "Machine learning is",
        "Physics is",
        "Astronomy is",
        "Computer science is",
        "Philosophy is",
        "Question: What is a loss function? Answer:",
        "Explain why model training requires validation datasets.",
        "To prevent overfitting, a model should",
        "Why do stars shine?"
    ]
    
    # Checkpoints exists verify
    if not os.path.exists(V4_CP_PATH):
        print(f"Error: Previous V4 checkpoint not found at {V4_CP_PATH}")
        return
    if not os.path.exists(V5_CP_PATH):
        print(f"Error: Phase 12B V5 checkpoint not found at {V5_CP_PATH}")
        return
        
    # Load V4
    v4_cp = torch.load(V4_CP_PATH, map_location=device)
    v4_cfg = ModelConfig(**v4_cp["config"])
    v4_model = CollisionTransformer(v4_cfg).to(device)
    v4_model.load_state_dict(v4_cp["model_state_dict"])
    v4_param_count = sum(p.numel() for p in v4_model.parameters())
    
    # Load V5
    v5_cp = torch.load(V5_CP_PATH, map_location=device)
    v5_cfg = ModelConfig(**v5_cp["config"])
    v5_model = CollisionTransformer(v5_cfg).to(device)
    v5_model.load_state_dict(v5_cp["model_state_dict"])
    v5_param_count = sum(p.numel() for p in v5_model.parameters())
    
    # Evaluate Validation sets
    print("\nRunning loss evaluation on datasets...")
    v4_val_loss, v4_val_perp = evaluate_model(v4_model, V4_VAL_BIN, v4_cfg.max_seq_len, device)
    v5_val_loss, v5_val_perp = evaluate_model(v5_model, V5_VAL_BIN, v5_cfg.max_seq_len, device)
    
    # Evaluate Test sets
    # V4 doesn't have a test set in its dataset dir, but let's see how it behaves on V5 test set as generalization
    v4_test_loss, v4_test_perp = evaluate_model(v4_model, V5_TEST_BIN, v4_cfg.max_seq_len, device)
    v5_test_loss, v5_test_perp = evaluate_model(v5_model, V5_TEST_BIN, v5_cfg.max_seq_len, device)
    
    print("\nGenerating outputs and computing quality metrics...")
    
    v4_metrics_list = []
    v5_metrics_list = []
    
    v4_speeds = []
    v5_speeds = []
    
    print("\n" + "="*80)
    print("PROMPT COMPARISONS")
    print("="*80)
    
    for idx, prompt in enumerate(prompts):
        print(f"\nPROMPT {idx+1}: {prompt}")
        print("-" * 50)
        
        # V4 generation
        v4_ids, v4_text, v4_speed = generate_sample(v4_model, tokenizer, prompt, device, max_tokens=50)
        v4_prompt_len = len(tokenizer.encode(prompt, bos=True))
        v4_m = calculate_quality_metrics(v4_ids, v4_prompt_len, tokenizer)
        v4_metrics_list.append(v4_m)
        v4_speeds.append(v4_speed)
        print(f"[V4 (Dataset v4)]: {v4_text.strip()}")
        print(f"  Speed: {v4_speed:.1f} tok/s | Repetition: {v4_m['repetition_rate']:.1%} | Terminated: {v4_m['terminated']}")
        
        # V5 generation
        v5_ids, v5_text, v5_speed = generate_sample(v5_model, tokenizer, prompt, device, max_tokens=50)
        v5_prompt_len = len(tokenizer.encode(prompt, bos=True))
        v5_m = calculate_quality_metrics(v5_ids, v5_prompt_len, tokenizer)
        v5_metrics_list.append(v5_m)
        v5_speeds.append(v5_speed)
        print(f"[V5 (Dataset v5-expanded)]: {v5_text.strip()}")
        print(f"  Speed: {v5_speed:.1f} tok/s | Repetition: {v5_m['repetition_rate']:.1%} | Terminated: {v5_m['terminated']}")
        
    # Aggregate quality metrics
    def avg_metric(m_list, key):
        vals = [m[key] for m in m_list]
        return np.mean(vals)
        
    print("\n" + "="*80)
    print("SUMMARY COMPARISON TABLE")
    print("="*80)
    
    print(f"| Metric | V4 Model (Dataset v4) | V5 Model (Dataset v5-exp) |")
    print(f"|---|---|---|")
    print(f"| Dataset Version | collision_dataset_v4 | collision_dataset_v5_expanded |")
    print(f"| Parameter Count | {v4_param_count:,} | {v5_param_count:,} |")
    print(f"| Validation Loss | {v4_val_loss:.4f} | {v5_val_loss:.4f} |")
    print(f"| Validation Perplexity | {v4_val_perp:.2f} | {v5_val_perp:.2f} |")
    # Clarify test loss evaluation
    print(f"| Test Loss (V5 Test split) | {v4_test_loss:.4f} (Zero-shot) | {v5_test_loss:.4f} |")
    print(f"| Test Perplexity (V5 Test split) | {v4_test_perp:.2f} (Zero-shot) | {v5_test_perp:.2f} |")
    print(f"| Avg Repetition Rate | {avg_metric(v4_metrics_list, 'repetition_rate'):.1%} | {avg_metric(v5_metrics_list, 'repetition_rate'):.1%} |")
    print(f"| Avg Unique Token Ratio | {avg_metric(v4_metrics_list, 'unique_token_ratio'):.1%} | {avg_metric(v5_metrics_list, 'unique_token_ratio'):.1%} |")
    print(f"| Avg Repeated 2-grams | {avg_metric(v4_metrics_list, 'repeated_2grams'):.1f} | {avg_metric(v5_metrics_list, 'repeated_2grams'):.1f} |")
    print(f"| Avg Repeated 3-grams | {avg_metric(v4_metrics_list, 'repeated_3grams'):.1f} | {avg_metric(v5_metrics_list, 'repeated_3grams'):.1f} |")
    print(f"| Avg Generated Length | {avg_metric(v4_metrics_list, 'length'):.1f} tokens | {avg_metric(v5_metrics_list, 'length'):.1f} tokens |")
    print(f"| UNK Token Frequency | {avg_metric(v4_metrics_list, 'unk_frequency'):.4f} | {avg_metric(v5_metrics_list, 'unk_frequency'):.4f} |")
    print(f"| Sentence Termination Rate | {avg_metric(v4_metrics_list, 'terminated'):.1%} | {avg_metric(v5_metrics_list, 'terminated'):.1%} |")
    print(f"| Avg Generation Speed | {np.mean(v4_speeds):.1f} tok/s | {np.mean(v5_speeds):.1f} tok/s |")
    print("="*80)

if __name__ == "__main__":
    main()
