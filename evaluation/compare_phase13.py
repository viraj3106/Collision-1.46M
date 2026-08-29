import os
import sys
import time
import torch
import numpy as np
import torch.nn as nn

# Resolve project root path and insert into Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model.config import ModelConfig
from model.transformer import CollisionTransformer
from data.tokenize import BPETokenizer

# Paths
BASE_CP_PATH = "checkpoints/phase12b/collision-3.38m-phase12b-best.pt"
INSTRUCT_CP_PATH = "checkpoints/phase13/collision-instruct-3.37m-best.pt"
TOKENIZER_DIR = "artifacts/tokenizer"
PROMPTS_FILE = "evaluation/phase13_prompts.txt"
OUTPUT_FILE = "experiments/phase13/generation_comparison.txt"

def generate_base(model, tokenizer, prompt, device, max_tokens=100, temp=0.7, top_k=50):
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

def generate_instruct(model, tokenizer, prompt, device, max_tokens=100, temp=0.7, top_k=50):
    model.eval()
    formatted = f"<|user|>\n{prompt}\n\n<|assistant|>\n"
    ids = tokenizer.encode(formatted, bos=True)
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
    print("Starting Phase 13 Base vs Instruct Comparison...")
    
    # Load tokenizer
    tokenizer = BPETokenizer()
    tokenizer.load(TOKENIZER_DIR)
    
    # Load prompts
    if not os.path.exists(PROMPTS_FILE):
        print(f"Error: Prompts file not found at {PROMPTS_FILE}")
        return
        
    with open(PROMPTS_FILE, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]
        
    # Load Base model
    print(f"Loading Base model from {BASE_CP_PATH}...")
    base_cp = torch.load(BASE_CP_PATH, map_location=device)
    base_cfg = ModelConfig(**base_cp["config"])
    base_model = CollisionTransformer(base_cfg).to(device)
    base_model.load_state_dict(base_cp["model_state_dict"])
    
    # Load Instruct model
    print(f"Loading Instruct model from {INSTRUCT_CP_PATH}...")
    instruct_cp = torch.load(INSTRUCT_CP_PATH, map_location=device)
    instruct_cfg = ModelConfig(**instruct_cp["config"])
    instruct_model = CollisionTransformer(instruct_cfg).to(device)
    instruct_model.load_state_dict(instruct_cp["model_state_dict"])
    
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    base_metrics = []
    instruct_metrics = []
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("COLLISION Phase 13 — Base vs Instruct Generation Comparison\n")
        f.write("============================================================\n\n")
        
        for idx, prompt in enumerate(prompts):
            print(f"Evaluating prompt {idx+1}/{len(prompts)}...")
            f.write(f"PROMPT {idx+1}: {prompt}\n")
            f.write("-" * 50 + "\n")
            
            # Base generation
            base_ids, base_text = generate_base(base_model, tokenizer, prompt, device)
            base_m = calculate_quality_metrics(base_ids, len(tokenizer.encode(prompt, bos=True)), tokenizer)
            base_metrics.append(base_m)
            f.write(f"BASE model response:\n  {base_text.strip()}\n")
            f.write(f"  Repetition Rate: {base_m['repetition_rate']:.1%} | Unique Ratio: {base_m['unique_token_ratio']:.1%} | Terminated: {base_m['terminated']}\n\n")
            
            # Instruct generation
            inst_ids, inst_text = generate_instruct(instruct_model, tokenizer, prompt, device)
            inst_m = calculate_quality_metrics(inst_ids, len(tokenizer.encode(f"<|user|>\n{prompt}\n\n<|assistant|>\n", bos=True)), tokenizer)
            instruct_metrics.append(inst_m)
            f.write(f"INSTRUCT model response:\n  {inst_text.strip()}\n")
            f.write(f"  Repetition Rate: {inst_m['repetition_rate']:.1%} | Unique Ratio: {inst_m['unique_token_ratio']:.1%} | Terminated: {inst_m['terminated']}\n")
            f.write("=" * 80 + "\n\n")
            
    # Aggregation
    def get_avg(metrics_list, key):
        return np.mean([m[key] for m in metrics_list])
        
    print("\n==================================================")
    print("AGGREGATED QUALITY METRICS SUMMARY")
    print("==================================================")
    print(f"| Metric | Base Model (V5) | Instruct Model (V5-SFT) |")
    print(f"|---|---|---|")
    print(f"| Avg Repetition Rate | {get_avg(base_metrics, 'repetition_rate'):.1%} | {get_avg(instruct_metrics, 'repetition_rate'):.1%} |")
    print(f"| Avg Unique Token Ratio | {get_avg(base_metrics, 'unique_token_ratio'):.1%} | {get_avg(instruct_metrics, 'unique_token_ratio'):.1%} |")
    print(f"| Avg Repeated 2-grams | {get_avg(base_metrics, 'repeated_2grams'):.1f} | {get_avg(instruct_metrics, 'repeated_2grams'):.1f} |")
    print(f"| Avg Repeated 3-grams | {get_avg(base_metrics, 'repeated_3grams'):.1f} | {get_avg(instruct_metrics, 'repeated_3grams'):.1f} |")
    print(f"| Avg Response Length | {get_avg(base_metrics, 'length'):.1f} tokens | {get_avg(instruct_metrics, 'length'):.1f} tokens |")
    print(f"| Sentence Termination Rate | {get_avg(base_metrics, 'terminated'):.1%} | {get_avg(instruct_metrics, 'terminated'):.1%} |")
    print("==================================================")

if __name__ == "__main__":
    main()
