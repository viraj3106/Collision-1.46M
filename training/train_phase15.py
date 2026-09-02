import os
import time
import json
import psutil
import sys
import argparse
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# Resolve project root path and insert into Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model.config import ModelConfig
from model.transformer import CollisionTransformer
from data.tokenize import BPETokenizer
from training.checkpoint import save_checkpoint
from training.scheduler import CosineWarmupScheduler

# Paths
CONFIG_YAML_PATH = "configs/collision_10m.yaml"
EXP_DIR = "experiments/phase15"
CP_DIR = "checkpoints/phase15"
TOKENIZER_DIR = "artifacts/tokenizer"
DATASET_DIR = "datasets/collision_dataset_v5_expanded"
TEST_BIN = os.path.join(DATASET_DIR, "test.bin")

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

class NonOverlappingTokenDataset(Dataset):
    def __init__(self, bin_path: str, seq_len: int):
        if not os.path.exists(bin_path):
            raise FileNotFoundError(f"Binary token file not found at {bin_path}.")
        self.data = np.fromfile(bin_path, dtype=np.uint16)
        self.seq_len = seq_len

    def __len__(self):
        return max(0, (len(self.data) - 1) // self.seq_len)

    def __getitem__(self, idx):
        start = idx * self.seq_len
        x = torch.from_numpy(self.data[start : start + self.seq_len].astype(np.int64))
        y = torch.from_numpy(self.data[start + 1 : start + self.seq_len + 1].astype(np.int64))
        return x, y

def get_process_memory():
    try:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024) # MB
    except Exception:
        return 0.0

def evaluate_full_dataset(model, bin_path, seq_len, device):
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

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
        
    return logits

def generate_sample(model, tokenizer, prompt, device, max_tokens=100, temp=0.8, top_k=50, top_p=0.9):
    model.eval()
    ids = tokenizer.encode(prompt, bos=True)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    
    start_time = time.time()
    tokens_generated = 0
    
    with torch.no_grad():
        for _ in range(max_tokens):
            x_cond = x if x.size(1) <= model.config.max_seq_len else x[:, -model.config.max_seq_len:]
            logits, _ = model(x_cond)
            next_token_logits = logits[0, -1, :]
            
            if temp > 0.0:
                next_token_logits = next_token_logits / temp
                filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
                probs = F.softmax(filtered_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits).unsqueeze(0)
                
            x = torch.cat((x, next_token.unsqueeze(0)), dim=1)
            tokens_generated += 1
            if next_token.item() == tokenizer.special_tokens.get("[EOS]", 259):
                break
                
    elapsed = time.time() - start_time
    latency = elapsed * 1000 # ms
    tok_per_sec = tokens_generated / max(0.0001, elapsed)
    return x[0].tolist(), tokenizer.decode(x[0].tolist()), tok_per_sec, latency

def calculate_quality_metrics(token_ids, prompt_len, tokenizer):
    gen_ids = token_ids[prompt_len:]
    if len(gen_ids) == 0:
        return {
            "repetition_rate": 0.0,
            "unique_token_ratio": 0.0,
            "repeated_unigram_ratio": 0.0,
            "repeated_bigram_ratio": 0.0,
            "repeated_trigram_ratio": 0.0,
            "length": 0,
            "terminated": False
        }
    
    unique_tokens = set(gen_ids)
    unique_ratio = len(unique_tokens) / len(gen_ids)
    repetition_rate = 1.0 - unique_ratio
    
    n2_grams = list(zip(gen_ids[:-1], gen_ids[1:]))
    n2_repeats = len(n2_grams) - len(set(n2_grams))
    n2_ratio = n2_repeats / len(n2_grams) if len(n2_grams) > 0 else 0.0
    
    n3_grams = list(zip(gen_ids[:-2], gen_ids[1:-1], gen_ids[2:]))
    n3_repeats = len(n3_grams) - len(set(n3_grams))
    n3_ratio = n3_repeats / len(n3_grams) if len(n3_grams) > 0 else 0.0
    
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
        "repeated_unigram_ratio": repetition_rate,
        "repeated_bigram_ratio": n2_ratio,
        "repeated_trigram_ratio": n3_ratio,
        "length": len(gen_ids),
        "terminated": terminated
    }

def run_audit_on_checkpoint(checkpoint_path, tokenizer, device, model_cfg):
    print(f"Running generation audit on {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = CollisionTransformer(model_cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    prompts = [
        "What is artificial intelligence?",
        "Computer science is",
        "The future of technology",
        "An algorithm is",
        "Space exploration",
        "Why does the Earth orbit the Sun?",
        "Machine learning is",
        "Photosynthesis is"
    ]
    
    audit_results = []
    for prompt in prompts:
        prompt_len = len(tokenizer.encode(prompt, bos=True))
        ids, text, speed, latency = generate_sample(
            model, tokenizer, prompt, device, max_tokens=100, temp=0.8, top_k=50, top_p=0.9
        )
        metrics = calculate_quality_metrics(ids, prompt_len, tokenizer)
        metrics["speed"] = speed
        metrics["latency"] = latency
        metrics["prompt"] = prompt
        metrics["output"] = text
        audit_results.append(metrics)
        
    return audit_results

def main():
    parser = argparse.ArgumentParser(description="COLLISION-10M production pretraining")
    parser.add_argument("--benchmark", action="store_true", help="Perform CPU benchmark and exit")
    parser.add_argument("--train", action="store_true", help="Perform production training run")
    parser.add_argument("--audit", action="store_true", help="Perform audits on existing checkpoints")
    args = parser.parse_args()

    os.makedirs(EXP_DIR, exist_ok=True)
    os.makedirs(CP_DIR, exist_ok=True)

    # 1. Load config
    if not os.path.exists(CONFIG_YAML_PATH):
        raise FileNotFoundError(f"Config file not found at {CONFIG_YAML_PATH}")
    with open(CONFIG_YAML_PATH, "r") as f:
        config_data = yaml.safe_load(f)
        
    import inspect
    sig = inspect.signature(ModelConfig)
    valid_keys = set(sig.parameters.keys())
    filtered_model_data = {k: v for k, v in config_data["model"].items() if k in valid_keys}
    
    model_cfg = ModelConfig(**filtered_model_data)
    device = torch.device("cpu")
    
    # Random weights init
    model = CollisionTransformer(model_cfg).to(device)
    param_count = model.get_parameter_count()
    
    print(f"========================================")
    print(f"COLLISION-10M PARAMETERS: {param_count:,}")
    print(f"========================================")
    
    tokenizer = BPETokenizer()
    tokenizer.load(TOKENIZER_DIR)

    # Datasets
    train_bin = os.path.join(DATASET_DIR, "train.bin")
    val_bin = os.path.join(DATASET_DIR, "val.bin")
    
    # 2. Benchmark Mode
    if args.benchmark:
        print("\nRunning CPU training benchmark...")
        train_dataset = TokenDataset(train_bin, model_cfg.max_seq_len)
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, drop_last=True)
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4)
        optimizer.zero_grad()
        
        start_time = time.time()
        step = 0
        memory_usages = []
        
        for x, y in train_loader:
            if step >= 20:
                break
            x, y = x.to(device), y.to(device)
            logits, loss = model(x, y)
            loss_scaled = loss / 4
            loss_scaled.backward()
            
            if (step + 1) % 4 == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                
            memory_usages.append(get_process_memory())
            step += 1
            
        elapsed = time.time() - start_time
        steps_per_sec = step / elapsed
        tokens_per_sec = (4 * model_cfg.max_seq_len) * steps_per_sec
        avg_mem = np.mean(memory_usages)
        
        print("\n========================================")
        print("          CPU BENCHMARK REPORT")
        print("========================================")
        print(f"Throughput: {tokens_per_sec:.2f} tokens/second")
        print(f"Memory usage: {avg_mem:.2f} MB (peak: {max(memory_usages):.2f} MB)")
        print("========================================\n")
        
        # Save estimate info
        with open(os.path.join(EXP_DIR, "benchmark_estimates.json"), "w", encoding="utf-8") as f:
            json.dump({
                "tokens_per_sec": tokens_per_sec,
                "memory_mb": avg_mem,
                "parameter_count": param_count
            }, f, indent=2)
        return

    # 3. Training Mode
    if args.train:
        train_dataset = TokenDataset(train_bin, model_cfg.max_seq_len)
        val_dataset = TokenDataset(val_bin, model_cfg.max_seq_len)
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, drop_last=True)
        
        learning_rate = 6e-4
        min_lr = 6e-5
        warmup_steps = 150
        weight_decay = 0.01
        grad_accumulation = 4
        grad_clipping = 1.0
        max_steps = 9766  # 10,000,384 tokens
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = CosineWarmupScheduler(
            optimizer, 
            warmup_steps=warmup_steps, 
            total_steps=max_steps, 
            base_lr=learning_rate, 
            min_lr=min_lr
        )
        
        print(f"Starting {max_steps} steps of 10M pretraining on CPU...")
        log_csv_path = os.path.join(EXP_DIR, "training_log.csv")
        with open(log_csv_path, "w", encoding="utf-8") as f:
            f.write("step,processed_tokens,train_loss,val_loss,val_perplexity,learning_rate,tokens_per_second,elapsed_time,cpu_memory\n")
            
        model.train()
        optimizer.zero_grad()
        
        step = 0
        running_loss = 0.0
        start_time = time.time()
        total_start_time = time.time()
        best_val_loss = float('inf')
        best_checkpoint_path = ""
        best_step = 0
        
        # Exact token schedule steps
        checkpoint_steps = {
            1465: "collision-10m-step-001465.pt", # 1.5M tokens
            2930: "collision-10m-step-002930.pt", # 3.0M tokens
            4883: "collision-10m-step-004883.pt", # 5.0M tokens
            7324: "collision-10m-step-007324.pt", # 7.5M tokens
            9766: "collision-10m-step-009766.pt"  # 10.0M tokens
        }
        
        finished_training = False
        while not finished_training:
            for x, y in train_loader:
                if step >= max_steps:
                    finished_training = True
                    break
                    
                x, y = x.to(device), y.to(device)
                logits, loss = model(x, y)
                loss_scaled = loss / grad_accumulation
                loss_scaled.backward()
                running_loss += loss.item()
                
                if (step + 1) % grad_accumulation == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clipping)
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    
                step += 1
                
                # Periodically print stats and write training loss
                if step % 10 == 0:
                    avg_train_loss = running_loss / 10
                    running_loss = 0.0
                    elapsed = time.time() - start_time
                    steps_per_sec = 10 / elapsed
                    tokens_per_sec = (4 * model_cfg.max_seq_len) * steps_per_sec
                    cpu_mem = get_process_memory()
                    current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else optimizer.param_groups[0]['lr']
                    
                    print(f"Step {step}/{max_steps} | Train Loss: {avg_train_loss:.4f} | LR: {current_lr:.6f} | Speed: {tokens_per_sec:.1f} tok/s | Mem: {cpu_mem:.1f} MB")
                    start_time = time.time()
                
                # Periodically evaluate validation loss (every 500 steps and at exact checkpoints)
                if step % 500 == 0 or step in checkpoint_steps:
                    val_loss, perp = evaluate_full_dataset(model, os.path.join(DATASET_DIR, "val.bin"), model_cfg.max_seq_len, device)
                    avg_train_loss_rec = (running_loss / (step % 10 if step % 10 != 0 else 10)) if step % 10 != 0 else avg_train_loss
                    elapsed_total = time.time() - total_start_time
                    cpu_mem = get_process_memory()
                    processed_tokens = step * 4 * model_cfg.max_seq_len
                    
                    with open(log_csv_path, "a", encoding="utf-8") as f:
                        f.write(f"{step},{processed_tokens},{avg_train_loss_rec:.4f},{val_loss:.4f},{perp:.2f},{current_lr:.6f},{tokens_per_sec:.1f},{elapsed_total:.1f},{cpu_mem:.1f}\n")
                        
                    print(f"\n--- VALIDATION --- Step {step} | Train Loss: {avg_train_loss_rec:.4f} | Val Loss: {val_loss:.4f} | Perplexity: {perp:.2f}\n")
                    
                    # If this is one of our scheduled checkpoint steps, save it
                    if step in checkpoint_steps:
                        cp_name = checkpoint_steps[step]
                        cp_path = os.path.join(CP_DIR, cp_name)
                        save_checkpoint(
                            model, optimizer, scheduler, step, 0, avg_train_loss_rec, val_loss,
                            model_cfg.__dict__, {"save_dir": TOKENIZER_DIR}, cp_path
                        )
                        print(f"Saved scheduled checkpoint: {cp_path}")
                        
                    # Save best validation loss checkpoint
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_step = step
                        best_checkpoint_path = os.path.join(CP_DIR, "collision-10m-best.pt")
                        save_checkpoint(
                            model, optimizer, scheduler, step, 0, avg_train_loss_rec, val_loss,
                            model_cfg.__dict__, {"save_dir": TOKENIZER_DIR}, best_checkpoint_path
                        )
                        print(f"Saved new best checkpoint: {best_checkpoint_path}")
                        
                    model.train()
                    start_time = time.time()
                    
        # Make sure final checkpoint is saved
        final_checkpoint_path = os.path.join(CP_DIR, "collision-10m-step-009766.pt")
        if not os.path.exists(final_checkpoint_path):
            save_checkpoint(
                model, optimizer, scheduler, step, 0, avg_train_loss_rec, val_loss,
                model_cfg.__dict__, {"save_dir": TOKENIZER_DIR}, final_checkpoint_path
            )
            print(f"Saved final checkpoint: {final_checkpoint_path}")
            
        print("\nTraining completed successfully!")
        
        # Run generation audits on all checkpoints
        print("\nStarting generation quality audits...")
        audit_results = {}
        for step_val, cp_name in checkpoint_steps.items():
            cp_path = os.path.join(CP_DIR, cp_name)
            res = run_audit_on_checkpoint(cp_path, tokenizer, device, model_cfg)
            audit_results[f"{step_val/1000:.1f}M"] = res
            
        # Audit the BEST checkpoint
        res_best = run_audit_on_checkpoint(os.path.join(CP_DIR, "collision-10m-best.pt"), tokenizer, device, model_cfg)
        audit_results["BEST"] = res_best
        
        with open(os.path.join(EXP_DIR, "audit_results.json"), "w", encoding="utf-8") as f:
            json.dump(audit_results, f, indent=2)
        print("Generation audit completed and saved.")
        
        # Evaluate best checkpoint on test split
        print("\nEvaluating best model on test split...")
        best_model = CollisionTransformer(model_cfg).to(device)
        best_cp = torch.load(os.path.join(CP_DIR, "collision-10m-best.pt"), map_location=device)
        best_model.load_state_dict(best_cp["model_state_dict"])
        test_loss, test_ppx = evaluate_full_dataset(best_model, TEST_BIN, model_cfg.max_seq_len, device)
        
        # Save final metrics
        metrics = {
            "best_step": best_step,
            "best_val_loss": best_val_loss,
            "best_val_ppx": np.exp(best_val_loss),
            "test_loss": test_loss,
            "test_ppx": test_ppx,
            "total_tokens_processed": max_steps * 4 * model_cfg.max_seq_len
        }
        with open(os.path.join(EXP_DIR, "final_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
            
        print("\n========================================")
        print("            TRAINING RESULTS")
        print("========================================")
        print(f"Total tokens processed: {metrics['total_tokens_processed']:,}")
        print(f"Best checkpoint step: {best_step}")
        print(f"Best validation loss: {best_val_loss:.4f}")
        print(f"Best validation perplexity: {metrics['best_val_ppx']:.2f}")
        print(f"Test loss: {test_loss:.4f}")
        print(f"Test perplexity: {test_ppx:.2f}")
        print("========================================")

if __name__ == "__main__":
    main()
