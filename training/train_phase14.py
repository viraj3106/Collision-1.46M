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
import matplotlib.pyplot as plt

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
EXP_DIR = "experiments/phase14"
CP_DIR = "checkpoints/phase14"
TOKENIZER_DIR = "artifacts/tokenizer"
DATASET_DIR = "datasets/collision_dataset_v5_expanded"

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

def run_evaluation(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    steps = 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            total_loss += loss.item()
            steps += 1
            if steps >= 50: # Cap at 50 batches for quick val
                break
    mean_loss = total_loss / max(1, steps)
    perplexity = np.exp(mean_loss) if mean_loss < 20 else float('inf')
    return mean_loss, perplexity

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

def main():
    parser = argparse.ArgumentParser(description="COLLISION-10M pretraining")
    parser.add_argument("--benchmark", action="store_true", help="Perform a short CPU benchmark and report resource usage estimates")
    parser.add_argument("--train", action="store_true", help="Perform actual training")
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
    
    # Random weight initialization (from scratch)
    print("Initializing COLLISION-10M from random weights...")
    model = CollisionTransformer(model_cfg).to(device)
    param_count = model.get_parameter_count()
    
    print(f"========================================")
    print(f"COLLISION-10M PARAMETERS: {param_count:,}")
    print(f"========================================")
    
    # Datasets
    train_bin = os.path.join(DATASET_DIR, "train.bin")
    val_bin = os.path.join(DATASET_DIR, "val.bin")
    
    train_dataset = TokenDataset(train_bin, model_cfg.max_seq_len)
    val_dataset = TokenDataset(val_bin, model_cfg.max_seq_len)
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    # Optimizer & Scheduler
    learning_rate = 6e-4
    min_lr = 6e-5
    warmup_steps = 150
    weight_decay = 0.01
    grad_accumulation = 4
    grad_clipping = 1.0
    max_steps = 1500

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = CosineWarmupScheduler(
        optimizer, 
        warmup_steps=warmup_steps, 
        total_steps=max_steps, 
        base_lr=learning_rate, 
        min_lr=min_lr
    )

    # 2. Benchmarking Mode
    if args.benchmark or not args.train:
        print("\nRunning a short 20-step CPU benchmark...")
        model.train()
        optimizer.zero_grad()
        
        start_time = time.time()
        step = 0
        memory_usages = []
        
        for x, y in train_loader:
            if step >= 20:
                break
            x, y = x.to(device), y.to(device)
            logits, loss = model(x, y)
            loss_scaled = loss / grad_accumulation
            loss_scaled.backward()
            
            if (step + 1) % grad_accumulation == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clipping)
                optimizer.step()
                optimizer.zero_grad()
                
            memory_usages.append(get_process_memory())
            step += 1
            
        elapsed = time.time() - start_time
        steps_per_sec = step / elapsed
        tokens_per_sec = (4 * model_cfg.max_seq_len) * steps_per_sec
        steps_per_hour = steps_per_sec * 3600
        est_total_hours = max_steps / steps_per_hour
        avg_mem = np.mean(memory_usages)
        
        print("\n========================================")
        print("          CPU BENCHMARK REPORT")
        print("========================================")
        print(f"Throughput: {tokens_per_sec:.2f} tokens/second")
        print(f"Processing speed: {steps_per_hour:.2f} steps/hour")
        print(f"Memory usage: {avg_mem:.2f} MB (peak: {max(memory_usages):.2f} MB)")
        print(f"Estimated 1500-step training time: {est_total_hours:.2f} hours ({est_total_hours * 60:.1f} minutes)")
        print("========================================\n")
        
        # Save estimates to config config.json for verification
        est_config = {
            "tokens_per_sec_estimate": tokens_per_sec,
            "steps_per_hour_estimate": steps_per_hour,
            "memory_usage_mb_estimate": avg_mem,
            "estimated_total_training_minutes": est_total_hours * 60,
            "parameter_count": param_count
        }
        with open(os.path.join(EXP_DIR, "benchmark_estimates.json"), "w", encoding="utf-8") as f:
            json.dump(est_config, f, indent=2)
            
        # Return and exit if ONLY benchmark was requested
        if args.benchmark and not args.train:
            return

    # 3. Actual Training Loop
    if args.train:
        # Check if checkpoint exists
        final_checkpoint_path = os.path.join(CP_DIR, "collision-10m-step-001500.pt")
        if os.path.exists(final_checkpoint_path):
            print("Final checkpoint already exists. Skipping training run...")
            return
            
        print(f"Starting {max_steps} steps of 10M pretraining on CPU...")
        log_csv_path = os.path.join(EXP_DIR, "training_log.csv")
        with open(log_csv_path, "w", encoding="utf-8") as f:
            f.write("step,train_loss,validation_loss,validation_perplexity,learning_rate,tokens_per_second,elapsed_time,cpu_memory\n")

        model.train()
        optimizer.zero_grad()
        
        step = 0
        running_loss = 0.0
        start_time = time.time()
        total_start_time = time.time()
        best_val_loss = float('inf')
        best_checkpoint_path = ""
        best_step = 0
        
        history = {
            "steps": [],
            "train_loss": [],
            "val_loss": []
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
                    
                if step % 500 == 0:
                    val_loss, perp = run_evaluation(model, val_loader, device)
                    avg_train_loss_rec = (running_loss / (step % 10 if step % 10 != 0 else 10)) if step % 10 != 0 else avg_train_loss
                    elapsed_total = time.time() - total_start_time
                    cpu_mem = get_process_memory()
                    
                    history["steps"].append(step)
                    history["train_loss"].append(avg_train_loss_rec)
                    history["val_loss"].append(val_loss)
                    
                    with open(log_csv_path, "a", encoding="utf-8") as f:
                        f.write(f"{step},{avg_train_loss_rec:.4f},{val_loss:.4f},{perp:.2f},{current_lr:.6f},{tokens_per_sec:.1f},{elapsed_total:.1f},{cpu_mem:.1f}\n")
                        
                    print(f"\n--- VALIDATION --- Step {step} | Train Loss: {avg_train_loss_rec:.4f} | Val Loss: {val_loss:.4f} | Perplexity: {perp:.2f}\n")
                    
                    # Save checkpoint
                    cp_name = f"collision-10m-step-{step:06d}.pt"
                    cp_path = os.path.join(CP_DIR, cp_name)
                    save_checkpoint(
                        model, optimizer, scheduler, step, 0, avg_train_loss_rec, val_loss,
                        model_cfg.__dict__, {"save_dir": TOKENIZER_DIR}, cp_path
                    )
                    
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_step = step
                        best_checkpoint_path = os.path.join(CP_DIR, "collision-10m-best.pt")
                        save_checkpoint(
                            model, optimizer, scheduler, step, 0, avg_train_loss_rec, val_loss,
                            model_cfg.__dict__, {"save_dir": TOKENIZER_DIR}, best_checkpoint_path
                        )
                    
                    model.train()
                    start_time = time.time()
                    
        # Save final checkpoint
        final_checkpoint_path = os.path.join(CP_DIR, f"collision-10m-step-{step:06d}.pt")
        if not os.path.exists(final_checkpoint_path):
            save_checkpoint(
                model, optimizer, scheduler, step, 0, avg_train_loss_rec, val_loss,
                model_cfg.__dict__, {"save_dir": TOKENIZER_DIR}, final_checkpoint_path
            )

        print("\nTraining completed successfully!")

if __name__ == "__main__":
    main()
