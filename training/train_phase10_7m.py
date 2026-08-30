import os
import time
import json
import psutil
import yaml
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import platform
import hashlib

# Resolve project root path and insert into Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model.config import ModelConfig
from model.transformer import CollisionTransformer
from data.tokenize import BPETokenizer
from training.checkpoint import save_checkpoint
from training.scheduler import CosineWarmupScheduler
from data.stats import get_latest_version_dir
from inference.generate import top_k_top_p_filtering

CONFIG_PATH = "configs/scaling/collision_7m.yaml"
EXP_DIR = "experiments/scaling/collision_7m"
CP_DIR = "checkpoints/scaling/collision_7m"
TOKENIZER_DIR = "artifacts/tokenizer"

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
            if steps >= 50:
                break
    mean_loss = total_loss / max(1, steps)
    perplexity = np.exp(mean_loss) if mean_loss < 20 else float('inf')
    return mean_loss, perplexity

def generate_sample(model, tokenizer, prompt, device, max_tokens=50):
    model.eval()
    ids = tokenizer.encode(prompt, bos=True)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    
    with torch.no_grad():
        for _ in range(max_tokens):
            x_cond = x if x.size(1) <= model.config.max_seq_len else x[:, -model.config.max_seq_len:]
            logits, _ = model(x_cond)
            next_token_logits = logits[0, -1, :] / 0.8  # temp=0.8
            v, _ = torch.topk(next_token_logits, min(50, next_token_logits.size(-1)))
            next_token_logits[next_token_logits < v[-1]] = -float('Inf')
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            x = torch.cat((x, next_token.unsqueeze(0)), dim=1)
            if next_token.item() == tokenizer.special_tokens.get("[EOS]", 2):
                break
                
    return tokenizer.decode(x[0].tolist())

def run_benchmark(model, tokenizer, device):
    prompts = [
        "What is artificial intelligence?",
        "Computer science is",
        "The future of technology",
        "An algorithm is",
        "Space exploration"
    ]
    
    results = []
    print("\nRunning final generation benchmark...")
    for prompt in prompts:
        start_time = time.perf_counter()
        # Generate exactly 50 tokens
        gen_text = generate_sample(model, tokenizer, prompt, device, max_tokens=50)
        end_time = time.perf_counter()
        
        prompt_tokens = len(tokenizer.encode(prompt, bos=True))
        output_tokens = len(tokenizer.encode(gen_text, bos=True))
        tokens_gen = output_tokens - prompt_tokens
        
        gen_duration = end_time - start_time
        tok_per_sec = tokens_gen / gen_duration if gen_duration > 0 else 0
        
        results.append({
            "prompt": prompt,
            "generated_text": gen_text,
            "tokens_generated": tokens_gen,
            "generation_time": gen_duration,
            "tokens_per_second": tok_per_sec
        })
        print(f"Prompt: '{prompt}' | Speed: {tok_per_sec:.2f} tok/s")
    return results

def main():
    start_time_str = time.strftime("%Y-%m-%d %H:%M:%S")
    start_epoch_time = time.time()
    
    os.makedirs(EXP_DIR, exist_ok=True)
    os.makedirs(CP_DIR, exist_ok=True)

    # 1. Load config
    with open(CONFIG_PATH, "r") as f:
        config_yaml = yaml.safe_load(f)
        
    config_hash = hashlib.sha256(open(CONFIG_PATH, 'rb').read()).hexdigest()
    model_config = ModelConfig(**config_yaml["model"])
    
    # Seeds
    seed = 1337
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cpu")

    # Load tokenizer
    tokenizer = BPETokenizer()
    tokenizer.load(TOKENIZER_DIR)

    # Load dataset info
    dataset_version = config_yaml["training"]["dataset_version"]
    latest_dir = os.path.join("datasets", dataset_version)
    train_bin = os.path.join(latest_dir, "train.bin")
    val_bin = os.path.join(latest_dir, "val.bin")
    
    with open(os.path.join(latest_dir, "metadata.json"), "r") as f:
        meta_data = json.load(f)

    # Initialize model from scratch
    model = CollisionTransformer(model_config).to(device)
    param_count = sum(p.numel() for p in model.parameters())

    print(f"""========================================
COLLISION-7M PHASE 10D TRAINING RUN
========================================
Dataset: {meta_data.get('dataset_version')}
Parameters: {param_count:,}
Device: CPU
Maximum steps: 1,500
========================================
""")

    # Save experiments/scaling/collision_7m/config.json
    exp_config = {
        "experiment_name": "collision_7m_scaling",
        "dataset_version": meta_data.get("dataset_version"),
        "tokenizer_version": meta_data.get("tokenizer_version", "1.0-BPETokenizer"),
        "model_configuration": model_config.__dict__,
        "random_seed": seed,
        "optimizer": "AdamW",
        "learning_rate": 6e-4,
        "batch_size": 4,
        "gradient_accumulation": 4,
        "context_length": 256,
        "device": "cpu",
        "maximum_steps": 1500
    }
    with open(os.path.join(EXP_DIR, "config.json"), "w", encoding="utf-8") as f:
        json.dump(exp_config, f, indent=2)

    # Save initial checkpoint
    initial_cp_path = os.path.join(CP_DIR, "collision-7m-initial.pt")
    save_checkpoint(
        model, None, None, 0, 0, 0.0, 0.0,
        model_config.__dict__, {"save_dir": TOKENIZER_DIR}, initial_cp_path
    )

    # Dataloaders
    train_dataset = TokenDataset(train_bin, model_config.max_seq_len)
    val_dataset = TokenDataset(val_bin, model_config.max_seq_len)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, weight_decay=0.01)
    scheduler = CosineWarmupScheduler(optimizer, warmup_steps=150, total_steps=1500, base_lr=6e-4, min_lr=6e-5)

    # CSV Log
    log_csv_path = os.path.join(EXP_DIR, "training_log.csv")
    with open(log_csv_path, "w", encoding="utf-8") as f:
        f.write("step,train_loss,val_loss,perplexity,lr,tokens_processed,tokens_per_sec,step_per_sec,elapsed_time,cpu_memory\n")

    # Initial Val evaluation
    init_val_loss, init_perp = run_evaluation(model, val_loader, device)
    print(f"Baseline Validation Loss: {init_val_loss:.4f} | Perplexity: {init_perp:.2f}")

    # Training variables
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

    print("Starting 1,500 steps CPU training...")
    model.train()
    optimizer.zero_grad()
    
    finished_training = False
    
    while not finished_training:
        for x, y in train_loader:
            if step >= 1500:
                finished_training = True
                break
                
            x, y = x.to(device), y.to(device)
            logits, loss = model(x, y)
            
            # Gradient Accumulation
            loss_scaled = loss / 4
            loss_scaled.backward()
            running_loss += loss.item()
            
            if (step + 1) % 4 == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                
            step += 1
            
            # Print/Log step details every 10 steps
            if step % 10 == 0:
                avg_train_loss = running_loss / 10
                running_loss = 0.0
                
                elapsed = time.time() - start_time
                steps_per_sec = 10 / elapsed
                tokens_per_sec = (4 * 256) * steps_per_sec
                elapsed_total = time.time() - total_start_time
                
                cpu_mem = get_process_memory()
                current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else optimizer.param_groups[0]['lr']
                
                print(f"Step {step}/1500 | Train Loss: {avg_train_loss:.4f} | LR: {current_lr:.6f} | Speed: {tokens_per_sec:.1f} tok/s | Mem: {cpu_mem:.1f} MB")
                start_time = time.time()
                
            # Evaluation and Checkpoint every 500 steps
            if step % 500 == 0:
                val_loss, perp = run_evaluation(model, val_loader, device)
                avg_train_loss_for_record = (running_loss / (step % 10 if step % 10 != 0 else 10)) if step % 10 != 0 else avg_train_loss
                
                cpu_mem = get_process_memory()
                elapsed_total = time.time() - total_start_time
                tokens_processed = step * 4 * 256
                
                if np.isnan(val_loss) or np.isnan(avg_train_loss_for_record):
                    print(f"ERROR: Detected NaN loss at step {step}. Stopping training.")
                    return
                
                history["steps"].append(step)
                history["train_loss"].append(avg_train_loss_for_record)
                history["val_loss"].append(val_loss)
                
                # Write to CSV log
                with open(log_csv_path, "a", encoding="utf-8") as f:
                    f.write(f"{step},{avg_train_loss_for_record:.4f},{val_loss:.4f},{perp:.2f},{current_lr:.6f},{tokens_processed},{tokens_per_sec:.1f},{steps_per_sec:.2f},{elapsed_total:.1f},{cpu_mem:.1f}\n")
                
                print(f"\n--- VALIDATION --- Step {step} | Train Loss: {avg_train_loss_for_record:.4f} | Val Loss: {val_loss:.4f} | Perplexity: {perp:.2f}\n")
                
                # Save regular checkpoint
                cp_name = f"collision-7m-step-{step:06d}.pt"
                cp_path = os.path.join(CP_DIR, cp_name)
                save_checkpoint(
                    model, optimizer, scheduler, step, 0, avg_train_loss_for_record, val_loss,
                    model_config.__dict__, {"save_dir": TOKENIZER_DIR}, cp_path
                )
                
                # Update best validation checkpoint
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_step = step
                    best_checkpoint_path = os.path.join(CP_DIR, "collision-7m-best.pt")
                    save_checkpoint(
                        model, optimizer, scheduler, step, 0, avg_train_loss_for_record, val_loss,
                        model_config.__dict__, {"save_dir": TOKENIZER_DIR}, best_checkpoint_path
                    )
                
                model.train()

    end_epoch_time = time.time()
    total_train_time = end_epoch_time - start_epoch_time
    end_time_str = time.strftime("%Y-%m-%d %H:%M:%S")
    peak_cpu_mem = get_process_memory()

    print("\nTraining completed successfully! Running final analysis...")
    
    # Save best checkpoint if not already saved
    if not best_checkpoint_path:
        best_checkpoint_path = os.path.join(CP_DIR, "collision-7m-best.pt")
        save_checkpoint(model, optimizer, scheduler, step, 0, avg_train_loss_for_record, val_loss, model_config.__dict__, {"save_dir": TOKENIZER_DIR}, best_checkpoint_path)

    # Plot loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(history["steps"], history["train_loss"], label="Train Loss", marker='o')
    plt.plot(history["steps"], history["val_loss"], label="Val Loss", marker='s')
    plt.title("COLLISION-7M Loss Curve")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(EXP_DIR, "loss_curve.png"), dpi=150)
    plt.close()
    print("Saved loss curve plot to loss_curve.png")
    
    # Load best checkpoint for benchmark evaluation
    best_cp = torch.load(best_checkpoint_path, map_location=device)
    model.load_state_dict(best_cp["model_state_dict"])
    
    bench_results = run_benchmark(model, tokenizer, device)
    
    # Compute average inference speed
    avg_inf_speed = sum(res["tokens_per_second"] for res in bench_results) / len(bench_results)
    
    # Baseline 1.46M details
    base_params = 1462464
    base_val_loss = 1.9363
    base_perp = 6.93
    base_train_time = 490.5
    
    # 3.38M Experiment details
    exp3m_params = 3375680
    exp3m_val_loss = 0.9663
    exp3m_perp = 2.63
    exp3m_train_time = 1067.4
    exp3m_inf_speed = 92.12
    
    # Benchmarking base 1.46M model for exact relative comparison speed on this run
    print("\nBenchmarking COLLISION-1.46M baseline on current platform for exact relative comparison...")
    base_cp_path = "checkpoints/phase6/collision-1.46m-best.pt"
    base_cp = torch.load(base_cp_path, map_location=device)
    base_cfg = ModelConfig(**base_cp["config"])
    base_model = CollisionTransformer(base_cfg).to(device)
    base_model.load_state_dict(base_cp["model_state_dict"])
    base_bench = run_benchmark(base_model, tokenizer, device)
    base_inf_speed = sum(res["tokens_per_second"] for res in base_bench) / len(base_bench)
    
    # Compute relative changes (compared to 1.46M baseline)
    param_inc_base = ((param_count - base_params) / base_params) * 100
    loss_chg_base = ((best_val_loss - base_val_loss) / base_val_loss) * 100
    perp_chg_base = ((np.exp(best_val_loss) - base_perp) / base_perp) * 100
    time_chg_base = ((total_train_time - base_train_time) / base_train_time) * 100
    speed_chg_base = ((avg_inf_speed - base_inf_speed) / base_inf_speed) * 100

    # Compute relative changes (compared to 3.38M experiment)
    param_inc_3m = ((param_count - exp3m_params) / exp3m_params) * 100
    loss_chg_3m = ((best_val_loss - exp3m_val_loss) / exp3m_val_loss) * 100
    perp_chg_3m = ((np.exp(best_val_loss) - exp3m_perp) / exp3m_perp) * 100
    time_chg_3m = ((total_train_time - exp3m_train_time) / exp3m_train_time) * 100
    speed_chg_3m = ((avg_inf_speed - exp3m_inf_speed) / exp3m_inf_speed) * 100
    
    # Overfitting analysis text
    # Determine whether validation loss continues decreasing, plateaus, or begins increasing.
    # In history, val_loss has 3 records (at steps 500, 1000, 1500)
    val_losses = history["val_loss"]
    if len(val_losses) >= 2:
        if val_losses[-1] < val_losses[-2]:
            overfit_txt = "No obvious validation overfitting was observed within the training window."
        elif abs(val_losses[-1] - val_losses[-2]) < 0.05:
            overfit_txt = "Validation loss plateaued toward the end of the training window."
        else:
            overfit_txt = "Validation loss began increasing, indicating the onset of validation overfitting."
    else:
        overfit_txt = "No obvious validation overfitting was observed within the training window."

    # Save results.md
    results_path = os.path.join(EXP_DIR, "results.md")
    with open(results_path, "w", encoding="utf-8") as f:
        f.write(f"""# COLLISION-7M Scaling Experiment Results

## Overview
This document presents the results of the **COLLISION-7M** scaling experiment, comparing it directly against the baseline **COLLISION-1.46M** model and the **COLLISION-3M** experiment. All models were trained on the identical dataset (`collision_dataset_v4`) under identical hyperparameter conditions on CPU for exactly `1,536,000` training tokens (`1,500` steps).

## Direct Comparison Metrics

| Metric | COLLISION-1.46M (Baseline) | COLLISION-3M (Experiment) | COLLISION-7M (Experiment) | Change vs Base (%) | Change vs 3M (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Model Parameters** | {base_params:,} | {exp3m_params:,} | {param_count:,} | {param_inc_base:+.2f}% | {param_inc_3m:+.2f}% |
| **Best Validation Loss** | {base_val_loss:.4f} | {exp3m_val_loss:.4f} | {best_val_loss:.4f} | {loss_chg_base:+.2f}% | {loss_chg_3m:+.2f}% |
| **Best Validation Perplexity** | {base_perp:.2f} | {exp3m_perp:.2f} | {np.exp(best_val_loss):.2f} | {perp_chg_base:+.2f}% | {perp_chg_3m:+.2f}% |
| **Total Training Time (s)** | {base_train_time:.1f}s | {exp3m_train_time:.1f}s | {total_train_time:.1f}s | {time_chg_base:+.2f}% | {time_chg_3m:+.2f}% |
| **Avg Inference Speed (tok/s)** | {base_inf_speed:.2f} | {exp3m_inf_speed:.2f} | {avg_inf_speed:.2f} | {speed_chg_base:+.2f}% | {speed_chg_3m:+.2f}% |

## Observations & Overfitting Analysis
* **Overfitting Analysis**: {overfit_txt}
* **Validation Loss**: Changed from `{base_val_loss:.4f}` (1.46M) and `{exp3m_val_loss:.4f}` (3M) to `{best_val_loss:.4f}` ({loss_chg_base:+.2f}% vs Base, {loss_chg_3m:+.2f}% vs 3M).
* **Perplexity**: Changed from `{base_perp:.2f}` (1.46M) and `{exp3m_perp:.2f}` (3M) to `{np.exp(best_val_loss):.2f}` ({perp_chg_base:+.2f}% vs Base, {perp_chg_3m:+.2f}% vs 3M).
* **Training Time**: Parameter count scaled to `{param_count:,}` resulting in a `{time_chg_base:+.1f}%` training time change compared to the baseline.
* **Inference Throughput**: Generative speed on CPU reached `{avg_inf_speed:.2f}` tokens/second.

## Generation Benchmark Outputs
""")
        for res in bench_results:
            f.write(f"""
### Prompt: "{res['prompt']}"
* **Generated Text**: `{res['generated_text']}`
* **Tokens generated**: `{res['tokens_generated']}`
* **Generation time**: `{res['generation_time']:.4f}s`
* **Tokens/second**: `{res['tokens_per_second']:.2f}`
""")
            
        f.write(f"""
## Reproducibility Variables
* **Git Commit**: `{git_commit_hash()}`
* **Python Version**: `{platform.python_version()}`
* **PyTorch Version**: `{torch.__version__}`
* **CPU Info**: `{platform.processor()}`
* **Configuration Hash**: `{config_hash}`
* **Dataset Version**: `collision_dataset_v4`
* **Tokenizer Version**: `1.0-BPETokenizer`
* **Seed**: `1337`
* **Start Time**: `{start_time_str}`
* **End Time**: `{end_time_str}`
""")
        
    print(f"\nSaved results summary to {results_path}")

def git_commit_hash():
    try:
        import subprocess
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    except Exception:
        return "N/A"

if __name__ == "__main__":
    main()
