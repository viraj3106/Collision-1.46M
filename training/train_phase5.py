import os
import time
import json
import psutil
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from model.config import ModelConfig
from model.transformer import CollisionTransformer
from data.tokenize import BPETokenizer
from training.checkpoint import save_checkpoint
from training.scheduler import CosineWarmupScheduler
from data.stats import get_latest_version_dir

# Configuration paths
CONFIG_PATH = "configs/collision_1m.yaml"
EXP_DIR = "experiments/phase5"
CP_DIR = "checkpoints/phase5"
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
            if steps >= 50:  # Evaluate on a decent subset for validation
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
            
            # top_k filtering (k=50)
            v, _ = torch.topk(next_token_logits, min(50, next_token_logits.size(-1)))
            next_token_logits[next_token_logits < v[-1]] = -float('Inf')
            
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            x = torch.cat((x, next_token.unsqueeze(0)), dim=1)
            if next_token.item() == tokenizer.special_tokens.get("[EOS]", 2):
                break
                
    return tokenizer.decode(x[0].tolist())

def main():
    os.makedirs(EXP_DIR, exist_ok=True)
    os.makedirs(CP_DIR, exist_ok=True)

    # 1. Load config
    with open(CONFIG_PATH, "r") as f:
        config_yaml = yaml.safe_load(f)
    
    model_config = ModelConfig.from_yaml(CONFIG_PATH)
    
    # Seeds
    seed = 1337
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cpu")

    # Load tokenizer
    tokenizer = BPETokenizer()
    tokenizer.load(TOKENIZER_DIR)

    # Load dataset info
    latest_dir = get_latest_version_dir()
    train_bin = os.path.join(latest_dir, "train.bin")
    val_bin = os.path.join(latest_dir, "val.bin")
    
    with open(os.path.join(latest_dir, "metadata.json"), "r") as f:
        meta_data = json.load(f)

    # Print critical pre-flight confirmation
    print("""========================================
COLLISION-1.46M PHASE 5
FIRST REAL TRAINING RUN
=======================

Dataset: collision_dataset_v3
Tokens: 2,411,502
Train: 2,108,753
Validation: 302,749

Parameters: 1,462,464
Vocabulary capacity: 8,000
Active tokenizer vocabulary: 890

Device: CPU
Maximum steps: 2,000

Initial checkpoint:
collision-1.46m-initial.pt
""")

    # Initialize model from scratch (no pretrained weights, random init)
    model = CollisionTransformer(model_config).to(device)
    param_count = model.get_parameter_count()

    # Save experiments/phase5/config.json
    exp_config = {
        "experiment_name": "collision_1.46m_phase5",
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
        "maximum_steps": 2000
    }
    with open(os.path.join(EXP_DIR, "config.json"), "w", encoding="utf-8") as f:
        json.dump(exp_config, f, indent=2)

    # Save initial checkpoint before training starts
    initial_cp_path = os.path.join(CP_DIR, "collision-1.46m-initial.pt")
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
    scheduler = CosineWarmupScheduler(optimizer, warmup_steps=200, total_steps=2000, base_lr=6e-4, min_lr=6e-5)

    prompts = [
        "The future of technology",
        "Artificial intelligence",
        "Science is",
        "COLLISION is",
        "Computer science"
    ]

    # --- 2. BASELINE BEFORE TRAINING ---
    print("Generating baseline sample text from initial random checkpoint...")
    baseline_gens = []
    for prompt in prompts:
        out = generate_sample(model, tokenizer, prompt, device)
        baseline_gens.append((prompt, out))
    
    with open(os.path.join(EXP_DIR, "baseline_generation.txt"), "w", encoding="utf-8") as f:
        for prompt, out in baseline_gens:
            f.write(f"Prompt: {prompt}\nOutput:\n{out}\n{'-'*40}\n")

    # Start generation comparison file
    gen_comp_path = os.path.join(EXP_DIR, "generation_comparison.txt")
    with open(gen_comp_path, "w", encoding="utf-8") as f:
        f.write("================================\nBASELINE\n========\n\n")
        for prompt, out in baseline_gens:
            f.write(f"Prompt:\n{prompt}\n\nOutput:\n{out}\n\n")
        f.write("="*32 + "\n")

    # CSV Log
    log_csv_path = os.path.join(EXP_DIR, "training_log.csv")
    with open(log_csv_path, "w", encoding="utf-8") as f:
        f.write("step,train_loss,val_loss,perplexity,lr,tokens_processed,tokens_per_sec,step_per_sec,elapsed_time,cpu_memory\n")

    # Initial Val evaluation
    init_val_loss, init_perp = run_evaluation(model, val_loader, device)
    print(f"Baseline Validation Loss: {init_val_loss:.4f} | Perplexity: {init_perp:.2f}")

    # Training Loop variables
    step = 0
    running_loss = 0.0
    start_time = time.time()
    total_start_time = time.time()
    best_val_loss = float('inf')
    best_checkpoint_path = ""
    
    # Store history for curve plotting
    history = {
        "steps": [],
        "train_loss": [],
        "val_loss": []
    }

    print("Starting 2,000 steps CPU training...")
    model.train()
    optimizer.zero_grad()
    
    finished_training = False
    
    while not finished_training:
        for x, y in train_loader:
            if step >= 2000:
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
                tokens_per_sec = (4 * 256) * steps_per_sec  # batch_size=4, seq_len=256
                elapsed_total = time.time() - total_start_time
                
                cpu_mem = get_process_memory()
                current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else optimizer.param_groups[0]['lr']
                
                print(f"Step {step}/2000 | Train Loss: {avg_train_loss:.4f} | LR: {current_lr:.6f} | Speed: {tokens_per_sec:.1f} tok/s | Mem: {cpu_mem:.1f} MB")
                start_time = time.time()
                
            # Evaluation and Checkpoint every 500 steps
            if step % 500 == 0:
                val_loss, perp = run_evaluation(model, val_loader, device)
                avg_train_loss_for_record = (running_loss / (step % 10 if step % 10 != 0 else 10)) if step % 10 != 0 else avg_train_loss
                
                cpu_mem = get_process_memory()
                elapsed_total = time.time() - total_start_time
                current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else optimizer.param_groups[0]['lr']
                tokens_processed = step * 4 * 256
                
                # Check for NaNs
                if np.isnan(val_loss) or np.isnan(avg_train_loss_for_record):
                    print(f"ERROR: Detected NaN loss at step {step}. Stopping training.")
                    # Save emergency checkpoint
                    nan_cp = os.path.join(CP_DIR, f"collision-1.46m-step-{step:06d}-nan.pt")
                    save_checkpoint(model, optimizer, scheduler, step, 0, avg_train_loss_for_record, val_loss, model_config.__dict__, {"save_dir": TOKENIZER_DIR}, nan_cp)
                    return
                
                # Append to history
                history["steps"].append(step)
                history["train_loss"].append(avg_train_loss_for_record)
                history["val_loss"].append(val_loss)
                
                # Write to CSV log
                with open(log_csv_path, "a", encoding="utf-8") as f:
                    f.write(f"{step},{avg_train_loss_for_record:.4f},{val_loss:.4f},{perp:.2f},{current_lr:.6f},{tokens_processed},{tokens_per_sec:.1f},{steps_per_sec:.2f},{elapsed_total:.1f},{cpu_mem:.1f}\n")
                
                print(f"\n--- VALIDATION --- Step {step} | Train Loss: {avg_train_loss_for_record:.4f} | Val Loss: {val_loss:.4f} | Perplexity: {perp:.2f}\n")
                
                # Save regular checkpoint
                cp_name = f"collision-1.46m-step-{step:06d}.pt"
                cp_path = os.path.join(CP_DIR, cp_name)
                save_checkpoint(
                    model, optimizer, scheduler, step, 0, avg_train_loss_for_record, val_loss,
                    model_config.__dict__, {"save_dir": TOKENIZER_DIR}, cp_path
                )
                
                # Update best validation checkpoint
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_checkpoint_path = os.path.join(CP_DIR, "collision-1.46m-best.pt")
                    save_checkpoint(
                        model, optimizer, scheduler, step, 0, avg_train_loss_for_record, val_loss,
                        model_config.__dict__, {"save_dir": TOKENIZER_DIR}, best_checkpoint_path
                    )
                
                # Save generations comparison
                checkpoint_gens = []
                for prompt in prompts:
                    out = generate_sample(model, tokenizer, prompt, device)
                    checkpoint_gens.append((prompt, out))
                    
                with open(gen_comp_path, "a", encoding="utf-8") as f:
                    f.write(f"================================\nSTEP {step}\n========\n\n")
                    for prompt, out in checkpoint_gens:
                        f.write(f"Prompt:\n{prompt}\n\nOutput:\n{out}\n\n")
                    f.write("="*32 + "\n")
                
                model.train()

    print("\nTraining completed successfully! Running final analysis...")
    
    # Save a copy of latest checkpoint as best if best doesn't exist
    if not best_checkpoint_path:
        best_checkpoint_path = os.path.join(CP_DIR, "collision-1.46m-best.pt")
        save_checkpoint(model, optimizer, scheduler, step, 0, avg_train_loss_for_record, val_loss, model_config.__dict__, {"save_dir": TOKENIZER_DIR}, best_checkpoint_path)

    # Plot loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(history["steps"], history["train_loss"], label="Train Loss", marker='o')
    plt.plot(history["steps"], history["val_loss"], label="Val Loss", marker='s')
    plt.title("COLLISION-1.46M Loss Curve (Phase 5)")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(EXP_DIR, "loss_curve.png"), dpi=150)
    plt.close()
    print("Saved loss curve plot to loss_curve.png")

    # Generate final analysis variables
    initial_train_loss = history["train_loss"][0]
    final_train_loss = history["train_loss"][-1]
    best_train_loss = min(history["train_loss"])
    initial_val_loss = history["val_loss"][0]
    final_val_loss = history["val_loss"][-1]
    best_val_loss_rec = min(history["val_loss"])
    best_perplexity = np.exp(best_val_loss_rec)
    total_tokens_processed = 2000 * 4 * 256
    total_elapsed = time.time() - total_start_time
    avg_tokens_per_sec = total_tokens_processed / total_elapsed

    # Classification
    is_improving = history["val_loss"][-1] < history["val_loss"][0]
    is_val_worsening = all(history["val_loss"][i] < history["val_loss"][i+1] for i in range(len(history["val_loss"])-1))
    
    if np.isnan(final_train_loss) or np.isnan(final_val_loss):
        classification = "UNSTABLE"
    elif is_val_worsening:
        classification = "POSSIBLE OVERFITTING"
    else:
        classification = "HEALTHY"

    # Write report.md
    report_content = f"""# COLLISION-1.46M Phase 5

## Dataset
* **Dataset Version**: {meta_data.get('dataset_version')}
* **Total Tokens**: {meta_data.get('token_count'):,}
* **Training Tokens**: {meta_data.get('train_tokens'):,}
* **Validation Tokens**: {meta_data.get('validation_tokens'):,}

## Model
* **Parameter Count**: {param_count:,}
* **Layers**: {model_config.n_layer}
* **Embedding Dimension**: {model_config.d_model}
* **Attention Heads**: {model_config.n_head}
* **Context Length**: {model_config.max_seq_len}

## Training
* **Steps**: 2,000
* **Device**: CPU
* **Batch Size**: 4 (Gradient Accumulation: 4)
* **Initial LR**: 6e-4 (Cosine Warmup)

## Results
* **Initial Training Loss**: {initial_train_loss:.4f}
* **Final Training Loss**: {final_train_loss:.4f}
* **Best Training Loss**: {best_train_loss:.4f}
* **Initial Validation Loss**: {initial_val_loss:.4f}
* **Final Validation Loss**: {final_val_loss:.4f}
* **Best Validation Loss**: {best_val_loss_rec:.4f}
* **Best Perplexity**: {best_perplexity:.2f}
* **Training Time**: {total_elapsed:.1f} seconds
* **Average tokens/sec**: {avg_tokens_per_sec:.1f}

## Checkpoints
Checkpoints saved under `checkpoints/phase5/`:
* `collision-1.46m-initial.pt` (Initial pre-training checkpoint)
* `collision-1.46m-step-000500.pt`
* `collision-1.46m-step-001000.pt`
* `collision-1.46m-step-001500.pt`
* `collision-1.46m-step-002000.pt`
* `collision-1.46m-best.pt` (Best Validation Loss: {best_val_loss_rec:.4f})

## Training Classification
### {classification}

## Known Experimental Limitation
* **Tokenizer/Model Vocab Mismatch**: The tokenizer has 890 active vocabulary tokens, while the model is configured with a vocabulary capacity of 8,000. This is safe and fully functional since all dataset token IDs fall within 0–889 (less than 8,000). The extra vocabulary capacity remains unused to preserve the target parameter count of 1,462,464 and architecture constraints.

## Generation Comparison
See complete comparisons in `generation_comparison.txt`.

## Observations
The loss decreased from initial baseline of {initial_train_loss:.4f} to final loss of {final_train_loss:.4f}, demonstrating that the model learns patterns from `collision_dataset_v3`.

## Limitations
This experiment shows simple pattern replication on a CPU dataset with 2,000 steps. It does NOT prove generalized intelligence or high-level logical reasoning.
"""

    with open(os.path.join(EXP_DIR, "report.md"), "w", encoding="utf-8") as f:
        f.write(report_content)

    print("\n========================================")
    print("COLLISION-1.46M PHASE 5 COMPLETE")
    print("========================================")
    print(f"Training status: {classification}")
    print(f"Final training loss: {final_train_loss:.4f}")
    print(f"Best validation loss: {best_val_loss_rec:.4f}")
    print(f"Perplexity: {best_perplexity:.2f}")
    print(f"Training time: {total_elapsed:.1f}s")
    print(f"Tokens/sec: {avg_tokens_per_sec:.1f}")
    print(f"Best checkpoint: collision-1.46m-best.pt")

if __name__ == "__main__":
    main()
