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
import re

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
CONFIG_JSON_PATH = "experiments/phase12b/config.json"
EXP_DIR = "experiments/phase12b"
CP_DIR = "checkpoints/phase12b"
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
            # Evaluate on up to 50 batches for quick validation during training
            if steps >= 50:
                break
    mean_loss = total_loss / max(1, steps)
    perplexity = np.exp(mean_loss) if mean_loss < 20 else float('inf')
    return mean_loss, perplexity

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

def evaluate_full_dataset(model, bin_path, seq_len, device):
    """Run full evaluation on validation or test set."""
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

def generate_sample(model, tokenizer, prompt, device, max_tokens=50, temp=0.8, top_k=50):
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
                
    return x[0].tolist(), tokenizer.decode(x[0].tolist())

def calculate_quality_metrics(token_ids, prompt_len, tokenizer):
    """
    Calculate generation quality metrics:
    - repetition rate: percentage of duplicate tokens
    - unique token ratio: number of unique tokens / total generated tokens
    - repeated n-grams: 2-grams and 3-grams repetition
    - average generated length
    - invalid/unknown token frequency: number of [UNK] tokens
    - sentence termination rate: ends with EOS or punctuation
    - prompt conditioning: prompt length vs output length
    """
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
    
    # 1. Repetition Rate & Unique Token Ratio
    unique_tokens = set(gen_ids)
    unique_ratio = len(unique_tokens) / len(gen_ids)
    repetition_rate = 1.0 - unique_ratio
    
    # 2. Repeated n-grams
    n2_grams = list(zip(gen_ids[:-1], gen_ids[1:]))
    n2_repeats = len(n2_grams) - len(set(n2_grams))
    
    n3_grams = list(zip(gen_ids[:-2], gen_ids[1:-1], gen_ids[2:]))
    n3_repeats = len(n3_grams) - len(set(n3_grams))
    
    # 3. Invalid/unknown token frequency
    unk_id = tokenizer.special_tokens.get("[UNK]", 257)
    unk_count = gen_ids.count(unk_id)
    unk_freq = unk_count / len(gen_ids)
    
    # 4. Sentence termination
    eos_id = tokenizer.special_tokens.get("[EOS]", 259)
    terminated = False
    if gen_ids[-1] == eos_id:
        terminated = True
    else:
        # Check text decoding for ending punctuation
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
    start_time_str = time.strftime("%Y-%m-%d %H:%M:%S")
    start_epoch_time = time.time()
    
    os.makedirs(EXP_DIR, exist_ok=True)
    os.makedirs(CP_DIR, exist_ok=True)

    # 1. Load config
    with open(CONFIG_JSON_PATH, "r") as f:
        exp_config = json.load(f)
        
    model_cfg_dict = exp_config["model_configuration"]
    model_config = ModelConfig(**model_cfg_dict)
    
    # Seeds
    seed = exp_config["random_seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(exp_config["device"])

    # Load tokenizer
    tokenizer = BPETokenizer()
    tokenizer.load(TOKENIZER_DIR)

    # Load dataset info
    train_bin = os.path.join(DATASET_DIR, "train.bin")
    val_bin = os.path.join(DATASET_DIR, "val.bin")
    test_bin = os.path.join(DATASET_DIR, "test.bin")
    
    # Verify dataset exists
    if not os.path.exists(train_bin) or not os.path.exists(val_bin) or not os.path.exists(test_bin):
        raise FileNotFoundError("Missing one or more V5 dataset binary files (train.bin, val.bin, test.bin).")

    # Initialize model from scratch (fresh random initialization)
    model = CollisionTransformer(model_config).to(device)
    
    # Programmatic verification
    param_count = sum(p.numel() for p in model.parameters())
    trainable_param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"========================================")
    print(f"COLLISION Phase 12B Training Experiment")
    print(f"========================================")
    print(f"Parameters: {param_count:,}")
    print(f"Trainable Parameters: {trainable_param_count:,}")
    print(f"Vocabulary Size: {model_config.vocab_size}")
    print(f"Context Length: {model_config.max_seq_len}")
    print(f"Number of Layers: {model_config.n_layer}")
    print(f"Embedding Dimension: {model_config.d_model}")
    print(f"Attention Heads: {model_config.n_head}")
    print(f"========================================")
    
    # Verify parameters match the target
    assert param_count == 3375680, f"Expected 3,375,680 parameters, got {param_count}"
    assert trainable_param_count == 3375680, f"Expected 3,375,680 trainable parameters, got {trainable_param_count}"
    
    # Update config.json metadata
    exp_config["verified_parameter_count"] = param_count
    exp_config["verified_trainable_parameter_count"] = trainable_param_count
    exp_config["verified_vocab_size"] = model_config.vocab_size
    exp_config["verified_context_length"] = model_config.max_seq_len
    exp_config["verified_n_layer"] = model_config.n_layer
    exp_config["verified_d_model"] = model_config.d_model
    exp_config["verified_n_head"] = model_config.n_head
    exp_config["verified_d_ff"] = model_config.d_ff
    
    with open(CONFIG_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(exp_config, f, indent=2)

    # Standardized evaluation prompts
    eval_prompts = [
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
    
    # Generate baseline output from random model
    print("\nGenerating baseline outputs from random model...")
    random_gen_results = []
    for prompt in eval_prompts:
        prompt_len = len(tokenizer.encode(prompt, bos=True))
        tok_ids, text = generate_sample(model, tokenizer, prompt, device, max_tokens=50)
        metrics = calculate_quality_metrics(tok_ids, prompt_len, tokenizer)
        random_gen_results.append({
            "prompt": prompt,
            "output": text,
            "metrics": metrics
        })

    # Dataloaders
    train_dataset = TokenDataset(train_bin, model_config.max_seq_len)
    val_dataset = TokenDataset(val_bin, model_config.max_seq_len)
    train_loader = DataLoader(train_dataset, batch_size=exp_config["batch_size"], shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=exp_config["batch_size"], shuffle=False)

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=exp_config["learning_rate"], weight_decay=exp_config["weight_decay"])
    scheduler = CosineWarmupScheduler(
        optimizer, 
        warmup_steps=exp_config["warmup_steps"], 
        total_steps=exp_config["maximum_steps"], 
        base_lr=exp_config["learning_rate"], 
        min_lr=exp_config["min_lr"]
    )

    # CSV Log initialization
    log_csv_path = os.path.join(EXP_DIR, "training_log.csv")
    final_checkpoint_path = os.path.join(CP_DIR, "collision-3.38m-phase12b-step-001500.pt")
    if not os.path.exists(final_checkpoint_path):
        with open(log_csv_path, "w", encoding="utf-8") as f:
            f.write("step,train_loss,validation_loss,validation_perplexity,learning_rate,tokens_per_second,elapsed_time,cpu_memory\n")

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
    
    intermediate_generations = {}

    final_checkpoint_path = os.path.join(CP_DIR, "collision-3.38m-phase12b-step-001500.pt")
    if os.path.exists(final_checkpoint_path):
        print("Final checkpoint already exists. Loading training history from training_log.csv...")
        import csv
        with open(log_csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                h_step = int(row["step"])
                h_train = float(row["train_loss"])
                h_val = float(row["validation_loss"])
                history["steps"].append(h_step)
                history["train_loss"].append(h_train)
                history["val_loss"].append(h_val)
                if h_val < best_val_loss:
                    best_val_loss = h_val
                    best_step = h_step
        step = 1500
        best_checkpoint_path = os.path.join(CP_DIR, "collision-3.38m-phase12b-best.pt")
        avg_train_loss_for_record = history["train_loss"][-1]
        val_loss = history["val_loss"][-1]
        finished_training = True
        total_train_time = 997.1
        
        # Load intermediate checkpoints and generate samples
        for step_num in [500, 1000]:
            cp_path = os.path.join(CP_DIR, f"collision-3.38m-phase12b-step-{step_num:06d}.pt")
            if os.path.exists(cp_path):
                print(f"Loading checkpoint for step {step_num} to generate intermediate samples...")
                temp_cp = torch.load(cp_path, map_location=device)
                model.load_state_dict(temp_cp["model_state_dict"])
                step_gen = []
                for prompt in eval_prompts:
                    prompt_len = len(tokenizer.encode(prompt, bos=True))
                    tok_ids, text = generate_sample(model, tokenizer, prompt, device, max_tokens=50)
                    metrics = calculate_quality_metrics(tok_ids, prompt_len, tokenizer)
                    step_gen.append({
                        "prompt": prompt,
                        "output": text,
                        "metrics": metrics
                    })
                intermediate_generations[step_num] = step_gen
    else:
        print(f"Starting {exp_config['maximum_steps']} steps CPU training...")
        model.train()
        optimizer.zero_grad()
        finished_training = False

    while not finished_training:
        for x, y in train_loader:
            if step >= exp_config["maximum_steps"]:
                finished_training = True
                break
                
            x, y = x.to(device), y.to(device)
            logits, loss = model(x, y)
            
            # Gradient Accumulation
            loss_scaled = loss / exp_config["gradient_accumulation"]
            loss_scaled.backward()
            running_loss += loss.item()
            
            if (step + 1) % exp_config["gradient_accumulation"] == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=exp_config["gradient_clipping"])
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
                tokens_per_sec = (exp_config["batch_size"] * model_config.max_seq_len) * steps_per_sec
                elapsed_total = time.time() - total_start_time
                
                cpu_mem = get_process_memory()
                current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else optimizer.param_groups[0]['lr']
                
                print(f"Step {step}/{exp_config['maximum_steps']} | Train Loss: {avg_train_loss:.4f} | LR: {current_lr:.6f} | Speed: {tokens_per_sec:.1f} tok/s | Mem: {cpu_mem:.1f} MB")
                start_time = time.time()
                
            # Evaluation and Checkpoint every 500 steps
            if step % 500 == 0:
                val_loss, perp = run_evaluation(model, val_loader, device)
                avg_train_loss_for_record = (running_loss / (step % 10 if step % 10 != 0 else 10)) if step % 10 != 0 else avg_train_loss
                
                cpu_mem = get_process_memory()
                elapsed_total = time.time() - total_start_time
                tokens_per_sec = (exp_config["batch_size"] * model_config.max_seq_len) / (time.time() - start_time)
                
                history["steps"].append(step)
                history["train_loss"].append(avg_train_loss_for_record)
                history["val_loss"].append(val_loss)
                
                # Write to CSV log
                with open(log_csv_path, "a", encoding="utf-8") as f:
                    f.write(f"{step},{avg_train_loss_for_record:.4f},{val_loss:.4f},{perp:.2f},{current_lr:.6f},{tokens_per_sec:.1f},{elapsed_total:.1f},{cpu_mem:.1f}\n")
                
                print(f"\n--- VALIDATION --- Step {step} | Train Loss: {avg_train_loss_for_record:.4f} | Val Loss: {val_loss:.4f} | Perplexity: {perp:.2f}\n")
                
                # Save regular checkpoint
                cp_name = f"collision-3.38m-phase12b-step-{step:06d}.pt"
                cp_path = os.path.join(CP_DIR, cp_name)
                save_checkpoint(
                    model, optimizer, scheduler, step, 0, avg_train_loss_for_record, val_loss,
                    model_config.__dict__, {"save_dir": TOKENIZER_DIR}, cp_path
                )
                
                # Save intermediate generations
                print(f"Generating samples at step {step}...")
                step_gen = []
                for prompt in eval_prompts:
                    prompt_len = len(tokenizer.encode(prompt, bos=True))
                    tok_ids, text = generate_sample(model, tokenizer, prompt, device, max_tokens=50)
                    metrics = calculate_quality_metrics(tok_ids, prompt_len, tokenizer)
                    step_gen.append({
                        "prompt": prompt,
                        "output": text,
                        "metrics": metrics
                    })
                intermediate_generations[step] = step_gen
                
                # Update best validation checkpoint
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_step = step
                    best_checkpoint_path = os.path.join(CP_DIR, "collision-3.38m-phase12b-best.pt")
                    save_checkpoint(
                        model, optimizer, scheduler, step, 0, avg_train_loss_for_record, val_loss,
                        model_config.__dict__, {"save_dir": TOKENIZER_DIR}, best_checkpoint_path
                    )
                
                model.train()
                start_time = time.time()

    end_epoch_time = time.time()
    total_train_time = end_epoch_time - start_epoch_time
    
    # Save final checkpoint explicitly
    final_checkpoint_path = os.path.join(CP_DIR, f"collision-3.38m-phase12b-step-{step:06d}.pt")
    if not os.path.exists(final_checkpoint_path):
        save_checkpoint(
            model, optimizer, scheduler, step, 0, avg_train_loss_for_record, val_loss,
            model_config.__dict__, {"save_dir": TOKENIZER_DIR}, final_checkpoint_path
        )
        
    print("\nTraining completed successfully! Running final analysis...")
    
    # Load best checkpoint for final evaluations and benchmark
    best_cp = torch.load(best_checkpoint_path, map_location=device)
    model.load_state_dict(best_cp["model_state_dict"])
    
    # Run full dataset evaluations for best model
    val_loss_best, val_perp_best = evaluate_full_dataset(model, val_bin, model_config.max_seq_len, device)
    
    # ONLY evaluate on the TEST set after training has finished
    test_loss_best, test_perp_best = evaluate_full_dataset(model, test_bin, model_config.max_seq_len, device)
    
    print(f"\n========================================")
    print(f"FINAL METRICS (Best Checkpoint - Step {best_step})")
    print(f"========================================")
    print(f"Best Validation Loss: {val_loss_best:.4f}")
    print(f"Best Validation Perplexity: {val_perp_best:.2f}")
    print(f"Best Test Loss: {test_loss_best:.4f}")
    print(f"Best Test Perplexity: {test_perp_best:.2f}")
    print(f"========================================")

    # Generate outputs from best model
    print("Generating samples from best checkpoint...")
    best_gen_results = []
    for prompt in eval_prompts:
        prompt_len = len(tokenizer.encode(prompt, bos=True))
        tok_ids, text = generate_sample(model, tokenizer, prompt, device, max_tokens=50)
        metrics = calculate_quality_metrics(tok_ids, prompt_len, tokenizer)
        best_gen_results.append({
            "prompt": prompt,
            "output": text,
            "metrics": metrics
        })

    # Generate outputs from final model (if different from best)
    final_cp = torch.load(final_checkpoint_path, map_location=device)
    model.load_state_dict(final_cp["model_state_dict"])
    val_loss_final, val_perp_final = evaluate_full_dataset(model, val_bin, model_config.max_seq_len, device)
    test_loss_final, test_perp_final = evaluate_full_dataset(model, test_bin, model_config.max_seq_len, device)
    
    print("Generating samples from final checkpoint...")
    final_gen_results = []
    for prompt in eval_prompts:
        prompt_len = len(tokenizer.encode(prompt, bos=True))
        tok_ids, text = generate_sample(model, tokenizer, prompt, device, max_tokens=50)
        metrics = calculate_quality_metrics(tok_ids, prompt_len, tokenizer)
        final_gen_results.append({
            "prompt": prompt,
            "output": text,
            "metrics": metrics
        })

    # 10. Write generation comparison to text file
    gen_comp_path = os.path.join(EXP_DIR, "generation_comparison.txt")
    with open(gen_comp_path, "w", encoding="utf-8") as f:
        f.write("COLLISION Phase 12B Generation Comparison\n")
        f.write("=========================================\n\n")
        
        for idx, prompt in enumerate(eval_prompts):
            f.write(f"PROMPT {idx+1}: {prompt}\n")
            f.write("-" * 40 + "\n")
            f.write(f"Random Baseline:\n  {random_gen_results[idx]['output'].strip()}\n")
            f.write(f"  Repetition Rate: {random_gen_results[idx]['metrics']['repetition_rate']:.2%}\n")
            f.write(f"  Unique Token Ratio: {random_gen_results[idx]['metrics']['unique_token_ratio']:.2%}\n")
            f.write(f"  Repeated 2-grams: {random_gen_results[idx]['metrics']['repeated_2grams']}\n")
            f.write(f"  Terminated: {random_gen_results[idx]['metrics']['terminated']}\n\n")
            
            for step_num in sorted(intermediate_generations.keys()):
                step_res = intermediate_generations[step_num][idx]
                f.write(f"Step {step_num} Checkpoint:\n  {step_res['output'].strip()}\n")
                f.write(f"  Repetition Rate: {step_res['metrics']['repetition_rate']:.2%}\n")
                f.write(f"  Unique Token Ratio: {step_res['metrics']['unique_token_ratio']:.2%}\n")
                f.write(f"  Repeated 2-grams: {step_res['metrics']['repeated_2grams']}\n")
                f.write(f"  Terminated: {step_res['metrics']['terminated']}\n\n")
                
            f.write(f"Best Checkpoint (Step {best_step}):\n  {best_gen_results[idx]['output'].strip()}\n")
            f.write(f"  Repetition Rate: {best_gen_results[idx]['metrics']['repetition_rate']:.2%}\n")
            f.write(f"  Unique Token Ratio: {best_gen_results[idx]['metrics']['unique_token_ratio']:.2%}\n")
            f.write(f"  Repeated 2-grams: {best_gen_results[idx]['metrics']['repeated_2grams']}\n")
            f.write(f"  Terminated: {best_gen_results[idx]['metrics']['terminated']}\n\n")
            
            f.write(f"Final Checkpoint (Step 1500):\n  {final_gen_results[idx]['output'].strip()}\n")
            f.write(f"  Repetition Rate: {final_gen_results[idx]['metrics']['repetition_rate']:.2%}\n")
            f.write(f"  Unique Token Ratio: {final_gen_results[idx]['metrics']['unique_token_ratio']:.2%}\n")
            f.write(f"  Repeated 2-grams: {final_gen_results[idx]['metrics']['repeated_2grams']}\n")
            f.write(f"  Terminated: {final_gen_results[idx]['metrics']['terminated']}\n\n")
            f.write("=" * 80 + "\n\n")

    print(f"Saved generation comparisons to {gen_comp_path}")

    # Plot loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(history["steps"], history["train_loss"], label="Train Loss", marker='o')
    plt.plot(history["steps"], history["val_loss"], label="Val Loss", marker='s')
    plt.title("COLLISION Phase 12B Loss Curve")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    loss_curve_path = os.path.join(EXP_DIR, "loss_curve.png")
    plt.savefig(loss_curve_path, dpi=150)
    plt.close()
    print(f"Saved loss curve plot to {loss_curve_path}")
    
    # Save run details in configuration json
    exp_config["training_completed"] = True
    exp_config["best_step"] = best_step
    exp_config["best_val_loss"] = val_loss_best
    exp_config["best_val_perplexity"] = val_perp_best
    exp_config["best_test_loss"] = test_loss_best
    exp_config["best_test_perplexity"] = test_perp_best
    exp_config["final_val_loss"] = val_loss_final
    exp_config["final_val_perplexity"] = val_perp_final
    exp_config["final_test_loss"] = test_loss_final
    exp_config["final_test_perplexity"] = test_perp_final
    exp_config["training_time_seconds"] = total_train_time
    
    with open(CONFIG_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(exp_config, f, indent=2)

if __name__ == "__main__":
    main()
