import os
import time
import json
import psutil
import sys
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
BASE_CHECKPOINT_PATH = "checkpoints/phase12b/collision-3.38m-phase12b-best.pt"
EXP_DIR = "experiments/phase13"
CP_DIR = "checkpoints/phase13"
TOKENIZER_DIR = "artifacts/tokenizer"
DATASET_DIR = "datasets/collision_instruct_v1"

class InstructDataset(Dataset):
    def __init__(self, jsonl_path: str, tokenizer, seq_len: int):
        self.examples = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.examples.append(json.loads(line))
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.pad_id = tokenizer.special_tokens.get("[PAD]", 256)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        text = f"<|user|>\n{ex['instruction']}\n\n<|assistant|>\n{ex['response']}"
        ids = self.tokenizer.encode(text, bos=True, eos=True)
        
        # Truncate if too long (need seq_len + 1 tokens to get seq_len inputs/targets)
        if len(ids) > self.seq_len + 1:
            ids = ids[:self.seq_len + 1]
            
        # Pad if too short
        pad_len = (self.seq_len + 1) - len(ids)
        if pad_len > 0:
            ids = ids + [self.pad_id] * pad_len
            
        x = torch.tensor(ids[:-1], dtype=torch.long)
        y = torch.tensor(ids[1:], dtype=torch.long)
        
        # Mask padding so CrossEntropyLoss ignores them (-100 is PyTorch's ignore_index default)
        y_masked = y.clone()
        y_masked[y_masked == self.pad_id] = -100
        
        return x, y_masked

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
            # Forward call with custom loss calculation to support ignore_index=-100
            logits, _ = model(x)
            loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-100)
            total_loss += loss.item()
            steps += 1
            if steps >= 50: # Cap val evaluation at 50 batches for speed
                break
    mean_loss = total_loss / max(1, steps)
    perplexity = np.exp(mean_loss) if mean_loss < 20 else float('inf')
    return mean_loss, perplexity

def evaluate_full_dataset(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    steps = 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-100)
            total_loss += loss.item()
            steps += 1
    mean_loss = total_loss / max(1, steps)
    perplexity = np.exp(mean_loss) if mean_loss < 20 else float('inf')
    return mean_loss, perplexity

def generate_instruct_sample(model, tokenizer, prompt, device, max_tokens=100, temp=0.7, top_k=50):
    model.eval()
    formatted_prompt = f"<|user|>\n{prompt}\n\n<|assistant|>\n"
    ids = tokenizer.encode(formatted_prompt, bos=True)
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
    prompt_len = len(ids)
    response_ids = generated_ids[prompt_len:]
    response_text = tokenizer.decode(response_ids)
    return generated_ids, response_text

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
    os.makedirs(EXP_DIR, exist_ok=True)
    os.makedirs(CP_DIR, exist_ok=True)

    # 1. Load configuration and tokenizer
    tokenizer = BPETokenizer()
    tokenizer.load(TOKENIZER_DIR)
    
    device = torch.device("cpu")
    print(f"Loading base checkpoint from {BASE_CHECKPOINT_PATH}...")
    checkpoint = torch.load(BASE_CHECKPOINT_PATH, map_location=device)
    
    model_config = ModelConfig(**checkpoint["config"])
    model = CollisionTransformer(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Loaded model with {param_count:,} parameters.")
    
    # 2. Datasets
    train_dataset = InstructDataset(os.path.join(DATASET_DIR, "train.jsonl"), tokenizer, model_config.max_seq_len)
    val_dataset = InstructDataset(os.path.join(DATASET_DIR, "val.jsonl"), tokenizer, model_config.max_seq_len)
    test_dataset = InstructDataset(os.path.join(DATASET_DIR, "test.jsonl"), tokenizer, model_config.max_seq_len)
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    # Hyperparameters
    max_steps = 1500
    learning_rate = 5e-5
    min_lr = 5e-6
    warmup_steps = 150
    weight_decay = 0.01
    grad_accumulation = 4
    grad_clipping = 1.0

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = CosineWarmupScheduler(
        optimizer, 
        warmup_steps=warmup_steps, 
        total_steps=max_steps, 
        base_lr=learning_rate, 
        min_lr=min_lr
    )

    # Log file
    log_csv_path = os.path.join(EXP_DIR, "training_log.csv")
    with open(log_csv_path, "w", encoding="utf-8") as f:
        f.write("step,train_loss,validation_loss,validation_perplexity,learning_rate,tokens_per_second,elapsed_time,cpu_memory\n")

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

    # Verify if instruct checkpoint exists (resume)
    final_checkpoint_path = os.path.join(CP_DIR, "collision-instruct-3.37m-step-001500.pt")
    if os.path.exists(final_checkpoint_path):
        print("Final instruction checkpoint already exists. Skipping training loop and running final analysis...")
        step = 1500
        # Reconstruct history if log exists
        if os.path.exists(log_csv_path) and os.path.getsize(log_csv_path) > 120:
            import csv
            with open(log_csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    h_step = int(row["step"])
                    history["steps"].append(h_step)
                    history["train_loss"].append(float(row["train_loss"]))
                    history["val_loss"].append(float(row["validation_loss"]))
            best_val_loss = min(history["val_loss"])
            best_step = history["steps"][history["val_loss"].index(best_val_loss)]
        else:
            best_val_loss = 1.00 # placeholder
            best_step = 1500
        best_checkpoint_path = os.path.join(CP_DIR, "collision-instruct-3.37m-best.pt")
        total_train_time = 0.0
    else:
        print(f"Starting {max_steps} steps of Instruction Fine-Tuning (SFT) on CPU...")
        model.train()
        optimizer.zero_grad()
        finished_training = False
        
        while not finished_training:
            for x, y in train_loader:
                if step >= max_steps:
                    finished_training = True
                    break
                    
                x, y = x.to(device), y.to(device)
                logits, _ = model(x)
                # Compute loss ignoring -100 label index
                loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-100)
                
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
                    tokens_per_sec = (4 * model_config.max_seq_len) * steps_per_sec
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
                    cp_name = f"collision-instruct-3.37m-step-{step:06d}.pt"
                    cp_path = os.path.join(CP_DIR, cp_name)
                    save_checkpoint(
                        model, optimizer, scheduler, step, 0, avg_train_loss_rec, val_loss,
                        model_config.__dict__, {"save_dir": TOKENIZER_DIR}, cp_path
                    )
                    
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_step = step
                        best_checkpoint_path = os.path.join(CP_DIR, "collision-instruct-3.37m-best.pt")
                        save_checkpoint(
                            model, optimizer, scheduler, step, 0, avg_train_loss_rec, val_loss,
                            model_config.__dict__, {"save_dir": TOKENIZER_DIR}, best_checkpoint_path
                        )
                        
                    model.train()
                    start_time = time.time()

        # Save final checkpoint
        final_checkpoint_path = os.path.join(CP_DIR, f"collision-instruct-3.37m-step-{step:06d}.pt")
        if not os.path.exists(final_checkpoint_path):
            save_checkpoint(
                model, optimizer, scheduler, step, 0, avg_train_loss_rec, val_loss,
                model_config.__dict__, {"save_dir": TOKENIZER_DIR}, final_checkpoint_path
            )

    # 4. Final Evaluations on Best checkpoint
    best_cp_path = os.path.join(CP_DIR, "collision-instruct-3.37m-best.pt")
    if os.path.exists(best_cp_path):
        print(f"Loading best checkpoint from {best_cp_path} for final evaluations...")
        best_cp = torch.load(best_cp_path, map_location=device)
        model.load_state_dict(best_cp["model_state_dict"])
        
    val_loss_best, val_perp_best = evaluate_full_dataset(model, val_loader, device)
    test_loss_best, test_perp_best = evaluate_full_dataset(model, test_loader, device)
    
    print("\n========================================")
    print("INSTRUCT MODEL FINAL METRICS (Best)")
    print("========================================")
    print(f"Validation Loss: {val_loss_best:.4f}")
    print(f"Validation Perplexity: {val_perp_best:.2f}")
    print(f"Test Loss: {test_loss_best:.4f}")
    print(f"Test Perplexity: {test_perp_best:.2f}")
    print("========================================")

    # Save details to training log curve plot
    plt.figure(figsize=(10, 6))
    plt.plot(history["steps"], history["train_loss"], label="Train Loss", marker='o')
    plt.plot(history["steps"], history["val_loss"], label="Val Loss", marker='s')
    plt.title("COLLISION-Instruct-3.37M Loss Curve")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(EXP_DIR, "loss_curve.png"), dpi=150)
    plt.close()
    
    # Save training status details in config config.json for Instruct
    exp_config = {
        "experiment_name": "collision_instruct_phase13",
        "base_model_checkpoint": BASE_CHECKPOINT_PATH,
        "dataset_version": "collision_instruct_v1",
        "best_step": best_step,
        "best_val_loss": val_loss_best,
        "best_val_perplexity": val_perp_best,
        "best_test_loss": test_loss_best,
        "best_test_perplexity": test_perp_best
    }
    with open(os.path.join(EXP_DIR, "config.json"), "w", encoding="utf-8") as f:
        json.dump(exp_config, f, indent=2)

if __name__ == "__main__":
    main()
