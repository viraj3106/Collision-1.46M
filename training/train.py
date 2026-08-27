import os
import time
import argparse
import psutil
import yaml
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model.config import ModelConfig
from model.transformer import CollisionTransformer
from data.tokenize import BPETokenizer
from training.checkpoint import save_checkpoint, load_checkpoint
from training.scheduler import CosineWarmupScheduler
from collision.config import DEFAULT_CONFIG_PATH, CHECKPOINT_DIR, EXPERIMENT_DIR, TOKENIZER_DIR, PROCESSED_DATA_DIR
from data.stats import get_latest_version_dir

class TokenDataset(Dataset):
    def __init__(self, bin_path: str, seq_len: int):
        if not os.path.exists(bin_path):
            raise FileNotFoundError(f"Binary token file not found at {bin_path}. Run tokenization first.")
        self.data = np.fromfile(bin_path, dtype=np.uint16)
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.data) - self.seq_len - 1)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.data[idx : idx + self.seq_len].astype(np.int64))
        y = torch.from_numpy(self.data[idx + 1 : idx + self.seq_len + 1].astype(np.int64))
        return x, y

def get_cpu_info():
    try:
        return psutil.cpu_percent()
    except Exception:
        return 0.0

def get_process_memory():
    try:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024) # MB
    except Exception:
        return 0.0

def generate_text_sample(model, tokenizer, prompt="The system", max_tokens=30, device="cpu"):
    model.eval()
    ids = tokenizer.encode(prompt, bos=True)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    
    with torch.no_grad():
        for _ in range(max_tokens):
            x_cond = x if x.size(1) <= model.config.max_seq_len else x[:, -model.config.max_seq_len:]
            logits, _ = model(x_cond)
            next_token_logits = logits[0, -1, :]
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.argmax(probs).unsqueeze(0).unsqueeze(0)
            x = torch.cat((x, next_token), dim=1)
            
    return tokenizer.decode(x[0].tolist())

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
            if steps >= 10:
                break
    mean_loss = total_loss / max(1, steps)
    perplexity = np.exp(mean_loss) if mean_loss < 20 else float('inf')
    return mean_loss, perplexity

def main():
    parser = argparse.ArgumentParser(description="Train COLLISION-1M decoder model")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH, help="Path to config yaml")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size override")
    parser.add_argument("--learning-rate", type=float, default=None, help="Learning rate override")
    parser.add_argument("--max-steps", type=int, default=None, help="Maximum number of training steps override")
    parser.add_argument("--checkpoint-interval", type=int, default=None, help="Save checkpoint interval override")
    parser.add_argument("--device", type=str, default=None, help="Device override: cpu, cuda")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--smoke-test", action="store_true", help="Run a quick smoke test verify loop")
    parser.add_argument("--profile", action="store_true", help="Profile CPU training speeds and performance metrics")
    parser.add_argument("--cpu-safe", action="store_true", help="Use conservative safe configurations for standard CPU laptop")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed for training reproducibility")
    args = parser.parse_args()

    # Load configuration
    with open(args.config, "r") as f:
        config_yaml = yaml.safe_load(f)
    
    model_config = ModelConfig.from_yaml(args.config)
    
    # Load training configs with fallback to CLI or defaults
    train_cfg = config_yaml.get("training", {})
    
    batch_size = int(args.batch_size or train_cfg.get("batch_size", 8))
    learning_rate = float(args.learning_rate or train_cfg.get("learning_rate", 6e-4))
    max_steps = int(args.max_steps or train_cfg.get("max_steps", 1000))
    checkpoint_interval = int(args.checkpoint_interval or train_cfg.get("checkpoint_interval", 100))
    device_name = str(args.device or train_cfg.get("device", "cpu"))
    grad_accum_steps = int(train_cfg.get("gradient_accumulation_steps", 1))


    if args.cpu_safe:
        batch_size = 4
        checkpoint_interval = 25
        grad_accum_steps = max(grad_accum_steps, 2)
        print(f"CPU_SAFE mode active: batch size set to 4, checkpoint interval set to 25, gradient accumulation set to {grad_accum_steps}.")

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    print(f"Random seed set to {args.seed}")

    # Initialize device
    device = torch.device("cuda" if torch.cuda.is_available() and device_name == "cuda" else "cpu")
    print(f"Using training device: {device}")

    # Load tokenizer
    tokenizer = BPETokenizer()
    try:
        tokenizer.load(TOKENIZER_DIR)
        print("Tokenizer loaded successfully.")
    except Exception:
        print("Error: Tokenizer files not found. Training a default tokenizer first...")
        from data.tokenize import main as train_tok
        train_tok()
        tokenizer.load(TOKENIZER_DIR)

    # Initialize model
    model = CollisionTransformer(model_config).to(device)
    param_count = model.get_parameter_count()
    print(f"COLLISION-1M model initialized. Programmatic parameter count: {param_count:,}")

    # Discover datasets
    latest_dir = get_latest_version_dir()
    if latest_dir:
        train_bin = os.path.join(latest_dir, "train.bin")
        val_bin = os.path.join(latest_dir, "val.bin")
        meta_path = os.path.join(latest_dir, "metadata.json")
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        dataset_tokens = metadata.get("token_count", 0)
    else:
        train_bin = os.path.join(PROCESSED_DATA_DIR, "train.bin")
        val_bin = os.path.join(PROCESSED_DATA_DIR, "val.bin")
        dataset_tokens = 0

    if not os.path.exists(train_bin) or not os.path.exists(val_bin):
        print("Tokenized binaries not found. Automatically tokenizing dataset...")
        from data.tokenize import main as run_tok
        run_tok()
        latest_dir = get_latest_version_dir()
        if latest_dir:
            train_bin = os.path.join(latest_dir, "train.bin")
            val_bin = os.path.join(latest_dir, "val.bin")

    train_dataset = TokenDataset(train_bin, model_config.max_seq_len)
    val_dataset = TokenDataset(val_bin, model_config.max_seq_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    total_train_steps = min(max_steps, len(train_loader) * args.epochs)
    warmup_steps = int(0.1 * total_train_steps)
    scheduler = CosineWarmupScheduler(optimizer, warmup_steps, total_train_steps, learning_rate, min_lr=learning_rate*0.1)

    start_step = 0
    start_epoch = 0
    best_val_loss = float('inf')

    latest_cp_path = os.path.join(CHECKPOINT_DIR, "latest.pt")
    if args.resume and os.path.exists(latest_cp_path):
        print(f"Resuming training from checkpoint: {latest_cp_path}")
        checkpoint = load_checkpoint(latest_cp_path, model, optimizer, scheduler)
        start_step = checkpoint["step"]
        start_epoch = checkpoint["epoch"]
        best_val_loss = checkpoint.get("val_loss", float('inf'))

    # Smoke Test Mode
    if args.smoke_test:
        print("--- RUNNING COLLISION-1M SMOKE TEST ---")
        smoke_steps = 5
        model.train()
        for s, (x, y) in enumerate(train_loader):
            if s >= smoke_steps:
                break
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits, loss = model(x, y)
            loss.backward()
            optimizer.step()
            print(f"Smoke step {s+1}/{smoke_steps} | Training Loss: {loss.item():.4f}")
            
        val_loss, perp = run_evaluation(model, val_loader, device)
        print(f"Smoke Validation | Loss: {val_loss:.4f} | Perplexity: {perp:.2f}")

        gen_text = generate_text_sample(model, tokenizer, prompt="COLLISION-1M is", max_tokens=15, device=device)
        print(f"Smoke generated sample: {gen_text}")

        test_cp_path = os.path.join(CHECKPOINT_DIR, "smoke_test_checkpoint.pt")
        save_checkpoint(
            model, optimizer, scheduler, smoke_steps, 0, loss.item(), val_loss,
            model_config.__dict__, {"save_dir": TOKENIZER_DIR}, test_cp_path
        )
        load_checkpoint(test_cp_path, model)
        print("Smoke test checkpoint verified and successfully loaded.")
        print("COLLISION-1M initialized successfully.")
        return

    # CPU Profile Mode
    if args.profile:
        print("--- RUNNING CPU PROFILING (100 STEPS) ---")
        profile_steps = 100
        model.train()
        
        start_time = time.time()
        running_loss = 0.0
        step_times = []
        
        step = 0
        optimizer.zero_grad()
        
        while step < profile_steps:
            for x, y in train_loader:
                if step >= profile_steps:
                    break
                
                step_start = time.time()
                x, y = x.to(device), y.to(device)
                
                # Forward & backward pass
                logits, loss = model(x, y)
                loss_scaled = loss / grad_accum_steps
                loss_scaled.backward()
                
                if (step + 1) % grad_accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                
                step += 1
                step_time = time.time() - step_start
                step_times.append(step_time)
                running_loss += loss.item()
                
                if step % 10 == 0:
                    print(f"Profile Step {step}/{profile_steps} | Loss: {loss.item():.4f} | Step time: {step_time:.4f}s")
        
        total_time = time.time() - start_time
        avg_step_time = np.mean(step_times)
        steps_per_sec = 1.0 / avg_step_time if avg_step_time > 0 else 0.0
        tokens_per_sec = (batch_size * model_config.max_seq_len) * steps_per_sec
        cpu_mem = get_process_memory()
        
        # Run validation
        val_loss, perp = run_evaluation(model, val_loader, device)
        
        # Save profile checkpoint
        profile_cp_path = os.path.join(CHECKPOINT_DIR, "collision-1m-profile.pt")
        save_checkpoint(
            model, optimizer, scheduler, step, 0, running_loss / profile_steps, val_loss,
            model_config.__dict__, {"save_dir": TOKENIZER_DIR}, profile_cp_path
        )
        
        # Time estimations
        est_1k = time.strftime('%H:%M:%S', time.gmtime(1000 * avg_step_time))
        est_5k = time.strftime('%H:%M:%S', time.gmtime(5000 * avg_step_time))
        est_10k = time.strftime('%H:%M:%S', time.gmtime(10000 * avg_step_time))

        print("\n## COLLISION TRAINING PROFILE\n")
        print(f"Parameters:          {param_count:,}")
        print(f"Dataset tokens:      {dataset_tokens:,}")
        print(f"Tokens/sec:          {tokens_per_sec:,.2f}")
        print(f"Steps/sec:           {steps_per_sec:.2f}")
        print(f"Average step time:   {avg_step_time:.4f} seconds")
        print(f"CPU Memory Usage:    {cpu_mem:.1f} MB")
        print(f"Average Loss:        {running_loss / profile_steps:.4f}")
        print(f"Val Loss:            {val_loss:.4f}")
        print(f"Perplexity:          {perp:.2f}")
        print(f"Estimated 1K steps:  {est_1k}")
        print(f"Estimated 5K steps:  {est_5k}")
        print(f"Estimated 10K steps: {est_10k}")
        
        # Save logs
        profile_stats = {
            "parameters": param_count,
            "tokens_per_sec": tokens_per_sec,
            "steps_per_sec": steps_per_sec,
            "avg_step_time": avg_step_time,
            "est_1k": est_1k,
            "est_5k": est_5k,
            "est_10k": est_10k,
            "date": time.strftime('%Y-%m-%d %H:%M:%S')
        }
        with open(os.path.join(EXPERIMENT_DIR, "profile_stats.json"), "w", encoding="utf-8") as f:
            json.dump(profile_stats, f, indent=2)
            
        print("\nCOLLISION-1M Phase 2 profile completed successfully.")
        return

    # Normal training loop
    if dataset_tokens < 100000:
        print("\nWARNING: Dataset is extremely small. This is suitable for testing only.")
        try:
            response = input("Do you want to proceed with training anyway? (y/n): ")
            if response.strip().lower() != 'y':
                print("Training cancelled by user.")
                return
        except EOFError:
            print("Non-interactive environment detected. Auto-confirming warning.")
    elif dataset_tokens < 1000000:
        print("\nWARNING: This dataset is small for a serious language-model experiment.")
        try:
            response = input("Do you want to proceed with training anyway? (y/n): ")
            if response.strip().lower() != 'y':
                print("Training cancelled by user.")
                return
        except EOFError:
            print("Non-interactive environment detected. Auto-confirming warning.")

    print(f"Beginning training. Target steps: {total_train_steps}. Epochs: {args.epochs}")

    step = start_step
    epoch = start_epoch
    tokens_processed = step * batch_size * model_config.max_seq_len

    exp_log_path = os.path.join(EXPERIMENT_DIR, "training_log.csv")
    if not os.path.exists(exp_log_path):
        os.makedirs(EXPERIMENT_DIR, exist_ok=True)
        with open(exp_log_path, "w", encoding="utf-8") as f:
            f.write("timestamp,step,train_loss,val_loss,perplexity,cpu_util,tokens_processed\n")

    model.train()
    running_loss = 0.0
    start_time = time.time()
    optimizer.zero_grad()

    try:
        for ep in range(start_epoch, args.epochs):
            epoch = ep
            for x, y in train_loader:
                if step >= total_train_steps:
                    break

                x, y = x.to(device), y.to(device)
                
                # Gradient Accumulation
                logits, loss = model(x, y)
                loss_scaled = loss / grad_accum_steps
                loss_scaled.backward()
                
                if (step + 1) % grad_accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()

                step += 1
                tokens_processed += batch_size * model_config.max_seq_len
                running_loss += loss.item()

                if step % 10 == 0:
                    avg_train_loss = running_loss / 10
                    running_loss = 0.0
                    elapsed = time.time() - start_time
                    steps_per_sec = 10 / elapsed
                    start_time = time.time()
                    
                    remaining_steps = total_train_steps - step
                    eta_sec = remaining_steps / max(0.001, steps_per_sec)
                    eta_str = time.strftime('%H:%M:%S', time.gmtime(eta_sec))

                    cpu_util = get_cpu_info()
                    current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else optimizer.param_groups[0]['lr']
                    print(f"Step {step}/{total_train_steps} | Epoch {epoch+1} | Loss: {avg_train_loss:.4f} | LR: {current_lr:.6f} | "
                          f"Speed: {steps_per_sec:.2f} steps/s | ETA: {eta_str} | CPU: {cpu_util:.1f}% | Tokens: {tokens_processed}")

                if step % checkpoint_interval == 0:
                    val_loss, perp = run_evaluation(model, val_loader, device)
                    print(f"--- Running Validation ---")
                    print(f"Step {step} | Val Loss: {val_loss:.4f} | Perplexity: {perp:.2f}")
                    
                    # Save checkpoint
                    cp_name = f"collision-1m-step-{step:06d}.pt"
                    cp_path = os.path.join(CHECKPOINT_DIR, cp_name)
                    save_checkpoint(
                        model, optimizer, scheduler, step, epoch, loss.item(), val_loss,
                        model_config.__dict__, {"save_dir": TOKENIZER_DIR}, cp_path
                    )
                    save_checkpoint(
                        model, optimizer, scheduler, step, epoch, loss.item(), val_loss,
                        model_config.__dict__, {"save_dir": TOKENIZER_DIR}, latest_cp_path
                    )

                    with open(exp_log_path, "a", encoding="utf-8") as f:
                        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')},{step},{loss.item():.4f},{val_loss:.4f},{perp:.2f},{cpu_util:.1f},{tokens_processed}\n")

                    model.train()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving emergency checkpoint...")
        interrupted_cp_path = os.path.join(CHECKPOINT_DIR, "collision-1m-interrupted.pt")
        save_checkpoint(
            model, optimizer, scheduler, step, epoch, running_loss / max(1, step % 10), 0.0,
            model_config.__dict__, {"save_dir": TOKENIZER_DIR}, interrupted_cp_path
        )
        print(f"Emergency checkpoint saved to {interrupted_cp_path}. Exiting.")
        return

    # Final Save
    val_loss, perp = run_evaluation(model, val_loader, device)
    final_cp_path = os.path.join(CHECKPOINT_DIR, "latest.pt")
    save_checkpoint(
        model, optimizer, scheduler, step, epoch, running_loss / max(1, step % 10), val_loss,
        model_config.__dict__, {"save_dir": TOKENIZER_DIR}, final_cp_path
    )
    print("COLLISION-1M checkpoint saved successfully.")

if __name__ == "__main__":
    main()
