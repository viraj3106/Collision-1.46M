import os
import time
import argparse
import psutil
import yaml
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

class TokenDataset(Dataset):
    def __init__(self, bin_path: str, seq_len: int):
        if not os.path.exists(bin_path):
            raise FileNotFoundError(f"Binary token file not found at {bin_path}. Run tokenization first.")
        # Load the binary data into memory using uint16
        self.data = np.fromfile(bin_path, dtype=np.uint16)
        self.seq_len = seq_len

    def __len__(self):
        # We need seq_len + 1 tokens for x and y
        return max(0, len(self.data) - self.seq_len - 1)

    def __getitem__(self, idx):
        # Slice the array
        x = torch.from_numpy(self.data[idx : idx + self.seq_len].astype(np.int64))
        y = torch.from_numpy(self.data[idx + 1 : idx + self.seq_len + 1].astype(np.int64))
        return x, y

def get_cpu_info():
    try:
        return psutil.cpu_percent()
    except Exception:
        return 0.0

def generate_text_sample(model, tokenizer, prompt="The system", max_tokens=30, device="cpu"):
    model.eval()
    ids = tokenizer.encode(prompt, bos=True)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    
    with torch.no_grad():
        for _ in range(max_tokens):
            # Crop inputs to max context length
            x_cond = x if x.size(1) <= model.config.max_seq_len else x[:, -model.config.max_seq_len:]
            logits, _ = model(x_cond)
            # Focus on next token logits
            next_token_logits = logits[0, -1, :]
            # Simple greedy or top-k sampling (greedy here for basic verification)
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
            if steps >= 10:  # Cap validation steps for CPU speed
                break
    mean_loss = total_loss / max(1, steps)
    perplexity = np.exp(mean_loss) if mean_loss < 20 else float('inf')
    return mean_loss, perplexity

def main():
    parser = argparse.ArgumentParser(description="Train COLLISION-1M decoder model")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH, help="Path to config yaml")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=6e-4, help="Learning rate")
    parser.add_argument("--max-steps", type=int, default=1000, help="Maximum number of training steps")
    parser.add_argument("--checkpoint-interval", type=int, default=100, help="Save checkpoint every N steps")
    parser.add_argument("--device", type=str, default="cpu", help="device to train on: cpu, cuda")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--smoke-test", action="store_true", help="Run a quick smoke test verify loop")
    parser.add_argument("--cpu-safe", action="store_true", help="Use conservative safe configurations for standard CPU laptop")
    args = parser.parse_args()

    # Apply cpu-safe overrides
    if args.cpu_safe:
        args.batch_size = 4
        args.checkpoint_interval = 25
        print("CPU_SAFE mode active: batch size set to 4, checkpoint interval set to 25.")

    # Initialize device
    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"Using training device: {device}")

    # Load configuration
    model_config = ModelConfig.from_yaml(args.config)
    
    # Load tokenizer
    tokenizer = BPETokenizer()
    try:
        tokenizer.load(TOKENIZER_DIR)
        print("Tokenizer loaded successfully.")
    except Exception:
        print("Error: Tokenizer files not found. Training a default tokenizer on sample.txt first...")
        from data.tokenize import main as train_tok
        train_tok()
        tokenizer.load(TOKENIZER_DIR)

    # Initialize model
    model = CollisionTransformer(model_config).to(device)
    param_count = model.get_parameter_count()
    print(f"COLLISION-1M model initialized. Programmatic parameter count: {param_count:,}")

    # Datasets
    train_bin = os.path.join(PROCESSED_DATA_DIR, "train.bin")
    val_bin = os.path.join(PROCESSED_DATA_DIR, "val.bin")
    
    if not os.path.exists(train_bin) or not os.path.exists(val_bin):
        print("Tokenized binaries not found. Automatically running data.tokenize...")
        from data.tokenize import main as run_tok
        run_tok()

    train_dataset = TokenDataset(train_bin, model_config.max_seq_len)
    val_dataset = TokenDataset(val_bin, model_config.max_seq_len)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    
    # Warmup and decay setup
    total_train_steps = min(args.max_steps, len(train_loader) * args.epochs)
    warmup_steps = int(0.1 * total_train_steps)
    scheduler = CosineWarmupScheduler(optimizer, warmup_steps, total_train_steps, args.learning_rate, min_lr=args.learning_rate*0.1)

    start_step = 0
    start_epoch = 0
    best_val_loss = float('inf')

    # Handle resume training
    latest_cp_path = os.path.join(CHECKPOINT_DIR, "latest.pt")
    if args.resume and os.path.exists(latest_cp_path):
        print(f"Resuming training from checkpoint: {latest_cp_path}")
        checkpoint = load_checkpoint(latest_cp_path, model, optimizer, scheduler)
        start_step = checkpoint["step"]
        start_epoch = checkpoint["epoch"]
        best_val_loss = checkpoint.get("val_loss", float('inf'))

    # Handle smoke-test override
    if args.smoke_test:
        print("--- RUNNING COLLISION-1M SMOKE TEST ---")
        # Run 5 training steps, 2 validation steps, generate, save temporary checkpoint, load, print parameters.
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

        # Text generation
        gen_text = generate_text_sample(model, tokenizer, prompt="COLLISION-1M is", max_tokens=15, device=device)
        print(f"Smoke generated sample: {gen_text}")

        # Save temporary checkpoint
        test_cp_path = os.path.join(CHECKPOINT_DIR, "smoke_test_checkpoint.pt")
        save_checkpoint(
            model, optimizer, scheduler, smoke_steps, 0, loss.item(), val_loss,
            model_config.__dict__, {"save_dir": TOKENIZER_DIR}, test_cp_path
        )
        # Verify load
        load_checkpoint(test_cp_path, model)
        print("Smoke test checkpoint verified and successfully loaded.")
        print("COLLISION-1M initialized successfully.")
        return

    # Normal training loop
    print(f"Beginning training. Target steps: {total_train_steps}. Epochs: {args.epochs}")
    step = start_step
    epoch = start_epoch
    tokens_processed = step * args.batch_size * model_config.max_seq_len

    # Track experiments metadata
    exp_log_path = os.path.join(EXPERIMENT_DIR, "training_log.csv")
    if not os.path.exists(exp_log_path):
        os.makedirs(EXPERIMENT_DIR, exist_ok=True)
        with open(exp_log_path, "w", encoding="utf-8") as f:
            f.write("timestamp,step,train_loss,val_loss,perplexity,cpu_util,tokens_processed\n")

    model.train()
    running_loss = 0.0
    start_time = time.time()

    for ep in range(start_epoch, args.epochs):
        epoch = ep
        for x, y in train_loader:
            if step >= total_train_steps:
                break

            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits, loss = model(x, y)
            loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            step += 1
            tokens_processed += args.batch_size * model_config.max_seq_len
            running_loss += loss.item()

            # Logging & checkpoints
            if step % 10 == 0:
                avg_train_loss = running_loss / 10
                running_loss = 0.0
                elapsed = time.time() - start_time
                steps_per_sec = 10 / elapsed
                start_time = time.time()
                
                # Estimate remaining time
                remaining_steps = total_train_steps - step
                eta_sec = remaining_steps / max(0.001, steps_per_sec)
                eta_str = time.strftime('%H:%M:%S', time.gmtime(eta_sec))

                cpu_util = get_cpu_info()
                current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else optimizer.param_groups[0]['lr']
                print(f"Step {step}/{total_train_steps} | Epoch {epoch+1} | Loss: {avg_train_loss:.4f} | LR: {current_lr:.6f} | "
                      f"Speed: {steps_per_sec:.2f} steps/s | ETA: {eta_str} | CPU: {cpu_util:.1f}% | Tokens: {tokens_processed}")


            if step % args.checkpoint_interval == 0:
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
                # Keep latest updated
                save_checkpoint(
                    model, optimizer, scheduler, step, epoch, loss.item(), val_loss,
                    model_config.__dict__, {"save_dir": TOKENIZER_DIR}, latest_cp_path
                )

                # Log metadata
                with open(exp_log_path, "a", encoding="utf-8") as f:
                    f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')},{step},{loss.item():.4f},{val_loss:.4f},{perp:.2f},{cpu_util:.1f},{tokens_processed}\n")

                model.train()

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
