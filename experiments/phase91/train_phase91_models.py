import os
import sys
import time
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model.config import ModelConfig
from model.transformer import CollisionTransformer
from training.checkpoint import save_checkpoint
from training.scheduler import CosineWarmupScheduler

class NonOverlappingTokenDataset(Dataset):
    def __init__(self, bin_path: str, seq_len: int):
        self.data = np.fromfile(bin_path, dtype=np.uint16)
        self.seq_len = seq_len

    def __len__(self):
        return max(0, (len(self.data) - 1) // self.seq_len)

    def __getitem__(self, idx):
        start = idx * self.seq_len
        x = torch.from_numpy(self.data[start : start + self.seq_len].astype(np.int64))
        y = torch.from_numpy(self.data[start + 1 : start + self.seq_len + 1].astype(np.int64))
        return x, y

def train_model(dataset_dir: str, save_dir: str, config_name: str, random_seed: int = 42):
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{config_name}] Training starting on device: {device}...")

    config_path = os.path.join(PROJECT_ROOT, "configs/collision_10m.yaml")
    cfg = ModelConfig.from_yaml(config_path)
    model = CollisionTransformer(cfg).to(device)

    train_bin = os.path.join(dataset_dir, "train.bin")
    val_bin = os.path.join(dataset_dir, "val.bin")

    seq_len = cfg.max_seq_len
    batch_size = 16
    lr = 5e-4
    total_steps = 3000

    train_dataset = NonOverlappingTokenDataset(train_bin, seq_len)
    val_dataset = NonOverlappingTokenDataset(val_bin, seq_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = CosineWarmupScheduler(optimizer, warmup_steps=200, total_steps=total_steps, base_lr=lr, min_lr=1e-5)

    os.makedirs(save_dir, exist_ok=True)
    history = []

    model.train()
    step = 0
    start_time = time.time()

    data_iter = iter(train_loader)

    for step in range(1, total_steps + 1):
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            x, y = next(data_iter)

        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if step % 500 == 0 or step == total_steps:
            model.eval()
            val_loss = 0.0
            val_steps = 0
            with torch.no_grad():
                for vx, vy in val_loader:
                    vx, vy = vx.to(device), vy.to(device)
                    _, vloss = model(vx, vy)
                    val_loss += vloss.item()
                    val_steps += 1
            val_loss = val_loss / max(1, val_steps)
            val_ppl = np.exp(val_loss) if val_loss < 20 else float('inf')
            model.train()

            elapsed = time.time() - start_time
            print(f"[{config_name}] Step {step}/{total_steps} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f} | Time: {elapsed:.1f}s")
            
            history.append({
                "step": step,
                "train_loss": round(loss.item(), 4),
                "val_loss": round(val_loss, 4),
                "val_perplexity": round(val_ppl, 2),
                "tokens_seen": step * batch_size * seq_len,
                "elapsed_seconds": round(elapsed, 1)
            })

    # Save final model
    model_save_path = os.path.join(save_dir, "model.pt")
    torch.save(model.state_dict(), model_save_path)
    print(f"[{config_name}] Saved final model checkpoint to {model_save_path}")

    config_info = {
        "config_name": config_name,
        "dataset_dir": dataset_dir,
        "checkpoint_path": model_save_path,
        "random_seed": random_seed,
        "batch_size": batch_size,
        "sequence_length": seq_len,
        "learning_rate": lr,
        "total_steps": total_steps,
        "training_tokens": total_steps * batch_size * seq_len,
        "final_train_loss": round(loss.item(), 4),
        "final_val_loss": round(val_loss, 4),
        "final_val_ppl": round(val_ppl, 2)
    }

    return config_info, history

def main():
    v5_dir = os.path.join(PROJECT_ROOT, "datasets/collision_dataset_v5_expanded")
    v9_dir = os.path.join(PROJECT_ROOT, "datasets/collision_dataset_v9_redesigned")

    control_save = os.path.join(PROJECT_ROOT, "models/phase91_control_10m")
    v9_save = os.path.join(PROJECT_ROOT, "models/phase91_v9_10m")

    # 1. Train Control Model
    control_cfg, control_hist = train_model(v5_dir, control_save, "phase91_control_10m")
    
    # 2. Train Experiment Model
    v9_cfg, v9_hist = train_model(v9_dir, v9_save, "phase91_v9_10m")

    out_dir = os.path.join(PROJECT_ROOT, "experiments/phase91")
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "phase91_control_training_config.json"), "w") as f:
        json.dump(control_cfg, f, indent=2)

    with open(os.path.join(out_dir, "phase91_v9_training_config.json"), "w") as f:
        json.dump(v9_cfg, f, indent=2)

    curves = {
        "control": control_hist,
        "v9_experiment": v9_hist
    }
    with open(os.path.join(out_dir, "phase91_training_curves.json"), "w") as f:
        json.dump(curves, f, indent=2)

    print("Phase 91 Controlled Retraining Complete!")

if __name__ == "__main__":
    main()
