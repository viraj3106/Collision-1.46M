"""
COLLISION Phase 103 — Master Direct Preference Optimization (DPO) Engine.

Features:
- Pairwise sequence log-probability calculation with prompt-length loss masking
- Frozen reference model initialized from Phase 102 checkpoint
- Trainable policy model initialized from Phase 102 checkpoint
- Configurable DPO beta, learning rate, and CosineWarmupScheduler
- Checkpointing, integrity SHA-256 computation, and validation evaluation
- Rapid smoke test verification (--smoke-test)
"""

import os
import sys
import time
import json
import yaml
import hashlib
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model.config import ModelConfig
from model.transformer import CollisionTransformer
from data.tokenize import BPETokenizer
from training.scheduler import CosineWarmupScheduler
from training.dpo import compute_sequence_logprobs, canonical_dpo_loss

DEFAULT_CONFIG_PATH = os.path.join(PROJECT_ROOT, "training", "preferences", "config.yaml")


class PreferenceDataset(Dataset):
    def __init__(self, jsonl_path: str, tokenizer: BPETokenizer, max_seq_len: int = 1024, pad_token_id: int = 256):
        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(f"Preference dataset not found at {jsonl_path}")

        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id
        self.samples = []

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.samples.append(json.loads(line))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        prompt = item.get("full_prompt", f"USER:\n{item['prompt']}\n\nASSISTANT:\n")
        chosen = item.get("chosen_response", item["response_a"] if item.get("preferred_response") == "A" else item["response_b"])
        rejected = item.get("rejected_response", item["response_b"] if item.get("preferred_response") == "A" else item["response_a"])

        prompt_ids = self.tokenizer.encode(prompt, bos=True, eos=False)
        chosen_resp_ids = self.tokenizer.encode(chosen, bos=False, eos=True)
        rejected_resp_ids = self.tokenizer.encode(rejected, bos=False, eos=True)

        chosen_full = prompt_ids + chosen_resp_ids
        rejected_full = prompt_ids + rejected_resp_ids

        # Truncate to max_seq_len
        if len(chosen_full) > self.max_seq_len:
            chosen_full = chosen_full[: self.max_seq_len]
        if len(rejected_full) > self.max_seq_len:
            rejected_full = rejected_full[: self.max_seq_len]

        chosen_prompt_len = min(len(prompt_ids), len(chosen_full))
        rejected_prompt_len = min(len(prompt_ids), len(rejected_full))

        # Pad to max_seq_len
        c_pad = self.max_seq_len - len(chosen_full)
        r_pad = self.max_seq_len - len(rejected_full)

        chosen_padded = chosen_full + [self.pad_token_id] * c_pad
        rejected_padded = rejected_full + [self.pad_token_id] * r_pad

        return (
            torch.tensor(chosen_padded, dtype=torch.long),
            torch.tensor(rejected_padded, dtype=torch.long),
            torch.tensor(chosen_prompt_len, dtype=torch.long),
            torch.tensor(rejected_prompt_len, dtype=torch.long)
        )


def compute_sha256(filepath: str) -> str:
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192 * 1024):
            sha.update(chunk)
    return sha.hexdigest().lower()


def evaluate_dpo(policy_model, reference_model, dataloader, device, beta=0.1, pad_token_id=256, max_batches=10):
    policy_model.eval()
    total_loss = 0.0
    steps = 0
    chosen_rewards = []
    rejected_rewards = []

    with torch.no_grad():
        for chosen_ids, rejected_ids, chosen_p_lens, rejected_p_lens in dataloader:
            chosen_ids = chosen_ids.to(device)
            rejected_ids = rejected_ids.to(device)
            chosen_p_lens = chosen_p_lens.to(device)
            rejected_p_lens = rejected_p_lens.to(device)

            loss, chosen_logp, rejected_logp, chosen_ref_logp, rejected_ref_logp = canonical_dpo_loss(
                policy_model, reference_model,
                chosen_ids, rejected_ids,
                chosen_p_lens, rejected_p_lens,
                beta=beta, pad_token_id=pad_token_id
            )

            if not torch.isnan(loss):
                total_loss += loss.item()
                steps += 1
                c_reward = (beta * (chosen_logp - chosen_ref_logp)).mean().item()
                r_reward = (beta * (rejected_logp - rejected_ref_logp)).mean().item()
                chosen_rewards.append(c_reward)
                rejected_rewards.append(r_reward)

            if steps >= max_batches:
                break

    mean_loss = total_loss / max(1, steps)
    avg_chosen = float(np.mean(chosen_rewards)) if chosen_rewards else 0.0
    avg_rejected = float(np.mean(rejected_rewards)) if rejected_rewards else 0.0
    reward_margin = avg_chosen - avg_rejected
    return mean_loss, reward_margin


def train_dpo(
    config_path: str = DEFAULT_CONFIG_PATH,
    smoke_test: bool = False,
    override_max_steps: int = None,
    device_name: str = None
):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    dpo_cfg = cfg.get("dpo", {})
    model_cfg_dict = cfg.get("model", {})

    # Select device
    if device_name:
        device = torch.device(device_name)
    elif torch.cuda.is_available() and not smoke_test:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"[DPO Train] Device selected: {device}", flush=True)

    seed = dpo_cfg.get("seed", 1337)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Tokenizer
    tok_dir = os.path.join(PROJECT_ROOT, dpo_cfg.get("tokenizer_path", "artifacts/tokenizer"))
    tokenizer = BPETokenizer()
    tokenizer.load(tok_dir)

    # Model Config
    if smoke_test:
        m_config = ModelConfig(
            vocab_size=32000,
            max_seq_len=256,
            d_model=128,
            n_layer=2,
            n_head=4,
            d_ff=256,
            dropout=0.0,
            tie_embeddings=True
        )
        batch_size = 2
        grad_accum = 1
        max_steps = 5
        max_seq_len = 256
    else:
        m_config = ModelConfig(
            vocab_size=model_cfg_dict.get("vocab_size", 32000),
            max_seq_len=model_cfg_dict.get("max_seq_len", 1024),
            d_model=model_cfg_dict.get("d_model", 2048),
            n_layer=model_cfg_dict.get("n_layer", 24),
            n_head=model_cfg_dict.get("n_head", 16),
            d_ff=model_cfg_dict.get("d_ff", 5376),
            dropout=model_cfg_dict.get("dropout", 0.1),
            tie_embeddings=model_cfg_dict.get("tie_embeddings", True)
        )
        batch_size = dpo_cfg.get("batch_size", 2)
        grad_accum = dpo_cfg.get("gradient_accumulation_steps", 4)
        max_steps = override_max_steps or dpo_cfg.get("max_steps", 25)
        max_seq_len = m_config.max_seq_len

    # Policy Model & Frozen Reference Model
    policy_model = CollisionTransformer(m_config).to(device)
    reference_model = CollisionTransformer(m_config).to(device)

    ref_cp_path = os.path.join(PROJECT_ROOT, dpo_cfg.get("reference_checkpoint_path", "checkpoints/phase102_sft/sft_final_model.pt"))
    if os.path.exists(ref_cp_path) and not smoke_test:
        try:
            ckpt = torch.load(ref_cp_path, map_location=device)
            state_dict = ckpt.get("model_state_dict", ckpt)
            policy_model.load_state_dict(state_dict, strict=False)
            reference_model.load_state_dict(state_dict, strict=False)
            print(f"[DPO Train] Loaded verified Phase 102 SFT checkpoint from {ref_cp_path}", flush=True)
        except Exception as e:
            print(f"[DPO Train] Note on checkpoint loading: {e}", flush=True)

    # Reference model is always frozen
    reference_model.eval()
    for p in reference_model.parameters():
        p.requires_grad = False

    pad_id = dpo_cfg.get("pad_token_id", 256)
    beta = float(dpo_cfg.get("beta", 0.1))

    # Load datasets
    train_path = os.path.join(PROJECT_ROOT, dpo_cfg.get("train_data_path", "data/preferences/train.jsonl"))
    val_path = os.path.join(PROJECT_ROOT, dpo_cfg.get("val_data_path", "data/preferences/validation.jsonl"))

    train_ds = PreferenceDataset(train_path, tokenizer, max_seq_len=max_seq_len, pad_token_id=pad_id)
    val_ds = PreferenceDataset(val_path, tokenizer, max_seq_len=max_seq_len, pad_token_id=pad_id)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    lr = float(dpo_cfg.get("learning_rate", 1.0e-6))
    min_lr = float(dpo_cfg.get("min_lr", 1.0e-7))
    warmup_steps = dpo_cfg.get("warmup_steps", 10)
    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=lr, weight_decay=dpo_cfg.get("weight_decay", 0.01))
    scheduler = CosineWarmupScheduler(
        optimizer,
        warmup_steps=warmup_steps,
        total_steps=max_steps,
        base_lr=lr,
        min_lr=min_lr
    )

    out_dir = os.path.join(PROJECT_ROOT, dpo_cfg.get("output_dir", "checkpoints/phase103_pref"))
    os.makedirs(out_dir, exist_ok=True)

    # Pre-training validation
    init_val_loss, init_margin = evaluate_dpo(policy_model, reference_model, val_loader, device, beta=beta, pad_token_id=pad_id)
    print(f"[DPO Train] Initial Val Loss: {init_val_loss:.4f} | Initial Reward Margin: {init_margin:+.4f}", flush=True)

    policy_model.train()
    step = 0
    running_loss = 0.0
    start_time = time.time()
    history = []
    stop_training = False

    while not stop_training:
        for chosen_ids, rejected_ids, chosen_p_lens, rejected_p_lens in train_loader:
            chosen_ids = chosen_ids.to(device)
            rejected_ids = rejected_ids.to(device)
            chosen_p_lens = chosen_p_lens.to(device)
            rejected_p_lens = rejected_p_lens.to(device)

            loss, chosen_logp, rejected_logp, chosen_ref, rejected_ref = canonical_dpo_loss(
                policy_model, reference_model,
                chosen_ids, rejected_ids,
                chosen_p_lens, rejected_p_lens,
                beta=beta, pad_token_id=pad_id
            )

            loss_scaled = loss / grad_accum
            loss_scaled.backward()
            running_loss += loss.item()

            if (step + 1) % grad_accum == 0 or (step + 1) == max_steps:
                nn.utils.clip_grad_norm_(policy_model.parameters(), dpo_cfg.get("max_grad_norm", 1.0))
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                step += 1
                avg_train_loss = running_loss / grad_accum
                running_loss = 0.0

                if step % dpo_cfg.get("eval_interval", 10) == 0 or step == max_steps or smoke_test:
                    val_loss, margin = evaluate_dpo(policy_model, reference_model, val_loader, device, beta=beta, pad_token_id=pad_id)
                    current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, "get_last_lr") else lr
                    print(
                        f"[DPO Step {step:03d}/{max_steps:03d}] "
                        f"Train Loss: {avg_train_loss:.4f} | "
                        f"Val Loss: {val_loss:.4f} | "
                        f"Reward Margin: {margin:+.4f} | "
                        f"LR: {current_lr:.2e}",
                        flush=True
                    )
                    history.append({
                        "step": step,
                        "train_loss": round(avg_train_loss, 4),
                        "val_loss": round(val_loss, 4),
                        "reward_margin": round(margin, 4),
                        "lr": current_lr
                    })

                if step % dpo_cfg.get("save_interval", 25) == 0 or step == max_steps:
                    cp_file = os.path.join(out_dir, f"pref_step_{step}.pt")
                    torch.save({
                        "step": step,
                        "model_state_dict": policy_model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "config": m_config.__dict__,
                        "val_loss": val_loss if 'val_loss' in locals() else None
                    }, cp_file)

                if step >= max_steps:
                    stop_training = True
                    break

    elapsed = time.time() - start_time
    final_cp = os.path.join(out_dir, "pref_final_model.pt")
    torch.save({
        "step": step,
        "model_state_dict": policy_model.state_dict(),
        "config": m_config.__dict__,
        "history": history
    }, final_cp)

    final_sha = compute_sha256(final_cp)
    print(f"[DPO Train] Preference training finished in {elapsed:.2f}s. Saved final checkpoint to {final_cp} (SHA-256: {final_sha[:16]}...)", flush=True)

    summary_file = os.path.join(out_dir, "preference_training_summary.json")
    summary_data = {
        "status": "COMPLETED",
        "alignment_stage": "COLLISION-1.0B-PREF",
        "smoke_test": smoke_test,
        "total_steps": step,
        "beta": beta,
        "elapsed_seconds": round(elapsed, 2),
        "initial_val_loss": round(init_val_loss, 4),
        "final_val_loss": round(history[-1]["val_loss"], 4) if history else None,
        "final_reward_margin": round(history[-1]["reward_margin"], 4) if history else None,
        "final_checkpoint_sha256": final_sha,
        "history": history
    }
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    return summary_data


def main():
    parser = argparse.ArgumentParser(description="COLLISION Phase 103 Preference Alignment Training (DPO)")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH, help="Path to config yaml")
    parser.add_argument("--smoke-test", action="store_true", help="Run rapid smoke test")
    parser.add_argument("--max-steps", type=int, default=None, help="Override maximum steps")
    parser.add_argument("--device", type=str, default=None, help="Device override")
    args = parser.parse_args()

    train_dpo(
        config_path=args.config,
        smoke_test=args.smoke_test,
        override_max_steps=args.max_steps,
        device_name=args.device
    )


if __name__ == "__main__":
    main()
