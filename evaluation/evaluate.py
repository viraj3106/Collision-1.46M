import os
import time
import json
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from model.config import ModelConfig
from model.transformer import CollisionTransformer
from data.tokenize import BPETokenizer
from training.train import TokenDataset, run_evaluation
from inference.generate import generate
from collision.config import CHECKPOINT_DIR, TOKENIZER_DIR, EXPERIMENT_DIR, PROCESSED_DATA_DIR

def main():
    parser = argparse.ArgumentParser(description="Evaluate COLLISION-1M models")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint .pt file")
    parser.add_argument("--device", type=str, default="cpu", help="Run device: cpu, cuda")
    args = parser.parse_args()

    # Load tokenizer
    tokenizer = BPETokenizer()
    tokenizer.load(TOKENIZER_DIR)

    # Locate checkpoint
    cp_path = args.checkpoint
    if cp_path is None:
        cp_path = os.path.join(CHECKPOINT_DIR, "latest.pt")

    if not os.path.exists(cp_path):
        print(f"Error: Checkpoint file not found at {cp_path}. Please train the model first.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"Loading checkpoint {cp_path} onto {device} for evaluation...")
    checkpoint = torch.load(cp_path, map_location=device)

    # Recreate config & model
    model_cfg = ModelConfig(**checkpoint["config"])
    model = CollisionTransformer(model_cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Load dataset validation slice
    val_bin = os.path.join(PROCESSED_DATA_DIR, "val.bin")
    if not os.path.exists(val_bin):
        print(f"Error: Processed validation data not found at {val_bin}. Run tokenization first.")
        return

    val_dataset = TokenDataset(val_bin, model_cfg.max_seq_len)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    print("Evaluating loss and perplexity...")
    val_loss, perplexity = run_evaluation(model, val_loader, device)

    print("\n--- EVALUATION RESULTS ---")
    print(f"Checkpoint Step: {checkpoint.get('step', 'N/A')}")
    print(f"Epochs Completed: {checkpoint.get('epoch', 'N/A') + 1}")
    print(f"Stored Training Loss: {checkpoint.get('train_loss', 0.0):.4f}")
    print(f"Calculated Validation Loss: {val_loss:.4f}")
    print(f"Calculated Perplexity: {perplexity:.2f}")
    print("--------------------------\n")

    # Generate samples from fixed prompts
    prompts = [
        "COLLISION-1M is",
        "We are training",
        "A language model learns"
    ]
    print("Generating test samples using checkpoint:")
    for prompt in prompts:
        gen = generate(model, tokenizer, prompt=prompt, max_tokens=30, temperature=0.7, device=device)
        print(f"Prompt: '{prompt}' | Output: '{gen}'")

    # Log details to experiments
    os.makedirs(EXPERIMENT_DIR, exist_ok=True)
    exp_record = {
        "date": time.strftime('%Y-%m-%d %H:%M:%S'),
        "checkpoint": os.path.basename(cp_path),
        "step": checkpoint.get('step', 0),
        "train_loss": checkpoint.get('train_loss', 0.0),
        "val_loss": val_loss,
        "perplexity": perplexity,
        "vocab_size": model_cfg.vocab_size,
        "d_model": model_cfg.d_model,
        "n_layer": model_cfg.n_layer,
        "n_head": model_cfg.n_head
    }
    
    exp_hist_path = os.path.join(EXPERIMENT_DIR, "experiments_history.jsonl")
    with open(exp_hist_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(exp_record) + "\n")
    print(f"Logged experiment results to {exp_hist_path}")

    # Write formatted markdown report
    report_path = os.path.join(EXPERIMENT_DIR, "evaluation_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# COLLISION-1M Evaluation Report\n\n")
        f.write(f"Generated on: {exp_record['date']}\n\n")
        f.write(f"## Model Configuration\n")
        f.write(f"- **Checkpoint**: {exp_record['checkpoint']}\n")
        f.write(f"- **Step**: {exp_record['step']}\n")
        f.write(f"- **Layers**: {exp_record['n_layer']}\n")
        f.write(f"- **Heads**: {exp_record['n_head']}\n")
        f.write(f"- **Embedding Dim**: {exp_record['d_model']}\n")
        f.write(f"- **Vocab Size**: {exp_record['vocab_size']}\n\n")
        f.write(f"## Performance Metrics\n")
        f.write(f"| Metric | Value |\n")
        f.write(f"| --- | --- |\n")
        f.write(f"| Train Loss | {exp_record['train_loss']:.4f} |\n")
        f.write(f"| Val Loss | {val_loss:.4f} |\n")
        f.write(f"| Perplexity | {perplexity:.2f} |\n\n")
        f.write(f"## Sample Generations\n")
        for prompt in prompts:
            gen = generate(model, tokenizer, prompt=prompt, max_tokens=30, temperature=0.7, device=device)
            f.write(f"- **Prompt**: *\"{prompt}\"*\n  - **Generation**: \"{gen}\"\n")
    print(f"Logged formatted markdown report to {report_path}")


if __name__ == "__main__":
    main()
