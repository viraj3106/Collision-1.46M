import os
import argparse
import torch
import numpy as np

from model.config import ModelConfig
from model.transformer import CollisionTransformer
from data.tokenize import BPETokenizer
from inference.generate import generate
from collision.config import CHECKPOINT_DIR, TOKENIZER_DIR

def main():
    parser = argparse.ArgumentParser(description="Compare COLLISION-1M Checkpoints")
    parser.add_argument("--prompts-file", type=str, default="evaluation/prompts.txt", help="Path to prompts file")
    parser.add_argument("--device", type=str, default="cpu", help="Device: cpu, cuda")
    args = parser.parse_args()

    # Load prompts
    if not os.path.exists(args.prompts_file):
        print(f"Error: Prompts file not found at {args.prompts_file}")
        return
    with open(args.prompts_file, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]

    # Load tokenizer
    tokenizer = BPETokenizer()
    tokenizer.load(TOKENIZER_DIR)

    # Discovers checkpoints
    if not os.path.exists(CHECKPOINT_DIR):
        print(f"No checkpoints folder found at {CHECKPOINT_DIR}")
        return

    checkpoint_files = [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".pt") and f != "latest.pt"]
    checkpoint_files = sorted(checkpoint_files)
    
    # Also add latest.pt to end if exists
    if os.path.exists(os.path.join(CHECKPOINT_DIR, "latest.pt")):
        checkpoint_files.append("latest.pt")

    if not checkpoint_files:
        print("No checkpoints found to compare.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu")

    results = []

    print("CHECKPOINT COMPARISON\n")
    print(f"{'Checkpoint':<30} | {'Step':<8} | {'Train Loss':<10} | {'Val Loss':<10} | {'Perplexity':<10}")
    print("-" * 80)

    for cp_name in checkpoint_files:
        cp_path = os.path.join(CHECKPOINT_DIR, cp_name)
        try:
            checkpoint = torch.load(cp_path, map_location=device)
            step = checkpoint.get("step", 0)
            train_loss = checkpoint.get("train_loss", 0.0)
            val_loss = checkpoint.get("val_loss", 0.0)
            perp = np.exp(val_loss) if val_loss < 20 else float('inf')
            
            print(f"{cp_name:<30} | {step:<8} | {train_loss:<10.4f} | {val_loss:<10.4f} | {perp:<10.2f}")
            
            results.append({
                "name": cp_name,
                "path": cp_path,
                "step": step,
                "checkpoint_data": checkpoint
            })
        except Exception as e:
            print(f"Error loading {cp_name}: {e}")

    print("\n" + "=" * 80 + "\n")
    print("GENERATED SAMPLE COMPARISONS FOR PROMPTS:\n")

    for prompt in prompts:
        print(f"Prompt: \"{prompt}\"")
        print("-" * 40)
        for res in results:
            cp_name = res["name"]
            checkpoint = res["checkpoint_data"]
            
            # Init model
            model_cfg = ModelConfig(**checkpoint["config"])
            model = CollisionTransformer(model_cfg).to(device)
            model.load_state_dict(checkpoint["model_state_dict"])
            
            # Generate
            out = generate(model, tokenizer, prompt=prompt, max_tokens=25, temperature=0.7, device=device)
            safe_out = out.encode('ascii', errors='replace').decode('ascii')
            print(f"  [{cp_name}]: \"{safe_out}\"")

        print()

if __name__ == "__main__":
    main()
