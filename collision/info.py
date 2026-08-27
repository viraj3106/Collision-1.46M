import os
import argparse
import torch
from model.config import ModelConfig
from collision.config import DEFAULT_CONFIG_PATH, CHECKPOINT_DIR

def get_training_status():
    if not os.path.exists(CHECKPOINT_DIR):
        return "NOT TRAINED"
    files = [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".pt")]
    if not files:
        return "NOT TRAINED"
    return f"TRAINED (latest: {sorted(files)[-1]})"

def main():
    parser = argparse.ArgumentParser(description="Show COLLISION Model Information")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH, help="Path to config yaml")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Error: Config file not found at {args.config}")
        return

    config = ModelConfig.from_yaml(args.config)
    param_count = config.calculate_parameter_count()
    status = get_training_status()
    device = "CUDA" if torch.cuda.is_available() else "CPU"

    print("## COLLISION-1M\n")
    print(f"Architecture: Decoder Transformer")
    print(f"Parameters: {param_count:,}")
    print(f"Vocabulary: {config.vocab_size}")
    print(f"Context: {config.max_seq_len}")
    print(f"Layers: {config.n_layer}")
    print(f"Heads: {config.n_head}")
    print(f"Embedding: {config.d_model}")
    print(f"Device: {device}")
    print(f"Training status: {status}")

if __name__ == "__main__":
    main()
