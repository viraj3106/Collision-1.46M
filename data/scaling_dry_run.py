import os
import sys
import yaml
import torch

# Resolve project root path and insert into Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model.config import ModelConfig
from model.transformer import CollisionTransformer
from data.tokenize import BPETokenizer

def get_param_count(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def dry_run_config(config_path):
    print(f"\n==================================================")
    print(f"Dry-running config: {config_path}")
    print(f"==================================================")
    
    if not os.path.exists(config_path):
        print(f"Error: Configuration file not found at {config_path}")
        return None
        
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
        
    model_cfg_dict = cfg.get("model", {})
    training_cfg_dict = cfg.get("training", {})
    
    # 1. Build the model config and construct model
    model_config = ModelConfig(**model_cfg_dict)
    model = CollisionTransformer(model_config)
    
    # 2 & 3. Calculate actual parameters
    total_params, trainable_params = get_param_count(model)
    print(f"Model constructed successfully.")
    print(f"Actual Total Parameters:     {total_params:,}")
    print(f"Actual Trainable Parameters: {trainable_params:,}")
    
    # Verify tokenizer path
    tokenizer_dir = training_cfg_dict.get("tokenizer_dir", "artifacts/tokenizer")
    if not os.path.exists(tokenizer_dir):
        print(f"Warning: Tokenizer directory not found at {tokenizer_dir}")
    else:
        # Try loading tokenizer
        tokenizer = BPETokenizer()
        tokenizer.load(tokenizer_dir)
        print(f"Tokenizer loaded successfully from {tokenizer_dir} (Vocab size: {len(tokenizer.vocab)})")
        
    # Verify dataset paths
    dataset_version = training_cfg_dict.get("dataset_version", "collision_dataset_v4")
    dataset_dir = os.path.join("datasets", dataset_version)
    train_bin = os.path.join(dataset_dir, "train.bin")
    val_bin = os.path.join(dataset_dir, "val.bin")
    
    if not os.path.exists(train_bin) or not os.path.exists(val_bin):
        print(f"Warning: Dataset binaries not found under {dataset_dir}")
    else:
        print(f"Dataset path validated successfully (train.bin and val.bin exist in {dataset_dir})")
        
    # 6. Verify optimizer construction
    lr = float(training_cfg_dict.get("learning_rate", 6e-4))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    print(f"Optimizer (AdamW) constructed successfully with learning rate {lr}")
    
    # Verify context length
    print(f"Context length (max_seq_len): {model_config.max_seq_len}")
    print(f"Seed: {training_cfg_dict.get('seed', 1337)}")
    print(f"Training Token Budget: {training_cfg_dict.get('training_token_budget', 1536000)}")
    
    return {
        "config_path": config_path,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "n_layer": model_config.n_layer,
        "d_model": model_config.d_model,
        "n_head": model_config.n_head,
        "max_seq_len": model_config.max_seq_len,
        "learning_rate": lr,
        "seed": training_cfg_dict.get("seed", 1337),
        "budget": training_cfg_dict.get("training_token_budget", 1536000)
    }

def main():
    configs = [
        "configs/scaling/collision_3m.yaml",
        "configs/scaling/collision_7m.yaml",
        "configs/scaling/collision_15m.yaml"
    ]
    
    results = []
    for c in configs:
        res = dry_run_config(c)
        if res:
            results.append(res)
            
    print("\n==================================================")
    print("               DRY RUN VERIFICATION SUMMARY       ")
    print("==================================================")
    for res in results:
        print(f"File: {os.path.basename(res['config_path'])} | Layers: {res['n_layer']} | d_model: {res['d_model']} | Heads: {res['n_head']} | Params: {res['total_params']:,}")
    print("==================================================\n")

if __name__ == "__main__":
    main()
