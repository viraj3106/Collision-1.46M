import os
import json
import time
import torch
import numpy as np

from model.config import ModelConfig
from model.transformer import CollisionTransformer
from data.tokenize import BPETokenizer
from data.stats import get_latest_version_dir
from collision.config import CHECKPOINT_DIR, TOKENIZER_DIR, EXPERIMENT_DIR, PROCESSED_DATA_DIR
from inference.generate import generate
from training.train import get_process_memory, get_cpu_info

def check_readiness():
    print("==================================================")
    print("COLLISION-1M PRE-TRAINING READINESS CHECK")
    print("==================================================\n")

    latest_dir = get_latest_version_dir()
    if not latest_dir:
        print("Error: No prepared dataset version directory found.")
        print("NOT READY — DATASET TOO SMALL (0 tokens).")
        return

    meta_path = os.path.join(latest_dir, "metadata.json")
    if not os.path.exists(meta_path):
        print(f"Error: metadata.json missing in {latest_dir}")
        return

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    # 1. Dataset Integrity Checks
    print("--- 1. DATASET INTEGRITY VERIFICATION ---")
    train_bin = os.path.join(latest_dir, "train.bin")
    val_bin = os.path.join(latest_dir, "val.bin")
    
    train_exists = os.path.exists(train_bin)
    val_exists = os.path.exists(val_bin)
    vocab_exists = os.path.exists(os.path.join(TOKENIZER_DIR, "vocab.json"))
    merges_exists = os.path.exists(os.path.join(TOKENIZER_DIR, "merges.json"))
    
    print(f"  train.bin exists:        {train_exists}")
    print(f"  val.bin exists:          {val_exists}")
    print(f"  tokenizer exists:        {vocab_exists and merges_exists}")
    print(f"  metadata.json exists:    True")
    print(f"  dataset version:         {meta.get('dataset_version', 'N/A')}")
    
    if not (train_exists and val_exists and vocab_exists and merges_exists):
        print("Integrity Check: FAILED (missing binary or tokenizer files)")
        return

    # Check token count matches actual binary files
    actual_train_tokens = os.path.getsize(train_bin) // 2
    actual_val_tokens = os.path.getsize(val_bin) // 2
    actual_total_tokens = actual_train_tokens + actual_val_tokens
    meta_total_tokens = meta.get("token_count", 0)
    
    match = actual_total_tokens == meta_total_tokens
    print(f"  Token count matches binary: {match} (Actual: {actual_total_tokens:,}, Metadata: {meta_total_tokens:,})")
    
    if not match:
        print("Integrity Check: FAILED (token count mismatch)")
        return
    print("Integrity Check: PASSED\n")

    # 2. Dataset Size Check
    print("--- 2. DATASET SIZE VERIFICATION ---")
    if actual_total_tokens < 1_000_000:
        print("  COLLISION IS NOT READY FOR SERIOUS TRAINING.")
        readiness = "NOT READY"
        reason = f"Dataset size is {actual_total_tokens:,} tokens. Minimum required is 1,000,000 tokens (missing {1_000_000 - actual_total_tokens:,} tokens)."
    else:
        print("  COLLISION DATASET IS LARGE ENOUGH FOR THE FIRST REAL EXPERIMENT.")
        readiness = "READY FOR FIRST REAL TRAINING"
        reason = "Dataset meets the minimum target of 1,000,000 tokens."
    print(f"  Decision status: {readiness}\n")

    # 3. Tokenizer Verification
    print("--- 3. TOKENIZER ENCODE/DECODE VERIFICATION ---")
    tokenizer = BPETokenizer()
    tokenizer.load(TOKENIZER_DIR)
    
    test_phrases = [
        "The future of artificial intelligence.",
        "COLLISION is learning from data.",
        "Computers can process information.",
        "Science and technology are changing rapidly."
    ]
    
    for phrase in test_phrases:
        tokens = tokenizer.encode(phrase, bos=True, eos=True)
        decoded = tokenizer.decode(tokens)
        print(f"  Original: \"{phrase}\"")
        print(f"  Tokens:   {tokens}")
        print(f"  Decoded:  \"{decoded.strip()}\"")
        print("-" * 30)
    print()

    # 4. Model Configuration Verification
    print("--- 4. MODEL PARAMETERS VERIFICATION ---")
    latest_cp_path = os.path.join(CHECKPOINT_DIR, "latest.pt")
    if not os.path.exists(latest_cp_path):
        print("Error: No trained model checkpoint found.")
        return
        
    checkpoint = torch.load(latest_cp_path, map_location="cpu")
    model_cfg = ModelConfig(**checkpoint["config"])
    model = CollisionTransformer(model_cfg)
    model.load_state_dict(checkpoint["model_state_dict"])
    param_count = model.get_parameter_count()
    
    print(f"  Model:        COLLISION-1M")
    print(f"  Parameters:   {param_count:,}")
    print(f"  Vocabulary:   {model_cfg.vocab_size}")
    print(f"  Context:      {model_cfg.max_seq_len}")
    print(f"  Layers:       {model_cfg.n_layer}")
    print(f"  Embedding:    {model_cfg.d_model}")
    print(f"  Heads:        {model_cfg.n_head}")
    print(f"  Device:       CPU\n")

    # 5. Checkpoint Verification
    print("--- 5. PROFILE CHECKPOINT VERIFICATION ---")
    profile_cp = os.path.join(CHECKPOINT_DIR, "collision-1m-profile.pt")
    if os.path.exists(profile_cp):
        try:
            profile_checkpoint = torch.load(profile_cp, map_location="cpu")
            print(f"  Profile checkpoint loads successfully.")
            print(f"  Step:         {profile_checkpoint.get('step', 0)}")
            print(f"  Epoch:        {profile_checkpoint.get('epoch', 0)}")
            print(f"  Train Loss:   {profile_checkpoint.get('train_loss', 0.0):.4f}")
        except Exception as e:
            print(f"  Error loading profile checkpoint: {e}")
    else:
        print("  Profile checkpoint collision-1m-profile.pt not found.")
    print()

    # 6. Benchmark speed check (using saved stats or quick load)
    print("--- 6. SPEED & MEMORY VERIFICATION ---")
    profile_stats_path = os.path.join(EXPERIMENT_DIR, "profile_stats.json")
    tokens_per_sec = 0.0
    steps_per_sec = 0.0
    cpu_mem = get_process_memory()
    
    if os.path.exists(profile_stats_path):
        with open(profile_stats_path, "r") as f:
            p_stats = json.load(f)
        tokens_per_sec = p_stats.get("tokens_per_sec", 0.0)
        steps_per_sec = p_stats.get("steps_per_sec", 0.0)
        print(f"  Steps/sec:           {steps_per_sec:.2f}")
        print(f"  Tokens/sec:          {tokens_per_sec:.2f}")
        print(f"  CPU Memory:          {cpu_mem:.1f} MB")
    else:
        print("  Profiling statistics file missing.")
    print()

    # 7. Generation Verification
    print("--- 7. GENERATION COMPARISON SAMPLES ---")
    gen_prompts = [
        "The future of technology",
        "Artificial intelligence",
        "Science is",
        "COLLISION is"
    ]
    for prompt in gen_prompts:
        out = generate(model, tokenizer, prompt=prompt, max_tokens=25, temperature=0.7, device="cpu")
        safe_out = out.encode('ascii', errors='replace').decode('ascii')
        print(f"  Prompt: \"{prompt}\" -> Generated: \"{safe_out.strip()}\"")
    print()

    # 8. Readiness Decision
    print("========================================")
    print("READINESS SUMMARY DECISION:")
    print(f"STATUS: {readiness}")
    print(f"Reason: {reason}")
    print("========================================\n")

    # Generate markdown report
    os.makedirs(EXPERIMENT_DIR, exist_ok=True)
    report_path = os.path.join(EXPERIMENT_DIR, "phase4_readiness.md")
    
    rec_command = "python -m training.train --config configs/collision_1m_cpu.yaml --max-steps 5000"
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# COLLISION-1M Pre-Training Readiness Check\n\n")
        f.write(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"## Evaluation Status\n")
        f.write(f"- **Readiness Decision**: **{readiness}**\n")
        f.write(f"- **Justification**: {reason}\n\n")
        
        f.write(f"## Dataset Inspection\n")
        f.write(f"- **Dataset Version**: {meta.get('dataset_version', 'N/A')}\n")
        f.write(f"- **Documents**: {meta.get('number_of_documents', 0)}\n")
        f.write(f"- **Total Tokens**: {actual_total_tokens:,}\n")
        f.write(f"- **Training Tokens**: {actual_train_tokens:,}\n")
        f.write(f"- **Validation Tokens**: {actual_val_tokens:,}\n")
        f.write(f"- **Vocabulary Size**: {meta.get('vocabulary_size', 0)}\n\n")
        
        f.write(f"## Model Specifications\n")
        f.write(f"- **Model Parameters**: {param_count:,}\n")
        f.write(f"- **Layers**: {model_cfg.n_layer}\n")
        f.write(f"- **Heads**: {model_cfg.n_head}\n")
        f.write(f"- **Embedding Size**: {model_cfg.d_model}\n")
        f.write(f"- **Context Length**: {model_cfg.max_seq_len}\n\n")
        
        f.write(f"## CPU Speed & Memory\n")
        f.write(f"- **Steps/Second**: {steps_per_sec:.2f}\n")
        f.write(f"- **Tokens/Second**: {tokens_per_sec:.2f}\n")
        f.write(f"- **Memory Usage**: {cpu_mem:.1f} MB\n\n")
        
        f.write(f"## Recommended Next Step\n")
        if readiness == "NOT READY":
            f.write("Dataset must be scaled to at least 1,000,000 tokens before training. Please provide additional training texts inside `data/raw/` and rebuild using `python -m data.build`.\n")
        else:
            f.write(f"To begin real training, run:\n```bash\n{rec_command}\n```\n")

    print(f"Readiness report written successfully to {report_path}")
    print("\nCOLLISION-1M Phase 4 readiness check completed.")
    
    if readiness == "NOT READY":
        print("\nNOT READY — DATASET TOO SMALL.")
    else:
        print("\nREADY FOR FIRST REAL TRAINING")

if __name__ == "__main__":
    check_readiness()
