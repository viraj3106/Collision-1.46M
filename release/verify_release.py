import os
import sys
import hashlib
import torch

# Resolve project root and insert into sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model.config import ModelConfig
from model.transformer import CollisionTransformer
from data.tokenize import BPETokenizer
from inference.generate import generate

EXPECTED_SHA256 = "d256d46d962d6416fe22d2cfe80b13df0574279fb980d7d8576c2bdcf3775b97"
EXPECTED_PARAMS = 10282304

def calculate_sha256(filepath):
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest().lower()

def main():
    print("Starting COLLISION Release Verification...")
    
    # 1. Verify existence of model files
    model_dir = os.path.join(PROJECT_ROOT, "models", "collision-10m")
    model_pt = os.path.join(model_dir, "model.pt")
    config_json = os.path.join(model_dir, "config.json")
    tokenizer_json = os.path.join(model_dir, "tokenizer.json")
    generation_config_json = os.path.join(model_dir, "generation_config.json")
    
    required_files = [model_pt, config_json, tokenizer_json, generation_config_json]
    for path in required_files:
        if not os.path.exists(path):
            print(f"FAILED: File {path} does not exist.")
            sys.exit(1)
    print("OK: Model configuration and checkpoint files exist.")

    # 2. Verify tokenizer files exist
    tokenizer_dir = os.path.join(model_dir, "tokenizer")
    tk_config = os.path.join(tokenizer_dir, "config.json")
    tk_vocab = os.path.join(tokenizer_dir, "vocab.json")
    tk_merges = os.path.join(tokenizer_dir, "merges.json")
    tk_stats = os.path.join(tokenizer_dir, "stats.json")
    
    tokenizer_files = [tk_config, tk_vocab, tk_merges, tk_stats]
    for path in tokenizer_files:
        if not os.path.exists(path):
            print(f"FAILED: Tokenizer file {path} does not exist.")
            sys.exit(1)
    print("OK: Tokenizer files exist.")

    # 3. Verify SHA256 checksum
    actual_sha256 = calculate_sha256(model_pt)
    if actual_sha256 != EXPECTED_SHA256:
        print(f"FAILED: SHA256 mismatch. Expected: {EXPECTED_SHA256}, Got: {actual_sha256}")
        sys.exit(1)
    print("OK: Checkpoint SHA256 checksum matches expected value.")

    # 4. Load tokenizer and verify BPE functionality
    tokenizer = BPETokenizer()
    try:
        tokenizer.load(tokenizer_dir)
    except Exception as e:
        print(f"FAILED: Tokenizer failed to load: {e}")
        sys.exit(1)
        
    test_prompt = "Artificial intelligence is"
    encoded_ids = tokenizer.encode(test_prompt, bos=True)
    if not encoded_ids or len(encoded_ids) == 0:
        print("FAILED: Tokenizer encode returned empty output.")
        sys.exit(1)
        
    decoded_text = tokenizer.decode(encoded_ids)
    if test_prompt not in decoded_text:
        print(f"FAILED: Tokenizer roundtrip encode/decode failed. Expected '{test_prompt}' to be in '{decoded_text}'")
        sys.exit(1)
    print("OK: Tokenizer loaded and verified successfully.")

    # 5. Load model and verify parameter count
    try:
        device = torch.device("cpu")
        checkpoint = torch.load(model_pt, map_location=device)
        model_cfg = ModelConfig(**checkpoint["config"])
        model = CollisionTransformer(model_cfg)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"FAILED: Failed to load model weights/checkpoint: {e}")
        sys.exit(1)
        
    param_count = sum(p.numel() for p in model.parameters())
    if param_count != EXPECTED_PARAMS:
        print(f"FAILED: Parameter count mismatch. Expected: {EXPECTED_PARAMS}, Got: {param_count}")
        sys.exit(1)
    print(f"OK: Model loaded. Parameter count is exactly {param_count:,}.")

    # 6. Run API-independent local inference
    try:
        generated_out = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=test_prompt,
            max_tokens=20,
            temperature=0.7,
            top_k=50,
            top_p=0.9,
            device="cpu"
        )
    except Exception as e:
        print(f"FAILED: Local inference execution failed: {e}")
        sys.exit(1)
        
    if not generated_out or len(generated_out.strip()) == 0:
        print("FAILED: Generated output is empty.")
        sys.exit(1)
        
    print(f"OK: Local inference verified. Sample completion: '{generated_out}'")
    
    print("\nCOLLISION RELEASE VERIFICATION PASSED\n")

if __name__ == "__main__":
    main()
