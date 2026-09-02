import os
import sys
import json
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
    print("Starting COLLISION Hugging Face Package Verification...")
    hf_dir = os.path.join(PROJECT_ROOT, "release", "huggingface")

    # 1. Verify required files exist
    required_files = [
        os.path.join(hf_dir, "README.md"),
        os.path.join(hf_dir, "model_metadata.json"),
        os.path.join(hf_dir, "checksums.sha256"),
        os.path.join(hf_dir, "DESCRIPTION.md"),
        os.path.join(hf_dir, "model.pt"),
        os.path.join(hf_dir, "config.json"),
        os.path.join(hf_dir, "tokenizer.json"),
        os.path.join(hf_dir, "generation_config.json"),
        os.path.join(hf_dir, "tokenizer", "config.json"),
        os.path.join(hf_dir, "tokenizer", "vocab.json"),
        os.path.join(hf_dir, "tokenizer", "merges.json"),
        os.path.join(hf_dir, "tokenizer", "stats.json")
    ]
    
    for f in required_files:
        if not os.path.exists(f):
            print(f"FAILED: Required Hugging Face package file is missing: {f}")
            sys.exit(1)
    print("OK: All required package files exist.")

    # 2. Verify no unexpected secret files exist
    unexpected_exclusions = [".env", ".git", ".idea", ".vscode"]
    for root, dirs, files in os.walk(hf_dir):
        for f in files:
            if any(exc in f for exc in unexpected_exclusions):
                print(f"FAILED: Unexpected or secret file found in package: {os.path.join(root, f)}")
                sys.exit(1)
    print("OK: No secret or unexpected files found in the package folder.")

    # 3. Verify model weights load and parameters count matches
    model_pt = os.path.join(hf_dir, "model.pt")
    
    # Check SHA256 matches
    actual_sha256 = calculate_sha256(model_pt)
    if actual_sha256 != EXPECTED_SHA256:
        print(f"FAILED: Checkpoint SHA256 mismatch. Expected: {EXPECTED_SHA256}, Got: {actual_sha256}")
        sys.exit(1)
    print("OK: Checkpoint SHA256 checksum matches.")

    try:
        device = torch.device("cpu")
        checkpoint = torch.load(model_pt, map_location=device)
        model_cfg = ModelConfig(**checkpoint["config"])
        model = CollisionTransformer(model_cfg)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"FAILED: Failed to load model weights: {e}")
        sys.exit(1)

    param_count = sum(p.numel() for p in model.parameters())
    if param_count != EXPECTED_PARAMS:
        print(f"FAILED: Parameter count mismatch. Expected: {EXPECTED_PARAMS}, Got: {param_count}")
        sys.exit(1)
    print(f"OK: Model loaded successfully. Parameter count is exactly {param_count:,}.")

    # 4. Verify tokenizer loads
    tokenizer_dir = os.path.join(hf_dir, "tokenizer")
    tokenizer = BPETokenizer()
    try:
        tokenizer.load(tokenizer_dir)
    except Exception as e:
        print(f"FAILED: Tokenizer failed to load: {e}")
        sys.exit(1)
    print("OK: Tokenizer loaded successfully.")

    # 5. Verify local inference works
    try:
        generated_out = generate(
            model=model,
            tokenizer=tokenizer,
            prompt="Artificial intelligence is",
            max_tokens=15,
            temperature=0.7,
            top_k=50,
            top_p=0.9,
            device="cpu"
        )
    except Exception as e:
        print(f"FAILED: Text generation failed: {e}")
        sys.exit(1)

    if not generated_out or len(generated_out.strip()) == 0:
        print("FAILED: Text generation returned empty result.")
        sys.exit(1)
    print(f"OK: Local text generation works. Result: '{generated_out}'")

    # 6. Verify metadata is valid
    metadata_file = os.path.join(hf_dir, "model_metadata.json")
    try:
        with open(metadata_file, "r", encoding="utf-8") as fp:
            meta = json.load(fp)
        # Assert crucial keys exist
        assert meta["name"] == "COLLISION-10M"
        assert meta["parameters"] == EXPECTED_PARAMS
        assert meta["checkpoint_sha256"] == EXPECTED_SHA256
    except Exception as e:
        print(f"FAILED: Model metadata validation failed: {e}")
        sys.exit(1)
    print("OK: Model metadata is valid.")

    # 7. Verify license documentation is referenced
    license_decision_path = os.path.join(PROJECT_ROOT, "release", "LICENSE_DECISION.md")
    license_path = os.path.join(PROJECT_ROOT, "LICENSE")
    if not os.path.exists(license_decision_path) or not os.path.exists(license_path):
        print("FAILED: License files are missing in project root / release folder.")
        sys.exit(1)
    print("OK: License documentation exists.")

    print("\nCOLLISION HUGGING FACE PACKAGE VERIFIED\n")

if __name__ == "__main__":
    main()
