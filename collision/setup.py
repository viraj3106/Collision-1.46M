import os
import sys

def main():
    print("Setting up COLLISION-1M project structure...")
    
    # Required directories
    dirs = [
        "data/raw",
        "data/processed",
        "data/samples",
        "checkpoints",
        "experiments",
        "artifacts/tokenizer",
        "configs"
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"Directory verified/created: {d}")

    # Generate sample raw dataset if not present
    sample_path = os.path.join("data", "raw", "sample.txt")
    if not os.path.exists(sample_path):
        from data.prepare import generate_sample_text
        with open(sample_path, "w", encoding="utf-8") as f:
            f.write(generate_sample_text())
        print(f"Created demo raw dataset at {sample_path}")

    # Check dependencies
    try:
        import torch
        import numpy
        print(f"PyTorch version: {torch.__version__}")
        print(f"NumPy version: {numpy.__version__}")
    except ImportError as e:
        print(f"Warning: Missing dependency: {e}. Please run 'pip install -r requirements.txt'")

    print("\nCOLLISION-1M initialized successfully.")

if __name__ == "__main__":
    main()
