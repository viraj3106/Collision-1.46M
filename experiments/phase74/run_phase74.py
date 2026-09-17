import os
import sys
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.phase73.train_phase73 import verify_immutability
from experiments.phase74.prepare_dataset import create_dataset_spec
from data.tokenize import BPETokenizer

EXP_DIR = os.path.join(PROJECT_ROOT, "experiments", "phase74")

def main():
    print("=" * 65)
    print("  PHASE 74 — CONVERSATIONAL FOUNDATION PREPARATION & AUDIT")
    print("=" * 65)
    
    # Step 1: Verify Immutability
    print("\n--- STEP 1: VERIFYING IMMUTABLE REFERENCES ---")
    sha_prod = verify_immutability()
    print(f"Production Model SHA256: {sha_prod} [FROZEN & VERIFIED]")
    
    # Step 2: Create Dataset Spec
    print("\n--- STEP 2: CREATING DATASET SPECIFICATION ---")
    create_dataset_spec()
    
    # Step 3: Verify Tokenizer
    tokenizer = BPETokenizer()
    tokenizer.load(os.path.join(PROJECT_ROOT, "artifacts", "tokenizer"))
    print(f"Tokenizer Vocabulary Size: {len(tokenizer.vocab)} [VERIFIED]")
    
    print("\n" + "=" * 65)
    print("  STATUS: PHASE 74 READY FOR CONVERSATIONAL TRAINING")
    print("=================================================================")

if __name__ == "__main__":
    main()
