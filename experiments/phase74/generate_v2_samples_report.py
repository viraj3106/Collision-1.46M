import os
import sys
import json
import re

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

SEED_PATH = os.path.join(PROJECT_ROOT, "data", "instructions", "collision_conversation_v2_seed", "v2_seed.jsonl")

def generate_v2_seed_samples_report():
    if not os.path.exists(SEED_PATH):
        raise FileNotFoundError(f"Missing v2_seed.jsonl at {SEED_PATH}")
        
    records = []
    with open(SEED_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
                
    out_md_path = os.path.join(PROJECT_ROOT, "experiments", "phase74", "reports", "v2_seed_samples.md")
    os.makedirs(os.path.dirname(out_md_path), exist_ok=True)
    
    with open(out_md_path, "w", encoding="utf-8") as f:
        f.write("# PHASE 74 — V2 SEED COMPLETE 100-CONVERSATION SAMPLES REPORT\n\n")
        f.write("This report presents ALL 100 hand-authored organic human-AI conversations in the Phase 74 v2 Seed set.\n\n---\n\n")
        
        for idx, r in enumerate(records, 1):
            f.write(f"### Conversation #{idx:03d} (ID: `{r['id']}`, Category: `{r['category']}`, Domain: `{r['domain']}`)\n")
            for t in r['turns']:
                f.write(f"* **{t['role'].capitalize()}**: {t['content']}\n")
            f.write("\n---\n\n")
            
    print(f"Saved complete 100-sample report to {out_md_path}")

if __name__ == "__main__":
    generate_v2_seed_samples_report()
