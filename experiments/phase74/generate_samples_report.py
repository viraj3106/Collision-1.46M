import os
import sys
import json
import re

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

GOLD_SET_PATH = os.path.join(PROJECT_ROOT, "data", "instructions", "collision_conversation_v1_gold", "gold_set.jsonl")

def generate_samples_report():
    if not os.path.exists(GOLD_SET_PATH):
        raise FileNotFoundError("gold_set.jsonl missing!")
        
    records = []
    with open(GOLD_SET_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
                
    one_turn = [r for r in records if len(r["turns"]) == 2][:10]
    two_turn = [r for r in records if len(r["turns"]) == 4][:10]
    three_turn = [r for r in records if len(r["turns"]) == 6][:10]
    
    out_md_path = os.path.join(PROJECT_ROOT, "experiments", "phase74", "reports", "gold_set_samples.md")
    os.makedirs(os.path.dirname(out_md_path), exist_ok=True)
    
    with open(out_md_path, "w", encoding="utf-8") as f:
        f.write("# PHASE 74 — GOLD SET REPRESENTATIVE SAMPLES REPORT\n\n")
        f.write("This report presents representative organic human-style samples from the 300-conversation Gold Set.\n\n---\n\n")
        
        f.write("## 1. ONE-TURN CONVERSATIONS (10 SAMPLES)\n\n")
        for idx, r in enumerate(one_turn, 1):
            f.write(f"### Sample {idx:02d} (Category: `{r['category']}`, Domain: `{r['domain']}`)\n")
            f.write(f"* **User**: {r['turns'][0]['content']}\n")
            f.write(f"* **Assistant**: {r['turns'][1]['content']}\n\n")
            
        f.write("---\n\n## 2. TWO-TURN CONVERSATIONS (10 SAMPLES)\n\n")
        for idx, r in enumerate(two_turn, 1):
            f.write(f"### Sample {idx:02d} (Category: `{r['category']}`, Domain: `{r['domain']}`)\n")
            for t in r['turns']:
                f.write(f"* **{t['role'].capitalize()}**: {t['content']}\n")
            f.write("\n")
            
        f.write("---\n\n## 3. THREE-TURN CONVERSATIONS (10 SAMPLES)\n\n")
        for idx, r in enumerate(three_turn, 1):
            f.write(f"### Sample {idx:02d} (Category: `{r['category']}`, Domain: `{r['domain']}`)\n")
            for t in r['turns']:
                f.write(f"* **{t['role'].capitalize()}**: {t['content']}\n")
            f.write("\n")
            
    print(f"Generated Gold Set Samples Report -> {out_md_path}")

if __name__ == "__main__":
    generate_samples_report()
