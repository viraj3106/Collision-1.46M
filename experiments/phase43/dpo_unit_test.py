import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model.config import ModelConfig
from model.transformer import CollisionTransformer
from data.tokenize import BPETokenizer

EXP_DIR = os.path.join(PROJECT_ROOT, "experiments", "phase43")
TOKENIZER_DIR = os.path.join(PROJECT_ROOT, "artifacts", "tokenizer")
H3_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "phase37", "collision_10m_candidate_h3.pt")

os.makedirs(EXP_DIR, exist_ok=True)

def run_dpo_unit_test():
    print("\n--- STEP 3: CHOSEN / REJECTED GRADIENT DIRECTION UNIT TEST ---", flush=True)
    tokenizer = BPETokenizer()
    tokenizer.load(TOKENIZER_DIR)

    ck = torch.load(H3_PATH, map_location="cpu")
    cfg = ModelConfig(**ck["config"])
    policy_model = CollisionTransformer(cfg)
    policy_model.load_state_dict(ck["model_state_dict"])
    policy_model.train()

    test_pairs = [
        ("What is the capital of France?", "The capital of France is Paris.", "The capital of France is London."),
        ("What is 2 + 2?", "2 + 2 equals 4.", "2 + 2 equals 5."),
        ("Explain Python list append.", "append adds an item to the end of the list.", "append deletes all items in the list."),
        ("How to check disk usage in Linux?", "Use the df -h command to check disk space.", "Format the hard drive with mkfs."),
        ("What does HTML stand for?", "HyperText Markup Language.", "High Texture Machine Learning.")
    ]

    opt = torch.optim.AdamW(policy_model.parameters(), lr=1.0e-5)

    def compute_seq_logprob(model, prompt, response):
        comb = tokenizer.encode(prompt, bos=True) + tokenizer.encode(response, bos=False, eos=True)
        if len(comb) > 256: comb = comb[:256]
        x = torch.tensor([comb[:-1]], dtype=torch.long)
        y = torch.tensor([comb[1:]], dtype=torch.long)
        logits, _ = model(x)
        log_probs = F.log_softmax(logits, dim=-1)
        per_token_log_probs = torch.gather(log_probs, 2, y.unsqueeze(-1)).squeeze(-1)
        # Mask out prompt tokens
        prompt_len = len(tokenizer.encode(prompt, bos=True))
        resp_mask = torch.zeros_like(per_token_log_probs)
        resp_mask[0, max(0, prompt_len-1):] = 1.0
        seq_logprob = (per_token_log_probs * resp_mask).sum()
        return seq_logprob

    initial_logprobs = []
    for prompt, chosen, rejected in test_pairs:
        c_lp = compute_seq_logprob(policy_model, prompt, chosen).item()
        r_lp = compute_seq_logprob(policy_model, prompt, rejected).item()
        initial_logprobs.append({"prompt": prompt, "chosen_lp": c_lp, "rejected_lp": r_lp, "diff": c_lp - r_lp})

    # Test Canonical DPO loss vs Implemented Proxy loss
    # Canonical DPO step:
    ref_model = CollisionTransformer(cfg)
    ref_model.load_state_dict(ck["model_state_dict"])
    ref_model.eval()
    for p in ref_model.parameters(): p.requires_grad = False

    beta = 0.1
    losses = []

    opt.zero_grad()
    for prompt, chosen, rejected in test_pairs:
        pi_c = compute_seq_logprob(policy_model, prompt, chosen)
        pi_r = compute_seq_logprob(policy_model, prompt, rejected)
        with torch.no_grad():
            ref_c = compute_seq_logprob(ref_model, prompt, chosen)
            ref_r = compute_seq_logprob(ref_model, prompt, rejected)

        pi_logratios = pi_c - pi_r
        ref_logratios = ref_c - ref_r
        logits = pi_logratios - ref_logratios
        loss = -F.logsigmoid(beta * logits)
        loss.backward()
        losses.append(loss.item())

    opt.step()

    updated_logprobs = []
    for idx, (prompt, chosen, rejected) in enumerate(test_pairs):
        c_lp = compute_seq_logprob(policy_model, prompt, chosen).item()
        r_lp = compute_seq_logprob(policy_model, prompt, rejected).item()
        init = initial_logprobs[idx]
        c_delta = c_lp - init["chosen_lp"]
        r_delta = r_lp - init["rejected_lp"]
        updated_logprobs.append({
            "prompt": prompt,
            "init_chosen_lp": round(init["chosen_lp"], 4),
            "updated_chosen_lp": round(c_lp, 4),
            "chosen_lp_delta": round(c_delta, 4),
            "init_rejected_lp": round(init["rejected_lp"], 4),
            "updated_rejected_lp": round(r_lp, 4),
            "rejected_lp_delta": round(r_delta, 4),
            "gradient_direction_correct": (c_delta > r_delta)
        })

    all_gradient_directions_correct = all(u["gradient_direction_correct"] for u in updated_logprobs)

    result = {
        "status": "PASS" if all_gradient_directions_correct else "FAIL",
        "canonical_dpo_gradient_direction_correct": all_gradient_directions_correct,
        "sample_pair_results": updated_logprobs
    }

    out_file = os.path.join(EXP_DIR, "dpo_unit_test_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"DPO Unit Test Result saved to {out_file} (Status: {result['status']})")
    return result

if __name__ == "__main__":
    run_dpo_unit_test()
