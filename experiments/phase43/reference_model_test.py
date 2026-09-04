import os
import sys
import json
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model.config import ModelConfig
from model.transformer import CollisionTransformer

EXP_DIR = os.path.join(PROJECT_ROOT, "experiments", "phase43")
H3_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "phase37", "collision_10m_candidate_h3.pt")

os.makedirs(EXP_DIR, exist_ok=True)

def run_reference_model_test():
    print("\n--- STEP 4: REFERENCE MODEL FREEZING & GRADIENT TEST ---", flush=True)

    ck = torch.load(H3_PATH, map_location="cpu")
    cfg = ModelConfig(**ck["config"])

    policy_model = CollisionTransformer(cfg)
    policy_model.load_state_dict(ck["model_state_dict"])
    policy_model.train()

    ref_model = CollisionTransformer(cfg)
    ref_model.load_state_dict(ck["model_state_dict"])
    ref_model.eval()

    # Freeze reference model
    for p in ref_model.parameters():
        p.requires_grad = False

    policy_params_count = sum(1 for p in policy_model.parameters() if p.requires_grad)
    ref_params_count = sum(1 for p in ref_model.parameters() if p.requires_grad)

    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=1.0e-5)

    ref_weights_before = {n: p.clone() for n, p in ref_model.named_parameters()}

    x = torch.randint(0, 8000, (2, 64))
    y = torch.randint(0, 8000, (2, 64))

    optimizer.zero_grad()
    pol_logits, pol_loss = policy_model(x, y)
    with torch.no_grad():
        ref_logits, ref_loss = ref_model(x, y)

    loss = pol_loss + (pol_loss - ref_loss)
    loss.backward()

    ref_grads_received = any(p.grad is not None and torch.sum(torch.abs(p.grad)) > 0 for p in ref_model.parameters())

    optimizer.step()

    ref_weights_after = {n: p for n, p in ref_model.named_parameters()}

    max_weight_delta = 0.0
    for n in ref_weights_before:
        diff = torch.max(torch.abs(ref_weights_after[n] - ref_weights_before[n])).item()
        if diff > max_weight_delta:
            max_weight_delta = diff

    ref_strictly_frozen = (ref_params_count == 0) and (not ref_grads_received) and (max_weight_delta == 0.0)

    result = {
        "status": "PASS" if ref_strictly_frozen else "FAIL",
        "policy_trainable_param_groups": policy_params_count,
        "reference_trainable_param_groups": ref_params_count,
        "reference_received_gradients": ref_grads_received,
        "max_reference_weight_delta": max_weight_delta,
        "reference_strictly_frozen": ref_strictly_frozen
    }

    out_file = os.path.join(EXP_DIR, "reference_model_test_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"Reference Model Test Result saved to {out_file} (Status: {result['status']})")
    return result

if __name__ == "__main__":
    run_reference_model_test()
