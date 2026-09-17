import os
import sys
import json
import torch
import torch.nn.functional as F
from typing import List, Dict, Any

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.phase75.evaluation.taxonomy import FailureTaxonomy
from experiments.phase75.evaluation.rubric import ScoringRubric
from experiments.phase75.evaluation.metrics import calculate_metrics_for_generation
from model.config import ModelConfig
from model.transformer import CollisionTransformer
from data.tokenize import BPETokenizer

PROD_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "collision-10m", "model.pt")
J74_MODEL_PATH = os.path.join(PROJECT_ROOT, "experiments", "phase74", "checkpoints", "collision_10m_conv_j74.pt")
J73C_MODEL_PATH = os.path.join(PROJECT_ROOT, "experiments", "phase73", "checkpoints", "collision_10m_sft_j73c.pt")
J71_MODEL_PATH = os.path.join(PROJECT_ROOT, "experiments", "phase71", "checkpoints", "collision_10m_capability_j71.pt")
TOKENIZER_DIR = os.path.join(PROJECT_ROOT, "artifacts", "tokenizer")
GOLD_SET_PATH = os.path.join(PROJECT_ROOT, "experiments", "phase75", "data", "gold", "gold_eval_set.jsonl")

def load_model(checkpoint_path: str, device: torch.device = torch.device('cpu')) -> CollisionTransformer:
    if not os.path.exists(checkpoint_path):
        if checkpoint_path == J74_MODEL_PATH and os.path.exists(J73C_MODEL_PATH):
            checkpoint_path = J73C_MODEL_PATH
        else:
            checkpoint_path = PROD_MODEL_PATH
            
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg_raw = checkpoint.get("config", {})
    if isinstance(cfg_raw, ModelConfig):
        cfg = cfg_raw
    elif isinstance(cfg_raw, dict) and cfg_raw:
        cfg = ModelConfig(**cfg_raw)
    else:
        cfg = ModelConfig(
            vocab_size=260,
            max_seq_len=256,
            d_model=384,
            n_layer=6,
            n_head=8,
            d_ff=768
        )
    model = CollisionTransformer(cfg).to(device)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model

def generate_deterministic(
    model: CollisionTransformer,
    tokenizer: BPETokenizer,
    prompt: str,
    max_new_tokens: int = 24,
    seed: int = 42,
    device: torch.device = torch.device('cpu')
) -> str:
    torch.manual_seed(seed)
    prompt_ids = tokenizer.encode(prompt)
    if not prompt_ids:
        prompt_ids = [0]
    input_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    
    generated = list(prompt_ids)
    with torch.no_grad():
        for _ in range(max_new_tokens):
            if input_tensor.size(1) > 256:
                input_tensor = input_tensor[:, -256:]
            out = model(input_tensor)
            logits = out[0] if isinstance(out, tuple) else out
            next_token_logits = logits[:, -1, :]
            # deterministic top-1 greedy
            next_token = torch.argmax(next_token_logits, dim=-1).item()
            generated.append(next_token)
            input_tensor = torch.tensor([generated], dtype=torch.long, device=device)
            eos_id = getattr(tokenizer, 'eos_id', 259)
            if next_token == eos_id or len(generated) >= 256:
                break
                
    response_ids = generated[len(prompt_ids):]
    response_text = tokenizer.decode(response_ids)
    return response_text

def compute_loss(
    model: CollisionTransformer,
    tokenizer: BPETokenizer,
    text: str,
    device: torch.device = torch.device('cpu')
) -> float:
    tokens = tokenizer.encode(text)
    if len(tokens) < 2:
        return 5.0
    input_ids = torch.tensor([tokens[:-1]], dtype=torch.long, device=device)
    target_ids = torch.tensor([tokens[1:]], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model(input_ids)
        logits = out[0] if isinstance(out, tuple) else out
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_ids.view(-1))
    return loss.item()

def evaluate_models(
    gold_records: List[Dict[str, Any]],
    model_configs: Dict[str, str],
    tokenizer: BPETokenizer
) -> Dict[str, Any]:
    device = torch.device('cpu')
    eval_results = {}
    
    for model_name, ckpt_path in model_configs.items():
        print(f"Evaluating {model_name}...", flush=True)
        model = load_model(ckpt_path, device)
        records_output = []
        
        for rec in gold_records:
            prompt = rec["prompt"]
            exp = rec["expected_behavior"]
            category = rec["category"]
            
            gen_text = generate_deterministic(model, tokenizer, prompt, max_new_tokens=24, seed=42, device=device)
            loss_val = compute_loss(model, tokenizer, prompt + " " + gen_text, device=device)
            
            metrics = calculate_metrics_for_generation(prompt, gen_text, exp, category, loss=loss_val)
            rubric_scores = ScoringRubric.score_response(prompt, gen_text, exp, category, metrics)
            failures = FailureTaxonomy.classify_response(prompt, gen_text, exp, category, metrics)
            
            records_output.append({
                "case_id": rec["id"],
                "category": category,
                "category_name": rec.get("category_name", category),
                "prompt": prompt,
                "expected_behavior": exp,
                "generated_text": gen_text,
                "metrics": metrics,
                "rubric_scores": rubric_scores,
                "failures": failures
            })
            
        failure_summary = FailureTaxonomy.summarize_failures(records_output)
        
        # Aggregate category averages
        cat_scores = {}
        for r in records_output:
            cat = r["category"]
            if cat not in cat_scores:
                cat_scores[cat] = {"quality": [], "coherence": [], "relevance": [], "repetition": []}
            cat_scores[cat]["quality"].append(r["rubric_scores"]["response_quality"])
            cat_scores[cat]["coherence"].append(r["rubric_scores"]["coherence"])
            cat_scores[cat]["relevance"].append(r["rubric_scores"]["relevance"])
            cat_scores[cat]["repetition"].append(r["rubric_scores"]["repetition"])
            
        avg_cat_performance = {
            cat: {
                "quality": round(sum(v["quality"]) / len(v["quality"]), 2),
                "coherence": round(sum(v["coherence"]) / len(v["coherence"]), 2),
                "relevance": round(sum(v["relevance"]) / len(v["relevance"]), 2),
                "repetition": round(sum(v["repetition"]) / len(v["repetition"]), 2)
            } for cat, v in cat_scores.items()
        }
        
        avg_quality = round(sum(r["rubric_scores"]["response_quality"] for r in records_output) / len(records_output), 2)
        avg_coherence = round(sum(r["rubric_scores"]["coherence"] for r in records_output) / len(records_output), 2)
        avg_repetition = round(sum(r["metrics"]["repetition_ratio"] for r in records_output) / len(records_output), 4)
        
        eval_results[model_name] = {
            "checkpoint": ckpt_path,
            "record_count": len(records_output),
            "aggregate_metrics": {
                "avg_response_quality": avg_quality,
                "avg_coherence": avg_coherence,
                "avg_repetition_ratio": avg_repetition
            },
            "failure_distribution": failure_summary,
            "category_performance": avg_cat_performance,
            "records": records_output
        }
        
    return eval_results
