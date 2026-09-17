import os
import sys
import json
import torch
from typing import List, Dict, Any, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.phase75.evaluation.evaluator import load_model, PROD_MODEL_PATH, J73C_MODEL_PATH
from experiments.phase75.evaluation.taxonomy import FailureTaxonomy
from experiments.phase75.evaluation.rubric import ScoringRubric
from experiments.phase75.evaluation.metrics import calculate_metrics_for_generation
from data.tokenize import BPETokenizer

STRESS_CONTEXT_LENGTHS = [32, 64, 96, 128, 160, 192, 256, 384]

def run_context_stress_test(
    model_path: str,
    gold_records: List[Dict[str, Any]],
    tokenizer: BPETokenizer,
    lengths: List[int] = STRESS_CONTEXT_LENGTHS
) -> Dict[int, Dict[str, Any]]:
    device = torch.device("cpu")
    model = load_model(model_path, device)
    
    results_by_length = {}
    
    for length in lengths:
        print(f"Stress-testing context length: {length} tokens...", flush=True)
        length_records = []
        for rec in gold_records[:5]:  # Controlled deterministic diagnostic subset
            prompt = rec["prompt"]
            exp = rec["expected_behavior"]
            cat = rec["category"]
            
            # Prepare context truncated or padded to target token length
            p_ids = tokenizer.encode(prompt)
            eff_len = min(length, getattr(model.config, 'max_seq_len', 256))
            if len(p_ids) > eff_len:
                p_ids = p_ids[-eff_len:]
            elif len(p_ids) < eff_len:
                p_ids = [0] * (eff_len - len(p_ids)) + p_ids
                
            input_tensor = torch.tensor([p_ids], dtype=torch.long, device=device)
            
            generated = list(p_ids)
            with torch.no_grad():
                for _ in range(16):
                    if input_tensor.size(1) > eff_len:
                        input_tensor = input_tensor[:, -eff_len:]
                    out = model(input_tensor)
                    logits = out[0] if isinstance(out, tuple) else out
                    next_token = torch.argmax(logits[:, -1, :], dim=-1).item()
                    generated.append(next_token)
                    input_tensor = torch.tensor([generated], dtype=torch.long, device=device)
                    if next_token == getattr(tokenizer, 'eos_id', 259) or len(generated) >= length + 24:
                        break
                        
            resp_ids = generated[len(p_ids):]
            gen_text = tokenizer.decode(resp_ids)
            
            metrics = calculate_metrics_for_generation(prompt, gen_text, exp, cat, loss=1.5)
            rubric_scores = ScoringRubric.score_response(prompt, gen_text, exp, cat, metrics)
            failures = FailureTaxonomy.classify_response(prompt, gen_text, exp, cat, metrics)
            
            length_records.append({
                "prompt": prompt,
                "gen_text": gen_text,
                "rubric": rubric_scores,
                "failures": failures
            })
            
        avg_quality = round(sum(r["rubric"]["response_quality"] for r in length_records) / len(length_records), 2)
        avg_coherence = round(sum(r["rubric"]["coherence"] for r in length_records) / len(length_records), 2)
        frag_count = sum(1 for r in length_records if "FRAGMENTATION" in r["failures"])
        irrel_count = sum(1 for r in length_records if "IRRELEVANCE" in r["failures"])
        
        results_by_length[length] = {
            "context_length": length,
            "sample_count": len(length_records),
            "avg_response_quality": avg_quality,
            "avg_coherence": avg_coherence,
            "fragmentation_count": frag_count,
            "irrelevance_count": irrel_count
        }
        
    return results_by_length
