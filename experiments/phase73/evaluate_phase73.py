import os
import sys
import json
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from typing import List, Dict, Any, Tuple
from model.config import ModelConfig
from model.transformer import CollisionTransformer
from data.tokenize import BPETokenizer
from experiments.phase72.phase72_diagnostic import (
    get_fixed_evaluation_prompts,
    classify_response,
    generate_text
)
from data.audit_generation_quality import calculate_repetition_metrics

PROD_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "collision-10m", "model.pt")
J52_MODEL_PATH = os.path.join(PROJECT_ROOT, "experiments", "phase52", "checkpoints", "collision_10m_sft_j52.pt")
J71_MODEL_PATH = os.path.join(PROJECT_ROOT, "experiments", "phase71", "checkpoints", "collision_10m_capability_j71.pt")
TOKENIZER_DIR = os.path.join(PROJECT_ROOT, "artifacts", "tokenizer")

def load_all_evaluation_models(j73a_path: str, j73b_path: str, j73c_path: str):
    tokenizer = BPETokenizer()
    tokenizer.load(TOKENIZER_DIR)
    
    def load_ckpt(path: str) -> CollisionTransformer:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        cfg_dict = ckpt.get("config", {})
        config = cfg_dict if isinstance(cfg_dict, ModelConfig) else ModelConfig(**cfg_dict)
        model = CollisionTransformer(config)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        return model

    models = {
        "baseline": load_ckpt(PROD_MODEL_PATH),
        "j52": load_ckpt(J52_MODEL_PATH),
        "j71": load_ckpt(J71_MODEL_PATH),
        "j73a": load_ckpt(j73a_path),
        "j73b": load_ckpt(j73b_path),
        "j73c": load_ckpt(j73c_path)
    }
    return models, tokenizer

def evaluate_all_candidates(models: dict, tokenizer: BPETokenizer) -> Tuple[dict, list]:
    prompts_dict = get_fixed_evaluation_prompts()
    all_generations = []
    
    results = {
        m_name: {
            "overall_capability": 0.0,
            "coherence": 0.0,
            "instruction_following": 0.0,
            "unique_token_ratio": 0.0,
            "unigram_repetition": 0.0,
            "bigram_repetition": 0.0,
            "avg_response_length": 0.0,
            "empty_rate": 0.0,
            "fragmented_rate": 0.0,
            "repetitive_rate": 0.0,
            "category_scores": {}
        } for m_name in models
    }

    print("\nEvaluating all candidates across fixed evaluation prompts...", flush=True)
    
    for cat_name, cat_prompts in prompts_dict.items():
        for p_idx, prompt in enumerate(cat_prompts):
            for m_name, model in models.items():
                torch.manual_seed(42 + p_idx)
                resp = generate_text(model, tokenizer, prompt, max_tokens=32, temperature=0.0)
                labels = classify_response(prompt, resp, tokenizer)
                u_ratio, unigram_rep, bigram_rep, _, _ = calculate_repetition_metrics(resp, tokenizer)
                
                gen_entry = {
                    "category": cat_name,
                    "prompt": prompt,
                    "model": m_name,
                    "temperature": 0.0,
                    "response": resp,
                    "labels": labels,
                    "metrics": {
                        "length": len(resp),
                        "unique_token_ratio": round(u_ratio, 4),
                        "unigram_repetition": round(unigram_rep, 4),
                        "bigram_repetition": round(bigram_rep, 4)
                    }
                }
                all_generations.append(gen_entry)

    total_prompts = sum(len(v) for v in prompts_dict.values())
    
    for m_name in models:
        m_gens = [g for g in all_generations if g["model"] == m_name]
        
        coherent_cnt = sum(1 for g in m_gens if "COHERENT" in g["labels"])
        inst_cnt = sum(1 for g in m_gens if "INSTRUCTION_FOLLOWING" in g["labels"])
        empty_cnt = sum(1 for g in m_gens if "EMPTY" in g["labels"])
        frag_cnt = sum(1 for g in m_gens if "FRAGMENTED" in g["labels"])
        rep_cnt = sum(1 for g in m_gens if "REPETITIVE" in g["labels"])
        
        avg_u_ratio = sum(g["metrics"]["unique_token_ratio"] for g in m_gens) / total_prompts
        avg_unigram_rep = sum(g["metrics"]["unigram_repetition"] for g in m_gens) / total_prompts
        avg_bigram_rep = sum(g["metrics"]["bigram_repetition"] for g in m_gens) / total_prompts
        avg_len = sum(g["metrics"]["length"] for g in m_gens) / total_prompts
        
        coherence_score = round(coherent_cnt / total_prompts, 4)
        inst_score = round(inst_cnt / total_prompts, 4)
        capability_score = round(0.5 * coherence_score + 0.3 * inst_score + 0.2 * avg_u_ratio, 4)
        
        cat_scores = {}
        for cat_name in prompts_dict:
            cat_gens = [g for g in m_gens if g["category"] == cat_name]
            c_coh = sum(1 for g in cat_gens if "COHERENT" in g["labels"]) / len(cat_gens)
            c_inst = sum(1 for g in cat_gens if "INSTRUCTION_FOLLOWING" in g["labels"]) / len(cat_gens)
            cat_scores[cat_name] = round(0.5 * c_coh + 0.5 * c_inst, 4)

        results[m_name] = {
            "overall_capability": capability_score,
            "coherence": coherence_score,
            "instruction_following": inst_score,
            "unique_token_ratio": round(avg_u_ratio, 4),
            "unigram_repetition": round(avg_unigram_rep, 4),
            "bigram_repetition": round(avg_bigram_rep, 4),
            "avg_response_length": round(avg_len, 2),
            "empty_rate": round(empty_cnt / total_prompts, 4),
            "fragmented_rate": round(frag_cnt / total_prompts, 4),
            "repetitive_rate": round(rep_cnt / total_prompts, 4),
            "category_scores": cat_scores
        }
        
    return results, all_generations

