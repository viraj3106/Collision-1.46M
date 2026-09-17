import os
import sys
import json
import math
import hashlib
import random
import re
from typing import List, Dict, Any, Tuple
import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model.config import ModelConfig
from model.transformer import CollisionTransformer
from data.tokenize import BPETokenizer
from inference.generate import top_k_top_p_filtering
from data.audit_generation_quality import calculate_repetition_metrics

EXP_DIR = os.path.join(PROJECT_ROOT, "experiments", "phase72")
REPORTS_DIR = os.path.join(EXP_DIR, "reports")
os.makedirs(EXP_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

PROD_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "collision-10m", "model.pt")
J52_MODEL_PATH = os.path.join(PROJECT_ROOT, "experiments", "phase52", "checkpoints", "collision_10m_sft_j52.pt")
J71_MODEL_PATH = os.path.join(PROJECT_ROOT, "experiments", "phase71", "checkpoints", "collision_10m_capability_j71.pt")
TOKENIZER_DIR = os.path.join(PROJECT_ROOT, "artifacts", "tokenizer")
SFT_V3_DIR = os.path.join(PROJECT_ROOT, "data", "instructions", "collision_sft_v3")

EXPECTED_SHA256 = "d256d46d962d6416fe22d2cfe80b13df0574279fb980d7d8576c2bdcf3775b97"

def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)

def get_sha256(path: str) -> str:
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha.update(chunk)
    return sha.hexdigest()

def verify_immutability() -> Dict[str, str]:
    if not os.path.exists(PROD_MODEL_PATH):
        raise FileNotFoundError(f"Production model missing at {PROD_MODEL_PATH}")
    prod_sha = get_sha256(PROD_MODEL_PATH)
    if prod_sha.lower() != EXPECTED_SHA256.lower():
        raise ValueError(f"CRITICAL: Production SHA256 mismatch! Expected {EXPECTED_SHA256}, got {prod_sha}")
    
    if not os.path.exists(J52_MODEL_PATH):
        raise FileNotFoundError(f"J52 model missing at {J52_MODEL_PATH}")
    j52_sha = get_sha256(J52_MODEL_PATH)
    
    if not os.path.exists(J71_MODEL_PATH):
        raise FileNotFoundError(f"J71 model missing at {J71_MODEL_PATH}")
    j71_sha = get_sha256(J71_MODEL_PATH)
    
    return {
        "production": prod_sha,
        "j52": j52_sha,
        "j71": j71_sha
    }

def load_models() -> Tuple[CollisionTransformer, CollisionTransformer, CollisionTransformer, BPETokenizer]:
    tokenizer = BPETokenizer()
    tokenizer.load(TOKENIZER_DIR)
    
    def load_ckpt(path: str) -> CollisionTransformer:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        cfg_dict = ckpt.get("config", {})
        if isinstance(cfg_dict, ModelConfig):
            config = cfg_dict
        else:
            config = ModelConfig(**cfg_dict)
        model = CollisionTransformer(config)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        return model

    m_prod = load_ckpt(PROD_MODEL_PATH)
    m_j52 = load_ckpt(J52_MODEL_PATH)
    m_j71 = load_ckpt(J71_MODEL_PATH)
    
    return m_prod, m_j52, m_j71, tokenizer

def run_tokenizer_diagnostic(tokenizer: BPETokenizer) -> Dict[str, Any]:
    vocab_size = len(tokenizer.vocab)
    
    sample_texts = {
        "sft_v3": [],
        "orig_corpus": [],
    }
    
    sft_train = os.path.join(SFT_V3_DIR, "train.jsonl")
    if os.path.exists(sft_train):
        with open(sft_train, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    text = item.get("prompt", "") + " " + item.get("response", "")
                    sample_texts["sft_v3"].append(text)
    
    latest_clean = os.path.join(PROJECT_ROOT, "data", "processed", "v4", "cleaned.txt")
    if not os.path.exists(latest_clean):
        for root, dirs, files in os.walk(os.path.join(PROJECT_ROOT, "data")):
            if "cleaned.txt" in files:
                latest_clean = os.path.join(root, "cleaned.txt")
                break
    if os.path.exists(latest_clean):
        with open(latest_clean, "r", encoding="utf-8") as f:
            sample_texts["orig_corpus"] = [f.read(50000)]

    def analyze_tokens(texts: List[str]) -> Dict[str, Any]:
        if not texts:
            return {}
        all_ids = []
        sample_lengths = []
        unk_count = 0
        total_tokens = 0
        fragmented_tokens = 0
        
        for text in texts[:100]:
            ids = tokenizer.encode(text, bos=False, eos=False)
            sample_lengths.append(len(ids))
            all_ids.extend(ids)
            for idx in ids:
                total_tokens += 1
                if idx == tokenizer.special_tokens.get("[UNK]", 257):
                    unk_count += 1
                token_val = tokenizer.inverse_vocab.get(idx, b"")
                if isinstance(token_val, bytes) and len(token_val) <= 2 and idx < 260:
                    fragmented_tokens += 1

        avg_len = sum(sample_lengths) / len(sample_lengths) if sample_lengths else 0
        sorted_lens = sorted(sample_lengths)
        median_len = sorted_lens[len(sorted_lens)//2] if sorted_lens else 0
        max_len = max(sample_lengths) if sample_lengths else 0
        
        rare_count = sum(1 for idx in all_ids if idx >= 260 and idx not in tokenizer.merges.values())
        
        return {
            "sample_count": len(sample_lengths),
            "total_tokens": total_tokens,
            "avg_tokens_per_sample": round(avg_len, 2),
            "median_tokens_per_sample": median_len,
            "max_seq_len": max_len,
            "unknown_token_rate": round(unk_count / total_tokens, 6) if total_tokens else 0.0,
            "token_fragmentation_ratio": round(fragmented_tokens / total_tokens, 4) if total_tokens else 0.0,
            "rare_token_percentage": round(rare_count / total_tokens, 4) if total_tokens else 0.0,
        }

    return {
        "vocabulary_size": vocab_size,
        "special_tokens": tokenizer.special_tokens,
        "sft_v3_analysis": analyze_tokens(sample_texts["sft_v3"]),
        "orig_corpus_analysis": analyze_tokens(sample_texts["orig_corpus"]),
    }

def audit_sft_dataset() -> Dict[str, Any]:
    train_file = os.path.join(SFT_V3_DIR, "train.jsonl")
    val_file = os.path.join(SFT_V3_DIR, "validation.jsonl")
    
    train_records = []
    val_records = []
    if os.path.exists(train_file):
        with open(train_file, "r", encoding="utf-8") as f:
            train_records = [json.loads(line) for line in f if line.strip()]
    if os.path.exists(val_file):
        with open(val_file, "r", encoding="utf-8") as f:
            val_records = [json.loads(line) for line in f if line.strip()]
            
    all_records = train_records + val_records
    total_records = len(all_records)
    
    prompt_lens = [len(r.get("prompt", "")) for r in all_records]
    resp_lens = [len(r.get("response", "")) for r in all_records]
    
    prompts = [r.get("prompt", "").strip() for r in all_records]
    unique_prompts = len(set(prompts))
    duplicate_ratio = round(1.0 - (unique_prompts / total_records), 4) if total_records else 0.0
    
    tech_count = 0
    conv_count = 0
    qna_count = 0
    expl_count = 0
    completion_count = 0
    
    for r in all_records:
        inst = r.get("prompt", "").lower()
        cat = r.get("type", "").lower()
        if "structured_technical" in cat or "code" in inst or "function" in inst or "algorithm" in inst:
            tech_count += 1
        else:
            conv_count += 1
            
        if "explain" in inst or "overview" in inst or "what is" in inst:
            expl_count += 1
        elif "?" in inst or "query" in inst:
            qna_count += 1
        else:
            completion_count += 1

    return {
        "total_records": total_records,
        "train_count": len(train_records),
        "validation_count": len(val_records),
        "prompt_length_stats": {
            "avg": round(sum(prompt_lens) / total_records, 2) if total_records else 0,
            "min": min(prompt_lens) if prompt_lens else 0,
            "max": max(prompt_lens) if prompt_lens else 0
        },
        "response_length_stats": {
            "avg": round(sum(resp_lens) / total_records, 2) if total_records else 0,
            "min": min(resp_lens) if resp_lens else 0,
            "max": max(resp_lens) if resp_lens else 0
        },
        "duplicate_prompt_ratio": duplicate_ratio,
        "technical_ratio": round(tech_count / total_records, 4) if total_records else 0.0,
        "conversational_ratio": round(conv_count / total_records, 4) if total_records else 0.0,
        "explanatory_examples": expl_count,
        "qna_examples": qna_count,
        "completion_examples": completion_count
    }

def get_fixed_evaluation_prompts() -> Dict[str, List[str]]:
    # 7 categories x 3 representative prompts = 21 prompts for ultra-fast diagnostic
    return {
        "general_knowledge": [
            "Artificial intelligence is",
            "The solar system consists of",
            "Photosynthesis is the process by which"
        ],
        "explanation": [
            "Explain what computer RAM does in simple terms.",
            "Explain the concept of gravity to a high school student.",
            "Explain how vaccines prepare the immune system."
        ],
        "instruction_following": [
            "List 3 major benefits of exercise.",
            "Summarize the main goal of machine learning in one sentence.",
            "Write a 3-step guide for making tea."
        ],
        "technical": [
            "Define the time complexity of binary search.",
            "What is a primary key in a relational database?",
            "Explain the purpose of an API interface."
        ],
        "completion": [
            "In modern computing systems, CPU architecture plays a key role because",
            "When developing scalable web applications, engineers must ensure that",
            "The primary objective of statistical data analysis is to"
        ],
        "conversational": [
            "Hello! How can you help me today?",
            "Can you assist me with a quick question?",
            "What kind of tasks are you designed to perform?"
        ],
        "reasoning": [
            "If all mammals breathe air, and a dolphin is a mammal, what follows?",
            "If A is larger than B, and B is larger than C, how does A compare to C?",
            "A train travels 60 miles in 1 hour. What is its average speed?"
        ]
    }

def classify_response(prompt: str, response: str, tokenizer: BPETokenizer) -> List[str]:
    labels = []
    resp_strip = response.strip()
    
    if not resp_strip:
        labels.append("EMPTY")
        return labels
        
    unique_ratio, unigram_rep, bigram_rep, _, _ = calculate_repetition_metrics(resp_strip, tokenizer)
    if unigram_rep > 0.45 or bigram_rep > 0.55:
        labels.append("REPETITIVE")
        
    if "\ufffd" in response or re.search(r'(?:[^\s\w\.\,\?\!\-\:\;\"]){3,}', response):
        labels.append("TOKENIZER_ARTIFACT")
        
    words = resp_strip.split()
    short_words = [w for w in words if len(w) == 1 and w.isalpha()]
    if len(words) > 0 and (len(short_words) / len(words) > 0.3 or "  " in resp_strip):
        labels.append("FRAGMENTED")
        
    prompt_lower = prompt.lower()
    if any(k in prompt_lower for k in ["list", "summarize", "write", "provide", "name", "state", "explain"]):
        if len(words) >= 4 and not ("EMPTY" in labels or "FRAGMENTED" in labels):
            labels.append("INSTRUCTION_FOLLOWING")
        else:
            labels.append("KNOWLEDGE_FAILURE")
            
    if not ("EMPTY" in labels or "FRAGMENTED" in labels or "REPETITIVE" in labels or "TOKENIZER_ARTIFACT" in labels):
        labels.append("COHERENT")
        
    return labels

def generate_text(model: CollisionTransformer, tokenizer: BPETokenizer, prompt: str, 
                  max_tokens: int = 24, temperature: float = 0.0, top_k: int = 50, top_p: float = 0.9) -> str:
    model.eval()
    ids = tokenizer.encode(prompt, bos=True)
    x = torch.tensor([ids], dtype=torch.long)
    
    with torch.no_grad():
        for _ in range(max_tokens):
            x_cond = x if x.size(1) <= model.config.max_seq_len else x[:, -model.config.max_seq_len:]
            logits, _ = model(x_cond)
            next_token_logits = logits[0, -1, :].clone()
            
            if temperature > 0.0:
                next_token_logits = next_token_logits / temperature
                filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
                probs = F.softmax(filtered_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits).unsqueeze(0)
                
            x = torch.cat((x, next_token.unsqueeze(0)), dim=1)
            if next_token.item() == tokenizer.special_tokens.get("[EOS]", 259):
                break
                
    gen_ids = x[0][len(ids):].tolist()
    return tokenizer.decode(gen_ids)

def run_benchmark(m_prod: CollisionTransformer, m_j52: CollisionTransformer, m_j71: CollisionTransformer, 
                  tokenizer: BPETokenizer) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    prompts_dict = get_fixed_evaluation_prompts()
    models = {
        "baseline": m_prod,
        "j52": m_j52,
        "j71": m_j71
    }
    
    all_generations = []
    print("\nRunning Fast Generation Benchmark...", flush=True)
    
    results_by_model = {
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

    for cat_name, cat_prompts in prompts_dict.items():
        for p_idx, prompt in enumerate(cat_prompts):
            for m_name, model in models.items():
                set_seed(42 + p_idx)
                resp_greedy = generate_text(model, tokenizer, prompt, temperature=0.0)
                labels_greedy = classify_response(prompt, resp_greedy, tokenizer)
                u_ratio, unigram_rep, bigram_rep, _, _ = calculate_repetition_metrics(resp_greedy, tokenizer)
                
                gen_entry = {
                    "category": cat_name,
                    "prompt": prompt,
                    "model": m_name,
                    "temperature": 0.0,
                    "response": resp_greedy,
                    "labels": labels_greedy,
                    "metrics": {
                        "length": len(resp_greedy),
                        "unique_token_ratio": round(u_ratio, 4),
                        "unigram_repetition": round(unigram_rep, 4),
                        "bigram_repetition": round(bigram_rep, 4)
                    }
                }
                all_generations.append(gen_entry)
                
                for temp in [0.5, 1.0]:
                    set_seed(42 + p_idx)
                    resp_samp = generate_text(model, tokenizer, prompt, temperature=temp)
                    labels_samp = classify_response(prompt, resp_samp, tokenizer)
                    all_generations.append({
                        "category": cat_name,
                        "prompt": prompt,
                        "model": m_name,
                        "temperature": temp,
                        "response": resp_samp,
                        "labels": labels_samp,
                    })

    greedy_gens = [g for g in all_generations if g["temperature"] == 0.0]
    total_evals_per_model = len(prompts_dict) * 3 # 21
    
    for m_name in models:
        m_gens = [g for g in greedy_gens if g["model"] == m_name]
        
        coherent_cnt = sum(1 for g in m_gens if "COHERENT" in g["labels"])
        inst_cnt = sum(1 for g in m_gens if "INSTRUCTION_FOLLOWING" in g["labels"])
        empty_cnt = sum(1 for g in m_gens if "EMPTY" in g["labels"])
        frag_cnt = sum(1 for g in m_gens if "FRAGMENTED" in g["labels"])
        rep_cnt = sum(1 for g in m_gens if "REPETITIVE" in g["labels"])
        
        avg_u_ratio = sum(g["metrics"]["unique_token_ratio"] for g in m_gens) / total_evals_per_model
        avg_unigram_rep = sum(g["metrics"]["unigram_repetition"] for g in m_gens) / total_evals_per_model
        avg_bigram_rep = sum(g["metrics"]["bigram_repetition"] for g in m_gens) / total_evals_per_model
        avg_len = sum(g["metrics"]["length"] for g in m_gens) / total_evals_per_model
        
        coherence_score = round(coherent_cnt / total_evals_per_model, 4)
        inst_score = round(inst_cnt / total_evals_per_model, 4)
        capability_score = round(0.5 * coherence_score + 0.3 * inst_score + 0.2 * avg_u_ratio, 4)
        
        cat_scores = {}
        for cat_name in prompts_dict:
            cat_gens = [g for g in m_gens if g["category"] == cat_name]
            c_coh = sum(1 for g in cat_gens if "COHERENT" in g["labels"]) / len(cat_gens)
            c_inst = sum(1 for g in cat_gens if "INSTRUCTION_FOLLOWING" in g["labels"]) / len(cat_gens)
            cat_scores[cat_name] = round(0.5 * c_coh + 0.5 * c_inst, 4)

        results_by_model[m_name] = {
            "overall_capability": capability_score,
            "coherence": coherence_score,
            "instruction_following": inst_score,
            "unique_token_ratio": round(avg_u_ratio, 4),
            "unigram_repetition": round(avg_unigram_rep, 4),
            "bigram_repetition": round(avg_bigram_rep, 4),
            "avg_response_length": round(avg_len, 2),
            "empty_rate": round(empty_cnt / total_evals_per_model, 4),
            "fragmented_rate": round(frag_cnt / total_evals_per_model, 4),
            "repetitive_rate": round(rep_cnt / total_evals_per_model, 4),
            "category_scores": cat_scores
        }
        
    return results_by_model, all_generations

