import os
import sys
import pytest
import hashlib
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.phase72.phase72_diagnostic import (
    verify_immutability,
    load_models,
    run_tokenizer_diagnostic,
    audit_sft_dataset,
    get_fixed_evaluation_prompts,
    classify_response,
    generate_text
)
from data.tokenize import BPETokenizer

PROD_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "collision-10m", "model.pt")
EXPECTED_SHA256 = "d256d46d962d6416fe22d2cfe80b13df0574279fb980d7d8576c2bdcf3775b97"

def test_checkpoint_immutability():
    sha_dict = verify_immutability()
    assert sha_dict["production"].lower() == EXPECTED_SHA256.lower()

def test_tokenizer_loading():
    tokenizer = BPETokenizer()
    tokenizer.load(os.path.join(PROJECT_ROOT, "artifacts", "tokenizer"))
    assert len(tokenizer.vocab) > 260
    assert "[BOS]" in tokenizer.special_tokens

def test_evaluation_set_consistency():
    prompts = get_fixed_evaluation_prompts()
    assert len(prompts) == 7
    for cat, p_list in prompts.items():
        assert len(p_list) > 0

def test_classification_logic():
    tokenizer = BPETokenizer()
    tokenizer.load(os.path.join(PROJECT_ROOT, "artifacts", "tokenizer"))
    
    empty_labels = classify_response("test", "", tokenizer)
    assert "EMPTY" in empty_labels
    
    rep_labels = classify_response("test", "the the the the the the the the the the", tokenizer)
    assert "REPETITIVE" in rep_labels

def test_deterministic_generation():
    m_prod, _, _, tokenizer = load_models()
    torch.manual_seed(42)
    res1 = generate_text(m_prod, tokenizer, "Artificial intelligence is", max_tokens=10, temperature=0.0)
    torch.manual_seed(42)
    res2 = generate_text(m_prod, tokenizer, "Artificial intelligence is", max_tokens=10, temperature=0.0)
    assert res1 == res2
