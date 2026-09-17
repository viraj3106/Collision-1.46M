import os
import sys
import pytest
import json
import hashlib

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.phase73.train_phase73 import verify_immutability
from data.tokenize import BPETokenizer

GOLD_SET_PATH = os.path.join(PROJECT_ROOT, "data", "instructions", "collision_conversation_v1_gold", "gold_set.jsonl")
REJECTED_DIR = os.path.join(PROJECT_ROOT, "data", "instructions", "collision_conversation_v1_rejected")
EXPECTED_SHA256 = "d256d46d962d6416fe22d2cfe80b13df0574279fb980d7d8576c2bdcf3775b97"

def test_checkpoint_immutability():
    sha = verify_immutability()
    assert sha.lower() == EXPECTED_SHA256.lower()

def test_tokenizer_loading():
    tokenizer = BPETokenizer()
    tokenizer.load(os.path.join(PROJECT_ROOT, "artifacts", "tokenizer"))
    assert len(tokenizer.vocab) > 260

def test_rejected_archive_exists():
    reason_p = os.path.join(REJECTED_DIR, "REJECTION_REASON.txt")
    assert os.path.exists(reason_p), "Rejection reason file missing from archive!"
    with open(reason_p, "r", encoding="utf-8") as f:
        content = f.read()
    assert "SYNTHETIC_TEMPLATE_CONTAMINATION" in content

def test_gold_set_structure_and_counts():
    assert os.path.exists(GOLD_SET_PATH), "gold_set.jsonl missing!"
    records = []
    with open(GOLD_SET_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
                
    assert len(records) == 300
    
    turn_counts = [len(r["turns"]) // 2 for r in records]
    assert sum(1 for c in turn_counts if c == 1) == 100
    assert sum(1 for c in turn_counts if c == 2) == 100
    assert sum(1 for c in turn_counts if c == 3) == 100
    
    for r in records[:20]:
        assert "id" in r
        assert "category" in r
        assert "domain" in r
        assert "turns" in r
        for t in r["turns"]:
            assert t["role"] in ["user", "assistant"]
            assert len(t["content"]) > 0

def test_gold_set_audit_reports():
    audit_p = os.path.join(PROJECT_ROOT, "experiments", "phase74", "reports", "gold_set_audit.json")
    samples_p = os.path.join(PROJECT_ROOT, "experiments", "phase74", "reports", "gold_set_samples.md")
    
    assert os.path.exists(audit_p), "gold_set_audit.json missing!"
    assert os.path.exists(samples_p), "gold_set_samples.md missing!"
    
    with open(audit_p, "r", encoding="utf-8") as f:
        audit = json.load(f)
    assert audit["total_conversations"] == 300
    assert audit["placeholder_contamination_count"] == 0
    assert audit["exact_duplicate_prompts"] == 0
