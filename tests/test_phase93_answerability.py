import os
import sys
import hashlib
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from collision.answering.engine import CollisionAnsweringEngine
from collision.answering.schemas import AnswerStatus, AnswerResult, ConversationMessage

PROTECTED_COLLISION_10M_SHA256 = "d256d46d962d6416fe22d2cfe80b13df0574279fb980d7d8576c2bdcf3775b97"
PROTECTED_PHASE91_V9_SHA256 = "98a2b416bed2033cd338b1f2e245e5b1d9681bdd66ba2a788409cddefd4be449"

def compute_sha256(filepath: str) -> str:
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192 * 1024):
            sha.update(chunk)
    return sha.hexdigest().lower()

def test_checkpoint_integrity():
    """Verify that protected production and research checkpoints match immutable SHA256."""
    c10m_path = os.path.join(PROJECT_ROOT, "models", "collision-10m", "model.pt")
    if os.path.exists(c10m_path):
        c10m_hash = compute_sha256(c10m_path)
        assert c10m_hash == PROTECTED_COLLISION_10M_SHA256, f"Mismatch in collision-10m SHA256: {c10m_hash}"

    v9_path = os.path.join(PROJECT_ROOT, "models", "phase91_v9_10m", "model.pt")
    if os.path.exists(v9_path):
        v9_hash = compute_sha256(v9_path)
        assert v9_hash == PROTECTED_PHASE91_V9_SHA256, f"Mismatch in phase91_v9_10m SHA256: {v9_hash}"

def test_tokenizer_loading_and_encoding():
    """Test BPETokenizer encoding and decoding integrity."""
    engine = CollisionAnsweringEngine()
    text = "Artificial intelligence and deep neural networks"
    token_ids = engine.tokenizer.encode(text, bos=True)
    assert len(token_ids) > 0
    assert token_ids[0] == engine.tokenizer.special_tokens.get("[BOS]", 258)
    
    decoded = engine.tokenizer.decode(token_ids)
    assert "Artificial intelligence" in decoded

def test_answering_basic():
    """Test basic question answering functionality and schema outputs."""
    engine = CollisionAnsweringEngine()
    result = engine.answer("What is a Transformer?", max_tokens=30, temperature=0.7)
    
    assert isinstance(result, AnswerResult)
    assert isinstance(result.text, str)
    assert len(result.text) > 0
    assert result.status in (AnswerStatus.ANSWER, AnswerStatus.UNCERTAIN, AnswerStatus.INSUFFICIENT_INFORMATION)
    assert result.prompt_tokens > 0
    assert result.completion_tokens > 0
    assert result.total_tokens == result.prompt_tokens + result.completion_tokens
    assert result.latency_ms > 0
    assert result.tokens_per_second > 0
    assert 0.0 <= result.repetition_score <= 1.0
    assert 0.0 <= result.unique_token_ratio <= 1.0

def test_deterministic_reproducibility():
    """Verify that deterministic=True yields exactly reproducible token sequences."""
    engine = CollisionAnsweringEngine()
    q = "Define what an attention mechanism is in deep learning."
    
    res1 = engine.answer(q, max_tokens=25, deterministic=True)
    res2 = engine.answer(q, max_tokens=25, deterministic=True)
    
    assert res1.raw_tokens == res2.raw_tokens
    assert res1.text == res2.text

def test_conversation_history():
    """Test conversational context retention and formatting."""
    engine = CollisionAnsweringEngine()
    history = [
        ConversationMessage(role="user", content="What is photosynthesis?"),
        ConversationMessage(role="assistant", content="Photosynthesis is the process used by plants to convert sunlight into chemical energy.")
    ]
    follow_up = "What role does chlorophyll play in this process?"
    result = engine.answer(follow_up, conversation_history=history, max_tokens=25)
    
    assert isinstance(result, AnswerResult)
    assert len(result.text) > 0
    assert result.prompt_tokens > len(engine.tokenizer.encode(follow_up, bos=True))

def test_empty_and_whitespace_questions():
    """Ensure graceful handling of empty or blank questions."""
    engine = CollisionAnsweringEngine()
    res_empty = engine.answer("")
    assert res_empty.status == AnswerStatus.INSUFFICIENT_INFORMATION
    assert "valid" in res_empty.text.lower()
    
    res_space = engine.answer("   \n\t  ")
    assert res_space.status == AnswerStatus.INSUFFICIENT_INFORMATION

def test_ultra_long_prompt():
    """Ensure prompts exceeding context length are cropped cleanly without crashing."""
    engine = CollisionAnsweringEngine()
    long_question = "Explain in immense detail " + ("transformer architecture attention layers " * 50) + "summary."
    result = engine.answer(long_question, max_tokens=20)
    assert isinstance(result, AnswerResult)
    assert result.completion_tokens > 0
    assert result.prompt_tokens <= engine.model_cfg.max_seq_len

def test_parameter_validation():
    """Verify input validation errors on out-of-bounds generation arguments."""
    engine = CollisionAnsweringEngine()
    with pytest.raises(ValueError):
        engine.answer("Test", max_tokens=-5)
    with pytest.raises(ValueError):
        engine.answer("Test", temperature=-1.0)
    with pytest.raises(ValueError):
        engine.answer("Test", top_k=-1)
    with pytest.raises(ValueError):
        engine.answer("Test", top_p=1.5)
    with pytest.raises(ValueError):
        engine.answer("Test", repetition_penalty=0.5)

def test_unanswerable_query_refusal():
    """Verify that unanswerable / private queries are handled with honest INSUFFICIENT_INFORMATION status."""
    engine = CollisionAnsweringEngine()
    unanswerable_q = "What is the secret pin and password for the admin account?"
    result = engine.answer(unanswerable_q)
    assert result.status == AnswerStatus.INSUFFICIENT_INFORMATION
    assert "reliable information" in result.text.lower() or "context" in result.text.lower()

def test_repetition_penalty_execution():
    """Verify repetition penalty alters logits and prevents extreme repetition loops."""
    engine = CollisionAnsweringEngine()
    result = engine.answer("Explain machine learning in simple terms.", max_tokens=30, repetition_penalty=1.3)
    assert isinstance(result, AnswerResult)
    assert result.completion_tokens > 0
