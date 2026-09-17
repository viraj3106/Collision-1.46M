"""
Phase 98 Test Suite — Grounded Synthesis Engine.
"""

import os
import sys
import hashlib
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from collision.rag.schemas import DocumentChunk
from collision.rag.index import VectorIndex
from collision.rag.retriever import DocumentRetriever
from collision.web.search import MockWebSearchProvider
from collision.routing.schemas import RouteMode, FusedEvidence
from collision.grounding.schemas import AnswerType, GroundedSynthesisResult
from collision.grounding.engine import GroundedSynthesisEngine


def test_checkpoint_integrity_phase98():
    """Verify protected model checkpoint SHA-256 hashes are immutable."""
    flagship_path = os.path.join(PROJECT_ROOT, "models", "collision-10m", "model.pt")
    research_path = os.path.join(PROJECT_ROOT, "models", "phase91_v9_10m", "model.pt")

    assert os.path.exists(flagship_path)
    assert os.path.exists(research_path)

    with open(flagship_path, "rb") as f:
        flagship_hash = hashlib.sha256(f.read()).hexdigest()
    with open(research_path, "rb") as f:
        research_hash = hashlib.sha256(f.read()).hexdigest()

    assert flagship_hash == "d256d46d962d6416fe22d2cfe80b13df0574279fb980d7d8576c2bdcf3775b97"
    assert research_hash == "98a2b416bed2033cd338b1f2e245e5b1d9681bdd66ba2a788409cddefd4be449"


def test_extractive_grounded_synthesis():
    """Test extraction-first grounded answering from local documentation."""
    index = VectorIndex()
    index.add([DocumentChunk(
        document_id="arch_spec",
        source="arch_spec.md",
        chunk_id=0,
        text="COLLISION 10M operates with 6 transformer layers and an embedding dimension of 512."
    )])
    retriever = DocumentRetriever(index=index)
    engine = GroundedSynthesisEngine(local_retriever=retriever)

    res = engine.answer("What is the embedding dimension in COLLISION 10M?", mode=RouteMode.LOCAL)
    assert res is not None
    assert res.answer_type in (AnswerType.EXTRACTIVE_ANSWER, AnswerType.GROUNDED_ANSWER)
    assert "512" in res.answer
    assert len(res.sources) > 0


def test_model_only_mode_separation():
    """Test that model-only questions are cleanly separated and not labeled grounded."""
    engine = GroundedSynthesisEngine()
    res = engine.answer("Hello, who are you?", mode=RouteMode.MODEL)
    assert res is not None
    assert res.answer_type == AnswerType.MODEL_ONLY
    assert res.route == RouteMode.MODEL
    assert len(res.sources) == 0
