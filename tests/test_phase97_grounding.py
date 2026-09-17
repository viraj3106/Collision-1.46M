"""
Phase 97 Test Suite — Grounding Validation & End-to-End Audit.
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
from collision.routing.engine import AdaptiveKnowledgeEngine
from collision.routing.schemas import RouteMode, VerifiedAnswerResult, FusedEvidence


def test_checkpoint_integrity_phase97():
    """Verify protected model checkpoint SHA-256 hashes are immutable."""
    flagship_path = os.path.join(PROJECT_ROOT, "models", "collision-10m", "model.pt")
    research_path = os.path.join(PROJECT_ROOT, "models", "phase91_v9_10m", "model.pt")

    assert os.path.exists(flagship_path), f"Flagship checkpoint not found: {flagship_path}"
    assert os.path.exists(research_path), f"Research checkpoint not found: {research_path}"

    with open(flagship_path, "rb") as f:
        flagship_hash = hashlib.sha256(f.read()).hexdigest()
    with open(research_path, "rb") as f:
        research_hash = hashlib.sha256(f.read()).hexdigest()

    assert flagship_hash == "d256d46d962d6416fe22d2cfe80b13df0574279fb980d7d8576c2bdcf3775b97"
    assert research_hash == "98a2b416bed2033cd338b1f2e245e5b1d9681bdd66ba2a788409cddefd4be449"


def test_end_to_end_grounding_audit():
    """Test full pipeline with local knowledge indexing and verification."""
    index = VectorIndex()
    index.add([DocumentChunk(
        document_id="arch_spec",
        source="arch_spec.md",
        chunk_id=0,
        text="COLLISION 10M operates with 6 transformer layers and an embedding dimension of 512."
    )])
    retriever = DocumentRetriever(index=index)
    engine = AdaptiveKnowledgeEngine(local_retriever=retriever)

    res = engine.answer("What is the embedding dimension in COLLISION 10M?", mode=RouteMode.LOCAL)
    assert res is not None
    assert res.status in ("ANSWER", "UNCERTAIN", "INSUFFICIENT_INFORMATION")
    assert len(res.fused_evidence) > 0
    assert len(res.sources) > 0
    assert any("arch_spec.md" in s.get("title", "") or "arch_spec.md" in s.get("url", "") for s in res.sources)


def test_grounding_refusal_on_empty_evidence():
    """Test that queries with empty/irrelevant evidence abstain honestly."""
    engine = AdaptiveKnowledgeEngine()
    res = engine.answer("asldkfjasldkfj qwpeoiruqweoiru ???", mode=RouteMode.AUTO)
    assert res.status == "INSUFFICIENT_INFORMATION"
    assert len(res.sources) == 0
