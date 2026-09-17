"""
Phase 98 Test Suite — Conflict Handling.
"""

import os
import sys
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from collision.rag.schemas import DocumentChunk
from collision.rag.index import VectorIndex
from collision.rag.retriever import DocumentRetriever
from collision.grounding.engine import GroundedSynthesisEngine
from collision.grounding.schemas import AnswerType
from collision.routing.schemas import RouteMode


def test_multi_source_conflict_handling():
    index = VectorIndex()
    index.add([
        DocumentChunk(
            document_id="spec_v1",
            source="spec_v1.md",
            chunk_id=0,
            text="The cluster node contains 64 processor cores and 128GB RAM."
        ),
        DocumentChunk(
            document_id="spec_v2",
            source="spec_v2.md",
            chunk_id=1,
            text="The cluster node contains 8 processor cores and 128GB RAM."
        )
    ])

    retriever = DocumentRetriever(index=index)
    engine = GroundedSynthesisEngine(local_retriever=retriever)

    res = engine.answer("How many processor cores are in the cluster node?", mode=RouteMode.LOCAL)
    assert res is not None
    assert res.answer_type == AnswerType.CONFLICTING_EVIDENCE
    assert res.status == "CONFLICTING_EVIDENCE"
    assert len(res.sources) >= 2
