"""
Phase 98 Test Suite — Extraction Fallback.
"""

import os
import sys
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from collision.routing.schemas import FusedEvidence, RouteMode
from collision.grounding.schemas import ExtractedFact, AnswerType
from collision.grounding.fallback import ExtractionFallbackHandler


def test_extraction_fallback_construction():
    handler = ExtractionFallbackHandler()

    facts = [
        ExtractedFact(
            fact_text="The company was founded in 2018.",
            fact_type="date",
            source="company_history.md",
            url="https://company.org/history",
            document_id="doc1",
            chunk_id=0,
            original_chunk_text="The company was founded in 2018 by researchers.",
            relevance_score=0.88,
            extraction_confidence=0.94
        )
    ]

    evidence = [
        FusedEvidence(
            source="company_history.md",
            url="https://company.org/history",
            document_id="doc1",
            chunk_id=0,
            text="The company was founded in 2018 by researchers.",
            score=0.88,
            source_type="WEB"
        )
    ]

    res = handler.construct_fallback_answer(
        question="When was the company founded?",
        extracted_facts=facts,
        fused_evidence=evidence,
        route=RouteMode.WEB
    )

    assert res is not None
    assert res.answer_type == AnswerType.EXTRACTIVE_ANSWER
    assert "founded in 2018" in res.answer
    assert res.is_fallback_used is True
    assert len(res.sources) == 1
    assert res.sources[0]["url"] == "https://company.org/history"
