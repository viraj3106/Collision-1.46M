"""
Phase 98 Test Suite — Evidence Extraction.
"""

import os
import sys
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from collision.routing.schemas import FusedEvidence
from collision.grounding.extractor import EvidenceExtractor


def test_date_and_number_extraction():
    extractor = EvidenceExtractor()
    evidence = [
        FusedEvidence(
            source="python_release.md",
            url="https://python.org/3.13",
            document_id="py_doc",
            chunk_id=0,
            text="Python 3.13 was officially released on October 7, 2024. It introduced a new JIT compiler.",
            score=0.9,
            source_type="WEB"
        )
    ]

    facts = extractor.extract_relevant_spans(
        question="What was the release date of Python 3.13 in 2024?",
        fused_evidence=evidence
    )

    assert len(facts) > 0
    top_fact = facts[0]
    assert "October 7, 2024" in top_fact.fact_text
    assert top_fact.fact_type in ("date", "specification", "span")
    assert top_fact.url == "https://python.org/3.13"
    assert top_fact.chunk_id == 0


def test_specification_extraction():
    extractor = EvidenceExtractor()
    evidence = [
        FusedEvidence(
            source="collision_spec.md",
            url="local://collision_spec.md",
            document_id="spec_doc",
            chunk_id=0,
            text="COLLISION-10M has 10,485,760 parameters and uses 6 layers.",
            score=0.85,
            source_type="LOCAL"
        )
    ]

    facts = extractor.extract_relevant_spans(
        question="How many parameters are in COLLISION 10M?",
        fused_evidence=evidence
    )

    assert len(facts) > 0
    assert "10,485,760" in facts[0].fact_text
    assert facts[0].relevance_score > 0.15
