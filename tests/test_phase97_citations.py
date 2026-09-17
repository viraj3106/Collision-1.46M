"""
Phase 97 Test Suite — Citation Verification Auditor.
"""

import os
import sys
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from evaluation.phase97_citations import CitationAuditor
from collision.routing.schemas import FusedEvidence, ClaimVerification


def test_valid_and_hallucinated_citations():
    auditor = CitationAuditor()
    evidence = [
        FusedEvidence(
            source="https://docs.python.org/3.13",
            url="https://docs.python.org/3.13",
            document_id="py313_doc",
            chunk_id=0,
            text="Python 3.13 was released on October 7, 2024 with JIT compiler support.",
            score=0.9,
            source_type="WEB"
        )
    ]

    answer = "Python 3.13 includes JIT compiler improvements."
    cited_sources = [
        "https://docs.python.org/3.13",
        "https://fake-citation-domain.org/invented-page"  # Hallucinated
    ]

    res = auditor.audit_citations(
        answer_text=answer,
        cited_sources=cited_sources,
        retrieved_evidence=evidence
    )

    assert res.total_citations_provided == 2
    assert res.valid_citations == 1
    assert res.hallucinated_citations == 1
    assert res.citation_correctness == 0.5


def test_citation_completeness():
    auditor = CitationAuditor()
    evidence = [
        FusedEvidence(
            source="doc1.md",
            url="doc1.md",
            document_id="d1",
            chunk_id=0,
            text="COLLISION has 6 layers.",
            score=0.85,
            source_type="LOCAL"
        )
    ]

    claim_verifications = [
        ClaimVerification(
            claim_text="COLLISION has 6 layers.",
            status="SUPPORTED",
            supporting_evidence=["doc1.md"],
            similarity_score=0.85
        )
    ]

    res = auditor.audit_citations(
        answer_text="COLLISION has 6 layers.",
        cited_sources=["doc1.md"],
        retrieved_evidence=evidence,
        claim_verifications=claim_verifications
    )

    assert res.citation_completeness == 1.0
