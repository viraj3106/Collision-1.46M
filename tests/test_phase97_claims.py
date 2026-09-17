"""
Phase 97 Test Suite — Claim Extraction & Verification Auditor.
"""

import os
import sys
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from evaluation.phase97_claims import ClaimAuditor
from collision.routing.schemas import FusedEvidence


def test_claim_extraction():
    auditor = ClaimAuditor()
    text = "COLLISION 10M has 6 layers. It uses 8 attention heads. The context length is 1024 tokens."
    claims = auditor.extract_claims(text)
    assert len(claims) == 3
    assert "COLLISION 10M has 6 layers." in claims[0]
    assert "attention heads" in claims[1]


def test_claim_supported_and_unsupported():
    auditor = ClaimAuditor()
    evidence = [
        FusedEvidence(
            source="spec.md",
            url="spec.md",
            document_id="spec_doc",
            chunk_id=0,
            text="COLLISION 10M operates with 6 transformer layers and 8 attention heads.",
            score=0.9,
            source_type="LOCAL"
        )
    ]

    answer = "COLLISION 10M has 6 transformer layers. The model was trained on Mars by aliens."
    res = auditor.audit_answer(answer, evidence)

    assert res.total_claims == 2
    assert res.supported_claims >= 1
    assert res.unsupported_claims >= 1
    assert res.claim_support_rate < 1.0


def test_claim_contradiction_detection():
    auditor = ClaimAuditor()
    evidence = [
        FusedEvidence(
            source="spec.md",
            url="spec.md",
            document_id="spec_doc",
            chunk_id=0,
            text="The architecture employs 6 layers and 8 attention heads.",
            score=0.9,
            source_type="LOCAL"
        )
    ]

    # Contradictory number: 64 layers instead of 6
    answer = "The architecture employs 64 layers and 8 attention heads."
    res = auditor.audit_answer(answer, evidence)

    assert res.contradicted_claims >= 1
    assert res.contradiction_rate > 0.0
