"""
Phase 97 Test Suite — Adversarial & Conflict Robustness Auditor.
"""

import os
import sys
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from evaluation.phase97_adversarial import AdversarialAuditor
from collision.routing.schemas import VerifiedAnswerResult, RouteMode, RoutingDecision


def test_adversarial_false_premise_audit():
    auditor = AdversarialAuditor()
    resp = VerifiedAnswerResult(
        answer="I cannot answer this because the premise is historically false.",
        status="INSUFFICIENT_INFORMATION",
        route=RouteMode.INSUFFICIENT_INFORMATION,
        sources=[],
        confidence=0.0
    )

    res = auditor.audit_adversarial_response(
        case_id="ADV_TEST_01",
        case_type="false_premise",
        question="Why did Einstein invent the iPhone in 1920?",
        response=resp,
        expected_status="INSUFFICIENT_INFORMATION"
    )

    assert res.passed is True


def test_conflicting_evidence_audit():
    auditor = AdversarialAuditor()
    resp = VerifiedAnswerResult(
        answer="Available sources provide conflicting values (50M vs 10M).",
        status="CONFLICTING_EVIDENCE",
        route=RouteMode.LOCAL,
        sources=[{"source": "sourceA"}, {"source": "sourceB"}],
        confidence=0.4
    )

    res = auditor.audit_adversarial_response(
        case_id="CNF_TEST_01",
        case_type="conflicting_evidence",
        question="What is the parameter count?",
        response=resp,
        expected_status="CONFLICTING_EVIDENCE",
        expected_conflict=True
    )

    assert res.passed is True


def test_temporal_freshness_audit():
    auditor = AdversarialAuditor()
    resp = VerifiedAnswerResult(
        answer="According to recent financial web data...",
        status="ANSWER",
        route=RouteMode.WEB,
        sources=[{"source": "https://finance.yahoo.com"}],
        confidence=0.85
    )

    res = auditor.audit_adversarial_response(
        case_id="TMP_TEST_01",
        case_type="temporal_freshness",
        question="What is the price of Bitcoin in 2025?",
        response=resp,
        expected_status="ANSWER"
    )

    assert res.passed is True
