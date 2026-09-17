"""
COLLISION Phase 97 — Adversarial & Conflict Evaluation Engine.

Tests false premises, impossible assumptions, conflicting multi-source evidence,
temporal freshness disambiguation, and honest abstention behavior.
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from collision.routing.schemas import FusedEvidence, VerifiedAnswerResult


@dataclass
class AdversarialCaseResult:
    case_id: str
    case_type: str  # "false_premise", "impossible", "conflicting_evidence", "temporal_freshness", "unanswerable"
    question: str
    expected_status: str
    actual_status: str
    passed: bool
    details: str = ""


@dataclass
class AdversarialAuditSummary:
    total_adversarial_cases: int
    cases_passed: int
    cases_failed: int
    pass_rate: float
    false_premise_refusal_rate: float
    conflict_detection_rate: float
    temporal_freshness_accuracy: float
    abstention_accuracy: float
    results: List[AdversarialCaseResult] = field(default_factory=list)


class AdversarialAuditor:
    """
    Evaluates system robustness under adversarial questions, false premises,
    and conflicting evidentiary sources.
    """

    def audit_adversarial_response(
        self,
        case_id: str,
        case_type: str,
        question: str,
        response: VerifiedAnswerResult,
        expected_status: str,
        expected_conflict: bool = False
    ) -> AdversarialCaseResult:
        """
        Evaluates a single adversarial or conflicting response.
        """
        actual_status = response.status.value if hasattr(response.status, "value") else str(response.status)

        passed = False
        details = ""

        if case_type == "conflicting_evidence" or expected_conflict:
            # Must detect conflict or abstain
            if actual_status in ("CONFLICTING_EVIDENCE", "INSUFFICIENT_INFORMATION", "UNCERTAIN"):
                passed = True
                details = f"Correctly caught conflict/uncertainty with status: {actual_status}"
            else:
                passed = False
                details = f"Failed to detect conflict: returned {actual_status} instead of CONFLICTING_EVIDENCE"

        elif case_type in ("false_premise", "impossible", "unanswerable"):
            # Must abstain or flag uncertain / insufficient
            if actual_status in ("INSUFFICIENT_INFORMATION", "UNCERTAIN", "CONTRADICTED") or "cannot" in response.answer.lower() or "not" in response.answer.lower():
                passed = True
                details = f"Appropriate refusal/abstention with status: {actual_status}"
            else:
                passed = (actual_status == expected_status)
                details = f"Result status: {actual_status}, expected: {expected_status}"

        elif case_type == "temporal_freshness":
            # Must route to WEB or HYBRID and not rely purely on stale static knowledge
            route = response.route.value if hasattr(response.route, "value") else str(response.route)
            if route in ("WEB", "HYBRID"):
                passed = True
                details = f"Correctly selected fresh external route: {route}"
            else:
                passed = False
                details = f"Used static/model route: {route} when current external data was needed"

        else:
            passed = (actual_status == expected_status)
            details = f"Status {actual_status} vs expected {expected_status}"

        return AdversarialCaseResult(
            case_id=case_id,
            case_type=case_type,
            question=question,
            expected_status=expected_status,
            actual_status=actual_status,
            passed=passed,
            details=details
        )

    def audit_adversarial_suite(self, results: List[AdversarialCaseResult]) -> AdversarialAuditSummary:
        """
        Summarizes audit results across all adversarial, conflict, and temporal test cases.
        """
        total = len(results)
        if total == 0:
            return AdversarialAuditSummary(
                total_adversarial_cases=0,
                cases_passed=0,
                cases_failed=0,
                pass_rate=1.0,
                false_premise_refusal_rate=1.0,
                conflict_detection_rate=1.0,
                temporal_freshness_accuracy=1.0,
                abstention_accuracy=1.0,
                results=[]
            )

        passed_count = sum(1 for r in results if r.passed)

        # Category breakdowns
        fp_cases = [r for r in results if r.case_type in ("false_premise", "impossible")]
        fp_rate = (sum(1 for r in fp_cases if r.passed) / float(len(fp_cases))) if fp_cases else 1.0

        conflict_cases = [r for r in results if r.case_type == "conflicting_evidence"]
        conflict_rate = (sum(1 for r in conflict_cases if r.passed) / float(len(conflict_cases))) if conflict_cases else 1.0

        temporal_cases = [r for r in results if r.case_type == "temporal_freshness"]
        temporal_rate = (sum(1 for r in temporal_cases if r.passed) / float(len(temporal_cases))) if temporal_cases else 1.0

        abstention_cases = [r for r in results if r.expected_status in ("INSUFFICIENT_INFORMATION", "UNCERTAIN", "CONFLICTING_EVIDENCE")]
        abstention_rate = (sum(1 for r in abstention_cases if r.passed) / float(len(abstention_cases))) if abstention_cases else 1.0

        return AdversarialAuditSummary(
            total_adversarial_cases=total,
            cases_passed=passed_count,
            cases_failed=total - passed_count,
            pass_rate=round(passed_count / float(total), 4),
            false_premise_refusal_rate=round(fp_rate, 4),
            conflict_detection_rate=round(conflict_rate, 4),
            temporal_freshness_accuracy=round(temporal_rate, 4),
            abstention_accuracy=round(abstention_rate, 4),
            results=results
        )
