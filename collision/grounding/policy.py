"""
COLLISION Phase 98 — Final Answer Policy.

Deterministic policy engine that gates final answer acceptance, conflict declaration,
abstention, extractive routing, and model-only classification.
"""

from typing import List, Optional
from collision.grounding.schemas import AnswerType, GroundedSynthesisResult
from collision.routing.schemas import RouteMode, FusedEvidence, VerificationResult


class FinalAnswerPolicy:
    """
    Applies deterministic validation rules before accepting an answer.
    """

    @staticmethod
    def determine_answer_type(
        route: RouteMode,
        has_conflict: bool,
        has_evidence: bool,
        is_extracted: bool,
        is_model_synthesized: bool,
        is_verified_supported: bool
    ) -> AnswerType:
        """
        Determines the final answer type.
        """
        if has_conflict:
            return AnswerType.CONFLICTING_EVIDENCE

        if route == RouteMode.INSUFFICIENT_INFORMATION or not has_evidence:
            if route == RouteMode.MODEL:
                return AnswerType.MODEL_ONLY
            return AnswerType.INSUFFICIENT_INFORMATION

        if route == RouteMode.MODEL:
            return AnswerType.MODEL_ONLY

        if is_model_synthesized and is_verified_supported:
            return AnswerType.MODEL_SYNTHESIZED_GROUNDED_ANSWER

        if is_extracted:
            return AnswerType.EXTRACTIVE_ANSWER

        if is_verified_supported:
            return AnswerType.GROUNDED_ANSWER

        return AnswerType.EXTRACTIVE_ANSWER
