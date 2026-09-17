from typing import List, Optional
from collision.routing.schemas import FusedEvidence, VerificationResult

class ConfidenceScorer:
    """
    Measurable, signal-derived confidence estimation for routed and grounded answers.
    Combines retrieval strength, claim verification rates, source diversity,
    and contradiction penalties without manufacturing arbitrary scores.
    """

    @staticmethod
    def calculate_confidence(
        verification: Optional[VerificationResult],
        fused_evidence: List[FusedEvidence],
        repetition_score: float = 0.0,
        is_model_only: bool = False
    ) -> float:
        """
        Calculates a calibrated confidence score in [0.0, 1.0].
        """
        if is_model_only:
            # Model-only conversational baseline confidence
            base = 0.85
            if repetition_score > 0.35:
                base -= 0.40
            return max(0.1, min(1.0, base))

        if verification is None:
            return 0.0

        if verification.has_conflicting_evidence:
            return 0.30

        if verification.contradicted_claims:
            return 0.10

        if not verification.supported and not verification.evidence_used:
            return 0.0

        # 1. Claim support score (0.0 to 0.50)
        claim_score = verification.confidence * 0.50

        # 2. Retrieval strength score (0.0 to 0.30)
        avg_retrieval = sum(e.score for e in fused_evidence) / max(1, len(fused_evidence)) if fused_evidence else 0.0
        ret_score = min(0.30, avg_retrieval * 0.6)

        # 3. Source diversity score (0.0 to 0.20)
        num_sources = len(set(e.source for e in fused_evidence))
        diversity_score = min(0.20, num_sources * 0.10)

        total_confidence = claim_score + ret_score + diversity_score

        # Repetition penalty
        if repetition_score > 0.35:
            total_confidence -= 0.30

        return max(0.0, min(1.0, round(total_confidence, 4)))
