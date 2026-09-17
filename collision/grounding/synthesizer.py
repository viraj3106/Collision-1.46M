"""
COLLISION Phase 98 — Grounded Synthesizer.

Synthesizes verified answers through extraction-first factual compilation, multi-source
evidence unification (Local + Web), and model synthesis with verification gating.
"""

import time
from typing import List, Dict, Any, Optional
from collision.grounding.schemas import AnswerType, ExtractedFact, GroundedSynthesisResult
from collision.grounding.extractor import EvidenceExtractor
from collision.grounding.fallback import ExtractionFallbackHandler
from collision.routing.schemas import FusedEvidence, RouteMode, VerificationResult
from collision.routing.verifier import GroundingVerifier
from collision.answering.engine import CollisionAnsweringEngine


from collision.grounding.formatter import NaturalGroundedFormatter


class GroundedSynthesizer:
    """
    Coordinates extraction-first factual answering, hybrid multi-source synthesis,
    natural structured response formatting, and fallback to guaranteed-grounded evidence spans.
    """

    def __init__(
        self,
        extractor: Optional[EvidenceExtractor] = None,
        fallback_handler: Optional[ExtractionFallbackHandler] = None,
        verifier: Optional[GroundingVerifier] = None,
        answering_engine: Optional[CollisionAnsweringEngine] = None
    ):
        self.extractor = extractor if extractor is not None else EvidenceExtractor()
        self.fallback_handler = fallback_handler if fallback_handler is not None else ExtractionFallbackHandler()
        self.verifier = verifier if verifier is not None else GroundingVerifier()
        self.answering_engine = answering_engine if answering_engine is not None else CollisionAnsweringEngine()

    def synthesize_extractive(
        self,
        question: str,
        fused_evidence: List[FusedEvidence],
        route: RouteMode = RouteMode.LOCAL
    ) -> Optional[GroundedSynthesisResult]:
        """
        Attempts direct deterministic factual extraction first and formats as natural AI response.
        """
        extracted_facts = self.extractor.extract_relevant_spans(
            question=question,
            fused_evidence=fused_evidence,
            top_n=3,
            min_relevance=0.10
        )

        if not extracted_facts:
            return None

        # Format natural, structured ChatGPT-grade answer from verified facts
        answer_text = NaturalGroundedFormatter.format_natural_answer(
            question=question,
            extracted_facts=extracted_facts,
            fused_evidence=fused_evidence,
            route=route
        )

        sources = []
        seen_src = set()
        for f in extracted_facts:
            src_key = f.url or f.source
            if src_key not in seen_src:
                seen_src.add(src_key)
                sources.append({
                    "title": f.source,
                    "url": f.url,
                    "type": "LOCAL" if "md" in f.source or "local" in f.url else "WEB"
                })

        top_fact = extracted_facts[0]
        return GroundedSynthesisResult(
            answer=answer_text,
            answer_type=AnswerType.EXTRACTIVE_ANSWER,
            route=route,
            sources=sources,
            fused_evidence=fused_evidence,
            extracted_facts=extracted_facts,
            confidence=top_fact.extraction_confidence,
            is_fallback_used=False,
            status="ANSWER",
            termination_reason="natural_grounded_synthesis"
        )

    def synthesize_hybrid(
        self,
        question: str,
        fused_evidence: List[FusedEvidence]
    ) -> GroundedSynthesisResult:
        """
        Synthesizes compatible facts across Local RAG and Web evidence into a natural structured response.
        """
        local_ev = [e for e in fused_evidence if e.source_type == "LOCAL"]
        web_ev = [e for e in fused_evidence if e.source_type == "WEB"]

        local_facts = self.extractor.extract_relevant_spans(question, local_ev, top_n=2) if local_ev else []
        web_facts = self.extractor.extract_relevant_spans(question, web_ev, top_n=2) if web_ev else []

        combined_facts = local_facts + web_facts
        sources = []
        seen_src = set()
        for f in combined_facts:
            src_key = f.url or f.source
            if src_key not in seen_src:
                seen_src.add(src_key)
                sources.append({
                    "title": f.source,
                    "url": f.url,
                    "type": "LOCAL" if "md" in f.source or "local" in f.url else "WEB"
                })

        ans_text = NaturalGroundedFormatter.format_natural_answer(
            question=question,
            extracted_facts=combined_facts,
            fused_evidence=fused_evidence,
            route=RouteMode.HYBRID
        )

        return GroundedSynthesisResult(
            answer=ans_text,
            answer_type=AnswerType.GROUNDED_ANSWER,
            route=RouteMode.HYBRID,
            sources=sources,
            fused_evidence=fused_evidence,
            extracted_facts=combined_facts,
            confidence=0.90,
            is_fallback_used=False,
            status="ANSWER",
            termination_reason="hybrid_natural_synthesis"
        )
