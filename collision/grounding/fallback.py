import re
from typing import List, Dict, Any, Optional
from collision.grounding.schemas import ExtractedFact, AnswerType, GroundedSynthesisResult
from collision.routing.schemas import FusedEvidence, RouteMode, VerificationResult
from collision.rag.embeddings import LocalEmbeddingModel


from collision.grounding.formatter import NaturalGroundedFormatter


class ExtractionFallbackHandler:
    """
    Safely recovers from ungrounded neural generation by assembling
    direct, verified factual statements from extracted evidence chunks in natural AI assistant format.
    """

    def construct_fallback_answer(
        self,
        question: str,
        extracted_facts: List[ExtractedFact],
        fused_evidence: List[FusedEvidence],
        route: RouteMode = RouteMode.LOCAL
    ) -> Optional[GroundedSynthesisResult]:
        """
        Builds a verified extractive answer from extracted facts or highest-relevance evidence.
        Returns None if evidence has no substantive content overlap with the user's question.
        """
        if not extracted_facts and not fused_evidence:
            return None

        # 1. If we have structured extracted facts, construct natural ChatGPT-like answer
        if extracted_facts:
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
                is_fallback_used=True,
                status="ANSWER",
                termination_reason="extraction_fallback"
            )

        # 2. Check if top fused evidence text has genuine relevance to the question
        if fused_evidence and question:
            q_content_words = set(re.findall(r"\b[a-z0-9_]{3,}\b", question.lower())) - LocalEmbeddingModel.STOP_WORDS
            top_ev = fused_evidence[0]
            ev_words = set(re.findall(r"\b[a-z0-9_]{3,}\b", top_ev.text.lower()))

            if q_content_words and q_content_words.intersection(ev_words):
                sources = [{
                    "title": top_ev.source,
                    "url": top_ev.url,
                    "type": top_ev.source_type
                }]

                formatted_text = NaturalGroundedFormatter.format_natural_answer(
                    question=question,
                    extracted_facts=[],
                    fused_evidence=[top_ev],
                    route=route
                )

                return GroundedSynthesisResult(
                    answer=formatted_text,
                    answer_type=AnswerType.EXTRACTIVE_ANSWER,
                    route=route,
                    sources=sources,
                    fused_evidence=fused_evidence,
                    extracted_facts=[],
                    confidence=min(1.0, float(top_ev.score)),
                    is_fallback_used=True,
                    status="ANSWER",
                    termination_reason="extraction_fallback_raw"
                )

        return None
