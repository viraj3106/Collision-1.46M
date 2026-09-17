"""
COLLISION Phase 98 — Grounded Synthesis Schemas.

Defines answer types, extracted facts, synthesis results, and provenance metadata.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from collision.routing.schemas import RouteMode, FusedEvidence, VerificationResult


class AnswerType(str, Enum):
    GROUNDED_ANSWER = "GROUNDED_ANSWER"
    EXTRACTIVE_ANSWER = "EXTRACTIVE_ANSWER"
    MODEL_SYNTHESIZED_GROUNDED_ANSWER = "MODEL_SYNTHESIZED_GROUNDED_ANSWER"
    INSUFFICIENT_INFORMATION = "INSUFFICIENT_INFORMATION"
    CONFLICTING_EVIDENCE = "CONFLICTING_EVIDENCE"
    MODEL_ONLY = "MODEL_ONLY"


@dataclass
class ExtractedFact:
    fact_text: str
    fact_type: str  # "date", "number", "entity", "specification", "definition", "span"
    source: str
    url: str
    document_id: str
    chunk_id: int
    original_chunk_text: str
    relevance_score: float = 0.0
    extraction_confidence: float = 1.0
    claim_linkage: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fact_text": self.fact_text,
            "fact_type": self.fact_type,
            "source": self.source,
            "url": self.url,
            "document_id": self.document_id,
            "chunk_id": self.chunk_id,
            "relevance_score": round(self.relevance_score, 4),
            "extraction_confidence": round(self.extraction_confidence, 4),
            "claim_linkage": self.claim_linkage
        }


@dataclass
class GroundedSynthesisResult:
    answer: str
    answer_type: AnswerType
    route: RouteMode
    sources: List[Dict[str, Any]] = field(default_factory=list)
    fused_evidence: List[FusedEvidence] = field(default_factory=list)
    extracted_facts: List[ExtractedFact] = field(default_factory=list)
    verification: Optional[VerificationResult] = None
    confidence: float = 1.0
    is_fallback_used: bool = False
    status: str = "ANSWER"
    latency_ms: float = 0.0
    routing_latency_ms: float = 0.0
    retrieval_latency_ms: float = 0.0
    synthesis_latency_ms: float = 0.0
    verification_latency_ms: float = 0.0
    fallback_latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    termination_reason: str = "eos"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "answer_type": self.answer_type.value if isinstance(self.answer_type, AnswerType) else str(self.answer_type),
            "status": self.status,
            "route": self.route.value if isinstance(self.route, RouteMode) else str(self.route),
            "sources": self.sources,
            "fused_evidence": [e.to_dict() for e in self.fused_evidence],
            "extracted_facts": [f.to_dict() for f in self.extracted_facts],
            "verification": self.verification.to_dict() if self.verification else None,
            "confidence": round(self.confidence, 4),
            "is_fallback_used": self.is_fallback_used,
            "latency_ms": round(self.latency_ms, 2),
            "routing_latency_ms": round(self.routing_latency_ms, 2),
            "retrieval_latency_ms": round(self.retrieval_latency_ms, 2),
            "synthesis_latency_ms": round(self.synthesis_latency_ms, 2),
            "verification_latency_ms": round(self.verification_latency_ms, 2),
            "fallback_latency_ms": round(self.fallback_latency_ms, 2),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "termination_reason": self.termination_reason
        }
