from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional

class RouteMode(str, Enum):
    AUTO = "AUTO"
    MODEL = "MODEL"
    LOCAL = "LOCAL"
    WEB = "WEB"
    HYBRID = "HYBRID"
    INSUFFICIENT_INFORMATION = "INSUFFICIENT_INFORMATION"

@dataclass
class RoutingDecision:
    mode: RouteMode
    reason: str
    confidence: float
    local_score: float = 0.0
    web_score: float = 0.0
    is_conversational_or_reasoning: bool = False
    requires_current_web: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode.value if isinstance(self.mode, RouteMode) else str(self.mode),
            "reason": self.reason,
            "confidence": round(self.confidence, 4),
            "local_score": round(self.local_score, 4),
            "web_score": round(self.web_score, 4),
            "is_conversational_or_reasoning": self.is_conversational_or_reasoning,
            "requires_current_web": self.requires_current_web
        }

@dataclass
class FusedEvidence:
    source: str
    url: str
    document_id: str
    chunk_id: int
    text: str
    score: float
    source_type: str # "LOCAL" or "WEB"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "url": self.url,
            "document_id": self.document_id,
            "chunk_id": self.chunk_id,
            "text": self.text,
            "score": round(self.score, 4),
            "source_type": self.source_type
        }

@dataclass
class ClaimVerification:
    claim_text: str
    status: str # "SUPPORTED", "UNSUPPORTED", "CONTRADICTED", "UNCERTAIN"
    supporting_evidence: List[str] = field(default_factory=list)
    similarity_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim_text": self.claim_text,
            "status": self.status,
            "supporting_evidence": self.supporting_evidence,
            "similarity_score": round(self.similarity_score, 4)
        }

@dataclass
class VerificationResult:
    supported: bool
    status: str # "ANSWER", "UNCERTAIN", "INSUFFICIENT_INFORMATION", "CONFLICTING_EVIDENCE"
    score: float
    claims: List[ClaimVerification] = field(default_factory=list)
    unsupported_claims: List[str] = field(default_factory=list)
    contradicted_claims: List[str] = field(default_factory=list)
    evidence_used: List[str] = field(default_factory=list)
    has_conflicting_evidence: bool = False
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "supported": self.supported,
            "status": self.status,
            "score": round(self.score, 4),
            "claims": [c.to_dict() for c in self.claims],
            "unsupported_claims": self.unsupported_claims,
            "contradicted_claims": self.contradicted_claims,
            "evidence_used": self.evidence_used,
            "has_conflicting_evidence": self.has_conflicting_evidence,
            "confidence": round(self.confidence, 4)
        }

@dataclass
class VerifiedAnswerResult:
    answer: str
    status: str # "ANSWER", "UNCERTAIN", "INSUFFICIENT_INFORMATION", "CONFLICTING_EVIDENCE"
    route: RouteMode
    sources: List[Dict[str, Any]] = field(default_factory=list)
    fused_evidence: List[FusedEvidence] = field(default_factory=list)
    verification: Optional[VerificationResult] = None
    confidence: float = 1.0
    latency_ms: float = 0.0
    routing_latency_ms: float = 0.0
    retrieval_latency_ms: float = 0.0
    generation_latency_ms: float = 0.0
    verification_latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    termination_reason: str = "eos"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "status": self.status,
            "route": self.route.value if isinstance(self.route, RouteMode) else str(self.route),
            "sources": self.sources,
            "fused_evidence": [e.to_dict() for e in self.fused_evidence],
            "verification": self.verification.to_dict() if self.verification else None,
            "confidence": round(self.confidence, 4),
            "latency_ms": round(self.latency_ms, 2),
            "routing_latency_ms": round(self.routing_latency_ms, 2),
            "retrieval_latency_ms": round(self.retrieval_latency_ms, 2),
            "generation_latency_ms": round(self.generation_latency_ms, 2),
            "verification_latency_ms": round(self.verification_latency_ms, 2),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "termination_reason": self.termination_reason
        }
