from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from collision.rag.schemas import RAGStatus

class RetrievalMode(str, Enum):
    AUTO = "AUTO"
    LOCAL = "LOCAL"
    WEB = "WEB"
    HYBRID = "HYBRID"

@dataclass
class WebSearchItem:
    title: str
    url: str
    snippet: str
    domain: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class WebSearchResult:
    query: str
    results: List[WebSearchItem] = field(default_factory=list)
    total_found: int = 0
    latency_ms: float = 0.0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "results": [r.to_dict() for r in self.results],
            "total_found": self.total_found,
            "latency_ms": round(self.latency_ms, 2),
            "error": self.error
        }

@dataclass
class WebDocument:
    url: str
    title: str
    domain: str
    text: str
    extracted_paragraphs: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class WebSource:
    title: str
    url: str
    domain: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class WebEvidenceChunk:
    url: str
    title: str
    domain: str
    text: str
    similarity_score: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "url": self.url,
            "title": self.title,
            "domain": self.domain,
            "text": self.text,
            "similarity_score": round(self.similarity_score, 4)
        }

@dataclass
class GroundingResult:
    answer: str
    status: RAGStatus
    sources: List[WebSource] = field(default_factory=list)
    evidence: List[WebEvidenceChunk] = field(default_factory=list)
    retrieval_mode: RetrievalMode = RetrievalMode.AUTO
    confidence: float = 1.0
    latency_ms: float = 0.0
    search_latency_ms: float = 0.0
    fetch_latency_ms: float = 0.0
    generation_latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    termination_reason: str = "eos"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "status": self.status.value if isinstance(self.status, RAGStatus) else str(self.status),
            "sources": [s.to_dict() for s in self.sources],
            "evidence": [e.to_dict() for e in self.evidence],
            "retrieval_mode": self.retrieval_mode.value if isinstance(self.retrieval_mode, RetrievalMode) else str(self.retrieval_mode),
            "confidence": round(self.confidence, 4),
            "latency_ms": round(self.latency_ms, 2),
            "search_latency_ms": round(self.search_latency_ms, 2),
            "fetch_latency_ms": round(self.fetch_latency_ms, 2),
            "generation_latency_ms": round(self.generation_latency_ms, 2),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "termination_reason": self.termination_reason
        }
