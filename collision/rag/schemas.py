from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional

class RAGStatus(str, Enum):
    ANSWER = "ANSWER"
    UNCERTAIN = "UNCERTAIN"
    INSUFFICIENT_INFORMATION = "INSUFFICIENT_INFORMATION"

@dataclass
class DocumentChunk:
    document_id: str
    source: str
    chunk_id: int
    text: str
    token_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class RetrievalItem:
    chunk: DocumentChunk
    similarity_score: float
    source: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk": self.chunk.to_dict(),
            "similarity_score": round(self.similarity_score, 4),
            "source": self.source
        }

@dataclass
class RAGResult:
    answer: str
    status: RAGStatus
    sources: List[str] = field(default_factory=list)
    retrieved_chunks: List[RetrievalItem] = field(default_factory=list)
    confidence: float = 1.0
    latency_ms: float = 0.0
    retrieval_latency_ms: float = 0.0
    generation_latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    repetition_score: float = 0.0
    termination_reason: str = "eos"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "status": self.status.value if isinstance(self.status, RAGStatus) else str(self.status),
            "sources": self.sources,
            "retrieved_chunks": [c.to_dict() for c in self.retrieved_chunks],
            "confidence": round(self.confidence, 4),
            "latency_ms": round(self.latency_ms, 2),
            "retrieval_latency_ms": round(self.retrieval_latency_ms, 2),
            "generation_latency_ms": round(self.generation_latency_ms, 2),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "repetition_score": round(self.repetition_score, 4),
            "termination_reason": self.termination_reason
        }
