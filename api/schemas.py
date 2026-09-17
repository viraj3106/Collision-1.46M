from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum


class StatusEnum(str, Enum):
    ANSWERED = "ANSWERED"
    INSUFFICIENT_INFORMATION = "INSUFFICIENT_INFORMATION"
    CONFLICT = "CONFLICT"
    ERROR = "ERROR"


class ModeEnum(str, Enum):
    MODEL = "MODEL"
    LOCAL = "LOCAL"
    WEB = "WEB"
    HYBRID = "HYBRID"
    INSUFFICIENT_INFORMATION = "INSUFFICIENT_INFORMATION"
    AUTO = "AUTO"


class ClaimSupportStatus(str, Enum):
    SUPPORTED = "SUPPORTED"
    UNCERTAIN = "UNCERTAIN"
    UNSUPPORTED = "UNSUPPORTED"
    CONTRADICTED = "CONTRADICTED"


# --- Phase 99 Production Ask Request / Response Schemas ---

class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000, description="User query to answer")
    mode: Optional[str] = Field(default="AUTO", description="Routing mode: AUTO, LOCAL, WEB, HYBRID, MODEL")
    include_sources: Optional[bool] = Field(default=True, description="Whether to include source provenance")
    include_claims: Optional[bool] = Field(default=True, description="Whether to include claim verification status")


class SourceProvenance(BaseModel):
    source_id: str
    title: str
    url: str
    source_type: str = "LOCAL"
    retrieval_score: float = 0.0
    relevance: float = 0.0
    snippet: Optional[str] = None


class ClaimItem(BaseModel):
    text: str
    support_status: str
    evidence_ids: List[str] = Field(default_factory=list)


class LatencyBreakdown(BaseModel):
    routing_ms: float = 0.0
    retrieval_ms: float = 0.0
    generation_ms: float = 0.0
    verification_ms: float = 0.0
    total_ms: float = 0.0


class AskResponse(BaseModel):
    answer: str
    status: str
    mode: str
    confidence: float
    sources: List[SourceProvenance] = Field(default_factory=list)
    claims: List[ClaimItem] = Field(default_factory=list)
    latency: LatencyBreakdown
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ReadyChecks(BaseModel):
    database: Optional[str] = "ok"
    model: Optional[str] = "ok"
    model_checkpoint: bool
    tokenizer: bool
    local_index: bool
    web_provider: bool


class ReadyResponse(BaseModel):
    status: str
    service: str = "collision"
    checks: ReadyChecks


class ErrorDetail(BaseModel):
    code: str
    message: str


class StructuredErrorResponse(BaseModel):
    status: str = "error"
    error: ErrorDetail
    latency: Optional[Dict[str, float]] = None


# --- Legacy Schemas (Preserved for compatibility) ---

class GenerateRequest(BaseModel):
    model: str = Field(default="collision-10m")
    prompt: str = Field(..., min_length=1)
    max_tokens: int = Field(default=100, gt=0, le=256)
    temperature: float = Field(default=0.7, gt=0.0)
    top_k: int = Field(default=50, ge=0)
    top_p: float = Field(default=0.9, gt=0.0, le=1.0)
    web_search: Optional[str] = Field(default="auto")

class UsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class PerformanceInfo(BaseModel):
    latency_ms: float
    tokens_per_second: float

class SourceInfo(BaseModel):
    title: str
    url: str
    snippet: Optional[str] = None

class GenerateResponse(BaseModel):
    id: str
    object: str = "text_completion"
    model: str
    text: str
    web_search_used: bool = False
    sources: List[SourceInfo] = Field(default_factory=list)
    usage: UsageInfo
    performance: PerformanceInfo

class HealthResponse(BaseModel):
    status: str
    service: Optional[str] = "collision"
    version: Optional[str] = "1.0"
    model: Optional[str] = "collision-10m"
    device: Optional[str] = "cpu"

class ModelInfo(BaseModel):
    id: str
    object: str = "model"

class ModelListResponse(BaseModel):
    data: List[ModelInfo]

class FeedbackRequest(BaseModel):
    user_id: Optional[str] = Field(default="anonymous")
    prompt: str = Field(..., min_length=1)
    model: str = Field(default="collision-10m")
    response: str = Field(..., min_length=1)
    rating: str = Field(...)
    feedback: Optional[str] = Field(default="")
    category: Optional[str] = Field(default="general")
    consent: bool = Field(default=True)

class FeedbackResponse(BaseModel):
    id: int
    status: str = "success"
    message: str = "Feedback recorded successfully."
