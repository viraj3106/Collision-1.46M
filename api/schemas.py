from pydantic import BaseModel, Field
from typing import List, Optional

class GenerateRequest(BaseModel):
    model: str = Field(default="collision-10m")
    prompt: str = Field(..., min_length=1)
    max_tokens: int = Field(default=100, gt=0, le=256)
    temperature: float = Field(default=0.7, gt=0.0)
    top_k: int = Field(default=50, ge=0)
    top_p: float = Field(default=0.9, gt=0.0, le=1.0)

class UsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class PerformanceInfo(BaseModel):
    latency_ms: float
    tokens_per_second: float

class GenerateResponse(BaseModel):
    id: str
    object: str = "text_completion"
    model: str
    text: str
    usage: UsageInfo
    performance: PerformanceInfo

class HealthResponse(BaseModel):
    status: str
    model: str
    device: str

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
    rating: str = Field(..., example="thumbs_up")
    feedback: Optional[str] = Field(default="")
    category: Optional[str] = Field(default="general")
    consent: bool = Field(default=True)

class FeedbackResponse(BaseModel):
    id: int
    status: str = "success"
    message: str = "Feedback recorded successfully."

