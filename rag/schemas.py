from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class SearchItem(BaseModel):
    title: str
    url: str
    snippet: str
    source: Optional[str] = "web"

class SearchResult(BaseModel):
    query: str
    results: List[SearchItem] = Field(default_factory=list)
    total_found: int = 0
    latency_ms: float = 0.0
    error: Optional[str] = None

class SourceItem(BaseModel):
    title: str
    url: str
    snippet: Optional[str] = None

class RAGRequest(BaseModel):
    query: str
    mode: str = Field(default="auto")  # "auto", "on", "off"
    top_k: int = Field(default=3, ge=1, le=10)
    max_tokens: int = Field(default=100)

class RAGTokenBudget(BaseModel):
    total_budget: int = 256
    query_tokens: int = 0
    context_tokens: int = 0
    system_tokens: int = 0
    max_completion_tokens: int = 0
    remaining_tokens: int = 0

class RAGResponse(BaseModel):
    query: str
    mode_used: str  # "auto", "on", "off"
    web_search_used: bool
    context_text: str = ""
    prompt_formatted: str = ""
    generated_answer: Optional[str] = None
    sources: List[SourceItem] = Field(default_factory=list)
    token_budget: Optional[RAGTokenBudget] = None
    search_latency_ms: float = 0.0
    fetch_latency_ms: float = 0.0
    rank_latency_ms: float = 0.0
    total_rag_latency_ms: float = 0.0
    error: Optional[str] = None
