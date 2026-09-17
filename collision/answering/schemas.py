from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional

class AnswerStatus(str, Enum):
    ANSWER = "ANSWER"
    UNCERTAIN = "UNCERTAIN"
    INSUFFICIENT_INFORMATION = "INSUFFICIENT_INFORMATION"

@dataclass
class ConversationMessage:
    role: str # "user", "assistant", "system"
    content: str

    def to_dict(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}

@dataclass
class AnswerResult:
    text: str
    status: AnswerStatus
    confidence_score: float = 1.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    tokens_per_second: float = 0.0
    repetition_score: float = 0.0
    unique_token_ratio: float = 1.0
    termination_reason: str = "eos"
    model_name: str = "collision-10m"
    raw_tokens: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value if isinstance(self.status, AnswerStatus) else str(self.status)
        return d
