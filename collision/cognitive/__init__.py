"""
COLLISION Cognitive Intelligence Subsystem.
"""

from collision.cognitive.engine import CollisionCognitiveEngine, CognitiveResponse
from collision.cognitive.reasoning import CognitiveReasoningEngine, CognitiveReasoningResult
from collision.cognitive.strategy import CognitiveStrategyEngine, StrategicAnalysisResult
from collision.cognitive.dialogue import CognitiveDialogueEngine, DialogueTurnResult
from collision.cognitive.knowledge import CrossDomainKnowledgeBase, DomainKnowledgeResult

__all__ = [
    "CollisionCognitiveEngine",
    "CognitiveResponse",
    "CognitiveReasoningEngine",
    "CognitiveReasoningResult",
    "CognitiveStrategyEngine",
    "StrategicAnalysisResult",
    "CognitiveDialogueEngine",
    "DialogueTurnResult",
    "CrossDomainKnowledgeBase",
    "DomainKnowledgeResult"
]
