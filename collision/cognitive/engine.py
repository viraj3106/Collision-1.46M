"""
COLLISION Cognitive Intelligence Engine.

Unified entry point integrating:
1. Cognitive Dialogue & Empathy
2. Chain-of-Thought Reasoning & Root Cause Analysis
3. Strategic Persuasion, Negotiation & Fallacy Deconstruction
4. Universal Cross-Domain Knowledge Base
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

from collision.cognitive.reasoning import CognitiveReasoningEngine, CognitiveReasoningResult
from collision.cognitive.strategy import CognitiveStrategyEngine, StrategicAnalysisResult
from collision.cognitive.dialogue import CognitiveDialogueEngine, DialogueTurnResult
from collision.cognitive.knowledge import CrossDomainKnowledgeBase, DomainKnowledgeResult


class CognitiveResponse(BaseModel):
    query: str
    cognitive_mode: str
    answer: str
    confidence: float = 1.0
    domain: Optional[str] = None
    followups: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CollisionCognitiveEngine:
    """
    Unified coordinator for advanced cognitive reasoning, dialogue, persuasion analysis,
    and cross-domain expert knowledge.
    """

    @classmethod
    def evaluate(cls, query: str) -> Optional[CognitiveResponse]:
        """
        Evaluates query across cognitive reasoning, strategic persuasion, dialogue,
        and cross-domain knowledge modules.
        """
        if not query or not query.strip():
            return None

        q = query.strip()

        # 1. Check Strategic Persuasion, Negotiation, Biases & Fallacies
        strat_res = CognitiveStrategyEngine.analyze_query(q)
        if strat_res is not None:
            return CognitiveResponse(
                query=q,
                cognitive_mode="STRATEGY_PERSUASION",
                answer=strat_res.formatted_output,
                confidence=1.0,
                domain="Strategy & Psychology",
                metadata={"analysis_type": strat_res.analysis_type}
            )

        # 2. Check Chain-of-Thought Reasoning, 5-Whys, Fermi Estimation & Problem Solving
        reason_res = CognitiveReasoningEngine.analyze_query(q)
        if reason_res is not None:
            return CognitiveResponse(
                query=q,
                cognitive_mode="CHAIN_OF_THOUGHT_REASONING",
                answer=reason_res.formatted_output,
                confidence=1.0,
                domain="Cognitive Reasoning",
                metadata={"reasoning_type": reason_res.reasoning_type}
            )

        # 3. Check Deep Cross-Domain Knowledge Base (AI, Physics, CS, Bio, Philosophy, Econ)
        know_res = CrossDomainKnowledgeBase.query_knowledge(q)
        if know_res is not None:
            return CognitiveResponse(
                query=q,
                cognitive_mode="CROSS_DOMAIN_KNOWLEDGE",
                answer=know_res.formatted_output,
                confidence=1.0,
                domain=know_res.domain,
                metadata={"concept_title": know_res.concept_title}
            )

        # 4. Check Open-Ended Dialogue, Empathy, Philosophy, Humor & Advice
        dialogue_res = CognitiveDialogueEngine.analyze_dialogue(q)
        if dialogue_res is not None:
            return CognitiveResponse(
                query=q,
                cognitive_mode="CONVERSATIONAL_DIALOGUE",
                answer=dialogue_res.response,
                confidence=1.0,
                domain="Dialogue & Empathy",
                followups=dialogue_res.suggested_followups,
                metadata={"intent_category": dialogue_res.intent_category, "tone": dialogue_res.tone}
            )

        return None
