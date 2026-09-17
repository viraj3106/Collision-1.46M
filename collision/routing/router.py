import time
from typing import Optional, List
from collision.rag.schemas import RetrievalItem
from collision.rag.retriever import DocumentRetriever
from collision.routing.schemas import RouteMode, RoutingDecision
from collision.routing.classifier import QueryClassifier

class AdaptiveKnowledgeRouter:
    """
    Intelligent query router that evaluates local evidence confidence,
    temporal/web signals, and query characteristics to deterministically
    route queries to MODEL, LOCAL, WEB, HYBRID, or INSUFFICIENT_INFORMATION.
    """

    def __init__(
        self,
        classifier: Optional[QueryClassifier] = None,
        local_relevance_threshold: float = 0.10
    ):
        self.classifier = classifier if classifier is not None else QueryClassifier()
        self.local_relevance_threshold = local_relevance_threshold

    def route(
        self,
        query: str,
        local_retriever: Optional[DocumentRetriever] = None,
        explicit_mode: RouteMode = RouteMode.AUTO
    ) -> RoutingDecision:
        """
        Computes the optimal routing decision for a given question.
        """
        if not query or not str(query).strip():
            return RoutingDecision(
                mode=RouteMode.INSUFFICIENT_INFORMATION,
                reason="Empty query",
                confidence=0.0
            )

        clean_q = str(query).strip()

        # 1. Explicit override
        if explicit_mode != RouteMode.AUTO:
            return RoutingDecision(
                mode=explicit_mode,
                reason=f"Explicit override to {explicit_mode.value}",
                confidence=1.0
            )

        # 2. Check unanswerable / private queries
        if self.classifier.is_unanswerable_private(clean_q):
            return RoutingDecision(
                mode=RouteMode.INSUFFICIENT_INFORMATION,
                reason="Query demands private credentials or unverifiable future events",
                confidence=1.0
            )

        # 3. Check hybrid candidate (e.g. comparison between local project and external literature)
        if self.classifier.is_hybrid_candidate(clean_q):
            return RoutingDecision(
                mode=RouteMode.HYBRID,
                reason="Comparative query requires both local repository knowledge and external literature",
                confidence=0.90,
                requires_current_web=True
            )

        # 4. Probe local retriever if available
        local_score = 0.0
        local_items: List[RetrievalItem] = []
        if local_retriever is not None:
            local_items = local_retriever.retrieve(
                query=clean_q,
                top_k=3,
                relevance_threshold=self.local_relevance_threshold
            )
            if local_items:
                local_score = max(item.similarity_score for item in local_items)

        requires_web = self.classifier.requires_current_web(clean_q)
        is_model = self.classifier.is_model_suitable(clean_q)

        # 5. Routing logic
        # A. Strong local match without temporal external requirement -> LOCAL
        if local_score >= self.local_relevance_threshold and not requires_web:
            return RoutingDecision(
                mode=RouteMode.LOCAL,
                reason=f"Local documents contain strong supporting evidence (similarity: {local_score:.3f})",
                confidence=min(1.0, 0.5 + local_score),
                local_score=local_score,
                web_score=0.0
            )

        # B. Local match WITH temporal requirement -> HYBRID
        if local_score >= self.local_relevance_threshold and requires_web:
            return RoutingDecision(
                mode=RouteMode.HYBRID,
                reason="Local context exists but external/temporal updates are also required",
                confidence=0.85,
                local_score=local_score,
                web_score=0.85,
                requires_current_web=True
            )

        # C. Conversational or logic/math without local document matches -> MODEL
        if is_model and local_score < self.local_relevance_threshold:
            return RoutingDecision(
                mode=RouteMode.MODEL,
                reason="Conversational, mathematical, or generic reasoning query suitable for direct model answer",
                confidence=0.90,
                local_score=local_score,
                is_conversational_or_reasoning=True
            )

        # D. Temporal or external information required -> WEB
        if requires_web or local_score < self.local_relevance_threshold:
            return RoutingDecision(
                mode=RouteMode.WEB,
                reason="External or current web information required",
                confidence=0.85,
                local_score=local_score,
                web_score=0.85,
                requires_current_web=requires_web
            )

        # Default fallback
        return RoutingDecision(
            mode=RouteMode.MODEL,
            reason="General knowledge query routed to model",
            confidence=0.70
        )
