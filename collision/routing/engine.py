import time
from typing import List, Optional, Dict, Any
from collision.rag.schemas import RetrievalItem
from collision.rag.retriever import DocumentRetriever
from collision.answering.engine import CollisionAnsweringEngine
from collision.answering.schemas import AnswerStatus, AnswerResult
from collision.web.schemas import WebEvidenceChunk, WebDocument
from collision.web.search import WebSearchProvider, MockWebSearchProvider
from collision.web.fetch import safe_fetch_page
from collision.web.extractor import WebPageExtractor
from collision.web.ranker import WebEvidenceRanker
from collision.routing.schemas import (
    RouteMode,
    RoutingDecision,
    FusedEvidence,
    VerificationResult,
    VerifiedAnswerResult
)
from collision.routing.router import AdaptiveKnowledgeRouter
from collision.routing.fusion import EvidenceFusion
from collision.routing.verifier import GroundingVerifier
from collision.routing.confidence import ConfidenceScorer

class AdaptiveKnowledgeEngine:
    """
    Unified Knowledge Routing & Evidence Verification Engine for COLLISION (Phase 96).
    Orchestrates routing, multi-source evidence fusion, generation, and claim-level verification.
    """

    def __init__(
        self,
        local_retriever: Optional[DocumentRetriever] = None,
        search_provider: Optional[WebSearchProvider] = None,
        answering_engine: Optional[CollisionAnsweringEngine] = None,
        router: Optional[AdaptiveKnowledgeRouter] = None,
        fusion: Optional[EvidenceFusion] = None,
        verifier: Optional[GroundingVerifier] = None,
        extractor: Optional[WebPageExtractor] = None,
        ranker: Optional[WebEvidenceRanker] = None
    ):
        self.local_retriever = local_retriever
        self.search_provider = search_provider if search_provider is not None else MockWebSearchProvider()
        self.answering_engine = answering_engine if answering_engine is not None else CollisionAnsweringEngine()
        self.router = router if router is not None else AdaptiveKnowledgeRouter()
        self.fusion = fusion if fusion is not None else EvidenceFusion()
        self.verifier = verifier if verifier is not None else GroundingVerifier()
        self.extractor = extractor if extractor is not None else WebPageExtractor()
        self.ranker = ranker if ranker is not None else WebEvidenceRanker()

    def _format_grounded_prompt(
        self,
        question: str,
        fused_evidence: List[FusedEvidence],
        max_completion_tokens: int = 80
    ) -> str:
        header = "Answer using ONLY the provided evidence. If insufficient, say so. Do not invent facts.\n\nEvidence:\n"
        ev_lines = []
        for idx, ev in enumerate(fused_evidence, 1):
            src_label = f"{ev.source_type}: {ev.source}"
            ev_lines.append(f"[{idx}] ({src_label}) {ev.text.strip()}")

        context_block = "\n".join(ev_lines)
        footer = f"\n\nQuestion:\n{question}\n\nAnswer:"
        full_prompt = header + context_block + footer

        encoded = self.answering_engine.tokenizer.encode(full_prompt, bos=True)
        max_prompt_len = self.answering_engine.model_cfg.max_seq_len - max_completion_tokens

        if len(encoded) > max_prompt_len and len(fused_evidence) > 1:
            return self._format_grounded_prompt(question, fused_evidence[:-1], max_completion_tokens)

        return full_prompt

    def _fetch_and_rank_web(self, query: str, top_k: int = 3) -> List[WebEvidenceChunk]:
        search_res = self.search_provider.search(query, max_results=top_k)
        if search_res.error or not search_res.results:
            return []

        web_docs: List[WebDocument] = []
        for item in search_res.results:
            html_text = None
            if not item.url.startswith("https://example.com") and not item.url.startswith("mock://"):
                html_text, _ = safe_fetch_page(item.url, timeout_sec=2.5)

            if html_text:
                doc = self.extractor.extract(html_text, url=item.url, title=item.title)
            else:
                doc = self.extractor.extract(item.snippet, url=item.url, title=item.title)
            web_docs.append(doc)

        return self.ranker.rank_evidence(query, web_docs, top_k=top_k)

    def answer(
        self,
        question: str,
        mode: RouteMode = RouteMode.AUTO,
        top_k: int = 3,
        max_tokens: int = 80,
        temperature: float = 0.7,
        deterministic: bool = False
    ) -> VerifiedAnswerResult:
        """
        End-to-end knowledge routing, retrieval, generation, and grounding verification.
        """
        t_total_start = time.perf_counter()

        # Step 1: Query Routing
        t_route_start = time.perf_counter()
        routing_dec: RoutingDecision = self.router.route(
            query=question,
            local_retriever=self.local_retriever,
            explicit_mode=mode
        )
        routing_lat_ms = (time.perf_counter() - t_route_start) * 1000.0

        # Step 2: Handle Insufficient / Unanswerable queries
        if routing_dec.mode == RouteMode.INSUFFICIENT_INFORMATION:
            tot_lat_ms = (time.perf_counter() - t_total_start) * 1000.0
            return VerifiedAnswerResult(
                answer="I do not have sufficient reliable information to answer this question accurately.",
                status="INSUFFICIENT_INFORMATION",
                route=RouteMode.INSUFFICIENT_INFORMATION,
                sources=[],
                fused_evidence=[],
                verification=VerificationResult(supported=False, status="INSUFFICIENT_INFORMATION", score=0.0, confidence=0.0),
                confidence=0.0,
                latency_ms=tot_lat_ms,
                routing_latency_ms=routing_lat_ms,
                prompt_tokens=len(self.answering_engine.tokenizer.encode(question, bos=True)),
                completion_tokens=15,
                total_tokens=len(self.answering_engine.tokenizer.encode(question, bos=True)) + 15,
                termination_reason="insufficient_information_route"
            )

        # Step 3: Handle MODEL-Only Mode
        if routing_dec.mode == RouteMode.MODEL:
            t_gen_start = time.perf_counter()
            gen_res: AnswerResult = self.answering_engine.answer(
                question=question,
                max_tokens=max_tokens,
                temperature=temperature,
                deterministic=deterministic
            )
            gen_lat_ms = (time.perf_counter() - t_gen_start) * 1000.0
            tot_lat_ms = (time.perf_counter() - t_total_start) * 1000.0

            conf = ConfidenceScorer.calculate_confidence(
                verification=None,
                fused_evidence=[],
                repetition_score=gen_res.repetition_score,
                is_model_only=True
            )

            return VerifiedAnswerResult(
                answer=gen_res.text,
                status="ANSWER" if gen_res.status == AnswerStatus.ANSWER else "UNCERTAIN",
                route=RouteMode.MODEL,
                sources=[],
                fused_evidence=[],
                verification=None,
                confidence=conf,
                latency_ms=tot_lat_ms,
                routing_latency_ms=routing_lat_ms,
                retrieval_latency_ms=0.0,
                generation_latency_ms=gen_lat_ms,
                verification_latency_ms=0.0,
                prompt_tokens=gen_res.prompt_tokens,
                completion_tokens=gen_res.completion_tokens,
                total_tokens=gen_res.total_tokens,
                termination_reason=gen_res.termination_reason
            )

        # Step 4: Multi-Source Retrieval Phase
        t_ret_start = time.perf_counter()
        local_items: List[RetrievalItem] = []
        web_items: List[WebEvidenceChunk] = []

        if routing_dec.mode in (RouteMode.LOCAL, RouteMode.HYBRID) and self.local_retriever:
            local_items = self.local_retriever.retrieve(question, top_k=top_k)

        if routing_dec.mode in (RouteMode.WEB, RouteMode.HYBRID):
            web_items = self._fetch_and_rank_web(question, top_k=top_k)

        ret_lat_ms = (time.perf_counter() - t_ret_start) * 1000.0

        # Step 5: Evidence Fusion
        fused_evidence: List[FusedEvidence] = self.fusion.fuse(local_items=local_items, web_items=web_items, max_fused_chunks=top_k)

        if not fused_evidence:
            tot_lat_ms = (time.perf_counter() - t_total_start) * 1000.0
            return VerifiedAnswerResult(
                answer="I do not have sufficient reliable information in the retrieved evidence to answer this question.",
                status="INSUFFICIENT_INFORMATION",
                route=routing_dec.mode,
                sources=[],
                fused_evidence=[],
                verification=VerificationResult(supported=False, status="INSUFFICIENT_INFORMATION", score=0.0, confidence=0.0),
                confidence=0.0,
                latency_ms=tot_lat_ms,
                routing_latency_ms=routing_lat_ms,
                retrieval_latency_ms=ret_lat_ms,
                prompt_tokens=len(self.answering_engine.tokenizer.encode(question, bos=True)),
                completion_tokens=18,
                total_tokens=len(self.answering_engine.tokenizer.encode(question, bos=True)) + 18,
                termination_reason="empty_fused_evidence"
            )

        # Step 6: Grounded Generation Phase
        grounded_prompt = self._format_grounded_prompt(question, fused_evidence, max_completion_tokens=max_tokens)
        t_gen_start = time.perf_counter()
        gen_res: AnswerResult = self.answering_engine.answer(
            question=grounded_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            deterministic=deterministic
        )
        gen_lat_ms = (time.perf_counter() - t_gen_start) * 1000.0

        # Step 7: Grounding Verification Phase
        t_ver_start = time.perf_counter()
        ver_res: VerificationResult = self.verifier.verify(gen_res.text, fused_evidence)
        ver_lat_ms = (time.perf_counter() - t_ver_start) * 1000.0
        tot_lat_ms = (time.perf_counter() - t_total_start) * 1000.0

        conf = ConfidenceScorer.calculate_confidence(
            verification=ver_res,
            fused_evidence=fused_evidence,
            repetition_score=gen_res.repetition_score
        )

        # Extract verified sources used
        sources = []
        seen_src = set()
        for ev in fused_evidence:
            if ev.url not in seen_src:
                seen_src.add(ev.url)
                sources.append({
                    "title": ev.source,
                    "url": ev.url,
                    "type": ev.source_type
                })

        return VerifiedAnswerResult(
            answer=gen_res.text,
            status=ver_res.status,
            route=routing_dec.mode,
            sources=sources,
            fused_evidence=fused_evidence,
            verification=ver_res,
            confidence=conf,
            latency_ms=tot_lat_ms,
            routing_latency_ms=routing_lat_ms,
            retrieval_latency_ms=ret_lat_ms,
            generation_latency_ms=gen_lat_ms,
            verification_latency_ms=ver_lat_ms,
            prompt_tokens=gen_res.prompt_tokens,
            completion_tokens=gen_res.completion_tokens,
            total_tokens=gen_res.total_tokens,
            termination_reason=gen_res.termination_reason
        )
