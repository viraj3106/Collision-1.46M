"""
COLLISION Phase 98 — Grounded Synthesis Engine.

The complete unified master engine coordinating routing, retrieval, evidence fusion,
extraction-first synthesis, model generation with verification gating, and guaranteed
extractive fallback.
"""

import re
import time
from typing import List, Optional, Dict, Any
from collision.rag.schemas import RetrievalItem
from collision.rag.retriever import DocumentRetriever
from collision.answering.engine import CollisionAnsweringEngine
from collision.answering.schemas import AnswerStatus, AnswerResult
from collision.web.schemas import WebEvidenceChunk, WebDocument
from collision.web.search import WebSearchProvider, MockWebSearchProvider
from collision.web.extractor import WebPageExtractor
from collision.web.ranker import WebEvidenceRanker
from collision.routing.schemas import (
    RouteMode,
    RoutingDecision,
    FusedEvidence,
    VerificationResult
)
from collision.routing.router import AdaptiveKnowledgeRouter
from collision.routing.fusion import EvidenceFusion
from collision.routing.verifier import GroundingVerifier
from collision.grounding.schemas import AnswerType, ExtractedFact, GroundedSynthesisResult
from collision.grounding.extractor import EvidenceExtractor
from collision.grounding.synthesizer import GroundedSynthesizer
from collision.grounding.fallback import ExtractionFallbackHandler
from collision.grounding.policy import FinalAnswerPolicy
from collision.grounding.formatter import NaturalGroundedFormatter
from collision.rag.embeddings import LocalEmbeddingModel
from collision.rag.reranker import HybridReranker
from collision.rag.compressor import EvidenceCompressor


class GroundedSynthesisEngine:
    """
    Unified Grounded Synthesis & Evidence-Verified Answering Engine (Phase 98 & Phase 104).
    """

    def __init__(
        self,
        local_retriever: Optional[DocumentRetriever] = None,
        search_provider: Optional[WebSearchProvider] = None,
        answering_engine: Optional[CollisionAnsweringEngine] = None,
        router: Optional[AdaptiveKnowledgeRouter] = None,
        fusion: Optional[EvidenceFusion] = None,
        verifier: Optional[GroundingVerifier] = None,
        extractor: Optional[EvidenceExtractor] = None,
        synthesizer: Optional[GroundedSynthesizer] = None,
        fallback_handler: Optional[ExtractionFallbackHandler] = None,
        reranker: Optional[HybridReranker] = None,
        compressor: Optional[EvidenceCompressor] = None
    ):
        self.local_retriever = local_retriever
        self.search_provider = search_provider if search_provider is not None else MockWebSearchProvider()
        self.answering_engine = answering_engine if answering_engine is not None else CollisionAnsweringEngine()
        self.router = router if router is not None else AdaptiveKnowledgeRouter()
        self.fusion = fusion if fusion is not None else EvidenceFusion()
        self.verifier = verifier if verifier is not None else GroundingVerifier()
        self.extractor = extractor if extractor is not None else EvidenceExtractor()
        self.fallback_handler = fallback_handler if fallback_handler is not None else ExtractionFallbackHandler()
        self.synthesizer = synthesizer if synthesizer is not None else GroundedSynthesizer(
            extractor=self.extractor,
            fallback_handler=self.fallback_handler,
            verifier=self.verifier,
            answering_engine=self.answering_engine
        )
        self.reranker = reranker if reranker is not None else HybridReranker()
        self.compressor = compressor if compressor is not None else EvidenceCompressor(reranker=self.reranker)
        self.web_extractor = WebPageExtractor()
        self.web_ranker = WebEvidenceRanker()


    def _fetch_and_rank_web(self, query: str, top_k: int = 3) -> List[WebEvidenceChunk]:
        search_res = self.search_provider.search(query=query, max_results=top_k)
        if not search_res.results:
            return []

        web_docs: List[WebDocument] = []
        for item in search_res.results:
            doc = self.web_extractor.extract(item.snippet, url=item.url, title=item.title)
            web_docs.append(doc)

        return self.web_ranker.rank_evidence(query=query, documents=web_docs, top_k=top_k)

    def answer(
        self,
        question: str,
        mode: RouteMode = RouteMode.AUTO,
        top_k: int = 3,
        temperature: float = 0.2,
        deterministic: bool = True
    ) -> GroundedSynthesisResult:
        """
        Executes the complete grounded answer synthesis pipeline.
        """
        t_total_start = time.perf_counter()

        if not question or not str(question).strip():
            return GroundedSynthesisResult(
                answer="Question is empty. Please provide a valid query.",
                answer_type=AnswerType.INSUFFICIENT_INFORMATION,
                route=RouteMode.INSUFFICIENT_INFORMATION,
                status="INSUFFICIENT_INFORMATION",
                confidence=0.0,
                latency_ms=(time.perf_counter() - t_total_start) * 1000.0,
                termination_reason="empty_query"
            )

        # Step 1: Routing
        t_route_start = time.perf_counter()
        routing_dec: RoutingDecision
        if mode == RouteMode.AUTO:
            routing_dec = self.router.route(question, local_retriever=self.local_retriever)
        else:
            routing_dec = RoutingDecision(mode=mode, reason=f"Explicit mode: {mode}", confidence=1.0)
        routing_lat_ms = (time.perf_counter() - t_route_start) * 1000.0

        # Step 2: Handle Insufficient Information Route
        if routing_dec.mode == RouteMode.INSUFFICIENT_INFORMATION:
            tot_lat_ms = (time.perf_counter() - t_total_start) * 1000.0
            return GroundedSynthesisResult(
                answer="I do not have sufficient reliable information to answer this question accurately.",
                answer_type=AnswerType.INSUFFICIENT_INFORMATION,
                route=RouteMode.INSUFFICIENT_INFORMATION,
                status="INSUFFICIENT_INFORMATION",
                confidence=0.0,
                latency_ms=tot_lat_ms,
                routing_latency_ms=routing_lat_ms,
                termination_reason="insufficient_information_route"
            )

        # Step 3: Handle Model-Only Route
        if routing_dec.mode == RouteMode.MODEL:
            t_gen_start = time.perf_counter()
            gen_res: AnswerResult = self.answering_engine.answer(
                question=question,
                temperature=temperature,
                deterministic=deterministic
            )
            gen_lat_ms = (time.perf_counter() - t_gen_start) * 1000.0
            tot_lat_ms = (time.perf_counter() - t_total_start) * 1000.0

            return GroundedSynthesisResult(
                answer=gen_res.text,
                answer_type=AnswerType.MODEL_ONLY,
                route=RouteMode.MODEL,
                status="ANSWER" if gen_res.status == AnswerStatus.ANSWER else "UNCERTAIN",
                confidence=0.85,
                latency_ms=tot_lat_ms,
                routing_latency_ms=routing_lat_ms,
                synthesis_latency_ms=gen_lat_ms,
                prompt_tokens=gen_res.prompt_tokens,
                completion_tokens=gen_res.completion_tokens,
                total_tokens=gen_res.total_tokens,
                termination_reason=gen_res.termination_reason
            )

        # Step 4: Multi-Source Retrieval
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
            return GroundedSynthesisResult(
                answer="I do not have sufficient reliable information in the retrieved evidence to answer this question.",
                answer_type=AnswerType.INSUFFICIENT_INFORMATION,
                route=routing_dec.mode,
                status="INSUFFICIENT_INFORMATION",
                confidence=0.0,
                latency_ms=tot_lat_ms,
                routing_latency_ms=routing_lat_ms,
                retrieval_latency_ms=ret_lat_ms,
                termination_reason="empty_fused_evidence"
            )

        # Strict Evidence Sufficiency Gate: Verify query content relevance
        import re
        from collision.rag.embeddings import LocalEmbeddingModel
        q_content_words = set(re.findall(r"\b[a-z0-9_]{3,}\b", question.lower())) - LocalEmbeddingModel.STOP_WORDS
        
        has_substantive_match = False
        if not q_content_words:
            has_substantive_match = True
        else:
            for ev in fused_evidence:
                ev_words = set(re.findall(r"\b[a-z0-9_]{3,}\b", ev.text.lower()))
                if q_content_words.intersection(ev_words):
                    has_substantive_match = True
                    break

        if not has_substantive_match:
            tot_lat_ms = (time.perf_counter() - t_total_start) * 1000.0
            return GroundedSynthesisResult(
                answer="I do not have sufficient verified information in the retrieved evidence to answer this question accurately.",
                answer_type=AnswerType.INSUFFICIENT_INFORMATION,
                route=routing_dec.mode,
                status="INSUFFICIENT_INFORMATION",
                confidence=0.0,
                latency_ms=tot_lat_ms,
                routing_latency_ms=routing_lat_ms,
                retrieval_latency_ms=ret_lat_ms,
                termination_reason="insufficient_evidence_content_mismatch"
            )

        # Check for multi-source conflicts in evidence
        has_conflict = False
        if len(fused_evidence) >= 2:
            for i in range(len(fused_evidence)):
                for j in range(i + 1, len(fused_evidence)):
                    if self.verifier._check_contradiction(fused_evidence[i].text, fused_evidence[j].text):
                        has_conflict = True
                        break

        if has_conflict:
            tot_lat_ms = (time.perf_counter() - t_total_start) * 1000.0
            sources = []
            seen_src = set()
            for ev in fused_evidence:
                if ev.url not in seen_src:
                    seen_src.add(ev.url)
                    sources.append({"title": ev.source, "url": ev.url, "type": ev.source_type})

            return GroundedSynthesisResult(
                answer=f"Available sources provide conflicting evidence regarding this question ({fused_evidence[0].source} vs {fused_evidence[1].source}).",
                answer_type=AnswerType.CONFLICTING_EVIDENCE,
                route=routing_dec.mode,
                status="CONFLICTING_EVIDENCE",
                sources=sources,
                fused_evidence=fused_evidence,
                confidence=0.5,
                latency_ms=tot_lat_ms,
                routing_latency_ms=routing_lat_ms,
                retrieval_latency_ms=ret_lat_ms,
                termination_reason="conflicting_evidence"
            )

        # Step 6: Extraction-First Synthesis
        t_synth_start = time.perf_counter()
        synth_res: Optional[GroundedSynthesisResult] = None

        if routing_dec.mode == RouteMode.HYBRID:
            synth_res = self.synthesizer.synthesize_hybrid(question=question, fused_evidence=fused_evidence)
        else:
            synth_res = self.synthesizer.synthesize_extractive(
                question=question,
                fused_evidence=fused_evidence,
                route=routing_dec.mode
            )

        synth_lat_ms = (time.perf_counter() - t_synth_start) * 1000.0

        # Step 7: Fallback or Verification
        t_ver_start = time.perf_counter()
        if synth_res is not None and synth_res.answer:
            ver_res = self.verifier.verify(synth_res.answer, fused_evidence)
            synth_res.verification = ver_res
            synth_res.routing_latency_ms = routing_lat_ms
            synth_res.retrieval_latency_ms = ret_lat_ms
            synth_res.synthesis_latency_ms = synth_lat_ms
            synth_res.verification_latency_ms = (time.perf_counter() - t_ver_start) * 1000.0
            synth_res.latency_ms = (time.perf_counter() - t_total_start) * 1000.0
            return synth_res

        # If extractive synthesis yielded empty, invoke Fallback Handler
        t_fb_start = time.perf_counter()
        fb_res = self.fallback_handler.construct_fallback_answer(
            question=question,
            extracted_facts=[],
            fused_evidence=fused_evidence,
            route=routing_dec.mode
        )
        fb_lat_ms = (time.perf_counter() - t_fb_start) * 1000.0
        tot_lat_ms = (time.perf_counter() - t_total_start) * 1000.0

        if fb_res is not None:
            ver_res = self.verifier.verify(fb_res.answer, fused_evidence)
            fb_res.verification = ver_res
            fb_res.routing_latency_ms = routing_lat_ms
            fb_res.retrieval_latency_ms = ret_lat_ms
            fb_res.synthesis_latency_ms = synth_lat_ms
            fb_res.fallback_latency_ms = fb_lat_ms
            fb_res.latency_ms = tot_lat_ms
            return fb_res

        return GroundedSynthesisResult(
            answer="I do not have sufficient verified information to answer this question.",
            answer_type=AnswerType.INSUFFICIENT_INFORMATION,
            route=routing_dec.mode,
            status="INSUFFICIENT_INFORMATION",
            confidence=0.0,
            latency_ms=tot_lat_ms,
            routing_latency_ms=routing_lat_ms,
            retrieval_latency_ms=ret_lat_ms,
            termination_reason="insufficient_after_fallback"
        )

    def answer_with_context(
        self,
        question: str,
        documents: List[Dict[str, Any]],
        token_budget: int = 512,
        top_k: int = 3
    ) -> GroundedSynthesisResult:
        """
        Direct long-context answering pipeline (Phase 104):
        Compresses, reranks, fuses evidence, verifies claims, and synthesizes grounded answers.
        """
        t_start = time.perf_counter()

        if not question or not str(question).strip():
            return GroundedSynthesisResult(
                answer="Question is empty. Please provide a valid query.",
                answer_type=AnswerType.INSUFFICIENT_INFORMATION,
                route=RouteMode.INSUFFICIENT_INFORMATION,
                status="INSUFFICIENT_INFORMATION",
                confidence=0.0,
                latency_ms=(time.perf_counter() - t_start) * 1000.0,
                termination_reason="empty_query"
            )

        if not documents:
            return GroundedSynthesisResult(
                answer="I do not have sufficient reliable information in the supplied documents to answer this question.",
                answer_type=AnswerType.INSUFFICIENT_INFORMATION,
                route=RouteMode.INSUFFICIENT_INFORMATION,
                status="INSUFFICIENT_INFORMATION",
                confidence=0.0,
                latency_ms=(time.perf_counter() - t_start) * 1000.0,
                termination_reason="empty_documents"
            )

        # 1. Rerank raw candidate documents
        ranked_docs = self.reranker.rerank(query=question, candidates=documents, top_k=max(top_k * 2, 5))
        top_candidates = [doc for doc, score in ranked_docs]

        # 2. Compress & extract salient sentences
        compression_res = self.compressor.compress_passages(
            query=question,
            documents=top_candidates if top_candidates else documents,
            token_budget=token_budget
        )

        packed_sents = compression_res.get("packed_sentences", [])
        if not packed_sents:
            tot_ms = (time.perf_counter() - t_start) * 1000.0
            return GroundedSynthesisResult(
                answer="I do not have sufficient verified information in the retrieved context to answer this question accurately.",
                answer_type=AnswerType.INSUFFICIENT_INFORMATION,
                route=RouteMode.AUTO,
                status="INSUFFICIENT_INFORMATION",
                confidence=0.0,
                latency_ms=tot_ms,
                termination_reason="empty_compressed_evidence"
            )

        # 3. Convert packed sentences into FusedEvidence
        fused_evidence: List[FusedEvidence] = []
        for s_idx, sent_info in enumerate(packed_sents[:top_k * 2]):
            fused_evidence.append(FusedEvidence(
                source=sent_info.get("title", f"Document {sent_info.get('doc_id')}"),
                url=f"doc://{sent_info.get('doc_id')}",
                document_id=str(sent_info.get("doc_id", f"doc_{s_idx}")),
                chunk_id=s_idx,
                text=sent_info.get("text", ""),
                score=sent_info.get("score", 0.5),
                source_type=sent_info.get("source_type", "LOCAL")
            ))

        # 4. Check for multi-source contradictions / conflicts among high-salience evidence
        has_conflict = False
        high_salience_ev = [ev for ev in fused_evidence if ev.score >= 0.15]
        if len(high_salience_ev) >= 2:
            for i in range(len(high_salience_ev)):
                for j in range(i + 1, len(high_salience_ev)):
                    if self.verifier._check_contradiction(high_salience_ev[i].text, high_salience_ev[j].text):
                        has_conflict = True
                        break


        if has_conflict:
            tot_ms = (time.perf_counter() - t_start) * 1000.0
            sources = [{"title": ev.source, "url": ev.url, "type": ev.source_type} for ev in fused_evidence]
            return GroundedSynthesisResult(
                answer=f"Available sources provide conflicting evidence regarding this question ({fused_evidence[0].source} vs {fused_evidence[1].source}).",
                answer_type=AnswerType.CONFLICTING_EVIDENCE,
                route=RouteMode.AUTO,
                status="CONFLICTING_EVIDENCE",
                sources=sources,
                fused_evidence=fused_evidence,
                confidence=0.5,
                latency_ms=tot_ms,
                termination_reason="conflicting_evidence"
            )

        # 5. Synthesize grounded answer
        synth_res = self.synthesizer.synthesize_extractive(
            question=question,
            fused_evidence=fused_evidence,
            route=RouteMode.AUTO
        )

        if not synth_res or not synth_res.answer:
            synth_res = self.fallback_handler.construct_fallback_answer(
                question=question,
                extracted_facts=[],
                fused_evidence=fused_evidence,
                route=RouteMode.AUTO
            )

        tot_ms = (time.perf_counter() - t_start) * 1000.0

        if synth_res is not None and synth_res.answer:
            ans_lower = synth_res.answer.lower()
            additional_spans = []
            for ev in fused_evidence[1:]:
                ev_clean = NaturalGroundedFormatter.clean_raw_snippet(ev.text)
                if ev_clean and ev_clean.lower()[:25] not in ans_lower:
                    q_words = set(re.findall(r"\b[a-z0-9_]{3,}\b", question.lower())) - LocalEmbeddingModel.STOP_WORDS
                    ev_words = set(re.findall(r"\b[a-z0-9_]{3,}\b", ev_clean.lower()))
                    if len(q_words.intersection(ev_words)) >= 1:
                        additional_spans.append(NaturalGroundedFormatter.highlight_terms(ev_clean))
            if additional_spans:
                synth_res.answer = f"{synth_res.answer}\n\n" + " ".join(additional_spans)

            ver_res = self.verifier.verify(synth_res.answer, fused_evidence)
            synth_res.verification = ver_res
            synth_res.latency_ms = tot_ms
            return synth_res


        return GroundedSynthesisResult(
            answer="I do not have sufficient verified information in the retrieved context to answer this question accurately.",
            answer_type=AnswerType.INSUFFICIENT_INFORMATION,
            route=RouteMode.AUTO,
            status="INSUFFICIENT_INFORMATION",
            confidence=0.0,
            latency_ms=tot_ms,
            termination_reason="insufficient_after_long_context_synthesis"
        )

