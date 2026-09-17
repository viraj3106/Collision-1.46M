import time
import urllib.parse
from typing import List, Optional, Union, Dict, Any
from collision.rag.schemas import RAGStatus, RetrievalItem
from collision.rag.retriever import DocumentRetriever
from collision.answering.engine import CollisionAnsweringEngine
from collision.answering.schemas import AnswerStatus, AnswerResult
from collision.web.schemas import (
    RetrievalMode,
    WebSource,
    WebEvidenceChunk,
    WebDocument,
    GroundingResult
)
from collision.web.search import WebSearchProvider, MockWebSearchProvider
from collision.web.fetch import safe_fetch_page
from collision.web.extractor import WebPageExtractor
from collision.web.ranker import WebEvidenceRanker

class LiveWebGroundingEngine:
    """
    Live Web Grounding Engine for COLLISION (Phase 95).
    Orchestrates Query Routing (Local RAG -> Web Fallback), Web Search,
    Safe Web Fetching, Text Extraction, Evidence Ranking, and Grounded Generation.
    """

    def __init__(
        self,
        search_provider: Optional[WebSearchProvider] = None,
        local_retriever: Optional[DocumentRetriever] = None,
        answering_engine: Optional[CollisionAnsweringEngine] = None,
        extractor: Optional[WebPageExtractor] = None,
        ranker: Optional[WebEvidenceRanker] = None
    ):
        self.search_provider = search_provider if search_provider is not None else MockWebSearchProvider()
        self.local_retriever = local_retriever
        self.answering_engine = answering_engine if answering_engine is not None else CollisionAnsweringEngine()
        self.extractor = extractor if extractor is not None else WebPageExtractor()
        self.ranker = ranker if ranker is not None else WebEvidenceRanker()

    def _format_web_grounded_prompt(
        self,
        question: str,
        evidence_chunks: List[WebEvidenceChunk],
        max_completion_tokens: int = 80
    ) -> str:
        """
        Formats a strict web-grounded QA prompt within the 256-token context budget.
        """
        header = "Answer using ONLY the retrieved web evidence. If evidence is insufficient, say so. Do not invent facts or citations.\n\nWeb Evidence:\n"

        evidence_lines = []
        for idx, chunk in enumerate(evidence_chunks, 1):
            evidence_lines.append(f"[{idx}] ({chunk.domain}) {chunk.text.strip()}")

        context_block = "\n".join(evidence_lines)
        footer = f"\n\nQuestion:\n{question}\n\nAnswer:"

        full_prompt = header + context_block + footer
        encoded = self.answering_engine.tokenizer.encode(full_prompt, bos=True)

        max_prompt_len = self.answering_engine.model_cfg.max_seq_len - max_completion_tokens
        if len(encoded) > max_prompt_len and len(evidence_chunks) > 1:
            # Drop lowest-scoring chunk until it fits
            return self._format_web_grounded_prompt(question, evidence_chunks[:-1], max_completion_tokens)

        return full_prompt

    def answer(
        self,
        question: str,
        mode: RetrievalMode = RetrievalMode.LOCAL if hasattr(RetrievalMode, "LOCAL") else "AUTO",
        top_k: int = 3,
        relevance_threshold: float = 0.10,
        max_tokens: int = 80,
        temperature: float = 0.7,
        deterministic: bool = False
    ) -> GroundingResult:
        """
        Executes grounded answering with automated local RAG -> web search fallback.
        """
        total_start = time.perf_counter()
        t_search_start = 0.0
        search_lat_ms = 0.0
        fetch_lat_ms = 0.0

        # Normalization of mode string
        mode_val = mode.value if isinstance(mode, RetrievalMode) else str(mode).upper()

        # Step 1: Check Local RAG if permitted
        if mode_val in ("LOCAL", "AUTO", "HYBRID") and self.local_retriever is not None:
            local_items: List[RetrievalItem] = self.local_retriever.retrieve(
                query=question,
                top_k=top_k,
                relevance_threshold=relevance_threshold
            )
            if local_items:
                # Local evidence found! Answer using Local RAG
                header = "Answer using ONLY the provided context facts. If insufficient, say so.\n\nContext:\n"
                ctx_lines = [f"[{i+1}] ({item.source}) {item.chunk.text.strip()}" for i, item in enumerate(local_items)]
                full_prompt = header + "\n".join(ctx_lines) + f"\n\nQuestion:\n{question}\n\nAnswer:"

                t_gen_start = time.perf_counter()
                gen_res: AnswerResult = self.answering_engine.answer(
                    question=full_prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    deterministic=deterministic
                )
                gen_lat_ms = (time.perf_counter() - t_gen_start) * 1000.0
                tot_lat_ms = (time.perf_counter() - total_start) * 1000.0

                sources = [
                    WebSource(title=item.source, url=f"local://{item.source}", domain="local_documents")
                    for item in local_items
                ]
                # Deduplicate sources
                unique_sources = []
                seen_src = set()
                for s in sources:
                    if s.url not in seen_src:
                        seen_src.add(s.url)
                        unique_sources.append(s)

                return GroundingResult(
                    answer=gen_res.text,
                    status=RAGStatus.ANSWER if gen_res.status == AnswerStatus.ANSWER else RAGStatus.UNCERTAIN,
                    sources=unique_sources,
                    evidence=[],
                    retrieval_mode=RetrievalMode.LOCAL,
                    confidence=gen_res.confidence_score,
                    latency_ms=tot_lat_ms,
                    search_latency_ms=0.0,
                    fetch_latency_ms=0.0,
                    generation_latency_ms=gen_lat_ms,
                    prompt_tokens=gen_res.prompt_tokens,
                    completion_tokens=gen_res.completion_tokens,
                    total_tokens=gen_res.total_tokens,
                    termination_reason=gen_res.termination_reason
                )

        if mode_val == "LOCAL":
            # Local only requested but no evidence found
            tot_lat_ms = (time.perf_counter() - total_start) * 1000.0
            return GroundingResult(
                answer="I do not have sufficient reliable information in the local documents to answer this question.",
                status=RAGStatus.INSUFFICIENT_INFORMATION,
                sources=[],
                evidence=[],
                retrieval_mode=RetrievalMode.LOCAL,
                confidence=0.0,
                latency_ms=tot_lat_ms,
                prompt_tokens=len(self.answering_engine.tokenizer.encode(question, bos=True)),
                completion_tokens=18,
                total_tokens=len(self.answering_engine.tokenizer.encode(question, bos=True)) + 18,
                termination_reason="insufficient_local_evidence"
            )

        # Step 2: Web Search Fallback
        t_search_start = time.perf_counter()
        search_res = self.search_provider.search(question, max_results=top_k)
        search_lat_ms = (time.perf_counter() - t_search_start) * 1000.0

        if search_res.error or not search_res.results:
            tot_lat_ms = (time.perf_counter() - total_start) * 1000.0
            return GroundingResult(
                answer="I do not have sufficient reliable information from the web to answer this question.",
                status=RAGStatus.INSUFFICIENT_INFORMATION,
                sources=[],
                evidence=[],
                retrieval_mode=RetrievalMode.WEB,
                confidence=0.0,
                latency_ms=tot_lat_ms,
                search_latency_ms=search_lat_ms,
                fetch_latency_ms=0.0,
                generation_latency_ms=0.0,
                prompt_tokens=len(self.answering_engine.tokenizer.encode(question, bos=True)),
                completion_tokens=18,
                total_tokens=len(self.answering_engine.tokenizer.encode(question, bos=True)) + 18,
                termination_reason="web_search_no_results"
            )

        # Step 3: Fetch & Extract Web Pages
        t_fetch_start = time.perf_counter()
        web_docs: List[WebDocument] = []

        for item in search_res.results:
            html_text = None
            if not item.url.startswith("https://example.com") and not item.url.startswith("mock://"):
                html_text, _ = safe_fetch_page(item.url, timeout_sec=2.5)

            if html_text:
                doc = self.extractor.extract(html_text, url=item.url, title=item.title)
            else:
                # Use snippet as fallback document content
                doc = self.extractor.extract(item.snippet, url=item.url, title=item.title)
            web_docs.append(doc)

        fetch_lat_ms = (time.perf_counter() - t_fetch_start) * 1000.0

        # Step 4: Rank Evidence Chunks
        ranked_evidence = self.ranker.rank_evidence(
            query=question,
            documents=web_docs,
            top_k=top_k,
            relevance_threshold=relevance_threshold
        )

        if not ranked_evidence:
            tot_lat_ms = (time.perf_counter() - total_start) * 1000.0
            return GroundingResult(
                answer="I do not have sufficient reliable web information to answer this question.",
                status=RAGStatus.INSUFFICIENT_INFORMATION,
                sources=[],
                evidence=[],
                retrieval_mode=RetrievalMode.WEB,
                confidence=0.0,
                latency_ms=tot_lat_ms,
                search_latency_ms=search_lat_ms,
                fetch_latency_ms=fetch_lat_ms,
                generation_latency_ms=0.0,
                prompt_tokens=len(self.answering_engine.tokenizer.encode(question, bos=True)),
                completion_tokens=18,
                total_tokens=len(self.answering_engine.tokenizer.encode(question, bos=True)) + 18,
                termination_reason="insufficient_web_evidence_threshold"
            )

        # Step 5: Format Web Prompt & Generate Answer
        grounded_prompt = self._format_web_grounded_prompt(question, ranked_evidence, max_completion_tokens=max_tokens)
        t_gen_start = time.perf_counter()
        gen_res: AnswerResult = self.answering_engine.answer(
            question=grounded_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            deterministic=deterministic
        )
        gen_lat_ms = (time.perf_counter() - t_gen_start) * 1000.0
        tot_lat_ms = (time.perf_counter() - total_start) * 1000.0

        # Extract unique sources from evidence
        sources = []
        seen_src = set()
        for chunk in ranked_evidence:
            if chunk.url not in seen_src:
                seen_src.add(chunk.url)
                sources.append(WebSource(title=chunk.title, url=chunk.url, domain=chunk.domain))

        rag_status = RAGStatus.ANSWER
        if gen_res.status == AnswerStatus.INSUFFICIENT_INFORMATION:
            rag_status = RAGStatus.INSUFFICIENT_INFORMATION
        elif gen_res.status == AnswerStatus.UNCERTAIN or gen_res.repetition_score > 0.40:
            rag_status = RAGStatus.UNCERTAIN

        return GroundingResult(
            answer=gen_res.text,
            status=rag_status,
            sources=sources,
            evidence=ranked_evidence,
            retrieval_mode=RetrievalMode.WEB,
            confidence=gen_res.confidence_score,
            latency_ms=tot_lat_ms,
            search_latency_ms=search_lat_ms,
            fetch_latency_ms=fetch_lat_ms,
            generation_latency_ms=gen_lat_ms,
            prompt_tokens=gen_res.prompt_tokens,
            completion_tokens=gen_res.completion_tokens,
            total_tokens=gen_res.total_tokens,
            termination_reason=gen_res.termination_reason
        )
