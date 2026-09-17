import time
from typing import List, Optional, Union, Dict, Any
from collision.rag.schemas import RAGStatus, RAGResult, RetrievalItem, DocumentChunk
from collision.rag.retriever import DocumentRetriever
from collision.answering.engine import CollisionAnsweringEngine
from collision.answering.schemas import AnswerStatus, AnswerResult

class GroundedRAGEngine:
    """
    Grounded RAG Engine for COLLISION (Phase 94).
    Orchestrates local document retrieval, relevance thresholding,
    grounded prompt synthesis, source citation tracking, and model generation.
    """

    def __init__(
        self,
        retriever: DocumentRetriever,
        answering_engine: Optional[CollisionAnsweringEngine] = None
    ):
        self.retriever = retriever
        self.answering_engine = answering_engine if answering_engine is not None else CollisionAnsweringEngine()

    def _format_grounded_prompt(
        self,
        question: str,
        retrieval_items: List[RetrievalItem],
        max_completion_tokens: int = 80
    ) -> str:
        """
        Formats a strict grounded QA prompt within the 256-token context budget.
        """
        # Header instruction
        header = "Answer the question using ONLY the provided context facts. If the context does not contain the answer, say the information is insufficient.\n\nContext:\n"
        
        # Format context lines with source attribution
        context_lines = []
        for idx, item in enumerate(retrieval_items, 1):
            src = item.source
            txt = item.chunk.text.strip()
            context_lines.append(f"[{idx}] (Source: {src}) {txt}")

        context_block = "\n".join(context_lines)
        footer = f"\n\nQuestion:\n{question}\n\nAnswer:"

        full_prompt = header + context_block + footer
        encoded = self.answering_engine.tokenizer.encode(full_prompt, bos=True)

        # Ensure full prompt + completion tokens fits within max_seq_len (256)
        max_prompt_len = self.answering_engine.model_cfg.max_seq_len - max_completion_tokens

        if len(encoded) > max_prompt_len and len(retrieval_items) > 1:
            # Drop lowest-scoring chunks until it fits
            return self._format_grounded_prompt(question, retrieval_items[:-1], max_completion_tokens)

        return full_prompt

    def answer(
        self,
        question: str,
        top_k: int = 3,
        relevance_threshold: float = 0.20,
        max_tokens: int = 80,
        temperature: float = 0.7,
        top_k_sampling: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        deterministic: bool = False
    ) -> RAGResult:
        """
        Executes end-to-end grounded RAG answering.
        """
        total_start = time.perf_counter()

        # 1. Retrieval Phase
        t_ret_start = time.perf_counter()
        retrieved_items = self.retriever.retrieve(
            query=question,
            top_k=top_k,
            relevance_threshold=relevance_threshold
        )
        t_ret_end = time.perf_counter()
        retrieval_latency_ms = (t_ret_end - t_ret_start) * 1000.0

        # 2. Threshold Check
        if not retrieved_items:
            total_latency_ms = (time.perf_counter() - total_start) * 1000.0
            return RAGResult(
                answer="I do not have sufficient reliable information in the provided documents to answer this question.",
                status=RAGStatus.INSUFFICIENT_INFORMATION,
                sources=[],
                retrieved_chunks=[],
                confidence=0.0,
                latency_ms=total_latency_ms,
                retrieval_latency_ms=retrieval_latency_ms,
                generation_latency_ms=0.0,
                prompt_tokens=len(self.answering_engine.tokenizer.encode(question, bos=True)),
                completion_tokens=18,
                total_tokens=len(self.answering_engine.tokenizer.encode(question, bos=True)) + 18,
                repetition_score=0.0,
                termination_reason="insufficient_evidence_threshold"
            )

        # 3. Grounded Prompt Formulation
        grounded_prompt = self._format_grounded_prompt(question, retrieved_items, max_completion_tokens=max_tokens)
        sources_used = sorted(list({item.source for item in retrieved_items}))

        # 4. Model Generation Phase
        t_gen_start = time.perf_counter()
        gen_result: AnswerResult = self.answering_engine.answer(
            question=grounded_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k_sampling,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            deterministic=deterministic
        )
        t_gen_end = time.perf_counter()
        generation_latency_ms = (t_gen_end - t_gen_start) * 1000.0
        total_latency_ms = (time.perf_counter() - total_start) * 1000.0

        # 5. Status & Grounding Classification
        rag_status = RAGStatus.ANSWER
        if gen_result.status == AnswerStatus.INSUFFICIENT_INFORMATION:
            rag_status = RAGStatus.INSUFFICIENT_INFORMATION
        elif gen_result.status == AnswerStatus.UNCERTAIN or gen_result.repetition_score > 0.40:
            rag_status = RAGStatus.UNCERTAIN

        return RAGResult(
            answer=gen_result.text,
            status=rag_status,
            sources=sources_used,
            retrieved_chunks=retrieved_items,
            confidence=gen_result.confidence_score,
            latency_ms=total_latency_ms,
            retrieval_latency_ms=retrieval_latency_ms,
            generation_latency_ms=generation_latency_ms,
            prompt_tokens=gen_result.prompt_tokens,
            completion_tokens=gen_result.completion_tokens,
            total_tokens=gen_result.total_tokens,
            repetition_score=gen_result.repetition_score,
            termination_reason=gen_result.termination_reason
        )
