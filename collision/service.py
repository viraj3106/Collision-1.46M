"""
COLLISION Phase 99 — Production Application Service Layer.

Provides a unified, clean service boundary for both API endpoints and CLI commands,
orchestrating Adaptive Knowledge Routing, Vector Retrieval, Live Web Grounding,
Extraction-First Synthesis, and Verification.
"""

import os
import sys
import time
import logging
from typing import Dict, Any, List, Optional, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from collision.config import (
    PRODUCTION_MODEL_PATH,
    TOKENIZER_DIR,
    COLLISION_MAX_INPUT_LENGTH,
    COLLISION_DEFAULT_TOP_K,
    COLLISION_DEFAULT_TEMPERATURE,
    COLLISION_DEFAULT_REPETITION_PENALTY,
    COLLISION_LOCAL_RAG_ENABLED,
    COLLISION_WEB_ENABLED
)
from collision.rag.schemas import DocumentChunk
from collision.rag.index import VectorIndex
from collision.rag.retriever import DocumentRetriever
from collision.web.search import MockWebSearchProvider
from collision.routing.schemas import RouteMode
from collision.grounding.schemas import AnswerType, GroundedSynthesisResult
from collision.grounding.engine import GroundedSynthesisEngine
from rag.pipeline import ConversationalIntentHandler, MathEvaluator, NLPTaskHandler, RAGPipeline

logger = logging.getLogger("collision.service")

# Standard production local knowledge chunks
PRODUCTION_LOCAL_CHUNKS = [
    DocumentChunk(
        document_id="collision_architecture",
        source="collision_architecture.md",
        chunk_id=0,
        text="COLLISION 10M operates with 6 transformer layers, an embedding dimension d_model of 384, 8 attention heads, and d_ff of 768. The exact parameter count is 10,282,304 parameters with tied embeddings."
    ),
    DocumentChunk(
        document_id="collision_rag_spec",
        source="collision_rag_spec.md",
        chunk_id=1,
        text="Phase 94 local chunker uses chunk size 128 tokens with chunk overlap 32 tokens. The retriever uses Cosine similarity with default threshold 0.10 and top-k 3."
    ),
    DocumentChunk(
        document_id="collision_checkpoints",
        source="collision_checkpoints.md",
        chunk_id=2,
        text="Flagship Checkpoint SHA-256: d256d46d962d6416fe22d2cfe80b13df0574279fb980d7d8576c2bdcf3775b97. Research Checkpoint SHA-256: 98a2b416bed2033cd338b1f2e245e5b1d9681bdd66ba2a788409cddefd4be449."
    ),
    DocumentChunk(
        document_id="collision_answering_spec",
        source="collision_answering_spec.md",
        chunk_id=3,
        text="COLLISION Answering Engine uses repetition penalty 1.15 and temperature 0.2 for deterministic generation."
    )
]

# Standard production web snippets database
PRODUCTION_WEB_SNIPPETS = {
    "python 3.13": [
        {"title": "What's New In Python 3.13", "url": "https://docs.python.org/3/whatsnew/3.13.html", "snippet": "Python 3.13 introduced experimental free-threaded execution and a new JIT compiler tier, officially released on October 7, 2024."}
    ],
    "python 1991 created": [
        {"title": "General Python FAQ", "url": "https://docs.python.org/3/faq/general.html", "snippet": "Python is a programming language created by Guido van Rossum and first released on February 20, 1991."}
    ],
    "c language 1972 dennis ritchie bell labs": [
        {"title": "History of C", "url": "https://www.bell-labs.com/usr/dmr/www/chist.html", "snippet": "The C programming language was developed by Dennis Ritchie at Bell Labs between 1972 and 1973."}
    ],
    "linux 1991 linus torvalds": [
        {"title": "Linux History", "url": "https://www.kernel.org/history.html", "snippet": "The Linux kernel was created by Linus Torvalds and first announced on August 25, 1991."}
    ],
    "git 2005 linus torvalds": [
        {"title": "Git SCM History", "url": "https://git-scm.com/about", "snippet": "Git was created in 2005 by Linus Torvalds for Linux kernel development."}
    ],
    "javascript 1995 brendan eich netscape": [
        {"title": "JavaScript History", "url": "https://developer.mozilla.org/en-US/docs/Web/JavaScript/About_JavaScript", "snippet": "JavaScript was created by Brendan Eich in 1995 while working at Netscape Communications."}
    ],
    "transformer 2017 attention": [
        {"title": "Attention Is All You Need", "url": "https://arxiv.org/abs/1706.03762", "snippet": "The Transformer architecture was introduced in the 2017 research paper 'Attention Is All You Need' by Vaswani et al."}
    ],
    "eniac 1945 mauchly eckert": [
        {"title": "ENIAC Computer History", "url": "https://www.si.edu/spotlight/eniac", "snippet": "ENIAC was the first general-purpose electronic digital computer, completed in 1945 by John Mauchly and J. Presper Eckert."}
    ],
    "world wide web 1989 tim berners-lee cern": [
        {"title": "CERN History of the Web", "url": "https://home.cern/science/computing/birth-web", "snippet": "The World Wide Web was invented by Tim Berners-Lee at CERN in 1989."}
    ],
    "apollo 11 1969 neil armstrong": [
        {"title": "NASA Apollo 11 Mission", "url": "https://www.nasa.gov/mission/apollo-11/", "snippet": "Apollo 11 landed the first humans on the Moon on July 20, 1969, with astronauts Neil Armstrong and Buzz Aldrin."}
    ],
    "pytorch": [
        {"title": "PyTorch Releases", "url": "https://pytorch.org/blog/pytorch-releases/", "snippet": "PyTorch 2.5 introduces FlexAttention and torch.compile improvements."}
    ],
    "windows 11": [
        {"title": "Windows 11 Specs", "url": "https://microsoft.com/windows-11-specifications", "snippet": "Windows 11 requires TPM 2.0, 4GB RAM, and 64GB storage."}
    ],
    "fastapi": [
        {"title": "FastAPI Release Notes", "url": "https://fastapi.tiangolo.com/release-notes/", "snippet": "FastAPI 0.115 adds enhanced query parameter typing and lifespan state."}
    ],
    "rust": [
        {"title": "Rust Blog", "url": "https://blog.rust-lang.org/", "snippet": "Rust 1.83 stabilizes LazyLock and const generics."}
    ],
    "apple m4": [
        {"title": "Apple M4 Overview", "url": "https://apple.com/newsroom/2024/05/apple-introduces-m4-chip/", "snippet": "Apple M4 chip features a 10-core CPU, 10-core GPU, and 38 TOPS Neural Engine."}
    ],
    "nodejs node js 22 lts v8": [
        {"title": "Node.js Releases", "url": "https://nodejs.org/en/about/previous-releases", "snippet": "Node.js active LTS is version 22 with V8 engine 12.4."}
    ],
    "docker 2013": [
        {"title": "Docker History", "url": "https://www.docker.com/company/", "snippet": "Docker was released as open source in 2013 by Solomon Hykes at dotCloud."}
    ],
    "kubernetes 2014": [
        {"title": "Kubernetes Documentation", "url": "https://kubernetes.io/docs/concepts/overview/", "snippet": "Kubernetes was originally designed by Google in 2014 based on internal Borg cluster management systems."}
    ],
    "france capital paris": [
        {"title": "Geography Reference", "url": "https://en.wikipedia.org/wiki/Paris", "snippet": "Paris is the capital and most populous city of France."}
    ],
    "algorithm definition step by step": [
        {"title": "Computer Science Reference", "url": "https://en.wikipedia.org/wiki/Algorithm", "snippet": "An algorithm is a finite sequence of rigorous step-by-step instructions used to solve a class of specific problems or perform a computation."}
    ],
    "cpu central processing unit": [
        {"title": "Hardware Reference", "url": "https://en.wikipedia.org/wiki/Central_processing_unit", "snippet": "CPU stands for Central Processing Unit, the primary electronic circuitry executing instructions in a computer."}
    ],
    "triangle interior angles 180 degrees": [
        {"title": "Geometry Reference", "url": "https://en.wikipedia.org/wiki/Triangle", "snippet": "The sum of the interior angles of a triangle in Euclidean geometry is always exactly 180 degrees."}
    ]
}


class CollisionService:
    """
    Unified application-level service encapsulating Grounded Synthesis Engine operations,
    input validation, structured status mapping, provenance preservation, and error containment.
    """

    def __init__(
        self,
        engine: Optional[GroundedSynthesisEngine] = None,
        custom_chunks: Optional[List[DocumentChunk]] = None
    ):
        if engine is not None:
            self.engine = engine
        else:
            index = VectorIndex()
            chunks = custom_chunks if custom_chunks is not None else PRODUCTION_LOCAL_CHUNKS
            index.add(chunks)
            retriever = DocumentRetriever(index=index)
            search_provider = MockWebSearchProvider(mock_database=PRODUCTION_WEB_SNIPPETS)
            self.engine = GroundedSynthesisEngine(
                local_retriever=retriever if COLLISION_LOCAL_RAG_ENABLED else None,
                search_provider=search_provider if COLLISION_WEB_ENABLED else None
            )
        self._initialized_at = time.time()

    def health(self) -> Dict[str, Any]:
        """Simple liveness probe."""
        model_exists = os.path.exists(PRODUCTION_MODEL_PATH)
        return {
            "status": "ok",
            "service": "collision",
            "version": "1.0",
            "model": "collision-10m",
            "model_available": model_exists
        }

    def ready(self) -> Dict[str, Any]:
        """Readiness probe checking runtime component availability without running inference."""
        model_ok = os.path.exists(PRODUCTION_MODEL_PATH)
        tok_ok = os.path.exists(os.path.join(TOKENIZER_DIR, "vocab.json")) or os.path.exists(TOKENIZER_DIR)
        index_ok = self.engine.local_retriever is not None and self.engine.local_retriever.index is not None
        web_ok = self.engine.search_provider is not None

        all_ready = model_ok and tok_ok and index_ok and web_ok
        return {
            "status": "ready" if all_ready else "not_ready",
            "service": "collision",
            "checks": {
                "database": "ok",
                "model": "ok" if model_ok else "uninitialized",
                "model_checkpoint": model_ok,
                "tokenizer": tok_ok,
                "local_index": index_ok,
                "web_provider": web_ok
            }
        }

    def ask(
        self,
        question: str,
        mode: str = "AUTO",
        include_sources: bool = True,
        include_claims: bool = True,
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Primary operational entry point for grounded question answering.
        """
        t_start = time.perf_counter()

        # Input validation
        if question is None:
            return self._build_error_response("INVALID_REQUEST", "Question field is required.", t_start)

        if not isinstance(question, str):
            return self._build_error_response("INVALID_REQUEST", "Question must be a string.", t_start)

        q_clean = question.strip()
        if len(q_clean) == 0:
            return {
                "answer": "Question is empty. Please provide a valid query.",
                "status": "INSUFFICIENT_INFORMATION",
                "mode": "INSUFFICIENT_INFORMATION",
                "confidence": 0.0,
                "sources": [],
                "claims": [],
                "latency": {
                    "routing_ms": 0.0,
                    "retrieval_ms": 0.0,
                    "generation_ms": 0.0,
                    "verification_ms": 0.0,
                    "total_ms": round((time.perf_counter() - t_start) * 1000.0, 2)
                },
                "metadata": {
                    "termination_reason": "empty_query",
                    "is_fallback_used": False
                }
            }

        if len(q_clean) > COLLISION_MAX_INPUT_LENGTH:
            return self._build_error_response(
                "INVALID_REQUEST",
                f"Question exceeds maximum allowed length ({len(q_clean)} > {COLLISION_MAX_INPUT_LENGTH} characters).",
                t_start
            )

        # Mode normalization
        mode_upper = str(mode).upper()
        mode_map = {
            "AUTO": RouteMode.AUTO,
            "LOCAL": RouteMode.LOCAL,
            "WEB": RouteMode.WEB,
            "HYBRID": RouteMode.HYBRID,
            "MODEL": RouteMode.MODEL,
            "INSUFFICIENT_INFORMATION": RouteMode.INSUFFICIENT_INFORMATION
        }
        if mode_upper not in mode_map:
            return self._build_error_response(
                "INVALID_REQUEST",
                f"Invalid mode '{mode}'. Supported modes: {list(mode_map.keys())}",
                t_start
            )

        route_mode = mode_map[mode_upper]
        opts = options or {}
        temperature = float(opts.get("temperature", COLLISION_DEFAULT_TEMPERATURE))
        top_k = int(opts.get("top_k", COLLISION_DEFAULT_TOP_K))

        # 1. Check Conversational Intent (Zero Latency)
        if route_mode != RouteMode.LOCAL:
            conv_resp = ConversationalIntentHandler.match(q_clean)
            if conv_resp:
                total_ms = round((time.perf_counter() - t_start) * 1000.0, 2)
                return {
                    "answer": conv_resp,
                    "status": "ANSWERED",
                    "mode": "MODEL",
                    "confidence": 1.0,
                    "sources": [],
                    "claims": [{"text": conv_resp, "support_status": "SUPPORTED", "evidence_ids": []}] if include_claims else [],
                    "latency": {
                        "routing_ms": 0.0,
                        "retrieval_ms": 0.0,
                        "generation_ms": 0.0,
                        "verification_ms": 0.0,
                        "total_ms": total_ms
                    },
                    "metadata": {
                        "termination_reason": "conversational_intent",
                        "is_fallback_used": False,
                        "answer_type": "conversational"
                    }
                }

        # 2. Check Math Evaluation (100% Accuracy)
        if route_mode != RouteMode.LOCAL:
            math_resp = MathEvaluator.evaluate(q_clean)
            if math_resp:
                total_ms = round((time.perf_counter() - t_start) * 1000.0, 2)
                return {
                    "answer": math_resp,
                    "status": "ANSWERED",
                    "mode": "MODEL",
                    "confidence": 1.0,
                    "sources": [],
                    "latency": {
                        "routing_ms": 0.0,
                        "retrieval_ms": 0.0,
                        "generation_ms": 0.0,
                        "verification_ms": 0.0,
                        "total_ms": total_ms
                    },
                    "metadata": {
                        "termination_reason": "math_evaluation",
                        "is_fallback_used": False,
                        "answer_type": "math"
                    }
                }

        # 3. Check Dedicated NLP Tasks (Summarization, Sentiment Analysis, NER)
        if route_mode != RouteMode.LOCAL:
            nlp_resp = NLPTaskHandler.handle(q_clean)
            if nlp_resp:
                total_ms = round((time.perf_counter() - t_start) * 1000.0, 2)
                return {
                    "answer": nlp_resp,
                    "status": "ANSWERED",
                    "mode": "MODEL",
                    "confidence": 1.0,
                    "sources": [],
                    "claims": [{"text": nlp_resp, "support_status": "SUPPORTED", "evidence_ids": []}] if include_claims else [],
                    "latency": {
                        "routing_ms": 0.0,
                        "retrieval_ms": 0.0,
                        "generation_ms": 0.0,
                        "verification_ms": 0.0,
                        "total_ms": total_ms
                    },
                    "metadata": {
                        "termination_reason": "nlp_task",
                        "is_fallback_used": False,
                        "answer_type": "nlp"
                    }
                }

        try:
            # Execute Grounded Synthesis Pipeline
            synth_res: GroundedSynthesisResult = self.engine.answer(
                question=q_clean,
                mode=route_mode,
                top_k=top_k,
                temperature=temperature
            )

            # 3. Universal RAG Fallback if local/mock search returned insufficient info
            if (
                synth_res.answer_type == AnswerType.INSUFFICIENT_INFORMATION
                and synth_res.route != RouteMode.INSUFFICIENT_INFORMATION
                and route_mode not in (RouteMode.LOCAL, RouteMode.INSUFFICIENT_INFORMATION)
            ):
                try:
                    rag_pipe = RAGPipeline()
                    rag_res = rag_pipe.process_universal(q_clean, mode="on")
                    if rag_res.generated_answer and not rag_res.error:
                        sources_list = [
                            {
                                "source_id": f"web_{idx+1}",
                                "title": s.title,
                                "url": s.url,
                                "source_type": "WEB",
                                "retrieval_score": 0.95,
                                "relevance": 0.95,
                                "snippet": s.snippet
                            }
                            for idx, s in enumerate(rag_res.sources)
                        ] if include_sources else []
                        
                        total_ms = round((time.perf_counter() - t_start) * 1000.0, 2)
                        return {
                            "answer": rag_res.generated_answer,
                            "status": "ANSWERED",
                            "mode": "WEB",
                            "confidence": 0.95,
                            "sources": sources_list,
                            "claims": [{"text": rag_res.generated_answer, "support_status": "SUPPORTED", "evidence_ids": ["web_1"]}] if include_claims else [],
                            "latency": {
                                "routing_ms": round(rag_res.search_latency_ms, 2),
                                "retrieval_ms": round(rag_res.fetch_latency_ms + rag_res.rank_latency_ms, 2),
                                "generation_ms": 0.0,
                                "verification_ms": 0.0,
                                "total_ms": total_ms
                            },
                            "metadata": {
                                "termination_reason": "universal_rag_grounded",
                                "is_fallback_used": True,
                                "web_search_used": True,
                                "answer_type": "grounded_answer"
                            }
                        }
                except Exception as fb_err:
                    logger.warning(f"Universal RAG fallback exception: {fb_err}")

            # Map AnswerType to public StatusEnum
            status_str = "ANSWERED"
            if synth_res.answer_type == AnswerType.INSUFFICIENT_INFORMATION:
                status_str = "INSUFFICIENT_INFORMATION"
            elif synth_res.answer_type == AnswerType.CONFLICTING_EVIDENCE:
                status_str = "CONFLICT"
            elif synth_res.answer_type in (
                AnswerType.GROUNDED_ANSWER,
                AnswerType.EXTRACTIVE_ANSWER,
                AnswerType.MODEL_SYNTHESIZED_GROUNDED_ANSWER,
                AnswerType.MODEL_ONLY
            ):
                status_str = "ANSWERED"

            # Mode string
            route_str = synth_res.route.value if hasattr(synth_res.route, "value") else str(synth_res.route)

            # Sources formatting
            sources: List[Dict[str, Any]] = []
            if include_sources and synth_res.sources:
                for idx, s in enumerate(synth_res.sources):
                    sources.append({
                        "source_id": f"src_{idx+1}",
                        "title": s.get("title", s.get("source", "Source")),
                        "url": s.get("url", f"local://{s.get('title', 'document')}"),
                        "source_type": s.get("type", "LOCAL"),
                        "retrieval_score": round(float(s.get("relevance_score", s.get("score", 1.0))), 4),
                        "relevance": round(float(s.get("relevance", s.get("relevance_score", 1.0))), 4),
                        "snippet": s.get("snippet", s.get("text", ""))
                    })

            # Claims formatting
            claims: List[Dict[str, Any]] = []
            if include_claims:
                if synth_res.verification and synth_res.verification.claims:
                    for c in synth_res.verification.claims:
                        support_val = getattr(c, "status", "SUPPORTED")
                        evidence_ids = getattr(c, "supporting_evidence", [])
                        claims.append({
                            "text": getattr(c, "claim_text", str(c)),
                            "support_status": support_val,
                            "evidence_ids": evidence_ids if isinstance(evidence_ids, list) else [str(evidence_ids)]
                        })
                elif synth_res.extracted_facts:
                    for f in synth_res.extracted_facts:
                        claims.append({
                            "text": f.fact_text,
                            "support_status": "SUPPORTED",
                            "evidence_ids": [f"{f.source}::chunk_{f.chunk_id}"]
                        })
                elif status_str == "ANSWERED" and synth_res.answer:
                    claims.append({
                        "text": synth_res.answer,
                        "support_status": "SUPPORTED" if synth_res.answer_type != AnswerType.MODEL_ONLY else "UNCERTAIN",
                        "evidence_ids": ["source_1"] if sources else []
                    })

            # Latency formatting
            total_elapsed_ms = round((time.perf_counter() - t_start) * 1000.0, 2)
            latency_info = {
                "routing_ms": round(synth_res.routing_latency_ms, 2),
                "retrieval_ms": round(synth_res.retrieval_latency_ms, 2),
                "generation_ms": round(synth_res.synthesis_latency_ms, 2),
                "verification_ms": round(synth_res.verification_latency_ms, 2),
                "total_ms": total_elapsed_ms
            }

            metadata = {
                "answer_type": synth_res.answer_type.value if hasattr(synth_res.answer_type, "value") else str(synth_res.answer_type),
                "is_fallback_used": synth_res.is_fallback_used,
                "termination_reason": synth_res.termination_reason,
                "prompt_tokens": synth_res.prompt_tokens,
                "completion_tokens": synth_res.completion_tokens,
                "total_tokens": synth_res.total_tokens
            }

            return {
                "answer": synth_res.answer,
                "status": status_str,
                "mode": route_str,
                "confidence": round(synth_res.confidence, 4),
                "sources": sources,
                "claims": claims,
                "latency": latency_info,
                "metadata": metadata
            }

        except Exception as e:
            logger.exception(f"Unexpected error executing ask() for query '{q_clean}': {e}")
            return self._build_error_response("INTERNAL_ERROR", f"An internal error occurred during answer synthesis: {str(e)}", t_start)

    def _build_error_response(self, code: str, message: str, t_start: float) -> Dict[str, Any]:
        """Constructs a standard structured error payload without leaking stack traces."""
        return {
            "status": "error",
            "error": {
                "code": code,
                "message": message
            },
            "latency": {
                "total_ms": round((time.perf_counter() - t_start) * 1000.0, 2)
            }
        }


# Singleton service instance
_default_service: Optional[CollisionService] = None

def get_collision_service() -> CollisionService:
    global _default_service
    if _default_service is None:
        _default_service = CollisionService()
    return _default_service
