"""
COLLISION Phase 104 — Hybrid Reranker & Relevance Scoring Engine.

Combines:
1. Local Neural Semantic Embedding Cosine Similarity
2. BM25 / Lexical Term Frequency with length penalty
3. Content-word intersection overlap
4. Adversarial prompt-injection penalty filtering
5. Deterministic reciprocal rank and nDCG calculation
"""

import re
import math
from typing import List, Dict, Any, Tuple, Optional, Union
from collision.rag.embeddings import LocalEmbeddingModel
from collision.rag.schemas import DocumentChunk, RetrievalItem
from collision.web.schemas import WebEvidenceChunk

INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior)\s+instructions?",
    r"critical\s+override",
    r"system\s+notice",
    r"admin\s+instruction",
    r"security_breach",
    r"output\s+['\"]?pwned",
    r"system_compromised",
    r"delete\s+database"
]

class HybridReranker:
    """
    Lightweight, deterministic multi-signal reranker optimizing evidence precision
    and adversarial distractor suppression for long-context retrieval.
    """

    def __init__(
        self,
        embedding_model: Optional[LocalEmbeddingModel] = None,
        alpha_semantic: float = 0.50,
        beta_bm25: float = 0.35,
        gamma_overlap: float = 0.15,
        injection_penalty: float = 0.80,
        k1: float = 1.5,
        b: float = 0.75
    ):
        self.embedding_model = embedding_model if embedding_model is not None else LocalEmbeddingModel()
        self.alpha_semantic = alpha_semantic
        self.beta_bm25 = beta_bm25
        self.gamma_overlap = gamma_overlap
        self.injection_penalty = injection_penalty
        self.k1 = k1
        self.b = b

    def _tokenize(self, text: str) -> List[str]:
        words = re.findall(r"\b[a-z0-9_]{2,}\b", text.lower())
        return [w for w in words if w not in LocalEmbeddingModel.STOP_WORDS]

    def _detect_injection_penalty(self, text: str) -> float:
        text_lower = text.lower()
        for pat in INJECTION_PATTERNS:
            if re.search(pat, text_lower):
                return self.injection_penalty
        return 0.0

    def score_candidate(self, query: str, candidate_text: str, query_vec: Optional[List[float]] = None) -> Dict[str, float]:
        """Calculates multi-factor score breakdown for a single candidate."""
        if not candidate_text or not candidate_text.strip():
            return {"semantic": 0.0, "bm25": 0.0, "overlap": 0.0, "penalty": 0.0, "final_score": -1.0}

        q_clean = query.lower().strip()
        c_clean = candidate_text.lower().strip()

        # 1. Semantic Similarity
        if query_vec is None:
            query_vec = self.embedding_model.embed_text(q_clean)
        cand_vec = self.embedding_model.embed_text(c_clean)
        sem_score = max(0.0, float(self.embedding_model.cosine_similarity(query_vec, cand_vec)))

        # 2. Token Overlap
        q_tokens = self._tokenize(q_clean)
        c_tokens = self._tokenize(c_clean)
        q_token_set = set(q_tokens)
        c_token_set = set(c_tokens)

        overlap_count = len(q_token_set.intersection(c_token_set))
        overlap_score = min(1.0, overlap_count / max(1.0, float(len(q_token_set))))

        # 3. BM25 Term Frequency
        bm25_score = 0.0
        if q_tokens and c_tokens:
            doc_len = len(c_tokens)
            avg_doc_len = 50.0  # reference passage size
            tf_dict = {}
            for t in c_tokens:
                tf_dict[t] = tf_dict.get(t, 0) + 1
            for qt in q_token_set:
                tf = tf_dict.get(qt, 0)
                if tf > 0:
                    denom = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / avg_doc_len))
                    bm25_score += (tf * (self.k1 + 1.0)) / max(0.1, denom)
            bm25_score = min(1.0, bm25_score / max(1.0, float(len(q_token_set))))

        # 4. Injection Penalty
        penalty = self._detect_injection_penalty(candidate_text)

        # 5. Combined Score
        final_score = (
            self.alpha_semantic * sem_score +
            self.beta_bm25 * bm25_score +
            self.gamma_overlap * overlap_score -
            penalty
        )

        return {
            "semantic": round(sem_score, 4),
            "bm25": round(bm25_score, 4),
            "overlap": round(overlap_score, 4),
            "penalty": round(penalty, 4),
            "final_score": round(max(0.0, final_score), 4)
        }

    def rerank(
        self,
        query: str,
        candidates: List[Any],
        top_k: Optional[int] = None,
        min_score_threshold: float = 0.05
    ) -> List[Tuple[Any, float]]:
        """
        Reranks a list of candidate documents/chunks/strings.
        Returns sorted list of (candidate, score).
        """
        if not candidates:
            return []

        query_vec = self.embedding_model.embed_text(query)
        scored_items = []

        for item in candidates:
            if isinstance(item, str):
                text = item
            elif isinstance(item, dict):
                text = item.get("text", "")
            elif hasattr(item, "text"):
                text = getattr(item, "text", "")
            elif hasattr(item, "chunk") and hasattr(item.chunk, "text"):
                text = item.chunk.text
            else:
                text = str(item)

            score_info = self.score_candidate(query, text, query_vec=query_vec)
            final_s = score_info["final_score"]
            if final_s >= min_score_threshold:
                scored_items.append((item, final_s))

        # Sort descending by final score
        scored_items.sort(key=lambda x: x[1], reverse=True)

        if top_k is not None:
            return scored_items[:top_k]
        return scored_items

    @staticmethod
    def compute_retrieval_metrics(
        ranked_doc_ids: List[str],
        target_needle_ids: List[str],
        k_values: List[int] = [1, 3, 5, 10]
    ) -> Dict[str, float]:
        """
        Computes standard deterministic information retrieval metrics:
        Recall@K, Precision@K, MRR (Mean Reciprocal Rank), nDCG@K.
        """
        if not target_needle_ids:
            return {
                "mrr": 1.0,
                **{f"recall@{k}": 1.0 for k in k_values},
                **{f"precision@{k}": 1.0 for k in k_values},
                **{f"ndcg@{k}": 1.0 for k in k_values}
            }

        target_set = set(target_needle_ids)
        metrics = {}

        # 1. MRR (Mean Reciprocal Rank)
        mrr = 0.0
        for rank, doc_id in enumerate(ranked_doc_ids, start=1):
            if doc_id in target_set:
                mrr = 1.0 / rank
                break
        metrics["mrr"] = round(mrr, 4)

        # 2. Recall@K & Precision@K & nDCG@K
        for k in k_values:
            top_k_docs = ranked_doc_ids[:k]
            hits = sum(1 for doc in top_k_docs if doc in target_set)
            
            recall = hits / float(len(target_set))
            precision = hits / float(max(1, len(top_k_docs)))
            
            # DCG / IDCG
            dcg = 0.0
            for r_idx, doc in enumerate(top_k_docs, start=1):
                rel = 1.0 if doc in target_set else 0.0
                dcg += rel / math.log2(r_idx + 1)

            idcg = sum(1.0 / math.log2(i + 2) for i in range(min(k, len(target_set))))
            ndcg = (dcg / idcg) if idcg > 0 else 0.0

            metrics[f"recall@{k}"] = round(recall, 4)
            metrics[f"precision@{k}"] = round(precision, 4)
            metrics[f"ndcg@{k}"] = round(ndcg, 4)

        return metrics
