"""
COLLISION Phase 98 — Evidence Extractor.

Extracts deterministic factual spans, dates, numbers, specifications, and entities
directly from verified evidence chunks without risking neural hallucination.
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from collision.routing.schemas import FusedEvidence
from collision.rag.embeddings import LocalEmbeddingModel
from collision.grounding.schemas import ExtractedFact


class EvidenceExtractor:
    """
    Deterministic factual extractor that isolates the most relevant verified
    evidence spans answering the user's question.
    """

    def __init__(self, embedding_model: Optional[LocalEmbeddingModel] = None):
        self.embedding_model = embedding_model if embedding_model is not None else LocalEmbeddingModel()

    def _split_into_sentences(self, text: str) -> List[str]:
        if not text:
            return []
        raw = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s.strip() for s in raw if len(s.strip()) > 5]

    def _classify_fact_type(self, text: str) -> str:
        t_lower = text.lower()
        if re.search(r"\b(?:january|february|march|april|may|june|july|august|september|october|november|december|\b\d{4}\b)", t_lower):
            return "date"
        if re.search(r"\b\d+(?:\.\d+)?\s*(?:parameters|layers|heads|million|tokens|gb|mb|cores|tops|%)", t_lower):
            return "specification"
        if re.search(r"\b\d+\b", t_lower):
            return "number"
        if re.search(r"\b(?:is|are|defined as|means|refers to)\b", t_lower):
            return "definition"
        return "span"

    def extract_relevant_spans(
        self,
        question: str,
        fused_evidence: List[FusedEvidence],
        top_n: int = 3,
        min_relevance: float = 0.12
    ) -> List[ExtractedFact]:
        """
        Extracts verified factual statements answering the user's query from fused evidence.
        """
        if not question or not fused_evidence:
            return []

        q_clean = question.strip()
        q_vec = self.embedding_model.embed_text(q_clean)
        q_content_words = set(re.findall(r"\b[a-z0-9_]{3,}\b", q_clean.lower())) - LocalEmbeddingModel.STOP_WORDS

        candidate_facts: List[Tuple[float, ExtractedFact]] = []

        for ev in fused_evidence:
            sentences = self._split_into_sentences(ev.text)
            if not sentences:
                sentences = [ev.text.strip()]

            for s in sentences:
                s_words = set(re.findall(r"\b[a-z0-9_]{3,}\b", s.lower()))
                word_overlap = len(q_content_words.intersection(s_words))

                # Require at least 1 content word overlap if query has content words
                if q_content_words and word_overlap == 0:
                    continue

                s_vec = self.embedding_model.embed_text(s)
                sim_score = float(self.embedding_model.cosine_similarity(q_vec, s_vec))

                # Combined score: embedding similarity + keyword overlap boost
                combined_score = sim_score + (0.12 * min(word_overlap, 4))

                if combined_score >= min_relevance or (word_overlap >= 1 and sim_score > 0.03):
                    fact_type = self._classify_fact_type(s)
                    fact = ExtractedFact(
                        fact_text=s,
                        fact_type=fact_type,
                        source=ev.source,
                        url=ev.url,
                        document_id=ev.document_id,
                        chunk_id=ev.chunk_id,
                        original_chunk_text=ev.text,
                        relevance_score=combined_score,
                        extraction_confidence=min(1.0, 0.5 + combined_score * 0.5),
                        claim_linkage=f"{ev.source}::chunk_{ev.chunk_id}"
                    )
                    candidate_facts.append((combined_score, fact))

        # Sort by relevance score descending
        candidate_facts.sort(key=lambda x: x[0], reverse=True)

        # Deduplicate facts with identical or near-identical text
        seen_texts = set()
        final_facts: List[ExtractedFact] = []

        for score, fact in candidate_facts:
            norm_text = re.sub(r"\s+", " ", fact.fact_text.lower()).strip()
            if norm_text not in seen_texts:
                seen_texts.add(norm_text)
                final_facts.append(fact)
                if len(final_facts) >= top_n:
                    break

        return final_facts
