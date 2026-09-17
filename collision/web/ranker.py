from typing import List, Optional
from collision.web.schemas import WebDocument, WebEvidenceChunk
from collision.rag.embeddings import LocalEmbeddingModel

class WebEvidenceRanker:
    """
    Ranks extracted web paragraphs and evidence chunks deterministically
    against the user query using the local CPU embedding model.
    """

    def __init__(
        self,
        embedding_model: Optional[LocalEmbeddingModel] = None,
        default_top_k: int = 5,
        default_relevance_threshold: float = 0.10
    ):
        self.embedding_model = embedding_model if embedding_model is not None else LocalEmbeddingModel()
        self.default_top_k = default_top_k
        self.default_relevance_threshold = default_relevance_threshold

    def rank_evidence(
        self,
        query: str,
        documents: List[WebDocument],
        top_k: Optional[int] = None,
        relevance_threshold: Optional[float] = None
    ) -> List[WebEvidenceChunk]:
        """
        Segments documents into evidence candidates, computes similarity against query,
        and returns top-k chunks passing relevance_threshold.
        """
        if not query or not documents:
            return []

        k = top_k if top_k is not None else self.default_top_k
        threshold = relevance_threshold if relevance_threshold is not None else self.default_relevance_threshold

        query_vec = self.embedding_model.embed_text(query)

        candidates: List[WebEvidenceChunk] = []

        import re
        query_words = set(re.findall(r"\b[a-z0-9_]{3,}\b", query.lower())) - LocalEmbeddingModel.STOP_WORDS

        for doc in documents:
            # If document has discrete paragraphs, score each paragraph
            chunks = doc.extracted_paragraphs if doc.extracted_paragraphs else [doc.text]
            for c_idx, chunk_text in enumerate(chunks):
                chunk_str = chunk_text.strip()
                if len(chunk_str) < 15:
                    continue

                chunk_vec = self.embedding_model.embed_text(chunk_str)
                score = float(self.embedding_model.cosine_similarity(query_vec, chunk_vec))

                chunk_words = set(re.findall(r"\b[a-z0-9_]{3,}\b", chunk_str.lower()))
                word_overlap = len(query_words.intersection(chunk_words)) if query_words else 0

                if score >= threshold or (word_overlap >= 1 and score >= 0.03):
                    ranking_score = score + 0.10 * min(word_overlap, 3)
                    candidates.append(WebEvidenceChunk(
                        url=doc.url,
                        title=doc.title,
                        domain=doc.domain,
                        text=chunk_str,
                        similarity_score=ranking_score
                    ))

        # Sort deterministically: score descending, then domain, url, text ascending
        candidates.sort(key=lambda c: (-c.similarity_score, c.domain, c.url, c.text[:30]))

        return candidates[:k]
