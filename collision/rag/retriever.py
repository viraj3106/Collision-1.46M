import re
from typing import List, Optional, Set
from collision.rag.schemas import DocumentChunk, RetrievalItem
from collision.rag.index import VectorIndex
from collision.rag.embeddings import LocalEmbeddingModel

class DocumentRetriever:
    """
    Retrieval coordinator that queries the VectorIndex and enforces
    minimum relevance confidence thresholds to prevent context contamination.
    """

    def __init__(
        self,
        index: VectorIndex,
        default_top_k: int = 5,
        default_relevance_threshold: float = 0.10
    ):
        self.index = index
        self.default_top_k = default_top_k
        self.default_relevance_threshold = default_relevance_threshold

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        relevance_threshold: Optional[float] = None
    ) -> List[RetrievalItem]:
        """
        Retrieves top-k chunks matching the query that strictly exceed relevance_threshold.
        If no chunk meets the threshold or if chunks lack content-word overlap with query, returns an empty list.
        """
        if not query or not str(query).strip():
            return []

        k = top_k if top_k is not None else self.default_top_k
        threshold = relevance_threshold if relevance_threshold is not None else self.default_relevance_threshold

        clean_q = query.lower().strip()
        q_content_words = set(re.findall(r"\b[a-z0-9_]{3,}\b", clean_q)) - LocalEmbeddingModel.STOP_WORDS

        raw_results = self.index.search(query, top_k=k)

        items: List[RetrievalItem] = []
        for chunk, score in raw_results:
            chunk_words = set(re.findall(r"\b[a-z0-9_]{3,}\b", chunk.text.lower()))
            overlap = len(q_content_words.intersection(chunk_words)) if q_content_words else 0
            if score >= threshold or (overlap >= 2 and score >= 0.03):
                if q_content_words and overlap == 0:
                    continue
                ranking_score = score + 0.10 * min(overlap, 4)
                items.append(RetrievalItem(
                    chunk=chunk,
                    similarity_score=ranking_score,
                    source=chunk.source
                ))

        items.sort(key=lambda x: x.similarity_score, reverse=True)
        return items
