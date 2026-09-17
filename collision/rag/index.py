import os
import json
from typing import List, Tuple, Union, Optional
import numpy as np
from collision.rag.schemas import DocumentChunk
from collision.rag.embeddings import LocalEmbeddingModel

class VectorIndex:
    """
    Deterministic local vector index for storing and searching document chunk embeddings.
    Supports in-memory cosine search, deterministic tie-breaking, and disk persistence (save/load).
    """

    def __init__(self, embedding_model: Optional[LocalEmbeddingModel] = None):
        self.embedding_model = embedding_model if embedding_model is not None else LocalEmbeddingModel()
        self.chunks: List[DocumentChunk] = []
        self.embeddings: np.ndarray = np.zeros((0, self.embedding_model.dimension), dtype=np.float32)

    def size(self) -> int:
        return len(self.chunks)

    def clear(self):
        self.chunks = []
        self.embeddings = np.zeros((0, self.embedding_model.dimension), dtype=np.float32)

    def add(self, chunks: List[DocumentChunk]):
        """
        Embeds and adds document chunks to the vector index.
        """
        if not chunks:
            return

        texts = [c.text for c in chunks]
        new_vecs = self.embedding_model.embed_batch(texts)

        self.chunks.extend(chunks)
        if self.embeddings.shape[0] == 0:
            self.embeddings = new_vecs
        else:
            self.embeddings = np.vstack([self.embeddings, new_vecs])

    def search(
        self,
        query: Union[str, np.ndarray],
        top_k: int = 5
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Searches the index for chunks most similar to the query.
        Returns list of (DocumentChunk, similarity_score) sorted deterministically.
        """
        if self.size() == 0:
            return []

        if isinstance(query, str):
            query_vec = self.embedding_model.embed_text(query)
        else:
            query_vec = query

        # Cosine similarity against all stored embeddings
        scores = np.dot(self.embeddings, query_vec)

        # Build candidate list with tie-breaking metadata
        candidates = []
        for idx, (chunk, score) in enumerate(zip(self.chunks, scores)):
            candidates.append((chunk, float(score), idx))

        # Sort: score descending, then doc_id, chunk_id, idx ascending for deterministic order
        candidates.sort(key=lambda item: (-item[1], item[0].document_id, item[0].chunk_id, item[2]))

        top_candidates = candidates[:max(1, top_k)]
        return [(c[0], c[1]) for c in top_candidates]

    def save(self, dirpath: str):
        """
        Saves index metadata, chunks, and embeddings matrix to disk.
        """
        os.makedirs(dirpath, exist_ok=True)

        chunks_data = [c.to_dict() for c in self.chunks]
        chunks_path = os.path.join(dirpath, "chunks.json")
        with open(chunks_path, "w", encoding="utf-8") as f:
            json.dump(chunks_data, f, indent=2)

        meta_path = os.path.join(dirpath, "index_meta.json")
        meta = {
            "dimension": self.embedding_model.dimension,
            "total_chunks": len(self.chunks)
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        vectors_path = os.path.join(dirpath, "vectors.npy")
        np.save(vectors_path, self.embeddings)

    def load(self, dirpath: str):
        """
        Loads saved index chunks and embeddings from disk.
        """
        chunks_path = os.path.join(dirpath, "chunks.json")
        vectors_path = os.path.join(dirpath, "vectors.npy")

        if not os.path.exists(chunks_path) or not os.path.exists(vectors_path):
            raise FileNotFoundError(f"Index files missing in directory: {dirpath}")

        with open(chunks_path, "r", encoding="utf-8") as f:
            chunks_data = json.load(f)

        self.chunks = [DocumentChunk(**item) for item in chunks_data]
        self.embeddings = np.load(vectors_path).astype(np.float32)
