import re
import math
import hashlib
from typing import List, Union, Set
import numpy as np

class LocalEmbeddingModel:
    """
    Deterministic, CPU-first local embedding model for COLLISION RAG.
    Generates normalized dense representations from word and subword n-gram features
    using fixed-dimensional feature hashing, stop-word downweighting, and L2 normalization.
    Requires no GPU, no internet, and zero external API dependencies.
    """

    STOP_WORDS: Set[str] = {
        "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
        "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but",
        "by", "can", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from",
        "further", "had", "has", "have", "having", "he", "her", "here", "hers", "herself", "him",
        "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just",
        "me", "more", "most", "my", "myself", "no", "nor", "not", "now", "of", "off", "on", "once",
        "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own", "s", "same", "she",
        "should", "so", "some", "such", "t", "than", "that", "the", "their", "theirs", "them",
        "themselves", "then", "there", "these", "they", "this", "those", "through", "to", "too",
        "under", "until", "up", "very", "was", "we", "were", "what", "when", "where", "which",
        "while", "who", "whom", "why", "will", "with", "you", "your", "yours", "yourself"
    }

    def __init__(self, dimension: int = 256):
        self.dimension = dimension

    def _hash_token(self, token: str, seed: int = 0) -> int:
        """Deterministically hashes a string token to a bucket index."""
        h = hashlib.md5((f"{seed}:{token}").encode("utf-8")).digest()
        val = int.from_bytes(h[:4], byteorder="little")
        return val % self.dimension

    def _hash_sign(self, token: str, seed: int = 1) -> float:
        """Deterministically generates a +1 or -1 sign for feature hashing."""
        h = hashlib.md5((f"{seed}:{token}").encode("utf-8")).digest()
        val = int.from_bytes(h[4:8], byteorder="little")
        return 1.0 if (val % 2 == 0) else -1.0

    def embed_text(self, text: str) -> np.ndarray:
        """
        Embeds a single string into an L2-normalized float32 vector of shape (dimension,).
        """
        if not text or not str(text).strip():
            return np.zeros(self.dimension, dtype=np.float32)

        clean = text.lower().strip()
        tokens = re.findall(r"\w+|[^\w\s]", clean)
        if not tokens:
            return np.zeros(self.dimension, dtype=np.float32)

        vec = np.zeros(self.dimension, dtype=np.float32)

        # 1. Unigrams with content-word boosting and stop-word filtering
        for token in tokens:
            if token in self.STOP_WORDS:
                continue
            idx = self._hash_token(token, seed=0)
            sign = self._hash_sign(token, seed=1)
            weight = 2.0 + math.log1p(len(token))
            vec[idx] += sign * weight

        # 2. Bigrams for phrase preservation
        content_tokens = [t for t in tokens if t not in self.STOP_WORDS]
        if len(content_tokens) > 1:
            for t1, t2 in zip(content_tokens[:-1], content_tokens[1:]):
                bigram = f"{t1}_{t2}"
                idx = self._hash_token(bigram, seed=2)
                sign = self._hash_sign(bigram, seed=3)
                vec[idx] += sign * 2.5

        # 3. Subword character n-grams (3-grams) for robust spelling / morphological matching
        for token in tokens:
            if token not in self.STOP_WORDS and len(token) >= 3:
                for i in range(len(token) - 2):
                    tri = token[i:i+3]
                    idx = self._hash_token(tri, seed=4)
                    sign = self._hash_sign(tri, seed=5)
                    vec[idx] += sign * 0.8

        # L2 Normalization
        norm = np.linalg.norm(vec)
        if norm > 1e-9:
            vec = vec / norm
        else:
            vec = np.zeros(self.dimension, dtype=np.float32)

        return vec.astype(np.float32)

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """
        Embeds a batch of texts into a 2D array of shape (N, dimension).
        """
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)

        embeddings = [self.embed_text(t) for t in texts]
        return np.vstack(embeddings).astype(np.float32)

    @staticmethod
    def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Computes cosine similarity between two 1D vectors.
        Assumes vectors are L2-normalized.
        """
        dot = float(np.dot(vec1, vec2))
        return max(-1.0, min(1.0, dot))
