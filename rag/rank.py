import re
import math
from typing import List, Tuple, Dict, Any

STOPWORDS = {
    'who', 'what', 'where', 'when', 'why', 'how', 'is', 'was', 'are', 'were',
    'do', 'does', 'did', 'the', 'a', 'an', 'and', 'or', 'in', 'on', 'at', 'to',
    'for', 'of', 'with', 'about', 'by', 'from', 'as', 'that', 'this', 'it'
}

def _stem(word: str) -> str:
    w = word.lower()
    for suffix in ('ing', 'tions', 'tion', 'ers', 'er', 'ed', 'es', 's'):
        if len(w) > len(suffix) + 3 and w.endswith(suffix):
            return w[:-len(suffix)]
    return w

class LexicalRanker:
    """
    Lightweight BM25 / Term-Frequency lexical ranker with stemming, stopword filtering,
    and search-provider position prior for ordering retrieved passages.
    """
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b

    def _tokenize(self, text: str) -> List[str]:
        return [_stem(w) for w in re.findall(r'\w+', text) if len(w) > 1 and w.lower() not in STOPWORDS]

    def score_passages(self, query: str, passages: List[str]) -> List[Tuple[float, str]]:
        if not passages:
            return []

        query_terms = self._tokenize(query)
        if not query_terms:
            return [(1.0 / (idx + 1.0), p) for idx, p in enumerate(passages)]

        doc_tokens = [self._tokenize(p) for p in passages]
        doc_lens = [len(dt) for dt in doc_tokens]
        avg_dl = sum(doc_lens) / max(1, len(doc_lens))
        num_docs = len(passages)

        # Calculate IDF with smoothing floor
        idf = {}
        for term in set(query_terms):
            df = sum(1 for dt in doc_tokens if term in dt)
            idf[term] = math.log((num_docs - df + 0.5) / (df + 0.5) + 1.0) + 1.0

        scored = []
        for idx, (p, dt, dl) in enumerate(zip(passages, doc_tokens, doc_lens)):
            score = 0.0
            tf_dict = {}
            for term in dt:
                tf_dict[term] = tf_dict.get(term, 0) + 1

            for q_term in query_terms:
                if q_term in tf_dict:
                    tf = tf_dict[q_term]
                    denom = tf + self.k1 * (1.0 - self.b + self.b * (dl / max(1.0, avg_dl)))
                    score += idf.get(q_term, 1.0) * (tf * (self.k1 + 1.0)) / max(0.1, denom)

            # Positional bias preserves the high-confidence search engine ranking
            position_bias = 5.0 / (idx + 1.0)
            final_score = score + position_bias
            scored.append((final_score, p))

        # Sort descending by score
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored


