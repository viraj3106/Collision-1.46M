import re
from typing import List, Optional, Set
from collision.rag.schemas import RetrievalItem
from collision.web.schemas import WebEvidenceChunk
from collision.routing.schemas import FusedEvidence

class EvidenceFusion:
    """
    Normalizes, deduplicates, and ranks multi-source evidence (Local RAG + Web).
    Preserves full provenance metadata without artificially inflating confidence.
    """

    @staticmethod
    def _normalize_text_key(text: str) -> str:
        clean = re.sub(r"\W+", " ", text.lower()).strip()
        words = clean.split()
        return " ".join(words[:15]) # Use first 15 words as fingerprint

    def fuse(
        self,
        local_items: Optional[List[RetrievalItem]] = None,
        web_items: Optional[List[WebEvidenceChunk]] = None,
        max_fused_chunks: int = 5
    ) -> List[FusedEvidence]:
        """
        Merges local and web evidence into a unified, deduplicated list of FusedEvidence.
        """
        fused: List[FusedEvidence] = []
        seen_keys: Set[str] = set()

        # 1. Ingest Local Evidence
        if local_items:
            for item in local_items:
                txt = item.chunk.text.strip()
                key = self._normalize_text_key(txt)
                if key in seen_keys:
                    continue
                seen_keys.add(key)

                fused.append(FusedEvidence(
                    source=item.source,
                    url=f"local://{item.source}",
                    document_id=item.chunk.document_id,
                    chunk_id=item.chunk.chunk_id,
                    text=txt,
                    score=float(item.similarity_score),
                    source_type="LOCAL"
                ))

        # 2. Ingest Web Evidence
        if web_items:
            for item in web_items:
                txt = item.text.strip()
                key = self._normalize_text_key(txt)
                if key in seen_keys:
                    continue
                seen_keys.add(key)

                fused.append(FusedEvidence(
                    source=item.domain,
                    url=item.url,
                    document_id=item.domain,
                    chunk_id=0,
                    text=txt,
                    score=float(item.similarity_score),
                    source_type="WEB"
                ))

        # Sort deterministically: score descending, then source_type, source, text
        fused.sort(key=lambda e: (-e.score, e.source_type, e.source, e.text[:20]))

        return fused[:max_fused_chunks]
