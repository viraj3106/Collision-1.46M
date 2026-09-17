"""
COLLISION Phase 97 — Citation Verification Engine.

Validates that attached sources and URLs are traceable to actual retrieved evidence,
verifies that cited chunks logically substantiate the corresponding claims, and
detects fabricated/hallucinated citations.
"""

import os
import sys
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from collision.routing.schemas import FusedEvidence, ClaimVerification
from collision.rag.embeddings import LocalEmbeddingModel


@dataclass
class CitationRecord:
    source_uri: str
    is_retrieved: bool
    is_relevant_to_answer: bool
    supported_claims_count: int
    matching_chunk_ids: List[str] = field(default_factory=list)
    relevance_score: float = 0.0


@dataclass
class CitationAuditResult:
    total_citations_provided: int
    valid_citations: int
    hallucinated_citations: int
    irrelevant_citations: int
    citation_correctness: float  # valid / total_citations
    citation_completeness: float  # supported claims with citation / total supported claims
    records: List[CitationRecord] = field(default_factory=list)


class CitationAuditor:
    """
    Validates citations attached to model answers against actual retrieved evidence.
    """

    def __init__(self, embedding_model: Optional[LocalEmbeddingModel] = None):
        self.embedding_model = embedding_model if embedding_model is not None else LocalEmbeddingModel()

    def audit_citations(
        self,
        answer_text: str,
        cited_sources: List[str],
        retrieved_evidence: List[FusedEvidence],
        claim_verifications: Optional[List[ClaimVerification]] = None,
        relevance_threshold: float = 0.10
    ) -> CitationAuditResult:
        """
        Audits citations attached to the answer against the retrieved evidence pool.
        """
        if not cited_sources:
            # No citations provided
            supported_count = 0
            if claim_verifications:
                supported_count = sum(1 for c in claim_verifications if c.status == "SUPPORTED")
            completeness = 1.0 if supported_count == 0 else 0.0
            return CitationAuditResult(
                total_citations_provided=0,
                valid_citations=0,
                hallucinated_citations=0,
                irrelevant_citations=0,
                citation_correctness=1.0,
                citation_completeness=completeness,
                records=[]
            )

        # Build index of retrieved evidence sources
        evidence_by_source: Dict[str, List[FusedEvidence]] = {}
        for ev in retrieved_evidence:
            source_key = ev.source.strip()
            if source_key not in evidence_by_source:
                evidence_by_source[source_key] = []
            evidence_by_source[source_key].append(ev)

        records: List[CitationRecord] = []
        valid_count = 0
        hallucinated_count = 0
        irrelevant_count = 0

        ans_vec = self.embedding_model.embed_text(answer_text) if answer_text else None

        for citation in cited_sources:
            clean_citation = citation.strip()
            is_retrieved = clean_citation in evidence_by_source

            if not is_retrieved:
                # Citation was not even in the retrieved evidence pool!
                hallucinated_count += 1
                records.append(CitationRecord(
                    source_uri=clean_citation,
                    is_retrieved=False,
                    is_relevant_to_answer=False,
                    supported_claims_count=0,
                    matching_chunk_ids=[],
                    relevance_score=0.0
                ))
                continue

            # Check if matching chunks actually support or relate to the answer
            matching_chunks = evidence_by_source[clean_citation]
            chunk_ids = [str(c.chunk_id) for c in matching_chunks]
            best_rel = 0.0

            for c in matching_chunks:
                c_vec = self.embedding_model.embed_text(c.text)
                if ans_vec is not None and len(ans_vec) > 0:
                    rel = self.embedding_model.cosine_similarity(ans_vec, c_vec)
                    if rel > best_rel:
                        best_rel = float(rel)

            # Count claims this source supported
            claim_support_count = 0
            if claim_verifications:
                for cv in claim_verifications:
                    if cv.status == "SUPPORTED" and clean_citation in cv.supporting_evidence:
                        claim_support_count += 1

            is_relevant = (best_rel >= relevance_threshold or claim_support_count > 0)
            if is_relevant:
                valid_count += 1
            else:
                irrelevant_count += 1

            records.append(CitationRecord(
                source_uri=clean_citation,
                is_retrieved=True,
                is_relevant_to_answer=is_relevant,
                supported_claims_count=claim_support_count,
                matching_chunk_ids=chunk_ids,
                relevance_score=round(best_rel, 4)
            ))

        total = len(cited_sources)
        correctness = valid_count / float(total) if total > 0 else 1.0

        # Completeness: fraction of supported claims that have a cited source
        total_supported_claims = 0
        supported_with_citation = 0
        if claim_verifications:
            for cv in claim_verifications:
                if cv.status == "SUPPORTED":
                    total_supported_claims += 1
                    if any(s in cited_sources for s in cv.supporting_evidence):
                        supported_with_citation += 1

        if total_supported_claims > 0:
            completeness = supported_with_citation / float(total_supported_claims)
        else:
            completeness = 1.0

        return CitationAuditResult(
            total_citations_provided=total,
            valid_citations=valid_count,
            hallucinated_citations=hallucinated_count,
            irrelevant_citations=irrelevant_count,
            citation_correctness=round(correctness, 4),
            citation_completeness=round(completeness, 4),
            records=records
        )
