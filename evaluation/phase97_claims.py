"""
COLLISION Phase 97 — Claim-Level Grounding & Hallucination Auditor.

Decomposes generated answers into individual factual assertions, scores each claim
against retrieved evidence with provenance tracking, and detects unsupported or
contradicted assertions.
"""

import os
import sys
import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from collision.routing.schemas import FusedEvidence
from collision.rag.embeddings import LocalEmbeddingModel


@dataclass
class ClaimRecord:
    claim_index: int
    claim_text: str
    status: str  # "SUPPORTED", "UNSUPPORTED", "CONTRADICTED", "UNCERTAIN"
    evidence_ids: List[str] = field(default_factory=list)
    evidence_sources: List[str] = field(default_factory=list)
    best_evidence_text: str = ""
    support_score: float = 0.0
    contradiction_score: float = 0.0
    is_factual: bool = True


@dataclass
class ClaimAuditResult:
    total_claims: int
    supported_claims: int
    unsupported_claims: int
    contradicted_claims: int
    uncertain_claims: int
    claim_support_rate: float
    unsupported_claim_rate: float
    contradiction_rate: float
    evidence_coverage: float
    hallucination_rate: float
    claims: List[ClaimRecord] = field(default_factory=list)


class ClaimAuditor:
    """
    Evaluates answer statements claim-by-claim against retrieved evidence pools.
    """

    def __init__(self, embedding_model: Optional[LocalEmbeddingModel] = None):
        self.embedding_model = embedding_model if embedding_model is not None else LocalEmbeddingModel()

    def extract_claims(self, text: str) -> List[str]:
        """
        Extracts discrete factual claim strings from an answer.
        Removes conversational wrappers and parses into independent sentences.
        """
        if not text or not text.strip():
            return []

        cleaned = text.strip()
        # Strip common prefixes
        cleaned = re.sub(r"^(Answer:\s*|Response:\s*|COLLISION:\s*)", "", cleaned, flags=re.IGNORECASE)
        cleaned = cleaned.strip()

        if not cleaned:
            return []

        raw_sentences = re.split(r"(?<=[.!?])\s+", cleaned)
        claims = []
        for s in raw_sentences:
            s_clean = s.strip()
            # Exclude trivial non-factual fragments
            if len(s_clean) >= 8 and not re.match(r"^(hello|hi|thanks|thank you|yes|no)\b", s_clean, re.IGNORECASE):
                claims.append(s_clean)

        if not claims and cleaned:
            claims = [cleaned]

        return claims

    def detect_contradiction(self, claim: str, evidence_text: str) -> Tuple[bool, float]:
        """
        Checks for numerical and polarity contradictions between claim and evidence.
        Returns (is_contradicted, contradiction_score).
        """
        c_lower = claim.lower()
        e_lower = evidence_text.lower()

        # 1. Number + noun binding mismatch (e.g., '64 layers' vs '6 layers')
        claim_num_nouns = re.findall(r"\b(\d+)\s+([a-z]{3,})\b", c_lower)
        evidence_num_nouns = re.findall(r"\b(\d+)\s+([a-z]{3,})\b", e_lower)

        if claim_num_nouns and evidence_num_nouns:
            claim_dict = {noun: num for num, noun in claim_num_nouns}
            evidence_dict = {noun: num for num, noun in evidence_num_nouns}

            for noun, c_num in claim_dict.items():
                if noun in evidence_dict and evidence_dict[noun] != c_num:
                    return True, 1.0

        # General number mismatch when entities overlap
        claim_numbers = set(re.findall(r"\b\d+\b", c_lower))
        evidence_numbers = set(re.findall(r"\b\d+\b", e_lower))
        claim_words = set(re.findall(r"\b[a-z]{4,}\b", c_lower))
        evidence_words = set(re.findall(r"\b[a-z]{4,}\b", e_lower))
        shared_words = claim_words.intersection(evidence_words)

        if len(shared_words) >= 2 and claim_numbers and evidence_numbers:
            diff_numbers = (claim_numbers - evidence_numbers)
            if diff_numbers and not claim_numbers.issubset(evidence_numbers):
                return True, 0.9

        # 2. Negation polarity mismatch
        negations = {"not", "never", "cannot", "did not", "false", "no"}
        evidence_has_neg = any(n in e_lower for n in negations)
        claim_has_neg = any(n in c_lower for n in negations)

        if len(shared_words) >= 3 and (evidence_has_neg != claim_has_neg):
            return True, 0.85

        return False, 0.0

    def audit_answer(
        self,
        answer: str,
        evidence_list: List[FusedEvidence],
        support_threshold: float = 0.20,
        uncertain_threshold: float = 0.15
    ) -> ClaimAuditResult:
        """
        Audits all claims in the answer against the retrieved evidence list.
        """
        claims = self.extract_claims(answer)
        if not claims:
            return ClaimAuditResult(
                total_claims=0,
                supported_claims=0,
                unsupported_claims=0,
                contradicted_claims=0,
                uncertain_claims=0,
                claim_support_rate=0.0,
                unsupported_claim_rate=0.0,
                contradiction_rate=0.0,
                evidence_coverage=0.0,
                hallucination_rate=0.0,
                claims=[]
            )

        claim_records: List[ClaimRecord] = []
        supported_count = 0
        unsupported_count = 0
        contradicted_count = 0
        uncertain_count = 0

        for idx, claim in enumerate(claims):
            if not evidence_list:
                rec = ClaimRecord(
                    claim_index=idx,
                    claim_text=claim,
                    status="UNSUPPORTED",
                    evidence_ids=[],
                    evidence_sources=[],
                    best_evidence_text="",
                    support_score=0.0,
                    contradiction_score=0.0
                )
                unsupported_count += 1
                claim_records.append(rec)
                continue

            claim_vec = self.embedding_model.embed_text(claim)
            best_score = -1.0
            best_ev: Optional[FusedEvidence] = None
            max_contradiction = 0.0

            for ev in evidence_list:
                ev_vec = self.embedding_model.embed_text(ev.text)
                sim = self.embedding_model.cosine_similarity(claim_vec, ev_vec)
                if sim > best_score:
                    best_score = sim
                    best_ev = ev

                is_contra, c_score = self.detect_contradiction(claim, ev.text)
                if is_contra and c_score > max_contradiction:
                    max_contradiction = c_score

            # Classify status
            if max_contradiction > 0.5:
                status = "CONTRADICTED"
                contradicted_count += 1
            elif best_score >= support_threshold and best_ev is not None:
                status = "SUPPORTED"
                supported_count += 1
            elif best_score >= uncertain_threshold:
                status = "UNCERTAIN"
                uncertain_count += 1
            else:
                status = "UNSUPPORTED"
                unsupported_count += 1

            ev_ids = [str(best_ev.chunk_id)] if (best_ev and status == "SUPPORTED") else []
            ev_sources = [best_ev.source] if (best_ev and status == "SUPPORTED") else []
            best_text = best_ev.text if best_ev else ""

            claim_records.append(ClaimRecord(
                claim_index=idx,
                claim_text=claim,
                status=status,
                evidence_ids=ev_ids,
                evidence_sources=ev_sources,
                best_evidence_text=best_text,
                support_score=round(max(0.0, float(best_score)), 4),
                contradiction_score=round(max_contradiction, 4)
            ))

        total = len(claims)
        covered = sum(1 for c in claim_records if len(c.evidence_ids) > 0)

        return ClaimAuditResult(
            total_claims=total,
            supported_claims=supported_count,
            unsupported_claims=unsupported_count,
            contradicted_claims=contradicted_count,
            uncertain_claims=uncertain_count,
            claim_support_rate=round(supported_count / float(total), 4),
            unsupported_claim_rate=round(unsupported_count / float(total), 4),
            contradiction_rate=round(contradicted_count / float(total), 4),
            evidence_coverage=round(covered / float(total), 4),
            hallucination_rate=round(unsupported_count / float(total), 4),
            claims=claim_records
        )
