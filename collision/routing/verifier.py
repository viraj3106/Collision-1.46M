import re
from typing import List, Tuple, Dict, Any, Optional
from collision.routing.schemas import FusedEvidence, ClaimVerification, VerificationResult
from collision.rag.embeddings import LocalEmbeddingModel

class GroundingVerifier:
    """
    Claim-level grounding verifier and citation checker.
    Validates that model outputs are strictly supported by retrieved evidence,
    detects hallucinations, identifies factual contradictions, and detects conflicting sources.
    """

    def __init__(self, embedding_model: Optional[LocalEmbeddingModel] = None):
        self.embedding_model = embedding_model if embedding_model is not None else LocalEmbeddingModel()

    def _split_into_claims(self, text: str) -> List[str]:
        """Splits answer text into discrete sentences/claims."""
        if not text:
            return []
        raw_sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        claims = [s.strip() for s in raw_sentences if len(s.strip()) > 10]
        if not claims and text.strip():
            claims = [text.strip()]
        return claims

    def _check_contradiction(self, claim: str, evidence_text: str) -> bool:
        """
        Detects genuine factual contradictions between two statements on the exact same attribute.
        Avoids false positives across distinct historical versions or orthogonal metrics.
        """
        c_lower = claim.lower()
        e_lower = evidence_text.lower()

        # 1. Direct attribute-specific numerical contradiction patterns
        attr_patterns = [
            r"(\d+)\s*(?:transformer\s+)?layers\b",
            r"(\d+)\s*attention\s+heads\b",
            r"(?:d_model|embedding\s+dimension)\s*(?:of\s*)?(\d+)\b",
            r"(?:d_ff|feedforward\s+dimension)\s*(?:of\s*)?(\d+)\b",
            r"(?:parameter\s+count|parameters)\s*(?:is|of)?\s*([\d,]+)\b",
            r"(?:chunk\s+size)\s*(?:of|is)?\s*(\d+)\b",
            r"(?:chunk\s+overlap)\s*(?:of|is)?\s*(\d+)\b",
            r"(\d+)[-\s]*(?:processor\s+|cpu\s+)?cores?\b",
            r"(\d+)\s*(?:gb|mb)\s*ram\b",
            r"(?:release\s+date|date)\s*(?:was|is)?\s*([a-z0-9\s]+?)(?=[.,;]|$)",
            r"(?:created|first\s+released|invented|developed)\s*(?:in|on)?\s*(\b\d{4}\b)"
        ]

        for p in attr_patterns:
            c_match = re.search(p, c_lower)
            e_match = re.search(p, e_lower)
            if c_match and e_match:
                c_val = c_match.group(1).replace(",", "")
                e_val = e_match.group(1).replace(",", "")
                if c_val != e_val:
                    # Check that both passages refer to the same subject
                    c_words = set(re.findall(r"\b[a-z]{4,}\b", c_lower))
                    e_words = set(re.findall(r"\b[a-z]{4,}\b", e_lower))
                    if len(c_words.intersection(e_words)) >= 1:
                        return True

        # 2. Direct negation contradiction on identical subject-predicate
        c_words = set(re.findall(r"\b[a-z]{4,}\b", c_lower))
        e_words = set(re.findall(r"\b[a-z]{4,}\b", e_lower))
        shared = c_words.intersection(e_words)

        neg_terms = {"not", "never", "cannot", "false", "disproven", "fake", "incorrect"}
        c_neg = any(n in c_lower for n in neg_terms)
        e_neg = any(n in e_lower for n in neg_terms)

        if len(shared) >= 3 and (c_neg != e_neg):
            # Check if negative particle applies directly to shared concept
            return True

        return False

    def verify(
        self,
        answer: str,
        evidence_list: List[FusedEvidence],
        similarity_threshold: float = 0.12
    ) -> VerificationResult:
        """
        Verifies answer against fused evidence, returning a structured VerificationResult.
        """
        if not answer or not answer.strip():
            return VerificationResult(
                supported=False,
                status="INSUFFICIENT_INFORMATION",
                score=0.0,
                confidence=0.0
            )

        clean_answer = answer.strip()
        if not evidence_list:
            # Model-only or unsupported
            return VerificationResult(
                supported=False,
                status="UNSUPPORTED",
                score=0.0,
                unsupported_claims=[clean_answer],
                confidence=0.0
            )

        claims = self._split_into_claims(clean_answer)
        claim_results: List[ClaimVerification] = []
        unsupported_claims: List[str] = []
        contradicted_claims: List[str] = []
        evidence_used: List[str] = []

        # Check for multi-source conflict in evidence list itself
        has_conflicting_evidence = False
        if len(evidence_list) >= 2:
            for i in range(len(evidence_list)):
                for j in range(i + 1, len(evidence_list)):
                    if self._check_contradiction(evidence_list[i].text, evidence_list[j].text):
                        has_conflicting_evidence = True
                        break

        total_similarity = 0.0

        for claim in claims:
            claim_vec = self.embedding_model.embed_text(claim)
            best_score = -1.0
            best_evidence: Optional[FusedEvidence] = None

            for ev in evidence_list:
                ev_vec = self.embedding_model.embed_text(ev.text)
                score = self.embedding_model.cosine_similarity(claim_vec, ev_vec)
                if score > best_score:
                    best_score = score
                    best_evidence = ev

            if best_evidence is not None and self._check_contradiction(claim, best_evidence.text):
                claim_results.append(ClaimVerification(
                    claim_text=claim,
                    status="CONTRADICTED",
                    supporting_evidence=[best_evidence.source],
                    similarity_score=best_score
                ))
                contradicted_claims.append(claim)
            elif best_score >= similarity_threshold and best_evidence is not None:
                claim_results.append(ClaimVerification(
                    claim_text=claim,
                    status="SUPPORTED",
                    supporting_evidence=[best_evidence.source],
                    similarity_score=best_score
                ))
                if best_evidence.source not in evidence_used:
                    evidence_used.append(best_evidence.source)
                total_similarity += best_score
            else:
                claim_results.append(ClaimVerification(
                    claim_text=claim,
                    status="UNSUPPORTED",
                    supporting_evidence=[],
                    similarity_score=best_score
                ))
                unsupported_claims.append(claim)

        # Compute overall status
        num_claims = max(1, len(claims))
        supported_count = sum(1 for c in claim_results if c.status == "SUPPORTED")
        support_ratio = supported_count / float(num_claims)

        if has_conflicting_evidence:
            final_status = "CONFLICTING_EVIDENCE"
            is_supported = False
        elif contradicted_claims:
            final_status = "CONTRADICTED"
            is_supported = False
        elif support_ratio >= 0.75:
            final_status = "ANSWER"
            is_supported = True
        elif support_ratio > 0.0:
            final_status = "UNCERTAIN"
            is_supported = False
        else:
            final_status = "INSUFFICIENT_INFORMATION"
            is_supported = False

        avg_score = total_similarity / max(1, supported_count)

        return VerificationResult(
            supported=is_supported,
            status=final_status,
            score=round(avg_score, 4),
            claims=claim_results,
            unsupported_claims=unsupported_claims,
            contradicted_claims=contradicted_claims,
            evidence_used=evidence_used,
            has_conflicting_evidence=has_conflicting_evidence,
            confidence=round(support_ratio, 4)
        )
