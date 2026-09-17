"""
COLLISION Phase 98 — Grounded Synthesis & Verified Release Module.
"""

from collision.grounding.schemas import (
    AnswerType,
    ExtractedFact,
    GroundedSynthesisResult
)
from collision.grounding.extractor import EvidenceExtractor
from collision.grounding.fallback import ExtractionFallbackHandler
from collision.grounding.synthesizer import GroundedSynthesizer
from collision.grounding.policy import FinalAnswerPolicy
from collision.grounding.engine import GroundedSynthesisEngine

__all__ = [
    "AnswerType",
    "ExtractedFact",
    "GroundedSynthesisResult",
    "EvidenceExtractor",
    "ExtractionFallbackHandler",
    "GroundedSynthesizer",
    "FinalAnswerPolicy",
    "GroundedSynthesisEngine"
]
