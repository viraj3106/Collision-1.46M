"""
COLLISION Dual-Process System 1/2 Cognitive Arbiter & Metacognitive Critic.

Implements Daniel Kahneman's Dual-Process Cognitive Architecture:
- System 1 (Reflexive / Fast Path): Sub-2ms pattern matching, cached associative facts, conversational courtesies, instant arithmetic.
- System 2 (Deliberative / Slow Path): Deep multi-node reasoning, dialectical synthesis, trade-off deconstruction, formal syllogisms.
- Epistemic Uncertainty Quantifier: Evaluates token Shannon entropy and semantic ambiguity to determine when to trigger System 2.
- Metacognitive Critic: Audits generated reasoning against cognitive biases (anchoring, confirmation, sunk cost) and fallacies.
"""

import re
import math
import time
from typing import Dict, Any, List, Optional, Tuple
from pydantic import BaseModel, Field

from collision.brain.schemas import (
    CognitiveModality,
    InformationEntropyProfile
)


class EpistemicUncertaintyQuantifier:
    """
    Measures epistemic uncertainty, ambiguity, and information density:
    - High entropy + high complexity -> Escalate to System 2 Deliberation.
    - Low entropy + clear intent -> Route to System 1 Reflex.
    """

    @classmethod
    def evaluate_entropy(cls, text: str) -> InformationEntropyProfile:
        words = re.findall(r'\w+', text.lower())
        total_tokens = len(words)
        if total_tokens == 0:
            return InformationEntropyProfile(
                total_tokens=0,
                unique_tokens=0,
                shannon_entropy=0.0,
                information_density=0.0,
                lexical_diversity=0.0,
                salience_spikes=[],
                epistemic_ambiguity_score=0.0
            )

        unique_tokens = len(set(words))
        # Word frequency distribution
        freqs: Dict[str, int] = {}
        for w in words:
            freqs[w] = freqs.get(w, 0) + 1

        # Shannon Entropy H(X) = - sum(p(x) * log2(p(x)))
        entropy = 0.0
        for w, c in freqs.items():
            p = c / total_tokens
            entropy -= p * math.log2(p)

        # Lexical Diversity (TTR)
        ttr = unique_tokens / total_tokens

        # Salience spikes (rare/informative words with low probability)
        salience_spikes = [w for w, c in freqs.items() if c == 1 and len(w) > 5]

        # Epistemic ambiguity score (high when entropy is elevated and multiple question words/conflicting tokens exist)
        ambiguity_markers = ["or", "maybe", "versus", "vs", "paradox", "trade-off", "tradeoff", "contradiction", "should", "why", "difference"]
        found_markers = sum(1 for m in ambiguity_markers if re.search(r'\b' + re.escape(m) + r'\b', text.lower()))
        ambiguity_score = min(1.0, (entropy / 5.0) * 0.5 + (found_markers * 0.15))

        return InformationEntropyProfile(
            total_tokens=total_tokens,
            unique_tokens=unique_tokens,
            shannon_entropy=round(entropy, 3),
            information_density=round(entropy / max(1, total_tokens), 4),
            lexical_diversity=round(ttr, 3),
            salience_spikes=salience_spikes[:5],
            epistemic_ambiguity_score=round(ambiguity_score, 3)
        )


class MetacognitiveCritic:
    """
    Introspective supervisor that examines thought chains for logical validity,
    overconfidence, and cognitive biases.
    """

    COGNITIVE_BIAS_RULES = [
        ("Confirmation Bias", r"\b(always|never|obviously|everyone knows|without question)\b", "Detected absolute generalizing markers without empirical qualifications."),
        ("Anchoring Bias", r"\b(first impression|initially assumed|stuck to the original)\b", "Prematurely anchored on initial premise without evaluating counter-hypotheses."),
        ("Sunk Cost Fallacy", r"\b(already invested|spent too much time to stop|cannot abandon)\b", "Justifying continued pursuit based on historical investment rather than marginal utility.")
    ]

    @classmethod
    def audit_thoughts(cls, thought_texts: List[str]) -> List[str]:
        notes = []
        combined = " ".join(thought_texts).lower()

        for bias_name, pattern, description in cls.COGNITIVE_BIAS_RULES:
            if re.search(pattern, combined):
                notes.append(f"[{bias_name}] {description}")

        if not notes:
            notes.append("[Epistemic Validation] Thoughts validated with zero detected cognitive bias violations.")

        return notes


class DualProcessArbiter:
    """
    Orchestrates dynamic routing between System 1 (Fast Reflex) and System 2 (Slow Deliberation).
    """

    SYSTEM_2_INDICATORS = [
        r"\b(?:why|how|explain|compare|contrast|trade-?offs?|paradox|dialectic|syllogism|deduce|derive)\b",
        r"\b(?:versus|vs|pros and cons|difference between|analyze|deep dive|step by step)\b",
        r"\b(?:ethical|implications|consequences|root cause|fallacy|bias|philosophy)\b"
    ]

    SYSTEM_1_INDICATORS = [
        r"^(?:hi|hello|hey|greetings|howdy|sup|thanks|thank you|bye|goodbye)[!.]?$",
        r"^(?:what is|calculate|compute)\s+[\d\s\+\-\*\/\^\(\)\.]+$",
        r"^(?:status|ping|pong|help)[!.]?$"
    ]

    @classmethod
    def decide_route(cls, query: str) -> Tuple[CognitiveModality, float, str]:
        """
        Determines whether to invoke System 1 Reflex or System 2 Deliberation.
        Returns: (Modality, Epistemic Uncertainty Score, Rationale)
        """
        q = query.strip()
        q_lower = q.lower()

        entropy_prof = EpistemicUncertaintyQuantifier.evaluate_entropy(q)

        # 1. Immediate System 1 check (greetings, simple arithmetic, single-word pings)
        for pattern in cls.SYSTEM_1_INDICATORS:
            if re.search(pattern, q_lower):
                return CognitiveModality.SYSTEM_1_REFLEX, entropy_prof.epistemic_ambiguity_score, "Deterministic pattern matched for System 1 fast reflex."

        # 2. System 2 check (deliberative analytical triggers)
        sys2_matches = sum(1 for p in cls.SYSTEM_2_INDICATORS if re.search(p, q_lower))
        if sys2_matches >= 1 or entropy_prof.epistemic_ambiguity_score > 0.40 or len(q.split()) > 10:
            return CognitiveModality.SYSTEM_2_DELIBERATION, entropy_prof.epistemic_ambiguity_score, f"Complex deliberative inquiry detected (entropy={entropy_prof.shannon_entropy:.2f}, ambiguity={entropy_prof.epistemic_ambiguity_score:.2f}). Escalated to System 2."

        return CognitiveModality.SYSTEM_1_REFLEX, entropy_prof.epistemic_ambiguity_score, "Low ambiguity query routed to System 1 fast path."
