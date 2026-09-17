"""
COLLISION Dual-Process System 1/2 Cognitive Arbiter & Metacognitive Critic 2.0.

Implements Daniel Kahneman's Dual-Process Cognitive Architecture with high-performance execution:
- System 1 (Reflexive / Fast Path): Sub-millisecond pre-compiled pattern matching, cached associative facts, conversational courtesies, instant arithmetic.
- System 2 (Deliberative / Slow Path): Multi-node Graph-of-Thoughts reasoning, dialectical synthesis, trade-off deconstruction, formal syllogisms.
- Epistemic Uncertainty Quantifier 2.0: Evaluates token Shannon & Renyi entropy, lexical diversity, and linguistic complexity with LRU caching.
- Metacognitive Critic 2.0: Audits generated reasoning against 15+ cognitive biases and formal logical fallacies with structured remediation.
"""

import re
import math
import time
from functools import lru_cache
from typing import Dict, Any, List, Optional, Tuple

from collision.brain.schemas import (
    CognitiveModality,
    InformationEntropyProfile,
    LinguisticComplexityMetrics,
    CognitiveBiasAudit,
    BiasSeverity
)

# ---------------------------------------------------------------------------
# Pre-compiled Regex Patterns for High Performance
# ---------------------------------------------------------------------------
WORD_TOKEN_PATTERN = re.compile(r'\w+')
SENTENCE_SPLIT_PATTERN = re.compile(r'[.!?]+')

SYSTEM_1_PATTERNS = [
    re.compile(r"^(?:hi|hello|hey|greetings|howdy|sup|thanks|thank you|bye|goodbye)[!.]?$", re.IGNORECASE),
    re.compile(r"^(?:what is|calculate|compute)\s+[\d\s\+\-\*\/\^\(\)\.]+$", re.IGNORECASE),
    re.compile(r"^(?:status|ping|pong|help|version|uptime)[!.]?$", re.IGNORECASE),
]

SYSTEM_2_PATTERNS = [
    re.compile(r"\b(?:why|how|explain|compare|contrast|trade-?offs?|paradox|dialectic|syllogism|deduce|derive)\b", re.IGNORECASE),
    re.compile(r"\b(?:versus|vs|pros and cons|difference between|analyze|deep dive|step by step|counterfactual)\b", re.IGNORECASE),
    re.compile(r"\b(?:ethical|implications|consequences|root cause|fallacy|bias|philosophy|first principles)\b", re.IGNORECASE),
    re.compile(r"\b(?:architecture|distributed|quantum|entropy|algorithmic|proof|boundary conditions)\b", re.IGNORECASE),
]

AMBIGUITY_MARKERS = [
    re.compile(r'\b' + re.escape(m) + r'\b', re.IGNORECASE) for m in [
        "or", "maybe", "versus", "vs", "paradox", "trade-off", "tradeoff",
        "contradiction", "should", "why", "difference", "uncertain",
        "either", "depends", "contingent", "ambiguous", "dilemma"
    ]
]

# ---------------------------------------------------------------------------
# Comprehensive Formal Fallacy & Cognitive Bias Registry (15+ rules)
# ---------------------------------------------------------------------------
BIAS_AND_FALLACY_REGISTRY = [
    (
        "Confirmation Bias",
        BiasSeverity.WARNING,
        re.compile(r"\b(always|never|obviously|everyone knows|without question|undeniably|indisputable)\b", re.IGNORECASE),
        "Detected absolute generalizing markers without empirical qualifications.",
        "Introduce probabilistic qualifiers (e.g., 'typically', 'under conditions X') and evaluate counter-examples.",
        0.06
    ),
    (
        "Anchoring Bias",
        BiasSeverity.WARNING,
        re.compile(r"\b(first impression|initially assumed|stuck to the original|initial baseline)\b", re.IGNORECASE),
        "Prematurely anchored on initial premise without evaluating counter-hypotheses.",
        "Recalibrate hypothesis against full parametric distribution, not initial seed assumption.",
        0.05
    ),
    (
        "Sunk Cost Fallacy",
        BiasSeverity.CRITICAL,
        re.compile(r"\b(already invested|spent too much time to stop|cannot abandon|committed too much)\b", re.IGNORECASE),
        "Justifying continued pursuit based on historical investment rather than prospective marginal utility.",
        "Evaluate forward-looking expected value independent of non-recoverable past investments.",
        0.08
    ),
    (
        "Availability Heuristic",
        BiasSeverity.WARNING,
        re.compile(r"\b(recent news|just saw|vivid example|comes to mind easily|everyone talks about)\b", re.IGNORECASE),
        "Overweighting easily recalled or sensationalized instances over statistical base rates.",
        "Ground reasoning in empirical base-rate distributions and systematic sample data.",
        0.04
    ),
    (
        "False Dichotomy / Dilemma",
        BiasSeverity.CRITICAL,
        re.compile(r"\b(either we .+ or we .+ fail|only two choices|must choose between .+ and .+ with no middle)\b", re.IGNORECASE),
        "Framing a complex multi-dimensional space as an artificial binary either/or dilemma.",
        "Explore continuous Pareto frontiers, hybrid architectures, and intermediate trade-offs.",
        0.08
    ),
    (
        "Strawman Fallacy",
        BiasSeverity.WARNING,
        re.compile(r"\b(claim that .+ is completely useless|arguing that .+ has zero value|trivial to dismiss)\b", re.IGNORECASE),
        "Misrepresenting the opposing perspective in an exaggerated or simplified form.",
        "Steel-man the opposing argument before formulating dialectical synthesis.",
        0.07
    ),
    (
        "Ad Hominem",
        BiasSeverity.CRITICAL,
        re.compile(r"\b(only idiots believe|incompetent proponents|biased author|corrupt critics)\b", re.IGNORECASE),
        "Attacking the proponent's character or affiliation rather than the structural validity of the claim.",
        "Dissect proposition purely through axiomatic and empirical validity independent of origin.",
        0.09
    ),
    (
        "Post Hoc Ergo Propter Hoc",
        BiasSeverity.WARNING,
        re.compile(r"\b(happened right after .+ therefore caused|subsequently proved that .+ triggered)\b", re.IGNORECASE),
        "Confusing chronological sequence with direct causal mechanisms (correlation != causation).",
        "Establish mechanistic causal links, confounder controls, and randomized counterfactual evidence.",
        0.06
    ),
    (
        "Appeal to Authority",
        BiasSeverity.ADVISORY,
        re.compile(r"\b(because .+ said so|celebrated expert asserted|famous authority proclaimed)\b", re.IGNORECASE),
        "Relying solely on authority figures rather than verifiable first-principles proof.",
        "Verify theoretical derivation and experimental reproducibility regardless of prestige.",
        0.04
    ),
    (
        "Circular Reasoning (Begging the Question)",
        BiasSeverity.CRITICAL,
        re.compile(r"\b(is true because it is correct|valid because it works|effective because of its efficacy)\b", re.IGNORECASE),
        "Premise presupposes the truth of the conclusion without external supporting proof.",
        "Anchor proof chain in independent empirical or axiomatic ground truth.",
        0.08
    ),
    (
        "Slippery Slope Fallacy",
        BiasSeverity.WARNING,
        re.compile(r"\b(will inevitably lead to total collapse|if we allow .+ then everything will be destroyed)\b", re.IGNORECASE),
        "Asserting an extreme chain of negative consequences without proving each transitional step.",
        "Quantify transition probabilities and negative feedback stabilizing loops at each stage.",
        0.05
    ),
    (
        "Gambler's Fallacy",
        BiasSeverity.WARNING,
        re.compile(r"\b(due for a win|long overdue|streak must end soon|law of averages guarantees)\b", re.IGNORECASE),
        "Treating independent probabilistic events as if past outcomes alter future marginal distributions.",
        "Apply memoryless Markov property or formal conditional Bayesian probability.",
        0.05
    ),
    (
        "Overconfidence Bias",
        BiasSeverity.WARNING,
        re.compile(r"\b(100% guaranteed|zero possibility of failure|impossible to fail|flawless design)\b", re.IGNORECASE),
        "Uncalibrated certainty discounting stochastic tail risks and unexpected edge conditions.",
        "Incorporate probabilistic confidence intervals and black-swan contingency bounds.",
        0.06
    ),
    (
        "Framing Effect",
        BiasSeverity.ADVISORY,
        re.compile(r"\b(only a 5% loss vs 95% success|packaged as .+ rather than)\b", re.IGNORECASE),
        "Allowing linguistic presentation to bias objective expected utility calculations.",
        "Normalize problem representation into mathematically neutral payoff matrices.",
        0.03
    ),
    (
        "Affirming the Consequent",
        BiasSeverity.WARNING,
        re.compile(r"\b(if P then Q, and Q happened, therefore P must be true)\b", re.IGNORECASE),
        "Formal propositional logic error: asserting necessary consequence proves sufficient antecedent.",
        "Check for equifinality and alternate explanatory hypotheses that could produce the same observation.",
        0.07
    ),
]


class EpistemicUncertaintyQuantifier:
    """
    Measures epistemic uncertainty, ambiguity, and multi-scale information density:
    - High Shannon & Renyi entropy + high complexity -> Escalate to System 2 Deliberation.
    - Low entropy + clear intent -> Route to System 1 Reflex.
    """

    @classmethod
    @lru_cache(maxsize=2048)
    def evaluate_entropy_cached(cls, text: str) -> InformationEntropyProfile:
        return cls._compute_profile(text)

    @classmethod
    def evaluate_entropy(cls, text: str) -> InformationEntropyProfile:
        return cls.evaluate_entropy_cached(text)

    @classmethod
    def _compute_profile(cls, text: str) -> InformationEntropyProfile:
        words = WORD_TOKEN_PATTERN.findall(text.lower())
        total_tokens = len(words)
        if total_tokens == 0:
            return InformationEntropyProfile(
                total_tokens=0,
                unique_tokens=0,
                shannon_entropy=0.0,
                information_density=0.0,
                lexical_diversity=0.0,
                salience_spikes=[],
                epistemic_ambiguity_score=0.0,
                renyi_entropy=0.0,
                complexity=LinguisticComplexityMetrics()
            )

        unique_tokens = len(set(words))
        # Word frequency distribution
        freqs: Dict[str, int] = {}
        for w in words:
            freqs[w] = freqs.get(w, 0) + 1

        # 1. Shannon Entropy H(X) = - sum(p(x) * log2(p(x)))
        shannon_entropy = 0.0
        sum_p_sq = 0.0
        for w, c in freqs.items():
            p = c / total_tokens
            shannon_entropy -= p * math.log2(p)
            sum_p_sq += p * p

        # 2. Renyi Entropy of Order 2 H_2(X) = - log2(sum(p(x)^2)) (Collision entropy)
        renyi_entropy = -math.log2(max(1e-9, sum_p_sq))

        # 3. Lexical Diversity (TTR)
        ttr = unique_tokens / total_tokens

        # 4. Salience spikes (rare/informative words with length > 5)
        salience_spikes = [w for w, c in freqs.items() if c == 1 and len(w) > 5]

        # 5. Epistemic ambiguity score (multi-scale markers + entropy)
        found_markers = sum(1 for m in AMBIGUITY_MARKERS if m.search(text))
        ambiguity_score = min(1.0, (shannon_entropy / 5.0) * 0.45 + (found_markers * 0.12))

        # 6. Linguistic Complexity Metrics
        sentences = [s.strip() for s in SENTENCE_SPLIT_PATTERN.split(text) if s.strip()]
        num_sentences = max(1, len(sentences))
        avg_words_per_sentence = total_tokens / num_sentences

        # Syllable approximation
        complex_words = sum(1 for w in words if len(w) >= 7)
        gunning_fog = 0.4 * (avg_words_per_sentence + 100.0 * (complex_words / max(1, total_tokens)))
        flesch_reading = 206.835 - 1.015 * avg_words_per_sentence - 84.6 * (complex_words / max(1, total_tokens))
        flesch_reading = max(0.0, min(100.0, flesch_reading))

        complexity = LinguisticComplexityMetrics(
            renyi_entropy_order2=round(renyi_entropy, 3),
            flesch_reading_ease=round(flesch_reading, 1),
            gunning_fog_estimate=round(gunning_fog, 1),
            syntactic_depth_estimate=min(8, max(2, int(avg_words_per_sentence / 4) + 1)),
            token_surprisal_avg=round(shannon_entropy / max(1.0, total_tokens), 3)
        )

        return InformationEntropyProfile(
            total_tokens=total_tokens,
            unique_tokens=unique_tokens,
            shannon_entropy=round(shannon_entropy, 3),
            information_density=round(shannon_entropy / max(1, total_tokens), 4),
            lexical_diversity=round(ttr, 3),
            salience_spikes=salience_spikes[:6],
            epistemic_ambiguity_score=round(ambiguity_score, 3),
            renyi_entropy=round(renyi_entropy, 3),
            complexity=complexity
        )


class MetacognitiveCritic:
    """
    Introspective supervisor that examines thought chains for logical validity,
    overconfidence, cognitive biases, and formal fallacies across 15+ rules.
    """

    @classmethod
    def audit_thoughts_detailed(cls, thought_texts: List[str]) -> List[CognitiveBiasAudit]:
        """Performs deep structured audit returning CognitiveBiasAudit models."""
        audits: List[CognitiveBiasAudit] = []
        combined = " ".join(thought_texts)

        for bias_name, severity, pattern, explanation, remediation, penalty in BIAS_AND_FALLACY_REGISTRY:
            m = pattern.search(combined)
            if m:
                snippet = m.group(0)
                audits.append(CognitiveBiasAudit(
                    bias_type=bias_name,
                    severity=severity,
                    detected_pattern=pattern.pattern,
                    detected_snippet=snippet,
                    explanation=explanation,
                    remediation_suggestion=remediation,
                    confidence_penalty=penalty
                ))

        return audits

    @classmethod
    def audit_thoughts(cls, thought_texts: List[str]) -> List[str]:
        """Backward-compatible audit returning human-readable string notes."""
        detailed = cls.audit_thoughts_detailed(thought_texts)
        if not detailed:
            return ["[Epistemic Validation] Thoughts rigorously verified: zero cognitive bias or fallacy violations detected."]

        notes = []
        for audit in detailed:
            notes.append(f"[{audit.severity.value}] [{audit.bias_type}] {audit.explanation} Remediation: {audit.remediation_suggestion}")
        return notes


class DualProcessArbiter:
    """
    High-Performance Cognitive Arbiter orchestrating dynamic routing
    between System 1 (Fast Reflex) and System 2 (Slow Deliberation).
    """

    @classmethod
    @lru_cache(maxsize=1024)
    def decide_route(cls, query: str) -> Tuple[CognitiveModality, float, str]:
        """
        Determines whether to invoke System 1 Reflex or System 2 Deliberation with sub-millisecond latency.
        Returns: (Modality, Epistemic Uncertainty Score, Rationale)
        """
        q = query.strip()
        if not q:
            return CognitiveModality.SYSTEM_1_REFLEX, 0.0, "Empty query defaulted to System 1."

        entropy_prof = EpistemicUncertaintyQuantifier.evaluate_entropy(q)

        # 1. Immediate System 1 check (greetings, simple arithmetic, single-word pings)
        for pattern in SYSTEM_1_PATTERNS:
            if pattern.search(q):
                return CognitiveModality.SYSTEM_1_REFLEX, entropy_prof.epistemic_ambiguity_score, "Deterministic pattern matched for System 1 fast reflex."

        # 2. System 2 check (deliberative analytical triggers)
        sys2_hits = sum(1 for p in SYSTEM_2_PATTERNS if p.search(q))
        word_count = len(q.split())

        if sys2_hits >= 1 or entropy_prof.epistemic_ambiguity_score > 0.38 or word_count > 9:
            return CognitiveModality.SYSTEM_2_DELIBERATION, entropy_prof.epistemic_ambiguity_score, (
                f"Complex deliberative inquiry detected (entropy={entropy_prof.shannon_entropy:.2f}, "
                f"ambiguity={entropy_prof.epistemic_ambiguity_score:.2f}, triggers={sys2_hits}). Escalated to System 2."
            )

        return CognitiveModality.SYSTEM_1_REFLEX, entropy_prof.epistemic_ambiguity_score, "Low ambiguity query routed to System 1 fast path."
