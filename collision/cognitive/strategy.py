"""
COLLISION Cognitive Strategy, Persuasion & Rhetoric Subsystem.

Provides comprehensive deconstruction of Cialdini's 6 Principles of Influence,
negotiation frameworks (BATNA, Anchoring), cognitive bias identification,
logical fallacy detection, rhetorical analysis, and strategic communication.
"""

import re
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class StrategicAnalysisResult(BaseModel):
    query: str
    analysis_type: str
    summary: str
    principles_identified: List[str] = Field(default_factory=list)
    fallacies_or_biases: List[str] = Field(default_factory=list)
    actionable_tactics: List[str] = Field(default_factory=list)
    formatted_output: str


class CognitiveStrategyEngine:
    """
    Analyzes persuasive psychology, negotiation tactics, cognitive biases, and logical fallacies.
    """

    CIALDINI_PRINCIPLES = {
        "reciprocity": {
            "name": "Reciprocity",
            "mechanism": "People feel deeply obligated to return favors, concessions, or kind gestures.",
            "application": "Give upfront, unexpected, and personalized value before making an ask.",
            "counter_tactic": "Recognize the upfront favor as a calculated opener; accept genuine gifts without feeling bound to concede."
        },
        "scarcity": {
            "name": "Scarcity",
            "mechanism": "Perceived value surges when availability, time, or opportunity is limited.",
            "application": "Highlight exclusive insights, limited edition windows, or what they stand to lose.",
            "counter_tactic": "Pause and evaluate the intrinsic utility of the item independent of its availability constraint."
        },
        "authority": {
            "name": "Authority",
            "mechanism": "People defer to recognized expertise, credentials, titles, and perceived status.",
            "application": "Establish credible track records, cite authoritative data, and present professional polish.",
            "counter_tactic": "Verify the credential's actual relevance to the specific claim and check for consensus."
        },
        "consistency": {
            "name": "Commitment & Consistency",
            "mechanism": "Once people take a stand or make a public commitment, they strive to align subsequent actions with that identity.",
            "application": "Secure small initial agreements ('foot-in-the-door') before escalating to larger requests.",
            "counter_tactic": "Beware of escalating commitments; do not fear changing course when new data emerges."
        },
        "liking": {
            "name": "Liking & Rapport",
            "mechanism": "People are far more receptive to those they like, find similar to themselves, or receive genuine compliments from.",
            "application": "Find genuine common ground, mirror communication styles, and express sincere appreciation.",
            "counter_tactic": "Separate the messenger from the proposal; evaluate the terms on their objective merits."
        },
        "consensus": {
            "name": "Consensus / Social Proof",
            "mechanism": "In situations of uncertainty, people look to the actions and behaviors of peers to guide their own decisions.",
            "application": "Showcase user metrics, testimonials from comparable peers, and adoption momentum.",
            "counter_tactic": "Recognize that herd behavior can be manufactured or misguided; evaluate first-principles data."
        }
    }

    LOGICAL_FALLACIES = {
        "ad hominem": {
            "name": "Ad Hominem",
            "definition": "Attacking the character, motive, or background of the person making an argument instead of the argument itself.",
            "example": "\"You can't trust their code review because they are only a junior developer.\"",
            "remedy": "Refocus the discussion strictly on the objective logic, code, and factual premises."
        },
        "strawman": {
            "name": "Strawman",
            "definition": "Misrepresenting or exaggerating an opponent's position to make it easier to attack.",
            "example": "\"They want to add unit tests, which means they want to stop all feature delivery completely.\"",
            "remedy": "Restate your exact position clearly ('steelmanning') and ask the other party to address that specific formulation."
        },
        "false dilemma": {
            "name": "False Dilemma (Black-or-White)",
            "definition": "Presenting only two extreme alternatives when nuanced middle-ground options exist.",
            "example": "\"Either we rewrite the entire architecture in Rust, or the company fails.\"",
            "remedy": "Introduce hybrid options, phased migrations, and intermediate trade-offs."
        },
        "slippery slope": {
            "name": "Slippery Slope",
            "definition": "Asserting that a minor initial step will inevitably trigger a catastrophic chain of events without causal proof.",
            "example": "\"If we allow one remote work day, nobody will ever come into the office and productivity will collapse.\"",
            "remedy": "Demand empirical evidence for each transitional link in the alleged causal chain."
        },
        "appeal to emotion": {
            "name": "Appeal to Emotion",
            "definition": "Manipulating emotional responses (fear, pity, outrage) in place of valid logical reasoning.",
            "example": "\"Think of the disaster if this doesn't pass immediately!\"",
            "remedy": "Acknowledge the emotional sentiment while anchoring decisions back to verified metrics and causal logic."
        },
        "circular reasoning": {
            "name": "Circular Reasoning (Begging the Question)",
            "definition": "The conclusion of the argument is assumed in the premise.",
            "example": "\"Our algorithm is the most reliable because it produces flawless results, and we know it's flawless because of its reliability.\"",
            "remedy": "Identify the circular loop and require independent external validation criteria."
        }
    }

    COGNITIVE_BIASES = {
        "confirmation bias": {
            "name": "Confirmation Bias",
            "description": "The tendency to search for, interpret, and recall information in a way that confirms preexisting beliefs while ignoring contrary evidence.",
            "mitigation": "Actively seek disconfirming data and conduct 'red team' stress tests."
        },
        "sunk cost fallacy": {
            "name": "Sunk Cost Fallacy",
            "description": "Continuing an endeavor solely because of past unrecoverable investments of time, money, or effort.",
            "mitigation": "Base future decisions exclusively on prospective forward-looking marginal costs and expected returns."
        },
        "anchoring bias": {
            "name": "Anchoring Bias",
            "description": "Disproportionately relying on the first piece of information encountered (the 'anchor') when making decisions.",
            "mitigation": "Establish independent objective valuation models before reviewing initial offers."
        },
        "framing effect": {
            "name": "Framing Effect",
            "description": "Drawing different conclusions from the same information depending on how it is presented (e.g., '90% survival rate' vs '10% mortality rate').",
            "mitigation": "Re-frame problems in both gain and loss formats to assess objective symmetry."
        },
        "availability heuristic": {
            "name": "Availability Heuristic",
            "description": "Overestimating the likelihood of events that are easily recalled from memory (e.g., recent vivid news) over statistical base rates.",
            "mitigation": "Rely on actuarial base rates and comprehensive statistical distributions rather than recent anecdotes."
        }
    }

    @classmethod
    def analyze_query(cls, query: str) -> Optional[StrategicAnalysisResult]:
        """
        Detects if query relates to persuasion, negotiation, manipulation tactics,
        cognitive biases, or logical fallacy identification.
        """
        q = query.strip()
        q_lower = q.lower()

        # 1. Cialdini's Principles of Influence
        if any(kw in q_lower for kw in ["cialdini", "principles of influence", "principles of persuasion", "influence tactics", "how to persuade", "how to influence", "psychology of persuasion"]):
            return cls.explain_cialdini_principles(q)

        # 2. Negotiation Strategy & BATNA
        if any(kw in q_lower for kw in ["negotiate", "negotiation", "batna", "anchoring offer", "salary negotiation", "win-win framing", "bargaining tactics"]):
            return cls.explain_negotiation_strategy(q)

        # 3. Logical Fallacies
        if any(kw in q_lower for kw in ["logical fallacy", "logical fallacies", "fallacy", "ad hominem", "strawman", "false dilemma", "slippery slope"]):
            return cls.explain_fallacies(q)

        # 4. Cognitive Biases
        if any(kw in q_lower for kw in ["cognitive bias", "cognitive biases", "sunk cost", "confirmation bias", "anchoring bias", "framing effect"]):
            return cls.explain_biases(q)

        # 5. Rhetoric & Persuasive Framing
        if any(kw in q_lower for kw in ["how to frame", "persuasive framing", "manipulation tactics", "psychological framing", "dark patterns"]):
            return cls.explain_persuasive_framing(q)

        return None

    @classmethod
    def explain_cialdini_principles(cls, query: str) -> StrategicAnalysisResult:
        """
        Explains Dr. Robert Cialdini's 6 Core Principles of Persuasion and Ethical Influence.
        """
        formatted = [
            "### 🎯 The 6 Core Principles of Persuasion (Robert Cialdini)",
            "Persuasion science identifies six fundamental psychological levers that drive human compliance, consensus, and decision-making:\n"
        ]

        principles = []
        for key, p in cls.CIALDINI_PRINCIPLES.items():
            principles.append(p["name"])
            formatted.append(f"#### **{len(principles)}. {p['name']}**")
            formatted.append(f"• **Psychological Mechanism**: {p['mechanism']}")
            formatted.append(f"• **Strategic Application**: {p['application']}")
            formatted.append(f"• **Defense & Counter-Tactic**: *{p['counter_tactic']}*\n")

        formatted.append("```")
        formatted.append("┌────────────────────────────────────────────────────────────────────────┐")
        formatted.append("│                     ETHICAL INFLUENCE MATRIX                           │")
        formatted.append("│  Reciprocity ➔ Commitment ➔ Social Proof ➔ Authority ➔ Liking ➔ Scarcity │")
        formatted.append("└────────────────────────────────────────────────────────────────────────┘")
        formatted.append("```\n")
        formatted.append("**💡 Key Takeaway**:\n> True strategic persuasion aligns mutual incentives ethically rather than through coercive manipulation.")

        return StrategicAnalysisResult(
            query=query,
            analysis_type="CialdiniInfluence",
            summary="Overview of Cialdini's 6 psychological principles of persuasion and ethical influence.",
            principles_identified=principles,
            actionable_tactics=["Offer upfront value", "Secure incremental commitments", "Demonstrate verified peer proof"],
            formatted_output="\n".join(formatted)
        )

    @classmethod
    def explain_negotiation_strategy(cls, query: str) -> StrategicAnalysisResult:
        """
        Provides tactical guidance on principled negotiation (Harvard Negotiation Project).
        """
        formatted = [
            "### 🤝 Strategic Negotiation & Tactical Influence Framework",
            "Principled negotiation separates the people from the problem and focuses on underlying interests rather than rigid positions:\n",
            "#### **1. Establish Your BATNA (Best Alternative to a Negotiated Agreement)**",
            "• **Concept**: Your BATNA is your greatest source of leverage. A strong walk-away alternative prevents accepting unfavorable terms.",
            "• **Tactic**: Deeply develop your alternative options *before* entering the negotiation room.\n",
            "#### **2. Strategic Anchoring & First Offers**",
            "• **Concept**: The initial credible numeric anchor heavily pulls the final settlement range.",
            "• **Tactic**: Make the first offer if you have strong market data; anchor with a precise, researched number rather than a round figure.\n",
            "#### **3. Value Creation vs Value Claiming (Integrative Framing)**",
            "• **Concept**: Expanding the pie by trading low-cost/high-value concessions across non-monetary dimensions (timeline, scope, equity, flexibility).",
            "• **Tactic**: Never make a unilateral concession; always use conditional phrasing: *\"If we can agree on X, then I can adjust Y.\"*\n",
            "#### **4. Cognitive Mirroring & Calibrated Questions (Chris Voss)**",
            "• **Concept**: Using \"How\" and \"What\" open-ended questions to allow the counterparty to solve your constraints.",
            "• **Tactic**: Use phrasing like *\"How am I supposed to do that?\"* to politely push back without generating hostility.\n",
            "**🎯 Tactical Checklist**:",
            "1. Clarify target, aspiration, and reservation limits.",
            "2. Identify the counterparty's core psychological drivers and institutional constraints.",
            "3. Trade concessions across multi-issue packages rather than single-point stalemates."
        ]

        return StrategicAnalysisResult(
            query=query,
            analysis_type="NegotiationStrategy",
            summary="Principled negotiation framework covering BATNA leverage, anchoring, integrative trade-offs, and calibrated questioning.",
            principles_identified=["BATNA Leverage", "Precise Anchoring", "Integrative Trade-offs", "Calibrated Questions"],
            actionable_tactics=["Strengthen walk-away alternative", "Anchor with precision", "Use conditional concessions"],
            formatted_output="\n".join(formatted)
        )

    @classmethod
    def explain_fallacies(cls, query: str) -> StrategicAnalysisResult:
        """
        Explains common logical fallacies and how to detect & counter them.
        """
        formatted = [
            "### 🛡️ Logical Fallacy Identification & Deconstruction",
            "A logical fallacy is a flaw in reasoning that invalidates an argument regardless of how persuasive it sounds:\n"
        ]

        fallacies_list = []
        for k, f in cls.LOGICAL_FALLACIES.items():
            fallacies_list.append(f["name"])
            formatted.append(f"#### **• {f['name']}**")
            formatted.append(f"  * **Definition**: {f['definition']}")
            formatted.append(f"  * **Example**: {f['example']}")
            formatted.append(f"  * **Remedy**: *{f['remedy']}*\n")

        formatted.append("**💡 Critical Thinking Rule**:\n> Always evaluate the validity of premises and causal inferences independently of rhetorical delivery.")

        return StrategicAnalysisResult(
            query=query,
            analysis_type="LogicalFallacies",
            summary="Deconstruction of major informal logical fallacies with detection criteria and remedies.",
            fallacies_or_biases=fallacies_list,
            actionable_tactics=["Demand empirical causal links", "Refocus ad hominems back to evidence", "Steelman opponent claims"],
            formatted_output="\n".join(formatted)
        )

    @classmethod
    def explain_biases(cls, query: str) -> StrategicAnalysisResult:
        """
        Explains key cognitive biases and systematic heuristics that skew decision-making.
        """
        formatted = [
            "### 🧠 Cognitive Biases & Systematic Decision Pitfalls",
            "Cognitive biases are systematic deviations from normative rationality, rooted in evolutionary mental shortcuts (heuristics):\n"
        ]

        biases_list = []
        for k, b in cls.COGNITIVE_BIASES.items():
            biases_list.append(b["name"])
            formatted.append(f"#### **• {b['name']}**")
            formatted.append(f"  * **Mechanics**: {b['description']}")
            formatted.append(f"  * **Mitigation Protocol**: *{b['mitigation']}*\n")

        formatted.append("**💡 Strategic Insight**:\n> Rational decision systems implement institutional guardrails (checklists, pre-mortems, red teams) rather than relying purely on individual willpower.")

        return StrategicAnalysisResult(
            query=query,
            analysis_type="CognitiveBiases",
            summary="Catalog of pervasive cognitive biases and procedural debiasing protocols.",
            fallacies_or_biases=biases_list,
            actionable_tactics=["Conduct project pre-mortems", "Separate marginal future costs from sunk costs", "Seek disconfirming data"],
            formatted_output="\n".join(formatted)
        )

    @classmethod
    def explain_persuasive_framing(cls, query: str) -> StrategicAnalysisResult:
        """
        Explains rhetorical framing, message architecture, and psychological structuring.
        """
        formatted = [
            "### 🎨 Persuasive Framing & Strategic Message Architecture",
            "Framing determines the cognitive lens through which an audience interprets information:\n",
            "#### **1. Gain vs Loss Framing (Prospect Theory)**",
            "• **Principle**: Losses loom approximately 2x larger than equivalent gains (loss aversion).",
            "• **Application**: Frame prevention behaviors around avoiding unnecessary losses, and innovation around upside gains.\n",
            "#### **2. Identity-Based Alignment**",
            "• **Principle**: People act to maintain alignment with their self-identity (e.g. \"As innovators...\").",
            "• **Application**: Anchor proposals to the audience's aspirational values and professional standards.\n",
            "#### **3. The 'Because' Heuristic (Ellen Langer)**",
            "• **Principle**: Providing an explicit reason increases compliance by up to 50%+, even with simple justifications.",
            "• **Application**: Always substantiate requests with a concise causal rationale.\n",
            "#### **4. Counter-Manipulation & Dark Pattern Defense**",
            "• **Principle**: Spotting artificial urgency (countdowns), guilt induction, and loaded questions.",
            "• **Application**: Enforce a mandatory cooling-off period before committing under high-pressure framing."
        ]

        return StrategicAnalysisResult(
            query=query,
            analysis_type="PersuasiveFraming",
            summary="Strategic overview of cognitive framing, loss aversion mechanics, and defense against coercive manipulation.",
            principles_identified=["Loss Aversion Framing", "Identity Alignment", "Causal Rationale Heuristic"],
            actionable_tactics=["Enforce cooling-off periods", "Re-frame in dual gain/loss perspectives"],
            formatted_output="\n".join(formatted)
        )
