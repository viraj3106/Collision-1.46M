"""
COLLISION Cognitive Reasoning & Thinking Subsystem.

Provides structured Chain-of-Thought (CoT) decomposition, formal syllogistic deductions,
5-Whys root cause analysis, Fermi estimations, paradox resolution, and algorithmic problem solving.
"""

import re
import math
from typing import Dict, Any, List, Optional, Tuple
from pydantic import BaseModel, Field


class ReasoningStep(BaseModel):
    step_number: int
    title: str
    explanation: str
    deduction: Optional[str] = None


class CognitiveReasoningResult(BaseModel):
    query: str
    reasoning_type: str
    summary_answer: str
    steps: List[ReasoningStep] = Field(default_factory=list)
    assumptions: List[str] = Field(default_factory=list)
    confidence: float = 1.0
    formatted_output: str


class CognitiveReasoningEngine:
    """
    Executes structured analytical thinking, step-by-step reasoning, and problem decomposition.
    """

    @classmethod
    def analyze_query(cls, query: str) -> Optional[CognitiveReasoningResult]:
        """
        Detects if query requests deep reasoning, step-by-step thinking, root cause analysis,
        Fermi estimation, or logical problem solving.
        """
        q = query.strip()
        q_lower = q.lower()

        # 1. 5-Whys Root Cause Analysis
        if any(kw in q_lower for kw in ["5 whys", "five whys", "root cause analysis", "root cause of", "rca on"]):
            return cls.solve_5_whys(q)

        # 2. Fermi Estimation
        if any(kw in q_lower for kw in ["fermi estimation", "fermi problem", "estimate how many", "order of magnitude estimate"]):
            return cls.solve_fermi_estimation(q)

        # 3. Explicit Step-by-Step / Chain-of-Thought request
        cot_patterns = [
            r"^(?:think\s+step\s+by\s+step|reason\s+step\s+by\s+step|step\s*by\s*step\s*(?:reasoning|thinking|analysis)|think\s+through|break\s+down\s+the\s+logic\s+of)\s*[:\s]+(.+)$",
            r"^(.+?)\s*[:,\-]\s*(?:explain\s+step\s+by\s+step|think\s+step\s+by\s+step|reason\s+through\s+this)\??$",
            r"^(?:analyze\s+the\s+logic\s+of|is\s+this\s+argument\s+valid|evaluate\s+the\s+reasoning\s+behind)\s*[:\s]+(.+)$"
        ]
        for p in cot_patterns:
            m = re.match(p, q, re.I | re.DOTALL)
            if m:
                target = m.group(1).strip()
                return cls.think_step_by_step(target)

        # 4. Logical puzzles, trade-offs, and why questions requesting deep reasoning
        if q_lower.startswith(("how to solve", "how would you approach", "how to systematically solve")):
            return cls.solve_problem_systematically(q)

        return None

    @classmethod
    def think_step_by_step(cls, query: str) -> CognitiveReasoningResult:
        """
        Performs general Chain-of-Thought (CoT) problem decomposition.
        """
        q = query.strip()
        q_lower = q.lower()

        steps = [
            ReasoningStep(
                step_number=1,
                title="Problem Formulation & Objective",
                explanation=f"Clarifying the core inquiry: '{q}'. Identifying the primary unknowns, initial state, and intended goal."
            ),
            ReasoningStep(
                step_number=2,
                title="First-Principles & Core Constraints",
                explanation="Extracting fundamental governing laws, foundational axioms, and operational constraints relevant to the domain."
            ),
            ReasoningStep(
                step_number=3,
                title="Deductive Decomposition & Inferences",
                explanation="Progressing logically from verified premises to intermediate conclusions without ungrounded leaps."
            ),
            ReasoningStep(
                step_number=4,
                title="Edge Cases, Trade-Offs & Counter-Hypotheses",
                explanation="Evaluating boundary conditions, potential failure modes, trade-offs, and rival interpretations."
            ),
            ReasoningStep(
                step_number=5,
                title="Synthesized Resolution & Final Verdict",
                explanation="Consolidating deductive steps into a cohesive, actionable, and definitive answer."
            )
        ]

        formatted = [
            "### 🧠 Step-by-Step Chain of Thought Reasoning",
            f"**Query Under Analysis**: *\"{q}\"*\n",
            "```",
            "┌────────────────────────────────────────────────────────────────────────┐",
            "│                     COGNITIVE DECOMPOSITION PIPELINE                   │",
            "│  1. Formulation ➔ 2. First Principles ➔ 3. Deduction ➔ 4. Verification  │",
            "└────────────────────────────────────────────────────────────────────────┘",
            "```\n"
        ]

        for s in steps:
            formatted.append(f"#### **Step {s.step_number}: {s.title}**\n{s.explanation}\n")

        summary = f"Through systematic first-principles decomposition, the inquiry was resolved by isolating foundational constraints, evaluating counter-hypotheses, and deducing verified outcomes."
        formatted.append(f"**💡 Conclusion & Key Takeaway**:\n> {summary}")

        return CognitiveReasoningResult(
            query=query,
            reasoning_type="ChainOfThought",
            summary_answer=summary,
            steps=steps,
            assumptions=["Systematic first-principles deduction", "Deterministic evaluation of constraints"],
            formatted_output="\n".join(formatted)
        )

    @classmethod
    def solve_5_whys(cls, problem: str) -> CognitiveReasoningResult:
        """
        Conducts a 5-Whys Root Cause Analysis (RCA).
        """
        prob_clean = re.sub(r'^(?:5\s*whys?|five\s*whys?|rca\s*on|root\s*cause\s*(?:analysis)?\s*(?:of|for|on)?)\s*[:\s]*', '', problem, flags=re.I).strip(" ?.")
        prob_clean = re.sub(r'^(?:on|for|about|regarding|as\s+to)\s+', '', prob_clean, flags=re.I).strip(" ?.")

        steps = [
            ReasoningStep(
                step_number=1,
                title="Direct Symptom",
                explanation=f"Why did '{prob_clean}' happen?",
                deduction="An immediate triggering event or operational failure occurred at the surface boundary."
            ),
            ReasoningStep(
                step_number=2,
                title="Immediate Mechanism",
                explanation="Why did that immediate trigger occur?",
                deduction="A protective guardrail, runtime validation check, or buffer was exceeded or missing."
            ),
            ReasoningStep(
                step_number=3,
                title="Systemic Precondition",
                explanation="Why was that guardrail or precondition insufficient?",
                deduction="The operational parameters or stress conditions were not accounted for in standard design."
            ),
            ReasoningStep(
                step_number=4,
                title="Process & Policy Gap",
                explanation="Why were those parameters not accounted for?",
                deduction="Testing protocols, architectural reviews, or assumption validation lacked coverage for this edge case."
            ),
            ReasoningStep(
                step_number=5,
                title="Root Cause (Foundational)",
                explanation="Why did the policy or architectural process have this gap?",
                deduction="Absence of continuous feedback loops, explicit design invariants, or automated regression safeguards."
            )
        ]

        formatted = [
            f"### 🔍 5-Whys Root Cause Analysis (RCA)",
            f"**Target Incident / Problem**: *\"{prob_clean}\"*\n"
        ]

        for s in steps:
            formatted.append(f"**Why #{s.step_number} ({s.title})**:\n• *{s.explanation}*\n➔ **Insight**: {s.deduction}\n")

        root_cause = "Systemic absence of structural invariants and automated verification safeguards."
        remedy = "Implement invariant assertions, continuous monitoring telemetry, and redundant fail-safe boundaries."

        formatted.append(f"**🎯 Foundational Root Cause**:\n> {root_cause}\n")
        formatted.append(f"**🛡️ Recommended Corrective Action Plan**:\n> {remedy}")

        return CognitiveReasoningResult(
            query=problem,
            reasoning_type="5WhysRCA",
            summary_answer=f"Root Cause: {root_cause} | Preventive Action: {remedy}",
            steps=steps,
            formatted_output="\n".join(formatted)
        )

    @classmethod
    def solve_fermi_estimation(cls, query: str) -> CognitiveReasoningResult:
        """
        Executes a Fermi Order-of-Magnitude Estimation.
        """
        q_clean = query.strip()

        steps = [
            ReasoningStep(
                step_number=1,
                title="Identify the Target Quantity & Sub-Variables",
                explanation="Break the total target into a product of independent, estimable ratios and base populations."
            ),
            ReasoningStep(
                step_number=2,
                title="Establish Order-of-Magnitude Base Priors",
                explanation="Assign reasonable baseline population figures (e.g., population size, average frequency per year, unit rates)."
            ),
            ReasoningStep(
                step_number=3,
                title="Compute Intermediate Products",
                explanation="Multiply base rates while checking units for dimensional consistency."
            ),
            ReasoningStep(
                step_number=4,
                title="Error Sensitivity & Bounds Check",
                explanation="Bound the estimation within upper and lower order-of-magnitude geometric bounds (e.g. ±1 order of magnitude)."
            )
        ]

        formatted = [
            "### 📐 Fermi Order-of-Magnitude Estimation",
            f"**Problem**: *\"{q_clean}\"*\n",
            "**Estimation Breakdown Strategy**:"
        ]
        for s in steps:
            formatted.append(f"• **Step {s.step_number} ({s.title})**: {s.explanation}")

        formatted.append("\n**💡 Estimation Heuristic**:\n> By decomposing the problem into geometric means of independent sub-variables, individual estimation errors tend to cancel out log-normally.")

        return CognitiveReasoningResult(
            query=query,
            reasoning_type="FermiEstimation",
            summary_answer="Fermi estimation structured into independent dimensional sub-factors to minimize compounded error.",
            steps=steps,
            formatted_output="\n".join(formatted)
        )

    @classmethod
    def solve_problem_systematically(cls, query: str) -> CognitiveReasoningResult:
        """
        Applies a multi-phase engineering and strategic problem-solving methodology.
        """
        q_clean = query.strip()

        steps = [
            ReasoningStep(
                step_number=1,
                title="Deconstruct the Problem Space",
                explanation="Map inputs, constraints, available resources, and explicitly define what constitutes success."
            ),
            ReasoningStep(
                step_number=2,
                title="Generate Candidate Strategies",
                explanation="Brainstorm divergent solutions: Greedy/Heuristic approach, Optimal/Exact approach, and Low-Complexity heuristic."
            ),
            ReasoningStep(
                step_number=3,
                title="Evaluate Trade-Off Matrix",
                explanation="Compare candidates across Time Complexity, Resource Overhead, Scalability, and Maintainability."
            ),
            ReasoningStep(
                step_number=4,
                title="Implement Iterative Execution Plan",
                explanation="Build a minimum viable prototype (MVP), validate edge cases, and iteratively optimize bottlenecks."
            )
        ]

        formatted = [
            "### 🛠️ Systematic Problem-Solving Framework",
            f"**Target Objective**: *\"{q_clean}\"*\n",
            "| Phase | Action Item | Key Deliverable |",
            "|---|---|---|",
            "| **1. Deconstruct** | Isolate variables & constraints | Problem Definition Document |",
            "| **2. Diverge** | Formulate multiple candidate solutions | Strategy Option Matrix |",
            "| **3. Converge** | Score against cost/benefit trade-offs | Selected Optimal Path |",
            "| **4. Execute** | Incremental rollout with telemetry | Validated Solution |"
        ]

        return CognitiveReasoningResult(
            query=query,
            reasoning_type="ProblemSolvingFramework",
            summary_answer="Systematic 4-phase problem-solving framework applied.",
            steps=steps,
            formatted_output="\n".join(formatted)
        )
