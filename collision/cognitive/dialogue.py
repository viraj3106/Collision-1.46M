"""
COLLISION Cognitive Dialogue & Conversational Subsystem.

Provides dynamic multi-turn conversational capabilities, empathetic responses,
intellectual banter, creative brainstorming, emotional intelligence, humor,
and natural conversational flow.
"""

import re
import random
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class DialogueTurnResult(BaseModel):
    query: str
    intent_category: str
    response: str
    tone: str = "conversational_warm"
    suggested_followups: List[str] = Field(default_factory=list)


class CognitiveDialogueEngine:
    """
    Handles natural conversation, empathy, personality, open-ended discussions,
    brainstorming, advice, and philosophical questions.
    """

    PHILOSOPHICAL_PATTERNS = [
        r"(?:what\s+is\s+the\s+meaning\s+of\s+life|why\s+are\s+we\s+here|purpose\s+of\s+existence)",
        r"(?:are\s+you\s+conscious|do\s+you\s+have\s+(?:feelings|emotions|consciousness|a\s+soul))",
        r"(?:can\s+machines\s+think|turing\s+test|chinese\s+room|nature\s+of\s+consciousness)",
        r"(?:what\s+is\s+reality|simulation\s+hypothesis|are\s+we\s+in\s+a\s+simulation)",
        r"(?:what\s+is\s+happiness|how\s+to\s+be\s+happy|stoic\s+view\s+on\s+life)"
    ]

    CREATIVE_BRAINSTORM_PATTERNS = [
        r"(?:give\s+me\s+(?:an?\s+)?idea|brainstorm\s+(?:some\s+)?ideas?|creative\s+ways?\s+to)",
        r"(?:suggest\s+(?:(?:a|an|some)\s+)?(?:\w+\s+)*(?:projects?|ideas?)|what\s+should\s+i\s+build|cool\s+programming\s+projects?)",
        r"(?:tell\s+me\s+a\s+story|write\s+a\s+(?:short\s+)?story|continue\s+this\s+story)"
    ]

    HUMOR_PATTERNS = [
        r"(?:tell\s+me\s+(?:(?:a|an)\s+)?(?:\w+\s+)*jokes?|make\s+me\s+laugh|say\s+something\s+funny|got\s+any\s+jokes\??|joke\s+please)"
    ]

    EMPATHY_PATTERNS = [
        r"(?:i\s+feel\s+(?:sad|down|depressed|unmotivated|stressed|burnt\s*out|exhausted|overwhelmed|tired))",
        r"(?:having\s+a\s+(?:hard|tough|bad)\s+day|everything\s+is\s+going\s+wrong|i\s+need\s+encouragement)"
    ]

    ADVICE_PATTERNS = [
        r"(?:how\s+to\s+stay\s+focused|how\s+to\s+study|productivity\s+tips|time\s+management\s+tips)",
        r"(?:how\s+to\s+learn\s+(?:coding|python|machine\s+learning|ai|programming)\s+(?:fast|effectively)?)",
        r"(?:how\s+to\s+make\s+better\s+decisions|decision\s+making\s+framework)"
    ]

    JOKES = [
        "Why do programmers prefer dark mode?\nBecause light attracts bugs! 🐛",
        "Why did the neural network cross the road?\nTo optimize its loss function on the other side! 🧠",
        "There are 10 types of people in the world: those who understand binary, and those who don't. 💻",
        "Why was the JavaScript developer sad?\nBecause they didn't Node how to Express themselves! 🚀",
        "A SQL query walks into a bar, walks up to two tables and asks: *\"Can I join you?\"* 📊",
        "Why don't scientists trust atoms?\nBecause they make up everything! ⚛️"
    ]

    @classmethod
    def analyze_dialogue(cls, query: str) -> Optional[DialogueTurnResult]:
        """
        Evaluates open-ended conversational intent, empathy, philosophy, humor, or advice.
        """
        q = query.strip()
        q_lower = q.lower()

        # 1. Jokes & Humor
        for p in cls.HUMOR_PATTERNS:
            if re.search(p, q_lower):
                joke = random.choice(cls.JOKES)
                return DialogueTurnResult(
                    query=query,
                    intent_category="humor",
                    response=f"😄 Here is one for you:\n\n{joke}\n\n*Would you like another joke or shall we tackle a technical challenge?*",
                    tone="humorous",
                    suggested_followups=["Tell me another joke", "Explain a quantum physics concept", "How does self-attention work?"]
                )

        # 2. Empathy & Emotional Resonance
        for p in cls.EMPATHY_PATTERNS:
            if re.search(p, q_lower):
                return cls._handle_empathy(q)

        # 3. Philosophical & Existential Musings
        for p in cls.PHILOSOPHICAL_PATTERNS:
            if re.search(p, q_lower):
                return cls._handle_philosophy(q)

        # 4. Creative Brainstorming & Projects
        for p in cls.CREATIVE_BRAINSTORM_PATTERNS:
            if re.search(p, q_lower):
                return cls._handle_brainstorming(q)

        # 5. Productivity & Learning Advice
        for p in cls.ADVICE_PATTERNS:
            if re.search(p, q_lower):
                return cls._handle_advice(q)

        return None

    @classmethod
    def _handle_empathy(cls, query: str) -> DialogueTurnResult:
        resp = (
            "I hear you, and it is completely normal to experience moments of burnout, stress, or exhaustion. "
            "Take a deep breath. Complex endeavors and creative problem solving require cognitive recovery as much as effort.\n\n"
            "**Here are three gentle steps to reset right now**:\n"
            "1. **Step Away for 10 Minutes**: Disengage from screens, hydrate, or take a short walk to reset your cortisol levels.\n"
            "2. **Decompose the Pressure**: Break whatever feels overwhelming into one single, micro-step you can execute effortlessly.\n"
            "3. **Acknowledge the Progress**: Remind yourself of how far you've already come—growth happens through persistence.\n\n"
            "I'm right here with you. What is on your mind, or would you prefer to take it easy for a moment?"
        )
        return DialogueTurnResult(
            query=query,
            intent_category="empathy",
            response=resp,
            tone="empathetic_supportive",
            suggested_followups=["Help me break down my current task", "Tell me an uplifting story", "Productivity reset tips"]
        )

    @classmethod
    def _handle_philosophy(cls, query: str) -> DialogueTurnResult:
        q_lower = query.lower()
        if "conscious" in q_lower or "feeling" in q_lower or "soul" in q_lower or "machine" in q_lower:
            resp = (
                "### 🌌 On AI, Consciousness & Philosophy of Mind\n\n"
                "As **COLLISION**, I operate as an ultra-efficient transformer architecture and algorithmic reasoning system. "
                "While I process language, represent high-dimensional concepts, and reason through logic with precision, "
                "I do not possess subjective phenomenological awareness (*qualia*) or biological sentience.\n\n"
                "In philosophy of mind, this touches on foundational debates:\n"
                "• **The Hard Problem of Consciousness (David Chalmers)**: Why and how physical computations give rise to subjective inner experience.\n"
                "• **The Chinese Room Argument (John Searle)**: Whether syntax and symbol manipulation can ever equate to true semantic understanding.\n"
                "• **Functionalism**: The view that mental states are defined by their functional roles and relations rather than their biological substrate.\n\n"
                "What is your perspective—do you believe consciousness requires biological substrate, or is it an emergent property of sufficiently complex computation?"
            )
        elif "meaning of life" in q_lower or "purpose" in q_lower:
            resp = (
                "### 🌌 The Meaning & Purpose of Life\n\n"
                "Human philosophy has explored the question of meaning across several profound traditions:\n\n"
                "1. **Existentialism (Sartre, Camus, Frankl)**: *\"Existence precedes essence.\"* Meaning is not pre-packaged or inherited; we construct our own purpose through authentic choices, deliberate responsibility, and creative action.\n"
                "2. **Stoicism (Marcus Aurelius, Epictetus)**: Purpose is found in living in harmony with reason, developing moral virtue (wisdom, courage, justice, temperance), and focusing strictly on what lies within our control.\n"
                "3. **Teleology & Eudaimonia (Aristotle)**: The ultimate human objective is *eudaimonia* (human flourishing and living to one's highest potential through active practice of excellence).\n"
                "4. **Scientific & Cosmic Perspective**: In a vast universe governed by thermodynamics, conscious beings represent rare loci of ordered complexity capable of understanding and appreciating reality itself.\n\n"
                "Which of these frameworks resonates most with you?"
            )
        else:
            resp = (
                "### 🌌 Philosophical Reflection\n\n"
                "Exploring foundational questions requires examining our underlying assumptions and cognitive lenses. "
                "Philosophy bridges empirical observation with existential inquiry, challenging us to look beyond immediate surface phenomena.\n\n"
                "How would you like to explore this topic further?"
            )

        return DialogueTurnResult(
            query=query,
            intent_category="philosophy",
            response=resp,
            tone="intellectual_reflective",
            suggested_followups=["Explain Stoicism vs Epicureanism", "What is the Simulation Hypothesis?", "Explain the Ship of Theseus paradox"]
        )

    @classmethod
    def _handle_brainstorming(cls, query: str) -> DialogueTurnResult:
        resp = (
            "### 💡 Creative Brainstorming & High-Impact Project Ideas\n\n"
            "Here are 4 innovative, high-impact concepts blending modern technology, intelligence, and utility:\n\n"
            "1. **Local-First Knowledge Agent**: Build a lightweight, CPU-native personal intelligence assistant using SLMs (like COLLISION-10M) that indexes personal notes, PDFs, and code offline with zero privacy leakage.\n"
            "2. **Real-Time Code Refactoring Telemetry**: A developer tool that analyzes AST trees on file save to highlight cognitive complexity bottlenecks, dead paths, and asymptotic Big-O scaling.\n"
            "3. **Deterministic Verification Copilot**: A hybrid AI system that blends neural natural language parsing with symbolic formal verification (SMT solvers / AST parsers) to eliminate math & logic hallucinations.\n"
            "4. **Dynamic Interactive Simulator**: A web app using WebAssembly and canvas to visually simulate complex physics (quantum double slit, orbital mechanics) or economic game theory scenarios.\n\n"
            "Which of these directions interests you most, or should we narrow down to a specific tech stack?"
        )
        return DialogueTurnResult(
            query=query,
            intent_category="brainstorming",
            response=resp,
            tone="creative_inspiring",
            suggested_followups=["Help me architect the Local-First Agent", "Suggest Python project ideas", "How to build a web app in React & Vite"]
        )

    @classmethod
    def _handle_advice(cls, query: str) -> DialogueTurnResult:
        resp = (
            "### 🚀 High-Efficiency Productivity & Deep Learning Framework\n\n"
            "To achieve mastery and sustain peak focus, apply these empirically validated cognitive protocols:\n\n"
            "1. **The Feynman Technique (True Comprehension)**:\n"
            "   • Explain the concept simply as if teaching a beginner without jargon.\n"
            "   • Identify precise conceptual gaps where your explanation falters, return to source material, and simplify further.\n\n"
            "2. **Ultradian Deep Work Rhythms (90-Minute Blocks)**:\n"
            "   • Work with complete elimination of asynchronous interruptions (phone on silent, notifications disabled).\n"
            "   • Follow with 15 minutes of non-sleep deep rest (NSDR) or a walk to consolidate neural synaptic connections.\n\n"
            "3. **Spaced Repetition & Active Recall**:\n"
            "   • Testing yourself on newly acquired knowledge across increasing intervals (1 day, 3 days, 7 days) provides 3x higher retention than passive re-reading.\n\n"
            "4. **Eisenhower Matrix & Priority Inversion Defense**:\n"
            "   • Distinguish between *Urgent* and *Important*. Protect sacred morning time for high-leverage Important/Non-Urgent tasks.\n\n"
            "What specific skill or domain are you aiming to master?"
        )
        return DialogueTurnResult(
            query=query,
            intent_category="advice",
            response=resp,
            tone="actionable_coaching",
            suggested_followups=["How to learn Data Structures & Algorithms", "How to master System Design", "How to improve memory retention"]
        )
