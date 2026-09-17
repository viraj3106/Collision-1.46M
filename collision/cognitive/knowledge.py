"""
COLLISION Universal Cross-Domain Knowledge Base & Problem Solver.

Provides comprehensive, structured, and authoritative conceptual knowledge
across Computer Science, Artificial Intelligence, Physics, Mathematics,
Biology, Chemistry, Philosophy, Economics, Psychology, History, and Everyday Problem Solving.
"""

import re
from typing import Dict, Any, List, Optional, Tuple
from pydantic import BaseModel, Field


class DomainKnowledgeResult(BaseModel):
    query: str
    domain: str
    concept_title: str
    summary: str
    key_mechanisms: List[str] = Field(default_factory=list)
    advantages_or_insights: List[str] = Field(default_factory=list)
    real_world_application: Optional[str] = None
    formatted_output: str


class CrossDomainKnowledgeBase:
    """
    Multi-domain expert knowledge base providing verified, deeply structured explanations.
    """

    KNOWLEDGE_REGISTRY: Dict[str, Dict[str, Any]] = {
        # -------------------------------------------------------------
        # 1. ARTIFICIAL INTELLIGENCE & MACHINE LEARNING
        # -------------------------------------------------------------
        "transformer_architecture": {
            "keywords": ["transformer", "attention mechanism", "self attention", "multi head attention", "vaswani", "how transformers work"],
            "domain": "Artificial Intelligence & ML",
            "title": "The Transformer Architecture & Self-Attention Mechanism",
            "summary": "The Transformer (introduced in 'Attention Is All You Need', 2017) is a neural architecture that replaces recurrent sequence processing with parallel self-attention mechanisms, capturing long-range dependencies efficiently.",
            "mechanisms": [
                "**Scaled Dot-Product Attention**: Computes attention weights via `Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V`.",
                "**Multi-Head Attention**: Projects Queries, Keys, and Values into multiple subspaces in parallel to capture distinct linguistic/semantic relationships.",
                "**Positional Encodings**: Injects order information (sinusoidal or learned/RoPE) into token embeddings since attention is permutation-invariant.",
                "**Feed-Forward Networks (FFN)**: Two-layer MLP with non-linear activation (GELU/SwiGLU) applied to each token independently."
            ],
            "insights": [
                "Eliminates sequential recurrence bottlenecks, enabling massive GPU/CPU parallelization during training.",
                "Foundation of all modern frontier LLMs and ultra-compact SLMs (like COLLISION-10M)."
            ],
            "application": "Natural language understanding, code generation, vision transformers (ViT), and multimodal AI systems."
        },
        "slm_vs_llm": {
            "keywords": ["slm", "small language model", "slm vs llm", "edge ai", "10m model", "compact model", "cpu inference"],
            "domain": "Artificial Intelligence & ML",
            "title": "Small Language Models (SLMs) vs Large Language Models (LLMs)",
            "summary": "SLMs (10M to 3B parameters) are ultra-compact neural models engineered for high efficiency, sub-5ms CPU latencies, and edge deployment, whereas heavy LLMs (70B+) require massive GPU clusters.",
            "mechanisms": [
                "**Memory Bandwidth Efficiency**: 10M models require only ~20-40 MB of RAM, easily fitting entirely in L3 CPU cache.",
                "**Zero-GPU Requirement**: Execute on standard CPUs, microcontrollers, and Raspberry Pi devices with sub-5ms latency.",
                "**Grounded Hybrid Intelligence**: Combining an SLM with RAG (Retrieval-Augmented Generation) and deterministic verification achieves near-zero hallucination without multi-gigabyte models."
            ],
            "insights": [
                "100x lower carbon footprint and compute cost compared to centralized cloud LLMs.",
                "Ideal for privacy-preserving, on-device intelligence and edge robotics."
            ],
            "application": "Edge IoT, embedded systems, local CLI assistants, offline enterprise search, and microservice agents."
        },
        "rag_retrieval_augmented_generation": {
            "keywords": ["rag", "retrieval augmented generation", "vector search", "dense retrieval", "how rag works"],
            "domain": "Artificial Intelligence & ML",
            "title": "Retrieval-Augmented Generation (RAG)",
            "summary": "RAG couples neural language models with external vector search or lexical retrieval systems to supply verified, up-to-date context dynamically at inference time.",
            "mechanisms": [
                "**Chunking & Indexing**: Raw documents are chunked (e.g. 128-512 tokens with overlap) and indexed into dense vector stores or BM25 lexical indices.",
                "**Semantic Retrieval**: Queries are embedded and compared via Cosine similarity to retrieve top-k relevant context passages.",
                "**Context Conditioning**: Retrieved evidence is prepended to the prompt context for conditioned generation.",
                "**Deterministic Verification**: Claims in the generation are cross-checked against source chunk citations to eliminate hallucinations."
            ],
            "insights": [
                "Enables zero-shot factual updating without costly model retraining.",
                "Provides auditable citations and traceability for enterprise compliance."
            ],
            "application": "Customer support bots, medical query answering, legal discovery, and live web-grounded search."
        },
        "backpropagation_gradient_descent": {
            "keywords": ["backprop", "backpropagation", "gradient descent", "adamw", "loss function", "how neural networks learn"],
            "domain": "Artificial Intelligence & ML",
            "title": "Backpropagation & Gradient Descent Optimization",
            "summary": "The foundational mathematical mechanism by which artificial neural networks update their internal weights to minimize prediction error.",
            "mechanisms": [
                "**Forward Pass**: Input data propagates through layers to compute predictions and calculate scalar loss `L(y_hat, y)`.",
                "**Reverse Auto-Differentiation**: Applies the mathematical Chain Rule backwards through computational graph nodes to compute partial derivatives `∂L/∂W`.",
                "**Parameter Update (AdamW / SGD)**: Adjusts weights in the opposite direction of the gradient: `W = W - lr * m_t / (sqrt(v_t) + eps) - lr * wd * W`."
            ],
            "insights": [
                "Enables scalable end-to-end optimization of billions of parameters simultaneously.",
                "AdamW decouples weight decay from gradient momentum, preventing parameter explosion."
            ],
            "application": "Training all modern deep learning models across NLP, computer vision, and speech."
        },

        # -------------------------------------------------------------
        # 2. COMPUTER SCIENCE & SOFTWARE SYSTEMS
        # -------------------------------------------------------------
        "big_o_complexity": {
            "keywords": ["big o", "time complexity", "space complexity", "asymptotic notation", "o(n)", "o(log n)", "o(n log n)"],
            "domain": "Computer Science & Systems",
            "title": "Asymptotic Analysis & Big-O Computational Complexity",
            "summary": "Big-O notation describes the upper bound of resource consumption (time or memory) of an algorithm as the input size `n` approaches infinity.",
            "mechanisms": [
                "**O(1) Constant Time**: Hash map lookups, array indexing, arithmetic operations.",
                "**O(log n) Logarithmic Time**: Binary search on sorted arrays, balanced binary tree operations.",
                "**O(n) Linear Time**: Linear search, single pass through an array or linked list.",
                "**O(n log n) Linearithmic Time**: Optimal comparison-based sorting (MergeSort, QuickSort, TimSort).",
                "**O(n^2) Quadratic Time**: Nested iterations, BubbleSort, naive pairwise comparisons.",
                "**O(2^n) & O(n!) Exponential / Factorial Time**: Exhaustive combinatorial search (Traveling Salesperson Problem, brute-force knapsack)."
            ],
            "insights": [
                "Focuses strictly on dominant asymptotic scaling terms, discarding constant multipliers.",
                "Crucial for engineering high-scale distributed systems and real-time software."
            ],
            "application": "System scaling, algorithm selection, database query indexing, and competitive programming."
        },
        "cap_theorem": {
            "keywords": ["cap theorem", "brewers theorem", "consistency availability partition", "distributed systems", "eventual consistency"],
            "domain": "Computer Science & Systems",
            "title": "The CAP Theorem in Distributed Computing",
            "summary": "Formulated by Eric Brewer, the CAP theorem states that any distributed data store can simultaneously guarantee at most two out of three properties during a network partition: Consistency, Availability, and Partition Tolerance.",
            "mechanisms": [
                "**Consistency (C)**: Every read receives the most recent write or an error (linearizable state).",
                "**Availability (A)**: Every non-failing node returns a non-error response for every request (without guarantee of newest data).",
                "**Partition Tolerance (P)**: The system continues to operate despite arbitrary network message drops or delays across nodes.",
                "**The Trade-Off (CP vs AP)**: Because network partitions are inevitable in real-world physical networks (P is non-negotiable), systems must choose between Consistency (CP, e.g. Raft, Paxos, Spanner, ZooKeeper) or Availability (AP, e.g. Cassandra, DynamoDB eventual consistency)."
            ],
            "insights": [
                "PACELC Theorem extends CAP by addressing the trade-off between Latency (L) and Consistency (C) even when no partition (E) exists."
            ],
            "application": "Database architecture, cloud infrastructure, banking ledgers, and microservice state design."
        },
        "solid_principles": {
            "keywords": ["solid principles", "solid design", "single responsibility", "open closed", "liskov", "dependency inversion"],
            "domain": "Computer Science & Systems",
            "title": "The SOLID Principles of Object-Oriented Software Design",
            "summary": "Five foundational design principles introduced by Robert C. Martin to build maintainable, understandable, and flexible software architectures.",
            "mechanisms": [
                "**S - Single Responsibility Principle (SRP)**: A module or class should have one, and only one, reason to change.",
                "**O - Open/Closed Principle (OCP)**: Software entities should be open for extension, but closed for modification (via interfaces/polymorphism).",
                "**L - Liskov Substitution Principle (LSP)**: Subtypes must be substitutable for their base types without altering program correctness.",
                "**I - Interface Segregation Principle (ISP)**: Clients should not be forced to depend upon interfaces they do not use.",
                "**D - Dependency Inversion Principle (DIP)**: High-level modules should depend upon abstractions, not concrete details."
            ],
            "insights": [
                "Dramatically reduces coupling and increases testability via dependency injection.",
                "Prevents cascading regressions when expanding codebases."
            ],
            "application": "Enterprise software engineering, clean architecture, SDK design, and domain-driven design (DDD)."
        },

        # -------------------------------------------------------------
        # 3. PHYSICS & MATHEMATICS
        # -------------------------------------------------------------
        "quantum_mechanics_core": {
            "keywords": ["quantum mechanics", "superposition", "entanglement", "schrodinger", "heisenberg uncertainty", "wave particle duality"],
            "domain": "Physics & Mathematics",
            "title": "Core Principles of Quantum Mechanics",
            "summary": "Quantum mechanics describes the fundamental physical behavior of matter and energy at atomic and subatomic scales, where physical quantities are quantized rather than continuous.",
            "mechanisms": [
                "**Wave-Particle Duality**: All particles (electrons, photons) exhibit both wave-like and particle-like properties (de Broglie wavelength `λ = h/p`).",
                "**Quantum Superposition**: A quantum state exists as a linear combination of basis states `|ψ⟩ = α|0⟩ + β|1⟩` until measurement collapses the wave function.",
                "**Quantum Entanglement**: Quantum states of two or more particles are fundamentally linked such that the state of one instantly determines the other regardless of distance (Einstein's 'spooky action at a distance').",
                "**Heisenberg Uncertainty Principle**: It is fundamentally impossible to simultaneously determine both the precise position `x` and momentum `p` of a particle: `Δx * Δp ≥ ℏ/2`."
            ],
            "insights": [
                "The universe is fundamentally probabilistic at microscopic scales, governed by complex probability amplitudes.",
                "Underpins semiconductor physics, lasers, MRI scanners, and quantum computing."
            ],
            "application": "Quantum computing (qubits), cryptography (QKD), nanotechnology, and atomic spectroscopy."
        },
        "general_special_relativity": {
            "keywords": ["special relativity", "general relativity", "einstein relativity", "spacetime", "time dilation", "e=mc2", "speed of light"],
            "domain": "Physics & Mathematics",
            "title": "Einstein's Special & General Theories of Relativity",
            "summary": "Revolutionized our understanding of space, time, mass, and gravity, unifying space and time into a 4-dimensional continuum.",
            "mechanisms": [
                "**Special Relativity (1905)**:\n  • Principle of Relativity: The laws of physics are identical in all inertial reference frames.\n  • Invariance of `c`: The speed of light in vacuum is constant for all observers (`c ≈ 299,792,458 m/s`).\n  • Time Dilation & Length Contraction: Moving clocks run slower (`t' = t / sqrt(1 - v^2/c^2)`).\n  • Mass-Energy Equivalence: `E = mc^2`.",
                "**General Relativity (1915)**:\n  • Principle of Equivalence: Gravitational force is locally indistinguishable from accelerated motion.\n  • Spacetime Curvature: Mass and energy warp the geometric curvature of spacetime (`G_μν + Λg_μν = (8πG/c^4) T_μν`); matter tells spacetime how to curve, and curved spacetime tells matter how to move."
            ],
            "insights": [
                "Explains gravitational lensing, black holes, gravitational waves (LIGO), and GPS satellite clock adjustments."
            ],
            "application": "GPS satellite synchronization, astrophysics, cosmology, and particle accelerator design."
        },
        "thermodynamics_laws": {
            "keywords": ["thermodynamics", "entropy", "laws of thermodynamics", "heat transfer", "second law of thermodynamics", "carnot"],
            "domain": "Physics & Mathematics",
            "title": "The Fundamental Laws of Thermodynamics",
            "summary": "Governs the relationships between heat, work, temperature, and energy transformations across all physical and chemical systems.",
            "mechanisms": [
                "**Zeroth Law (Thermal Equilibrium)**: If system A is in thermal equilibrium with B, and B with C, then A is in equilibrium with C (defines temperature).",
                "**First Law (Conservation of Energy)**: Energy cannot be created or destroyed, only transformed: `ΔU = Q - W`.",
                "**Second Law (Entropy & Arrow of Time)**: The total entropy `S` of an isolated system always increases over time (`ΔS_total ≥ 0`); natural processes are irreversible.",
                "**Third Law (Absolute Zero)**: As temperature approaches absolute zero (`0 Kelvin` / `-273.15°C`), the entropy of a pure crystalline substance approaches a constant minimum value (zero)."
            ],
            "insights": [
                "No heat engine can achieve 100% thermal efficiency; Carnot efficiency `η = 1 - T_cold/T_hot` sets the theoretical ceiling.",
                "Entropy explains the thermodynamic arrow of time."
            ],
            "application": "Combustion engines, refrigeration, power plants, chemical engineering, and computing heat dissipation."
        },

        # -------------------------------------------------------------
        # 4. BIOLOGY, MEDICINE & GENETICS
        # -------------------------------------------------------------
        "dna_genetics_crispr": {
            "keywords": ["dna", "rna", "genetics", "crispr", "gene editing", "protein synthesis", "central dogma"],
            "domain": "Biology, Medicine & Genetics",
            "title": "DNA, the Central Dogma & CRISPR Gene Editing",
            "summary": "The molecular foundation of biological inheritance, protein expression, and precision genomic engineering.",
            "mechanisms": [
                "**DNA Double Helix**: Antiparallel sugar-phosphate backbone with complementary base pairing: Adenine (A) pairs with Thymine (T), Guanine (G) pairs with Cytosine (C).",
                "**Central Dogma of Molecular Biology**: Information flows unidirectionally from `DNA ➔ (Transcription) ➔ mRNA ➔ (Translation on Ribosomes) ➔ Functional Protein`.",
                "**CRISPR-Cas9 Mechanism**: Bacterial adaptive immune system repurposed for genetic engineering. Guide RNA (gRNA) directs the Cas9 endonuclease enzyme to target specific genomic sequences for precision double-strand cleavage and editing."
            ],
            "insights": [
                "Enables targeted cure development for monogenic diseases (e.g. sickle cell anemia) and agricultural optimization.",
                "Epigenetic modifications (DNA methylation, histone acetylation) modulate gene expression without altering base sequences."
            ],
            "application": "Gene therapy, biotechnology, personalized medicine, forensics, and synthetic biology."
        },
        "neuroscience_neurons_synapses": {
            "keywords": ["neuroscience", "neuron", "synapse", "neurotransmitters", "dopamine", "action potential", "neuroplasticity"],
            "domain": "Biology & Neuroscience",
            "title": "Neurons, Synaptic Transmission & Neuroplasticity",
            "summary": "The structural and functional architecture of the central nervous system, driving cognition, memory, learning, and motor control.",
            "mechanisms": [
                "**Action Potential**: Electrochemical wave triggered when membrane potential reaches threshold (~ -55mV), causing rapid voltage-gated Na+ influx and subsequent K+ efflux.",
                "**Synaptic Transmission**: Action potential arrives at axon terminal, triggering calcium influx and vesicle release of neurotransmitters into the synaptic cleft.",
                "**Key Neurotransmitters**:\n  • *Dopamine*: Reward prediction error, motivation, motor planning.\n  • *Serotonin*: Mood regulation, sleep, emotional resilience.\n  • *GABA & Glutamate*: Primary inhibitory and excitatory neurotransmitters of the brain.",
                "**Hebbian Plasticity & Long-Term Potentiation (LTP)**: *\"Neurons that fire together, wire together.\"* Repeated synaptic stimulation strengthens synaptic efficacy, forming the physical basis of learning and memory."
            ],
            "insights": [
                "The adult brain retains structural neuroplasticity throughout life in response to targeted practice and cognitive challenge."
            ],
            "application": "Brain-computer interfaces (BCI), neuropharmacology, cognitive training, and artificial neural network inspiration."
        },

        # -------------------------------------------------------------
        # 5. PHILOSOPHY & ETHICS
        # -------------------------------------------------------------
        "ethics_frameworks": {
            "keywords": ["ethics", "utilitarianism", "deontology", "virtue ethics", "kant", "categorical imperative", "moral philosophy"],
            "domain": "Philosophy & Ethics",
            "title": "Major Normative Ethical Frameworks",
            "summary": "The three primary philosophical systems for evaluating moral duty, right action, and human character.",
            "mechanisms": [
                "**1. Utilitarianism (Consequentialism - Bentham, Mill)**:\n  • Principle: Actions are morally right if they maximize overall well-being and minimize suffering for the greatest number (*The Greatest Happiness Principle*).\n  • Focus: Outcomes and consequences.",
                "**2. Deontology (Duty-Based - Immanuel Kant)**:\n  • Principle: Actions are intrinsically right or wrong based on adherence to universal moral duties, regardless of consequences.\n  • Categorical Imperative: *\"Act only according to that maxim whereby you can at the same time will that it should become a universal law.\"*",
                "**3. Virtue Ethics (Character-Based - Aristotle)**:\n  • Principle: Morality stems from cultivating virtuous character traits (wisdom, courage, temperance, justice) finding the *Golden Mean* between excess and deficiency."
            ],
            "insights": [
                "Modern ethical decisions in AI safety, bioethics, and public policy frequently balance Utilitarian utility against Deontological rights constraints."
            ],
            "application": "AI alignment and safety, bioethics, legal jurisprudence, and corporate governance."
        },

        # -------------------------------------------------------------
        # 6. ECONOMICS, FINANCE & GAME THEORY
        # -------------------------------------------------------------
        "game_theory_nash_equilibrium": {
            "keywords": ["game theory", "nash equilibrium", "prisoners dilemma", "zero sum game", "dominant strategy", "minimax"],
            "domain": "Economics & Game Theory",
            "title": "Game Theory, Nash Equilibrium & The Prisoner's Dilemma",
            "summary": "The mathematical study of strategic interaction where the outcome for each participant depends upon the choices of all participants.",
            "mechanisms": [
                "**Nash Equilibrium (John Nash)**: A state of a game where no player can unilaterally improve their payoff by changing their strategy, given the strategies chosen by all other players.",
                "**The Prisoner's Dilemma**: Two suspects can either cooperate (stay silent) or defect (confess). Although mutual cooperation yields the highest collective payoff, individual rational self-interest leads both to defect—resulting in a Pareto-suboptimal Nash Equilibrium.",
                "**Zero-Sum vs Non-Zero-Sum**: In zero-sum games, one player's gain is exactly another's loss; in non-zero-sum games, trade and cooperation can create positive net value for all participants.",
                "**Tit-for-Tat Strategy**: In repeated Prisoner's Dilemmas, starting with cooperation and then mirroring the opponent's previous move proves empirically optimal for sustaining cooperation."
            ],
            "insights": [
                "Explains economic price wars, international nuclear deterrence (MAD), and open-source collaboration dynamics."
            ],
            "application": "Auction design, antitrust regulation, cybersecurity defense, and geopolitical negotiations."
        },
        "supply_demand_macroeconomics": {
            "keywords": ["supply and demand", "inflation", "gdp", "fiscal policy", "monetary policy", "interest rates", "macroeconomics"],
            "domain": "Economics & Finance",
            "title": "Core Principles of Micro & Macroeconomics",
            "summary": "The economic framework governing market price equilibrium, national output, monetary inflation, and fiscal stabilization.",
            "mechanisms": [
                "**Supply & Demand Equilibrium**: Market clearing price occurs where quantity supplied equals quantity demanded. Elasticity measures price sensitivity.",
                "**Inflation & Monetary Policy**: Central banks raise interest rates (contracting money supply) to cool inflation, or lower rates to stimulate economic growth and employment.",
                "**Fiscal Policy**: Government taxation and expenditure policies used to influence aggregate demand during economic cycles.",
                "**Gross Domestic Product (GDP)**: The total monetary value of all finished goods and services produced within a country: `GDP = Consumption + Investment + Government Spending + Net Exports (C + I + G + NX)`."
            ],
            "insights": [
                "Real economic growth stems from technological innovation and productivity gains (total factor productivity)."
            ],
            "application": "Corporate financial planning, investment portfolio allocation, and macroeconomic forecasting."
        }
    }

    @classmethod
    def query_knowledge(cls, query: str) -> Optional[DomainKnowledgeResult]:
        """
        Matches a user's question against the cross-domain knowledge registry.
        """
        q_lower = query.lower().strip()

        # Specific collision model queries should be handled by local RAG, not general cross-domain articles
        if any(kw in q_lower for kw in ["collision 10m", "collision model", "transformer layers are used in collision"]):
            return None

        best_match = None
        highest_score = 0

        for key, entry in cls.KNOWLEDGE_REGISTRY.items():
            score = 0
            for kw in entry["keywords"]:
                kw_lower = kw.lower()
                if kw_lower in q_lower:
                    word_count = len(kw_lower.split())
                    # Multi-word matches get higher weight; single-word matches require high relevance
                    score += word_count * 3 if word_count > 1 else 1

            # Require at least score 3 (e.g. a 2-word keyword or multiple keyword hits)
            if score > highest_score and score >= 3:
                highest_score = score
                best_match = entry

        if not best_match:
            return None

        # Build rich structured output
        formatted = [
            f"### {best_match['title']}",
            f"**Domain**: *{best_match['domain']}*\n",
            f"**Executive Summary**:\n> {best_match['summary']}\n",
            "**Core Mechanisms & Technical Foundations**:"
        ]
        for m in best_match["mechanisms"]:
            formatted.append(f"• {m}")

        if best_match.get("insights"):
            formatted.append("\n**Strategic Insights & Key Implications**:")
            for i in best_match["insights"]:
                formatted.append(f"• {i}")

        if best_match.get("application"):
            formatted.append(f"\n**Real-World Applications & Industry Use**:\n• {best_match['application']}")

        return DomainKnowledgeResult(
            query=query,
            domain=best_match["domain"],
            concept_title=best_match["title"],
            summary=best_match["summary"],
            key_mechanisms=best_match["mechanisms"],
            advantages_or_insights=best_match.get("insights", []),
            real_world_application=best_match.get("application"),
            formatted_output="\n".join(formatted)
        )
