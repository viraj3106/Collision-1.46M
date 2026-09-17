"""
COLLISION Cross-Domain Cognitive Lattice & Multi-Hop Concept Synthesizer 2.0.

Provides 25+ multi-disciplinary associative links across:
- Computer Science & Distributed Systems (CAP, Byzantine Faults, Raft, Paxos, Actor Model, P vs NP, Cache Coherency)
- Artificial Intelligence & Neural Systems (Transformers, Backpropagation, Scaling Laws, Info Bottleneck, MoE, KV Cache)
- Physics & Information Theory (Heisenberg, Shannon Entropy, Landauer's Principle, Quantum Entanglement, Thermodynamics)
- Philosophy of Mind & Cognitive Science (Chinese Room, Global Workspace, Dual-Process, Free Energy, Gödel Incompleteness)
- Economics & Systems Dynamics (Nash Equilibrium, Pareto Optimality, Prospect Theory, Cybernetics, Mechanism Design)
"""

import re
from functools import lru_cache
from typing import Dict, Any, List, Optional, Tuple, Set
from pydantic import BaseModel, Field


class LatticeConcept(BaseModel):
    concept_id: str
    name: str
    domain: str
    definition: str
    formal_equation: Optional[str] = None
    core_principles: List[str]
    boundary_conditions: Optional[str] = None
    related_concepts: List[str] = Field(default_factory=list)
    cross_disciplinary_analogy: Optional[str] = None
    counter_intuitive_insight: Optional[str] = None


class CrossDomainKnowledgeLattice:
    """
    Associative Semantic Lattice linking foundational concepts across 5 major scientific disciplines.
    """

    LATTICE_GRAPH: Dict[str, LatticeConcept] = {
        # -------------------------------------------------------------------
        # 1. Distributed Systems & Computer Science
        # -------------------------------------------------------------------
        "cap_theorem": LatticeConcept(
            concept_id="cap_theorem",
            name="CAP Theorem (Brewer's Theorem)",
            domain="Distributed Systems",
            definition="A distributed data store can simultaneously provide at most two out of three guarantees: Consistency, Availability, and Partition Tolerance.",
            formal_equation="In any network partition P: Consistency XOR Availability",
            core_principles=[
                "In the presence of network partition (P), a system must trade off between returning stale data (Availability) or failing writes (Consistency).",
                "PACELC theorem extends CAP to latency vs consistency trade-offs during normal execution."
            ],
            boundary_conditions="Applies strictly to distributed systems operating across unreliable network partitions.",
            related_concepts=["heisenberg_uncertainty", "nash_equilibrium", "byzantine_fault", "raft_consensus"],
            cross_disciplinary_analogy="Analogous to Heisenberg's Uncertainty Principle in quantum mechanics: measuring one parameter with absolute precision forces uncertainty in the conjugate parameter.",
            counter_intuitive_insight="Partition tolerance is not optional in real networks; hence, the real operational trade-off is always Availability vs Consistency (AP vs CP)."
        ),
        "byzantine_fault": LatticeConcept(
            concept_id="byzantine_fault",
            name="Byzantine Fault Tolerance (BFT)",
            domain="Distributed Systems & Cryptography",
            definition="The capability of a distributed computer network to function correctly and reach consensus despite nodes failing or propagating malicious/arbitrary misinformation.",
            formal_equation="N >= 3f + 1 (tolerates f arbitrary/malicious adversarial nodes)",
            core_principles=[
                "Guarantees safety and liveness under adversarial node behavior (Lamport et al.).",
                "Requires quorum thresholds and cryptographic signatures or multi-phase voting rounds."
            ],
            boundary_conditions="Consensus cannot be guaranteed if strictly more than 1/3 of nodes are malicious in asynchronous networks.",
            related_concepts=["cap_theorem", "raft_consensus", "paxos_algorithm", "nash_equilibrium"],
            cross_disciplinary_analogy="Echoes immune system pathogen defense: biological consensus is maintained by multi-point receptor validation even when rogue cells transmit false chemical signals.",
            counter_intuitive_insight="Even a single malicious node can prevent consensus in a 3-node system; a minimum of 4 nodes is mathematically required to withstand 1 traitor."
        ),
        "raft_consensus": LatticeConcept(
            concept_id="raft_consensus",
            name="Raft Distributed Consensus",
            domain="Distributed Systems",
            definition="An understandable consensus algorithm for managing replicated state machines via strong leader election, log replication, and safety invariants.",
            formal_equation="Quorum = floor(N/2) + 1",
            core_principles=[
                "Decomposes consensus into distinct subproblems: Leader Election, Log Replication, and Safety.",
                "Ensures state machine safety: If any server has applied a particular log entry, no other server will ever apply a different log entry for that index."
            ],
            boundary_conditions="Requires a strict majority quorum (>50%) of non-faulty nodes to commit new log entries.",
            related_concepts=["paxos_algorithm", "byzantine_fault", "cap_theorem"],
            cross_disciplinary_analogy="Analogous to parliamentary democratic governance: a leader holds executive authority until term expiry or loss of majority quorum confidence.",
            counter_intuitive_insight="Raft guarantees identical safety to Multi-Paxos while drastically reducing implementation edge-case complexity through strict leader dominance."
        ),
        "paxos_algorithm": LatticeConcept(
            concept_id="paxos_algorithm",
            name="Paxos Consensus Algorithm",
            domain="Distributed Systems",
            definition="A foundational protocol family for solving consensus in a network of unreliable or fallible processors (Leslie Lamport).",
            formal_equation="Phase 1 (Prepare/Promise) -> Phase 2 (Accept/Accepted) with monotonically increasing proposal numbers",
            core_principles=[
                "Guarantees that only a single value is ever chosen among competing proposers.",
                "Tolerates node crashes, network delays, and message reordering without compromising consistency."
            ],
            boundary_conditions="Assumes non-Byzantine crash-recovery model with asynchronous network bounds.",
            related_concepts=["raft_consensus", "byzantine_fault", "cap_theorem"],
            cross_disciplinary_analogy="Functions like a formal legislative floor vote with sequential amendments requiring majority confirmation before enactment.",
            counter_intuitive_insight="Paxos guarantees safety under all asynchronous conditions, but liveness can theoretically be disrupted by dueling proposers (FLP impossibility theorem)."
        ),
        "actor_model": LatticeConcept(
            concept_id="actor_model",
            name="Actor Model Concurrency",
            domain="Distributed Systems & Programming Languages",
            definition="A mathematical model of concurrent computation that treats 'actors' as universal primitives that communicate exclusively via asynchronous message passing.",
            formal_equation="State(t+1) = f(State(t), Message); spawns child actors and emits outbound messages",
            core_principles=[
                "Share-nothing architecture: No shared mutable memory; zero lock contention.",
                "Supervision trees provide fault isolation: 'Let it crash' philosophy with hierarchical recovery."
            ],
            boundary_conditions="Message ordering is typically non-deterministic across independent sender actors.",
            related_concepts=["global_workspace_theory", "cybernetics_feedback_loops", "mixture_of_experts"],
            cross_disciplinary_analogy="Resembles an organization of specialized human agents collaborating strictly via asynchronous email threads rather than shared physical whiteboards.",
            counter_intuitive_insight="Eliminating shared locks by copying messages frequently outperforms shared-memory architectures under high thread contention."
        ),
        "p_vs_np": LatticeConcept(
            concept_id="p_vs_np",
            name="P versus NP Problem",
            domain="Theoretical Computer Science",
            definition="A major unsolved question asking whether every problem whose solution can be verified quickly (polynomial time, NP) can also be solved quickly (polynomial time, P).",
            formal_equation="Is P = NP or P != NP?",
            core_principles=[
                "P: Decision problems solvable in polynomial time O(n^k) on a deterministic Turing machine.",
                "NP-Complete: The hardest problems in NP; finding a polynomial algorithm for any one solves all problems in NP."
            ],
            boundary_conditions="Defined strictly within the formal Turing machine computational complexity model.",
            related_concepts=["godel_incompleteness", "information_entropy", "amortized_complexity"],
            cross_disciplinary_analogy="The difference between appreciating a masterpiece symphony (verification / NP) versus composing one from scratch (generation / P).",
            counter_intuitive_insight="If P = NP, modern asymmetric public-key cryptography (RSA, ECC) collapses instantaneously since factorization and discrete logs become polynomial."
        ),
        "amortized_complexity": LatticeConcept(
            concept_id="amortized_complexity",
            name="Amortized Complexity Analysis",
            domain="Computer Science & Data Structures",
            definition="A method of analyzing algorithms that considers the average running time per operation over a worst-case sequence of operations.",
            formal_equation="Amortized Cost = Actual Cost + Potential Function Change: a_i = c_i + Φ(D_i) - Φ(D_{i-1})",
            core_principles=[
                "Occasional expensive operations (e.g. dynamic array resizing) are mathematically offset by numerous cheap operations.",
                "The Potential Method assigns dynamic energy Φ to data structures to bound total sequence cost."
            ],
            boundary_conditions="Valid across continuous sequences of operations; individual single-call latency may still spike.",
            related_concepts=["p_vs_np", "thermodynamic_entropy_second_law", "kv_cache_paging"],
            cross_disciplinary_analogy="Analogous to buying in bulk: high upfront capital outlay amortized across months yields a drastically lower average cost per item.",
            counter_intuitive_insight="A single operation taking O(N) worst-case time can still guarantee strict O(1) amortized performance across all production sequences."
        ),
        "cache_coherency_mesi": LatticeConcept(
            concept_id="cache_coherency_mesi",
            name="MESI Cache Coherence Protocol",
            domain="Computer Architecture & Hardware",
            definition="An invalidation-based cache coherence protocol maintaining memory consistency across multi-core symmetric multiprocessing (SMP) CPUs.",
            formal_equation="States: Modified (M), Exclusive (E), Shared (S), Invalid (I)",
            core_principles=[
                "Write-Invalidate protocol: Writing to a cache line broadcasts an invalidation bus message to all peer caches.",
                "Prevents multiple cores from observing conflicting values for the same physical memory address."
            ],
            boundary_conditions="Hardware bus snooping bandwidth limits scalability beyond large numbers of CPU cores without directory-based protocols.",
            related_concepts=["cap_theorem", "global_workspace_theory", "actor_model"],
            cross_disciplinary_analogy="Analogous to patent ownership registries: an inventor can hold exclusive rights (E) or license shared rights (S), but modifying the asset (M) revokes peer copies (I).",
            counter_intuitive_insight="True memory bottleneck in multicore scaling is rarely arithmetic ALU speed, but rather cross-core cache coherence bus invalidation traffic."
        ),

        # -------------------------------------------------------------------
        # 2. Artificial Intelligence, Neural Systems & SLMs
        # -------------------------------------------------------------------
        "transformer_attention": LatticeConcept(
            concept_id="transformer_attention",
            name="Transformer Self-Attention & Neural Routing",
            domain="Artificial Intelligence & ML",
            definition="A sequence modeling architecture utilizing scaled dot-product attention to compute dynamic contextual representations across all token positions simultaneously.",
            formal_equation="Attention(Q, K, V) = softmax((Q * K^T) / sqrt(d_k)) * V",
            core_principles=[
                "Permutation-invariant token interaction conditioned by positional encodings (RoPE / sinusoidal).",
                "Replaces sequential recurrent inductive bias with direct O(1) path length between any two tokens."
            ],
            boundary_conditions="Standard full attention exhibits quadratic O(N^2) memory and compute complexity with respect to context length N.",
            related_concepts=["global_workspace_theory", "information_entropy", "kv_cache_paging", "scaling_laws"],
            cross_disciplinary_analogy="Functions as a soft associative dynamic memory lookup, where queries broadcast across keys to weight value representations.",
            counter_intuitive_insight="Self-attention does not process order inherently; word order is entirely injected artificially through positional encodings."
        ),
        "backpropagation": LatticeConcept(
            concept_id="backpropagation",
            name="Backpropagation & Automatic Differentiation",
            domain="Artificial Intelligence & Mathematics",
            definition="An algorithm for efficiently computing the gradient of a loss function with respect to all neural network weights via the multivariate calculus chain rule.",
            formal_equation="∂L/∂w_ij = (∂L/∂y_j) * (∂y_j/∂net_j) * (∂net_j/∂w_ij)",
            core_principles=[
                "Reverse-mode automatic differentiation computes gradients of scalar loss with respect to millions of parameters in a single backward pass.",
                "Guarantees gradient computation in time proportional to the forward evaluation."
            ],
            boundary_conditions="Susceptible to vanishing/exploding gradients in deep un-normalized graphs without residual connections or normalization layers.",
            related_concepts=["transformer_attention", "scaling_laws", "information_bottleneck"],
            cross_disciplinary_analogy="Like tracing backwards through an intricate supply chain to allocate exact cost responsibility to each upstream supplier.",
            counter_intuitive_insight="Computing gradients for 10 million parameters takes roughly the same computational time as running just two forward inference passes."
        ),
        "scaling_laws": LatticeConcept(
            concept_id="scaling_laws",
            name="Neural Scaling Laws (Kaplan / Chinchilla)",
            domain="Artificial Intelligence & Empirical ML",
            definition="Power-law relationships governing the predictable improvement of language model cross-entropy loss as a function of compute (FLOPs), dataset size (tokens), and parameter count.",
            formal_equation="Loss(N, D) = (N_c / N)^alpha_N + (D_c / D)^alpha_D + L_0",
            core_principles=[
                "Model performance scales predictably as a power law of compute over multiple orders of magnitude.",
                "Chinchilla optimality dictates scaling parameters and dataset tokens in equal proportion (approx. 20 tokens per parameter)."
            ],
            boundary_conditions="Power laws break down when data quality degrades (repetition/synthetic collapse) or upon reaching irreducible task entropy L_0.",
            related_concepts=["transformer_attention", "information_entropy", "information_bottleneck"],
            cross_disciplinary_analogy="Echoes Allometric Scaling in biology: metabolic rates and lifespan scale predictably with animal body mass according to Kleiber's Law.",
            counter_intuitive_insight="Training an undertrained giant model is computationally wasteful compared to training a compact model (SLM) on vastly more high-quality tokens."
        ),
        "information_bottleneck": LatticeConcept(
            concept_id="information_bottleneck",
            name="Information Bottleneck Theory (Tishby)",
            domain="Information Theory & Deep Learning",
            definition="A theoretical framework asserting that deep neural networks learn by finding a minimal sufficient representation T that compresses input X while preserving mutual information with label Y.",
            formal_equation="min_{p(t|x)} I(X; T) - beta * I(T; Y)",
            core_principles=[
                "Two distinct training phases: Initial empirical fitting followed by gradual diffusion-like compression and noise discarding.",
                "Optimal representations trade off compression complexity against predictive sufficiency."
            ],
            boundary_conditions="Exact mutual information calculation becomes computationally intractable for deterministic continuous high-dimensional activations.",
            related_concepts=["information_entropy", "scaling_laws", "mechanistic_interpretability"],
            cross_disciplinary_analogy="Like an executive executive summary: strip away 99% of raw operational noise while preserving 100% of the strategic decision-making signal.",
            counter_intuitive_insight="Neural generalization occurs not merely during parameter fitting, but primarily when the network actively compresses and forgets irrelevant input variance."
        ),
        "mixture_of_experts": LatticeConcept(
            concept_id="mixture_of_experts",
            name="Sparse Mixture-of-Experts (MoE)",
            domain="Artificial Intelligence & Model Architecture",
            definition="An architectural paradigm where feed-forward layers are split into multiple specialized sub-networks ('experts'), with a gating router selecting top-K experts per token.",
            formal_equation="y = sum_{i in TopK} G(x)_i * Expert_i(x)",
            core_principles=[
                "Decouples total parameter capacity from per-token compute cost.",
                "Enables massive parameter scaling (e.g. 8x capacity) while maintaining constant inference FLOPs."
            ],
            boundary_conditions="Prone to routing collapse (load imbalance) where a few dominant experts starve others without auxiliary load-balancing loss.",
            related_concepts=["transformer_attention", "global_workspace_theory", "actor_model"],
            cross_disciplinary_analogy="Analogous to a multidisciplinary hospital: patients (tokens) are dynamically routed only to the relevant specialist clinic (expert) rather than seeing every doctor.",
            counter_intuitive_insight="An MoE model with 50B total parameters can execute inference faster and cheaper than a dense 15B model while retaining superior knowledge capacity."
        ),
        "mechanistic_interpretability": LatticeConcept(
            concept_id="mechanistic_interpretability",
            name="Mechanistic Interpretability & Superposition",
            domain="AI Safety & Cognitive ML",
            definition="The reverse-engineering of neural network weights into human-understandable computational circuits, features, and causal algorithms.",
            formal_equation="Feature Vector h = sum c_i * v_i in d-dimensional space where features > d (Superposition)",
            core_principles=[
                "Polysemanticity: Individual neurons often activate on multiple unrelated concepts due to feature superposition.",
                "Sparse Autoencoders (SAEs) decompose dense superposition representations into monosemantic, interpretable dictionary atoms."
            ],
            boundary_conditions="Reconstructing complete global circuit graphs for multi-billion parameter networks remains an open research frontier.",
            related_concepts=["information_bottleneck", "chinese_room", "transformer_attention"],
            cross_disciplinary_analogy="Like reverse-engineering compiled machine binary back into high-level structured C source code with explicit variable names.",
            counter_intuitive_insight="Neural networks represent far more independent semantic concepts than they have physical dimensions by exploiting near-orthogonal vectors in high-dimensional space."
        ),
        "kv_cache_paging": LatticeConcept(
            concept_id="kv_cache_paging",
            name="PagedAttention & KV Cache Optimization",
            domain="AI Systems & High-Throughput Inference",
            definition="A memory management algorithm inspired by virtual memory paging that eliminates fragmentation in Key-Value (KV) attention cache during LLM/SLM generation.",
            formal_equation="KV Block Table: Logical Token Sequence -> Physical Non-Contiguous Block Pointers",
            core_principles=[
                "Allocates KV cache memory dynamically in fixed-size blocks (pages) rather than large contiguous pre-allocated buffers.",
                "Enables memory sharing across parallel generation paths (beam search, parallel sampling) with zero duplication."
            ],
            boundary_conditions="Paged lookup adds minor pointer indirection overhead, heavily offset by 2-4x higher batch throughput.",
            related_concepts=["transformer_attention", "amortized_complexity", "cache_coherency_mesi"],
            cross_disciplinary_analogy="Directly mirrors OS virtual memory page tables translating virtual memory addresses to non-contiguous physical RAM pages.",
            counter_intuitive_insight="Up to 80% of GPU memory in legacy LLM serving was wasted on internal/external fragmentation before dynamic paging."
        ),

        # -------------------------------------------------------------------
        # 3. Physics, Thermodynamics & Information Theory
        # -------------------------------------------------------------------
        "heisenberg_uncertainty": LatticeConcept(
            concept_id="heisenberg_uncertainty",
            name="Heisenberg Uncertainty Principle",
            domain="Quantum Physics",
            definition="The fundamental limit on the precision with which certain pairs of physical properties (such as position and momentum) can be simultaneously known: Δx * Δp >= ℏ/2.",
            formal_equation="Δx * Δp >= ℏ / 2;  [X, P] = iℏ",
            core_principles=[
                "Not an experimental measurement artifact, but an intrinsic wave-like property of quantum wavefunctions.",
                "Non-commuting quantum operators prevent simultaneous sharp eigenstates."
            ],
            boundary_conditions="Applies to non-commuting conjugate observable pairs in quantum mechanical Hilbert spaces.",
            related_concepts=["cap_theorem", "information_entropy", "quantum_entanglement", "fourier_uncertainty"],
            cross_disciplinary_analogy="Directly isomorphic to the Gabor Fourier time-frequency limit: a sound cannot simultaneously possess instantaneous time localization and single-frequency precision.",
            counter_intuitive_insight="Uncertainty is not a limitation of our measuring instruments; a particle does not possess a definite position and momentum simultaneously."
        ),
        "information_entropy": LatticeConcept(
            concept_id="information_entropy",
            name="Shannon Information Entropy",
            domain="Information Theory & Mathematics",
            definition="The fundamental measure of average uncertainty, information content, and surprisal inherent in a stochastic variable's possible outcomes (Claude Shannon, 1948).",
            formal_equation="H(X) = - sum_{i=1}^n p(x_i) * log_2(p(x_i))",
            core_principles=[
                "Defines the absolute theoretical lower limit on lossless data compression (Source Coding Theorem).",
                "Information is quantified as the reduction in prior uncertainty upon receiving a message."
            ],
            boundary_conditions="Assumes known stationary probability distribution of independent or ergodic source tokens.",
            related_concepts=["heisenberg_uncertainty", "thermodynamic_entropy_second_law", "landauer_principle", "scaling_laws"],
            cross_disciplinary_analogy="Thermodynamic entropy measures molecular disorder in physical systems; Shannon entropy measures semantic unpredictability in symbolic messages.",
            counter_intuitive_insight="A completely predictable message conveys exactly 0 bits of information regardless of its length."
        ),
        "landauer_principle": LatticeConcept(
            concept_id="landauer_principle",
            name="Landauer's Principle",
            domain="Physics of Computation",
            definition="The physical principle establishing that any logically irreversible manipulation of information (such as erasing a bit) must dissipate a minimum amount of thermodynamic heat.",
            formal_equation="E_min = k_B * T * ln(2)  (~ 2.87 * 10^-21 Joules at 298 K per erased bit)",
            core_principles=[
                "Information is physical: Erasing computational state reduces information entropy and must increase physical thermodynamic entropy.",
                "Reversible computation (Fredkin/Toffoli gates) theoretically allows zero-energy computation."
            ],
            boundary_conditions="Applies strictly to logically irreversible operations where output cannot reconstruct input state.",
            related_concepts=["information_entropy", "thermodynamic_entropy_second_law", "heisenberg_uncertainty"],
            cross_disciplinary_analogy="Like sweeping dust from a table: you cannot remove the disorder from the table without dispersing it into the surrounding room air.",
            counter_intuitive_insight="Computing an answer does not inherently consume energy; it is the act of erasing intermediate scratchpad memory that dissipates heat."
        ),
        "quantum_entanglement": LatticeConcept(
            concept_id="quantum_entanglement",
            name="Quantum Entanglement & Non-Locality",
            domain="Quantum Mechanics",
            definition="A phenomenon where quantum particles become inextricably linked such that the quantum state of each particle cannot be described independently of the state of the others.",
            formal_equation="|Ψ⟩ = (1/sqrt(2)) * (|00⟩ + |11⟩) (Bell State)",
            core_principles=[
                "Violates Local Realism (Bell's Theorem experimental confirmation).",
                "Does not violate Special Relativity: Cannot be utilized for Faster-Than-Light (FTL) communication (No-Communication Theorem)."
            ],
            boundary_conditions="Entangled states are fragile and subject to rapid environmental decoherence.",
            related_concepts=["heisenberg_uncertainty", "information_entropy", "byzantine_fault"],
            cross_disciplinary_analogy="Like two copies of the same coin flipped across light-years that always land on the identical side, despite neither transmitting signals during flight.",
            counter_intuitive_insight="Spooky instantaneous correlation across cosmic distances is real, yet perfectly preserves Einstein's cosmic speed limit because measurements yield pure random entropy in isolation."
        ),
        "thermodynamic_entropy_second_law": LatticeConcept(
            concept_id="thermodynamic_entropy_second_law",
            name="Second Law of Thermodynamics",
            domain="Statistical Mechanics & Physics",
            definition="In an isolated system, total entropy (disorder) can never decrease over time; it can remain constant in reversible processes and must increase in irreversible spontaneous processes.",
            formal_equation="ΔS_universe >= 0;  S = k_B * ln(Ω)",
            core_principles=[
                "Establishes the fundamental Arrow of Time: macroscopic processes are fundamentally irreversible.",
                "Local order (e.g. living organisms, neural training) can only increase at the expense of greater external environmental entropy."
            ],
            boundary_conditions="Applies strictly to closed/isolated thermodynamic systems; open systems can export entropy.",
            related_concepts=["information_entropy", "landauer_principle", "active_inference_predictive_coding"],
            cross_disciplinary_analogy="Training a neural network creates intense local mathematical order (reducing parameter entropy) by consuming electrical power and releasing thermal heat.",
            counter_intuitive_insight="The universe does not evolve toward complexity, but rather toward maximizing microscopic disorder; complex structures are transient dissipative engines accelerating entropy production."
        ),
        "fourier_uncertainty": LatticeConcept(
            concept_id="fourier_uncertainty",
            name="Fourier Gabor Uncertainty Limit",
            domain="Signal Processing & Applied Mathematics",
            definition="The mathematical theorem stating that a function cannot be simultaneously sharply localized in both time and frequency domains.",
            formal_equation="Δt * Δf >= 1 / (4π)",
            core_principles=[
                "A short-duration impulse contains a broad spectrum of frequencies; a pure sine wave is completely unlocalized in time.",
                "Wavelet transforms balance time-frequency trade-offs via multi-resolution decomposition."
            ],
            boundary_conditions="Inherent mathematical property of continuous and discrete Fourier transforms.",
            related_concepts=["heisenberg_uncertainty", "information_entropy", "transformer_attention"],
            cross_disciplinary_analogy="Mathematically identical to the quantum Heisenberg principle: quantum wave mechanics is simply Fourier analysis applied to matter wavefunctions.",
            counter_intuitive_insight="You cannot ask 'what frequency is playing right now at this exact microsecond'—frequency requires temporal duration to exist."
        ),

        # -------------------------------------------------------------------
        # 4. Philosophy of Mind & Cognitive Science
        # -------------------------------------------------------------------
        "chinese_room": LatticeConcept(
            concept_id="chinese_room",
            name="The Chinese Room Argument (John Searle)",
            domain="Philosophy of Mind & AI Epistemology",
            definition="A famous thought experiment arguing that syntactic symbol manipulation according to formal algorithmic rules does not constitute semantic understanding or true intentional consciousness.",
            formal_equation="Syntax != Semantics (Rules(Symbols) != Meaning)",
            core_principles=[
                "A human locked in a room following rulebooks to translate Chinese characters creates the illusion of understanding without comprehending a single symbol.",
                "Counters 'Strong AI' functionalism: purely computational programs lack phenomenal consciousness and semantic grounding."
            ],
            boundary_conditions="Debated extensively by Systems Reply, Robot Reply, and Connectionist/Emergence theorists.",
            related_concepts=["global_workspace_theory", "dual_process_theory", "mechanistic_interpretability"],
            cross_disciplinary_analogy="Analogous to a compiled bytecode interpreter executing instructions without knowledge of the overarching business problem.",
            counter_intuitive_insight="An LLM can generate flawless essays about grief or quantum electrodynamics purely through next-token statistical correlation without subjective phenomenal experience."
        ),
        "global_workspace_theory": LatticeConcept(
            concept_id="global_workspace_theory",
            name="Global Workspace Theory (Baars / Dehaene)",
            domain="Cognitive Neuroscience & AI",
            definition="A cognitive architecture where specialized, parallel unconscious processors compete to broadcast information onto a central working memory 'stage' for global access.",
            formal_equation="Consciousness = Global Ignition + Synchronized Broadcasting across Distributed Modules",
            core_principles=[
                "Consciousness serves an integrative, broadcast function across decentralized specialized modules.",
                "Working memory acts as the conscious spotlight, coordinating perception, reasoning, language, and motor execution."
            ],
            boundary_conditions="Assumes cognitive systems have decentralized modular subsystems and a shared communicative blackboard.",
            related_concepts=["chinese_room", "dual_process_theory", "transformer_attention", "actor_model"],
            cross_disciplinary_analogy="Analogous to a publish-subscribe event bus or blackboard architecture in distributed microservices.",
            counter_intuitive_insight="Over 95% of human cognitive computation occurs unconsciously in parallel; conscious attention is a severely bottlenecked serial broadcast channel."
        ),
        "dual_process_theory": LatticeConcept(
            concept_id="dual_process_theory",
            name="Dual-Process Cognitive Theory (Kahneman / Tversky)",
            domain="Cognitive Psychology & Behavioral Economics",
            definition="A model positing two distinct modes of thought: System 1 (fast, instinctive, unconscious, emotional) and System 2 (slow, deliberative, logical, effortful).",
            formal_equation="Cognition = System 1 (Reflex / Heuristic) + System 2 (Deliberation / Metacognition)",
            core_principles=[
                "System 1 operates automatically with minimal effort but is susceptible to systematic cognitive biases.",
                "System 2 allocates deliberate attention for complex computation, audit, and logical validation."
            ],
            boundary_conditions="The two systems are not physical brain chambers, but functional operational modes.",
            related_concepts=["global_workspace_theory", "prospect_theory", "nash_equilibrium"],
            cross_disciplinary_analogy="Directly mirrors computer memory caching: L1 CPU cache (System 1) provides instant sub-nanosecond answers; fetching from SSD disk (System 2) executes thorough exhaustive retrieval.",
            counter_intuitive_insight="Most human errors occur not because System 2 reasons incorrectly, but because System 2 lazily endorses biased default answers generated by System 1."
        ),
        "active_inference_predictive_coding": LatticeConcept(
            concept_id="active_inference_predictive_coding",
            name="Active Inference & Free Energy Principle (Friston)",
            domain="Computational Neuroscience & Physics",
            definition="A unified theory of brain function asserting that biological organisms survive by continuously minimizing variational free energy (surprise) through predictive internal models and action.",
            formal_equation="F = E_{q}[ln q(s) - ln p(o, s)] = Complexity - Accuracy = KL(q(s) || p(s|o)) - ln p(o)",
            core_principles=[
                "The brain is a predictive engine: it does not passively receive sensation, but broadcasts top-down sensory predictions and computes bottom-up prediction errors.",
                "Active Inference: Organisms act upon the environment to change sensations into alignment with internal predictions."
            ],
            boundary_conditions="Requires high-dimensional generative internal models capable of continuous variational inference.",
            related_concepts=["information_entropy", "thermodynamic_entropy_second_law", "transformer_attention"],
            cross_disciplinary_analogy="Like wearing noise-canceling headphones: generating an inverted anti-sound wave to nullify incoming acoustic disorder.",
            counter_intuitive_insight="Perception is essentially 'controlled hallucination' constrained by incoming sensory prediction errors."
        ),
        "godel_incompleteness": LatticeConcept(
            concept_id="godel_incompleteness",
            name="Gödel's Incompleteness Theorems",
            domain="Mathematical Logic & Meta-Mathematics",
            definition="Two foundational theorems proving inherent limitations of every consistent formal axiomatic system capable of modeling basic arithmetic.",
            formal_equation="First: Consistent F => Incomplete F;  Second: Consistent F => F cannot prove Con(F)",
            core_principles=[
                "First Theorem: In any consistent formal system capable of basic arithmetic, there exist true statements that cannot be proven within the system.",
                "Second Theorem: No consistent system can prove its own consistency."
            ],
            boundary_conditions="Applies to recursively enumerable formal axiomatic systems expressive enough to represent Peano arithmetic.",
            related_concepts=["p_vs_np", "chinese_room", "information_entropy"],
            cross_disciplinary_analogy="Like trying to photograph the entire camera using the camera itself: the mechanism of observation cannot completely capture its own foundation.",
            counter_intuitive_insight="Mathematical truth is permanently larger than mathematical proof; no algorithm can ever capture all true mathematical theorems."
        ),

        # -------------------------------------------------------------------
        # 5. Economics, Game Theory & Systems Dynamics
        # -------------------------------------------------------------------
        "nash_equilibrium": LatticeConcept(
            concept_id="nash_equilibrium",
            name="Nash Equilibrium",
            domain="Economics & Game Theory",
            definition="A state in a non-cooperative game where no player has an incentive to unilaterally deviate from their chosen strategy given the strategies of all other players.",
            formal_equation="u_i(s_i^*, s_{-i}^*) >= u_i(s_i, s_{-i}^*)  for all s_i in S_i",
            core_principles=[
                "Stable strategic state where every agent plays a best response to peer strategies.",
                "Does not necessarily imply Pareto optimality (e.g., Prisoner's Dilemma leads to suboptimal mutual defection)."
            ],
            boundary_conditions="Assumes rational agents with common knowledge of game structure and payoff matrices.",
            related_concepts=["pareto_optimality", "cap_theorem", "byzantine_fault", "prospect_theory"],
            cross_disciplinary_analogy="Analogous to a stable local minimum in gradient descent optimization where no single parameter perturbation alone produces loss reduction.",
            counter_intuitive_insight="Individually rational optimal choices can mathematically guarantee collective systemic disaster when strategic incentives are misaligned."
        ),
        "pareto_optimality": LatticeConcept(
            concept_id="pareto_optimality",
            name="Pareto Optimality & Pareto Frontier",
            domain="Economics & Multi-Objective Optimization",
            definition="A state of allocation of resources from which it is impossible to reallocate so as to make any one individual better off without making at least one individual worse off.",
            formal_equation="Vector x dominates y if x_i >= y_i for all i and x_j > y_j for at least one j",
            core_principles=[
                "Represents the efficiency boundary in multi-objective trade-offs.",
                "First Fundamental Theorem of Welfare Economics: Competitive market equilibria under ideal conditions are Pareto efficient."
            ],
            boundary_conditions="Pareto optimality evaluates efficiency, not fairness, distribution equity, or social justice.",
            related_concepts=["nash_equilibrium", "cap_theorem", "prospect_theory"],
            cross_disciplinary_analogy="Directly mirrors the latency-throughput boundary in computer systems: you cannot lower latency further without sacrificing maximum throughput.",
            counter_intuitive_insight="A situation where one person holds 100% of wealth is Pareto optimal if taking one dollar to feed the starving harms the single owner."
        ),
        "prospect_theory": LatticeConcept(
            concept_id="prospect_theory",
            name="Prospect Theory (Kahneman & Tversky)",
            domain="Behavioral Economics",
            definition="A behavioral model demonstrating that humans evaluate outcomes relative to a reference point and are significantly more sensitive to losses than equivalent gains (loss aversion).",
            formal_equation="Value Function v(x): Convex for losses, Concave for gains; |v(-x)| > v(x) (~ 2x loss aversion)",
            core_principles=[
                "Loss Aversion: The psychological pain of losing $100 is roughly 2x more intense than the joy of gaining $100.",
                "Non-linear probability weighting: Humans overweight low-probability extreme events and underweight moderate/high probabilities."
            ],
            boundary_conditions="Applies to human decision-making under uncertainty and risk relative to subjective reference points.",
            related_concepts=["dual_process_theory", "nash_equilibrium", "pareto_optimality"],
            cross_disciplinary_analogy="Mirrors hysteresis in physics: the system's reaction curve on unloading does not follow the same path as on loading.",
            counter_intuitive_insight="People will take reckless irrational risks to avoid a certain loss, yet behave excessively risk-averse when protecting identical gains."
        ),
        "cybernetics_feedback_loops": LatticeConcept(
            concept_id="cybernetics_feedback_loops",
            name="Cybernetics & Homeostatic Feedback Loops",
            domain="Systems Engineering & Cybernetics",
            definition="The study of regulatory feedback systems, control mechanisms, and information flow governing machines, organisms, and autonomous agents (Norbert Wiener).",
            formal_equation="Error e(t) = Setpoint r(t) - Output y(t); Control u(t) = K_p * e(t) + K_i * int e + K_d * de/dt",
            core_principles=[
                "Negative Feedback: Dampens deviations and stabilizes dynamic systems toward homeostatic equilibrium.",
                "Positive Feedback: Amplifies perturbations leading to exponential growth, cascade tipping points, or runaway collapse."
            ],
            boundary_conditions="Feedback loops with excessive delay (latency) induce severe oscillatory instability and resonance.",
            related_concepts=["actor_model", "active_inference_predictive_coding", "thermodynamic_entropy_second_law"],
            cross_disciplinary_analogy="Functions identically across a building thermostat, biological glucose regulation, and an RL agent's reward-driven policy updates.",
            counter_intuitive_insight="Adding more aggressive control response to an unstable system with feedback delay exacerbates catastrophic oscillations rather than calming them."
        ),
        "mechanism_design": LatticeConcept(
            concept_id="mechanism_design",
            name="Mechanism Design (Reverse Game Theory)",
            domain="Microeconomics & Cryptographic Systems",
            definition="A field of economics that designs game rules, incentive structures, and protocols to achieve a desired systemic outcome despite self-interested, strategic participants.",
            formal_equation="Incentive Compatibility: u_i(v_i, v_i) >= u_i(v_i', v_i) (Truthful bidding is dominant)",
            core_principles=[
                "Revelation Principle: Any outcome achieved by any mechanism can be achieved by a direct-revelation mechanism where agents report their true preferences truthfully.",
                "Vickrey-Clarke-Groves (VCG) auctions align private incentives with social welfare maximization."
            ],
            boundary_conditions="Assumes participants are strategically self-interested and compute best responses.",
            related_concepts=["nash_equilibrium", "byzantine_fault", "cap_theorem"],
            cross_disciplinary_analogy="Like designing traffic road layouts and roundabouts such that every driver's selfish desire to arrive quickly naturally prevents gridlock for everyone.",
            counter_intuitive_insight="In a second-price sealed-bid auction, your mathematically optimal strategy is always to bid your exact true valuation, regardless of what anyone else bids."
        )
    }

    @classmethod
    @lru_cache(maxsize=512)
    def find_concept(cls, query: str) -> Optional[LatticeConcept]:
        q_lower = query.lower()
        q_norm = q_lower.replace("&", " and ").replace("-", " ")
        
        # 1. Direct name or concept_id match
        for cid, concept in cls.LATTICE_GRAPH.items():
            c_name_norm = concept.name.lower().replace("&", " and ").replace("-", " ")
            if concept.name.lower() in q_lower or c_name_norm in q_norm or cid.replace("_", " ") in q_norm:
                return concept

        # 2. Tokenized words match
        for cid, concept in cls.LATTICE_GRAPH.items():
            words = cid.split("_")
            if len(words) >= 2 and all(w in q_norm for w in words):
                return concept

        # 3. Match distinct key phrases from name
        for cid, concept in cls.LATTICE_GRAPH.items():
            name_words = [w for w in re.findall(r'\w+', concept.name.lower()) if len(w) > 3 and w not in ("principle", "theorem", "theory", "argument", "model", "analysis")]
            if name_words and sum(1 for w in name_words if w in q_norm) >= 2:
                return concept

        # 4. Secondary match in related concepts
        for cid, concept in cls.LATTICE_GRAPH.items():
            for rel in concept.related_concepts:
                if rel.replace("_", " ") in q_norm:
                    return concept

        return None

    @classmethod
    def synthesize_cross_disciplinary(cls, query: str) -> Optional[str]:
        concept = cls.find_concept(query)
        if not concept:
            return None

        out = [
            f"### Cross-Disciplinary Knowledge Synapse: {concept.name}",
            f"**Primary Domain**: *{concept.domain}*\n",
            f"**Formal Definition**:\n> {concept.definition}\n"
        ]

        if concept.formal_equation:
            out.append(f"**Governing Equation / Invariant**:\n`{concept.formal_equation}`\n")

        out.append("**Foundational Principles**:")
        for p in concept.core_principles:
            out.append(f"• {p}")

        if concept.boundary_conditions:
            out.append(f"\n**Boundary Constraints**: {concept.boundary_conditions}")

        if concept.cross_disciplinary_analogy:
            out.append(f"\n**Cross-Disciplinary Bridge & Analogy**:\n{concept.cross_disciplinary_analogy}")

        if concept.counter_intuitive_insight:
            out.append(f"\n💡 **Counter-Intuitive Insight**: {concept.counter_intuitive_insight}")

        if concept.related_concepts:
            related_names = [cls.LATTICE_GRAPH[rc].name for rc in concept.related_concepts if rc in cls.LATTICE_GRAPH]
            if related_names:
                out.append(f"\n**Associated Lattice Concepts**: {', '.join(related_names)}")

        return "\n".join(out)
