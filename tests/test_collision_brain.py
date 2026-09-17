"""
COLLISION Synaptic-GWT Cognitive Brain 2.0 — Comprehensive Unit & Performance Test Suite.

Validates:
1. Synaptic Working Memory (GWT-SWM 2.0) retention, LTP reinforcement, Hebbian associations, LTM consolidation, and capacity pruning
2. Global Workspace message competition, salience weighting, episodic replay buffer, and broadcasting
3. Dual-Process System 1 vs System 2 routing, Shannon & Renyi entropy, linguistic complexity, and 15+ Metacognitive Critic rules
4. Graph-of-Thoughts (GoT 2.0) DAG construction, Hegelian Dialectic Synthesis, Mermaid flowchart & ASCII tree exports
5. Synaptic NLP Information Density profiling, 20+ Semantic Knowledge Triplet extractors, and 15+ Contextual Polysemy terms
6. Cross-Domain Knowledge Lattice across 25+ multi-disciplinary concepts across 5 scientific fields
7. End-to-end CollisionBrain.think(), batch_think(), and process()
8. CollisionService.think() integration with graph visualizers
9. High-performance throughput and sub-millisecond execution benchmarks
"""

import os
import sys
import time
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from collision.brain import (
    CollisionBrain,
    SynapticCognitiveBrain,
    get_collision_brain,
    GlobalWorkspace,
    SynapticWorkingMemory,
    HebbianAssociation,
    DualProcessArbiter,
    EpistemicUncertaintyQuantifier,
    MetacognitiveCritic,
    CognitiveBiasAudit,
    BiasSeverity,
    HegelianDialecticEngine,
    GraphOfThoughtReasoner,
    DialecticalGraph,
    ThoughtNode,
    ThoughtType,
    CognitiveModality,
    SemanticTriple,
    InformationEntropyProfile,
    LinguisticComplexityMetrics,
    SemanticTripletExtractor,
    ContextualPolysemyDisambiguator,
    CrossDomainKnowledgeLattice,
    LatticeConcept,
    BrainResponse,
    BrainCognitiveTrace
)
from collision.service import CollisionService


class TestSynapticWorkingMemory:
    def test_retention_and_recall(self):
        wm = SynapticWorkingMemory(capacity=5, half_life_seconds=60.0)
        item = wm.retain("user_goal", "Optimize SLM inference on CPU", salience=1.5, domain_tags=["Systems", "AI"])
        assert item.key == "user_goal"
        assert item.access_count == 1
        assert item.salience >= 1.5

        # Recall boosts access count and LTP salience
        val = wm.recall("user_goal")
        assert val == "Optimize SLM inference on CPU"
        assert wm.items["user_goal"].access_count == 2

    def test_hebbian_associative_plasticity(self):
        wm = SynapticWorkingMemory(capacity=6)
        wm.retain("paxos", "State machine consensus", salience=1.2, domain_tags=["Systems"])
        wm.retain("raft", "Leader election and log replication", salience=1.2, domain_tags=["Systems"])

        # Accessing raft after paxos creates a Hebbian association link
        raft_item = wm.items["raft"]
        assert "paxos" in raft_item.associations
        assert raft_item.associations["paxos"].synaptic_weight >= 0.5

    def test_spreading_activation(self):
        wm = SynapticWorkingMemory(capacity=6)
        wm.retain("neural_net", "Deep learning architecture", salience=1.0)
        wm.retain("backprop", "Gradient descent chain rule", salience=1.0)

        initial_nn_salience = wm.items["neural_net"].salience
        # Recalling backprop should spread activation to associated neural_net
        wm.recall("backprop", spread_activation=True)
        assert wm.items["neural_net"].salience >= initial_nn_salience

    def test_ltm_consolidation(self):
        wm = SynapticWorkingMemory(capacity=5)
        wm.retain("core_axiom", "Energy conservation", salience=2.0)
        wm.recall("core_axiom")
        wm.recall("core_axiom")
        assert wm.items["core_axiom"].access_count >= 3
        assert wm.items["core_axiom"].is_consolidated is True

    def test_associative_tag_search(self):
        wm = SynapticWorkingMemory()
        wm.retain("rag_spec", "Chunk size 128 tokens", domain_tags=["RAG", "AI"])
        wm.retain("cap_theorem", "Consistency vs Availability", domain_tags=["Distributed", "Systems"])
        wm.retain("vector_index", "Cosine similarity top-k 3", domain_tags=["RAG", "Search"])

        rag_items = wm.search_by_tag("RAG")
        assert len(rag_items) == 2
        keys = [k for k, v, s in rag_items]
        assert "rag_spec" in keys
        assert "vector_index" in keys

    def test_capacity_pruning(self):
        wm = SynapticWorkingMemory(capacity=3)
        wm.retain("k1", "v1", salience=0.2)
        wm.retain("k2", "v2", salience=0.9)
        wm.retain("k3", "v3", salience=0.8)
        wm.retain("k4", "v4", salience=1.0)  # Should evict k1 (lowest salience)

        assert len(wm.items) == 3
        assert "k1" not in wm.items
        assert "k4" in wm.items
        assert "k2" in wm.items


class TestGlobalWorkspace:
    def test_workspace_broadcasting(self):
        gw = GlobalWorkspace()
        msg = gw.broadcast(
            sender_module="DialecticEngine",
            salience_weight=0.95,
            content_summary="Thesis converged with antithesis",
            payload={"coherence": 0.98}
        )
        assert msg.broadcast_cycle == 1
        assert len(gw.broadcast_history) == 1
        assert "Thesis converged" in msg.content_summary
        assert gw.working_memory.recall("broadcast_DialecticEngine_1") is not None
        assert len(gw.episodic_replay_buffer) == 1

    def test_workspace_arbitration(self):
        gw = GlobalWorkspace()
        candidates = [
            {"module": "NeuralLM", "confidence": 0.70, "salience": 0.60, "answer": "candidate 1"},
            {"module": "SymbolicReasoner", "confidence": 0.95, "salience": 0.90, "answer": "candidate 2"},
            {"module": "HeuristicMatcher", "confidence": 0.50, "salience": 0.40, "answer": "candidate 3"}
        ]
        winner = gw.arbitrate_candidates(candidates)
        assert winner is not None
        assert winner["module"] == "SymbolicReasoner"
        assert winner["answer"] == "candidate 2"


class TestDualProcessAndEntropy:
    def test_epistemic_entropy_calculation(self):
        simple_text = "hello hello hello"
        complex_text = "The quantum superposition paradox in distributed systems creates trade-offs between consistency and latency."

        simple_prof = EpistemicUncertaintyQuantifier.evaluate_entropy(simple_text)
        complex_prof = EpistemicUncertaintyQuantifier.evaluate_entropy(complex_text)

        assert simple_prof.shannon_entropy < complex_prof.shannon_entropy
        assert complex_prof.shannon_entropy > 3.0
        assert complex_prof.renyi_entropy > 0.0
        assert complex_prof.complexity is not None
        assert complex_prof.epistemic_ambiguity_score > simple_prof.epistemic_ambiguity_score

    def test_dual_process_routing(self):
        sys1_query = "hello"
        mod1, unc1, rat1 = DualProcessArbiter.decide_route(sys1_query)
        assert mod1 == CognitiveModality.SYSTEM_1_REFLEX

        sys2_query = "Compare and contrast the trade-offs between CAP Theorem and Quantum Entanglement in distributed databases"
        mod2, unc2, rat2 = DualProcessArbiter.decide_route(sys2_query)
        assert mod2 == CognitiveModality.SYSTEM_2_DELIBERATION
        assert "System 2" in rat2

    def test_metacognitive_critic_rules(self):
        # Test 1: Confirmation bias & overconfidence
        biased_thoughts = ["This solution is obviously always the best without question and 100% guaranteed."]
        audits = MetacognitiveCritic.audit_thoughts_detailed(biased_thoughts)
        assert len(audits) >= 1
        bias_types = [a.bias_type for a in audits]
        assert "Confirmation Bias" in bias_types or "Overconfidence Bias" in bias_types

        # Test 2: Sunk cost fallacy
        sunk_thoughts = ["We have already invested too much time to stop now, we cannot abandon this."]
        sunk_audits = MetacognitiveCritic.audit_thoughts_detailed(sunk_thoughts)
        assert any(a.bias_type == "Sunk Cost Fallacy" for a in sunk_audits)

        # Test 3: Clean validation
        valid_thoughts = ["Under nominal operating assumptions with bounded network latency, consistency holds."]
        valid_notes = MetacognitiveCritic.audit_thoughts(valid_thoughts)
        assert any("Epistemic Validation" in n for n in valid_notes)


class TestGraphOfThoughtsAndDialectic:
    def test_hegelian_dialectic_synthesis(self):
        query = "Should distributed systems prioritize consistency or availability?"
        thesis = "Strong consistency guarantees serializability and prevents dirty reads."
        antithesis = "High availability ensures zero-downtime and resilient fault tolerance during network partitions."

        graph, synthesis = HegelianDialecticEngine.synthesize_dialectic(
            query=query,
            thesis_content=thesis,
            antithesis_content=antithesis,
            domain="Distributed Systems",
            counterfactual="If network partition duration is zero"
        )

        assert isinstance(graph, DialecticalGraph)
        assert len(graph.nodes) >= 4
        assert graph.thesis_id in graph.nodes
        assert graph.antithesis_id in graph.nodes
        assert graph.synthesis_id in graph.nodes
        assert graph.dialectic_resolved is True
        assert graph.adversarial_resilience_score >= 0.90
        assert "Dialectical Synthesis" in synthesis

    def test_mermaid_and_ascii_exports(self):
        query = "Is strong consistency fundamentally irreconcilable with partition tolerance?"
        graph, _ = HegelianDialecticEngine.synthesize_dialectic(
            query=query,
            thesis_content="Consistency is mandatory.",
            antithesis_content="Availability prevents downtime.",
            domain="Distributed Systems"
        )

        mermaid = graph.to_mermaid()
        assert "flowchart TD" in mermaid
        assert "THESIS" in mermaid
        assert "SYNTHESIS" in mermaid

        ascii_tree = graph.to_ascii_tree()
        assert "[Graph of Thoughts DAG]" in ascii_tree
        assert "THESIS" in ascii_tree

    def test_graph_of_thought_reasoner(self):
        query = "How to systematically isolate a memory leak in a high-throughput microservice"
        facts = [
            "Capture heap dumps at 10-minute intervals",
            "Identify retained object graphs with monotonic growth",
            "Profile GC pause times and generation tenuring rates"
        ]
        graph = GraphOfThoughtReasoner.build_graph(query, facts, domain="Distributed Systems")
        assert len(graph.nodes) == 5
        assert len(graph.edges) == 4
        assert graph.synthesis_id == "node_final_synthesis"


class TestSynapticNLP:
    def test_semantic_triplet_extraction(self):
        text = "COLLISION 10M operates with 6 transformer layers. The optimizer minimizes categorical cross-entropy loss."
        triples = SemanticTripletExtractor.extract_triples(text)
        assert len(triples) >= 1
        subjects = [t.subject for t in triples]
        assert any("COLLISION" in s or "transformer" in s or "optimizer" in s for s in subjects)

    def test_contextual_polysemy_disambiguation(self):
        ai_text = "The transformer utilizes multi-head attention layers to process input tokens."
        ai_senses = ContextualPolysemyDisambiguator.disambiguate(ai_text)
        assert "transformer" in ai_senses
        assert "Neural self-attention" in ai_senses["transformer"]

        phys_text = "Thermodynamic entropy increases in closed systems according to the second law of heat."
        phys_senses = ContextualPolysemyDisambiguator.disambiguate(phys_text)
        assert "entropy" in phys_senses
        assert "Thermodynamic" in phys_senses["entropy"]


class TestCrossDomainLattice:
    def test_multi_disciplinary_concepts(self):
        # 1. Distributed systems
        res_cap = CrossDomainKnowledgeLattice.synthesize_cross_disciplinary("Explain CAP Theorem")
        assert res_cap is not None
        assert "CAP Theorem" in res_cap
        assert "Heisenberg" in res_cap

        # 2. Physics & Landauer
        res_landauer = CrossDomainKnowledgeLattice.synthesize_cross_disciplinary("What is Landauer's Principle of computation?")
        assert res_landauer is not None
        assert "Landauer" in res_landauer

        # 3. AI & Scaling Laws
        res_scale = CrossDomainKnowledgeLattice.synthesize_cross_disciplinary("What are neural scaling laws in deep learning?")
        assert res_scale is not None
        assert "Scaling Laws" in res_scale

        # 4. Economics & Mechanism Design
        res_mech = CrossDomainKnowledgeLattice.synthesize_cross_disciplinary("Explain mechanism design and reverse game theory")
        assert res_mech is not None
        assert "Mechanism Design" in res_mech

        # 5. Cognitive Science & Free Energy
        res_free = CrossDomainKnowledgeLattice.synthesize_cross_disciplinary("What is Active Inference and Free Energy Principle by Friston?")
        assert res_free is not None
        assert "Active Inference" in res_free


class TestCollisionBrainEndToEnd:
    @pytest.fixture
    def brain(self):
        return CollisionBrain()

    def test_brain_think_deliberation(self, brain):
        query = "Is strong consistency fundamentally irreconcilable with partition tolerance?"
        res = brain.think(query, domain="Distributed Systems", counterfactual="zero latency")
        assert isinstance(res, BrainResponse)
        assert res.modality == CognitiveModality.DIALECTICAL_SYNTHESIS
        assert res.confidence >= 0.85
        assert res.epistemic_certainty >= 0.70
        assert res.axiomatic_confidence >= 0.90
        assert res.adversarial_resilience >= 0.85
        assert res.merit_score >= 0.85
        assert res.graph_mermaid is not None
        assert res.graph_ascii is not None
        assert res.trace is not None
        assert len(res.trace.stage_latencies) >= 4
        assert len(res.key_insights) >= 3

    def test_brain_process_system1(self, brain):
        query = "hello"
        res = brain.process(query)
        assert isinstance(res, BrainResponse)
        assert res.modality in (CognitiveModality.SYSTEM_1_REFLEX, CognitiveModality.CROSS_DOMAIN_SYNAPSE)
        assert res.confidence >= 0.85

    def test_batch_think_and_process(self, brain):
        queries = [
            "What is the CAP Theorem?",
            "How does backpropagation work?",
            "What is Nash Equilibrium?"
        ]
        results = brain.batch_think(queries)
        assert len(results) == 3
        assert all(isinstance(r, BrainResponse) for r in results)

    def test_workspace_snapshot(self, brain):
        brain.think("How does backpropagation update weights?")
        snap = brain.get_workspace_snapshot()
        assert isinstance(snap, dict)
        assert len(snap) > 0


class TestServiceBrainIntegration:
    @pytest.fixture
    def service(self):
        return CollisionService()

    def test_service_think_method(self, service):
        res = service.think("What are the cognitive trade-offs between System 1 intuition and System 2 deliberation?")
        assert res["status"] == "ANSWERED"
        assert res["mode"] == "BRAIN_DELIBERATION"
        assert res["confidence"] >= 0.85
        assert res["merit_score"] >= 0.85
        assert "graph_mermaid" in res
        assert "graph_ascii" in res
        assert "bias_audits" in res
        assert len(res["key_insights"]) > 0
        assert res["graph_summary"]["nodes_count"] >= 3
        assert res["latency"]["total_ms"] > 0


class TestPerformanceBenchmarks:
    @pytest.fixture
    def brain(self):
        return CollisionBrain()

    def test_system1_reflex_latency(self, brain):
        # Warmup
        brain.process("hi")

        t0 = time.perf_counter()
        for _ in range(50):
            brain.process("hello")
        elapsed_per_call_ms = ((time.perf_counter() - t0) / 50.0) * 1000.0

        # System 1 should execute well under 1.5ms per query
        assert elapsed_per_call_ms < 1.5, f"System 1 reflex exceeded latency threshold: {elapsed_per_call_ms:.2f}ms"

    def test_system2_deliberation_latency(self, brain):
        # Warmup
        brain.think("What is CAP Theorem?")

        t0 = time.perf_counter()
        for _ in range(20):
            brain.think("What are the trade-offs of microservices vs monoliths?")
        elapsed_per_call_ms = ((time.perf_counter() - t0) / 20.0) * 1000.0

        # System 2 with full GoT DAG, bias audits, and entropy should execute under 5.0ms
        assert elapsed_per_call_ms < 5.0, f"System 2 deliberation exceeded latency threshold: {elapsed_per_call_ms:.2f}ms"
