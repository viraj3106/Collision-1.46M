"""
COLLISION Synaptic-GWT Cognitive Brain — Comprehensive Unit Test Suite.

Validates:
1. Synaptic Working Memory (GWT-SWM) retention, LTP reinforcement, temporal decay, and capacity pruning
2. Global Workspace message competition, salience weighting, and broadcasting
3. Dual-Process System 1 vs System 2 routing, Epistemic Shannon Entropy, and Metacognitive Critic checks
4. Graph-of-Thoughts (GoT) DAG construction & Hegelian Dialectical Synthesis (Thesis -> Antithesis -> Synthesis)
5. Synaptic NLP Information Density profiling, Semantic Knowledge Triplet extraction, and Contextual Polysemy disambiguation
6. Cross-Domain Knowledge Lattice & Multi-hop conceptual bridge synthesis
7. End-to-end CollisionBrain.think() and CollisionBrain.process()
8. CollisionService.think() integration
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
    DualProcessArbiter,
    EpistemicUncertaintyQuantifier,
    MetacognitiveCritic,
    HegelianDialecticEngine,
    GraphOfThoughtReasoner,
    DialecticalGraph,
    ThoughtNode,
    ThoughtType,
    CognitiveModality,
    SemanticTripletExtractor,
    ContextualPolysemyDisambiguator,
    CrossDomainKnowledgeLattice,
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
        assert gw.working_memory.recall(f"broadcast_DialecticEngine_1") is not None

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
        assert complex_prof.epistemic_ambiguity_score > simple_prof.epistemic_ambiguity_score

    def test_dual_process_routing(self):
        sys1_query = "hello"
        mod1, unc1, rat1 = DualProcessArbiter.decide_route(sys1_query)
        assert mod1 == CognitiveModality.SYSTEM_1_REFLEX

        sys2_query = "Compare and contrast the trade-offs between CAP Theorem and Quantum Entanglement in distributed databases"
        mod2, unc2, rat2 = DualProcessArbiter.decide_route(sys2_query)
        assert mod2 == CognitiveModality.SYSTEM_2_DELIBERATION
        assert "System 2" in rat2

    def test_metacognitive_critic(self):
        biased_thoughts = ["This solution is obviously always the best without question for every scenario."]
        notes = MetacognitiveCritic.audit_thoughts(biased_thoughts)
        assert any("Confirmation Bias" in n for n in notes)

        valid_thoughts = ["Under normal operating conditions with bounded network partition latency, consistency is preserved."]
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
            domain="Distributed Systems"
        )

        assert isinstance(graph, DialecticalGraph)
        assert len(graph.nodes) == 3
        assert graph.thesis_id in graph.nodes
        assert graph.antithesis_id in graph.nodes
        assert graph.synthesis_id in graph.nodes
        assert graph.dialectic_resolved is True
        assert "Dialectical Synthesis" in synthesis
        assert "Thesis" in synthesis
        assert "Antithesis" in synthesis

    def test_graph_of_thought_reasoner(self):
        query = "How to systematically isolate a memory leak in a high-throughput microservice"
        facts = [
            "Capture heap dumps at 10-minute intervals",
            "Identify retained object graphs with monotonic growth",
            "Profile GC pause times and generation tenuring rates"
        ]
        graph = GraphOfThoughtReasoner.build_graph(query, facts)
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

        elec_text = "The substation step-up transformer stepped up AC voltage on the electrical grid coil."
        elec_senses = ContextualPolysemyDisambiguator.disambiguate(elec_text)
        assert "transformer" in elec_senses
        assert "Electromagnetic" in elec_senses["transformer"]


class TestCrossDomainLattice:
    def test_cross_domain_concept_lookup(self):
        res = CrossDomainKnowledgeLattice.synthesize_cross_disciplinary("Explain CAP Theorem and its analogy")
        assert res is not None
        assert "CAP Theorem" in res
        assert "Heisenberg" in res

    def test_chinese_room_concept(self):
        res = CrossDomainKnowledgeLattice.synthesize_cross_disciplinary("What is the Chinese Room argument by John Searle?")
        assert res is not None
        assert "Chinese Room" in res
        assert "Syntax does not equate to Semantics" in res


class TestCollisionBrainEndToEnd:
    @pytest.fixture
    def brain(self):
        return CollisionBrain()

    def test_brain_think_deliberation(self, brain):
        query = "Is strong consistency fundamentally irreconcilable with partition tolerance?"
        res = brain.think(query, domain="Distributed Systems")
        assert isinstance(res, BrainResponse)
        assert res.modality == CognitiveModality.DIALECTICAL_SYNTHESIS
        assert res.confidence >= 0.95
        assert res.epistemic_certainty > 0.80
        assert res.trace is not None
        assert res.trace.graph_of_thoughts is not None
        assert len(res.trace.graph_of_thoughts.nodes) >= 3
        assert len(res.key_insights) >= 3
        assert len(res.followup_hypotheses) >= 1

    def test_brain_process_system1(self, brain):
        query = "hello"
        res = brain.process(query)
        assert isinstance(res, BrainResponse)
        assert res.modality in (CognitiveModality.SYSTEM_1_REFLEX, CognitiveModality.CROSS_DOMAIN_SYNAPSE)
        assert res.confidence >= 0.85

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
        assert res["confidence"] >= 0.95
        assert "Thesis" in res["answer"] or "Synthesis" in res["answer"] or "Dialectical" in res["answer"]
        assert len(res["key_insights"]) > 0
        assert res["graph_summary"]["nodes_count"] >= 3
        assert res["latency"]["total_ms"] > 0
