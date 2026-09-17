import os
import sys
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from collision.cognitive.reasoning import CognitiveReasoningEngine, CognitiveReasoningResult
from collision.cognitive.strategy import CognitiveStrategyEngine, StrategicAnalysisResult
from collision.cognitive.dialogue import CognitiveDialogueEngine, DialogueTurnResult
from collision.cognitive.knowledge import CrossDomainKnowledgeBase, DomainKnowledgeResult
from collision.cognitive.engine import CollisionCognitiveEngine, CognitiveResponse
from collision.service import CollisionService


class TestCognitiveReasoning:
    def test_think_step_by_step(self):
        res = CognitiveReasoningEngine.think_step_by_step("Why cannot information travel faster than light?")
        assert isinstance(res, CognitiveReasoningResult)
        assert res.reasoning_type == "ChainOfThought"
        assert len(res.steps) >= 4
        assert "Step-by-Step Chain of Thought" in res.formatted_output
        assert "Conclusion & Key Takeaway" in res.formatted_output

    def test_solve_5_whys(self):
        res = CognitiveReasoningEngine.solve_5_whys("5 whys on why the database latency spiked")
        assert isinstance(res, CognitiveReasoningResult)
        assert res.reasoning_type == "5WhysRCA"
        assert len(res.steps) == 5
        assert "Root Cause" in res.formatted_output
        assert "Action Plan" in res.formatted_output

    def test_fermi_estimation(self):
        res = CognitiveReasoningEngine.solve_fermi_estimation("fermi estimation of piano tuners in Chicago")
        assert isinstance(res, CognitiveReasoningResult)
        assert res.reasoning_type == "FermiEstimation"
        assert "Order-of-Magnitude" in res.formatted_output

    def test_problem_solving_framework(self):
        res = CognitiveReasoningEngine.solve_problem_systematically("how to systematically solve high memory fragmentation")
        assert isinstance(res, CognitiveReasoningResult)
        assert "Problem-Solving Framework" in res.formatted_output


class TestCognitiveStrategy:
    def test_cialdini_principles(self):
        res = CognitiveStrategyEngine.explain_cialdini_principles("What are the 6 principles of influence by Robert Cialdini?")
        assert isinstance(res, StrategicAnalysisResult)
        assert res.analysis_type == "CialdiniInfluence"
        assert len(res.principles_identified) == 6
        assert "Reciprocity" in res.formatted_output
        assert "Scarcity" in res.formatted_output
        assert "Social Proof" in res.formatted_output

    def test_negotiation_strategy(self):
        res = CognitiveStrategyEngine.explain_negotiation_strategy("How to negotiate salary using BATNA and anchoring?")
        assert isinstance(res, StrategicAnalysisResult)
        assert "BATNA" in res.formatted_output
        assert "Anchoring" in res.formatted_output
        assert "Calibrated Questions" in res.formatted_output

    def test_logical_fallacies(self):
        res = CognitiveStrategyEngine.explain_fallacies("Explain logical fallacies like ad hominem and strawman")
        assert isinstance(res, StrategicAnalysisResult)
        assert "Ad Hominem" in res.formatted_output
        assert "Strawman" in res.formatted_output
        assert "False Dilemma" in res.formatted_output

    def test_cognitive_biases(self):
        res = CognitiveStrategyEngine.explain_biases("What are the common cognitive biases in decision making?")
        assert isinstance(res, StrategicAnalysisResult)
        assert "Confirmation Bias" in res.formatted_output
        assert "Sunk Cost Fallacy" in res.formatted_output
        assert "Anchoring Bias" in res.formatted_output

    def test_persuasive_framing(self):
        res = CognitiveStrategyEngine.explain_persuasive_framing("How does persuasive framing and loss aversion work?")
        assert isinstance(res, StrategicAnalysisResult)
        assert "Prospect Theory" in res.formatted_output


class TestCognitiveDialogue:
    def test_humor_jokes(self):
        res = CognitiveDialogueEngine.analyze_dialogue("Tell me a joke")
        assert isinstance(res, DialogueTurnResult)
        assert res.intent_category == "humor"
        assert len(res.response) > 10

    def test_empathy_support(self):
        res = CognitiveDialogueEngine.analyze_dialogue("I feel overwhelmed and stressed with my projects")
        assert isinstance(res, DialogueTurnResult)
        assert res.intent_category == "empathy"
        assert "Step Away for 10 Minutes" in res.response

    def test_philosophy_consciousness(self):
        res = CognitiveDialogueEngine.analyze_dialogue("Are you conscious or do you have feelings?")
        assert isinstance(res, DialogueTurnResult)
        assert res.intent_category == "philosophy"
        assert "Chinese Room" in res.response or "Hard Problem" in res.response

    def test_creative_brainstorming(self):
        res = CognitiveDialogueEngine.analyze_dialogue("Suggest some creative AI project ideas")
        assert isinstance(res, DialogueTurnResult)
        assert res.intent_category == "brainstorming"
        assert "Local-First" in res.response

    def test_learning_advice(self):
        res = CognitiveDialogueEngine.analyze_dialogue("How to learn coding fast and productivity tips")
        assert isinstance(res, DialogueTurnResult)
        assert res.intent_category == "advice"
        assert "Feynman Technique" in res.response


class TestCrossDomainKnowledge:
    def test_transformer_knowledge(self):
        res = CrossDomainKnowledgeBase.query_knowledge("Explain the transformer architecture and attention mechanism")
        assert isinstance(res, DomainKnowledgeResult)
        assert res.domain == "Artificial Intelligence & ML"
        assert "Scaled Dot-Product Attention" in res.formatted_output

    def test_slm_vs_llm_knowledge(self):
        res = CrossDomainKnowledgeBase.query_knowledge("What is a small language model vs heavy LLM for CPU inference?")
        assert isinstance(res, DomainKnowledgeResult)
        assert "Memory Bandwidth" in res.formatted_output

    def test_cap_theorem_knowledge(self):
        res = CrossDomainKnowledgeBase.query_knowledge("Explain the CAP theorem in distributed systems")
        assert isinstance(res, DomainKnowledgeResult)
        assert res.domain == "Computer Science & Systems"
        assert "Consistency" in res.formatted_output
        assert "Partition Tolerance" in res.formatted_output

    def test_quantum_mechanics_knowledge(self):
        res = CrossDomainKnowledgeBase.query_knowledge("What is quantum superposition and entanglement in quantum mechanics?")
        assert isinstance(res, DomainKnowledgeResult)
        assert res.domain == "Physics & Mathematics"
        assert "Heisenberg Uncertainty" in res.formatted_output

    def test_ethics_knowledge(self):
        res = CrossDomainKnowledgeBase.query_knowledge("What is the difference between utilitarianism and deontology in ethics?")
        assert isinstance(res, DomainKnowledgeResult)
        assert res.domain == "Philosophy & Ethics"
        assert "Categorical Imperative" in res.formatted_output

    def test_game_theory_knowledge(self):
        res = CrossDomainKnowledgeBase.query_knowledge("What is a Nash equilibrium in game theory and prisoner's dilemma?")
        assert isinstance(res, DomainKnowledgeResult)
        assert res.domain == "Economics & Game Theory"
        assert "Prisoner's Dilemma" in res.formatted_output


class TestCognitiveServiceIntegration:
    @pytest.fixture
    def service(self):
        return CollisionService()

    def test_service_persuasion_query(self, service):
        res = service.ask("What are the 6 principles of influence by Robert Cialdini?")
        assert res["status"] == "ANSWERED"
        assert res["mode"] == "MODEL"
        assert res["confidence"] == 1.0
        assert "Reciprocity" in res["answer"]

    def test_service_reasoning_query(self, service):
        res = service.ask("Think step by step: Why can't information travel faster than light?")
        assert res["status"] == "ANSWERED"
        assert "Step-by-Step Chain of Thought" in res["answer"]

    def test_service_cross_domain_query(self, service):
        res = service.ask("Explain the SOLID principles of software design")
        assert res["status"] == "ANSWERED"
        assert "Single Responsibility" in res["answer"]

    def test_service_dialogue_query(self, service):
        res = service.ask("Tell me a funny joke")
        assert res["status"] == "ANSWERED"
        assert len(res["answer"]) > 10
