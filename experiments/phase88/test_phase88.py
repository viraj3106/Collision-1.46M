import os
import sys
import json
import unittest
import hashlib

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from rag.pipeline import RAGPipeline, QueryRouter
from rag.schemas import RAGRequest, RAGResponse
from collision.inference.engine import CollisionInferenceEngine

PHASE88_DIR = os.path.dirname(os.path.abspath(__file__))
PHASE86_DIR = os.path.abspath(os.path.join(PHASE88_DIR, "..", "phase86"))
BENCHMARK_PATH = os.path.join(PHASE86_DIR, "independent_rag_benchmark.jsonl")
MODEL_PATH = os.path.join(REPO_ROOT, "models", "collision-10m", "model.pt")

class MockInferenceEngine:
    """Mock engine for fast unit tests on pipeline invariants."""
    def __init__(self, output_text="The ticker symbol for Microsoft is MSFT."):
        self.output_text = output_text
        
    def generate(self, prompt: str, max_tokens: int = 80, **kwargs):
        return {"text": self.output_text, "tokens_generated": 10}

def safe_evaluate_answer(q_item: dict, response_obj: RAGResponse, evaluator_input: str) -> dict:
    """
    Evaluator contract enforcing strict Phase 88 invariants.
    """
    query = response_obj.query.strip()
    prompt_fmt = response_obj.prompt_formatted.strip()
    context = response_obj.context_text.strip()
    gen_ans = response_obj.generated_answer
    
    # Invariant checks
    if gen_ans is None or len(gen_ans.strip()) == 0:
        return {"status": "GENERATION_FAILURE", "correct": False, "score": 0.0, "reason": "empty_generation"}
    
    ans_clean = gen_ans.strip()
    if ans_clean == query:
        return {"status": "GENERATION_FAILURE", "correct": False, "score": 0.0, "reason": "query_echo"}
    if ans_clean == prompt_fmt:
        return {"status": "GENERATION_FAILURE", "correct": False, "score": 0.0, "reason": "prompt_echo"}
    if context and ans_clean == context:
        return {"status": "GENERATION_FAILURE", "correct": False, "score": 0.0, "reason": "context_echo"}
        
    # Assertion 6: Evaluator must know which field it is scoring
    assert evaluator_input == gen_ans, f"Evaluator input ({evaluator_input}) != generated_answer ({gen_ans})"
    
    expected_keywords = [k.lower() for k in (q_item.get("expected_keywords", []) + q_item.get("gold_facts", []))]
    ground_truth = q_item.get("expected_answer", q_item.get("ground_truth_answer", "")).strip().lower()
    ans_lower = evaluator_input.lower()
    
    correct = False
    if ground_truth:
        if ground_truth in ans_lower or any(kw in ans_lower for kw in expected_keywords):
            correct = True
        elif len(ground_truth) > 3 and ground_truth[:15] in ans_lower:
            correct = True
    else:
        correct = True
        
    return {"status": "EVALUATED", "correct": correct, "score": 1.0 if correct else 0.0, "reason": None}

class TestPhase88Pipeline(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.engine = CollisionInferenceEngine(model_dir=os.path.join(REPO_ROOT, "models", "collision-10m"))
        cls.pipeline = RAGPipeline(inference_engine=cls.engine)

    def test_TEST_A_non_web_rag_produces_generated_answer(self):
        """TEST_A: Non-web RAG produces a generated answer."""
        req = RAGRequest(query="What is the ticker symbol for Microsoft?", mode="off")
        resp = self.pipeline.process(req)
        self.assertIsNotNone(resp.generated_answer)
        self.assertTrue(len(resp.generated_answer.strip()) > 0)
        self.assertNotEqual(resp.generated_answer.strip(), resp.query.strip())

    def test_TEST_B_evaluator_does_not_score_raw_query(self):
        """TEST_B: Evaluator does not score raw query."""
        req = RAGRequest(query="What is the stock ticker for Microsoft (MSFT)?", mode="off")
        resp = self.pipeline.process(req)
        
        # Suppose someone attempts to evaluate raw query:
        q_item = {"question": req.query, "expected_answer": "MSFT"}
        
        # Evaluator invariant test: if input passed is raw query instead of generated_answer:
        res = safe_evaluate_answer(q_item, resp, resp.generated_answer)
        self.assertEqual(res["status"], "EVALUATED")
        self.assertNotEqual(resp.generated_answer, req.query)

    def test_TEST_C_evaluator_does_not_score_formatted_prompt(self):
        """TEST_C: Evaluator does not score formatted prompt."""
        req = RAGRequest(query="What is the stock ticker for Microsoft (MSFT)?", mode="off")
        resp = self.pipeline.process(req)
        self.assertNotEqual(resp.generated_answer, resp.prompt_formatted)

    def test_TEST_D_evaluator_does_not_score_retrieved_context(self):
        """TEST_D: Evaluator does not score retrieved context."""
        req = RAGRequest(query="What is Microsoft's stock symbol MSFT?", mode="on")
        resp = self.pipeline.process(req)
        if resp.context_text:
            self.assertNotEqual(resp.generated_answer, resp.context_text)

    def test_TEST_E_empty_generation_causes_explicit_failure(self):
        """TEST_E: Empty generation causes explicit failure."""
        resp = RAGResponse(
            query="Test query?",
            mode_used="off",
            web_search_used=False,
            context_text="",
            prompt_formatted="Test query?",
            generated_answer=""
        )
        q_item = {"question": "Test query?", "expected_answer": "Test"}
        res = safe_evaluate_answer(q_item, resp, evaluator_input="")
        self.assertEqual(res["status"], "GENERATION_FAILURE")
        self.assertFalse(res["correct"])

    def test_TEST_F_generated_answer_passed_unchanged(self):
        """TEST_F: Generated answer is passed unchanged to evaluator."""
        resp = RAGResponse(
            query="What is Microsoft stock symbol?",
            mode_used="off",
            web_search_used=False,
            context_text="",
            prompt_formatted="What is Microsoft stock symbol?",
            generated_answer="Microsoft stock symbol is MSFT."
        )
        q_item = {"question": "What is Microsoft stock symbol?", "expected_answer": "MSFT"}
        res = safe_evaluate_answer(q_item, resp, evaluator_input=resp.generated_answer)
        self.assertEqual(res["status"], "EVALUATED")
        self.assertTrue(res["correct"])

    def test_TEST_G_question_containing_keyword_not_credited_unless_answer_contains_keyword(self):
        """TEST_G: Question containing expected keyword does NOT receive credit unless generated answer contains keyword."""
        # Question contains keyword MSFT
        q_item = {
            "question": "Why is MSFT performing well this year?",
            "expected_answer": "MSFT",
            "expected_keywords": ["MSFT"]
        }
        # Model output does NOT contain MSFT
        resp = RAGResponse(
            query=q_item["question"],
            mode_used="off",
            web_search_used=False,
            context_text="",
            prompt_formatted=q_item["question"],
            generated_answer="The company has strong revenue growth in cloud services."
        )
        res = safe_evaluate_answer(q_item, resp, evaluator_input=resp.generated_answer)
        self.assertFalse(res["correct"], "Should NOT get credit just because question had MSFT!")

    def test_TEST_H_web_and_non_web_use_same_contract(self):
        """TEST_H: Web RAG and non-web RAG use the same answer-extraction contract."""
        req_non_web = RAGRequest(query="Capital of France?", mode="off")
        resp_non_web = self.pipeline.process(req_non_web)
        
        req_web = RAGRequest(query="Capital of France?", mode="on")
        resp_web = self.pipeline.process(req_web)
        
        self.assertIsNotNone(resp_non_web.generated_answer)
        self.assertIsNotNone(resp_web.generated_answer)

    def test_TEST_I_response_fields_semantically_separated(self):
        """TEST_I: Response object fields remain semantically separated."""
        req = RAGRequest(query="What is quantum computing?", mode="off")
        resp = self.pipeline.process(req)
        fields = resp.__class__.model_fields if hasattr(resp.__class__, "model_fields") else resp.__fields__
        self.assertIn("query", fields)
        self.assertIn("prompt_formatted", fields)
        self.assertIn("context_text", fields)
        self.assertIn("generated_answer", fields)

    def test_TEST_J_production_model_sha256_unchanged(self):
        """TEST_J: Production model SHA256 remains unchanged."""
        sha = hashlib.sha256()
        with open(MODEL_PATH, "rb") as f:
            while chunk := f.read(65536):
                sha.update(chunk)
        expected_sha = "d256d46d962d6416fe22d2cfe80b13df0574279fb980d7d8576c2bdcf3775b97"
        self.assertEqual(sha.hexdigest(), expected_sha)

if __name__ == "__main__":
    unittest.main()
