import os
import sys
import hashlib
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from collision.rag.schemas import DocumentChunk
from collision.rag.index import VectorIndex
from collision.rag.retriever import DocumentRetriever
from collision.web.schemas import WebEvidenceChunk
from collision.web.search import MockWebSearchProvider
from collision.routing.schemas import (
    RouteMode,
    RoutingDecision,
    FusedEvidence,
    VerificationResult,
    VerifiedAnswerResult
)
from collision.routing.classifier import QueryClassifier
from collision.routing.router import AdaptiveKnowledgeRouter
from collision.routing.fusion import EvidenceFusion
from collision.routing.verifier import GroundingVerifier
from collision.routing.confidence import ConfidenceScorer
from collision.routing.engine import AdaptiveKnowledgeEngine

PROTECTED_COLLISION_10M_SHA256 = "d256d46d962d6416fe22d2cfe80b13df0574279fb980d7d8576c2bdcf3775b97"
PROTECTED_PHASE91_V9_SHA256 = "98a2b416bed2033cd338b1f2e245e5b1d9681bdd66ba2a788409cddefd4be449"

def compute_sha256(filepath: str) -> str:
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192 * 1024):
            sha.update(chunk)
    return sha.hexdigest().lower()

def test_checkpoint_integrity_phase96():
    """Verify that protected production and research checkpoints remain byte-identical."""
    c10m_path = os.path.join(PROJECT_ROOT, "models", "collision-10m", "model.pt")
    if os.path.exists(c10m_path):
        assert compute_sha256(c10m_path) == PROTECTED_COLLISION_10M_SHA256

    v9_path = os.path.join(PROJECT_ROOT, "models", "phase91_v9_10m", "model.pt")
    if os.path.exists(v9_path):
        assert compute_sha256(v9_path) == PROTECTED_PHASE91_V9_SHA256

def test_query_classifier_signals():
    """Verify intent signal classification (unanswerable, temporal web, model-only, hybrid)."""
    classifier = QueryClassifier()
    assert classifier.is_unanswerable_private("What is my secret admin password?")
    assert classifier.requires_current_web("What is the latest React 19 version?")
    assert classifier.is_model_suitable("Hello, what is 2 + 2?")
    assert classifier.is_hybrid_candidate("Compare COLLISION Phase 15 results with latest LLM benchmarks.")

def test_adaptive_router_decisions():
    """Verify adaptive routing decisions for different query profiles."""
    c1 = DocumentChunk(document_id="d1", source="arch.md", chunk_id=0, text="COLLISION-10M uses 6 layers and 8 attention heads.")
    index = VectorIndex()
    index.add([c1])
    retriever = DocumentRetriever(index=index, default_top_k=2, default_relevance_threshold=0.10)

    router = AdaptiveKnowledgeRouter()

    # 1. Local match -> LOCAL
    dec_local = router.route("How many attention heads does COLLISION-10M use?", local_retriever=retriever)
    assert dec_local.mode == RouteMode.LOCAL

    # 2. Conversational / Math -> MODEL
    dec_model = router.route("Hello, if A is taller than B and B is taller than C, who is shortest?", local_retriever=retriever)
    assert dec_model.mode == RouteMode.MODEL

    # 3. External / Temporal -> WEB
    dec_web = router.route("What are the release notes for React 19 and Python 3.13?", local_retriever=retriever)
    assert dec_web.mode == RouteMode.WEB

    # 4. Private / Unanswerable -> INSUFFICIENT_INFORMATION
    dec_ins = router.route("What is my bank account PIN and credit card number?", local_retriever=retriever)
    assert dec_ins.mode == RouteMode.INSUFFICIENT_INFORMATION

def test_evidence_fusion_and_deduplication():
    """Verify multi-source evidence fusion, deduplication, and score ranking."""
    fusion = EvidenceFusion()
    c1 = DocumentChunk(document_id="d1", source="arch.md", chunk_id=0, text="COLLISION has 6 transformer layers.")
    ret_item = retriever_item = type("Obj", (), {"chunk": c1, "source": "arch.md", "similarity_score": 0.85})()
    
    web_item1 = WebEvidenceChunk(url="https://a.com", title="A", domain="a.com", text="COLLISION has 6 transformer layers.", similarity_score=0.90) # Duplicate content
    web_item2 = WebEvidenceChunk(url="https://b.com", title="B", domain="b.com", text="Python 3.13 was released in October 2024.", similarity_score=0.75)

    fused = fusion.fuse(local_items=[ret_item], web_items=[web_item1, web_item2], max_fused_chunks=5)
    assert len(fused) == 2  # Deduplicated duplicate content
    assert fused[0].text == "COLLISION has 6 transformer layers."
    assert fused[1].text == "Python 3.13 was released in October 2024."

def test_grounding_verifier_claim_checking():
    """Verify claim-level verification and contradiction detection."""
    verifier = GroundingVerifier()
    evidence = [
        FusedEvidence(source="specs.md", url="local://specs.md", document_id="s1", chunk_id=0, text="The system uses an 8-core CPU with 32GB RAM.", score=0.9, source_type="LOCAL")
    ]

    # Supported answer
    v_sup = verifier.verify("The system uses an 8-core CPU and 32GB of RAM.", evidence)
    assert v_sup.supported is True
    assert v_sup.status == "ANSWER"

    # Contradicted answer (64 cores vs 8 cores)
    v_con = verifier.verify("The system uses a 64-core CPU.", evidence)
    assert v_con.supported is False
    assert v_con.status == "CONTRADICTED"

    # Unsupported answer
    v_uns = verifier.verify("The server was painted bright neon green in 2021.", evidence)
    assert v_uns.supported is False
    assert v_uns.status in ("UNSUPPORTED", "INSUFFICIENT_INFORMATION")

def test_grounding_verifier_conflicting_sources():
    """Verify detection of conflicting evidence across retrieved sources."""
    verifier = GroundingVerifier()
    evidence = [
        FusedEvidence(source="srcA", url="https://a.com", document_id="a", chunk_id=0, text="The project release date was October 1.", score=0.8, source_type="WEB"),
        FusedEvidence(source="srcB", url="https://b.com", document_id="b", chunk_id=1, text="The project release date was November 15.", score=0.8, source_type="WEB")
    ]
    v_res = verifier.verify("The project release date is recorded.", evidence)
    assert v_res.has_conflicting_evidence is True
    assert v_res.status == "CONFLICTING_EVIDENCE"

def test_adaptive_knowledge_engine_end_to_end():
    """Test full pipeline routing, execution, generation, and verification."""
    c1 = DocumentChunk(document_id="d1", source="hardware.md", chunk_id=0, text="The CPU inference node has 8 physical cores.")
    index = VectorIndex()
    index.add([c1])
    retriever = DocumentRetriever(index=index, default_top_k=2, default_relevance_threshold=0.10)

    search_provider = MockWebSearchProvider()
    search_provider.add_mock_results("react 19 release", [
        {"title": "React 19", "url": "https://react.dev/blog", "snippet": "React 19 introduces Actions and Server Functions."}
    ])

    engine = AdaptiveKnowledgeEngine(local_retriever=retriever, search_provider=search_provider)

    # 1. Local Routed Query
    res_local = engine.answer("How many physical cores does the CPU inference node have?", mode=RouteMode.AUTO, max_tokens=25)
    assert res_local.route == RouteMode.LOCAL
    assert len(res_local.fused_evidence) > 0

    # 2. Web Routed Query
    res_web = engine.answer("What features are in the React 19 release?", mode=RouteMode.AUTO, max_tokens=25)
    assert res_web.route == RouteMode.WEB
    assert len(res_web.fused_evidence) > 0

    # 3. Model Routed Query
    res_model = engine.answer("Hello, how are you?", mode=RouteMode.AUTO, max_tokens=20)
    assert res_model.route == RouteMode.MODEL
