import os
import sys
import hashlib
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from collision.rag.schemas import DocumentChunk, RAGStatus
from collision.rag.index import VectorIndex
from collision.rag.retriever import DocumentRetriever
from collision.web.schemas import (
    RetrievalMode,
    WebSearchItem,
    WebSearchResult,
    WebDocument,
    WebSource,
    WebEvidenceChunk,
    GroundingResult
)
from collision.web.search import WebSearchProvider, MockWebSearchProvider
from collision.web.fetch import safe_fetch_page, SSRFProtectionError, validate_url
from collision.web.extractor import WebPageExtractor
from collision.web.ranker import WebEvidenceRanker
from collision.web.engine import LiveWebGroundingEngine

PROTECTED_COLLISION_10M_SHA256 = "d256d46d962d6416fe22d2cfe80b13df0574279fb980d7d8576c2bdcf3775b97"
PROTECTED_PHASE91_V9_SHA256 = "98a2b416bed2033cd338b1f2e245e5b1d9681bdd66ba2a788409cddefd4be449"

def compute_sha256(filepath: str) -> str:
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192 * 1024):
            sha.update(chunk)
    return sha.hexdigest().lower()

def test_checkpoint_integrity_phase95():
    """Verify that protected production and research checkpoints remain unmodified."""
    c10m_path = os.path.join(PROJECT_ROOT, "models", "collision-10m", "model.pt")
    if os.path.exists(c10m_path):
        assert compute_sha256(c10m_path) == PROTECTED_COLLISION_10M_SHA256

    v9_path = os.path.join(PROJECT_ROOT, "models", "phase91_v9_10m", "model.pt")
    if os.path.exists(v9_path):
        assert compute_sha256(v9_path) == PROTECTED_PHASE91_V9_SHA256

def test_mock_search_provider_deduplication_and_domain():
    """Test MockWebSearchProvider schema, duplicate URL elimination, and domain extraction."""
    provider = MockWebSearchProvider()
    provider.add_mock_results("python release", [
        {"title": "Python 3.13", "url": "https://python.org/release/3.13", "snippet": "Python 3.13 introduces free-threading."},
        {"title": "Python 3.13 Duplicate", "url": "https://python.org/release/3.13", "snippet": "Duplicate URL."},
        {"title": "Release Notes", "url": "https://docs.python.org/3/whatsnew", "snippet": "Overview of new features."}
    ])

    res: WebSearchResult = provider.search("python release", max_results=5)
    assert isinstance(res, WebSearchResult)
    assert res.total_found == 2  # Deduplicated from 3
    assert res.results[0].domain == "python.org"
    assert res.results[1].domain == "docs.python.org"

def test_fetch_ssrf_protection():
    """Verify that SSRF protection blocks private, link-local, and loopback IP requests."""
    with pytest.raises(SSRFProtectionError):
        validate_url("http://127.0.0.1:8000/secret")
    with pytest.raises(SSRFProtectionError):
        validate_url("http://localhost:8080/admin")
    with pytest.raises(SSRFProtectionError):
        validate_url("ftp://example.com/file")

def test_html_text_extractor():
    """Verify clean HTML body extraction and noise removal."""
    extractor = WebPageExtractor()
    html = """
    <html>
      <head><title>Test Page</title><script>var x = 1;</script></head>
      <body>
        <nav><a href='/home'>Home</a></nav>
        <h1>Main Heading</h1>
        <p>This is the primary readable body paragraph about transformer neural networks.</p>
        <footer>Copyright 2026</footer>
      </body>
    </html>
    """
    doc: WebDocument = extractor.extract(html, url="https://example.com/test", title="Test Page")
    assert doc.title == "Test Page"
    assert doc.domain == "example.com"
    assert "primary readable body paragraph" in doc.text
    assert "var x = 1" not in doc.text
    assert "Copyright" not in doc.text

def test_web_evidence_ranker():
    """Verify deterministic ranking of web evidence against user queries."""
    ranker = WebEvidenceRanker()
    doc1 = WebDocument(
        url="https://pytorch.org",
        title="PyTorch Docs",
        domain="pytorch.org",
        text="PyTorch 2.4 adds new compiler graph optimizations for CPU and CUDA.",
        extracted_paragraphs=["PyTorch 2.4 adds new compiler graph optimizations for CPU and CUDA."]
    )
    doc2 = WebDocument(
        url="https://recipes.com",
        title="Pasta Recipes",
        domain="recipes.com",
        text="Boil pasta in salted water for 10 minutes until al dente.",
        extracted_paragraphs=["Boil pasta in salted water for 10 minutes until al dente."]
    )

    evidence: list[WebEvidenceChunk] = ranker.rank_evidence("What optimizations were added to PyTorch?", [doc1, doc2], top_k=2)
    assert len(evidence) >= 1
    assert evidence[0].domain == "pytorch.org"
    assert "PyTorch 2.4" in evidence[0].text

def test_live_web_grounding_auto_local_fallback():
    """Verify AUTO mode: Uses local RAG when local evidence is present, falls back to web when not."""
    # 1. Local RAG setup
    c1 = DocumentChunk(document_id="d1", source="local_arch.md", chunk_id=0, text="COLLISION-10M has 6 transformer layers.")
    index = VectorIndex()
    index.add([c1])
    local_retriever = DocumentRetriever(index=index, default_top_k=2, default_relevance_threshold=0.10)

    # 2. Mock Web Search setup
    search_provider = MockWebSearchProvider()
    search_provider.add_mock_results("react 19 release", [
        {"title": "React 19", "url": "https://react.dev/blog/react-19", "snippet": "React 19 adds Actions and Server Components."}
    ])

    engine = LiveWebGroundingEngine(search_provider=search_provider, local_retriever=local_retriever)

    # Query A: Answerable from local RAG
    res_local = engine.answer("How many transformer layers are in COLLISION-10M?", mode=RetrievalMode.AUTO, max_tokens=25)
    assert res_local.retrieval_mode == RetrievalMode.LOCAL
    assert any("local_arch.md" in s.title for s in res_local.sources)

    # Query B: Not in local RAG -> triggers Web Search fallback
    res_web = engine.answer("What features are in the React 19 release?", mode=RetrievalMode.AUTO, max_tokens=25)
    assert res_web.retrieval_mode == RetrievalMode.WEB
    assert any("react.dev" in s.domain for s in res_web.sources)

def test_web_grounding_insufficient_evidence():
    """Verify that when web search returns 0 results, system honestly returns INSUFFICIENT_INFORMATION."""
    search_provider = MockWebSearchProvider() # Empty database
    engine = LiveWebGroundingEngine(search_provider=search_provider, local_retriever=None)

    res = engine.answer("What is the secret menu of the unannounced restaurant in 2045?", mode=RetrievalMode.WEB)
    assert res.status == RAGStatus.INSUFFICIENT_INFORMATION
    assert len(res.sources) == 0
    assert "information" in res.answer.lower()
