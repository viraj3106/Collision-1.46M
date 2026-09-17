import os
import sys
import hashlib
import tempfile
import pytest
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from collision.rag.schemas import DocumentChunk, RetrievalItem, RAGStatus, RAGResult
from collision.rag.chunker import DocumentChunker
from collision.rag.embeddings import LocalEmbeddingModel
from collision.rag.index import VectorIndex
from collision.rag.retriever import DocumentRetriever
from collision.rag.engine import GroundedRAGEngine
from collision.answering.engine import CollisionAnsweringEngine

PROTECTED_COLLISION_10M_SHA256 = "d256d46d962d6416fe22d2cfe80b13df0574279fb980d7d8576c2bdcf3775b97"
PROTECTED_PHASE91_V9_SHA256 = "98a2b416bed2033cd338b1f2e245e5b1d9681bdd66ba2a788409cddefd4be449"

def compute_sha256(filepath: str) -> str:
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192 * 1024):
            sha.update(chunk)
    return sha.hexdigest().lower()

def test_checkpoint_integrity_phase94():
    """Verify that protected production and research checkpoints remain unmodified."""
    c10m_path = os.path.join(PROJECT_ROOT, "models", "collision-10m", "model.pt")
    if os.path.exists(c10m_path):
        assert compute_sha256(c10m_path) == PROTECTED_COLLISION_10M_SHA256

    v9_path = os.path.join(PROJECT_ROOT, "models", "phase91_v9_10m", "model.pt")
    if os.path.exists(v9_path):
        assert compute_sha256(v9_path) == PROTECTED_PHASE91_V9_SHA256

def test_deterministic_chunking():
    """Test deterministic document ingestion and chunking across multiple calls."""
    chunker = DocumentChunker(chunk_size_words=20, chunk_overlap_words=5)
    sample_text = (
        "The COLLISION architecture is a 6-layer causal transformer. "
        "It uses 8 attention heads with an embedding dimension of 384. "
        "The model is optimized for CPU inference and operates on a vocabulary of 8000 tokens. "
        "Weights are tied between input and output embeddings."
    )

    chunks1 = chunker.chunk_text(sample_text, document_id="doc1", source="arch.md")
    chunks2 = chunker.chunk_text(sample_text, document_id="doc1", source="arch.md")

    assert len(chunks1) > 0
    assert len(chunks1) == len(chunks2)
    for c1, c2 in zip(chunks1, chunks2):
        assert c1.document_id == c2.document_id
        assert c1.chunk_id == c2.chunk_id
        assert c1.text == c2.text
        assert c1.source == c2.source

def test_empty_and_malformed_document_chunking():
    """Test chunker behavior on empty, whitespace, and non-existent files."""
    chunker = DocumentChunker()
    assert chunker.chunk_text("", "empty_doc", "empty.txt") == []
    assert chunker.chunk_text("   \n\t  ", "blank_doc", "blank.md") == []

    with pytest.raises(FileNotFoundError):
        chunker.ingest_file("non_existent_file_path_xyz.md")

def test_local_embeddings_generation_and_normalization():
    """Verify local embedding generation, deterministic hashing, and L2 unit normalization."""
    emb_model = LocalEmbeddingModel(dimension=256)
    t1 = "COLLISION Transformer causal attention"
    t2 = "COLLISION Transformer causal attention"
    t3 = "Completely different text about cooking pasta"

    v1 = emb_model.embed_text(t1)
    v2 = emb_model.embed_text(t2)
    v3 = emb_model.embed_text(t3)

    assert v1.shape == (256,)
    assert np.allclose(v1, v2)
    assert np.isclose(np.linalg.norm(v1), 1.0, atol=1e-5)
    assert np.isclose(np.linalg.norm(v3), 1.0, atol=1e-5)

    sim_same = emb_model.cosine_similarity(v1, v2)
    sim_diff = emb_model.cosine_similarity(v1, v3)
    assert np.isclose(sim_same, 1.0, atol=1e-5)
    assert sim_diff < sim_same

def test_vector_index_add_search_save_load():
    """Test vector index addition, similarity search, and disk persistence."""
    chunker = DocumentChunker(chunk_size_words=15, chunk_overlap_words=0)
    c1 = DocumentChunk(document_id="d1", source="hardware.md", chunk_id=0, text="The system uses an 8-core CPU with 32GB of RAM.")
    c2 = DocumentChunk(document_id="d2", source="network.md", chunk_id=0, text="The database API listens on TCP port 5432.")

    index = VectorIndex()
    index.add([c1, c2])
    assert index.size() == 2

    # Search
    results = index.search("How many CPU cores are in the system?", top_k=2)
    assert len(results) == 2
    top_chunk, score = results[0]
    assert top_chunk.source == "hardware.md"
    assert "8-core CPU" in top_chunk.text
    assert score > 0.20

    # Save and load in temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        index.save(tmp_dir)
        loaded_index = VectorIndex()
        loaded_index.load(tmp_dir)
        assert loaded_index.size() == 2
        loaded_results = loaded_index.search("How many CPU cores are in the system?", top_k=1)
        assert loaded_results[0][0].source == "hardware.md"
        assert np.isclose(loaded_results[0][1], score, atol=1e-5)

def test_retriever_threshold_rejection():
    """Test that retriever filters out low-similarity chunks when below threshold."""
    c1 = DocumentChunk(document_id="d1", source="hardware.md", chunk_id=0, text="The system uses an 8-core CPU.")
    index = VectorIndex()
    index.add([c1])
    retriever = DocumentRetriever(index=index, default_top_k=3, default_relevance_threshold=0.10)

    # Relevant query
    rel_items = retriever.retrieve("What CPU is used?")
    assert len(rel_items) >= 1
    assert rel_items[0].source == "hardware.md"

    # Irrelevant query below threshold
    irrel_items = retriever.retrieve("Quantum entanglement teleportation protocol for satellites in deep space", relevance_threshold=0.85)
    assert len(irrel_items) == 0

def test_grounded_rag_engine_answering_and_citations():
    """Test end-to-end grounded RAG answering, source propagation, and refusal on irrelevant queries."""
    doc_text = "The COLLISION project was released with a 10M parameter flagship model trained on 10 million tokens."
    chunker = DocumentChunker()
    chunks = chunker.chunk_text(doc_text, document_id="rel_doc", source="release_notes.md")

    index = VectorIndex()
    index.add(chunks)
    retriever = DocumentRetriever(index=index, default_top_k=2, default_relevance_threshold=0.15)
    rag_engine = GroundedRAGEngine(retriever=retriever)

    # 1. Grounded Question
    res_grounded = rag_engine.answer("How many parameters does the flagship model have?", max_tokens=25)
    assert isinstance(res_grounded, RAGResult)
    assert res_grounded.status in (RAGStatus.ANSWER, RAGStatus.UNCERTAIN)
    assert "release_notes.md" in res_grounded.sources
    assert len(res_grounded.retrieved_chunks) > 0
    assert res_grounded.latency_ms > 0
    assert res_grounded.retrieval_latency_ms > 0
    assert res_grounded.generation_latency_ms > 0

    # 2. Irrelevant Question (Rejected by Threshold)
    res_irrelevant = rag_engine.answer("What is the recipe for chocolate chip cookies?", relevance_threshold=0.80)
    assert res_irrelevant.status == RAGStatus.INSUFFICIENT_INFORMATION
    assert len(res_irrelevant.sources) == 0
    assert len(res_irrelevant.retrieved_chunks) == 0
    assert "information" in res_irrelevant.answer.lower()
