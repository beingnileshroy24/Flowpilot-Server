import pytest
import os
import shutil
import json
from app.ai.storage.lancedb_client import LanceDBManager
from app.ai.storage.hybrid_retriever import HybridRetriever
from app.ai.core.embedder import ModernBertEmbedderSingleton

TEST_DB_PATH = "./.lancedb_test_store"

@pytest.fixture(autouse=True)
def clean_db():
    if os.path.exists(TEST_DB_PATH):
        shutil.rmtree(TEST_DB_PATH)
    yield
    if os.path.exists(TEST_DB_PATH):
        shutil.rmtree(TEST_DB_PATH)

def test_hybrid_retriever_threshold_and_rrf():
    manager = LanceDBManager(db_path=TEST_DB_PATH)
    embedder = ModernBertEmbedderSingleton()
    retriever = HybridRetriever(lancedb_manager=manager, embedder=embedder)

    project_id = "test_proj_hybrid"
    
    # We will simulate embeddings by writing directly to LanceDB or mocking embedder.
    # Let's insert directly using LanceDBManager's insert_knowledge method:
    # Vector size: 768.
    
    # 1. Record A: Match in vector (close) and keyword
    vector_a = [0.01] * 768
    manager.insert_knowledge(
        vector=vector_a,
        entity_type="DOCUMENT",
        source_id="doc_a",
        project_id=project_id,
        created_at="2026-07-15T00:00:00Z",
        content_snippet="We need setup instructions for oauth SSO authentication.",
        metadata_struct={"chunk_idx": 0}
    )

    # 2. Record B: Match in vector (close) but NO keyword match
    vector_b = [0.012] * 768
    manager.insert_knowledge(
        vector=vector_b,
        entity_type="DOCUMENT",
        source_id="doc_b",
        project_id=project_id,
        created_at="2026-07-15T00:00:00Z",
        content_snippet="The developer John is working on databases.",
        metadata_struct={"chunk_idx": 0}
    )

    # 3. Record C: Low vector similarity (far) but keyword match
    # Since we search with query vector_a, let's make vector_c orthogonal or opposite
    vector_c = [0.2] * 768
    manager.insert_knowledge(
        vector=vector_c,
        entity_type="DOCUMENT",
        source_id="doc_c",
        project_id=project_id,
        created_at="2026-07-15T00:00:00Z",
        content_snippet="oauth SSO configuration credentials setup.",
        metadata_struct={"chunk_idx": 0}
    )

    # 4. Record D: Low vector similarity (far) and NO keyword match
    vector_d = [0.22] * 768
    manager.insert_knowledge(
        vector=vector_d,
        entity_type="DOCUMENT",
        source_id="doc_d",
        project_id=project_id,
        created_at="2026-07-15T00:00:00Z",
        content_snippet="Completely unrelated content about dogs.",
        metadata_struct={"chunk_idx": 0}
    )

    # Mock the embedder to return vector_a when searching for "oauth SSO setup"
    class MockEmbedder:
        def compute_embedding(self, query, prefix):
            return vector_a

    retriever.embedder = MockEmbedder()

    # Search with "oauth SSO setup"
    results = retriever.search_hybrid(
        query="oauth SSO setup",
        project_id=project_id,
        limit=5,
        similarity_threshold=0.72
    )

    # Let's verify:
    # - Record A matches both vector and keyword (high RRF score).
    # - Record B matches vector only (vector distance from A is close).
    # - Record C matches keyword only (vector distance from A is far, similarity < 0.72 so dropped from vector search, but keyword match brings it in).
    # - Record D is dropped (similarity too low and no keyword match).
    
    assert len(results) >= 2
    
    source_ids = [r["source_id"] for r in results]
    assert "doc_a" in source_ids
    assert "doc_b" in source_ids
    assert "doc_c" in source_ids
    assert "doc_d" not in source_ids
    
    # Doc A should be first because it is in both rank lists
    assert results[0]["source_id"] == "doc_a"
