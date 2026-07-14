"""
Unit tests for the knowledge base (vector store) retrieval layer.

These tests use embedding_mode="tfidf" deliberately — not the default
ChromaDB embedding — because the default requires a one-time internet
download of the embedding model, which CI/sandboxed environments may not
have. This mirrors exactly the reasoning documented in knowledge_base.py.
Retrieval QUALITY with the real embedding model will be better than what
these tests show; what's being verified here is that the plumbing (storage,
filtering, top-k ordering) is correct, independent of embedding quality.
"""

import sys
import shutil
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from knowledge_base import PlanningKnowledgeBase, PLANNING_DOCUMENTS

TEST_PERSIST_DIR = str(Path(__file__).parent / "_test_chroma_db")


@pytest.fixture(scope="module")
def kb():
    # clean slate for the test run
    shutil.rmtree(TEST_PERSIST_DIR, ignore_errors=True)
    store = PlanningKnowledgeBase(persist_dir=TEST_PERSIST_DIR, embedding_mode="tfidf")
    store.build()
    yield store
    shutil.rmtree(TEST_PERSIST_DIR, ignore_errors=True)


def test_all_documents_ingested(kb):
    all_docs = kb.collection.get()
    assert len(all_docs["ids"]) == len(PLANNING_DOCUMENTS)


def test_search_returns_requested_count(kb):
    results = kb.search("safety stock policy", k=2)
    assert len(results) == 2


def test_search_finds_relevant_policy_doc(kb):
    results = kb.search("what service level do we target for seasonal SKUs", k=3)
    titles = [r.metadata["title"] for r in results]
    # the safety stock policy doc explicitly covers seasonal service levels
    assert any("Safety Stock Policy" in t or "Seasonal" in t for t in titles)


def test_category_filter_restricts_results(kb):
    results = kb.search("demand", k=10, category_filter="policy")
    assert all(r.metadata["category"] == "policy" for r in results)
    assert len(results) > 0


def test_rebuild_does_not_duplicate_documents(kb):
    added_again = kb.build()  # should be a no-op, all docs already ingested
    assert added_again == 0
    all_docs = kb.collection.get()
    assert len(all_docs["ids"]) == len(PLANNING_DOCUMENTS)


def test_relevance_score_is_native_float(kb):
    results = kb.search("supplier risk", k=1)
    assert isinstance(results[0].relevance_score, float)
