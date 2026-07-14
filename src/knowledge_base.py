"""
knowledge_base.py — the RAG layer of PlanningCopilot.

WHY THIS EXISTS (read before adding more documents):
Prophet forecasts a number. It cannot tell you *why* your company decided to
carry extra safety stock on Electronics before Black Friday last year, or what
the agreed policy is for handling a supplier with a history of late shipments.
That's institutional knowledge that lives in planning policy documents and past
S&OP meeting notes — exactly the kind of unstructured text a vector DB is for.

This is deliberately NOT a general-purpose document store. It only holds two
kinds of documents:
  1. Planning policy documents (e.g. safety stock policy, supplier risk rules)
  2. Past S&OP meeting notes (SKU-specific decisions and context)

Same rule as the rest of the project: retrieval surfaces relevant text, but the
agent must attribute any number found in a retrieved document to that document
explicitly (e.g. "per the Q3 2025 S&OP notes...") rather than presenting it as
a live Prophet-computed figure. A policy target and a computed forecast are
different kinds of numbers and must never be blurred together in the response.
"""

import chromadb
from chromadb.utils import embedding_functions
from pathlib import Path
from dataclasses import dataclass


@dataclass
class RetrievedDoc:
    doc_id: str
    text: str
    metadata: dict
    relevance_score: float  # lower = more relevant (Chroma returns distance)


# ── Synthetic institutional knowledge corpus ─────────────────────────────────
# Mirrors what a real planning team's policy wiki + meeting notes archive
# would contain. Swap for real documents by pointing build() at a directory
# of your own markdown/text files instead.

PLANNING_DOCUMENTS = [
    {
        "id": "policy-safety-stock-001",
        "category": "policy",
        "title": "Safety Stock Policy — General Guidelines",
        "text": (
            "Company policy targets a 95% service level for all core SKUs, corresponding "
            "to a z-score of 1.65 in safety stock calculations. Seasonal SKUs (category: "
            "Seasonal) are held to a stricter 98% service level (z=2.05) during their peak "
            "season (October through December) due to the high cost of stockouts during "
            "gift-buying periods. Electronics category SKUs carry an additional 10% buffer "
            "on top of the standard formula due to historically volatile supplier lead times."
        ),
    },
    {
        "id": "policy-supplier-risk-001",
        "category": "policy",
        "title": "Supplier Risk Classification",
        "text": (
            "Suppliers are classified Tier 1 (reliable, <5% late shipment rate), Tier 2 "
            "(moderate risk, 5-15% late rate), or Tier 3 (high risk, >15% late rate). "
            "For Tier 3 suppliers, planners should add 7 additional days to the standard "
            "lead time assumption when calculating reorder points, reflecting the historical "
            "pattern of delays. Electronics suppliers are currently classified Tier 2 "
            "following the Q2 2025 shipping delays out of the primary distribution hub."
        ),
    },
    {
        "id": "policy-promo-planning-001",
        "category": "policy",
        "title": "Promotional Demand Planning Guidelines",
        "text": (
            "Promotions should be flagged in the planning system at least 21 days before "
            "the promo start date to allow adequate lead time for inventory buildup. "
            "Historical data shows promotional lift varies significantly by category: "
            "Snacks and Beverages typically see 50-90% demand lift during promotions, "
            "while Household and Electronics categories see more modest 20-40% lift. "
            "Planners should not assume a flat lift percentage across categories."
        ),
    },
    {
        "id": "sop-notes-2025-q3-001",
        "category": "meeting_notes",
        "title": "S&OP Meeting Notes — Q3 2025 — SKU-1007 (Electronics)",
        "text": (
            "Team flagged that SKU-1007 has shown consistent under-forecasting during "
            "back-to-school season (August-September) for the past two years. Demand "
            "planning agreed to apply a manual +15% adjustment to the statistical forecast "
            "for this SKU specifically during August and September going forward, pending "
            "further investigation into whether the seasonality model needs retuning. "
            "Action owner: category planning lead. Review at Q4 2025 meeting."
        ),
    },
    {
        "id": "sop-notes-2025-q3-002",
        "category": "meeting_notes",
        "title": "S&OP Meeting Notes — Q3 2025 — SKU-1009/1010 (Seasonal)",
        "text": (
            "Seasonal category SKUs (1009, 1010) both showed a 22% demand overshoot versus "
            "forecast in the November 2024 holiday period, resulting in stockouts in the "
            "first two weeks of December. Root cause was traced to the yearly seasonality "
            "component underestimating the holiday spike for newer SKUs with limited "
            "historical data. Decision: for Seasonal category SKUs with less than 18 months "
            "of history, apply the stricter 98% service level policy (see safety stock "
            "policy doc) rather than the standard 95%, until more data accumulates."
        ),
    },
    {
        "id": "sop-notes-2025-q2-001",
        "category": "meeting_notes",
        "title": "S&OP Meeting Notes — Q2 2025 — Electronics Supplier Delay",
        "text": (
            "Primary Electronics supplier experienced a 12-day shipping delay in May 2025 "
            "due to a port congestion issue, causing SKU-1007 and SKU-1008 stockouts for "
            "approximately one week. This informed the Q3 decision to reclassify Electronics "
            "suppliers as Tier 2 risk (see supplier risk policy) and add the standard 7-day "
            "buffer to lead time assumptions for both SKUs going forward."
        ),
    },
    {
        "id": "policy-exception-handling-001",
        "category": "policy",
        "title": "Demand Exception Escalation Policy",
        "text": (
            "When actual demand deviates more than 15% from the statistical forecast over "
            "a rolling 14-day window, this should be flagged as an exception for S&OP "
            "review. Deviations above 25% require same-week escalation to the category "
            "planning lead rather than waiting for the regular S&OP cycle, since sustained "
            "large deviations often indicate either a data quality issue or a genuine "
            "market shift that the forecasting model has not yet captured."
        ),
    },
]


class PlanningKnowledgeBase:
    """
    Wraps a Chroma collection over the planning document corpus.

    embedding_mode:
      "default"  — Chroma's built-in ONNX MiniLM-L6-v2 embedding. Free, runs
                    locally after a one-time ~90MB download on first use.
                    This is what you want for the actual deployed app.
      "tfidf"    — Lightweight scikit-learn TF-IDF embedding, no download
                    required. Used here only to verify the retrieval pipeline
                    logic works correctly in network-restricted environments
                    (e.g. CI, sandboxes) — semantic quality is lower than the
                    default mode, so don't use this for the real demo.
    """

    def __init__(self, persist_dir: str = "./chroma_db", embedding_mode: str = "default"):
        self.embedding_mode = embedding_mode
        self.client = chromadb.PersistentClient(path=persist_dir)

        if embedding_mode == "default":
            ef = embedding_functions.DefaultEmbeddingFunction()
        elif embedding_mode == "tfidf":
            ef = _TfidfEmbeddingFunction()
        else:
            raise ValueError(f"Unknown embedding_mode: {embedding_mode}")

        self.collection = self.client.get_or_create_collection(
            name="planning_knowledge",
            embedding_function=ef,
        )

    def build(self, documents: list[dict] = None, force_rebuild: bool = False):
        """Ingest the planning document corpus into the vector store."""
        documents = documents or PLANNING_DOCUMENTS

        if force_rebuild:
            existing_ids = self.collection.get()["ids"]
            if existing_ids:
                self.collection.delete(ids=existing_ids)

        existing_ids = set(self.collection.get()["ids"])
        new_docs = [d for d in documents if d["id"] not in existing_ids]

        if not new_docs:
            return 0

        self.collection.add(
            ids=[d["id"] for d in new_docs],
            documents=[d["text"] for d in new_docs],
            metadatas=[{"category": d["category"], "title": d["title"]} for d in new_docs],
        )
        return len(new_docs)

    def search(self, query: str, k: int = 3, category_filter: str = None) -> list[RetrievedDoc]:
        """
        Search planning documents relevant to a query. Optionally filter by
        category ("policy" or "meeting_notes").
        """
        where = {"category": category_filter} if category_filter else None
        results = self.collection.query(query_texts=[query], n_results=k, where=where)

        docs = []
        for i in range(len(results["ids"][0])):
            docs.append(RetrievedDoc(
                doc_id=results["ids"][0][i],
                text=results["documents"][0][i],
                metadata=results["metadatas"][0][i],
                relevance_score=results["distances"][0][i],
            ))
        return docs


class _TfidfEmbeddingFunction(embedding_functions.EmbeddingFunction):
    """Minimal TF-IDF embedding for offline testing — NOT used in the shipped
    default configuration. See docstring above for why this exists."""

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self._vectorizer = TfidfVectorizer(max_features=384)
        corpus_texts = [d["text"] for d in PLANNING_DOCUMENTS]
        self._vectorizer.fit(corpus_texts)

    def __call__(self, input):
        texts = input if isinstance(input, list) else [input]
        import numpy as np
        vecs = self._vectorizer.transform(texts).toarray()
        if vecs.shape[1] < 384:
            vecs = np.pad(vecs, ((0, 0), (0, 384 - vecs.shape[1])))
        return vecs.tolist()

    def name(self):
        return "tfidf-fallback"

    def get_config(self):
        return {}

    @staticmethod
    def build_from_config(config):
        return _TfidfEmbeddingFunction()


if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "default"

    kb = PlanningKnowledgeBase(embedding_mode=mode)
    added = kb.build()
    print(f"Ingested {added} new documents (mode={mode})")

    test_queries = [
        "what's our safety stock policy for seasonal items",
        "any issues with the Electronics supplier",
        "when should I escalate a demand exception",
    ]
    for q in test_queries:
        print(f"\nQuery: {q}")
        results = kb.search(q, k=2)
        for r in results:
            print(f"  [{r.metadata['title']}] (score={r.relevance_score:.3f})")
            print(f"    {r.text[:120]}...")
