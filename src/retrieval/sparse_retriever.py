"""
BM25 sparse (keyword) retriever.

BM25 excels at exact-match and rare-term queries where dense
embeddings may miss (e.g. specific part numbers, torque values,
model-year codes).

Design decisions:
  - The BM25 index is built *in memory* from the text stored in
    Pinecone metadata.  On first use we do a bulk fetch; after that
    the index is cached for the lifetime of the process.
  - Scores are normalised to [0, 1] by dividing by the maximum
    score in the result set, so they're comparable to cosine scores
    from the dense retriever.
  - Metadata filtering (make / model / year / subsystem) is applied
    *after* BM25 scoring via a simple post-filter.

For large corpora (>500K chunks) consider moving to an external
sparse index (e.g. Elasticsearch).  For the 100K-vector scale of
this project, in-memory BM25 is fast enough (<50 ms per query).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from rank_bm25 import BM25Okapi
from pinecone import Pinecone

from src.config import settings
from src.retrieval.dense_retriever import RetrievedChunk

logger = logging.getLogger(__name__)


def _tokenise(text: str) -> list[str]:
    """Simple whitespace + punctuation tokeniser for BM25."""
    # Lowercase, split on non-alphanumeric runs
    return re.findall(r"[a-z0-9]+", text.lower())


class SparseRetriever:
    """BM25 retriever backed by an in-memory index.

    Args:
        top_k: How many results to return (default from settings).
    """

    def __init__(self, top_k: int | None = None):
        self.top_k = top_k or settings.top_k_retrieval
        self.pc = Pinecone(api_key=settings.pinecone_api_key)
        self._index_name = settings.pinecone_index_name

        # Cache: populated on first call to retrieve()
        self._bm25: BM25Okapi | None = None
        self._corpus: list[dict] = []  # [{text, metadata}, …]

    # ── Public interface ──────────────────────────────────────────
    def retrieve(
        self,
        query: str,
        filters: dict | None = None,
    ) -> list[RetrievedChunk]:
        """Score *query* against the BM25 index and return top-k.

        Args:
            query:   Natural-language question.
            filters: Same filter dict as DenseRetriever (applied post-score).
        """
        self._ensure_index_loaded()

        tokenised_query = _tokenise(query)
        scores = self._bm25.get_scores(tokenised_query)  # numpy array

        # Pair scores with corpus entries and sort descending
        scored = sorted(
            zip(scores, self._corpus),
            key=lambda pair: pair[0],
            reverse=True,
        )

        # Normalise scores to [0, 1]
        max_score = scored[0][0] if scored and scored[0][0] > 0 else 1.0
        results: list[RetrievedChunk] = []

        for raw_score, doc in scored:
            if len(results) >= self.top_k:
                break
            if raw_score <= 0:
                break  # remaining scores are zero; nothing useful left

            meta = dict(doc.get("metadata", {}))

            # Post-filter by metadata
            if filters and not self._matches_filter(meta, filters):
                continue

            results.append(RetrievedChunk(
                text=doc["text"],
                score=float(raw_score / max_score),
                metadata=meta,
            ))

        logger.info("BM25 retrieval returned %d results.", len(results))
        return results

    # ── Index loading ─────────────────────────────────────────────
    def _ensure_index_loaded(self) -> None:
        """Fetch all vectors from Pinecone and build the BM25 index.

        This is a one-time cost; subsequent queries use the cache.
        For very large indexes, paginate the fetch; here we use
        Pinecone's fetch-all via list + fetch.
        """
        if self._bm25 is not None:
            return  # already cached

        logger.info("Building in-memory BM25 index from Pinecone…")
        index = self.pc.Index(self._index_name)

        # Pinecone serverless: use list() to get all IDs, then fetch in batches
        all_ids: list[str] = []
        for id_batch in index.list(limit=100):
            all_ids.extend(id_batch)

        if not all_ids:
            logger.warning("Pinecone index is empty; BM25 index will be empty.")
            self._corpus = []
            self._bm25 = BM25Okapi([[]])
            return

        # Fetch in batches of 1000
        corpus: list[dict] = []
        batch_size = 100
        for i in range(0, len(all_ids), batch_size):
            batch_ids = all_ids[i: i + batch_size]
            fetched = index.fetch(ids=batch_ids)
            for vec_id, vec_data in fetched.vectors.items():
                meta = dict(vec_data.metadata) if vec_data.metadata else {}
                text = meta.pop("text", "")
                corpus.append({"text": text, "metadata": meta})

        self._corpus = corpus
        tokenised_corpus = [_tokenise(doc["text"]) for doc in corpus]
        self._bm25 = BM25Okapi(tokenised_corpus)

        logger.info("BM25 index built: %d documents.", len(corpus))

    # ── Filter helper ─────────────────────────────────────────────
    @staticmethod
    def _matches_filter(metadata: dict, filters: dict) -> bool:
        """Return True if metadata satisfies all active filters."""
        for key in ("make", "model", "year", "subsystem"):
            if key in filters and filters[key] is not None:
                if metadata.get(key) != filters[key]:
                    return False
        return True
