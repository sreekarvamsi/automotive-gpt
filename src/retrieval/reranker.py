"""
Cohere cross-encoder reranker.

After dense + sparse retrieval produce a candidate pool, this
module sends the (query, candidate) pairs to Cohere's rerank
endpoint.  The cross-encoder scores each candidate in the context
of the query, producing much more accurate relevance scores than
bi-encoder cosine similarity alone.

This is the final ranking step before the candidates are handed
to the generation layer.  Typically improves answer relevance by
~12 % over retrieval-only baselines (see project metrics).
"""

from __future__ import annotations

import logging

import cohere

from src.config import settings
from src.retrieval.dense_retriever import RetrievedChunk

logger = logging.getLogger(__name__)


class Reranker:
    """Cohere reranker wrapper.

    Args:
        top_n: How many results to keep after reranking
               (default from settings.rerank_top_n).
    """

    def __init__(self, top_n: int | None = None):
        self.top_n = top_n or settings.rerank_top_n
        self.client = cohere.Client(api_key=settings.cohere_api_key)
        self.model = settings.cohere_rerank_model

    # ── Public interface ──────────────────────────────────────────
    def rerank(
        self,
        query: str,
        candidates: list[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        """Rerank *candidates* against *query* using Cohere.

        Args:
            query:      The user's question.
            candidates: Merged pool from dense + sparse retrieval.

        Returns:
            Top-N candidates sorted by descending rerank score.
            Each chunk's `score` is replaced with the Cohere
            relevance score (0–1).
        """
        if not candidates:
            return []

        # Cohere expects a list of document strings
        documents = [c.text for c in candidates]

        logger.info(
            "Reranking %d candidates (top_n=%d, model=%s)…",
            len(candidates), self.top_n, self.model,
        )

        response = self.client.rerank(
            model=self.model,
            query=query,
            documents=documents,
            top_n=min(self.top_n, len(candidates)),
        )

        # Build result list in reranked order
        # response.results is sorted by relevance_score descending
        reranked: list[RetrievedChunk] = []
        for result in response.results:
            original_idx = result.index
            original = candidates[original_idx]
            reranked.append(RetrievedChunk(
                text=original.text,
                score=float(result.relevance_score),
                metadata=original.metadata,
            ))

        logger.info(
            "Reranking complete. Kept %d / %d candidates.",
            len(reranked), len(candidates),
        )
        return reranked
