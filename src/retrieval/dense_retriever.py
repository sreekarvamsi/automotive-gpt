"""
Dense (semantic) retriever.

Embeds the user query with the same model used during ingestion
(text-embedding-3-large) and runs a cosine-similarity top-k search
against the Pinecone index.  Optional metadata filters (make, model,
year, subsystem) are applied server-side so only relevant vectors
are scored.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from openai import OpenAI
from pinecone import Pinecone

from src.config import settings

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    """A single chunk returned by any retriever, with its score."""
    text: str
    score: float                                   # 0–1 (cosine similarity or BM25-normalised)
    metadata: dict = field(default_factory=dict)   # source_file, page, …


class DenseRetriever:
    """Semantic retrieval via Pinecone cosine search.

    Args:
        top_k: Number of vectors to retrieve (default from settings).
    """

    def __init__(self, top_k: int | None = None):
        self.top_k = top_k or settings.top_k_retrieval
        self.openai = OpenAI(api_key=settings.openai_api_key)
        self.pc = Pinecone(api_key=settings.pinecone_api_key)
        self.index = self.pc.Index(settings.pinecone_index_name)

    # ── Public interface ──────────────────────────────────────────
    def retrieve(
        self,
        query: str,
        filters: dict | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve the top-k most semantically similar chunks.

        Args:
            query:   The user's natural-language question.
            filters: Optional dict with keys like make, model, year,
                     subsystem.  Converted to a Pinecone filter expression.

        Returns:
            List of RetrievedChunk sorted by descending similarity score.
        """
        query_vector = self._embed_query(query)
        pinecone_filter = self._build_filter(filters) if filters else None

        logger.info("Dense retrieval: top_k=%d, filter=%s", self.top_k, pinecone_filter)

        response = self.index.query(
            vector=query_vector,
            top_k=self.top_k,
            filter=pinecone_filter,
            include_metadata=True,
        )

        results: list[RetrievedChunk] = []
        for match in response.matches:
            meta = dict(match.metadata) if match.metadata else {}
            text = meta.pop("text", "")  # text was stored in metadata at upsert time
            results.append(RetrievedChunk(
                text=text,
                score=match.score,
                metadata=meta,
            ))

        logger.info("Dense retrieval returned %d results.", len(results))
        return results

    # ── Helpers ───────────────────────────────────────────────────
    def _embed_query(self, query: str) -> list[float]:
        """Embed a single query string."""
        response = self.openai.embeddings.create(
            model=settings.openai_embedding_model,
            input=[query],
        )
        return response.data[0].embedding

    @staticmethod
    def _build_filter(filters: dict) -> dict | None:
        """Convert a user-facing filter dict to a Pinecone filter expression.

        Supported keys: make, model, year, subsystem.
        Example input:  {"make": "Honda", "year": 2022}
        Example output: {"$and": [{"make": {"$eq": "Honda"}}, {"year": {"$eq": 2022}}]}
        """
        clauses: list[dict] = []
        for key in ("make", "model", "year", "subsystem"):
            if key in filters and filters[key] is not None:
                clauses.append({key: {"$eq": filters[key]}})

        if not clauses:
            return None
        if len(clauses) == 1:
            return clauses[0]
        return {"$and": clauses}
