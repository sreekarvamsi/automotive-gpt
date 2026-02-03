"""
Embedding generation via OpenAI text-embedding-3-large.

Chunks are embedded in batches (max 100 per API call) to stay
within OpenAI's rate limits.  Exponential-backoff retry is built
in so transient 429 / 5xx errors don't crash the pipeline.

Each returned EmbeddedChunk carries the original text, the 3072-
dimensional vector, and all metadata from the chunker.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from openai import OpenAI, RateLimitError, APIStatusError

from src.config import settings
from src.ingestion.chunker import Chunk

logger = logging.getLogger(__name__)

# OpenAI allows up to 2048 inputs per call, but we stay
# conservative to avoid hitting token-level rate limits.
_BATCH_SIZE = 100
_MAX_RETRIES = 5
_RETRY_BASE_DELAY = 1.0  # seconds


@dataclass
class EmbeddedChunk:
    """A chunk paired with its dense embedding vector."""
    text: str
    vector: list[float]                        # 3072-dimensional
    metadata: dict = field(default_factory=dict)


class Embedder:
    """Wraps the OpenAI embedding API with batching and retry.

    Usage:
        embedder = Embedder()
        embedded = embedder.embed(chunks)   # list[Chunk] → list[EmbeddedChunk]
    """

    def __init__(self):
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_embedding_model

    # ── Public interface ──────────────────────────────────────────
    def embed(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        """Embed a list of chunks, returning EmbeddedChunks.

        Processes in batches of _BATCH_SIZE.  Progress is logged.
        """
        if not chunks:
            return []

        results: list[EmbeddedChunk] = []
        total_batches = (len(chunks) + _BATCH_SIZE - 1) // _BATCH_SIZE

        for batch_idx in range(0, len(chunks), _BATCH_SIZE):
            batch = chunks[batch_idx: batch_idx + _BATCH_SIZE]
            texts = [c.text for c in batch]

            logger.info(
                "Embedding batch %d/%d (%d texts)…",
                batch_idx // _BATCH_SIZE + 1, total_batches, len(texts),
            )

            vectors = self._call_api(texts)

            for chunk, vector in zip(batch, vectors):
                results.append(EmbeddedChunk(
                    text=chunk.text,
                    vector=vector,
                    metadata=chunk.metadata,
                ))

        logger.info("Embedded %d chunks total.", len(results))
        return results

    # ── API call with retry ───────────────────────────────────────
    def _call_api(self, texts: list[str]) -> list[list[float]]:
        """Call OpenAI embeddings endpoint with exponential backoff.

        Returns vectors in the same order as the input texts.
        """
        for attempt in range(_MAX_RETRIES):
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=texts,
                )
                # OpenAI may return embeddings out of order; sort by
                # the index field to restore original order.
                sorted_data = sorted(response.data, key=lambda obj: obj.index)
                return [item.embedding for item in sorted_data]

            except RateLimitError:
                delay = _RETRY_BASE_DELAY * (2 ** attempt)
                logger.warning(
                    "Rate-limited by OpenAI. Retrying in %.1fs (attempt %d/%d)…",
                    delay, attempt + 1, _MAX_RETRIES,
                )
                time.sleep(delay)

            except APIStatusError as exc:
                if exc.status_code >= 500:
                    delay = _RETRY_BASE_DELAY * (2 ** attempt)
                    logger.warning(
                        "OpenAI server error %d. Retrying in %.1fs (attempt %d/%d)…",
                        exc.status_code, delay, attempt + 1, _MAX_RETRIES,
                    )
                    time.sleep(delay)
                else:
                    # 4xx errors other than 429 are not retryable
                    raise

        raise RuntimeError(
            f"Failed to embed batch after {_MAX_RETRIES} retries."
        )
