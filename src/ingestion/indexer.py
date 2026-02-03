"""
Pinecone vector-store indexer.

Responsibilities:
  - Create the Pinecone index if it doesn't already exist
    (dimension=3072, cosine metric, serverless spec).
  - Upsert EmbeddedChunks in batches of 500 (Pinecone's max).
  - Support *incremental* re-indexing: before upserting chunks from
    a file, delete any existing vectors that came from the same
    source_file.  This lets you re-process a single manual without
    touching the rest of the index.

Metadata schema stored per vector:
  source_file    – original filename
  page           – page number (1-indexed)
  section_type   – heading | paragraph | table | list
  chunk_id       – unique chunk identifier
  context_prefix – nearest heading (may be null)
  make           – vehicle make (populated downstream if available)
  model          – vehicle model
  year           – model year (int)
  subsystem      – brake | engine | electrical | …
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from pinecone import Pinecone, ServerlessSpec

from src.config import settings
from src.ingestion.embedder import EmbeddedChunk

logger = logging.getLogger(__name__)

_UPSERT_BATCH = 50
_DIMENSION = 3072  # text-embedding-3-large


class PineconeIndexer:
    """Manages upsert and lifecycle of the Pinecone index.

    Usage:
        indexer = PineconeIndexer()
        indexer.ensure_index()                        # create if missing
        indexer.delete_by_source("civic_2022.pdf")    # optional re-index
        indexer.upsert(embedded_chunks)
    """

    def __init__(self):
        self.pc = Pinecone(api_key=settings.pinecone_api_key)
        self.index_name = settings.pinecone_index_name
        self._index = None  # lazily initialised after ensure_index()

    # ── Index lifecycle ───────────────────────────────────────────
    def ensure_index(self) -> None:
        """Create the index if it doesn't exist; then connect."""
        existing = [idx.name for idx in self.pc.list_indexes()]
        if self.index_name not in existing:
            logger.info("Creating Pinecone index '%s'…", self.index_name)
            self.pc.create_index(
                name=self.index_name,
                dimension=_DIMENSION,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region=settings.pinecone_environment,
                ),
            )
            logger.info("Index '%s' created.", self.index_name)
        else:
            logger.info("Index '%s' already exists.", self.index_name)

        self._index = self.pc.Index(self.index_name)

    @property
    def index(self):
        if self._index is None:
            raise RuntimeError(
                "Index not initialised. Call ensure_index() first."
            )
        return self._index

    # ── Delete helpers ────────────────────────────────────────────
    def delete_by_source(self, source_file: str) -> None:
        """Delete all vectors that originated from *source_file*.

        Uses Pinecone's filter-based delete (requires the index to
        have been created with a filter-deletable namespace or the
        vectors to have the source_file metadata key).
        """
        logger.info("Deleting vectors for source '%s'…", source_file)
        try:
            self.index.delete(
                filter={"source_file": {"$eq": source_file}},
            )
            logger.info("Deleted vectors for '%s'.", source_file)
        except Exception as e:
            # On first run, vectors don't exist yet - this is fine
            logger.warning("Could not delete vectors (might be first run): %s", str(e))

    # ── Upsert ────────────────────────────────────────────────────
    def upsert(self, embedded_chunks: list[EmbeddedChunk]) -> int:
        """Upsert embedded chunks in batches.

        Returns the total number of vectors upserted.
        """
        if not embedded_chunks:
            logger.warning("No chunks to upsert.")
            return 0

        # Build the vector records
        records: list[tuple[str, list[float], dict[str, Any]]] = []
        for ec in embedded_chunks:
            vec_id = ec.metadata.get("chunk_id") or str(uuid.uuid4())
            # Ensure metadata values are Pinecone-compatible types
            meta = self._sanitise_metadata(ec.metadata)
            # Store the original text so we can return it at query time
            meta["text"] = ec.text
            records.append((vec_id, ec.vector, meta))

        # Batch upsert
        total = 0
        for i in range(0, len(records), _UPSERT_BATCH):
            batch = records[i: i + _UPSERT_BATCH]
            logger.info(
                "Upserting vectors %d–%d / %d…",
                i + 1, min(i + _UPSERT_BATCH, len(records)), len(records),
            )
            self.index.upsert(vectors=batch)
            total += len(batch)

        logger.info("Upserted %d vectors total.", total)
        return total

    # ── Metadata sanitisation ─────────────────────────────────────
    @staticmethod
    def _sanitise_metadata(meta: dict) -> dict:
        """Pinecone metadata values must be str | int | float | bool | list.

        Drop None values and convert anything else to str.
        """
        clean: dict[str, Any] = {}
        for k, v in meta.items():
            if v is None:
                continue
            if isinstance(v, (str, int, float, bool)):
                clean[k] = v
            elif isinstance(v, list):
                # Lists must be homogeneous; convert all to str
                clean[k] = [str(item) for item in v]
            else:
                clean[k] = str(v)
        return clean
