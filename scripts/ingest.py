#!/usr/bin/env python3
"""
CLI: Run the document ingestion pipeline.

Usage:
    python scripts/ingest.py --source ./data/manuals
    python scripts/ingest.py --source ./data/manuals --verbose
    python scripts/ingest.py --source ./data/manuals --file civic_2022.pdf   # single file

The script:
  1. Scans the source directory for .pdf, .docx, .html files.
  2. For each file: parse → chunk → embed → upsert to Pinecone.
  3. Before upserting, deletes any existing vectors for that source
     file (incremental re-indexing).
  4. Prints a summary at the end.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# Ensure the project root is on sys.path so `src.*` imports work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion.parser import parse_document
from src.ingestion.chunker import HybridChunker
from src.ingestion.embedder import Embedder
from src.ingestion.indexer import PineconeIndexer
from src.config import settings


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Ingest vehicle service manuals into the AutomotiveGPT vector store."
    )
    parser.add_argument(
        "--source", required=True,
        help="Path to folder containing manuals (or a single file).",
    )
    parser.add_argument(
        "--file", default=None,
        help="Process only this specific file (must be inside --source).",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable debug-level logging.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Parse and chunk only — do NOT embed or upsert.",
    )
    args = parser.parse_args()
    setup_logging(args.verbose)
    logger = logging.getLogger("ingest")

    source_path = Path(args.source)
    if not source_path.exists():
        logger.error("Source path does not exist: %s", source_path)
        sys.exit(1)

    # ── Discover files ──────────────────────────────────────────────
    supported = {".pdf", ".docx", ".html", ".htm"}
    if source_path.is_file():
        files = [source_path]
    else:
        files = sorted(f for f in source_path.iterdir() if f.suffix.lower() in supported)

    # Filter to single file if --file was specified
    if args.file:
        files = [f for f in files if f.name == args.file]
        if not files:
            logger.error("File '%s' not found in %s", args.file, source_path)
            sys.exit(1)

    if not files:
        logger.error("No supported documents found in %s", source_path)
        sys.exit(1)

    logger.info("Found %d document(s) to process.", len(files))

    # ── Initialise pipeline components ──────────────────────────────
    chunker = HybridChunker(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    if not args.dry_run:
        embedder = Embedder()
        indexer = PineconeIndexer()
        indexer.ensure_index()
    else:
        embedder = None
        indexer = None
        logger.info("DRY RUN mode — skipping embedding and upsert.")

    # ── Process each file ───────────────────────────────────────────
    total_sections = 0
    total_chunks = 0
    total_vectors = 0
    errors: list[str] = []
    pipeline_start = time.perf_counter()

    for file_path in files:
        logger.info("─" * 50)
        logger.info("Processing: %s (%s)", file_path.name, _human_size(file_path.stat().st_size))
        file_start = time.perf_counter()

        try:
            # 1. Parse
            sections = parse_document(file_path)
            total_sections += len(sections)
            logger.info("  Parsed → %d sections", len(sections))

            # 2. Chunk
            chunks = chunker.chunk(sections)
            total_chunks += len(chunks)
            logger.info("  Chunked → %d chunks", len(chunks))

            if args.dry_run:
                # Print chunk previews in dry-run mode
                for i, chunk in enumerate(chunks[:3]):
                    logger.debug("    Chunk %d: %s…", i, chunk.text[:80])
                if len(chunks) > 3:
                    logger.debug("    … and %d more chunks", len(chunks) - 3)
                continue

            # 3. Delete old vectors for this file (incremental re-index)
            indexer.delete_by_source(file_path.name)

            # 4. Embed
            embedded = embedder.embed(chunks)
            logger.info("  Embedded → %d vectors", len(embedded))

            # 5. Upsert
            count = indexer.upsert(embedded)
            total_vectors += count
            logger.info("  Upserted → %d vectors", count)

            elapsed = time.perf_counter() - file_start
            logger.info("  ✅ Done in %.1fs", elapsed)

        except Exception as exc:
            logger.error("  ❌ Failed: %s", exc)
            errors.append(f"{file_path.name}: {exc}")

    # ── Summary ─────────────────────────────────────────────────────
    total_time = time.perf_counter() - pipeline_start
    logger.info("=" * 50)
    logger.info("INGESTION COMPLETE")
    logger.info("  Files processed:  %d", len(files))
    logger.info("  Total sections:   %d", total_sections)
    logger.info("  Total chunks:     %d", total_chunks)
    if not args.dry_run:
        logger.info("  Total vectors:    %d", total_vectors)
    logger.info("  Errors:           %d", len(errors))
    logger.info("  Total time:       %.1fs", total_time)
    if errors:
        logger.warning("Errors encountered:")
        for err in errors:
            logger.warning("  - %s", err)
    logger.info("=" * 50)


def _human_size(size_bytes: int) -> str:
    """Format bytes as human-readable string."""
    for unit in ("B", "KB", "MB", "GB"):
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


if __name__ == "__main__":
    main()
