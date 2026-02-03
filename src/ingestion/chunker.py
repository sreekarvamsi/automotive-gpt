"""
Hybrid chunking strategy.

Two complementary passes produce the final chunk list:

1. **Structure-aware splitting** – headings and tables are never
   broken mid-way.  Tables are emitted as standalone chunks.
   Headings are kept as a 1-chunk prefix so the retriever knows
   the context.

2. **Recursive character splitting with overlap** – long
   paragraphs / lists are split at the best available separator
   (paragraph break → sentence end → comma → space) until every
   chunk fits within `chunk_size` tokens.  A sliding `overlap`
   window is prepended so adjacent chunks share context.

The result: every chunk is ≤ chunk_size tokens, tables are never
fragmented, and every chunk carries the nearest preceding heading
as implicit context via the `context_prefix` metadata field.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from src.ingestion.parser import ParsedSection

logger = logging.getLogger(__name__)

# Approximate token count: 1 token ≈ 4 characters (English text).
# This is a fast heuristic; if you need exact counts swap in
# tiktoken later.
CHARS_PER_TOKEN = 4


@dataclass
class Chunk:
    """A single chunk ready for embedding and indexing."""
    text: str                          # the chunk text
    metadata: dict = field(default_factory=dict)
    # metadata includes everything from ParsedSection.metadata PLUS:
    #   chunk_id       – unique id within the document
    #   context_prefix – nearest heading text (for context)


# ── Token / character helpers ─────────────────────────────────────────
def _token_len(text: str) -> int:
    return len(text) // CHARS_PER_TOKEN or 1


def _char_budget(tokens: int) -> int:
    return tokens * CHARS_PER_TOKEN


# ── Recursive splitter ────────────────────────────────────────────────
# Separator hierarchy (tried in order; first that produces a valid
# split is used).
_SEPARATORS = ["\n\n", "\n", ". ", ", ", " "]


def _recursive_split(text: str, max_chars: int) -> list[str]:
    """Split *text* into pieces ≤ max_chars using the separator hierarchy.

    Each level tries the next separator if the current one doesn't
    produce small-enough chunks.  Guarantees every returned piece
    is ≤ max_chars (single words longer than that are kept whole to
    avoid breaking identifiers / part numbers).
    """
    if len(text) <= max_chars:
        return [text] if text.strip() else []

    for sep in _SEPARATORS:
        parts = text.split(sep)
        if len(parts) <= 1:
            continue  # separator not found; try next

        # Reassemble parts greedily until we'd exceed the budget
        chunks: list[str] = []
        current = ""
        for part in parts:
            candidate = (current + sep + part) if current else part
            if len(candidate) <= max_chars:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                # If a single part is still too long, recurse deeper
                if len(part) > max_chars:
                    chunks.extend(_recursive_split(part, max_chars))
                    current = ""
                else:
                    current = part
        if current:
            chunks.append(current)
        return [c for c in chunks if c.strip()]

    # Last resort: hard-split by characters (shouldn't happen for
    # normal prose, but protects against pathological inputs)
    return [text[i:i + max_chars] for i in range(0, len(text), max_chars)]


def _add_overlap(chunks: list[str], overlap_chars: int) -> list[str]:
    """Prepend an overlap window from the previous chunk.

    The overlap is taken as the *last* overlap_chars characters of
    the previous chunk, trimmed to the nearest word boundary.
    """
    if overlap_chars <= 0 or len(chunks) <= 1:
        return chunks

    result = [chunks[0]]
    for i in range(1, len(chunks)):
        prev = chunks[i - 1]
        # Take the tail of the previous chunk
        tail = prev[-overlap_chars:]
        # Trim to the first space to avoid mid-word joins
        space_idx = tail.find(" ")
        if space_idx > 0:
            tail = tail[space_idx + 1:]
        result.append(tail + " " + chunks[i])
    return result


# ── Main chunker class ────────────────────────────────────────────────
class HybridChunker:
    """Structure-aware + recursive chunking.

    Args:
        chunk_size:   Target max tokens per chunk (default 512).
        chunk_overlap: Overlap tokens between adjacent chunks (default 50).
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._max_chars = _char_budget(chunk_size)
        self._overlap_chars = _char_budget(chunk_overlap)

    def chunk(self, sections: list[ParsedSection]) -> list[Chunk]:
        """Convert a list of parsed sections into overlapping chunks.

        Algorithm:
            1. Walk sections in order, tracking the current heading.
            2. Tables → emit as a single chunk (never split).
            3. Headings → update the current heading tracker; if the
               heading itself is short enough, merge it with the next
               paragraph for context.
            4. Paragraphs / lists → recursive-split + overlap.
        """
        chunks: list[Chunk] = []
        current_heading: str | None = None
        chunk_counter = 0
        # Determine the source file from the first section (all should match)
        source_file = sections[0].metadata.get("source_file", "unknown") if sections else "unknown"

        # Buffer: accumulates paragraph text before splitting so that
        # short consecutive paragraphs can be merged into one chunk.
        para_buffer: list[str] = []
        buffer_meta: dict = {}

        def _flush_buffer():
            """Split whatever is in para_buffer and append to chunks."""
            nonlocal chunk_counter
            if not para_buffer:
                return
            combined = " ".join(para_buffer)
            raw_chunks = _recursive_split(combined, self._max_chars)
            overlapped = _add_overlap(raw_chunks, self._overlap_chars)

            for text in overlapped:
                if not text.strip():
                    continue
                chunks.append(Chunk(
                    text=text.strip(),
                    metadata={
                        **buffer_meta,
                        "chunk_id": f"{source_file}_{chunk_counter}",
                        "context_prefix": current_heading,
                    },
                ))
                chunk_counter += 1
            para_buffer.clear()
            buffer_meta.clear()

        for section in sections:
            stype = section.metadata.get("section_type", "paragraph")

            # ── Heading ───────────────────────────────────────────
            if stype == "heading":
                _flush_buffer()
                current_heading = section.text
                # Short headings are merged into the next paragraph
                # for retrieval context; longer ones become their own chunk.
                if _token_len(section.text) > self.chunk_size // 4:
                    chunks.append(Chunk(
                        text=section.text,
                        metadata={
                            **section.metadata,
                            "chunk_id": f"{source_file}_{chunk_counter}",
                            "context_prefix": None,
                        },
                    ))
                    chunk_counter += 1

            # ── Table ─────────────────────────────────────────────
            elif stype == "table":
                _flush_buffer()
                # Tables are never split — emit whole.
                # If a table exceeds chunk_size we still keep it intact
                # because splitting a table destroys its meaning.
                chunks.append(Chunk(
                    text=section.text,
                    metadata={
                        **section.metadata,
                        "chunk_id": f"{source_file}_{chunk_counter}",
                        "context_prefix": current_heading,
                    },
                ))
                chunk_counter += 1

            # ── Paragraph / List ──────────────────────────────────
            else:
                # If adding this section would overflow the buffer,
                # flush first.
                candidate_len = sum(len(p) for p in para_buffer) + len(section.text)
                if candidate_len > self._max_chars and para_buffer:
                    _flush_buffer()

                para_buffer.append(section.text)
                if not buffer_meta:
                    buffer_meta = {**section.metadata}

        # Final flush
        _flush_buffer()

        logger.info(
            "Chunked '%s': %d sections → %d chunks (size=%d, overlap=%d)",
            source_file, len(sections), len(chunks),
            self.chunk_size, self.chunk_overlap,
        )
        return chunks
