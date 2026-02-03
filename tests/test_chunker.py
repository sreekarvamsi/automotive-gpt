"""Tests for src/ingestion/chunker.py"""

import pytest

from src.ingestion.parser import ParsedSection
from src.ingestion.chunker import HybridChunker, Chunk, _recursive_split, _add_overlap, _token_len


# ── Unit tests for internal helpers ──────────────────────────────────
class TestRecursiveSplit:
    def test_short_text_is_not_split(self):
        text = "This is a short sentence."
        result = _recursive_split(text, max_chars=200)
        assert result == [text]

    def test_splits_on_double_newline(self):
        text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
        result = _recursive_split(text, max_chars=30)
        assert len(result) >= 2
        # Each piece should be ≤ 30 chars
        for piece in result:
            assert len(piece) <= 30

    def test_splits_on_sentence_boundary(self):
        text = "First sentence here. Second sentence here. Third sentence here."
        result = _recursive_split(text, max_chars=40)
        assert len(result) >= 2
        for piece in result:
            assert len(piece) <= 40

    def test_empty_string(self):
        assert _recursive_split("", max_chars=100) == []

    def test_whitespace_only(self):
        assert _recursive_split("   \n\n  ", max_chars=100) == []


class TestAddOverlap:
    def test_no_overlap_on_single_chunk(self):
        chunks = ["Only one chunk here."]
        result = _add_overlap(chunks, overlap_chars=20)
        assert result == chunks

    def test_overlap_prepends_tail_of_previous(self):
        chunks = ["First chunk content here.", "Second chunk content here."]
        result = _add_overlap(chunks, overlap_chars=10)
        # Second chunk should have some text from the first prepended
        assert len(result[1]) > len(chunks[1])
        # First chunk is unchanged
        assert result[0] == chunks[0]

    def test_zero_overlap_returns_unchanged(self):
        chunks = ["A", "B", "C"]
        assert _add_overlap(chunks, overlap_chars=0) == chunks


class TestTokenLen:
    def test_approximate_length(self):
        # 40 chars ≈ 10 tokens at 4 chars/token
        assert _token_len("a" * 40) == 10

    def test_minimum_is_one(self):
        assert _token_len("") == 1
        assert _token_len("ab") == 1  # 2 chars // 4 = 0, clamped to 1


# ── Integration tests for HybridChunker ──────────────────────────────
def _make_section(text: str, section_type: str = "paragraph", page: int = 1) -> ParsedSection:
    return ParsedSection(
        text=text,
        metadata={"source_file": "test.pdf", "page": page, "section_type": section_type},
    )


class TestHybridChunker:
    def test_short_sections_merge_into_single_chunk(self):
        sections = [
            _make_section("Short para one."),
            _make_section("Short para two."),
            _make_section("Short para three."),
        ]
        chunker = HybridChunker(chunk_size=512, chunk_overlap=50)
        chunks = chunker.chunk(sections)
        # All three should merge into one chunk (total text is tiny)
        assert len(chunks) == 1
        assert "Short para one" in chunks[0].text
        assert "Short para three" in chunks[0].text

    def test_tables_are_never_split(self):
        # Create a table that exceeds normal chunk size
        big_table = "| " + " | ".join([f"Col{i}" for i in range(20)]) + " |\n"
        big_table += ("| " + " | ".join(["data"] * 20) + " |\n") * 10

        sections = [_make_section(big_table, section_type="table")]
        chunker = HybridChunker(chunk_size=64, chunk_overlap=10)  # very small chunk size
        chunks = chunker.chunk(sections)

        # Table should be a single chunk regardless of size
        table_chunks = [c for c in chunks if c.metadata.get("section_type") == "table"]
        assert len(table_chunks) == 1
        assert big_table.strip() in table_chunks[0].text

    def test_heading_becomes_context_prefix(self):
        sections = [
            _make_section("Brake System Overview", section_type="heading"),
            _make_section("The brake pads should be replaced every 30,000 miles."),
        ]
        chunker = HybridChunker(chunk_size=512, chunk_overlap=50)
        chunks = chunker.chunk(sections)

        # At least one chunk should have the heading as context_prefix
        prefixed = [c for c in chunks if c.metadata.get("context_prefix") == "Brake System Overview"]
        assert len(prefixed) >= 1

    def test_long_paragraph_is_split(self):
        # 200 words ≈ 1000 chars → should exceed a 128-token (512 char) budget
        long_text = " ".join(["word"] * 200)
        sections = [_make_section(long_text)]
        chunker = HybridChunker(chunk_size=128, chunk_overlap=20)
        chunks = chunker.chunk(sections)
        assert len(chunks) > 1
        # Each chunk should be ≤ budget (with some tolerance for overlap)
        for chunk in chunks:
            # Overlap can push slightly over; allow 2x budget
            assert len(chunk.text) <= 128 * 4 * 2  # chars

    def test_chunk_ids_are_unique(self):
        sections = [
            _make_section("Para " + str(i) + " with some content here.") for i in range(10)
        ]
        chunker = HybridChunker(chunk_size=128, chunk_overlap=20)
        chunks = chunker.chunk(sections)
        ids = [c.metadata["chunk_id"] for c in chunks]
        assert len(ids) == len(set(ids))  # all unique

    def test_empty_input_returns_empty(self):
        chunker = HybridChunker()
        assert chunker.chunk([]) == []

    def test_metadata_flows_through(self):
        sections = [
            ParsedSection(
                text="Test paragraph with enough content.",
                metadata={
                    "source_file": "manual.pdf",
                    "page": 42,
                    "section_type": "paragraph",
                    "make": "Honda",
                },
            )
        ]
        chunker = HybridChunker(chunk_size=512, chunk_overlap=50)
        chunks = chunker.chunk(sections)
        assert chunks[0].metadata["source_file"] == "manual.pdf"
        assert chunks[0].metadata["page"] == 42
        assert chunks[0].metadata["make"] == "Honda"
