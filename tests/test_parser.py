"""Tests for src/ingestion/parser.py"""

import tempfile
from pathlib import Path

import pytest

from src.ingestion.parser import (
    parse_document,
    ParsedSection,
    _clean,
    _table_to_text,
)


# ── Utilities ─────────────────────────────────────────────────────────
class TestUtilities:
    def test_clean_collapses_whitespace(self):
        assert _clean("  hello   world  ") == "hello world"

    def test_clean_strips_newlines(self):
        assert _clean("hello\n\nworld") == "hello world"

    def test_table_to_text_basic(self):
        rows = [["Make", "Model"], ["Honda", "Civic"]]
        result = _table_to_text(rows)
        assert "| Make" in result
        assert "| Honda" in result
        assert len(result.strip().split("\n")) == 2

    def test_table_to_text_empty(self):
        assert _table_to_text([]) == ""

    def test_table_to_text_ragged_rows(self):
        rows = [["A", "B", "C"], ["X"]]
        result = _table_to_text(rows)
        assert "| A" in result
        assert "| X" in result


# ── HTML Parser ───────────────────────────────────────────────────────
def _write_html(content: str) -> Path:
    tmp = tempfile.NamedTemporaryFile(suffix=".html", mode="w", delete=False, encoding="utf-8")
    tmp.write(f"<html><body>{content}</body></html>")
    tmp.close()
    return Path(tmp.name)


class TestHTMLParser:
    def test_parses_headings(self):
        sections = parse_document(_write_html("<h1>Engine Specs</h1><p>Some paragraph content here.</p>"))
        headings = [s for s in sections if s.metadata["section_type"] == "heading"]
        assert len(headings) >= 1
        assert headings[0].text == "Engine Specs"
        assert headings[0].metadata["level"] == 1

    def test_parses_unordered_list(self):
        sections = parse_document(_write_html("<ul><li>Item A</li><li>Item B</li><li>Item C</li></ul>"))
        lists = [s for s in sections if s.metadata["section_type"] == "list"]
        assert len(lists) == 1
        assert "• Item A" in lists[0].text
        assert lists[0].metadata["item_count"] == 3

    def test_parses_ordered_list(self):
        sections = parse_document(_write_html("<ol><li>Step one</li><li>Step two</li></ol>"))
        lists = [s for s in sections if s.metadata["section_type"] == "list"]
        assert "1. Step one" in lists[0].text
        assert "2. Step two" in lists[0].text

    def test_parses_table(self):
        html = "<table><tr><th>Spec</th><th>Value</th></tr><tr><td>Torque</td><td>25 N·m</td></tr></table>"
        sections = parse_document(_write_html(html))
        tables = [s for s in sections if s.metadata["section_type"] == "table"]
        assert len(tables) == 1
        assert "Torque" in tables[0].text
        assert tables[0].metadata["row_count"] == 2

    def test_strips_script_tags(self):
        sections = parse_document(_write_html('<script>alert("x")</script><p>Good content here.</p>'))
        all_text = " ".join(s.text for s in sections)
        assert "alert" not in all_text
        assert "Good content" in all_text

    def test_metadata_has_source_file(self):
        path = _write_html("<p>Hello world content here.</p>")
        sections = parse_document(path)
        for s in sections:
            assert s.metadata["source_file"] == path.name


# ── DOCX Parser ───────────────────────────────────────────────────────
def _create_docx(paragraphs: list[dict]) -> Path:
    from docx import Document
    doc = Document()
    for p in paragraphs:
        doc.add_paragraph(p["text"], style=p.get("style", "Normal"))
    tmp = tempfile.NamedTemporaryFile(suffix=".docx", delete=False)
    doc.save(tmp.name)
    tmp.close()
    return Path(tmp.name)


class TestDOCXParser:
    def test_parses_paragraphs(self):
        sections = parse_document(_create_docx([
            {"text": "First paragraph with enough content."},
            {"text": "Second paragraph with enough content."},
        ]))
        paras = [s for s in sections if s.metadata["section_type"] == "paragraph"]
        assert len(paras) >= 2

    def test_detects_headings(self):
        sections = parse_document(_create_docx([
            {"text": "Main Title", "style": "Heading 1"},
            {"text": "Body text here."},
        ]))
        headings = [s for s in sections if s.metadata["section_type"] == "heading"]
        assert len(headings) >= 1
        assert "Main Title" in headings[0].text

    def test_metadata_source_file(self):
        path = _create_docx([{"text": "Test content."}])
        sections = parse_document(path)
        for s in sections:
            assert s.metadata["source_file"] == path.name


# ── Entry point ───────────────────────────────────────────────────────
class TestParseDocument:
    def test_raises_on_missing_file(self):
        with pytest.raises(FileNotFoundError):
            parse_document("/nonexistent/path/file.pdf")

    def test_raises_on_unsupported_extension(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".xyz", delete=False)
        tmp.close()
        with pytest.raises(ValueError, match="Unsupported file type"):
            parse_document(tmp.name)
