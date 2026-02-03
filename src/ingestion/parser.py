"""
Multi-format document parser.

Supports PDF, DOCX, and HTML. Every parsed unit is returned as a
ParsedSection carrying source metadata (file, page, type) that
flows through chunking all the way into the vector store.

Tables are serialised as pipe-delimited text so they survive
chunking as a single retrievable unit.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import PyPDF2
from docx import Document as DocxDocument
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

SectionType = Literal["heading", "paragraph", "table", "list"]


# ── Data model ────────────────────────────────────────────────────────
@dataclass
class ParsedSection:
    """A single logical unit extracted from a source document."""
    text: str
    metadata: dict = field(default_factory=dict)
    # metadata keys: source_file, page (1-indexed), section_type, ...extras


# ── Utility helpers ──────────────────────────────────────────────────
def _clean(text: str) -> str:
    """Strip and collapse internal whitespace."""
    return re.sub(r"\s+", " ", text).strip()


def _table_to_text(rows: list[list[str]]) -> str:
    """Serialise a 2-D table into aligned pipe-delimited text.

    Example:
        | Make  | Model | Year |
        | Honda | Civic | 2022 |
    """
    if not rows:
        return ""
    col_count = max(len(row) for row in rows)
    widths = [0] * col_count
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    lines: list[str] = []
    for row in rows:
        padded = [row[i].ljust(widths[i]) if i < len(row) else " " * widths[i]
                  for i in range(col_count)]
        lines.append("| " + " | ".join(padded) + " |")
    return "\n".join(lines)


# ── PDF Parser ────────────────────────────────────────────────────────
class PDFParser:
    """Extract text from PDF pages using PyPDF2 layout mode.

    Heuristics classify each block as heading / list / paragraph:
      - Short, ALL-CAPS lines  →  heading
      - Lines starting with bullet or number markers  →  list
      - Everything else  →  paragraph
    """

    def parse(self, file_path: Path) -> list[ParsedSection]:
        sections: list[ParsedSection] = []
        with open(file_path, "rb") as fh:
            reader = PyPDF2.PdfReader(fh)
            for page_idx, page in enumerate(reader.pages, start=1):
                raw = page.extract_text() or ""
                blocks = [b.strip() for b in raw.split("\n\n") if b.strip()]

                for block in blocks:
                    stype: SectionType
                    if len(block) < 120 and block.upper() == block and len(block.split()) <= 12:
                        stype = "heading"
                    elif block.lstrip().startswith(("•", "-", "–", "1.", "a.")):
                        stype = "list"
                    else:
                        stype = "paragraph"

                    sections.append(ParsedSection(
                        text=_clean(block),
                        metadata={
                            "source_file": file_path.name,
                            "page": page_idx,
                            "section_type": stype,
                        },
                    ))

        logger.info("PDF '%s' → %d sections, %d pages",
                    file_path.name, len(sections), len(reader.pages))
        return sections


# ── DOCX Parser ───────────────────────────────────────────────────────
class DOCXParser:
    """Extract paragraphs and tables from .docx in document order.

    python-docx doesn't expose a unified body iterator, so we walk
    the raw XML children and dispatch by tag name.  This preserves
    the interleaved order of paragraphs and tables exactly as the
    author wrote them.
    """

    def parse(self, file_path: Path) -> list[ParsedSection]:
        doc = DocxDocument(str(file_path))
        sections: list[ParsedSection] = []

        for element in doc.element.body:
            tag = element.tag.split("}")[-1]  # strip XML namespace

            if tag == "p":
                from docx.text.paragraph import Paragraph
                para = Paragraph(element, doc)
                text = _clean(para.text)
                if not text:
                    continue

                style_name = (para.style.name or "").lower() if para.style else ""
                if "heading" in style_name:
                    stype: SectionType = "heading"
                elif style_name.startswith("list"):
                    stype = "list"
                else:
                    stype = "paragraph"

                sections.append(ParsedSection(
                    text=text,
                    metadata={
                        "source_file": file_path.name,
                        "page": 1,  # python-docx has no page tracking
                        "section_type": stype,
                        "style": style_name,
                    },
                ))

            elif tag == "tbl":
                from docx.table import Table
                table = Table(element, doc)
                rows = [[_clean(cell.text) for cell in row.cells]
                        for row in table.rows]

                sections.append(ParsedSection(
                    text=_table_to_text(rows),
                    metadata={
                        "source_file": file_path.name,
                        "page": 1,
                        "section_type": "table",
                        "row_count": len(rows),
                        "col_count": max((len(r) for r in rows), default=0),
                    },
                ))

        logger.info("DOCX '%s' → %d sections", file_path.name, len(sections))
        return sections


# ── HTML Parser ───────────────────────────────────────────────────────
class HTMLParser:
    """Extract structured content from HTML via BeautifulSoup.

    Walks the <body> top-down.  Script/style/nav tags are stripped
    before processing so only content nodes remain.
    """

    SKIP_TAGS = {"script", "style", "nav", "footer", "header", "noscript", "svg"}

    def parse(self, file_path: Path) -> list[ParsedSection]:
        html = file_path.read_text(encoding="utf-8")
        soup = BeautifulSoup(html, "html.parser")

        for tag in soup.find_all(self.SKIP_TAGS):
            tag.decompose()

        sections: list[ParsedSection] = []
        body_children = soup.body.children if soup.body else []

        for element in body_children:
            if not hasattr(element, "name") or element.name is None:
                continue
            name = element.name.lower()

            # ── Headings ──────────────────────────────────────────
            if name in ("h1", "h2", "h3", "h4", "h5", "h6"):
                text = _clean(element.get_text())
                if text:
                    sections.append(ParsedSection(
                        text=text,
                        metadata={
                            "source_file": file_path.name,
                            "page": 1,
                            "section_type": "heading",
                            "level": int(name[1]),
                        },
                    ))

            # ── Lists ─────────────────────────────────────────────
            elif name in ("ul", "ol"):
                items = [_clean(li.get_text()) for li in element.find_all("li")
                         if li.get_text().strip()]
                if items:
                    lines = []
                    for idx, item in enumerate(items, 1):
                        prefix = f"{idx}." if name == "ol" else "•"
                        lines.append(f"{prefix} {item}")
                    sections.append(ParsedSection(
                        text="\n".join(lines),
                        metadata={
                            "source_file": file_path.name,
                            "page": 1,
                            "section_type": "list",
                            "item_count": len(items),
                        },
                    ))

            # ── Tables ────────────────────────────────────────────
            elif name == "table":
                rows: list[list[str]] = []
                for tr in element.find_all("tr"):
                    cells = [_clean(td.get_text()) for td in tr.find_all(("td", "th"))]
                    if cells:
                        rows.append(cells)
                if rows:
                    sections.append(ParsedSection(
                        text=_table_to_text(rows),
                        metadata={
                            "source_file": file_path.name,
                            "page": 1,
                            "section_type": "table",
                            "row_count": len(rows),
                            "col_count": max(len(r) for r in rows),
                        },
                    ))

            # ── Generic blocks (p, div, section, …) ──────────────
            elif name in ("p", "div", "section", "article", "main", "aside"):
                text = _clean(element.get_text())
                if text and len(text) > 10:
                    sections.append(ParsedSection(
                        text=text,
                        metadata={
                            "source_file": file_path.name,
                            "page": 1,
                            "section_type": "paragraph",
                        },
                    ))

        logger.info("HTML '%s' → %d sections", file_path.name, len(sections))
        return sections


# ── Public entry point ────────────────────────────────────────────────
_PARSERS: dict[str, type] = {
    ".pdf": PDFParser,
    ".docx": DOCXParser,
    ".htm": HTMLParser,
    ".html": HTMLParser,
}


def parse_document(file_path: str | Path) -> list[ParsedSection]:
    """Parse any supported document → flat list of ParsedSections.

    Raises:
        FileNotFoundError: path does not exist.
        ValueError: unsupported file extension.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Document not found: {path}")

    suffix = path.suffix.lower()
    if suffix not in _PARSERS:
        raise ValueError(
            f"Unsupported file type '{suffix}'. Supported: {list(_PARSERS.keys())}"
        )

    parser = _PARSERS[suffix]()
    return parser.parse(path)
