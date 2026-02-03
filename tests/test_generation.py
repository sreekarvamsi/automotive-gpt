"""Tests for src/generation/ — prompts, generator, streamer."""

from unittest.mock import patch, MagicMock

import pytest

from src.retrieval.dense_retriever import RetrievedChunk
from src.generation.prompts import SYSTEM_PROMPT, FEW_SHOT_EXAMPLES, format_context
from src.generation.generator import AutomotiveGenerator, GenerationResult


# ── Helper ────────────────────────────────────────────────────────────
def _chunk(text: str, source: str = "manual.pdf", page: int = 5, score: float = 0.85) -> RetrievedChunk:
    return RetrievedChunk(
        text=text,
        score=score,
        metadata={"source_file": source, "page": page, "section_type": "paragraph", "chunk_id": "c1"},
    )


# ── Prompt formatting ─────────────────────────────────────────────────
class TestFormatContext:
    def test_formats_single_chunk(self):
        chunks = [_chunk("Oil capacity is 3.7 quarts.", source="civic.pdf", page=42)]
        result = format_context(chunks)
        assert "[Source 1 — civic.pdf, Page 42]" in result
        assert "Oil capacity is 3.7 quarts." in result

    def test_formats_multiple_chunks(self):
        chunks = [
            _chunk("Chunk A", source="a.pdf", page=10),
            _chunk("Chunk B", source="b.pdf", page=20),
            _chunk("Chunk C", source="c.pdf", page=30),
        ]
        result = format_context(chunks)
        assert "[Source 1 — a.pdf, Page 10]" in result
        assert "[Source 2 — b.pdf, Page 20]" in result
        assert "[Source 3 — c.pdf, Page 30]" in result

    def test_empty_chunks_returns_no_context_message(self):
        result = format_context([])
        assert "No relevant context" in result

    def test_missing_page_omits_page_str(self):
        chunk = RetrievedChunk(
            text="No page info.",
            score=0.5,
            metadata={"source_file": "test.pdf", "chunk_id": "x"},  # no "page" key
        )
        result = format_context([chunk])
        assert "Page" not in result
        assert "[Source 1 — test.pdf]" in result


# ── Message assembly ──────────────────────────────────────────────────
class TestMessageAssembly:
    def test_message_structure(self):
        chunks = [_chunk("Some context.")]
        messages = AutomotiveGenerator._build_messages(
            query="What is the oil capacity?",
            context_chunks=chunks,
            conversation_history=None,
        )

        # First message is system prompt
        assert messages[0]["role"] == "system"
        assert "AutomotiveGPT" in messages[0]["content"]

        # Few-shot examples follow (2 examples × 2 messages each = 4)
        assert messages[1]["role"] == "user"
        assert messages[2]["role"] == "assistant"
        assert messages[3]["role"] == "user"
        assert messages[4]["role"] == "assistant"

        # Last message is the current user question
        assert messages[-1]["role"] == "user"
        assert "What is the oil capacity?" in messages[-1]["content"]
        assert "Some context." in messages[-1]["content"]

    def test_conversation_history_is_injected(self):
        history = [
            {"role": "user", "content": "Previous question"},
            {"role": "assistant", "content": "Previous answer"},
        ]
        messages = AutomotiveGenerator._build_messages(
            query="Follow-up question",
            context_chunks=[_chunk("ctx")],
            conversation_history=history,
        )
        # History should appear after few-shot, before current question
        # Find "Previous question" in the messages
        prev_idx = next(i for i, m in enumerate(messages) if "Previous question" in m["content"])
        curr_idx = next(i for i, m in enumerate(messages) if "Follow-up question" in m["content"])
        assert prev_idx < curr_idx


# ── Confidence & source extraction ────────────────────────────────────
class TestConfidenceAndSources:
    def test_confidence_is_mean_of_top3(self):
        chunks = [
            _chunk("A", score=0.9),
            _chunk("B", score=0.8),
            _chunk("C", score=0.7),
            _chunk("D", score=0.1),  # 4th chunk should be ignored
        ]
        conf = AutomotiveGenerator._compute_confidence(chunks)
        expected = round((0.9 + 0.8 + 0.7) / 3, 3)
        assert conf == expected

    def test_confidence_empty_chunks_is_zero(self):
        assert AutomotiveGenerator._compute_confidence([]) == 0.0

    def test_extract_sources(self):
        chunks = [
            _chunk("A", source="a.pdf", page=10, score=0.9),
            _chunk("B", source="b.pdf", page=20, score=0.7),
        ]
        sources = AutomotiveGenerator._extract_sources(chunks)
        assert len(sources) == 2
        assert sources[0]["source_id"] == 1
        assert sources[0]["source_file"] == "a.pdf"
        assert sources[0]["page"] == 10
        assert sources[1]["source_file"] == "b.pdf"


# ── Generator (mocked OpenAI) ─────────────────────────────────────────
class TestAutomotiveGenerator:
    @patch("src.generation.generator.OpenAI")
    def test_generate_returns_result(self, mock_openai_cls):
        # Mock the OpenAI client
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="The oil capacity is 3.7 quarts."))]
        mock_response.usage = MagicMock(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        mock_client.chat.completions.create.return_value = mock_response

        generator = AutomotiveGenerator(streaming=False)
        chunks = [_chunk("Oil capacity: 3.7 quarts.", score=0.88)]

        result = generator.generate(
            query="What is the oil capacity?",
            context_chunks=chunks,
        )

        assert isinstance(result, GenerationResult)
        assert "3.7 quarts" in result.answer
        assert result.confidence > 0
        assert len(result.sources) == 1
        assert result.usage["total_tokens"] == 150

    @patch("src.generation.generator.OpenAI")
    def test_generate_with_empty_context(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="No information found."))]
        mock_response.usage = MagicMock(prompt_tokens=80, completion_tokens=10, total_tokens=90)
        mock_client.chat.completions.create.return_value = mock_response

        generator = AutomotiveGenerator(streaming=False)
        result = generator.generate(query="Random question", context_chunks=[])

        assert result.confidence == 0.0
        assert result.sources == []
