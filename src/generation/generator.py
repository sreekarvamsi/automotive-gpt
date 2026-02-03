"""
GPT-4-turbo generation with retrieval-augmented context.

This is the module that actually calls the LLM.  It:
  1. Assembles the full message list in the correct order:
       system prompt → few-shot examples → conversation history
       → current context + question
  2. Supports both streaming (via Streamer) and non-streaming modes.
  3. Returns a GenerationResult that bundles the answer text with
     lightweight confidence metadata (token usage, model, latency).

Confidence score is approximated from the rerank scores of the
retrieved chunks — if the top chunks scored high, the answer is
likely well-grounded; if they scored low, the model may be
hallucinating and the score reflects that risk.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Generator

from openai import OpenAI

from src.config import settings
from src.retrieval.dense_retriever import RetrievedChunk
from src.generation.prompts import SYSTEM_PROMPT, FEW_SHOT_EXAMPLES, format_context
from src.generation.streamer import Streamer

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    """The output of a generation call."""
    answer: str
    confidence: float                              # 0–1, approximated from retrieval scores
    sources: list[dict] = field(default_factory=list)  # [{source_file, page, score}, …]
    latency_ms: int = 0
    usage: dict = field(default_factory=dict)      # {prompt_tokens, completion_tokens, …}


class AutomotiveGenerator:
    """Assembles prompts and calls GPT-4-turbo.

    Args:
        streaming: If True, generate() returns a generator of tokens
                   instead of a complete string.  Default False.
    """

    def __init__(self, streaming: bool = False):
        self.streaming = streaming
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_model
        self.streamer = Streamer() if streaming else None

    # ── Public interface ──────────────────────────────────────────
    def generate(
        self,
        query: str,
        context_chunks: list[RetrievedChunk],
        conversation_history: list[dict] | None = None,
    ) -> GenerationResult | Generator[str, None, None]:
        """Generate an answer grounded in retrieved context.

        Args:
            query:               The user's current question.
            context_chunks:      Reranked chunks from HybridRetriever.
            conversation_history: Prior (role, content) dicts for multi-turn.

        Returns:
            GenerationResult (non-streaming) or a token generator (streaming).
        """
        messages = self._build_messages(query, context_chunks, conversation_history)

        if self.streaming:
            return self._generate_streaming(messages, context_chunks)
        return self._generate_sync(messages, context_chunks)

    # ── Streaming path ────────────────────────────────────────────
    def _generate_streaming(
        self,
        messages: list[dict],
        context_chunks: list[RetrievedChunk],
    ) -> Generator[str, None, None]:
        """Yield tokens one at a time via the Streamer."""
        stream_response = self.streamer.stream(messages)
        for token in stream_response:
            yield token
        # Note: full_text and usage are available on stream_response
        # after iteration, but generators can't return values directly.
        # The API layer handles caching the full text separately.

    # ── Non-streaming path ────────────────────────────────────────
    def _generate_sync(
        self,
        messages: list[dict],
        context_chunks: list[RetrievedChunk],
    ) -> GenerationResult:
        """Blocking call — returns complete GenerationResult."""
        start = time.perf_counter()

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.2,
            max_tokens=2048,
            timeout=30
        )

        elapsed_ms = int((time.perf_counter() - start) * 1000)
        answer = response.choices[0].message.content or ""

        return GenerationResult(
            answer=answer,
            confidence=self._compute_confidence(context_chunks),
            sources=self._extract_sources(context_chunks),
            latency_ms=elapsed_ms,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
        )

    # ── Message assembly ──────────────────────────────────────────
    @staticmethod
    def _build_messages(
        query: str,
        context_chunks: list[RetrievedChunk],
        conversation_history: list[dict] | None,
    ) -> list[dict]:
        """Build the full messages list in order.

        Layout:
          [0]   system prompt
          [1-4] few-shot examples (user/assistant pairs)
          [5…]  conversation history (prior turns)
          [-1]  current user turn (context + question)
        """
        messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

        # Few-shot examples — each example is a user + assistant pair
        for example in FEW_SHOT_EXAMPLES:
            user_content = (
                f"Context:\n{example['context']}\n\n"
                f"Question: {example['question']}"
            )
            messages.append({"role": "user", "content": user_content})
            messages.append({"role": "assistant", "content": example["answer"]})

        # Prior conversation turns
        if conversation_history:
            messages.extend(conversation_history)

        # Current turn — inject retrieved context above the question
        context_text = format_context(context_chunks)
        current_user_content = (
            f"Context from service manuals:\n{context_text}\n\n"
            f"Question: {query}"
        )
        messages.append({"role": "user", "content": current_user_content})

        return messages

    # ── Confidence & source extraction ────────────────────────────
    @staticmethod
    def _compute_confidence(chunks: list[RetrievedChunk]) -> float:
        """Approximate confidence from retrieval scores.

        Uses the mean of the top-3 rerank scores.  If no chunks were
        retrieved, confidence is 0.
        """
        if not chunks:
            return 0.0
        top_scores = [c.score for c in chunks[:3]]
        return round(sum(top_scores) / len(top_scores), 3)

    @staticmethod
    def _extract_sources(chunks: list[RetrievedChunk]) -> list[dict]:
        """Return a lightweight source list for the API response."""
        sources: list[dict] = []
        for idx, chunk in enumerate(chunks, start=1):
            sources.append({
                "source_id": idx,
                "source_file": chunk.metadata.get("source_file", "unknown"),
                "page": chunk.metadata.get("page"),
                "section_type": chunk.metadata.get("section_type"),
                "score": round(chunk.score, 3),
            })
        return sources
