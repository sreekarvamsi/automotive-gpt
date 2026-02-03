"""
Streaming response handler for OpenAI chat completions.

Wraps the OpenAI SDK's streaming mode so that tokens are yielded
one at a time as they arrive.  This gives the UI sub-second first-
token latency — critical for interactive technician workflows.

The streamer also accumulates the full response text so callers
can cache or log the complete answer after the stream is consumed.
"""

from __future__ import annotations

import logging
from typing import Generator

from openai import OpenAI

from src.config import settings

logger = logging.getLogger(__name__)


class StreamingResponse:
    """Wrapper around an OpenAI streaming completion.

    Attributes:
        full_text: Accumulates the complete response as tokens arrive.
                   Available after the generator is fully consumed.
    """

    def __init__(self, stream):
        self._stream = stream
        self.full_text: str = ""

    def __iter__(self) -> Generator[str, None, None]:
        """Yield individual text tokens from the stream."""
        chunks: list[str] = []
        for chunk in self._stream:
            # Each chunk has choices[0].delta.content
            content = chunk.choices[0].delta.content
            if content:
                chunks.append(content)
                yield content

        self.full_text = "".join(chunks)
        logger.debug("Stream complete. Total length: %d chars.", len(self.full_text))


class Streamer:
    """Creates streaming chat completions via OpenAI.

    Usage:
        streamer = Streamer()
        response = streamer.stream(messages=[...])
        for token in response:
            print(token, end="", flush=True)
        print()
        # response.full_text now contains the complete answer
    """

    def __init__(self):
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_model

    def stream(
        self,
        messages: list[dict],
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> StreamingResponse:
        """Open a streaming chat completion.

        Args:
            messages:    The full message list (system + few-shot + user).
            temperature: Lower = more deterministic (good for factual QA).
            max_tokens:  Maximum tokens in the response.

        Returns:
            A StreamingResponse that yields tokens when iterated.
        """
        logger.info("Opening streaming completion (model=%s, temp=%.1f)…", self.model, temperature)

        raw_stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )

        return StreamingResponse(raw_stream)
