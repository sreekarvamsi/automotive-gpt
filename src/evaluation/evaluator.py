"""
Evaluation pipeline — GPT-4-as-judge + automated metrics.

Metrics:
  1. Answer Relevance   – GPT-4 judges correctness vs ground truth (0–1).
  2. Faithfulness       – GPT-4 checks grounding in retrieved context (0–1).
  3. Citation Accuracy  – Automated: did we retrieve an expected source? (0 or 1).

Usage:
    from src.evaluation.evaluator import Evaluator, load_test_cases

    cases = load_test_cases("src/evaluation/test_cases/")
    evaluator = Evaluator()
    results = evaluator.run(cases)
    evaluator.save_report(results, "eval_report.json")
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from openai import OpenAI

from src.config import settings
from src.retrieval.hybrid_retriever import HybridRetriever
from src.generation.generator import AutomotiveGenerator

logger = logging.getLogger(__name__)


# ── Data models ───────────────────────────────────────────────────────
@dataclass
class TestCase:
    id: str
    question: str
    expected_answer: str
    filters: dict | None = None
    expected_sources: list[str] = field(default_factory=list)


@dataclass
class EvalResult:
    test_id: str
    question: str
    generated_answer: str
    expected_answer: str
    relevance_score: float
    faithfulness_score: float
    citation_accuracy: float
    latency_ms: int
    sources_retrieved: list[str]


# ── Judge prompts ─────────────────────────────────────────────────────
_RELEVANCE_PROMPT = """\
You are an expert automotive engineer evaluating answer quality.

Question: {question}
Expected answer (ground truth): {expected}
Generated answer: {generated}

Score 0.0–1.0 on correctness, completeness, and precision of technical values.
Respond ONLY with valid JSON: {{"score": <float>, "reasoning": "<string>"}}
"""

_FAITHFULNESS_PROMPT = """\
Evaluate whether the answer is fully grounded in the context (no hallucination).

Context:
{context}

Answer:
{answer}

Score 0.0–1.0 (1.0 = fully grounded, 0.0 = significant hallucination).
Respond ONLY with valid JSON: {{"score": <float>, "reasoning": "<string>"}}
"""


class Evaluator:
    """Runs the full evaluation suite."""

    def __init__(self):
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.retriever = HybridRetriever()
        self.generator = AutomotiveGenerator(streaming=False)

    def run(self, test_cases: list[TestCase]) -> list[EvalResult]:
        results: list[EvalResult] = []
        for i, tc in enumerate(test_cases, 1):
            logger.info("Evaluating %d/%d: %s", i, len(test_cases), tc.id)
            try:
                results.append(self._evaluate_single(tc))
            except Exception as exc:
                logger.error("Failed on %s: %s", tc.id, exc)
        self._log_summary(results)
        return results

    def _evaluate_single(self, tc: TestCase) -> EvalResult:
        start = time.perf_counter()
        chunks = self.retriever.retrieve(tc.question, tc.filters)
        gen_result = self.generator.generate(query=tc.question, context_chunks=chunks)
        elapsed_ms = int((time.perf_counter() - start) * 1000)

        context_text = "\n\n".join(c.text for c in chunks)
        sources = list({c.metadata.get("source_file", "") for c in chunks})

        relevance = self._judge(
            _RELEVANCE_PROMPT.format(
                question=tc.question,
                expected=tc.expected_answer,
                generated=gen_result.answer,
            )
        )
        faithfulness = self._judge(
            _FAITHFULNESS_PROMPT.format(context=context_text, answer=gen_result.answer)
        )
        citation_acc = 1.0 if (
            not tc.expected_sources or set(tc.expected_sources) & set(sources)
        ) else 0.0

        return EvalResult(
            test_id=tc.id,
            question=tc.question,
            generated_answer=gen_result.answer,
            expected_answer=tc.expected_answer,
            relevance_score=relevance,
            faithfulness_score=faithfulness,
            citation_accuracy=citation_acc,
            latency_ms=elapsed_ms,
            sources_retrieved=sources,
        )

    def _judge(self, prompt: str) -> float:
        resp = self.client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "Respond only with valid JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=200,
        )
        try:
            data = json.loads(resp.choices[0].message.content or "{}")
            return max(0.0, min(1.0, float(data.get("score", 0.0))))
        except (json.JSONDecodeError, ValueError):
            return 0.0

    @staticmethod
    def _log_summary(results: list[EvalResult]) -> None:
        if not results:
            return
        n = len(results)
        logger.info("=" * 60)
        logger.info("EVALUATION SUMMARY (%d test cases)", n)
        logger.info("  Answer Relevance:   %.1f%%", sum(r.relevance_score for r in results) / n * 100)
        logger.info("  Faithfulness:       %.1f%%", sum(r.faithfulness_score for r in results) / n * 100)
        logger.info("  Citation Accuracy:  %.1f%%", sum(r.citation_accuracy for r in results) / n * 100)
        logger.info("  Avg Latency:        %.0f ms", sum(r.latency_ms for r in results) / n)
        logger.info("=" * 60)

    @staticmethod
    def save_report(results: list[EvalResult], output_path: str) -> None:
        data = [
            {
                "test_id": r.test_id, "question": r.question,
                "generated_answer": r.generated_answer, "expected_answer": r.expected_answer,
                "relevance_score": r.relevance_score, "faithfulness_score": r.faithfulness_score,
                "citation_accuracy": r.citation_accuracy, "latency_ms": r.latency_ms,
                "sources_retrieved": r.sources_retrieved,
            }
            for r in results
        ]
        Path(output_path).write_text(json.dumps(data, indent=2))
        logger.info("Report saved → %s", output_path)


def load_test_cases(directory: str | Path) -> list[TestCase]:
    """Load test cases from *.json files in a directory."""
    cases: list[TestCase] = []
    for f in sorted(Path(directory).glob("*.json")):
        raw = json.loads(f.read_text())
        for item in (raw if isinstance(raw, list) else [raw]):
            cases.append(TestCase(
                id=item["id"],
                question=item["question"],
                expected_answer=item["expected_answer"],
                filters=item.get("filters"),
                expected_sources=item.get("expected_sources", []),
            ))
    logger.info("Loaded %d test cases from %s", len(cases), directory)
    return cases
