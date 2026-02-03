#!/usr/bin/env python3
"""
CLI: Run the evaluation suite.

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --test-dir src/evaluation/test_cases
    python scripts/evaluate.py --output eval_report.json
    python scripts/evaluate.py --test-id tc_001_oil_capacity   # single test

Runs every test case through the full pipeline (retrieve → generate)
and scores each answer with GPT-4-as-judge + automated citation checks.
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.evaluator import Evaluator, load_test_cases


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run the AutomotiveGPT evaluation suite."
    )
    parser.add_argument(
        "--test-dir",
        default="src/evaluation/test_cases",
        help="Directory containing test case JSON files.",
    )
    parser.add_argument(
        "--output",
        default="eval_report.json",
        help="Path for the output JSON report.",
    )
    parser.add_argument(
        "--test-id",
        default=None,
        help="Run only a single test case by ID.",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable debug logging.",
    )
    args = parser.parse_args()
    setup_logging(args.verbose)
    logger = logging.getLogger("evaluate")

    # ── Load test cases ───────────────────────────────────────────
    test_dir = Path(args.test_dir)
    if not test_dir.is_dir():
        logger.error("Test directory not found: %s", test_dir)
        sys.exit(1)

    test_cases = load_test_cases(test_dir)
    if not test_cases:
        logger.error("No test cases found in %s", test_dir)
        sys.exit(1)

    # Filter to single test if --test-id specified
    if args.test_id:
        test_cases = [tc for tc in test_cases if tc.id == args.test_id]
        if not test_cases:
            logger.error("Test case '%s' not found.", args.test_id)
            sys.exit(1)
        logger.info("Running single test: %s", args.test_id)

    logger.info("Running %d test case(s)…", len(test_cases))

    # ── Run evaluation ────────────────────────────────────────────
    evaluator = Evaluator()
    results = evaluator.run(test_cases)

    # ── Print results table ───────────────────────────────────────
    print()
    print("=" * 90)
    print(f"{'Test ID':<35} {'Relevance':>10} {'Faithful':>10} {'Citation':>10} {'Latency':>10}")
    print("-" * 90)
    for r in results:
        print(
            f"{r.test_id:<35} "
            f"{r.relevance_score:>9.1%} "
            f"{r.faithfulness_score:>9.1%} "
            f"{r.citation_accuracy:>9.1%} "
            f"{r.latency_ms:>8}ms"
        )
    print("-" * 90)

    if results:
        n = len(results)
        print(
            f"{'AVERAGE':<35} "
            f"{sum(r.relevance_score for r in results)/n:>9.1%} "
            f"{sum(r.faithfulness_score for r in results)/n:>9.1%} "
            f"{sum(r.citation_accuracy for r in results)/n:>9.1%} "
            f"{sum(r.latency_ms for r in results)/n:>7.0f}ms"
        )
    print("=" * 90)

    # ── Save report ───────────────────────────────────────────────
    evaluator.save_report(results, args.output)
    print(f"\n📄 Report saved → {args.output}")


if __name__ == "__main__":
    main()
