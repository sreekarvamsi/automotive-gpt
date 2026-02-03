"""Tests for src/retrieval/ — dense, sparse, reranker, hybrid."""

from unittest.mock import patch, MagicMock

import pytest

from src.retrieval.dense_retriever import DenseRetriever, RetrievedChunk
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.sparse_retriever import SparseRetriever
from src.retrieval.reranker import Reranker


# ── Helper ────────────────────────────────────────────────────────────
def _chunk(text: str, score: float = 0.9, chunk_id: str = "c1", source: str = "manual.pdf", page: int = 1) -> RetrievedChunk:
    return RetrievedChunk(
        text=text,
        score=score,
        metadata={"chunk_id": chunk_id, "source_file": source, "page": page},
    )


# ── DenseRetriever filter builder ────────────────────────────────────
class TestDenseRetrieverFilterBuilder:
    def test_empty_filters_returns_none(self):
        assert DenseRetriever._build_filter({}) is None

    def test_single_filter(self):
        result = DenseRetriever._build_filter({"make": "Honda"})
        assert result == {"make": {"$eq": "Honda"}}

    def test_multiple_filters_uses_and(self):
        result = DenseRetriever._build_filter({"make": "Honda", "year": 2022})
        assert "$and" in result
        clauses = result["$and"]
        assert {"make": {"$eq": "Honda"}} in clauses
        assert {"year": {"$eq": 2022}} in clauses

    def test_none_values_are_skipped(self):
        result = DenseRetriever._build_filter({"make": "Honda", "model": None, "year": 2022})
        # model=None should be skipped
        if "$and" in result:
            keys = [list(c.keys())[0] for c in result["$and"]]
            assert "model" not in keys
        else:
            assert "model" not in result

    def test_unsupported_keys_are_ignored(self):
        result = DenseRetriever._build_filter({"make": "Honda", "color": "red"})
        # "color" is not a supported filter key
        assert result == {"make": {"$eq": "Honda"}}


# ── RRF merge ─────────────────────────────────────────────────────────
class TestRecipocalRankFusion:
    def test_single_list_preserves_order(self):
        chunks = [
            _chunk("A", score=0.9, chunk_id="a"),
            _chunk("B", score=0.7, chunk_id="b"),
            _chunk("C", score=0.5, chunk_id="c"),
        ]
        merged = HybridRetriever._reciprocal_rank_fusion(chunks)
        # Order should be preserved (rank 1 gets highest RRF score)
        assert merged[0].text == "A"
        assert merged[1].text == "B"
        assert merged[2].text == "C"

    def test_duplicate_across_lists_is_merged(self):
        list_a = [_chunk("shared", score=0.9, chunk_id="shared_id")]
        list_b = [_chunk("shared", score=0.8, chunk_id="shared_id")]
        merged = HybridRetriever._reciprocal_rank_fusion(list_a, list_b)
        # Should be deduplicated to one entry
        assert len(merged) == 1
        # RRF score should be sum of both ranks: 1/(60+1) + 1/(60+1)
        expected_score = 1.0 / 61 + 1.0 / 61
        assert abs(merged[0].score - expected_score) < 1e-6

    def test_higher_combined_rank_wins(self):
        # A appears at rank 1 in list_a, rank 3 in list_b
        # B appears at rank 2 in list_a, rank 1 in list_b
        list_a = [
            _chunk("A", chunk_id="a"),
            _chunk("B", chunk_id="b"),
        ]
        list_b = [
            _chunk("B", chunk_id="b"),
            _chunk("X", chunk_id="x"),
            _chunk("A", chunk_id="a"),
        ]
        merged = HybridRetriever._reciprocal_rank_fusion(list_a, list_b)
        # B: 1/(60+2) + 1/(60+1) vs A: 1/(60+1) + 1/(60+3)
        # B should have a higher combined score
        texts = [c.text for c in merged]
        assert texts.index("B") < texts.index("A")

    def test_empty_lists(self):
        merged = HybridRetriever._reciprocal_rank_fusion([], [])
        assert merged == []


# ── SparseRetriever post-filter ───────────────────────────────────────
class TestSparseRetrieverFilter:
    def test_matches_all_filters(self):
        meta = {"make": "Honda", "year": 2022, "subsystem": "brake"}
        filters = {"make": "Honda", "year": 2022}
        assert SparseRetriever._matches_filter(meta, filters) is True

    def test_fails_on_mismatch(self):
        meta = {"make": "Honda", "year": 2022}
        filters = {"make": "Toyota"}
        assert SparseRetriever._matches_filter(meta, filters) is False

    def test_none_filter_values_are_skipped(self):
        meta = {"make": "Honda"}
        filters = {"make": "Honda", "year": None}
        assert SparseRetriever._matches_filter(meta, filters) is True

    def test_missing_metadata_key_fails(self):
        meta = {"make": "Honda"}
        filters = {"make": "Honda", "year": 2022}
        # meta doesn't have "year" → should fail
        assert SparseRetriever._matches_filter(meta, filters) is False


# ── Reranker (mocked) ─────────────────────────────────────────────────
class TestReranker:
    @patch("src.retrieval.reranker.cohere.Client")
    def test_reranker_reorders_by_score(self, mock_client_cls):
        # Mock the Cohere client to return reversed order with scores
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        # Simulate Cohere returning index 1 first (higher score)
        mock_client.rerank.return_value = MagicMock(results=[
            MagicMock(index=1, relevance_score=0.95),
            MagicMock(index=0, relevance_score=0.60),
        ])

        reranker = Reranker(top_n=2)
        candidates = [
            _chunk("Low relevance", score=0.9, chunk_id="low"),
            _chunk("High relevance", score=0.8, chunk_id="high"),
        ]
        result = reranker.rerank("test query", candidates)

        assert len(result) == 2
        assert result[0].text == "High relevance"
        assert result[0].score == 0.95
        assert result[1].text == "Low relevance"
        assert result[1].score == 0.60

    @patch("src.retrieval.reranker.cohere.Client")
    def test_reranker_handles_empty_candidates(self, mock_client_cls):
        reranker = Reranker(top_n=5)
        result = reranker.rerank("query", [])
        assert result == []
