"""
Hybrid retriever — the single entry point for all retrieval.

Pipeline:
    1. Run DenseRetriever  (semantic cosine search)   → top-K candidates
    2. Run SparseRetriever (BM25 keyword search)      → top-K candidates
    3. Merge via Reciprocal Rank Fusion (RRF)         → deduplicated pool
    4. Rerank merged pool with Cohere cross-encoder   → final top-N

Reciprocal Rank Fusion (RRF)
    score(d) = Σ  1 / (k + rank_i(d))
               i ∈ {dense, sparse}

    where k = 60 (standard default).  RRF is rank-based so it
    naturally handles the different score scales of cosine vs BM25
    without any normalisation.

The final output is a short list (typically 5 chunks) that feeds
directly into the generation prompt.
"""

from __future__ import annotations

import logging
from collections import defaultdict

from src.config import settings
from src.retrieval.dense_retriever import DenseRetriever, RetrievedChunk
from src.retrieval.sparse_retriever import SparseRetriever
from src.retrieval.reranker import Reranker

logger = logging.getLogger(__name__)

# RRF constant — higher k smooths out rank differences more.
# k=60 is the standard choice from the original RRF paper.
_RRF_K = 60


class HybridRetriever:
    """Orchestrates the full dense → sparse → RRF → rerank pipeline.

    Args:
        top_k:    Candidates to fetch from each sub-retriever.
        rerank_n: Final number of results after reranking.
    """

    def __init__(
        self,
        top_k: int | None = None,
        rerank_n: int | None = None,
    ):
        self.top_k = top_k or settings.top_k_retrieval
        self.rerank_n = rerank_n or settings.rerank_top_n

        self.dense = DenseRetriever(top_k=self.top_k)
        self.sparse = SparseRetriever(top_k=self.top_k)
        self.reranker = Reranker(top_n=self.rerank_n)

    # ── Public interface ──────────────────────────────────────────

    def retrieve(
        self, 
        query: str, 
        filters: dict | None = None, 
        top_k: int | None = None
    ) -> list[RetrievedChunk]:
        """Hybrid retrieval with comparison support"""
        
        # NEW CODE: Detect comparison queries
        comparison_keywords = ['compare', 'vs', 'versus', 'between', 'difference']
        is_comparison = any(kw in query.lower() for kw in comparison_keywords)
        
        if is_comparison:
            # Extract vehicle names
            vehicles = []
            vehicle_terms = {
                'civic': 'Honda Civic',
                'camry': 'Toyota Camry', 
                'f-150': 'Ford F-150',
                'f150': 'Ford F-150',
                'model 3': 'Tesla Model 3'
            }
            
            for term, full_name in vehicle_terms.items():
                if term in query.lower():
                    vehicles.append(full_name)
            
            # If we found multiple vehicles, do separate retrievals
            if len(vehicles) >= 2:
                all_results = []
                topic = query.lower()
                for vehicle in vehicles:
                    # Clean up the query
                    for term in vehicle_terms.keys():
                        topic = topic.replace(term, '')
                    for kw in comparison_keywords:
                        topic = topic.replace(kw, '')
                    topic = topic.strip()
                    
                    # Retrieve for each vehicle
                    vehicle_query = f"{topic} {vehicle}"
                    vehicle_results = self._retrieve_single(vehicle_query, filters, top_k=3)
                    all_results.extend(vehicle_results[:2])  # Top 2 from each
                
                return all_results[:5]  # Return top 5 total
        
        # EXISTING CODE: Normal single retrieval
        return self._retrieve_single(query, filters, top_k)

    def _retrieve_single(self, query: str, filters: dict | None = None, top_k: int | None = None):
        """Original retrieve logic (rename existing retrieve code to this)"""
        # Move all the existing retrieve() code here
        # (everything that was in retrieve() before)
        
        # ── Step 1 & 2: parallel retrieval ──────────────────────
        logger.info("HybridRetriever: running dense + sparse retrieval…")
        dense_results = self.dense.retrieve(query, filters)
        sparse_results = self.sparse.retrieve(query, filters)

        logger.info(
            "Dense: %d results | Sparse: %d results",
            len(dense_results), len(sparse_results),
        )

        # ── Step 3: RRF merge ───────────────────────────────────
        merged = self._reciprocal_rank_fusion(dense_results, sparse_results)
        logger.info("After RRF merge: %d unique candidates.", len(merged))

        # ── Step 4: Cohere reranking ────────────────────────────
        reranked = self.reranker.rerank(query, merged)
        logger.info("After reranking: %d final results.", len(reranked))

        return reranked

    # ── RRF merger ────────────────────────────────────────────────
    @staticmethod
    def _reciprocal_rank_fusion(
        *result_lists: list[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        """Merge multiple ranked lists via Reciprocal Rank Fusion.

        Documents are keyed by (source_file, chunk_id) so duplicates
        from dense and sparse are detected and their RRF scores summed.

        Returns candidates sorted by descending RRF score.
        """
        rrf_scores: dict[str, float] = defaultdict(float)
        chunk_map: dict[str, RetrievedChunk] = {}

        for ranked_list in result_lists:
            for rank, chunk in enumerate(ranked_list, start=1):
                key = chunk.metadata.get("chunk_id") or str(hash(chunk.text))
                rrf_scores[key] += 1.0 / (_RRF_K + rank)
                if key not in chunk_map:
                    chunk_map[key] = chunk

        sorted_keys = sorted(rrf_scores, key=lambda k: rrf_scores[k], reverse=True)
        merged: list[RetrievedChunk] = []
        for key in sorted_keys:
            chunk = chunk_map[key]
            merged.append(RetrievedChunk(
                text=chunk.text,
                score=rrf_scores[key],
                metadata=chunk.metadata,
            ))

        return merged
