import heapq
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List

import numpy as np
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer


class HybridRetrievalEngine:
    """
    Hybrid retrieval engine:
    BM25 sparse retrieval + dense semantic retrieval
    -> Reciprocal Rank Fusion (RRF)
    -> Cross-Encoder reranking
    """

    def __init__(
        self,
        corpus: List[str],
        dense_model: str = "BAAI/bge-large-en-v1.5",
        reranker_model: str = "BAAI/bge-reranker-large"
    ):
        if not corpus:
            raise ValueError("corpus must contain at least one document.")

        self.corpus = corpus

        # Prefer GPU, then Apple Silicon MPS, otherwise CPU.
        if torch.cuda.is_available():
            device = "cuda"
        elif (
            getattr(torch.backends, "mps", None) is not None
            and torch.backends.mps.is_available()
        ):
            device = "mps"
        else:
            device = "cpu"

        self._device = device
        self._encode_batch_size = 64 if device != "cpu" else 32
        self._rerank_batch_size = 32 if device != "cpu" else 16

        # Load expensive models once at startup.
        self.dense_encoder = SentenceTransformer(
            dense_model,
            device=self._device,
        )
        self.reranker = CrossEncoder(
            reranker_model,
            device=self._device,
        )

        # Build BM25 index once.
        self.tokenized_corpus = [
            doc.lower().split()
            for doc in corpus
        ]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

        # Precompute document embeddings once.
        self.doc_embeddings = self.dense_encoder.encode(
            corpus,
            batch_size=self._encode_batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32, copy=False)

        # BM25 and dense retrieval can run concurrently.
        self._executor = ThreadPoolExecutor(max_workers=2)

    def __del__(self):
        executor = getattr(self, "_executor", None)
        if executor is not None:
            executor.shutdown(wait=False)

    def close(self) -> None:
        """Release the retrieval thread pool."""
        executor = getattr(self, "_executor", None)
        if executor is not None:
            executor.shutdown(wait=False)
            self._executor = None

    def _rrf(
        self,
        rank_lists: List[List[int]],
        k: int = 60
    ) -> Dict[int, float]:
        """Fuse ranked lists using Reciprocal Rank Fusion."""
        rrf_scores: Dict[int, float] = {}

        for r_list in rank_lists:
            for rank, doc_id in enumerate(r_list):
                rrf_scores[doc_id] = (
                    rrf_scores.get(doc_id, 0.0)
                    + 1.0 / (k + rank + 1)
                )

        return rrf_scores

    def retrieve(
        self,
        query: str,
        top_candidates: int = 25,
        final_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Retrieve the most relevant documents for a query.

        Pipeline:
        1. BM25 retrieval
        2. Dense semantic retrieval
        3. RRF fusion
        4. Cross-Encoder reranking
        5. Return final_k documents
        """
        if not query or not query.strip():
            raise ValueError("query must not be empty.")

        if top_candidates < 1 or final_k < 1:
            raise ValueError("top_candidates and final_k must be >= 1.")

        effective_candidates = min(
            top_candidates,
            len(self.corpus),
        )

        def _sparse_ranks() -> List[int]:
            query_tokens = query.lower().split()
            bm25_scores = self.bm25.get_scores(query_tokens)

            if effective_candidates == len(bm25_scores):
                sparse_candidate_ids = np.arange(len(bm25_scores))
            else:
                sparse_candidate_ids = np.argpartition(
                    bm25_scores,
                    -effective_candidates,
                )[-effective_candidates:]

            ordered = sparse_candidate_ids[
                np.argsort(
                    bm25_scores[sparse_candidate_ids]
                )[::-1]
            ]
            return ordered.tolist()

        def _dense_ranks() -> List[int]:
            query_emb = self.dense_encoder.encode(
                [query],
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )[0].astype(np.float32, copy=False)

            # Document embeddings and query embedding are normalized,
            # so dot product is equivalent to cosine similarity.
            dense_scores = np.dot(
                self.doc_embeddings,
                query_emb,
            )

            if effective_candidates == len(dense_scores):
                dense_candidate_ids = np.arange(len(dense_scores))
            else:
                dense_candidate_ids = np.argpartition(
                    dense_scores,
                    -effective_candidates,
                )[-effective_candidates:]

            ordered = dense_candidate_ids[
                np.argsort(
                    dense_scores[dense_candidate_ids]
                )[::-1]
            ]
            return ordered.tolist()

        sparse_future = self._executor.submit(_sparse_ranks)
        dense_future = self._executor.submit(_dense_ranks)

        sparse_ranks = sparse_future.result()
        dense_ranks = dense_future.result()

        # Fuse sparse + dense rankings.
        fused = self._rrf(
            [sparse_ranks, dense_ranks],
            k=60,
        )

        top_doc_ids = [
            doc_id
            for doc_id, _ in heapq.nlargest(
                effective_candidates,
                fused.items(),
                key=lambda item: item[1],
            )
        ]

        # Cross-Encoder evaluates query/document pairs jointly.
        pairs = [
            [query, self.corpus[doc_id]]
            for doc_id in top_doc_ids
        ]

        rerank_scores = np.asarray(
            self.reranker.predict(
                pairs,
                batch_size=self._rerank_batch_size,
                show_progress_bar=False,
            )
        )

        final_count = min(final_k, len(rerank_scores))

        if final_count == len(rerank_scores):
            top_final = np.arange(len(rerank_scores))
        else:
            top_final = np.argpartition(
                rerank_scores,
                -final_count,
            )[-final_count:]

        ranked_indices = top_final[
            np.argsort(
                rerank_scores[top_final]
            )[::-1]
        ]

        return [
            {
                "doc_id": top_doc_ids[idx],
                "text": self.corpus[top_doc_ids[idx]],
                "score": float(rerank_scores[idx]),
            }
            for idx in ranked_indices
        ]
