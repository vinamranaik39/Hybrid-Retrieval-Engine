import numpy as np
import pytest

import src.hybrid_retrieval_engine as module


class DummyEncoder:
    def encode(
        self,
        texts,
        batch_size=None,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ):
        # Fixed deterministic embeddings for tests.
        return np.tile(
            np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
            (len(texts), 1),
        )


class DummyReranker:
    def predict(
        self,
        pairs,
        batch_size=None,
        show_progress_bar=False,
    ):
        return np.arange(len(pairs), 0, -1, dtype=np.float32)


@pytest.fixture
def engine(monkeypatch):
    monkeypatch.setattr(
        module,
        "SentenceTransformer",
        lambda *args, **kwargs: DummyEncoder(),
    )
    monkeypatch.setattr(
        module,
        "CrossEncoder",
        lambda *args, **kwargs: DummyReranker(),
    )

    return module.HybridRetrievalEngine(
        [
            "BM25 is a lexical retrieval method.",
            "Dense retrieval captures semantic meaning.",
            "RRF combines rankings from multiple retrievers.",
        ]
    )


def test_retrieve_returns_expected_structure(engine):
    results = engine.retrieve(
        "How does BM25 work?",
        top_candidates=3,
        final_k=2,
    )

    assert len(results) == 2
    assert "doc_id" in results[0]
    assert "text" in results[0]
    assert "score" in results[0]


def test_final_k_is_respected(engine):
    results = engine.retrieve(
        "retrieval",
        top_candidates=3,
        final_k=1,
    )

    assert len(results) == 1


def test_empty_query_is_rejected(engine):
    with pytest.raises(ValueError):
        engine.retrieve("")


def test_empty_corpus_is_rejected():
    with pytest.raises(ValueError):
        module.HybridRetrievalEngine([])


def test_rrf_combines_rankings(engine):
    scores = engine._rrf(
        [[0, 1], [1, 0]],
        k=60,
    )

    assert scores[0] == pytest.approx(scores[1])
