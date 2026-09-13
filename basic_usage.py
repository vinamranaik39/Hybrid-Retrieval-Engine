from pathlib import Path

from src.hybrid_retrieval_engine import HybridRetrievalEngine


def load_documents() -> list[str]:
    data_dir = Path("data/sample_documents")
    return [
        path.read_text(encoding="utf-8").strip()
        for path in sorted(data_dir.glob("*.txt"))
    ]


def main() -> None:
    documents = load_documents()
    engine = HybridRetrievalEngine(documents)

    query = "How does BM25 improve hybrid RAG retrieval?"
    results = engine.retrieve(
        query,
        top_candidates=5,
        final_k=3,
    )

    print(f"\nQuery: {query}\n")
    for rank, result in enumerate(results, start=1):
        print(f"Result #{rank}")
        print(f"Document ID: {result['doc_id']}")
        print(f"Score: {result['score']:.4f}")
        print(f"Text: {result['text']}\n")

    engine.close()


if __name__ == "__main__":
    main()
