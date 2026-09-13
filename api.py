import os
from glob import glob
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.hybrid_retrieval_engine import HybridRetrievalEngine


app = FastAPI(
    title="Hybrid RAG API",
    version="1.0.0",
    description=(
        "Hybrid Retrieval-Augmented Generation API using "
        "BM25 + dense embeddings + RRF + Cross-Encoder reranking."
    ),
)

DATA_PATH = Path("data/sample_documents")
document_paths = sorted(glob(str(DATA_PATH / "*.txt")))

if not document_paths:
    raise RuntimeError(
        f"No .txt documents found under {DATA_PATH.resolve()}"
    )

documents: List[str] = []
for file_path in document_paths:
    documents.append(
        Path(file_path).read_text(encoding="utf-8").strip()
    )

engine = HybridRetrievalEngine(documents)


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_candidates: int = Field(default=25, ge=1, le=100)
    final_k: int = Field(default=5, ge=1, le=25)


class AskRequest(QueryRequest):
    pass


def build_context(results: list[dict]) -> str:
    return "\n\n".join(
        f"[Document {item['doc_id']}]\n{item['text']}"
        for item in results
    )


def generate_with_openai(query: str, results: list[dict]) -> str:
    """
    Optional answer generation.

    Set OPENAI_API_KEY and optionally OPENAI_MODEL in the environment
    to turn retrieval into a full RAG answer. Without a key, /ask
    returns retrieved evidence instead of pretending generation occurred.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return (
            "LLM generation is not configured. "
            "Retrieved evidence:\n\n"
            + build_context(results)
        )

    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    model = os.getenv("OPENAI_MODEL", "gpt-5.6-luna")

    context = build_context(results)
    prompt = (
        "Answer the user's question using only the retrieved context. "
        "Do not invent facts. Cite the supporting document IDs inline.\n\n"
        f"Question: {query}\n\n"
        f"Retrieved context:\n{context}"
    )

    response = client.responses.create(
        model=model,
        input=prompt,
    )
    return response.output_text


@app.get("/health")
def health():
    return {
        "status": "ok",
        "documents": len(documents),
        "device": engine._device,
    }


@app.post("/retrieve")
def retrieve(request: QueryRequest):
    try:
        results = engine.retrieve(
            request.query,
            top_candidates=request.top_candidates,
            final_k=request.final_k,
        )
        return {
            "query": request.query,
            "results": results,
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/ask")
def ask(request: AskRequest):
    try:
        results = engine.retrieve(
            request.query,
            top_candidates=request.top_candidates,
            final_k=request.final_k,
        )
        answer = generate_with_openai(request.query, results)
        return {
            "query": request.query,
            "answer": answer,
            "sources": results,
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
