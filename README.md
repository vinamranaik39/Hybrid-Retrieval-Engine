Hybrid Retrieval Engine RAG

A production-oriented Hybrid Retrieval-Augmented Generation (RAG) system that combines:

BM25 sparse lexical retrieval

Dense semantic retrieval with Sentence Transformers

Reciprocal Rank Fusion (RRF)

Cross-Encoder reranking

GPU / MPS / CPU device selection

Concurrent sparse + dense retrieval

Efficient top-k selection

Optional FastAPI deployment

Optional LLM generation with OpenAI

Tests and sample documents

Why Hybrid RAG?

A vector-only retriever is strong at semantic similarity but can underperform on exact strings such as:

error codes

SKUs

identifiers

acronyms

domain-specific keywords

BM25 is strong at lexical matching, while dense embeddings capture meaning. Hybrid RAG uses both retrieval signals and merges their rankings before a second-stage reranker.

Architecture

                    USER QUERY
                         |
              +----------+----------+
              |                     |
              v                     v
           BM25              Dense Encoder
      lexical retrieval    semantic retrieval
              |                     |
              +----------+----------+
                         |
                        RRF
                  rank fusion
                         |
                  candidate pool
                         |
                  Cross-Encoder
                     reranking
                         |
                      Top-K
                         |
                 optional LLM
                         |
                   final answer

See docs/architecture.png for the visual architecture diagram.

Project Structure

hybrid-rag/
├── README.md
├── requirements.txt
├── .gitignore
├── .env.example
│
├── src/
│   └── hybrid_retrieval_engine.py
│
├── app/
│   └── api.py
│
├── examples/
│   └── basic_usage.py
│
├── tests/
│   └── test_hybrid_retrieval.py
│
├── data/
│   └── sample_documents/
│
└── docs/
    └── architecture.png

How the Retrieval Engine Works

1. Initialization

The system loads the dense encoder and Cross-Encoder once. It also builds the BM25 index and precomputes corpus embeddings.

This avoids repeatedly encoding the entire corpus for every query.

2. Sparse Retrieval

The query is tokenized and scored using BM25.

BM25 is useful when exact terms matter.

3. Dense Retrieval

The query is converted into a dense embedding and compared with precomputed document embeddings.

Because embeddings are normalized, a dot product is equivalent to cosine similarity.

4. Reciprocal Rank Fusion

The top-ranked BM25 and dense results are merged using:

[
RRF(d) = \sum_r \frac{1}{k + r(d)}
]

The implementation uses k=60.

5. Cross-Encoder Reranking

The RRF candidate pool is converted into:

[query, document]

pairs and scored jointly by a Cross-Encoder.

Only the final top-k documents are returned.

Performance-Oriented Design

The implementation includes:

device-aware inference: CUDA → MPS → CPU

batch encoding

batch reranking

concurrent BM25 and dense retrieval

np.argpartition() for partial top-k selection

heapq.nlargest() for fused candidates

float32 embeddings

For a very large corpus, the next scaling step would be replacing brute-force dense similarity with an ANN index such as FAISS or HNSW.

Installation

git clone <https://github.com/vinamranaik39/Hybrid-Retrieval-Engine.git>
cd hybrid-retrieval-engine-rag

python -m venv .venv

Windows PowerShell:

.\.venv\Scripts\Activate.ps1

Linux/macOS:

source .venv/bin/activate

Install dependencies:

pip install -r requirements.txt

The first run downloads the embedding and reranker models from Hugging Face.

Run the Example

python examples/basic_usage.py

Run Tests

pytest -q

Tests use deterministic dummy models, so the test suite does not need to download large ML models.

Run the API

uvicorn app.api:app --reload

Retrieval

POST /retrieve
Content-Type: application/json

{
  "query": "How does BM25 improve RAG?",
  "top_candidates": 5,
  "final_k": 3
}

Full RAG Answer

/ask performs retrieval and then uses an LLM if OPENAI_API_KEY is configured.

Create .env or export the variables in your shell:

OPENAI_API_KEY=your_key
OPENAI_MODEL=gpt-5.6-luna

Then:

POST /ask
Content-Type: application/json

{
  "query": "Why use BM25 together with dense retrieval?",
  "top_candidates": 5,
  "final_k": 3
}

Without an API key, /ask safely returns the retrieved evidence rather than fabricating an LLM-generated answer.

Interview Explanation

I built a Hybrid Retrieval Engine RAG retrieval engine because dense retrieval and lexical retrieval have different strengths. BM25 handles exact keyword matching, while dense embeddings capture semantic similarity. I run both retrieval paths, fuse their rankings using Reciprocal Rank Fusion, and then apply a Cross-Encoder for second-stage reranking. For deployment, I precompute document embeddings, select the best device, batch model inference, run sparse and dense retrieval concurrently, and use efficient top-k operations to reduce unnecessary sorting.

Limitations

This repository is intentionally compact and interview-friendly.

It currently uses:

in-memory BM25

in-memory document embeddings

brute-force dense similarity

a local sample corpus

optional external LLM generation

For larger production systems, consider:

FAISS/HNSW or a managed vector database

persistent document/index storage

document ingestion and chunking pipelines

authentication and authorization

request caching

observability and latency metrics

automated RAG evaluation

containerization and CI/CD


License

MIT