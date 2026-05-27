# doc-qa-agent

An AI-powered document Q&A system built from scratch to demonstrate RAG, agents, and evaluation — no orchestration frameworks.

Ask a question in the browser. An agent searches your private local documents (or the web as a fallback), returns a grounded answer with citations, and links directly to the source files.

---

## Architecture

```
User question (browser UI)
        │
        ▼
   Agent (Claude)  ←── system prompt: always search documents first
        │
        ├── search_documents(query)
        │       └── embed query → ChromaDB cosine search → top-3 chunks
        │                                └── returns source file paths
        └── web_search(query)          ← fallback if docs have no answer
                └── DuckDuckGo (ddgs)
        │
        ▼
Grounded answer + citations + agent plan with clickable source links
```

**RAG pipeline** (built from scratch):
- Fixed-size chunking with overlap (500 chars / 100 char overlap)
- Embeddings via `all-MiniLM-L6-v2` (384-dim, runs fully locally)
- Vector store: ChromaDB with cosine similarity

**Agent layer**: Claude tool use API — two tools, agentic loop with `max_iterations` guard

**Eval pipeline**: LLM-as-judge scoring for faithfulness and answer relevancy

**Benchmark suite**: 3 controlled experiments measuring how chunking strategy, top-k, and prompt strictness affect quality

**Web UI**: Simple browser search bar served directly from FastAPI — no frontend framework

---

## Project structure

```
doc-qa-agent/
├── src/
│   ├── ingest.py        # chunk → embed → store in ChromaDB (.txt, .pdf, .docx)
│   ├── retriever.py     # embed query → cosine search → top-k chunks
│   ├── qa.py            # retriever + Claude → grounded answer (no agent)
│   ├── agent.py         # Claude with search_documents + web_search tools
│   ├── main.py          # FastAPI app + serves web UI
│   ├── watcher.py       # watches ~/Downloads, auto-ingests new files
│   ├── test_setup.py    # Anthropic API smoke test
│   └── static/
│       └── index.html   # browser search UI
├── evals/
│   ├── eval_pipeline.py # 10-question LLM-as-judge eval
│   ├── run_benchmarks.py# 3 experiments: chunking / top-k / prompt strictness
│   └── eval_results.json
├── data/
│   ├── sample.txt                    # RAG concepts
│   ├── python_best_practices.txt     # Python patterns
│   ├── machine_learning_fundamentals.txt
│   └── fastapi_and_apis.txt
├── chroma_db/           # persisted vector store (generated, not committed)
├── .env                 # ANTHROPIC_API_KEY (not committed)
├── Makefile
└── requirements.txt
```

---

## Setup

**1. Clone and create a virtual environment**

```bash
git clone <repo-url>
cd doc-qa-agent
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**2. Add your Anthropic API key**

```bash
echo "ANTHROPIC_API_KEY=your-key-here" > .env
```

**3. Ingest documents**

```bash
make ingest
# Ingested N chunks from sample.txt
```

Drop `.txt`, `.pdf`, or `.docx` files into `data/` and run `make ingest` to add them.

---

## Usage

**Start the server**

```bash
make serve
```

Open **http://localhost:8000** in your browser. Type a question and press Enter.

The answer renders with full markdown formatting. The "Tools used" section shows which tool was called, the query, and clickable links to the source documents that were retrieved.

---

**API** (for programmatic use)

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How does chunk overlap prevent context loss?"}'
```

```json
{
  "answer": "Chunk overlap ensures sentences split at a boundary still appear complete in at least one chunk [1]...",
  "plan": [
    {
      "tool": "search_documents",
      "query": "chunk overlap context loss",
      "sources": ["/abs/path/to/data/sample.txt"]
    }
  ]
}
```

---

## Auto-ingest from Downloads

Run the watcher in a separate terminal. Any `.txt`, `.pdf`, or `.docx` file dropped into `~/Downloads` is automatically ingested into the knowledge base.

```bash
make watch
# Watching /Users/you/Downloads for new .txt, .pdf, .docx files...
```

---

## Evaluation

**Eval pipeline** — 10 test questions, LLM-as-judge scoring:

```bash
make eval
```

| Metric | Score |
|---|---|
| Avg faithfulness | 0.78 |
| Avg answer relevancy | 0.96 |

Full results saved to `evals/eval_results.json`.

**Benchmark suite** — 3 controlled experiments:

```bash
make benchmark
```

| Experiment | What varies | Fixed |
|---|---|---|
| 1 | Chunk size (256 / 500 / 1000) | k=3, current prompt |
| 2 | Top-k retrieval (1 / 3 / 5 / 10) | best chunk config |
| 3 | Prompt strictness (loose / current / strict) | best chunk + k |

Results saved to `evals/benchmark_results.json` and `evals/benchmark_summary.md`.

---

## Key design decisions

**No orchestration framework.** RAG is implemented from scratch — chunking, embedding, vector search, prompt construction. Every step is inspectable and debuggable.

**LLM-as-judge eval instead of RAGAS.** Same metrics (faithfulness, answer relevancy), implemented directly with Claude as the evaluator. No extra framework dependency, scoring logic fully visible.

**Cosine similarity, not L2.** ChromaDB is configured with `hnsw:space: cosine`. Cosine measures semantic direction, not magnitude — more appropriate for text embeddings.

**Overlap chunking.** 100-character overlap between 500-character chunks ensures sentences cut at boundaries still appear complete in at least one chunk.

**System prompt forces tool use.** Without explicit instruction, Claude answers from training knowledge. The system prompt ensures it always calls `search_documents` first, making it a true RAG system.

**Deduplication on ingest.** `already_ingested()` checks ChromaDB metadata before embedding, so re-running `make ingest` or the watcher on an already-loaded file is a no-op.

---

## Add your own documents

1. Drop any `.txt`, `.pdf`, or `.docx` file into `data/`
2. Run `make ingest`

Or use the watcher — any file saved to `~/Downloads` is ingested automatically.

> `chroma_db/` is gitignored. Every fresh clone needs `make ingest` before the API returns answers.
