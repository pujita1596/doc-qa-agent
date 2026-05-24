# doc-qa-agent

An AI-powered document Q&A system built from scratch to demonstrate RAG, agents, and evaluation — no orchestration frameworks.

A user asks a question. An agent decides whether to search the internal document knowledge base or the web, retrieves relevant context, and returns a grounded answer with citations and a full decision trace.

---

## Architecture

```
User question
     │
     ▼
Agent (Claude)  ←── system prompt: always search documents first
     │
     ├── search_documents(query)
     │       └── embed query → ChromaDB cosine search → top-3 chunks
     │
     └── web_search(query)          ← fallback if docs have no answer
             └── DuckDuckGo (ddgs)
     │
     ▼
Grounded answer + citations + agent plan
```

**RAG pipeline** (built from scratch):
- Fixed-size chunking with overlap (500 chars / 100 char overlap)
- Embeddings via `all-MiniLM-L6-v2` (384-dim, runs locally)
- Vector store: ChromaDB with cosine similarity

**Agent layer**: Claude tool use API — two tools, agentic loop with `max_iterations` guard

**Eval pipeline**: LLM-as-judge scoring faithfulness and answer relevancy across 10 test queries

**API**: FastAPI `POST /ask` returning `answer`, `plan`

---

## Project structure

```
doc-qa-agent/
├── src/
│   ├── ingest.py        # chunk → embed → store in ChromaDB
│   ├── retriever.py     # embed query → cosine search → top-k chunks
│   ├── qa.py            # retriever + Claude → grounded answer (no agent)
│   ├── agent.py         # Claude with search_documents + web_search tools
│   ├── main.py          # FastAPI app
│   └── test_setup.py    # Anthropic API smoke test
├── evals/
│   ├── eval_pipeline.py # 10-question LLM-as-judge eval
│   └── eval_results.json
├── data/
│   └── sample.txt       # knowledge base document
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
python src/ingest.py
# Ingested 6 chunks from data/sample.txt
```

Drop any `.txt` files into `data/` and pass the path to `ingest()` to add them to the knowledge base.

---

## Usage

**Start the API server**

```bash
make serve
# or directly:
uvicorn main:app --app-dir src --host 0.0.0.0 --port 8000 --reload
```

**Ask a question**

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How does chunk overlap prevent context loss?"}'
```

**Response**

```json
{
  "answer": "Chunk overlap ensures that sentences split at a boundary still appear in full in at least one chunk [1]. Consecutive chunks share 50–100 characters so no information is lost at the edges.",
  "plan": [
    {"tool": "search_documents", "query": "chunk overlap context loss"}
  ]
}
```

**Health check**

```bash
curl http://localhost:8000/health
# {"status": "ok"}
```

---

## Evaluation

Runs 10 test questions through the agent, then scores each answer with Claude as the judge.

```bash
python evals/eval_pipeline.py
```

**Metrics**

| Metric | What it measures |
|---|---|
| **Faithfulness** | Fraction of answer claims that are grounded in retrieved context |
| **Answer relevancy** | How directly the answer addresses the question |

**Results on sample.txt**

| Metric | Score |
|---|---|
| Avg faithfulness | 0.78 |
| Avg answer relevancy | 0.96 |

Full per-question results saved to `evals/eval_results.json`.

---

## Key design decisions

**No orchestration framework.** RAG is implemented from scratch — chunking, embedding, vector search, prompt construction. This makes every step inspectable and debuggable.

**LLM-as-judge eval instead of RAGAS.** Same metrics (faithfulness, answer relevancy), implemented directly with Claude as the evaluator. No extra framework dependency, and the scoring logic is fully visible.

**Cosine similarity, not L2.** ChromaDB is configured with `hnsw:space: cosine`. Cosine measures the angle between vectors (semantic direction), not magnitude — more appropriate for text embeddings.

**Overlap chunking.** 100-character overlap between 500-character chunks ensures sentences cut at boundaries still appear complete in at least one chunk.

**System prompt forces tool use.** Without an explicit instruction, Claude answers from training knowledge. The system prompt ensures it always calls `search_documents` first, making it a true RAG system rather than a chatbot.

---

## Add your own documents

1. Place any `.txt` file in `data/`
2. Call `ingest()` with the path:
   ```python
   # in src/ingest.py __main__, or from a script:
   from ingest import ingest
   ingest("data/yourfile.txt")
   ```
3. The new chunks are added to the existing ChromaDB collection — no need to re-ingest previous files

> **Note:** `chroma_db/` is gitignored. Every fresh clone must run `python src/ingest.py` before the API will return answers. The setup instructions above cover this.

PDF support can be added with `pypdf` — `load_text()` in [src/ingest.py](src/ingest.py) is the only function that needs updating.
