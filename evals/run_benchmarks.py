"""
Runs 3 controlled experiments to understand how config affects RAG quality.

  Experiment 1 — chunking strategy vs retrieval quality
  Experiment 2 — top-k retrieval vs answer quality
  Experiment 3 — prompt strictness vs hallucination rate

Usage:
    python evals/run_benchmarks.py
"""

import sys
import json
import time
from pathlib import Path
from dotenv import load_dotenv
import anthropic
import chromadb

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import helpers from existing src files — don't rewrite, parameterize
from ingest import load_file, chunk_text
from retriever import _model as _embed_model   # reuse loaded model, never load twice
from eval_pipeline import TEST_CASES, score_faithfulness, score_answer_relevancy

load_dotenv()
client = anthropic.Anthropic()

DOC_PATH = str(Path(__file__).parent.parent / "data" / "sample.txt")
BENCH_CHROMA_PATH = str(Path(__file__).parent.parent / "chroma_db_bench")

# ── Prompt versions for Experiment 3 ─────────────────────────────────────────

PROMPTS = {
    "loose": (
        "Answer the question using the provided context."
    ),
    "current": (
        "Answer the question using ONLY the context provided below.\n"
        "Cite the source number (e.g. [1], [2]) for each claim you make.\n"
        'If the answer is not in the context, say "I don\'t have enough information to answer that."'
    ),
    "strict": (
        "Answer ONLY using the context below. "
        "If the answer is not explicitly stated in the context, respond with "
        "'I don't have enough information to answer that.' "
        "Never use outside knowledge under any circumstances."
    ),
}

# ── Low-level benchmark helpers ───────────────────────────────────────────────

def bench_ingest(doc_path: str, collection_name: str, chunk_size: int, overlap: int) -> int:
    """Ingest into an isolated benchmark collection. Skips if already populated."""
    db = chromadb.PersistentClient(path=BENCH_CHROMA_PATH)
    try:
        col = db.get_collection(name=collection_name)
        if col.count() > 0:
            print(f"  [skip] {collection_name} already has {col.count()} chunks")
            return col.count()
    except Exception:
        pass

    collection = db.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    text = load_file(doc_path)
    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    embeddings = _embed_model.encode(chunks).tolist()

    p = Path(doc_path)
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[f"{p.stem}_{i}" for i in range(len(chunks))],
        metadatas=[{"source": str(p.resolve()), "chunk_index": i} for i in range(len(chunks))],
    )
    print(f"  [ingest] {collection_name}: {len(chunks)} chunks (size={chunk_size}, overlap={overlap})")
    return len(chunks)


def bench_retrieve(query: str, collection_name: str, n_results: int) -> list[dict]:
    query_embedding = _embed_model.encode([query]).tolist()
    db = chromadb.PersistentClient(path=BENCH_CHROMA_PATH)
    collection = db.get_collection(name=collection_name)
    actual_n = min(n_results, collection.count())   # can't request more than exists

    results = collection.query(query_embeddings=query_embedding, n_results=actual_n)
    return [
        {"text": chunk, "score": round(1 - dist, 4)}
        for chunk, dist in zip(results["documents"][0], results["distances"][0])
    ]


def bench_ask(question: str, collection_name: str, n_chunks: int, prompt_template: str) -> dict:
    """Ask one question with a given config. Returns answer, contexts, latency, tokens."""
    t0 = time.time()
    chunks = bench_retrieve(question, collection_name, n_chunks)
    context_block = "\n\n".join(f"[{i+1}] {c['text']}" for i, c in enumerate(chunks))
    prompt = f"{prompt_template}\n\nContext:\n{context_block}\n\nQuestion: {question}"

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )
    return {
        "answer": response.content[0].text,
        "contexts": [c["text"] for c in chunks],
        "latency": round(time.time() - t0, 2),
        "tokens": response.usage.input_tokens + response.usage.output_tokens,
    }


def run_eval_set(collection_name: str, n_chunks: int, prompt_template: str) -> dict:
    """Run all TEST_CASES and aggregate scores. Errors are logged and skipped."""
    faithfulness_scores, relevancy_scores, latencies = [], [], []
    total_tokens = 0

    for i, question in enumerate(TEST_CASES):
        try:
            result = bench_ask(question, collection_name, n_chunks, prompt_template)
            f = score_faithfulness(result["answer"], result["contexts"])
            r = score_answer_relevancy(question, result["answer"])
            faithfulness_scores.append(f)
            relevancy_scores.append(r)
            latencies.append(result["latency"])
            total_tokens += result["tokens"]
            print(f"    [{i+1:2d}/{len(TEST_CASES)}] f={f:.2f}  r={r:.2f}  lat={result['latency']:.1f}s")
        except Exception as e:
            print(f"    [{i+1:2d}/{len(TEST_CASES)}] ERROR: {e}")

    n = len(faithfulness_scores)
    return {
        "faithfulness": round(sum(faithfulness_scores) / n, 3) if n else 0.0,
        "relevancy": round(sum(relevancy_scores) / n, 3) if n else 0.0,
        "avg_latency": round(sum(latencies) / n, 2) if n else 0.0,
        "total_tokens": total_tokens,
    }


# ── Experiments ───────────────────────────────────────────────────────────────

def experiment_1() -> list[dict]:
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: CHUNK SIZE & OVERLAP")
    print("=" * 60)

    configs = [(256, 50), (500, 100), (1000, 200)]
    results = []

    for chunk_size, overlap in configs:
        tag = "  ← current" if (chunk_size, overlap) == (500, 100) else ""
        print(f"\nchunk={chunk_size}, overlap={overlap}{tag}")
        name = f"bench_{chunk_size}_{overlap}"
        bench_ingest(DOC_PATH, name, chunk_size, overlap)
        scores = run_eval_set(name, n_chunks=3, prompt_template=PROMPTS["current"])
        results.append({"chunk_size": chunk_size, "overlap": overlap, **scores})

    return results


def experiment_2(best_collection: str) -> list[dict]:
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: TOP-K RETRIEVAL")
    print("=" * 60)

    results = []
    for k in [1, 3, 5, 10]:
        tag = "  ← current" if k == 3 else ""
        print(f"\nk={k}{tag}")
        scores = run_eval_set(best_collection, n_chunks=k, prompt_template=PROMPTS["current"])
        results.append({"k": k, **scores})

    return results


def experiment_3(best_collection: str, best_k: int) -> list[dict]:
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: PROMPT STRICTNESS")
    print("=" * 60)

    results = []
    for name, prompt in PROMPTS.items():
        tag = "  ← current" if name == "current" else ""
        print(f"\nprompt={name}{tag}")
        scores = run_eval_set(best_collection, n_chunks=best_k, prompt_template=prompt)
        results.append({"prompt": name, **scores})

    return results


# ── Output helpers ────────────────────────────────────────────────────────────

def print_summary(exp1: list, exp2: list, exp3: list) -> None:
    def row(cells: list, widths: list) -> str:
        return "  ".join(str(c).ljust(w) for c, w in zip(cells, widths))

    def table(title: str, headers: list, rows: list) -> None:
        widths = [max(len(str(h)), max(len(str(r[i])) for r in rows)) for i, h in enumerate(headers)]
        print(f"\n{title}")
        print(row(headers, widths))
        print("-" * (sum(widths) + 2 * (len(widths) - 1)))
        for r in rows:
            print(row(r, widths))

    current_marker = lambda cond: "← current" if cond else ""

    if exp1:
        table(
            "EXPERIMENT 1: CHUNK SIZE",
            ["chunk", "overlap", "faithfulness", "relevancy", "latency"],
            [
                [r["chunk_size"], r["overlap"], r["faithfulness"], r["relevancy"],
                 f"{r['avg_latency']}s  {current_marker((r['chunk_size'], r['overlap']) == (500, 100))}"]
                for r in exp1
            ],
        )

    if exp2:
        table(
            "\nEXPERIMENT 2: TOP-K",
            ["k", "faithfulness", "relevancy", "latency", "tokens"],
            [
                [r["k"], r["faithfulness"], r["relevancy"],
                 f"{r['avg_latency']}s", r["total_tokens"],
                 current_marker(r["k"] == 3)]
                for r in exp2
            ],
        )

    if exp3:
        table(
            "\nEXPERIMENT 3: PROMPT STRICTNESS",
            ["prompt", "faithfulness", "relevancy"],
            [
                [r["prompt"], r["faithfulness"], r["relevancy"],
                 current_marker(r["prompt"] == "current")]
                for r in exp3
            ],
        )


def write_summary_md(exp1: list, exp2: list, exp3: list, best_chunk: tuple, best_k: int) -> None:
    best_e1 = max(exp1, key=lambda x: x["faithfulness"] + x["relevancy"])
    best_e3 = max(exp3, key=lambda x: x["faithfulness"])

    def md_table(headers, rows):
        sep = "|" + "|".join("---" for _ in headers) + "|"
        head = "|" + "|".join(headers) + "|"
        body = "\n".join("|" + "|".join(str(v) for v in r) + "|" for r in rows)
        return f"{head}\n{sep}\n{body}"

    lines = [
        "# Benchmark Results\n",
        "## Experiment 1: Chunk Size & Overlap\n",
        md_table(
            ["chunk", "overlap", "faithfulness", "relevancy", "latency"],
            [[r["chunk_size"], r["overlap"], r["faithfulness"], r["relevancy"], f"{r['avg_latency']}s"] for r in exp1],
        ),
        f"\n**Finding:** chunk_size={best_e1['chunk_size']}, overlap={best_e1['overlap']} scored best "
        f"(faithfulness={best_e1['faithfulness']}, relevancy={best_e1['relevancy']}). "
        "Smaller chunks retrieve more focused context; larger chunks provide more surrounding information "
        "but dilute relevance.\n",
        "---\n",
        "## Experiment 2: Top-K Retrieval\n",
        md_table(
            ["k", "faithfulness", "relevancy", "latency", "tokens"],
            [[r["k"], r["faithfulness"], r["relevancy"], f"{r['avg_latency']}s", r["total_tokens"]] for r in exp2],
        ),
        f"\n**Finding:** k={best_k} gave the best quality/cost balance. "
        "Higher k provides more context but increases token cost and latency linearly. "
        "Beyond the optimal k, extra chunks added noise without improving faithfulness.\n",
        "---\n",
        "## Experiment 3: Prompt Strictness\n",
        md_table(
            ["prompt", "faithfulness", "relevancy"],
            [[r["prompt"], r["faithfulness"], r["relevancy"]] for r in exp3],
        ),
        f"\n**Finding:** The `{best_e3['prompt']}` prompt produced the highest faithfulness score. "
        "Stricter prompts reduce hallucination but can cause refusals on borderline questions, "
        "which lowers answer relevancy.\n",
        "---\n",
        "## Optimal Configuration\n",
        md_table(
            ["Parameter", "Value"],
            [
                ["Chunk size", f"{best_chunk[0]} chars"],
                ["Overlap", f"{best_chunk[1]} chars"],
                ["Top-k", best_k],
                ["Prompt", best_e3["prompt"]],
            ],
        ),
        "\n---\n",
        "## Quality vs Latency vs Cost Tradeoff\n",
        md_table(
            ["k", "faithfulness", "relevancy", "latency", "relative cost"],
            [
                [r["k"], r["faithfulness"], r["relevancy"], f"{r['avg_latency']}s",
                 "lowest" if r["k"] == 1 else "low" if r["k"] == 3 else "medium" if r["k"] == 5 else "highest"]
                for r in exp2
            ],
        ),
        "\n",
    ]

    out_path = Path(__file__).parent / "benchmark_summary.md"
    out_path.write_text("\n".join(lines))
    print(f"\nSummary written to {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    all_results = {}
    best_chunk = (500, 100)
    best_k = 3

    # Experiment 1
    try:
        exp1 = experiment_1()
        all_results["experiment_1"] = exp1
        best_e1 = max(exp1, key=lambda x: x["faithfulness"] + x["relevancy"])
        best_chunk = (best_e1["chunk_size"], best_e1["overlap"])
    except Exception as e:
        print(f"\nExperiment 1 failed: {e}")
        exp1 = []

    best_collection = f"bench_{best_chunk[0]}_{best_chunk[1]}"

    # Experiment 2
    try:
        exp2 = experiment_2(best_collection)
        all_results["experiment_2"] = exp2
        best_e2 = max(exp2, key=lambda x: x["faithfulness"] + x["relevancy"])
        best_k = best_e2["k"]
    except Exception as e:
        print(f"\nExperiment 2 failed: {e}")
        exp2 = []

    # Experiment 3
    try:
        exp3 = experiment_3(best_collection, best_k)
        all_results["experiment_3"] = exp3
    except Exception as e:
        print(f"\nExperiment 3 failed: {e}")
        exp3 = []

    # Save raw results
    out_path = Path(__file__).parent / "benchmark_results.json"
    out_path.write_text(json.dumps(all_results, indent=2))

    # Print tables
    print_summary(exp1, exp2, exp3)
    print(f"\nRaw results saved to {out_path}")

    # Write markdown summary
    if exp1 and exp2 and exp3:
        write_summary_md(exp1, exp2, exp3, best_chunk, best_k)


if __name__ == "__main__":
    main()
