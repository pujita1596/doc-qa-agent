import sys
import json
from pathlib import Path
from dotenv import load_dotenv
import anthropic

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from agent import run_agent
from retriever import retrieve

load_dotenv()
client = anthropic.Anthropic()

# 10 questions grounded in sample.txt — mix of direct lookup and reasoning
TEST_CASES = [
    "Why does chunk overlap matter in a RAG pipeline?",
    "What are the two phases of a RAG pipeline?",
    "What embedding model is used and how many dimensions does it produce?",
    "What is ChromaDB and what does it store?",
    "How does the generation step work in RAG?",
    "What is a vector embedding?",
    "What happens if chunks are too large?",
    "Why is cosine similarity preferred over L2 distance for text embeddings?",
    "How does RAG reduce hallucination in LLM responses?",
    "What is the recommended chunk size and overlap mentioned in the document?",
]


def score_faithfulness(answer: str, contexts: list[str]) -> float:
    """What fraction of claims in the answer are supported by the retrieved context?"""
    context_str = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts))
    prompt = f"""You are an evaluator assessing whether an answer is grounded in the provided context.

Context:
{context_str}

Answer:
{answer}

Score the faithfulness from 0.0 to 1.0:
- 1.0 = every claim is directly supported by the context
- 0.5 = roughly half the claims are supported
- 0.0 = no claims are supported; the answer ignores the context

Reply with ONLY a decimal number, nothing else."""

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=8,
        messages=[{"role": "user", "content": prompt}],
    )
    return float(response.content[0].text.strip())


def score_answer_relevancy(question: str, answer: str) -> float:
    """How directly and completely does the answer address the question?"""
    prompt = f"""You are an evaluator assessing whether an answer is relevant to a question.

Question: {question}
Answer: {answer}

Score the relevancy from 0.0 to 1.0:
- 1.0 = the answer directly and completely addresses the question
- 0.5 = the answer partially addresses the question
- 0.0 = the answer is off-topic or irrelevant

Reply with ONLY a decimal number, nothing else."""

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=8,
        messages=[{"role": "user", "content": prompt}],
    )
    return float(response.content[0].text.strip())


def run_eval() -> None:
    results = []

    for i, question in enumerate(TEST_CASES):
        print(f"[{i+1}/{len(TEST_CASES)}] {question}")

        agent_result = run_agent(question)
        answer = agent_result["answer"]

        # Retrieve the same top-3 chunks the agent would have used
        chunks = retrieve(question, n_results=3)
        contexts = [c["text"] for c in chunks]

        faithfulness = score_faithfulness(answer, contexts)
        relevancy = score_answer_relevancy(question, answer)

        results.append({
            "question": question,
            "answer": answer,
            "plan": agent_result["plan"],
            "faithfulness": faithfulness,
            "answer_relevancy": relevancy,
        })

        print(f"  faithfulness={faithfulness:.2f}  relevancy={relevancy:.2f}")

    # Summary
    avg_f = sum(r["faithfulness"] for r in results) / len(results)
    avg_r = sum(r["answer_relevancy"] for r in results) / len(results)

    print(f"\n{'='*50}")
    print(f"Avg faithfulness:     {avg_f:.3f}")
    print(f"Avg answer relevancy: {avg_r:.3f}")
    print(f"{'='*50}")

    # Save full results for inspection
    out_path = Path(__file__).parent / "eval_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nFull results saved to {out_path}")


if __name__ == "__main__":
    run_eval()
