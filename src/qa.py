from dotenv import load_dotenv
import anthropic
from retriever import retrieve

load_dotenv()

client = anthropic.Anthropic()


def ask(question: str, n_chunks: int = 3) -> dict:
    chunks = retrieve(question, n_results=n_chunks)

    # Number each chunk so the model can cite [1], [2], etc.
    context_block = "\n\n".join(
        f"[{i + 1}] {c['text']}" for i, c in enumerate(chunks)
    )

    prompt = f"""Answer the question using ONLY the context provided below.
Cite the source number (e.g. [1], [2]) for each claim you make.
If the answer is not in the context, say "I don't have enough information to answer that."

Context:
{context_block}

Question: {question}"""

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )

    return {
        "answer": response.content[0].text,
        "sources": [
            {"chunk_index": i + 1, "score": c["score"], "text": c["text"], "metadata": c["metadata"]}
            for i, c in enumerate(chunks)
        ],
    }


if __name__ == "__main__":
    questions = [
        "Why does chunk overlap matter?",
        "What is cosine similarity used for?",
        "What happens in the generation step of RAG?",
    ]

    for q in questions:
        print(f"Q: {q}")
        result = ask(q)
        print(f"A: {result['answer']}")
        print(f"Sources used: {[s['chunk_index'] for s in result['sources']]}")
        print()
