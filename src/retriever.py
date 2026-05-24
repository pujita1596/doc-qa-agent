from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

# Load once at import time — reloading on every request adds ~1s latency
_model = SentenceTransformer("all-MiniLM-L6-v2")
_CHROMA_PATH = str(Path(__file__).resolve().parent.parent / "chroma_db")


def retrieve(query: str, n_results: int = 3, collection_name: str = "documents") -> list[dict]:
    query_embedding = _model.encode([query]).tolist()

    client = chromadb.PersistentClient(path=_CHROMA_PATH)
    collection = client.get_collection(name=collection_name)

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results,
    )

    # cosine space: distance = 1 - similarity, so similarity = 1 - distance
    chunks = results["documents"][0]
    distances = results["distances"][0]
    metadatas = results["metadatas"][0]

    return [
        {"text": chunk, "score": round(1 - dist, 4), "metadata": meta}
        for chunk, dist, meta in zip(chunks, distances, metadatas)
    ]


if __name__ == "__main__":
    query = "How does chunking work and why does overlap matter?"
    print(f"Query: {query}\n")
    for i, r in enumerate(retrieve(query)):
        print(f"--- Chunk {i + 1} (similarity: {r['score']}) ---")
        print(r["text"])
        print()
