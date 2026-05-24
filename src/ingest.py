from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer


def load_text(path: str) -> str:
    return Path(path).read_text()


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    """Slide a window over the text. Overlap keeps context at chunk boundaries."""
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start : start + chunk_size])
        start += chunk_size - overlap
    return chunks


def ingest(doc_path: str, collection_name: str = "documents") -> None:
    text = load_text(doc_path)
    chunks = chunk_text(text)

    # Embed each chunk — same model must be used at retrieval time
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks).tolist()

    # PersistentClient writes to disk so embeddings survive between runs
    chroma_path = str(Path(__file__).resolve().parent.parent / "chroma_db")
    client = chromadb.PersistentClient(path=chroma_path)
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},  # use cosine similarity, not L2
    )

    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[f"chunk_{i}" for i in range(len(chunks))],
        metadatas=[{"source": doc_path, "chunk_index": i} for i in range(len(chunks))],
    )

    print(f"Ingested {len(chunks)} chunks from {doc_path}")


if __name__ == "__main__":
    default_doc = Path(__file__).resolve().parent.parent / "data" / "sample.txt"
    ingest(str(default_doc))
