from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".docx"}

_CHROMA_PATH = str(Path(__file__).resolve().parent.parent / "chroma_db")


def load_file(path: str) -> str:
    p = Path(path)
    if p.suffix == ".txt":
        return p.read_text(errors="ignore")
    if p.suffix == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(str(p))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    if p.suffix == ".docx":
        from docx import Document
        doc = Document(str(p))
        return "\n".join(para.text for para in doc.paragraphs)
    raise ValueError(f"Unsupported file type: {p.suffix}")


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    """Slide a window over the text. Overlap keeps context at chunk boundaries."""
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start : start + chunk_size])
        start += chunk_size - overlap
    return [c for c in chunks if c.strip()]  # drop whitespace-only chunks


def already_ingested(collection, doc_path: str) -> bool:
    results = collection.get(where={"source": doc_path}, limit=1)
    return len(results["ids"]) > 0


def ingest(doc_path: str, collection_name: str = "documents") -> None:
    p = Path(doc_path)
    if p.suffix not in SUPPORTED_EXTENSIONS:
        print(f"Skipping unsupported file type: {p.name}")
        return

    client = chromadb.PersistentClient(path=_CHROMA_PATH)
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    if already_ingested(collection, str(p.resolve())):
        print(f"Already ingested, skipping: {p.name}")
        return

    text = load_file(doc_path)
    if not text.strip():
        print(f"No text extracted from {p.name}, skipping.")
        return

    chunks = chunk_text(text)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks).tolist()

    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[f"{p.stem}_{i}" for i in range(len(chunks))],
        metadatas=[{"source": str(p.resolve()), "filename": p.name, "chunk_index": i} for i in range(len(chunks))],
    )

    print(f"Ingested {len(chunks)} chunks from {p.name}")


if __name__ == "__main__":
    default_doc = Path(__file__).resolve().parent.parent / "data" / "sample.txt"
    ingest(str(default_doc))
