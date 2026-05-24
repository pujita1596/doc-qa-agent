FROM python:3.12-slim

WORKDIR /app

# Install dependencies first so this layer is cached on code-only changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY data/ data/

# Cache the embedding model inside the image so cold starts are fast
ENV SENTENCE_TRANSFORMERS_HOME=/app/.cache/sentence_transformers

# Pre-populate ChromaDB — bakes the vector store into the image
# No API key needed here; ingest only uses sentence-transformers + chromadb
RUN python src/ingest.py

EXPOSE 8000

# Railway injects $PORT; default to 8000 locally
CMD ["sh", "-c", "uvicorn main:app --app-dir src --host 0.0.0.0 --port ${PORT:-8000}"]
