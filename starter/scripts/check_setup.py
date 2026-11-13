"""Verify the local embedding model and persistent Chroma store."""
import os
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv

from src.embeddings import ChromaLocalEmbedder
from src.models import Chunk
from src.store import ChromaVectorStore


ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

provider = os.getenv("EMBEDDING_PROVIDER", "chroma-local")
if provider != "chroma-local":
    raise ValueError("This setup check supports EMBEDDING_PROVIDER=chroma-local only")
model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
persist_directory_value = os.getenv(
    "CHROMA_PERSIST_DIRECTORY", str(ROOT / "data" / "chroma")
)
if not persist_directory_value.strip():
    raise ValueError("CHROMA_PERSIST_DIRECTORY must not be empty")
persist_directory = Path(persist_directory_value)
if not persist_directory.is_absolute():
    persist_directory = (ROOT / persist_directory).resolve()

embedder = ChromaLocalEmbedder(model_name)
text = "A local setup check for persistent vector storage."
vector = embedder.embed_query(text)
store = ChromaVectorStore(persist_directory, embedder.dimensions)
strategy = "_setup_check"
chunk_id = str(uuid4())

try:
    store.upsert(
        strategy,
        [
            Chunk(
                chunk_id=chunk_id,
                document_id="setup-check",
                source="setup-check",
                page=1,
                text=text,
                metadata={"setup_check": True},
            )
        ],
        [vector],
    )
    hits = store.search(strategy, vector, filters={"setup_check": True})
    if not hits or hits[0]["chunk_id"] != chunk_id:
        raise RuntimeError("Chroma setup check could not retrieve its test vector")
finally:
    store.client.delete_collection(name=f"day6_{strategy}")

print(f"Embedding model: {embedder.model_name} ({embedder.dimensions} dimensions)")
print(f"Persistent Chroma path: {persist_directory}")
print("Setup OK. No Docker or API key is required for the local configuration.")
