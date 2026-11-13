"""Build and measure the three Day 06 ingestion configurations."""
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import platform
from time import perf_counter
from typing import Any

import chromadb
from dotenv import load_dotenv

from .chunking import by_paragraph, fixed_size
from .embeddings import ChromaLocalEmbedder
from .ingestion import load_corpus_units
from .models import Chunk, DocumentPage
from .store import ChromaVectorStore


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
EMBEDDING_BATCH_SIZE = 64

CONFIGURATIONS: dict[str, dict[str, Any]] = {
    "A": {
        "chunking": "fixed_size",
        "chunk_size": 500,
        "overlap": 50,
        "hypothesis": "Fixed windows provide consistent retrieval units but may split context.",
    },
    "B": {
        "chunking": "paragraph",
        "chunk_size": 550,
        "overlap": 0,
        "hypothesis": "Paragraph-aware chunks preserve local meaning within the model token limit.",
    },
    "C": {
        "chunking": "paragraph",
        "chunk_size": 550,
        "overlap": 0,
        "hypothesis": "The same chunks as B plus user-provided volume filters improve scoped retrieval.",
    },
}

EXPERIMENT_FIELDS = [
    "strategy",
    "chunking",
    "chunk_size",
    "overlap",
    "metadata",
    "filter",
    "vectorizer",
    "hypothesis",
    "dev_hit1",
    "dev_hit3",
    "test_hit1",
    "test_hit3",
    "notes",
]


def _configured_path(value: str | None, default: Path) -> Path:
    if value is None:
        return default.resolve()
    if not value.strip():
        raise ValueError("CHROMA_PERSIST_DIRECTORY must not be empty")
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _embed_in_batches(embedder: ChromaLocalEmbedder, chunks: list[Chunk]) -> list[list[float]]:
    vectors: list[list[float]] = []
    texts = [chunk.text for chunk in chunks]
    for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch = texts[start : start + EMBEDDING_BATCH_SIZE]
        batch_vectors = embedder.embed_documents(batch)
        if len(batch_vectors) != len(batch):
            raise RuntimeError(
                f"Embedder returned {len(batch_vectors)} vectors for {len(batch)} chunks"
            )
        vectors.extend(batch_vectors)
    if len(vectors) != len(chunks):
        raise RuntimeError("Embedding count does not match chunk count")
    return vectors


def _chunk_pages(pages: list[DocumentPage], strategy: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    for page in pages:
        if strategy == "fixed_size":
            chunks.extend(fixed_size(page, size=500, overlap=50))
        elif strategy == "paragraph":
            chunks.extend(by_paragraph(page, max_size=550))
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")
    if not chunks:
        raise ValueError(f"Chunking strategy {strategy!r} produced no chunks")
    return chunks


def _write_experiment_log(
    path: Path,
    results: dict[str, dict[str, Any]],
    model_name: str,
    run_id: str,
) -> None:
    existing: dict[str, dict[str, str]] = {}
    if path.exists():
        with path.open("r", encoding="utf-8-sig", newline="") as log_file:
            reader = csv.DictReader(log_file)
            for row in reader:
                strategy = (row.get("strategy") or "").strip()
                if strategy:
                    existing[strategy] = {field: row.get(field, "") or "" for field in EXPERIMENT_FIELDS}

    for strategy, result in results.items():
        config = CONFIGURATIONS[strategy]
        row = existing.setdefault(strategy, {field: "" for field in EXPERIMENT_FIELDS})
        row.update(
            {
                "strategy": strategy,
                "chunking": config["chunking"],
                "chunk_size": str(config["chunk_size"]),
                "overlap": str(config["overlap"]),
                "metadata": "document_id, source, logical_page, unit_id, volume",
                "filter": (
                    "At query time: filter by volume only when the question provides it"
                    if strategy == "C"
                    else "No filter"
                ),
                "vectorizer": f"{model_name}; 384 dimensions; cosine",
                "hypothesis": config["hypothesis"],
                "notes": (
                    f"Step 3 run {run_id}; chunks={result['chunk_count']}; "
                    f"chunk_characters={result['chunk_characters']}; "
                    f"chunk_tokens={result['chunk_tokens']}; "
                    f"index_seconds={result['index_seconds']:.3f}; "
                    f"reingest_count={result['reingest_count']}; local_cost_usd=0. "
                    f"{result['embedding_note']}"
                ),
            }
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="") as log_file:
        writer = csv.DictWriter(log_file, fieldnames=EXPERIMENT_FIELDS)
        writer.writeheader()
        for strategy in sorted(existing):
            writer.writerow(existing[strategy])
    temporary_path.replace(path)


def main() -> None:
    load_dotenv(ROOT / ".env")
    provider = os.getenv("EMBEDDING_PROVIDER", "chroma-local")
    if provider != "chroma-local":
        raise ValueError("This pipeline supports EMBEDDING_PROVIDER=chroma-local only")

    model_name = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    embedder = ChromaLocalEmbedder(model_name)
    persist_directory = _configured_path(
        os.getenv("CHROMA_PERSIST_DIRECTORY"),
        ROOT / "data" / "chroma",
    )
    corpus_path = ROOT / "data" / "corpus_units.jsonl"
    pages = load_corpus_units(corpus_path, minimum_units=30, minimum_characters=1500)
    active_document_ids = {page.document_id for page in pages}

    corpus_token_start = perf_counter()
    source_characters = sum(len(page.text) for page in pages)
    source_tokens = sum(embedder.count_tokens(page.text) for page in pages)
    corpus_token_seconds = perf_counter() - corpus_token_start

    store = ChromaVectorStore(persist_directory, embedder.dimensions)
    run_id = datetime.now().astimezone().isoformat(timespec="seconds")
    results: dict[str, dict[str, Any]] = {}
    strategy_chunks: dict[str, list[Chunk]] = {}
    strategy_vectors: dict[str, list[list[float]]] = {}
    strategy_token_counts: dict[str, list[int]] = {}

    for strategy in ("A", "B", "C"):
        config = CONFIGURATIONS[strategy]
        if strategy == "C":
            chunks = strategy_chunks["B"]
            vectors = strategy_vectors["B"]
            chunk_token_counts = strategy_token_counts["B"]
            chunking_seconds = 0.0
            token_count_seconds = 0.0
            embedding_seconds = 0.0
            embedding_note = "Chunks and embeddings reused from B; C changes query-time filtering only."
        else:
            chunking_start = perf_counter()
            chunks = _chunk_pages(pages, config["chunking"])
            chunking_seconds = perf_counter() - chunking_start

            token_count_start = perf_counter()
            chunk_token_counts = [embedder.count_tokens(chunk.text) for chunk in chunks]
            token_count_seconds = perf_counter() - token_count_start
            oversized = [
                (chunk.chunk_id, token_count)
                for chunk, token_count in zip(chunks, chunk_token_counts)
                if token_count + 2 > embedder.max_input_tokens
            ]
            if oversized:
                example_id, example_count = oversized[0]
                raise ValueError(
                    f"{len(oversized)} chunks exceed the {embedder.max_input_tokens}-token "
                    f"embedding limit; first is {example_id} with {example_count} content tokens. "
                    "Reduce the chunk size before embedding to avoid silent truncation."
                )

            embedding_start = perf_counter()
            vectors = _embed_in_batches(embedder, chunks)
            embedding_seconds = perf_counter() - embedding_start
            strategy_chunks[strategy] = chunks
            strategy_vectors[strategy] = vectors
            strategy_token_counts[strategy] = chunk_token_counts
            embedding_note = f"Embedded locally in batches of {EMBEDDING_BATCH_SIZE}."

        collection_count_before = store.count(strategy)
        storage_start = perf_counter()
        collection_count_after = store.synchronize(
            strategy,
            chunks,
            vectors,
            active_document_ids,
        )
        storage_seconds = perf_counter() - storage_start
        if collection_count_after != len(chunks):
            raise RuntimeError(
                f"Strategy {strategy} stored {collection_count_after} chunks, "
                f"expected {len(chunks)}"
            )

        reingest_start = perf_counter()
        reingest_count = store.synchronize(
            strategy,
            chunks,
            vectors,
            active_document_ids,
        )
        reingest_seconds = perf_counter() - reingest_start
        if reingest_count != collection_count_after:
            raise RuntimeError(
                f"Strategy {strategy} created duplicate or missing chunks on re-ingestion: "
                f"{collection_count_after} -> {reingest_count}"
            )

        chunk_lengths = [len(chunk.text) for chunk in chunks]
        chunk_tokens = sum(chunk_token_counts)
        index_seconds = (
            chunking_seconds
            + token_count_seconds
            + embedding_seconds
            + storage_seconds
        )
        results[strategy] = {
            "configuration": config,
            "chunk_count": len(chunks),
            "chunk_characters": sum(chunk_lengths),
            "chunk_tokens": chunk_tokens,
            "chunk_token_method": "Chroma ONNX tokenizer, content tokens without special tokens",
            "chunk_characters_min": min(chunk_lengths),
            "chunk_characters_max": max(chunk_lengths),
            "chunk_characters_mean": sum(chunk_lengths) / len(chunk_lengths),
            "chunk_tokens_min": min(chunk_token_counts),
            "chunk_tokens_max": max(chunk_token_counts),
            "chunk_tokens_mean": chunk_tokens / len(chunk_token_counts),
            "chunks_over_model_limit": 0,
            "chunking_seconds": chunking_seconds,
            "token_count_seconds": token_count_seconds,
            "embedding_seconds": embedding_seconds,
            "storage_seconds": storage_seconds,
            "reingest_seconds": reingest_seconds,
            "index_seconds": index_seconds,
            "total_with_reingest_seconds": index_seconds + reingest_seconds,
            "collection_count_before": collection_count_before,
            "collection_count_after": collection_count_after,
            "reingest_count": reingest_count,
            "embedding_note": embedding_note,
            "local_cost_usd": 0.0,
        }

    summary: dict[str, Any] = {
        "run_id": run_id,
        "python_version": platform.python_version(),
        "chromadb_version": chromadb.__version__,
        "embedding_provider": provider,
        "embedding_model": embedder.model_name,
        "embedding_dimensions": embedder.dimensions,
        "embedding_max_input_tokens": embedder.max_input_tokens,
        "vector_store": "Chroma PersistentClient",
        "vector_store_path": str(persist_directory),
        "distance_metric": "cosine",
        "search_type": "Chroma HNSW approximate nearest-neighbor",
        "corpus_path": str(corpus_path.relative_to(ROOT)),
        "corpus_units": len(pages),
        "source_characters": source_characters,
        "source_tokens": source_tokens,
        "source_token_method": "Chroma ONNX tokenizer, content tokens without special tokens",
        "source_token_count_seconds": corpus_token_seconds,
        "active_document_ids": sorted(active_document_ids),
        "local_embedding_cost_usd": 0.0,
        "strategies": results,
    }
    results_directory = ROOT / "results"
    results_directory.mkdir(parents=True, exist_ok=True)
    stats_path = results_directory / "step3_ingestion_stats.json"
    temporary_stats_path = stats_path.with_suffix(stats_path.suffix + ".tmp")
    temporary_stats_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_stats_path.replace(stats_path)
    _write_experiment_log(
        REPO_ROOT / "artifacts" / "experiment_log.csv",
        results,
        embedder.model_name,
        run_id,
    )

    print(f"Corpus: {len(pages)} units, {source_characters:,} characters, {source_tokens:,} tokens")
    print(f"Embedding: {embedder.model_name}, {embedder.dimensions} dimensions")
    print(f"Chroma: {persist_directory} (cosine)")
    for strategy, result in results.items():
        print(
            f"{strategy}: {result['chunk_count']:,} chunks, "
            f"{result['chunk_characters']:,} chars, {result['chunk_tokens']:,} tokens, "
            f"index={result['index_seconds']:.2f}s, "
            f"reingest count={result['reingest_count']:,}"
        )
    print(f"Stats: {stats_path}")
    print(f"Experiment log: {REPO_ROOT / 'artifacts' / 'experiment_log.csv'}")
    print("Step 3 ingestion complete. Local embedding cost: $0.00.")


if __name__ == "__main__":
    main()
