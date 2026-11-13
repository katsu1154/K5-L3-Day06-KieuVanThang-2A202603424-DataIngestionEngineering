"""Persistent local vector storage using Chroma."""
from pathlib import Path
import re
from typing import Any

import chromadb

from .models import Chunk


class ChromaVectorStore:
    def __init__(self, persist_directory: str | Path, dimensions: int) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be a positive integer")
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.dimensions = dimensions
        self.client = chromadb.PersistentClient(path=str(self.persist_directory))
        self._collections: dict[str, Any] = {}

    def _collection(self, strategy: str) -> Any:
        if not re.fullmatch(r"[A-Za-z0-9._-]{1,63}", strategy):
            raise ValueError("strategy must contain only letters, digits, '.', '_' or '-'")
        if strategy not in self._collections:
            collection = self.client.get_or_create_collection(
                name=f"day6_{strategy}",
                metadata={"hnsw:space": "cosine"},
                embedding_function=None,
            )
            if collection.metadata.get("hnsw:space") != "cosine":
                raise ValueError(f"Collection for strategy {strategy!r} is not using cosine distance")
            self._collections[strategy] = collection
        return self._collections[strategy]

    def upsert(self, strategy: str, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        if not chunks:
            return

        ids = [chunk.chunk_id for chunk in chunks]
        if len(ids) != len(set(ids)):
            raise ValueError("chunk IDs must be unique within an upsert")
        for vector in vectors:
            if len(vector) != self.dimensions:
                raise ValueError(
                    f"Expected vectors with {self.dimensions} dimensions; got {len(vector)}"
                )

        collection = self._collection(strategy)
        document_ids = {chunk.document_id for chunk in chunks}
        old_ids: set[str] = set()
        for document_id in document_ids:
            old_ids.update(
                collection.get(where={"document_id": document_id}, include=["metadatas"])["ids"]
            )

        metadatas: list[dict[str, str | int | float | bool]] = []
        for chunk in chunks:
            metadata: dict[str, Any] = {
                **chunk.metadata,
                "document_id": chunk.document_id,
                "source": chunk.source,
                "page": chunk.page,
            }
            if any(not isinstance(value, (str, int, float, bool)) for value in metadata.values()):
                raise ValueError("Chroma metadata values must be strings, numbers, or booleans")
            metadatas.append(metadata)

        collection.upsert(
            ids=ids,
            embeddings=vectors,
            documents=[chunk.text for chunk in chunks],
            metadatas=metadatas,
        )

        obsolete_ids = old_ids - set(ids)
        if obsolete_ids:
            collection.delete(ids=sorted(obsolete_ids))

    def synchronize(
        self,
        strategy: str,
        chunks: list[Chunk],
        vectors: list[list[float]],
        active_document_ids: set[str],
    ) -> int:
        chunk_document_ids = {chunk.document_id for chunk in chunks}
        if chunk_document_ids != active_document_ids:
            raise ValueError(
                "active_document_ids must exactly match the documents represented by chunks"
            )

        self.upsert(strategy, chunks, vectors)
        collection = self._collection(strategy)
        existing = collection.get(include=["metadatas"])
        obsolete_ids = [
            chunk_id
            for chunk_id, metadata in zip(existing["ids"], existing["metadatas"])
            if metadata is None or metadata.get("document_id") not in active_document_ids
        ]
        if obsolete_ids:
            collection.delete(ids=obsolete_ids)
        return collection.count()

    def count(self, strategy: str) -> int:
        return self._collection(strategy).count()

    def get_chunks(
        self,
        strategy: str,
        filters: dict[str, Any],
    ) -> list[dict[str, Any]]:
        collection = self._collection(strategy)
        conditions = [{key: value} for key, value in filters.items()]
        where: dict[str, Any] | None = None
        if conditions:
            where = conditions[0] if len(conditions) == 1 else {"$and": conditions}
        result = collection.get(
            where=where,
            include=["documents", "metadatas"],
        )
        return [
            {
                "chunk_id": chunk_id,
                "text": text,
                "metadata": metadata or {},
            }
            for chunk_id, text, metadata in zip(
                result["ids"], result["documents"], result["metadatas"]
            )
        ]

    def search(
        self,
        strategy: str,
        query_vector: list[float],
        top_k: int = 3,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        if len(query_vector) != self.dimensions:
            raise ValueError(
                f"Expected a query vector with {self.dimensions} dimensions; "
                f"got {len(query_vector)}"
            )

        collection = self._collection(strategy)
        count = collection.count()
        if count == 0:
            return []

        where: dict[str, Any] | None = None
        if filters:
            conditions = [{key: value} for key, value in filters.items()]
            where = conditions[0] if len(conditions) == 1 else {"$and": conditions}

        result = collection.query(
            query_embeddings=[query_vector],
            n_results=min(top_k, count),
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        hits: list[dict[str, Any]] = []
        for chunk_id, text, metadata, distance in zip(
            result["ids"][0],
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
        ):
            hits.append(
                {
                    "chunk_id": chunk_id,
                    "text": text,
                    "score": 1.0 - distance,
                    "metadata": metadata,
                }
            )
        return hits
