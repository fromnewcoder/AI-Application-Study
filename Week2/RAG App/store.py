"""
store.py — ChromaDB persistent vector store.

Wraps a single ChromaDB collection with convenience methods for:
  • Adding chunks (with duplicate detection via doc_id)
  • Querying top-k similar chunks
  • Listing ingested documents
  • Deleting a document by source path / URL
  • Clearing the entire collection
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import NamedTuple

import chromadb
from chromadb.config import Settings

from embeddings import LocalEmbeddingFunction
from ingestion import Chunk

# ── Default paths ──────────────────────────────────────────────────────────────

_HERE = Path(__file__).parent
DEFAULT_DB_DIR = _HERE / "db"
COLLECTION_NAME = "rag_documents"


# ── Result type ───────────────────────────────────────────────────────────────

class SearchResult(NamedTuple):
    text: str
    source: str
    source_type: str
    page: int | None
    chunk_index: int
    distance: float


# ── Store ─────────────────────────────────────────────────────────────────────

class VectorStore:
    """
    Persistent ChromaDB-backed document store.

    Parameters
    ----------
    db_dir:
        Directory where ChromaDB persists data.  Created if absent.
    collection_name:
        Name of the ChromaDB collection to use.
    """

    def __init__(
        self,
        db_dir: str | Path = DEFAULT_DB_DIR,
        collection_name: str = COLLECTION_NAME,
    ) -> None:
        db_dir = Path(db_dir)
        db_dir.mkdir(parents=True, exist_ok=True)

        self._ef = LocalEmbeddingFunction()
        self._client = chromadb.PersistentClient(
            path=str(db_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        self._col = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=self._ef,
            metadata={"hnsw:space": "cosine"},
        )

    # ── Ingestion ──────────────────────────────────────────────────────────────

    def add_chunks(self, chunks: list[Chunk], skip_existing: bool = True) -> int:
        """
        Add a list of Chunks to the collection.

        Returns the number of chunks actually written (0 if already present
        and skip_existing=True).
        """
        if not chunks:
            return 0

        doc_id = chunks[0].doc_id

        if skip_existing and self._doc_exists(doc_id):
            return 0

        # Delete stale version if re-ingesting
        if not skip_existing and self._doc_exists(doc_id):
            self._delete_by_doc_id(doc_id)

        self._col.add(
            ids=[c.chroma_id for c in chunks],
            documents=[c.text for c in chunks],
            metadatas=[c.chroma_metadata for c in chunks],
        )
        return len(chunks)

    def _doc_exists(self, doc_id: str) -> bool:
        result = self._col.get(where={"doc_id": {"$eq": doc_id}}, limit=1)
        return len(result["ids"]) > 0

    def _delete_by_doc_id(self, doc_id: str) -> None:
        self._col.delete(where={"doc_id": {"$eq": doc_id}})

    # ── Querying ───────────────────────────────────────────────────────────────

    def query(
        self,
        question: str,
        n_results: int = 5,
        source_filter: str | None = None,
    ) -> list[SearchResult]:
        """
        Retrieve the top-k most relevant chunks for a question.

        Parameters
        ----------
        question:   Natural-language query.
        n_results:  Number of chunks to return.
        source_filter: If set, restrict results to this source path/URL.
        """
        total = self._col.count()
        if total == 0:
            return []

        n_results = min(n_results, total)

        where = None
        if source_filter:
            where = {"source": {"$eq": source_filter}}

        results = self._col.query(
            query_texts=[question],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        out: list[SearchResult] = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            out.append(SearchResult(
                text=doc,
                source=meta.get("source", "unknown"),
                source_type=meta.get("source_type", "unknown"),
                page=meta.get("page"),
                chunk_index=meta.get("chunk_index", 0),
                distance=round(float(dist), 4),
            ))

        return out

    # ── Listing / info ─────────────────────────────────────────────────────────

    def list_documents(self) -> list[dict]:
        """
        Return one record per unique source document.

        Each record contains: source, source_type, chunk_count, doc_id.
        """
        if self._col.count() == 0:
            return []

        all_meta = self._col.get(include=["metadatas"])["metadatas"]
        seen: dict[str, dict] = {}
        for m in all_meta:
            src = m.get("source", "unknown")
            if src not in seen:
                seen[src] = {
                    "source": src,
                    "source_type": m.get("source_type", "unknown"),
                    "chunk_count": 0,
                    "doc_id": m.get("doc_id", ""),
                }
            seen[src]["chunk_count"] += 1

        return sorted(seen.values(), key=lambda x: x["source"])

    def total_chunks(self) -> int:
        return self._col.count()

    # ── Deletion ───────────────────────────────────────────────────────────────

    def delete_document(self, source: str) -> int:
        """
        Remove all chunks whose source matches the given path or URL.

        Returns the number of chunks deleted.
        """
        result = self._col.get(
            where={"source": {"$eq": source}},
            include=["metadatas"],
        )
        ids = result["ids"]
        if ids:
            self._col.delete(ids=ids)
        return len(ids)

    def clear(self) -> int:
        """Delete every chunk in the collection. Returns count removed."""
        count = self._col.count()
        if count > 0:
            all_ids = self._col.get()["ids"]
            self._col.delete(ids=all_ids)
        return count

    # ── Repr ───────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"VectorStore(chunks={self.total_chunks()}, "
            f"backend={self._ef.backend!r})"
        )
