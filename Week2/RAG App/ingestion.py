"""
ingestion.py — Document loaders and text chunker.

Supported sources:
  • PDF        (.pdf)   via PyMuPDF
  • Word       (.docx)  via python-docx
  • Plain text (.txt, .md, .rst, …) via built-in open()
  • Web URL    (http/https) via requests + BeautifulSoup

Chunking strategy:
  • Split on paragraph boundaries first (preserves natural units).
  • If a paragraph exceeds max_tokens, fall back to a sliding window with
    configurable overlap so no information is lost at chunk edges.
  • Each chunk carries metadata: source path/url, page number (PDF), chunk
    index, and a short preview for display purposes.
"""

from __future__ import annotations

import re
import textwrap
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

# ── Data model ───────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    text: str
    doc_id: str                    # stable identifier for the parent document
    chunk_index: int
    source: str                    # file path or URL
    source_type: str               # pdf | docx | text | url
    page: int | None = None        # PDF page number (1-indexed)
    metadata: dict = field(default_factory=dict)

    @property
    def chroma_id(self) -> str:
        """Unique ID suitable for ChromaDB (deterministic per chunk)."""
        return f"{self.doc_id}__chunk_{self.chunk_index:04d}"

    @property
    def chroma_metadata(self) -> dict:
        meta = {
            "source": self.source,
            "source_type": self.source_type,
            "chunk_index": self.chunk_index,
            "doc_id": self.doc_id,
        }
        if self.page is not None:
            meta["page"] = self.page
        meta.update(self.metadata)
        return meta

    def preview(self, width: int = 80) -> str:
        clean = " ".join(self.text.split())
        return textwrap.shorten(clean, width=width, placeholder="…")


# ── Chunking helpers ──────────────────────────────────────────────────────────

def _word_count(text: str) -> int:
    return len(text.split())


def _sliding_window(text: str, max_words: int, overlap: int) -> Iterator[str]:
    """Yield overlapping windows of approximately max_words words."""
    words = text.split()
    step = max(1, max_words - overlap)
    for start in range(0, len(words), step):
        chunk = " ".join(words[start : start + max_words])
        if chunk.strip():
            yield chunk
        if start + max_words >= len(words):
            break


def _split_into_chunks(
    text: str,
    max_words: int = 200,
    overlap: int = 30,
    min_words: int = 10,
) -> list[str]:
    """
    1. Split on blank lines (paragraphs / sections).
    2. Merge short paragraphs with their neighbour.
    3. Slide over paragraphs that are still too long.
    """
    # Step 1 — paragraph split
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]

    # Step 2 — merge tiny fragments with the previous paragraph
    merged: list[str] = []
    for para in paragraphs:
        if merged and _word_count(para) < min_words:
            merged[-1] += " " + para
        else:
            merged.append(para)

    # Step 3 — slide over any paragraph that is still too large
    chunks: list[str] = []
    for para in merged:
        if _word_count(para) <= max_words:
            chunks.append(para)
        else:
            chunks.extend(_sliding_window(para, max_words, overlap))

    return [c for c in chunks if _word_count(c) >= min_words]


# ── Loaders ───────────────────────────────────────────────────────────────────

def _make_doc_id(source: str) -> str:
    import hashlib
    return hashlib.md5(source.encode()).hexdigest()[:12]


def load_pdf(path: str | Path, **chunk_kwargs) -> list[Chunk]:
    """Extract text page-by-page from a PDF and chunk each page."""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise RuntimeError("PyMuPDF is required for PDF support: pip install pymupdf")

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    doc_id = _make_doc_id(str(path))
    chunks: list[Chunk] = []
    chunk_index = 0

    with fitz.open(str(path)) as pdf:
        for page_num, page in enumerate(pdf, start=1):
            raw = page.get_text("text")
            if not raw.strip():
                continue  # skip blank/image-only pages
            for text in _split_into_chunks(raw, **chunk_kwargs):
                chunks.append(Chunk(
                    text=text,
                    doc_id=doc_id,
                    chunk_index=chunk_index,
                    source=str(path),
                    source_type="pdf",
                    page=page_num,
                ))
                chunk_index += 1

    if not chunks:
        raise ValueError(f"No text could be extracted from '{path.name}'. "
                         "The PDF may be scanned/image-based.")
    return chunks


def load_docx(path: str | Path, **chunk_kwargs) -> list[Chunk]:
    """Extract text from a Word document and chunk it."""
    try:
        from docx import Document
    except ImportError:
        raise RuntimeError("python-docx is required: pip install python-docx")

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"DOCX not found: {path}")

    doc = Document(str(path))
    # Concatenate paragraphs; insert double newlines to preserve structure
    full_text = "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())

    doc_id = _make_doc_id(str(path))
    return [
        Chunk(
            text=text,
            doc_id=doc_id,
            chunk_index=i,
            source=str(path),
            source_type="docx",
        )
        for i, text in enumerate(_split_into_chunks(full_text, **chunk_kwargs))
    ]


def load_text(path: str | Path, **chunk_kwargs) -> list[Chunk]:
    """Load a plain-text or Markdown file and chunk it."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    raw = path.read_text(encoding="utf-8", errors="replace")
    # Strip Markdown headers/emphasis for cleaner embeddings (keep the text)
    raw = re.sub(r"^#{1,6}\s+", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"[*_`]{1,3}(.+?)[*_`]{1,3}", r"\1", raw)

    doc_id = _make_doc_id(str(path))
    return [
        Chunk(
            text=text,
            doc_id=doc_id,
            chunk_index=i,
            source=str(path),
            source_type="text",
        )
        for i, text in enumerate(_split_into_chunks(raw, **chunk_kwargs))
    ]


def load_url(url: str, **chunk_kwargs) -> list[Chunk]:
    """Fetch a web page and extract readable text from it."""
    try:
        import requests
        from bs4 import BeautifulSoup
    except ImportError:
        raise RuntimeError("requests and beautifulsoup4 are required: "
                           "pip install requests beautifulsoup4")

    try:
        resp = requests.get(url, timeout=15, headers={
            "User-Agent": "RAGApp/1.0 (https://github.com/rag-app; contact@example.com) Python/3.11"
        })
        resp.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(f"Failed to fetch URL '{url}': {e}")

    soup = BeautifulSoup(resp.text, "html.parser")

    # Remove noise elements
    for tag in soup(["script", "style", "nav", "footer", "header",
                     "aside", "form", "button", "iframe"]):
        tag.decompose()

    # Prefer <article> or <main> if available; else fall back to <body>
    container = (soup.find("article") or soup.find("main") or soup.body or soup)
    raw = container.get_text(separator="\n")
    # Collapse excessive whitespace
    raw = re.sub(r"\n{3,}", "\n\n", raw).strip()

    if len(raw.split()) < 30:
        raise ValueError(f"Very little text extracted from '{url}'. "
                         "The page may require JavaScript or authentication.")

    doc_id = _make_doc_id(url)
    return [
        Chunk(
            text=text,
            doc_id=doc_id,
            chunk_index=i,
            source=url,
            source_type="url",
            metadata={"title": soup.title.string.strip() if soup.title else url},
        )
        for i, text in enumerate(_split_into_chunks(raw, **chunk_kwargs))
    ]


# ── Unified entry point ───────────────────────────────────────────────────────

_EXT_MAP = {
    ".pdf":  load_pdf,
    ".docx": load_docx,
    ".doc":  load_docx,
    ".txt":  load_text,
    ".md":   load_text,
    ".rst":  load_text,
    ".text": load_text,
}


def ingest(source: str, **chunk_kwargs) -> list[Chunk]:
    """
    Detect source type and return a list of Chunks.

    Args:
        source: A file path or an http(s) URL.
        **chunk_kwargs: Passed to _split_into_chunks
                        (max_words, overlap, min_words).

    Returns:
        list[Chunk]

    Raises:
        ValueError: Unsupported file extension.
        FileNotFoundError / RuntimeError: I/O problems.
    """
    if source.startswith("http://") or source.startswith("https://"):
        return load_url(source, **chunk_kwargs)

    path = Path(source)
    ext = path.suffix.lower()
    loader = _EXT_MAP.get(ext)
    if loader is None:
        supported = ", ".join(sorted(_EXT_MAP.keys()))
        raise ValueError(
            f"Unsupported file type '{ext}'. Supported: {supported}, or an http(s) URL."
        )
    return loader(path, **chunk_kwargs)
