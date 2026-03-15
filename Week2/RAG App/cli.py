"""
cli.py — Polished command-line interface for the RAG app.

Commands
--------
  ingest   Add a document or URL to the knowledge base
  ask      Ask a question against ingested documents
  list     Show all ingested documents
  delete   Remove a document from the knowledge base
  clear    Wipe the entire knowledge base

Usage examples
--------------
  python cli.py ingest docs/report.pdf
  python cli.py ingest notes.md --max-words 150 --overlap 25
  python cli.py ingest https://en.wikipedia.org/wiki/Retrieval-augmented_generation
  python cli.py ask "What are the main findings?"
  python cli.py ask "Summarise section 3" --top-k 8
  python cli.py ask "What does the report say about costs?" --source docs/report.pdf
  python cli.py list
  python cli.py delete docs/report.pdf
  python cli.py clear
  python cli.py info
"""

from __future__ import annotations

# Fix Windows encoding issues
import sys
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import os
import sys
import time

import click
from rich import box
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

# ── Console setup ──────────────────────────────────────────────────────────────

THEME = Theme({
    "info":    "cyan",
    "success": "bold green",
    "warning": "bold yellow",
    "error":   "bold red",
    "muted":   "dim white",
    "heading": "bold white",
    "source":  "steel_blue1",
    "answer":  "white",
})

console = Console(theme=THEME, highlight=False, force_terminal=True)

# ── Lazy imports (keep startup fast) ──────────────────────────────────────────

def _get_store():
    from store import VectorStore
    return VectorStore()

def _get_pipeline(store):
    from pipeline import RAGPipeline
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        console.print(
            "[error]ANTHROPIC_API_KEY environment variable is not set.[/error]\n"
            "Export it before running:\n"
            "  [bold]export ANTHROPIC_API_KEY=sk-ant-...[/bold]"
        )
        sys.exit(1)
    return RAGPipeline(store=store, api_key=api_key)


# ── Shared banner ──────────────────────────────────────────────────────────────

def _banner():
    console.print()
    console.print(Panel.fit(
        "[bold white]RAG Document Q&A[/bold white]  [muted]powered by Claude[/muted]",
        border_style="dim white",
        padding=(0, 2),
    ))
    console.print()


# ── CLI group ──────────────────────────────────────────────────────────────────

@click.group()
def cli():
    """RAG Document Q&A — ingest documents, then ask questions."""
    pass


# ── ingest ─────────────────────────────────────────────────────────────────────

@cli.command()
@click.argument("source")
@click.option("--max-words",  default=200,  show_default=True,
              help="Max words per chunk.")
@click.option("--overlap",    default=30,   show_default=True,
              help="Word overlap between adjacent chunks.")
@click.option("--min-words",  default=10,   show_default=True,
              help="Discard chunks shorter than this.")
@click.option("--force",      is_flag=True, default=False,
              help="Re-ingest even if source already exists.")
def ingest(source: str, max_words: int, overlap: int, min_words: int, force: bool):
    """Ingest a document or URL into the knowledge base.

    SOURCE can be a file path (.pdf, .docx, .txt, .md) or an http(s) URL.
    """
    _banner()

    chunk_kwargs = dict(max_words=max_words, overlap=overlap, min_words=min_words)
    skip_existing = not force

    store = _get_store()

    # ── Load & chunk ──────────────────────────────────────────────────────────
    with Progress(
        SpinnerColumn(),
        TextColumn("[info]{task.description}[/info]"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Loading and chunking document…", total=None)
        try:
            from ingestion import ingest as _ingest
            chunks = _ingest(source, **chunk_kwargs)
        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            console.print(f"[error]X  {exc}[/error]")
            sys.exit(1)

        progress.update(task, description=f"Embedding {len(chunks)} chunks…")
        added = store.add_chunks(chunks, skip_existing=skip_existing)

    # ── Feedback ──────────────────────────────────────────────────────────────
    short_src = os.path.basename(source) if not source.startswith("http") else source

    if added == 0:
        console.print(
            f"[warning]⚠  '{short_src}' is already in the knowledge base.[/warning]\n"
            "Use [bold]--force[/bold] to re-ingest it."
        )
    else:
        console.print(
            f"[success]OK  Ingested '{short_src}'[/success]  "
            f"[muted]({added} chunks, {max_words}-word window, {overlap}-word overlap)[/muted]"
        )
        console.print(
            f"[muted]Total knowledge base: {store.total_chunks()} chunks[/muted]"
        )
    console.print()


# ── ask ────────────────────────────────────────────────────────────────────────

@cli.command()
@click.argument("question")
@click.option("--top-k",   default=5, show_default=True,
              help="Number of chunks to retrieve.")
@click.option("--source",  default=None,
              help="Restrict retrieval to this source path/URL.")
@click.option("--show-chunks", is_flag=True, default=False,
              help="Print the retrieved context chunks.")
def ask(question: str, top_k: int, source: str | None, show_chunks: bool):
    """Ask a question against the ingested documents."""
    _banner()

    store    = _get_store()
    pipeline = _get_pipeline(store)

    # ── Retrieve & generate ────────────────────────────────────────────────────
    with Progress(
        SpinnerColumn(),
        TextColumn("[info]{task.description}[/info]"),
        console=console,
        transient=True,
    ) as progress:
        t = progress.add_task("Searching knowledge base…", total=None)
        start = time.perf_counter()
        response = pipeline.ask(question, top_k=top_k, source_filter=source)
        elapsed = time.perf_counter() - start

    # ── Question echo ──────────────────────────────────────────────────────────
    console.print(Panel(
        Text(question, style="bold white"),
        title="[muted]Question[/muted]",
        border_style="dim white",
        padding=(0, 1),
    ))
    console.print()

    # ── Edge-case responses ────────────────────────────────────────────────────
    if response.empty_store or response.no_results:
        console.print(Panel(
            response.answer,
            title="[warning]Notice[/warning]",
            border_style="yellow",
            padding=(0, 1),
        ))
        console.print()
        return

    # ── Answer ─────────────────────────────────────────────────────────────────
    console.print(Panel(
        Markdown(response.answer),
        title="[success]Answer[/success]",
        border_style="green",
        padding=(1, 2),
    ))
    console.print()

    # ── Sources table ──────────────────────────────────────────────────────────
    table = Table(
        title="Retrieved Sources",
        box=box.SIMPLE_HEAD,
        header_style="bold white",
        border_style="dim white",
        show_lines=False,
        expand=False,
    )
    table.add_column("#",       style="muted",   width=3,  no_wrap=True)
    table.add_column("Source",  style="source",  max_width=45)
    table.add_column("Type",    style="info",    width=6)
    table.add_column("Page",    style="muted",   width=5)
    table.add_column("Chunk",   style="muted",   width=5)
    table.add_column("Score",   style="muted",   width=7)
    table.add_column("Preview", style="muted",   max_width=40)

    for i, s in enumerate(response.sources, 1):
        short = os.path.basename(s.source) if not s.source.startswith("http") else s.source
        table.add_row(
            str(i),
            short,
            s.source_type,
            str(s.page) if s.page else "—",
            str(s.chunk_index),
            f"{1 - s.distance:.2f}",      # cosine similarity (higher = better)
            s.text[:60].replace("\n", " ") + ("…" if len(s.text) > 60 else ""),
        )

    console.print(table)

    # ── Optional: show raw context chunks ─────────────────────────────────────
    if show_chunks:
        console.print()
        console.rule("[muted]Retrieved chunks[/muted]")
        for i, s in enumerate(response.sources, 1):
            short = os.path.basename(s.source) if not s.source.startswith("http") else s.source
            console.print(Panel(
                s.text,
                title=f"[muted]Chunk {i} · {short}[/muted]",
                border_style="dim white",
                padding=(0, 1),
            ))
            console.print()

    # ── Footer ─────────────────────────────────────────────────────────────────
    tokens = response.tokens_used
    console.print(
        f"[muted]↳  {response.context_chunks_used} chunks used · "
        f"{tokens.get('input',0):,} input tokens · "
        f"{tokens.get('output',0):,} output tokens · "
        f"{elapsed:.1f}s[/muted]"
    )
    console.print()


# ── list ────────────────────────────────────────────────────────────────────────

@cli.command(name="list")
def list_docs():
    """List all documents in the knowledge base."""
    _banner()
    store = _get_store()
    docs  = store.list_documents()

    if not docs:
        console.print("[warning]No documents have been ingested yet.[/warning]")
        console.print("[muted]Run:  python cli.py ingest <file_or_url>[/muted]")
        console.print()
        return

    table = Table(
        title=f"Knowledge Base  [muted]({store.total_chunks()} total chunks)[/muted]",
        box=box.SIMPLE_HEAD,
        header_style="bold white",
        border_style="dim white",
        show_lines=False,
    )
    table.add_column("Source",  style="source",  max_width=60)
    table.add_column("Type",    style="info",    width=7)
    table.add_column("Chunks",  style="success", width=7, justify="right")

    for d in docs:
        short = os.path.basename(d["source"]) if not d["source"].startswith("http") else d["source"]
        table.add_row(short, d["source_type"], str(d["chunk_count"]))

    console.print(table)
    console.print()


# ── delete ─────────────────────────────────────────────────────────────────────

@cli.command()
@click.argument("source")
@click.option("--yes", is_flag=True, default=False, help="Skip confirmation prompt.")
def delete(source: str, yes: bool):
    """Remove a document from the knowledge base by its source path or URL."""
    _banner()
    if not yes:
        click.confirm(
            f"Remove all chunks for '{source}' from the knowledge base?",
            abort=True,
        )

    store = _get_store()
    removed = store.delete_document(source)

    if removed == 0:
        console.print(f"[warning]⚠  No chunks found for '{source}'.[/warning]")
        console.print("[muted]Run `python cli.py list` to see available sources.[/muted]")
    else:
        console.print(f"[success]OK  Removed {removed} chunks for '{source}'.[/success]")
    console.print()


# ── clear ──────────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--yes", is_flag=True, default=False, help="Skip confirmation prompt.")
def clear(yes: bool):
    """Wipe the entire knowledge base (all documents and chunks)."""
    _banner()
    if not yes:
        click.confirm(
            "This will permanently delete ALL documents. Are you sure?",
            abort=True,
        )

    store   = _get_store()
    removed = store.clear()
    console.print(f"[success]OK  Knowledge base cleared ({removed} chunks removed).[/success]")
    console.print()


# ── info ───────────────────────────────────────────────────────────────────────

@cli.command()
def info():
    """Show knowledge base and embedding backend info."""
    _banner()
    from embeddings import EMBEDDING_BACKEND, LocalEmbeddingFunction
    ef    = LocalEmbeddingFunction()
    store = _get_store()

    table = Table(box=box.SIMPLE, show_header=False, border_style="dim white")
    table.add_column("Key",   style="muted",    min_width=22)
    table.add_column("Value", style="bold white")

    table.add_row("Embedding backend",  ef.backend)
    table.add_row("Embedding dims",     str(ef.dim))
    table.add_row("Vector store",       "ChromaDB (persistent)")
    table.add_row("LLM",                "claude-sonnet-4-20250514")
    table.add_row("Total chunks",       str(store.total_chunks()))
    table.add_row("Documents ingested", str(len(store.list_documents())))
    table.add_row("API key set",        "OK" if os.environ.get("ANTHROPIC_API_KEY") else "X  (set ANTHROPIC_API_KEY)")

    console.print(table)
    console.print()


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cli()
