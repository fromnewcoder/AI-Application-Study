"""
pipeline.py — RAG query pipeline.

Flow:
  1. Retrieve top-k relevant chunks from the vector store.
  2. Build a grounded prompt with the retrieved context.
  3. Call Claude (claude-sonnet-4-20250514) to generate an answer.
  4. Return a structured RAGResponse with the answer, sources, and debug info.

Edge cases handled:
  • Empty collection → friendly "no documents" message (no API call).
  • No relevant chunks found (all distances > threshold) → tells the user.
  • Long context → chunks are trimmed to stay within token budget.
  • API errors → raised with a clear message.
"""

from __future__ import annotations

import os
import textwrap
from dataclasses import dataclass, field

import anthropic

from store import SearchResult, VectorStore

# ── Config ────────────────────────────────────────────────────────────────────

MODEL = "MiniMax-M2.5"
DEFAULT_TOP_K = 5
MAX_CONTEXT_WORDS = 3000        # rough guard against huge prompts
DISTANCE_THRESHOLD = 1.0        # cosine distance; lower = more similar
                                 # (1.0 keeps everything; tighten to ~0.6 for strict)

_SYSTEM_PROMPT = """\
You are a precise, helpful research assistant.  You answer questions using ONLY
the context passages provided below.  Follow these rules strictly:

1. Base every claim on the provided context.  Do NOT use outside knowledge.
2. If the context does not contain enough information, say so clearly and explain
   what is missing — do NOT guess or hallucinate.
3. When useful, cite which source(s) support your answer (e.g. "According to
   [source_name], …").
4. Be concise and direct.  Prefer bullet points for multi-part answers.
5. If the question is ambiguous, briefly note the ambiguity, then answer the
   most plausible interpretation.
"""


# ── Response model ─────────────────────────────────────────────────────────────

@dataclass
class RAGResponse:
    question: str
    answer: str
    sources: list[SearchResult]
    model: str = MODEL
    context_chunks_used: int = 0
    tokens_used: dict = field(default_factory=dict)
    no_results: bool = False
    empty_store: bool = False

    @property
    def unique_sources(self) -> list[str]:
        seen, out = set(), []
        for s in self.sources:
            if s.source not in seen:
                seen.add(s.source)
                out.append(s.source)
        return out


# ── Helpers ────────────────────────────────────────────────────────────────────

def _format_context(results: list[SearchResult]) -> str:
    """Format retrieved chunks into a numbered context block."""
    lines: list[str] = []
    word_count = 0

    for i, r in enumerate(results, start=1):
        # Source label
        import os
        label = os.path.basename(r.source) if not r.source.startswith("http") else r.source
        page_info = f", page {r.page}" if r.page else ""
        header = f"[{i}] Source: {label}{page_info}"

        # Guard total context length
        chunk_words = r.text.split()
        if word_count + len(chunk_words) > MAX_CONTEXT_WORDS:
            remaining = MAX_CONTEXT_WORDS - word_count
            if remaining < 20:
                break
            chunk_text = " ".join(chunk_words[:remaining]) + " …[truncated]"
        else:
            chunk_text = r.text

        lines.append(f"{header}\n{chunk_text}")
        word_count += len(chunk_words)

    return "\n\n---\n\n".join(lines)


def _build_user_message(question: str, context: str) -> str:
    return f"""\
## Context passages

{context}

---

## Question

{question}

Please answer the question using only the context passages above.
"""


# ── Main pipeline ─────────────────────────────────────────────────────────────

class RAGPipeline:
    """
    Retrieval-Augmented Generation pipeline.

    Parameters
    ----------
    store:
        A VectorStore instance.
    api_key:
        Anthropic API key.  Defaults to ANTHROPIC_API_KEY env var.
    top_k:
        Number of chunks to retrieve per query.
    distance_threshold:
        Cosine distance cutoff (0 = identical, 2 = opposite).
        Results above this value are filtered out.
    """

    def __init__(
        self,
        store: VectorStore,
        api_key: str | None = None,
        top_k: int = DEFAULT_TOP_K,
        distance_threshold: float = DISTANCE_THRESHOLD,
    ) -> None:
        self.store = store
        self.top_k = top_k
        self.distance_threshold = distance_threshold
        self._client = anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY")
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def ask(
        self,
        question: str,
        top_k: int | None = None,
        source_filter: str | None = None,
    ) -> RAGResponse:
        """
        Answer a question using the RAG pipeline.

        Parameters
        ----------
        question:       The user's question.
        top_k:          Override the default number of chunks to retrieve.
        source_filter:  Restrict retrieval to a single source path/URL.

        Returns
        -------
        RAGResponse with the answer, sources, and usage metadata.
        """
        question = question.strip()
        if not question:
            raise ValueError("Question cannot be empty.")

        # ── Step 0: check collection is not empty ──────────────────────────────
        if self.store.total_chunks() == 0:
            return RAGResponse(
                question=question,
                answer=(
                    "⚠️  No documents have been ingested yet.\n"
                    "Run `python cli.py ingest <file_or_url>` to add documents first."
                ),
                sources=[],
                empty_store=True,
            )

        # ── Step 1: retrieve ──────────────────────────────────────────────────
        k = top_k or self.top_k
        results = self.store.query(
            question,
            n_results=k,
            source_filter=source_filter,
        )

        # Filter by distance threshold
        relevant = [r for r in results if r.distance <= self.distance_threshold]

        if not relevant:
            return RAGResponse(
                question=question,
                answer=(
                    "⚠️  No sufficiently relevant passages were found for your question.\n\n"
                    "Suggestions:\n"
                    "  • Try rephrasing the question.\n"
                    "  • Check which documents are loaded with `python cli.py list`.\n"
                    "  • Ingest additional documents with `python cli.py ingest <source>`."
                ),
                sources=results,   # include them for transparency
                no_results=True,
            )

        # ── Step 2: build prompt ──────────────────────────────────────────────
        context = _format_context(relevant)
        user_msg = _build_user_message(question, context)

        # ── Step 3: call Claude ───────────────────────────────────────────────
        response = self._client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )

        # Handle MiniMax response - may include thinking block first
        answer = ""
        for block in response.content:
            if block.type == "text":
                answer = block.text.strip()
                break
            elif block.type == "thinking":
                answer = block.thinking.strip()

        if not answer:
            answer = "No response generated"

        return RAGResponse(
            question=question,
            answer=answer,
            sources=relevant,
            context_chunks_used=len(relevant),
            tokens_used={
                "input": response.usage.input_tokens,
                "output": response.usage.output_tokens,
            },
        )
