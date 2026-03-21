# Week 2 Summary: Embeddings, Vector Databases & RAG

## Overview

Week 2 builds on Week 1's token foundations to explore how to **retrieve relevant information** from large documents and inject it into LLM prompts — the core of RAG (Retrieval-Augmented Generation).

---

## Theory Files

### `EmbeddingsTheory.md`
- **Embeddings** are dense, continuous, fixed-size vector representations of text that capture semantic meaning
- Similar concepts are positioned close together in vector space
- **Cosine similarity** (range -1 to 1) measures angle between vectors — 1.0 = identical, 0 = unrelated
- Models covered: `text-embedding-ada-002` (1536-dim), `all-MiniLM-L6-v2` (384-dim), Cohere, DeepSeek
- Use cases: semantic search, text similarity, clustering, recommendation, RAG, anomaly detection

### `ChromaDBTheory.md`
- **ChromaDB** is an open-source vector database with persistent storage and built-in embedding computation
- CRUD operations: `collection.add()`, `collection.get()`, `collection.update()`, `collection.delete()`
- **HNSW** (Hierarchical Navigable Small World) is the default ANN index — fast with high accuracy
- Distance metrics: **cosine** (semantic), **L2** (geometric), **IP** (inner product)
- Metadata filtering with operators: `$eq`, `$ne`, `$gt`, `$gte`, `$lt`, `$lte`, `$in`, `$exists`
- Supports client-server mode for production

### `RAGTheory.md`
- **RAG = Retrieval + Augmentation + Generation** — grounds LLM responses in your own data
- Reduces hallucinations, keeps knowledge current without retraining, enables citation
- **Chunking strategies:**
  - **Fixed-size**: Simple, predictable, may break semantic units
  - **Sliding window**: Overlap preserves context across boundaries
  - **Semantic**: Splits at paragraph/sentence boundaries using NLP (spaCy) — most natural
  - **Recursive**: Tries paragraph → sentence → character boundaries progressively
- **Overlap**: 10–20% for general text, 50%+ for technical documents
- Chunk size guide: <500 tokens (short), 500–1000 (standard), >1000 (detailed)
- Augmented prompting: inject retrieved context + instructions into prompt before LLM call
- Pipeline: Document → Chunking → Embedding → Vector DB → Retrieval → Augmentation → LLM

---

## Demos

### `EmbeddingsDemo.py`
- Basic embedding generation and similarity comparison
- Uses a model to embed text and compute cosine similarity between pairs

### `ChromaDBDemo.py`
- CRUD operations with ChromaDB
- Persistent client with metadata filtering demo

### `RAGDemo.py`
Complete RAG pipeline demo with 4 chunking strategies applied to an AI overview document:

```python
# 4 strategies compared:
fixed_chunking(text, chunk_size=200)           # non-overlapping character chunks
sliding_window_chunking(text, 200, step=100)    # overlapping windows
semantic_chunking(text, max_sentences=2)        # spaCy sentence boundaries
recursive_chunking(text, 150, overlap=30)       # paragraph/sentence fallback
```

- Stores chunks in Chroma (`./rag_demo_db`)
- Queries: "What is machine learning?", "neural networks deep learning", "language processing computers"
- Builds augmented prompts from retrieved context

### `RAGPDFDemo.py`
End-to-end PDF RAG pipeline using `AI_Study_Material.pdf`:

```
PDF → PyPDF2 extraction → Semantic chunking (spaCy) → ChromaDB → Retrieval → LLM answer
```

- **Dual LLM support**: MiniMax (`MiniMax-M2.5` via Anthropic) and DeepSeek (`deepseek-chat`)
- Stores chunk metadata: `chunk_id`, `source`, `strategy`, `char_count`
- 3 demo queries with LLM-generated answers
- Usage: `python RAGPDFDemo.py [pdf_path] [--llm minimax|deepseek]`

---

## RAG App (`RAG App/`)

A polished, production-ready CLI RAG application.

### Stack
| Layer | Technology |
|-------|-----------|
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (384-dim, runs locally) |
| Vector store | ChromaDB (persistent) |
| LLM | Claude `claude-sonnet-4-20250514` via Anthropic API |
| Document loaders | PyMuPDF, python-docx, requests + BeautifulSoup |
| CLI | `click` + `rich` |

### Smart Chunking Pipeline
1. Split on blank lines (paragraph boundaries)
2. Merge fragments < `--min-words` (default: 10) with neighbours
3. Slide over paragraphs > `--max-words` (default: 200) with `--overlap` (default: 30) words

### CLI Commands
```bash
python cli.py ingest path/to/report.pdf        # ingest a document
python cli.py ingest notes.md
python cli.py ingest https://example.com       # web URL
python cli.py ask "What are the key findings?"  # query the knowledge base
python cli.py ask "Summarise section 3" --top-k 8
python cli.py ask "Costs?" --source report.pdf  # filter by source
python cli.py list                             # list ingested documents
python cli.py delete path/to/report.pdf         # remove a document
python cli.py clear                            # wipe knowledge base
python cli.py info                             # show stats
```

### Edge Cases Handled
- Empty knowledge base → friendly message, no API call
- No relevant chunks → explains why + suggests next steps
- Very long documents → trimmed to 3000-word context budget
- Scanned/image-only PDFs → clear error message
- Duplicate documents → skips silently (use `--force` to re-ingest)
- Missing API key → immediate error with fix instructions

---

## Key Concepts Summary

| Concept | Key Point |
|---------|-----------|
| Embedding | Dense vector capturing semantic meaning; similar texts → similar vectors |
| Cosine similarity | Measures angle between vectors; 1 = identical, 0 = unrelated |
| ChromaDB | Open-source vector DB; HNSW index; persistent storage; metadata filtering |
| RAG | Retrieval + Augmentation + Generation; grounds LLM in your data |
| Chunking | Fixed (simple), Sliding (overlap), Semantic (NLP boundaries) |
| Overlap | 10–20% general / 50%+ technical; prevents context loss at boundaries |
| HNSW | Default ANN index in Chroma; fast approximate nearest neighbor search |
| Augmented prompt | Retrieved context injected into prompt with instructions for LLM |

---

## Dependencies

```bash
# Core
pip install chromadb tiktoken spacy
python -m spacy download en_core_web_md

# RAG App
pip install -r "Week2/RAG App/requirements.txt"
```
