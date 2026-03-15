# 📚 RAG Document Q&A

A polished, fully local Retrieval-Augmented Generation app.  
Ingest PDFs, Word docs, plain text, Markdown, and web URLs — then ask questions in plain English.

---

## Stack

| Layer | Technology |
|-------|-----------|
| Embeddings | `sentence-transformers` — `all-MiniLM-L6-v2` (384-dim, runs locally) |
| Vector store | `ChromaDB` — persistent on disk |
| LLM | `Claude claude-sonnet-4-20250514` via Anthropic API |
| Document loaders | PyMuPDF · python-docx · built-in · requests+BeautifulSoup |
| CLI | `click` + `rich` |

---

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your Anthropic API key
export ANTHROPIC_API_KEY=sk-ant-...

# 3. Ingest documents
python cli.py ingest path/to/report.pdf
python cli.py ingest notes.md
python cli.py ingest paper.docx
python cli.py ingest https://en.wikipedia.org/wiki/Retrieval-augmented_generation

# 4. Ask questions
python cli.py ask "What are the key findings?"
python cli.py ask "Summarise section 3" --top-k 8
python cli.py ask "What does it say about costs?" --source path/to/report.pdf

# 5. Manage the knowledge base
python cli.py list
python cli.py delete path/to/report.pdf
python cli.py clear
python cli.py info
```

---

## Project structure

```
rag_app/
├── embeddings.py   Embedding function (sentence-transformers → sklearn fallback)
├── ingestion.py    Document loaders (PDF / DOCX / TXT / MD / URL) + smart chunker
├── store.py        ChromaDB persistent vector store wrapper
├── pipeline.py     RAG query pipeline: retrieve → prompt → Claude
├── cli.py          Rich CLI: ingest / ask / list / delete / clear / info
├── requirements.txt
└── db/             ChromaDB persists here (auto-created)
```

---

## Chunking strategy

1. Split on blank lines (paragraph boundaries).
2. Merge fragments shorter than `--min-words` (default: 10) with their neighbour.
3. Slide over any paragraph still longer than `--max-words` (default: 200) with `--overlap` (default: 30) word overlap.

Tune per document:

```bash
# Dense academic paper → smaller chunks, more overlap
python cli.py ingest paper.pdf --max-words 150 --overlap 40

# Long narrative → larger chunks
python cli.py ingest book.txt --max-words 300 --overlap 50
```

---

## Edge cases handled

| Situation | Behaviour |
|-----------|-----------|
| Empty knowledge base | Friendly message, no API call made |
| No relevant chunks found | Explains why + suggests next steps |
| Very long document | Chunks trimmed to 3 000-word context budget |
| Scanned / image-only PDF | Clear error message |
| URL requires JavaScript | Clear error + word-count warning |
| Duplicate document | Skips silently (use `--force` to re-ingest) |
| Missing API key | Immediate error with fix instructions |

---

## Swapping the embedding model

Edit `embeddings.py`.  Drop-in replacements (all work with ChromaDB):

```python
# Larger, more accurate (slower, ~120 MB)
SentenceTransformer("all-mpnet-base-v2")     # 768-dim

# Multilingual
SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# Via ChromaDB's built-in wrapper
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
ef = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
```
