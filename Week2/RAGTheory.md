# RAG Architecture Deep Dive

## What is RAG?

**Retrieval-Augmented Generation (RAG)** combines:
1. **Retrieval** - Find relevant context from external knowledge base
2. **Augmentation** - Inject retrieved context into prompt
3. **Generation** - LLM generates response using augmented context

### Why RAG?
- Ground LLM responses in your own data
- Reduce hallucinations
- Keep knowledge current without retraining
- Enable source citation

---

## Chunking Strategies

### Why Chunk?

LLMs have context window limits. We must split large documents into smaller pieces that:
- Fit within the context window
- Preserve meaningful semantic units
- Enable precise retrieval

### Strategy 1: Fixed-Size Chunking

Split text into chunks of N characters or tokens.

**Pros:**
- Simple to implement
- Predictable chunk sizes
- Fast processing

**Cons:**
- May split sentences/paragraphs in middle
- Loses context at boundaries

```python
def fixed_chunking(text, chunk_size=500, chunk_overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - chunk_overlap  # overlap
    return chunks
```

### Strategy 2: Sliding Window

Similar to fixed-size but slides continuously instead of jumping.

```python
def sliding_window(text, window_size=500, step=100):
    chunks = []
    for start in range(0, len(text), step):
        end = start + window_size
        chunks.append(text[start:end])
    return chunks
```

### Strategy 3: Semantic Chunking

Split at natural boundaries (paragraphs, sentences).

**Methods:**
- Split by paragraph (`\n\n`)
- Split by sentence (use NLP like spaCy)
- Split by heading structure (Markdown, HTML)

```python
# Using spaCy for sentence-based chunking
def semantic_chunking(text, max_sentences=5):
    doc = nlp(text)
    chunks = []
    current_chunk = []
    sentence_count = 0

    for sent in doc.sents:
        current_chunk.append(sent.text)
        sentence_count += 1

        if sentence_count >= max_sentences:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            sentence_count = 0

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks
```

### Strategy Comparison

| Strategy | Best For | Limitations |
|----------|----------|-------------|
| Fixed-size | Uniform processing, simple use cases | May break semantic units |
| Sliding window | Time-series data, overlapping context | More redundant chunks |
| Semantic | Natural text, docs with clear structure | Requires NLP library |

---

## Overlap

**Why overlap?**
- Prevent context loss at chunk boundaries
- Ensure important information isn't split between chunks

**Common overlap sizes:**
- 10-20% of chunk size for general text
- 50%+ for technical documents where context is critical

```
Chunk 1: [============]
              ^ overlap ^

Chunk 2:        [============]
                    ^ overlap ^
```

---

## Retrieval Pipeline Design

```
┌─────────────────────────────────────────────────────────────────┐
│                        RAG PIPELINE                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐     │
│  │   DOCUMENTS  │───▶│   CHUNKING   │───▶│  EMBEDDING   │     │
│  └──────────────┘    └──────────────┘    └──────────────┘     │
│                                                  │              │
│                                                  ▼              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐     │
│  │   RESPONSE   │◀───│ GENERATION   │◀───│ RETRIEVAL    │     │
│  └──────────────┘    └──────────────┘    └──────────────┘     │
│                                                 │              │
│                                                 ▼              │
│                                        ┌──────────────┐        │
│                                        │  VECTOR DB   │        │
│                                        └──────────────┘        │
└─────────────────────────────────────────────────────────────────┘
```

### Step-by-Step Pipeline

#### 1. Document Loading
```python
from langchain.document_loaders import TextLoader, PDFLoader

loader = TextLoader("document.txt")
documents = loader.load()

# Or for PDFs
pdf_loader = PDFLoader("document.pdf")
documents = pdf_loader.load()
```

#### 2. Text Chunking
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " "]
)

chunks = splitter.split_documents(documents)
```

#### 3. Embedding Generation
```python
from langchain.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Embed each chunk
chunk_texts = [chunk.page_content for chunk in chunks]
chunk_embeddings = embeddings.embed_documents(chunk_texts)
```

#### 4. Vector Storage
```python
import chromadb

client = chromadb.PersistentClient(path="./vector_store")
collection = client.create_collection("documents")

# Store with metadata
collection.add(
    ids=[f"chunk_{i}" for i in range(len(chunks))],
    embeddings=chunk_embeddings,
    documents=chunk_texts,
    metadatas=[{"source": "doc1", "index": i} for i in range(len(chunks))]
)
```

#### 5. Retrieval
```python
def retrieve(query, top_k=5):
    # Embed query
    query_embedding = embeddings.embed_query(query)

    # Search vector DB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    return results['documents'][0], results['metadatas'][0]
```

#### 6. Augmented Prompting
```python
def generate_response(query, retrieved_docs):
    context = "\n\n".join(retrieved_docs)

    prompt = f"""Use the following context to answer the question.

Context:
{context}

Question: {query}

Answer:"""

    response = llm.generate(prompt)
    return response
```

---

## Augmented Prompting

### Basic Augmented Prompt

```
You are a helpful assistant. Use the provided context to answer questions.

Context:
{retrieved_chunk_1}
{retrieved_chunk_2}
...

Question: {user_question}

Answer based on the context above:
```

### Advanced: Multi-Chunk Context

```python
def build_augmented_prompt(query, chunks, max_tokens=3000):
    context_parts = []
    current_length = 0

    for chunk in chunks:
        chunk_length = len(chunk)  # approximate tokens
        if current_length + chunk_length > max_tokens:
            break
        context_parts.append(chunk)
        current_length += chunk_length

    context = "\n\n---\n\n".join(context_parts)

    return f"""Based on the following reference documents, answer the question.

Reference Documents:
{context}

Question: {query}

Instructions:
- Only use information from the reference documents
- If the answer is not in the documents, say "I don't have enough information"
- Cite relevant parts when possible

Answer:"""
```

### Techniques for Better Augmentation

1. **Ranked Context**: Put most relevant chunks first (closer to query position)
2. **Context Compression**: Summarize long retrieved chunks
3. **Windowed Context**: Include chunks before/after retrieved ones
4. **Hybrid Search**: Combine dense (embedding) + sparse (keyword) retrieval

---

## Chunking Decision Diagram

```
                    ┌─────────────────────┐
                    │   INPUT DOCUMENT    │
                    └──────────�──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  Choose Strategy    │
                    └──────────┬──────────┘
                               │
            ┌──────────────────┼──────────────────┐
            │                  │                  │
            ▼                  ▼                  ▼
   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
   │   Is it     │    │   Is it     │    │   Is it     │
   │ structured? │    │  technical? │    │ conversational│
   └──────┬──────┘    └──────┬──────┘    └──────┬──────┘
          │                  │                  │
    ┌─────┴─────┐      ┌─────┴─────┐      ┌─────┴─────┐
    │            │      │            │      │            │
   YES          NO    YES          NO    YES          NO
    │            │      │            │      │            │
    ▼            ▼      ▼            ▼      ▼            ▼
┌────────┐  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐
│Semantic│  │Fixed   │ │Semantic│ │Sliding │ │Semantic│ │ Sliding│
│by para-│  │or slid-│ │+ over- │ │window  │ │by sen- │ │ window │
│graph   │  │ing win-│ │lap 50%+│ │+ large │ │tence   │ │+ overlap│
│        │  │dow     │ │        │ │overlap │ │        │ │        │
└────────┘  └────────┘ └────────┘ └────────┘ └────────┘ └────────┘
            │            │            │            │            │
            └────────────┴────────────┴────────────┴────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │    CHUNK SIZE?     │
                    └──────────┬──────────┘
                               │
         ┌─────────────────────┼─────────────────────┐
         │                     │                     │
         ▼                     ▼                     ▼
   ┌───────────┐        ┌───────────┐        ┌───────────┐
   │ < 500     │        │ 500-1000  │        │ > 1000    │
   │ tokens    │        │ tokens    │        │ tokens    │
   │ (short    │        │ (standard)│        │ (detailed │
   │ docs)     │        │           │        │ docs)     │
   └───────────┘        └───────────┘        └───────────┘
         │                     │                     │
         └─────────────────────┴─────────────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │    OVERLAP?         │
                    └──────────┬──────────┘
                               │
                    ┌──────────┴──────────┐
                    │                     │
                    ▼                     ▼
              ┌─────────┐           ┌─────────┐
              │  10-20% │           │  20-50% │
              │ (general)│          │ (critical│
              │         │          │ context) │
              └─────────┘           └─────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  STORE IN VECTOR   │
                    │       DATABASE      │
                    └─────────────────────┘
```

---

## Summary

| Component | Key Decision |
|-----------|--------------|
| Chunking | Fixed (simple) vs Semantic (quality) vs Sliding (time-series) |
| Chunk Size | 256-1000 tokens depending on use case |
| Overlap | 10-50% depending on criticality |
| Retrieval | Semantic similarity + optional keyword filtering |
| Augmentation | Put relevant context first, limit total tokens |
