# 🔍 RAG Pipeline

A clean, zero-dependency implementation of **Retrieval-Augmented Generation** in
modern JavaScript (ESM, Node 18+).

Every layer of the stack is hand-written and documented so you can read, understand,
and modify each piece without wading through framework magic.

```
Documents → Chunker → Embedder → VectorStore → Claude (streaming) → Structured Answer
```

---

## Features

| Layer | What it does |
|---|---|
| **Chunker** | Sentence-aware sliding-window splitting with configurable size & overlap |
| **Embedder** | Voyage AI (real) or TF-IDF (offline fallback) with batching |
| **VectorStore** | In-memory cosine-similarity store with JSON export/import |
| **Retriever** | Top-k ANN search with score threshold & source filtering |
| **Generator** | Claude Sonnet via streaming SSE |
| **Structured output** | Every answer is typed JSON: `{answer, citations, confidence, reasoning}` |
| **Retries** | Exponential-backoff retry on 429 / 5xx across all API calls |
| **Tests** | 24 unit tests, Node built-in test runner (no extra deps) |

---

## Project layout

```
rag-app/
├── src/
│   ├── chunker.js       # Text → overlapping chunks
│   ├── embedder.js      # Chunks → float vectors  (+ retry helper)
│   ├── vectorStore.js   # Index + cosine retrieval
│   ├── rag.js           # Orchestrator (ingest / query / streaming)
│   └── rag.test.js      # Unit tests (node --test)
├── examples/
│   └── example.js       # End-to-end demo
└── package.json
```

---

## Quick start

### 1 — Clone & enter

```bash
git clone https://github.com/you/rag-pipeline
cd rag-pipeline
```

No `npm install` required — zero runtime dependencies.

### 2 — Set environment variables

```bash
# Required
export ANTHROPIC_API_KEY="sk-ant-..."

# Optional — enables real semantic embeddings (Voyage AI)
# Without this the pipeline uses TF-IDF (great for dev/testing)
export VOYAGE_API_KEY="pa-..."
```

### 3 — Run the example

```bash
node examples/example.js
```

You'll see streaming JSON tokens arrive in real time, then a formatted answer
with citations and a confidence rating.

### 4 — Run the tests

```bash
node --test src/rag.test.js
```

Expected: **24 pass, 0 fail** (no network calls needed).

---

## How it works — layer by layer

### 1. Chunker (`src/chunker.js`)

Raw text is split into overlapping windows so that answers which straddle a
natural boundary are not missed.

```
"Sentence A. Sentence B. Sentence C. Sentence D."
  chunk 0: [A, B]
  chunk 1: [B, C]   ← overlap keeps context
  chunk 2: [C, D]
```

**Key options:**

| Option | Default | Description |
|---|---|---|
| `chunkSize` | `300` | Target chunk size in tokens (≈ 4 chars/token) |
| `overlap` | `50` | Overlap between adjacent chunks in tokens |
| `source` | `"unknown"` | Metadata tag; shows up in citations |

```js
import { chunkText, chunkDocuments } from "./src/chunker.js";

// Single document
const chunks = chunkText(myText, { chunkSize: 300, overlap: 50, source: "wiki" });

// Multiple documents at once
const chunks = chunkDocuments([
  { text: "...", source: "doc1" },
  { text: "...", source: "doc2" },
]);
```

Each chunk: `{ id, text, source, index, tokens }`

---

### 2. Embedder (`src/embedder.js`)

Converts text chunks to dense float vectors.

**Primary path — Voyage AI** (set `VOYAGE_API_KEY`):
- Model: `voyage-3-lite` (swap to `voyage-3` for higher accuracy)
- Batches up to 64 texts per API call
- Automatic exponential-backoff retry on 429 / 5xx

**Fallback — TF-IDF** (no API key needed):
- Sparse bag-of-words vectors
- Useful for unit tests and offline development
- Works fine for small corpora (~hundreds of docs)

```js
import { embedTexts, cosineSimilarity } from "./src/embedder.js";

const { vectors, model } = await embedTexts(["hello world", "foo bar"], {
  apiKey: process.env.VOYAGE_API_KEY,  // omit → TF-IDF
});

const score = cosineSimilarity(vectors[0], vectors[1]); // 0..1
```

**`withRetry` helper** — reusable for any async operation:

```js
import { withRetry } from "./src/embedder.js";

const data = await withRetry(
  () => fetch("https://api.example.com/data").then(r => r.json()),
  { maxRetries: 4, initialDelay: 500 }  // doubles each attempt
);
```

---

### 3. VectorStore (`src/vectorStore.js`)

An in-memory nearest-neighbour index backed by brute-force cosine search.

Fast enough for ~50 k chunks. For larger corpora, drop in `hnswlib-node` or swap
the class for a Pinecone / Weaviate / pgvector client — the interface is the same.

```js
import { VectorStore } from "./src/vectorStore.js";

const store = new VectorStore();
store.add(chunks, vectors);          // parallel arrays

const results = store.query(queryVector, {
  topK:     5,      // how many results
  minScore: 0.1,    // filter below this cosine similarity
  sources:  ["doc1", "doc2"],  // optional: restrict to specific sources
});
// → [{ id, text, source, index, score }, …]

// Persist to disk and restore later (avoids re-embedding)
const json    = store.toJSON();
const store2  = VectorStore.fromJSON(json);
```

---

### 4. RAG orchestrator (`src/rag.js`)

The top-level class that wires everything together.

```js
import { RAGPipeline } from "./src/rag.js";

const pipeline = new RAGPipeline({
  anthropicApiKey: process.env.ANTHROPIC_API_KEY,  // required
  voyageApiKey:    process.env.VOYAGE_API_KEY,      // optional
  chunkSize: 300,   // tokens per chunk
  overlap:   50,    // overlap tokens
  topK:      5,     // chunks to retrieve
  minScore:  0.1,   // minimum similarity
});

// Ingest documents (chunk → embed → index)
await pipeline.ingest([
  { text: "...", source: "my-document" },
]);

// Query with streaming
const result = await pipeline.query("What is RAG?", {
  onToken: (tok) => process.stdout.write(tok),  // stream tokens
});

console.log(result);
// {
//   answer:          "RAG is ...",
//   citations:       [{ source: "my-document", passage: "..." }],
//   confidence:      "high",
//   reasoning:       "The context directly addresses the question.",
//   retrievedChunks: [{ id, text, source, score }, …]
// }
```

#### Ingest flow

```
ingest(docs)
  └─ chunkDocuments()          →  chunks[]
  └─ embedTexts(chunk.texts)   →  vectors[]
  └─ store.add(chunks, vectors)
```

#### Query flow

```
query(question)
  └─ embedTexts([question])    →  queryVector
  └─ store.query(queryVector)  →  topChunks[]
  └─ buildUserMessage()        →  prompt with context block
  └─ streamClaude(prompt)      →  SSE stream → onToken() callbacks
  └─ parseStructuredOutput()   →  { answer, citations, confidence, reasoning }
```

#### Structured output schema

Claude is instructed via system prompt to always return:

```json
{
  "answer":     "Plain prose answer grounded in context.",
  "citations":  [{ "source": "doc-name", "passage": "exact short quote" }],
  "confidence": "high | medium | low",
  "reasoning":  "One sentence explaining confidence level."
}
```

If Claude returns malformed JSON, `parseStructuredOutput` throws with the raw
text in the error message so you can diagnose prompt drift.

---

## Swap out any layer

The modules are deliberately decoupled. Here are common substitutions:

### Use OpenAI embeddings

```js
// Replace embedder.js's fetchEmbeddingBatch with:
const res = await fetch("https://api.openai.com/v1/embeddings", {
  method: "POST",
  headers: { Authorization: `Bearer ${apiKey}`, "Content-Type": "application/json" },
  body: JSON.stringify({ input: texts, model: "text-embedding-3-small" }),
});
const json = await res.json();
return json.data.map(d => d.embedding);
```

### Use Pinecone instead of the in-memory store

```js
// vectorStore.js → replace query() with:
const pinecone = new Pinecone({ apiKey });
const index    = pinecone.index("my-index");
const results  = await index.query({ vector: queryVector, topK, includeMetadata: true });
```

### Use a different LLM

Replace `streamClaude()` in `rag.js` — the rest of the pipeline stays unchanged.

---

## Configuration reference

| Constructor option | Type | Default | Description |
|---|---|---|---|
| `anthropicApiKey` | string | — | **Required.** Your Anthropic API key. |
| `voyageApiKey` | string | `undefined` | Voyage AI key. Absent → TF-IDF fallback. |
| `chunkSize` | number | `300` | Target chunk size (tokens). |
| `overlap` | number | `50` | Overlap between chunks (tokens). |
| `topK` | number | `5` | Chunks to retrieve per query. |
| `minScore` | number | `0.1` | Minimum cosine similarity to include a chunk. |

---

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | ✅ Yes | Passed to the Claude streaming API. |
| `VOYAGE_API_KEY` | ☑ Optional | Enables real semantic embeddings. Without it the pipeline uses TF-IDF. |

---

## Limitations & production notes

- **Vector store**: brute-force O(n) scan. Fine to ~50 k chunks; beyond that use HNSW.
- **No persistence by default**: call `pipeline.exportIndex()` and save the JSON to disk/S3 between runs to avoid re-embedding on restart.
- **Single-turn only**: the query method does not maintain conversation history. Wrap it in a chat loop and pass `history` into the user message for multi-turn.
- **TF-IDF fallback**: good for development but significantly worse retrieval quality vs. semantic embeddings on real questions. Always use Voyage (or OpenAI/Cohere) in production.

---

## License

MIT
