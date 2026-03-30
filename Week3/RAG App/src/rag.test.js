/**
 * rag.test.js
 * ──────────────────────────────────────────────────────────────────────────────
 * Unit tests for every layer of the RAG pipeline.
 * Run with:  node --test src/rag.test.js
 * (Node 18+ has a built-in test runner — no extra dependencies needed.)
 */

import { describe, it, before } from "node:test";
import assert from "node:assert/strict";

import { chunkText, chunkDocuments } from "./chunker.js";
import { embedTexts, cosineSimilarity, withRetry } from "./embedder.js";
import { VectorStore } from "./vectorStore.js";

// ── Chunker tests ─────────────────────────────────────────────────────────────

describe("chunkText", () => {
  it("returns empty array for empty input", () => {
    assert.deepEqual(chunkText(""), []);
    assert.deepEqual(chunkText(null), []);
  });

  it("produces at least one chunk for a short sentence", () => {
    const chunks = chunkText("Hello world. This is a test.", { chunkSize: 300 });
    assert.ok(chunks.length >= 1);
  });

  it("assigns correct source metadata", () => {
    const chunks = chunkText("Some sentence.", { source: "my-doc" });
    assert.equal(chunks[0].source, "my-doc");
  });

  it("splits long text into multiple chunks", () => {
    // Create a long document (~2000 chars) and chunk it at 50 tokens (~200 chars)
    const longText = Array(40).fill("This is a sentence about a topic.").join(" ");
    const chunks = chunkText(longText, { chunkSize: 50, overlap: 10 });
    assert.ok(chunks.length > 1, `Expected >1 chunk, got ${chunks.length}`);
  });

  it("each chunk has required fields", () => {
    const chunks = chunkText("First sentence. Second sentence.", { source: "test" });
    for (const c of chunks) {
      assert.ok("id"     in c, "missing id");
      assert.ok("text"   in c, "missing text");
      assert.ok("source" in c, "missing source");
      assert.ok("index"  in c, "missing index");
      assert.ok("tokens" in c, "missing tokens");
    }
  });

  it("IDs are unique across chunks", () => {
    const longText = Array(50).fill("Sentence here.").join(" ");
    const chunks = chunkText(longText, { chunkSize: 30 });
    const ids = chunks.map(c => c.id);
    assert.equal(new Set(ids).size, ids.length);
  });
});

describe("chunkDocuments", () => {
  it("combines chunks from multiple documents", () => {
    const docs = [
      { text: "First document sentence.", source: "doc1" },
      { text: "Second document sentence.", source: "doc2" },
    ];
    const chunks = chunkDocuments(docs);
    const sources = new Set(chunks.map(c => c.source));
    assert.ok(sources.has("doc1"));
    assert.ok(sources.has("doc2"));
  });
});

// ── Embedder tests ────────────────────────────────────────────────────────────

describe("embedTexts (TF-IDF fallback)", () => {
  it("returns a vector for each input text", async () => {
    const texts = ["hello world", "foo bar baz"];
    const { vectors, model } = await embedTexts(texts); // no apiKey → TF-IDF
    assert.equal(vectors.length, texts.length);
    assert.equal(model, "tfidf-fallback");
  });

  it("each vector is an array of numbers", async () => {
    const { vectors } = await embedTexts(["test sentence"]);
    assert.ok(Array.isArray(vectors[0]));
    assert.ok(vectors[0].length > 0);
    vectors[0].forEach(v => assert.equal(typeof v, "number"));
  });

  it("returns empty vectors for empty input", async () => {
    const { vectors } = await embedTexts([]);
    assert.deepEqual(vectors, []);
  });
});

describe("cosineSimilarity", () => {
  it("returns 1.0 for identical vectors", () => {
    const v = [1, 2, 3];
    assert.equal(cosineSimilarity(v, v), 1.0);
  });

  it("returns 0.0 for orthogonal vectors", () => {
    const sim = cosineSimilarity([1, 0], [0, 1]);
    assert.ok(Math.abs(sim) < 1e-9);
  });

  it("returns 0.0 for zero vectors", () => {
    assert.equal(cosineSimilarity([0, 0], [1, 2]), 0);
  });

  it("is commutative", () => {
    const a = [0.5, 0.3, 0.8];
    const b = [0.1, 0.9, 0.4];
    assert.equal(cosineSimilarity(a, b), cosineSimilarity(b, a));
  });
});

// ── Retry helper tests ────────────────────────────────────────────────────────

describe("withRetry", () => {
  it("resolves immediately on first success", async () => {
    let calls = 0;
    const result = await withRetry(async () => { calls++; return "ok"; });
    assert.equal(result, "ok");
    assert.equal(calls, 1);
  });

  it("retries on 429 error and eventually succeeds", async () => {
    let calls = 0;
    const result = await withRetry(
      async () => {
        calls++;
        if (calls < 3) {
          const e = new Error("rate limit"); e.status = 429; throw e;
        }
        return "success";
      },
      { maxRetries: 4, initialDelay: 1 } // tiny delay for tests
    );
    assert.equal(result, "success");
    assert.equal(calls, 3);
  });

  it("throws after exhausting retries", async () => {
    const err = new Error("server error"); err.status = 500;
    await assert.rejects(
      () => withRetry(async () => { throw err; }, { maxRetries: 2, initialDelay: 1 }),
      /server error/
    );
  });

  it("does not retry on 4xx (non-429) errors", async () => {
    let calls = 0;
    const err = new Error("bad request"); err.status = 400;
    await assert.rejects(
      () => withRetry(async () => { calls++; throw err; }, { maxRetries: 3, initialDelay: 1 }),
      /bad request/
    );
    assert.equal(calls, 1); // no retries for 400
  });
});

// ── VectorStore tests ─────────────────────────────────────────────────────────

describe("VectorStore", () => {
  let store;

  before(async () => {
    store = new VectorStore();
    // Build tiny TF-IDF embeddings to use as vectors
    const texts = [
      "Claude is an AI assistant made by Anthropic.",
      "RAG stands for Retrieval-Augmented Generation.",
      "Vector databases store embeddings for similarity search.",
    ];
    const { vectors } = await embedTexts(texts);
    const chunks = texts.map((text, i) => ({
      id: `chunk::${i}`, text, source: "test", index: i,
    }));
    store.add(chunks, vectors);
  });

  it("size reflects number of ingested chunks", () => {
    assert.equal(store.size, 3);
  });

  it("add() throws if chunks and vectors length mismatch", () => {
    assert.throws(
      () => store.add([{ id: "x", text: "hi", source: "s", index: 99 }], []),
      /same length/
    );
  });

  it("query() returns at most topK results", async () => {
    const { vectors: [qv] } = await embedTexts(["What is Anthropic?"]);
    const results = store.query(qv, { topK: 2 });
    assert.ok(results.length <= 2);
  });

  it("query() results are sorted by descending score", async () => {
    const { vectors: [qv] } = await embedTexts(["similarity search and vectors"]);
    const results = store.query(qv, { topK: 3 });
    for (let i = 1; i < results.length; i++) {
      assert.ok(results[i - 1].score >= results[i].score);
    }
  });

  it("query() results do not include raw vector", async () => {
    const { vectors: [qv] } = await embedTexts(["Claude AI"]);
    const results = store.query(qv, { topK: 1 });
    assert.ok(!("vector" in results[0]), "vector should not be in query results");
  });

  it("serialise/deserialise roundtrip preserves size", () => {
    const json   = store.toJSON();
    const store2 = VectorStore.fromJSON(json);
    assert.equal(store2.size, store.size);
  });
});
