/**
 * vectorStore.js
 * ──────────────────────────────────────────────────────────────────────────────
 * In-memory vector store with cosine-similarity retrieval.
 *
 * For production workloads, swap this class out for a Pinecone / Weaviate /
 * pgvector client — the interface (add / query) stays the same.
 *
 * Retrieval strategy: approximate nearest-neighbour via brute-force scan.
 * Fast enough up to ~50 k chunks; for larger corpora use HNSW (hnswlib-node).
 */

import { cosineSimilarity } from "./embedder.js";

export class VectorStore {
  constructor() {
    /** @type {Array<{id:string, text:string, source:string, index:number, vector:number[]}>} */
    this._items = [];
  }

  // ── Write ──────────────────────────────────────────────────────────────────

  /**
   * Add pre-embedded chunks to the store.
   *
   * @param {Array<{id:string, text:string, source:string, index:number}>} chunks
   * @param {number[][]} vectors  Parallel array of embedding vectors.
   */
  add(chunks, vectors) {
    if (chunks.length !== vectors.length) {
      throw new Error("[vectorStore] chunks and vectors must have the same length");
    }
    for (let i = 0; i < chunks.length; i++) {
      this._items.push({ ...chunks[i], vector: vectors[i] });
    }
    console.log(`[vectorStore] Store now holds ${this._items.length} chunks`);
  }

  // ── Read ───────────────────────────────────────────────────────────────────

  /** Total number of chunks indexed. */
  get size() { return this._items.length; }

  /**
   * Retrieve the top-k most similar chunks to a query vector.
   *
   * @param {number[]} queryVector   Embedding of the user's question.
   * @param {object}   [opts]
   * @param {number}   [opts.topK=5]          How many results to return.
   * @param {number}   [opts.minScore=0.0]    Filter out results below this similarity.
   * @param {string[]} [opts.sources]         Only return chunks from these sources.
   * @returns {Array<{id, text, source, index, score}>}
   */
  query(queryVector, { topK = 5, minScore = 0.0, sources } = {}) {
    let candidates = this._items;

    if (sources?.length) {
      candidates = candidates.filter(item => sources.includes(item.source));
    }

    const scored = candidates
      .map(item => ({ ...item, score: cosineSimilarity(queryVector, item.vector) }))
      .filter(item => item.score >= minScore)
      .sort((a, b) => b.score - a.score)
      .slice(0, topK);

    // Strip the raw vector from results — no need to send it upstream
    return scored.map(({ vector: _v, ...rest }) => rest);
  }

  // ── Persistence ────────────────────────────────────────────────────────────

  /** Serialize to plain JSON (suitable for fs.writeFile or localStorage). */
  toJSON() {
    return JSON.stringify(this._items);
  }

  /** Restore a store that was previously serialized with toJSON(). */
  static fromJSON(json) {
    const store = new VectorStore();
    store._items = JSON.parse(json);
    return store;
  }
}
