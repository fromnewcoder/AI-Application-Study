/**
 * chunker.js
 * ──────────────────────────────────────────────────────────────────────────────
 * Splits raw text into overlapping chunks suitable for embedding.
 *
 * Strategy: sentence-aware sliding window.
 *   1. Split text into sentences.
 *   2. Pack sentences into chunks ≤ chunkSize tokens (approx).
 *   3. Slide forward by (chunkSize - overlap) tokens, so adjacent chunks share
 *      context — this prevents the retriever from missing answers that straddle
 *      a boundary.
 *
 * Approximate token count: 1 token ≈ 4 chars (GPT / Claude rule-of-thumb).
 */

const APPROX_CHARS_PER_TOKEN = 4;

/**
 * @param {string} text          Raw document text.
 * @param {object} [opts]
 * @param {number} [opts.chunkSize=300]   Target chunk size in tokens.
 * @param {number} [opts.overlap=50]      Overlap between consecutive chunks (tokens).
 * @param {string} [opts.source]          Metadata: where did this text come from?
 * @returns {Array<{id: string, text: string, source: string, index: number}>}
 */
export function chunkText(text, { chunkSize = 300, overlap = 50, source = "unknown" } = {}) {
  if (!text || typeof text !== "string") return [];

  const chunkChars   = chunkSize * APPROX_CHARS_PER_TOKEN;
  const overlapChars = overlap   * APPROX_CHARS_PER_TOKEN;

  // ── 1. Sentence-split (handles . ! ? followed by whitespace / end-of-string)
  const sentences = text
    .replace(/\r\n/g, "\n")
    .split(/(?<=[.!?])\s+/)
    .map(s => s.trim())
    .filter(Boolean);

  const chunks = [];
  let buffer = "";
  let chunkIndex = 0;

  const flushBuffer = (buf) => {
    if (!buf.trim()) return;
    chunks.push({
      id:     `${source}::${chunkIndex}`,
      text:   buf.trim(),
      source,
      index:  chunkIndex++,
      tokens: Math.ceil(buf.length / APPROX_CHARS_PER_TOKEN),
    });
  };

  for (const sentence of sentences) {
    // If adding this sentence would exceed the chunk size, flush and start overlap
    if (buffer.length + sentence.length > chunkChars && buffer.length > 0) {
      flushBuffer(buffer);

      // Keep trailing chars for overlap
      buffer = buffer.slice(-overlapChars) + " " + sentence;
    } else {
      buffer = buffer ? buffer + " " + sentence : sentence;
    }
  }

  // Flush any remainder
  flushBuffer(buffer);

  return chunks;
}

/**
 * Convenience: chunk an array of {text, source} documents.
 * @param {Array<{text:string, source:string}>} docs
 * @param {object} [opts]
 * @returns {Array<Chunk>}
 */
export function chunkDocuments(docs, opts = {}) {
  return docs.flatMap(doc => chunkText(doc.text, { ...opts, source: doc.source }));
}
