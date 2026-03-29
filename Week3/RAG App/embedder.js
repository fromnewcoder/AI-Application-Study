/**
 * embedder.js
 * ──────────────────────────────────────────────────────────────────────────────
 * Generates dense vector embeddings for text using the Voyage AI API
 * (Anthropic's recommended embedding partner).
 *
 * Falls back gracefully to a local TF-IDF sparse representation if no API key
 * is present — useful for unit-testing the pipeline without network calls.
 *
 * Key features
 *   • Exponential-backoff retry on 429 / 5xx responses
 *   • Batched requests (Voyage accepts up to 128 texts per call)
 *   • Cosine-similarity helper bundled alongside
 */

// ── Config ─────────────────────────────────────────────────────────────────────

const VOYAGE_API_URL   = "https://api.voyageai.com/v1/embeddings";
const VOYAGE_MODEL     = "voyage-3-lite"; // fast & cheap; swap to voyage-3 for quality
const BATCH_SIZE       = 64;             // texts per API request
const MAX_RETRIES      = 4;
const INITIAL_DELAY_MS = 500;            // doubles on each retry

// ── Retry helper ───────────────────────────────────────────────────────────────

/**
 * Wraps an async function with exponential-backoff retry.
 * Retries on HTTP 429 (rate-limit) and 5xx (server errors).
 *
 * @template T
 * @param {() => Promise<T>} fn
 * @param {object} [opts]
 * @param {number} [opts.maxRetries]
 * @param {number} [opts.initialDelay]
 * @returns {Promise<T>}
 */
export async function withRetry(fn, { maxRetries = MAX_RETRIES, initialDelay = INITIAL_DELAY_MS } = {}) {
  let attempt = 0;
  let delay   = initialDelay;

  while (true) {
    try {
      return await fn();
    } catch (err) {
      const isRetryable = err.status === 429 || (err.status >= 500 && err.status < 600);

      if (!isRetryable || attempt >= maxRetries) throw err;

      attempt++;
      console.warn(`[embedder] Attempt ${attempt}/${maxRetries} failed (${err.status}). Retrying in ${delay}ms…`);
      await sleep(delay);
      delay *= 2; // exponential back-off
    }
  }
}

const sleep = (ms) => new Promise(res => setTimeout(res, ms));

// ── Voyage API ─────────────────────────────────────────────────────────────────

/**
 * Call the Voyage embedding endpoint for a batch of texts.
 * @param {string[]}  texts
 * @param {string}    apiKey
 * @returns {Promise<number[][]>}  Array of embedding vectors (float32).
 */
async function fetchEmbeddingBatch(texts, apiKey) {
  return withRetry(async () => {
    const res = await fetch(VOYAGE_API_URL, {
      method: "POST",
      headers: {
        "Content-Type":  "application/json",
        "Authorization": `Bearer ${apiKey}`,
      },
      body: JSON.stringify({ input: texts, model: VOYAGE_MODEL }),
    });

    if (!res.ok) {
      const err = new Error(`Voyage API error: ${res.statusText}`);
      err.status = res.status;
      throw err;
    }

    const json = await res.json();
    // Voyage returns [{object:'embedding', embedding:[...], index:N}, ...]
    return json.data
      .sort((a, b) => a.index - b.index)
      .map(item => item.embedding);
  });
}

// ── TF-IDF fallback ────────────────────────────────────────────────────────────
// Sparse bag-of-words vectors; useful for offline testing.

function buildTfidf(corpus) {
  const df   = {}; // document frequency per term
  const tfs  = corpus.map(text => {
    const freq = {};
    text.toLowerCase().split(/\W+/).filter(Boolean).forEach(t => {
      freq[t] = (freq[t] || 0) + 1;
    });
    Object.keys(freq).forEach(t => { df[t] = (df[t] || 0) + 1; });
    return freq;
  });

  const N     = corpus.length;
  const vocab = Object.keys(df).sort();

  const vectors = tfs.map(freq => {
    const total = Object.values(freq).reduce((a, b) => a + b, 0);
    return vocab.map(term => {
      const tf  = (freq[term] || 0) / total;
      const idf = Math.log((N + 1) / ((df[term] || 0) + 1));
      return tf * idf;
    });
  });

  return { vocab, vectors };
}

// ── Public API ─────────────────────────────────────────────────────────────────

/**
 * Embed an array of texts.  Returns a parallel array of float vectors.
 *
 * @param {string[]} texts
 * @param {object}   [opts]
 * @param {string}   [opts.apiKey]   Voyage API key. If absent → TF-IDF fallback.
 * @returns {Promise<{vectors: number[][], model: string}>}
 */
export async function embedTexts(texts, { apiKey } = {}) {
  if (!texts.length) return { vectors: [], model: "none" };

  // ── Voyage path ──
  if (apiKey) {
    const vectors = [];

    for (let i = 0; i < texts.length; i += BATCH_SIZE) {
      const batch = texts.slice(i, i + BATCH_SIZE);
      const batchVectors = await fetchEmbeddingBatch(batch, apiKey);
      vectors.push(...batchVectors);
      console.log(`[embedder] Embedded ${Math.min(i + BATCH_SIZE, texts.length)}/${texts.length} texts`);
    }

    return { vectors, model: VOYAGE_MODEL };
  }

  // ── TF-IDF fallback ──
  console.warn("[embedder] No VOYAGE_API_KEY — using TF-IDF fallback (for dev/testing only)");
  const { vectors } = buildTfidf(texts);
  return { vectors, model: "tfidf-fallback" };
}

// ── Similarity helpers ─────────────────────────────────────────────────────────

/**
 * Cosine similarity between two equal-length vectors.
 * Returns a value in [-1, 1].
 */
export function cosineSimilarity(a, b) {
  let dot = 0, magA = 0, magB = 0;
  for (let i = 0; i < a.length; i++) {
    dot  += a[i] * b[i];
    magA += a[i] * a[i];
    magB += b[i] * b[i];
  }
  if (magA === 0 || magB === 0) return 0;
  return dot / (Math.sqrt(magA) * Math.sqrt(magB));
}

/**
 * Dot-product similarity (faster when vectors are already L2-normalised).
 */
export function dotProduct(a, b) {
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += a[i] * b[i];
  return sum;
}
