/**
 * rag.js
 * ──────────────────────────────────────────────────────────────────────────────
 * The main RAG pipeline.  Ties together:
 *   chunker  →  embedder  →  vectorStore  →  Claude (streaming + structured output)
 *
 * Two public surface areas:
 *   1. RAGPipeline class  — ingest documents, ask questions, get streaming answers
 *   2. Structured output  — every answer is returned as { answer, citations, confidence }
 */

import { chunkDocuments }           from "./chunker.js";
import { embedTexts, withRetry }    from "./embedder.js";
import { VectorStore }              from "./vectorStore.js";

// ── Config ─────────────────────────────────────────────────────────────────────

const CLAUDE_API_URL = "https://api.anthropic.com/v1/messages";
const CLAUDE_MODEL   = "claude-sonnet-4-20250514";
const MAX_TOKENS     = 1024;

// ── Prompt templates ───────────────────────────────────────────────────────────

/**
 * System prompt — instructs Claude to:
 *   (a) answer only from provided context
 *   (b) return a strict JSON structure
 *   (c) cite sources
 */
const SYSTEM_PROMPT = `You are a precise, helpful assistant that answers questions using ONLY the context passages provided.

RULES:
1. Base your answer ENTIRELY on the provided context. Do not use outside knowledge.
2. If the context does not contain enough information, say so — do not hallucinate.
3. Always cite the source(s) you used using the "source" field from each passage.
4. Respond ONLY with valid JSON matching this exact schema:

{
  "answer": "<your answer in clear prose>",
  "citations": [
    { "source": "<source name>", "passage": "<exact short quote from context>" }
  ],
  "confidence": "high" | "medium" | "low",
  "reasoning": "<one sentence explaining why you chose this confidence level>"
}

Do not include markdown fences, preamble, or any text outside the JSON object.`;

/**
 * Build the user-turn message that includes retrieved context + the question.
 *
 * @param {string}   question
 * @param {object[]} retrievedChunks
 * @returns {string}
 */
function buildUserMessage(question, retrievedChunks) {
  const contextBlock = retrievedChunks
    .map((c, i) =>
      `--- Passage ${i + 1} (source: ${c.source}, similarity: ${c.score.toFixed(3)}) ---\n${c.text}`
    )
    .join("\n\n");

  return `CONTEXT:\n${contextBlock}\n\nQUESTION: ${question}`;
}

// ── Structured-output parser ───────────────────────────────────────────────────

/**
 * Parse and validate Claude's JSON response.
 *
 * @param {string} raw  Raw text from the API.
 * @returns {{ answer:string, citations:object[], confidence:string, reasoning:string }}
 */
function parseStructuredOutput(raw) {
  // Strip accidental markdown code fences just in case
  const cleaned = raw.replace(/^```(?:json)?\s*/m, "").replace(/\s*```$/m, "").trim();

  let parsed;
  try {
    parsed = JSON.parse(cleaned);
  } catch {
    throw new Error(`[rag] Claude returned non-JSON output:\n${raw}`);
  }

  // Light schema validation
  const required = ["answer", "citations", "confidence", "reasoning"];
  for (const key of required) {
    if (!(key in parsed)) throw new Error(`[rag] Missing key in structured output: "${key}"`);
  }

  if (!Array.isArray(parsed.citations)) {
    throw new Error('[rag] "citations" must be an array');
  }

  const validConfidence = ["high", "medium", "low"];
  if (!validConfidence.includes(parsed.confidence)) {
    parsed.confidence = "low"; // safe default
  }

  return parsed;
}

// ── Claude API call (streaming) ────────────────────────────────────────────────

/**
 * Call Claude with streaming and yield text delta chunks as they arrive.
 * Uses server-sent events (SSE) — compatible with both Node.js 18+ and browsers.
 *
 * @param {string} userMessage
 * @param {string} apiKey
 * @yields {string}   Text delta chunks.
 * @returns {string}  The fully accumulated response text.
 */
async function* streamClaude(userMessage, apiKey) {
  const response = await withRetry(async () => {
    const res = await fetch(CLAUDE_API_URL, {
      method: "POST",
      headers: {
        "Content-Type":      "application/json",
        "x-api-key":         apiKey,
        "anthropic-version": "2023-06-01",
      },
      body: JSON.stringify({
        model:      CLAUDE_MODEL,
        max_tokens: MAX_TOKENS,
        stream:     true,
        system:     SYSTEM_PROMPT,
        messages:   [{ role: "user", content: userMessage }],
      }),
    });

    if (!res.ok) {
      const err = new Error(`Claude API error: ${res.status} ${res.statusText}`);
      err.status = res.status;
      throw err;
    }

    return res;
  });

  // Parse the SSE stream
  const reader  = response.body.getReader();
  const decoder = new TextDecoder();
  let   buffer  = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });

    // SSE events are separated by double newlines
    const events = buffer.split("\n\n");
    buffer = events.pop(); // keep the incomplete last chunk

    for (const event of events) {
      const dataLine = event.split("\n").find(l => l.startsWith("data: "));
      if (!dataLine) continue;

      const data = dataLine.slice(6); // strip "data: "
      if (data === "[DONE]") return;

      let parsed;
      try { parsed = JSON.parse(data); } catch { continue; }

      if (parsed.type === "content_block_delta" && parsed.delta?.type === "text_delta") {
        yield parsed.delta.text;
      }
    }
  }
}

// ── Main pipeline class ────────────────────────────────────────────────────────

export class RAGPipeline {
  /**
   * @param {object} opts
   * @param {string} opts.anthropicApiKey   Required — Claude API key.
   * @param {string} [opts.voyageApiKey]    Optional — Voyage AI key for real embeddings.
   * @param {number} [opts.chunkSize=300]   Chunk size in tokens.
   * @param {number} [opts.overlap=50]      Overlap in tokens.
   * @param {number} [opts.topK=5]          Chunks to retrieve per query.
   * @param {number} [opts.minScore=0.1]    Minimum similarity threshold.
   */
  constructor(opts) {
    if (!opts?.anthropicApiKey) throw new Error("anthropicApiKey is required");
    this._opts  = opts;
    this._store = new VectorStore();
  }

  // ── Ingest ─────────────────────────────────────────────────────────────────

  /**
   * Add documents to the knowledge base.
   * Documents are chunked, embedded, and stored — ready to retrieve.
   *
   * @param {Array<{text:string, source:string}>} docs
   */
  async ingest(docs) {
    console.log(`[rag] Ingesting ${docs.length} document(s)…`);

    // 1. Chunk
    const chunks = chunkDocuments(docs, {
      chunkSize: this._opts.chunkSize ?? 300,
      overlap:   this._opts.overlap   ?? 50,
    });
    console.log(`[rag] Created ${chunks.length} chunks`);

    // 2. Embed
    const { vectors, model } = await embedTexts(
      chunks.map(c => c.text),
      { apiKey: this._opts.voyageApiKey }
    );
    console.log(`[rag] Embedded with model: ${model}`);

    // 3. Index
    this._store.add(chunks, vectors);
  }

  // ── Query ──────────────────────────────────────────────────────────────────

  /**
   * Ask a question against the indexed knowledge base.
   * Returns a structured answer **and** streams the raw JSON tokens as they
   * arrive from Claude so the UI can show a typing effect.
   *
   * @param {string}   question
   * @param {object}   [opts]
   * @param {Function} [opts.onToken]   Called with each streaming text delta.
   * @returns {Promise<{answer, citations, confidence, reasoning, retrievedChunks}>}
   */
  async query(question, { onToken } = {}) {
    if (this._store.size === 0) {
      throw new Error("[rag] Knowledge base is empty — call ingest() first");
    }

    // 1. Embed the question
    const { vectors: [queryVector] } = await embedTexts(
      [question],
      { apiKey: this._opts.voyageApiKey }
    );

    // 2. Retrieve top-k chunks
    const retrievedChunks = this._store.query(queryVector, {
      topK:     this._opts.topK     ?? 5,
      minScore: this._opts.minScore ?? 0.1,
    });

    if (retrievedChunks.length === 0) {
      return {
        answer:           "No relevant information found in the knowledge base.",
        citations:        [],
        confidence:       "low",
        reasoning:        "No chunks exceeded the minimum similarity threshold.",
        retrievedChunks:  [],
      };
    }

    console.log(`[rag] Retrieved ${retrievedChunks.length} chunks (top score: ${retrievedChunks[0].score.toFixed(3)})`);

    // 3. Build user message
    const userMessage = buildUserMessage(question, retrievedChunks);

    // 4. Stream Claude response
    let rawResponse = "";
    for await (const token of streamClaude(userMessage, this._opts.anthropicApiKey)) {
      rawResponse += token;
      onToken?.(token);
    }

    // 5. Parse structured output
    const structured = parseStructuredOutput(rawResponse);

    return { ...structured, retrievedChunks };
  }

  // ── Persistence ────────────────────────────────────────────────────────────

  /** Export the vector store as a JSON string. */
  exportIndex() { return this._store.toJSON(); }

  /** Restore a previously exported index (avoids re-embedding). */
  importIndex(json) {
    this._store = VectorStore.fromJSON(json);
    console.log(`[rag] Loaded index: ${this._store.size} chunks`);
  }
}
