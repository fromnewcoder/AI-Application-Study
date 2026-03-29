/**
 * example.js
 * ──────────────────────────────────────────────────────────────────────────────
 * End-to-end demonstration of the RAG pipeline.
 *
 * Run:
 *   ANTHROPIC_API_KEY=sk-... node examples/example.js
 *
 * Optional (for real embeddings instead of TF-IDF):
 *   VOYAGE_API_KEY=pa-... ANTHROPIC_API_KEY=sk-... node examples/example.js
 */

import { RAGPipeline } from "../src/rag.js";

// ── Sample knowledge base ─────────────────────────────────────────────────────

const DOCUMENTS = [
  {
    source: "anthropic-blog-claude",
    text: `Claude is Anthropic's AI assistant, designed with safety and helpfulness as core principles.
Claude 3 introduced a family of models: Haiku (fastest), Sonnet (balanced), and Opus (most capable).
Anthropic was founded in 2021 by former OpenAI researchers including Dario Amodei and Daniela Amodei.
The company's mission is the responsible development and maintenance of advanced AI for the long-term benefit of humanity.
Claude is trained using a technique called Constitutional AI, which guides the model toward helpful, harmless, and honest responses.
Constitutional AI uses a set of principles to critique and revise Claude's outputs during training.`,
  },
  {
    source: "rag-overview",
    text: `Retrieval-Augmented Generation (RAG) is a technique that grounds language model outputs in external knowledge.
Instead of relying solely on parametric knowledge baked into model weights, RAG retrieves relevant passages from a document store at inference time.
The retrieved passages are prepended to the prompt as context, allowing the model to answer questions accurately and cite sources.
RAG reduces hallucination because the model's answer is anchored to retrieved text rather than recalled from training.
Key components of a RAG system: chunker, embedding model, vector store, retriever, and a generator (LLM).
Popular vector stores include Pinecone, Weaviate, Qdrant, Chroma, and pgvector.`,
  },
  {
    source: "embeddings-explainer",
    text: `Text embeddings are dense numerical representations of text in a high-dimensional vector space.
Semantically similar texts cluster near each other in this space, enabling similarity search.
Embedding models like Voyage, OpenAI ada-002, and Cohere Embed convert raw text into float vectors.
Cosine similarity is the standard metric: it measures the angle between two vectors, returning 1.0 for identical semantics and 0.0 for unrelated text.
Typical embedding dimensions range from 512 to 3072 floats per text snippet.
Chunking strategy significantly impacts retrieval quality — overly long chunks dilute signal, while overly short chunks lose context.`,
  },
];

// ── Main ──────────────────────────────────────────────────────────────────────

async function main() {
  const anthropicApiKey = process.env.ANTHROPIC_API_KEY;
  const voyageApiKey    = process.env.VOYAGE_API_KEY; // optional

  if (!anthropicApiKey) {
    console.error("❌  Set ANTHROPIC_API_KEY environment variable first.");
    process.exit(1);
  }

  // 1. Initialise pipeline
  const pipeline = new RAGPipeline({ anthropicApiKey, voyageApiKey });

  // 2. Ingest documents
  console.log("\n📥  Ingesting documents…");
  await pipeline.ingest(DOCUMENTS);

  // 3. Ask questions with streaming
  const questions = [
    "What is Constitutional AI and how does it help Claude?",
    "Why does RAG reduce hallucination in language models?",
    "What are embedding dimensions and why do they matter?",
  ];

  for (const question of questions) {
    console.log(`\n${"─".repeat(70)}`);
    console.log(`❓  ${question}\n`);
    process.stdout.write("💬  Streaming answer: ");

    const result = await pipeline.query(question, {
      onToken: (token) => process.stdout.write(token),
    });

    console.log("\n");
    console.log(`📎  Citations:`);
    result.citations.forEach(c => {
      console.log(`    [${c.source}] "${c.passage}"`);
    });
    console.log(`🎯  Confidence: ${result.confidence} — ${result.reasoning}`);
    console.log(`📊  Retrieved ${result.retrievedChunks.length} chunks`);
  }

  // 4. Demonstrate index persistence
  console.log(`\n${"─".repeat(70)}`);
  console.log("💾  Demonstrating index export / import…");
  const exported = pipeline.exportIndex();
  const pipeline2 = new RAGPipeline({ anthropicApiKey, voyageApiKey });
  pipeline2.importIndex(exported);
  console.log(`✅  New pipeline loaded ${JSON.parse(exported).length} chunks from exported index.`);
}

main().catch(err => {
  console.error("\n❌  Fatal error:", err.message);
  process.exit(1);
});
