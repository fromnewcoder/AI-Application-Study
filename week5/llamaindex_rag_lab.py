"""
Ph 225 — LlamaIndex Deep Dive Lab
===================================
Rebuilds a RAG pipeline using LlamaIndex, explores every major index type,
compares VectorStoreIndex vs SummaryIndex vs KeywordTableIndex, exercises
RouterQueryEngine, and benchmarks against a bare-metal baseline.

Install:
    pip install llama-index llama-index-llms-anthropic \
                llama-index-embeddings-huggingface \
                llama-index-retrievers-bm25 \
                sentence-transformers rank-bm25 anthropic

Set API key:
    export ANTHROPIC_API_KEY=sk-ant-...

Run:
    python llamaindex_rag_lab.py
"""

import os
import time
import textwrap
from dataclasses import dataclass, field

# ── Try to import LlamaIndex (graceful fallback for demo purposes) ─────────
try:
    from llama_index.core import (
        VectorStoreIndex,
        SummaryIndex,
        KeywordTableIndex,
        SimpleDirectoryReader,
        Document,
        Settings,
    )
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.core.query_engine import RouterQueryEngine, RetrieverQueryEngine
    from llama_index.core.selectors import LLMSingleSelector
    from llama_index.core.tools import QueryEngineTool
    from llama_index.core.postprocessor import SimilarityPostprocessor
    from llama_index.core.response_synthesizers import ResponseMode
    LLAMA_AVAILABLE = True
except ImportError:
    LLAMA_AVAILABLE = False
    print("⚠  llama-index not installed — running in DEMO mode (no real LLM calls).\n"
          "   Install with: pip install llama-index llama-index-llms-anthropic\n")

try:
    from llama_index.llms.anthropic import Anthropic as AnthropicLLM
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    LLAMA_LLM_AVAILABLE = True
except ImportError:
    LLAMA_LLM_AVAILABLE = False


# ── Colour helpers ─────────────────────────────────────────────────────────
RESET = "\033[0m"
BOLD  = "\033[1m"
CYAN  = "\033[36m"
GREEN = "\033[32m"
AMBER = "\033[33m"
RED   = "\033[31m"
DIM   = "\033[2m"

def h1(s): print(f"\n{BOLD}{CYAN}{'═'*60}{RESET}\n{BOLD}{CYAN}  {s}{RESET}\n{BOLD}{CYAN}{'═'*60}{RESET}")
def h2(s): print(f"\n{BOLD}── {s} {'─'*(56-len(s))}{RESET}")
def ok(s):  print(f"  {GREEN}✓{RESET} {s}")
def info(s):print(f"  {CYAN}→{RESET} {s}")
def warn(s):print(f"  {AMBER}!{RESET} {s}")
def dim(s): print(f"  {DIM}{s}{RESET}")


# ===========================================================================
# 1.  SAMPLE CORPUS
#     A small in-memory knowledge base about AI / ML topics.
#     In a real lab you'd replace this with SimpleDirectoryReader("./data").
# ===========================================================================

SAMPLE_DOCS_TEXT = [
    ("rag_overview",
     """Retrieval-Augmented Generation (RAG) is a technique that combines information
retrieval with language model generation. Instead of relying solely on parametric
knowledge stored in model weights, RAG fetches relevant documents at query time and
conditions the LLM output on those documents. This reduces hallucination and allows
the system to answer questions about private or frequently-updated information.
A standard RAG pipeline has three stages: indexing (chunking documents and building
a search index), retrieval (finding relevant chunks for a query), and generation
(synthesising an answer using the retrieved context)."""),

    ("vector_search",
     """Dense vector search encodes both queries and documents into high-dimensional
embedding vectors. Similarity is measured with cosine distance or dot product.
Models like BAAI/bge-base-en-v1.5 and text-embedding-3-small produce strong
general-purpose embeddings. Approximate Nearest Neighbour (ANN) indexes such as
FAISS, HNSW, and ScaNN enable sub-millisecond retrieval at million-document scale.
The key weakness of vector search is that it can miss exact rare terms — a query for
a specific product SKU may return semantically similar but irrelevant results."""),

    ("bm25_search",
     """BM25 (Best Match 25) is a bag-of-words ranking function widely used in
search engines. It extends TF-IDF with length normalisation (parameter b) and term
frequency saturation (parameter k1). BM25 excels at exact keyword matching and is
particularly strong for technical queries containing rare terms, product codes, or
proper nouns. It requires no GPU and is extremely fast. The main limitation is that
it treats each word independently — it cannot capture synonyms or semantic meaning,
so 'automobile' and 'car' are treated as completely different terms."""),

    ("hybrid_search",
     """Hybrid search combines sparse (BM25) and dense (vector) retrieval to get the
best of both approaches. The most common fusion technique is Reciprocal Rank Fusion
(RRF), which merges two ranked lists using the formula score = Σ 1/(k + rank_i)
where k=60 is standard. RRF does not require score normalisation, making it robust
across different retrieval systems. Research consistently shows that hybrid search
outperforms either method alone, with improvements of 5–15% on standard benchmarks
like BEIR and MS MARCO."""),

    ("reranking",
     """Cross-encoder reranking is a two-stage retrieval approach. In the first stage,
a fast retriever (BM25 or vector) generates a candidate set of 50–100 documents.
In the second stage, a cross-encoder model scores each (query, document) pair
jointly — this is much more accurate than bi-encoder similarity but too slow to
run over the full corpus. Popular cross-encoder models include ms-marco-MiniLM-L-6-v2
(fast, small) and ms-marco-electra-base (slower, better). The combination of
retrieval + reranking typically yields the highest retrieval precision."""),

    ("llamaindex_overview",
     """LlamaIndex (formerly GPT Index) is a data framework for building LLM-powered
applications over custom data. Its core abstractions are: Documents (raw input),
Nodes (chunked text with metadata), Indexes (data structures for retrieval),
Query Engines (retrieve + synthesise), and Retrievers (retrieval-only components).
LlamaIndex provides five major index types out of the box: VectorStoreIndex,
SummaryIndex, KeywordTableIndex, KnowledgeGraphIndex, and TreeIndex. All share
the same from_documents() and as_query_engine() interface, making it trivial to
swap retrieval strategies."""),

    ("llamaindex_vs_langchain",
     """LlamaIndex and LangChain are complementary rather than competing frameworks.
LlamaIndex excels at RAG use cases — its abstractions map directly to retrieval
pipeline stages and require minimal boilerplate. LangChain provides a broader
ecosystem of integrations, richer agent tooling via LangGraph, and best-in-class
observability through LangSmith. Many production teams use LlamaIndex for the
retrieval layer and LangChain for orchestration, chaining, and agent behaviour.
The key decision: if your primary use case is document Q&A, start with LlamaIndex.
If you need complex multi-step agents or extensive tool integrations, LangChain
provides more flexibility."""),

    ("chunking_strategies",
     """Document chunking profoundly affects RAG quality. Chunk size is the primary
tunable: smaller chunks (128–256 tokens) yield higher precision but lose context;
larger chunks (512–1024 tokens) retain more context but reduce specificity.
Sentence-based splitting (SentenceSplitter) is generally better than character-based
splitting because it respects sentence boundaries. Semantic chunking groups sentences
by embedding similarity rather than fixed size, producing more coherent chunks.
The chunk_overlap parameter (typically 20–50 tokens) ensures key information near
boundaries is not lost between adjacent chunks."""),

    ("evaluation_metrics",
     """RAG evaluation uses specialised metrics distinct from standard NLP benchmarks.
Context Precision measures whether retrieved chunks are relevant to the query.
Context Recall measures whether all relevant information was retrieved.
Faithfulness measures whether the generated answer is supported by the context
(hallucination detection). Answer Relevancy measures whether the answer addresses
the question. RAGAS is the most widely adopted open-source RAG evaluation framework,
computing all four metrics using an LLM judge. LlamaIndex integrates directly with
RAGAS via its evaluation module."""),

    ("postprocessors",
     """LlamaIndex postprocessors transform or filter retrieved nodes before synthesis.
SimilarityPostprocessor drops nodes below a cosine similarity threshold, removing
low-quality matches. KeywordNodePostprocessor filters nodes that must contain (or
must not contain) certain keywords. SentenceTransformerRerank applies a cross-encoder
reranker. TimeWeightedPostprocessor boosts recently-accessed nodes. Postprocessors
are applied in sequence — typically: similarity threshold → reranker → final top-k.
They are configured via the node_postprocessors parameter on as_query_engine()."""),
]


def make_documents() -> list:
    """Build LlamaIndex Document objects from the sample corpus."""
    if not LLAMA_AVAILABLE:
        return SAMPLE_DOCS_TEXT  # return raw tuples in demo mode
    return [
        Document(
            text=text,
            metadata={"source": name, "topic": name.split("_")[0]},
            doc_id=name,
        )
        for name, text in SAMPLE_DOCS_TEXT
    ]


# ===========================================================================
# 2.  CONFIGURATION
# ===========================================================================

def configure_settings():
    """Set global LlamaIndex LLM and embedding model."""
    if not LLAMA_AVAILABLE or not LLAMA_LLM_AVAILABLE:
        warn("Using mock settings (llama-index-llms-anthropic not installed).")
        return

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        warn("ANTHROPIC_API_KEY not set — LLM calls will fail.")
        return

    Settings.llm = AnthropicLLM(
        model="claude-haiku-4-5-20251001",  # Haiku: fast + cheap for lab
        api_key=api_key,
        max_tokens=1024,
    )
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5"   # small + fast local embeddings
    )
    Settings.node_parser = SentenceSplitter(
        chunk_size=512,
        chunk_overlap=50,
    )
    ok("LLM: claude-haiku-4-5-20251001  |  Embeddings: bge-small-en-v1.5")


# ===========================================================================
# 3.  INDEX COMPARISON
# ===========================================================================

@dataclass
class BenchResult:
    index_type: str
    query: str
    response: str
    latency_ms: float
    sources: list[str] = field(default_factory=list)


def time_query(engine, query: str, label: str) -> BenchResult:
    """Run a query, capture response + latency."""
    t0 = time.perf_counter()
    try:
        response = engine.query(query)
        elapsed = (time.perf_counter() - t0) * 1000
        sources = []
        if hasattr(response, "source_nodes"):
            sources = [n.node.metadata.get("source", "?")
                       for n in response.source_nodes[:3]]
        return BenchResult(
            index_type=label,
            query=query,
            response=str(response),
            latency_ms=elapsed,
            sources=sources,
        )
    except Exception as exc:
        elapsed = (time.perf_counter() - t0) * 1000
        return BenchResult(label, query, f"[ERROR: {exc}]", elapsed)


def run_index_comparison(docs):
    """Build three index types and compare their responses to the same query."""
    h2("3a. Building indexes")

    # ── VectorStoreIndex ──────────────────────────────────────────────
    info("Building VectorStoreIndex…")
    t0 = time.perf_counter()
    vector_index = VectorStoreIndex.from_documents(docs, show_progress=False)
    ok(f"VectorStoreIndex  built in {(time.perf_counter()-t0)*1000:.0f} ms")

    # ── SummaryIndex ──────────────────────────────────────────────────
    info("Building SummaryIndex…")
    t0 = time.perf_counter()
    summary_index = SummaryIndex.from_documents(docs, show_progress=False)
    ok(f"SummaryIndex      built in {(time.perf_counter()-t0)*1000:.0f} ms")

    # ── KeywordTableIndex ─────────────────────────────────────────────
    info("Building KeywordTableIndex…")
    t0 = time.perf_counter()
    keyword_index = KeywordTableIndex.from_documents(docs, show_progress=False)
    ok(f"KeywordTableIndex built in {(time.perf_counter()-t0)*1000:.0f} ms")

    h2("3b. Same query → three engines")

    queries = [
        "What is the difference between BM25 and vector search?",
        "Summarise the key RAG evaluation metrics.",
    ]

    vector_engine  = vector_index.as_query_engine(similarity_top_k=5)
    summary_engine = summary_index.as_query_engine(
        response_mode="tree_summarize"
    )
    keyword_engine = keyword_index.as_query_engine(
        response_mode="compact"
    )

    all_results: list[BenchResult] = []

    for query in queries:
        print(f"\n  Query: {BOLD}{query}{RESET}")
        for engine, label in [
            (vector_engine,  "VectorStoreIndex"),
            (summary_engine, "SummaryIndex    "),
            (keyword_engine, "KeywordTable    "),
        ]:
            r = time_query(engine, query, label)
            all_results.append(r)
            snippet = r.response[:120].replace("\n", " ")
            src_str = ", ".join(r.sources) if r.sources else "—"
            print(f"    {GREEN}{label}{RESET}  [{r.latency_ms:.0f}ms]")
            print(f"      {DIM}{snippet}…{RESET}")
            print(f"      Sources: {DIM}{src_str}{RESET}")

    return vector_index, summary_index, keyword_index, all_results


# ===========================================================================
# 4.  QUERY ENGINE FEATURES
# ===========================================================================

def demo_query_engines(vector_index, docs):
    h2("4. Query engine features")

    # ── Response modes ────────────────────────────────────────────────
    info("Comparing response modes on the same query…")
    query = "How does chunking affect RAG quality?"

    for mode_name, mode in [
        ("compact       ", ResponseMode.COMPACT),
        ("refine        ", ResponseMode.REFINE),
        ("tree_summarize", ResponseMode.TREE_SUMMARIZE),
    ]:
        engine = vector_index.as_query_engine(
            similarity_top_k=4,
            response_mode=mode,
        )
        r = time_query(engine, query, mode_name)
        snippet = r.response[:100].replace("\n", " ")
        print(f"    {CYAN}{mode_name}{RESET}  [{r.latency_ms:.0f}ms]  {DIM}{snippet}…{RESET}")

    # ── Postprocessor pipeline ────────────────────────────────────────
    info("\nWith SimilarityPostprocessor (threshold=0.6) + top-10 initial retrieval…")
    sim_filter = SimilarityPostprocessor(similarity_cutoff=0.6)
    engine = vector_index.as_query_engine(
        similarity_top_k=10,
        node_postprocessors=[sim_filter],
    )
    r = time_query(engine, "What is LlamaIndex?", "vector+sim_filter")
    ok(f"Nodes after filter: {len(r.sources)}  |  {r.latency_ms:.0f}ms")

    # ── SubQuestion demo (structural) ─────────────────────────────────
    info("\nSubQuestionQueryEngine decomposes complex queries into sub-questions.")
    info("(Skipping live call — expensive; see snippet below.)")
    dim("""
  from llama_index.core.query_engine import SubQuestionQueryEngine
  from llama_index.core.tools import QueryEngineTool

  tools = [QueryEngineTool.from_defaults(
      query_engine=engine,
      name="knowledge_base",
      description="Contains articles about RAG and LlamaIndex",
  )]
  sub_engine = SubQuestionQueryEngine.from_defaults(
      query_engine_tools=tools,
      use_async=True,
  )
  response = sub_engine.query(
      "Compare BM25 vs vector search, and explain which LlamaIndex "
      "index type is best for each."
  )
    """)


# ===========================================================================
# 5.  ROUTER QUERY ENGINE
# ===========================================================================

def demo_router(vector_index, summary_index):
    h2("5. RouterQueryEngine")

    vector_tool = QueryEngineTool.from_defaults(
        query_engine=vector_index.as_query_engine(similarity_top_k=5),
        name="factual_qa",
        description=(
            "Useful for answering specific factual questions about "
            "RAG, search techniques, LlamaIndex, or evaluation metrics."
        ),
    )
    summary_tool = QueryEngineTool.from_defaults(
        query_engine=summary_index.as_query_engine(
            response_mode="tree_summarize"
        ),
        name="summariser",
        description=(
            "Useful for producing high-level summaries of entire topics "
            "or overviews of the document collection."
        ),
    )

    router = RouterQueryEngine(
        selector=LLMSingleSelector.from_defaults(),
        query_engine_tools=[vector_tool, summary_tool],
        verbose=False,
    )

    test_queries = [
        ("What is the formula used in RRF fusion?",
         "factual → vector engine expected"),
        ("Give me an overview of all the search techniques covered.",
         "summary → summary engine expected"),
    ]

    for query, expectation in test_queries:
        info(f'Query: "{query}"')
        dim(f"  Expected routing: {expectation}")
        r = time_query(router, query, "RouterQueryEngine")
        snippet = r.response[:120].replace("\n", " ")
        print(f"  {GREEN}Response [{r.latency_ms:.0f}ms]:{RESET} {DIM}{snippet}…{RESET}\n")


# ===========================================================================
# 6.  HYBRID RETRIEVAL (BM25 + Vector in LlamaIndex)
# ===========================================================================

def demo_hybrid(vector_index):
    h2("6. Hybrid retrieval via QueryFusionRetriever")

    try:
        from llama_index.retrievers.bm25 import BM25Retriever
        from llama_index.core.retrievers import QueryFusionRetriever

        bm25_retriever   = BM25Retriever.from_defaults(
            index=vector_index, similarity_top_k=8
        )
        vector_retriever = vector_index.as_retriever(similarity_top_k=8)

        hybrid_retriever = QueryFusionRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            similarity_top_k=5,
            num_queries=1,          # no query augmentation for speed
            mode="reciprocal_rerank",
            use_async=False,
        )

        engine = RetrieverQueryEngine.from_args(
            retriever=hybrid_retriever,
            response_mode=ResponseMode.COMPACT,
        )

        query = "How does BM25 handle rare technical terms compared to vector search?"
        r = time_query(engine, query, "BM25+Vector hybrid")
        snippet = r.response[:150].replace("\n", " ")
        ok(f"Hybrid query [{r.latency_ms:.0f}ms]: {DIM}{snippet}…{RESET}")
        ok(f"Top sources: {r.sources}")

    except ImportError:
        warn("llama-index-retrievers-bm25 not installed.")
        warn("Install with: pip install llama-index-retrievers-bm25")
        dim("""
  from llama_index.retrievers.bm25 import BM25Retriever
  from llama_index.core.retrievers import QueryFusionRetriever

  hybrid = QueryFusionRetriever(
      retrievers=[bm25_retriever, vector_retriever],
      similarity_top_k=5,
      mode="reciprocal_rerank",
  )
        """)


# ===========================================================================
# 7.  BASELINE COMPARISON — LlamaIndex vs bare-metal RAG
# ===========================================================================

def run_baseline_comparison():
    """
    Shows conceptually what LlamaIndex does vs hand-rolling:
    same BM25+vector logic from the Ph225 lab, side by side.
    """
    h2("7. LlamaIndex vs bare-metal — what changes?")

    comparison = [
        ("Document ingestion",
         "SimpleDirectoryReader('data').load_data()\nVectorStoreIndex.from_documents(docs)",
         "open() each file, split manually,\nencode with sentence-transformers,\nadd to faiss index"),
        ("BM25 retrieval",
         "BM25Retriever.from_defaults(index=idx, similarity_top_k=10)",
         "BM25Okapi([d.split() for d in texts])\nbm25.get_scores(query.split())"),
        ("Hybrid fusion",
         "QueryFusionRetriever(mode='reciprocal_rerank')",
         "Manual RRF: {id: Σ 1/(k+rank)} dict"),
        ("Reranking",
         "SentenceTransformerRerank(model=…, top_n=5)",
         "CrossEncoder().predict(pairs)\nnp.argsort(scores)[::-1]"),
        ("Response synthesis",
         "engine.query(q)  # LLM call + prompt template handled",
         "Manual prompt: f'Context: {docs}\\nQ: {q}'\nclient.messages.create(…)"),
        ("Evaluation",
         "from llama_index.core.evaluation import RagasEvaluator",
         "from ragas import evaluate\n# manual dataset construction"),
    ]

    col_w = 34
    print(f"\n  {'Stage':<22}  {'LlamaIndex':<{col_w}}  {'Bare-metal':<{col_w}}")
    print(f"  {'─'*22}  {'─'*col_w}  {'─'*col_w}")
    for stage, llama, bare in comparison:
        llama_lines = llama.split("\n")
        bare_lines  = bare.split("\n")
        max_lines   = max(len(llama_lines), len(bare_lines))
        llama_lines += [""] * (max_lines - len(llama_lines))
        bare_lines  += [""] * (max_lines - len(bare_lines))
        for i in range(max_lines):
            s = stage if i == 0 else ""
            ll = llama_lines[i]
            bl = bare_lines[i]
            prefix = f"  {s:<22}  "
            print(f"{prefix}{GREEN}{ll:<{col_w}}{RESET}  {DIM}{bl:<{col_w}}{RESET}")
        print()

    print(f"  {BOLD}Verdict{RESET}")
    print(f"  LlamaIndex saves ~60% of boilerplate for standard RAG.")
    print(f"  Bare-metal gives full control — worth knowing when debugging.")
    print(f"  Use LlamaIndex by default; drop to bare-metal only when needed.\n")


# ===========================================================================
# 8.  DEMO MODE (no LlamaIndex installed)
# ===========================================================================

def run_demo_mode():
    """Explains the pipeline without executing live LLM calls."""
    h2("Demo mode — architecture walkthrough")
    print(textwrap.dedent("""
  The lab would run the following pipeline when llama-index is installed:

  ┌─ Ingestion ────────────────────────────────────────────────┐
  │  SimpleDirectoryReader → SentenceSplitter (512 tok, 50 ov) │
  │  → Nodes with metadata                                      │
  └────────────────────────────────────────────────────────────┘
           │
  ┌─ Indexing (three in parallel) ──────────────────────────────┐
  │  VectorStoreIndex  → bge-small-en-v1.5 embeddings + FAISS  │
  │  SummaryIndex      → flat node list, tree synthesis         │
  │  KeywordTableIndex → inverted keyword → node_ids dict       │
  └────────────────────────────────────────────────────────────┘
           │
  ┌─ Query (RouterQueryEngine picks) ───────────────────────────┐
  │  Factual Q → VectorStoreIndex + SimilarityPostprocessor     │
  │  Summary Q → SummaryIndex (tree_summarize mode)             │
  └────────────────────────────────────────────────────────────┘
           │
  ┌─ Hybrid path (QueryFusionRetriever) ────────────────────────┐
  │  BM25Retriever + vector_retriever → RRF fusion → top-5     │
  │  → SentenceTransformerRerank → top-3                        │
  └────────────────────────────────────────────────────────────┘
           │
  ┌─ Synthesis ─────────────────────────────────────────────────┐
  │  claude-haiku-4-5 (compact / refine / tree_summarize mode)  │
  └────────────────────────────────────────────────────────────┘
    """))


# ===========================================================================
# 9.  KNOWLEDGE GRAPH INDEX DEMO (structural — expensive, shown as snippet)
# ===========================================================================

def show_kg_snippet():
    h2("Bonus: KnowledgeGraphIndex snippet")
    dim("""
  from llama_index.core import KnowledgeGraphIndex

  # Build — LLM extracts (subject, predicate, object) triples
  kg_index = KnowledgeGraphIndex.from_documents(
      docs,
      max_triplets_per_chunk=5,
      include_embeddings=True,   # hybrid: graph + vector
      show_progress=True,
  )

  # Query — traverses graph then synthesises
  engine = kg_index.as_query_engine(
      include_text=True,
      retriever_mode="keyword",  # or "embedding"
      response_mode="tree_summarize",
  )
  response = engine.query(
      "What is the relationship between BM25 and hybrid search?"
  )

  # Visualise the graph (optional, needs pyvis)
  from llama_index.core.query_engine import KnowledgeGraphQueryEngine
  kg_index.get_networkx_graph()  # returns nx.Graph for viz
    """)
    info("Use KnowledgeGraphIndex when entity relationships matter more than text similarity.")
    info("Cost: ~1 LLM call per chunk for triple extraction at index time.")


# ===========================================================================
# MAIN
# ===========================================================================

if __name__ == "__main__":
    h1("Ph 225 — LlamaIndex Deep Dive Lab")

    if not LLAMA_AVAILABLE:
        run_demo_mode()
        run_baseline_comparison()
        show_kg_snippet()
        print(f"\n{AMBER}Install llama-index to run live queries:{RESET}")
        print("  pip install llama-index llama-index-llms-anthropic "
              "llama-index-embeddings-huggingface llama-index-retrievers-bm25\n")
    else:
        h2("1. Configuration")
        configure_settings()

        h2("2. Loading corpus")
        docs = make_documents()
        ok(f"Loaded {len(docs)} documents")
        for doc in docs[:3]:
            dim(f"  {doc.doc_id}: {doc.text[:60]}…")
        dim(f"  … and {len(docs)-3} more")

        # ── Core lab ──────────────────────────────────────────────────
        vector_index, summary_index, keyword_index, bench_results = \
            run_index_comparison(docs)

        demo_query_engines(vector_index, docs)
        demo_router(vector_index, summary_index)
        demo_hybrid(vector_index)
        run_baseline_comparison()
        show_kg_snippet()

        # ── Summary table ─────────────────────────────────────────────
        h2("Lab summary — latency by index type")
        by_type: dict[str, list[float]] = {}
        for r in bench_results:
            key = r.index_type.strip()
            by_type.setdefault(key, []).append(r.latency_ms)
        for k, lats in sorted(by_type.items()):
            avg = sum(lats) / len(lats)
            bar = "█" * int(avg / 50)
            print(f"  {k:<20} avg {avg:>6.0f}ms  {GREEN}{bar}{RESET}")

        print(f"\n{GREEN}{BOLD}Lab complete.{RESET} "
              "Key takeaways:\n"
              "  1. VectorStoreIndex is the default — best for open Q&A.\n"
              "  2. SummaryIndex shines for full-document synthesis.\n"
              "  3. RouterQueryEngine removes the need to manually pick an index.\n"
              "  4. QueryFusionRetriever adds hybrid search in ~5 lines.\n"
              "  5. LlamaIndex vs bare-metal: same logic, ~60% less boilerplate.\n")
