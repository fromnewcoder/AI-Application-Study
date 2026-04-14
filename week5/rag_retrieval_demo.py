
"""
Ph 225 — Retrieval Improvements Demo
======================================
Demonstrates hybrid search (BM25 + vector), RRF fusion, cross-encoder
reranking, HyDE, MMR, and metadata filtering in a single runnable script.

Install dependencies:
    pip install rank-bm25 sentence-transformers faiss-cpu numpy anthropic

Set your API key:
    export ANTHROPIC_API_KEY=sk-ant-...

Run:
    python rag_retrieval_demo.py
"""

import os
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# 1. CORPUS  — small in-memory knowledge base with metadata
# ---------------------------------------------------------------------------

@dataclass
class Document:
    id: int
    text: str
    metadata: dict = field(default_factory=dict)


CORPUS = [
    Document(0, "RAG combines retrieval with generation. A retriever finds relevant documents, "
                "then an LLM generates an answer conditioned on those documents.",
             {"year": 2023, "topic": "rag", "source": "textbook"}),
    Document(1, "BM25 is a bag-of-words ranking function based on term frequency and inverse "
                "document frequency. It uses a length normalisation parameter b and a saturation "
                "parameter k1.",
             {"year": 2022, "topic": "bm25", "source": "paper"}),
    Document(2, "Dense retrieval encodes queries and documents into a shared embedding space. "
                "Similarity is measured with cosine or dot-product distance.",
             {"year": 2023, "topic": "vector", "source": "paper"}),
    Document(3, "Cross-encoders jointly encode the query and document together, producing a "
                "more accurate relevance score than bi-encoders at the cost of speed.",
             {"year": 2023, "topic": "reranking", "source": "paper"}),
    Document(4, "HyDE (Hypothetical Document Embeddings) generates a fake answer using an LLM, "
                "then retrieves documents similar to that fake answer instead of the raw query.",
             {"year": 2023, "topic": "hyde", "source": "paper"}),
    Document(5, "Maximal Marginal Relevance selects documents by balancing relevance to the query "
                "and diversity among selected documents, avoiding redundant results.",
             {"year": 2022, "topic": "mmr", "source": "paper"}),
    Document(6, "Metadata filtering lets you restrict retrieval to a subset of documents using "
                "structured fields such as date, author, or topic before running similarity search.",
             {"year": 2024, "topic": "metadata", "source": "blog"}),
    Document(7, "Reciprocal Rank Fusion (RRF) merges multiple ranked lists by summing "
                "1/(k + rank) scores. It is robust and requires no score normalisation.",
             {"year": 2023, "topic": "rrf", "source": "paper"}),
    Document(8, "Chunking strategy affects retrieval quality. Smaller chunks improve precision "
                "but lose context; larger chunks retain context but reduce specificity.",
             {"year": 2024, "topic": "chunking", "source": "blog"}),
    Document(9, "Sentence-BERT uses siamese networks to produce semantically meaningful sentence "
                "embeddings suitable for semantic textual similarity tasks.",
             {"year": 2021, "topic": "embeddings", "source": "paper"}),
]


# ---------------------------------------------------------------------------
# 2. METADATA FILTERING
# ---------------------------------------------------------------------------

def metadata_filter(docs: list[Document], filters: dict) -> list[Document]:
    """
    Keep only documents whose metadata matches ALL supplied filters.

    Example:
        metadata_filter(CORPUS, {"year": 2023, "source": "paper"})
    """
    result = []
    for doc in docs:
        match = all(doc.metadata.get(k) == v for k, v in filters.items())
        if match:
            result.append(doc)
    return result


# ---------------------------------------------------------------------------
# 3. BM25 RETRIEVAL
# ---------------------------------------------------------------------------

class BM25:
    """Minimal BM25Okapi implementation (no external dependency version)."""

    def __init__(self, corpus: list[Document], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_ids = [d.id for d in corpus]
        tokenised = [d.text.lower().split() for d in corpus]
        self.avgdl = sum(len(t) for t in tokenised) / len(tokenised)
        self.df: dict[str, int] = {}
        self.tf: list[dict[str, int]] = []
        self.N = len(tokenised)
        for tokens in tokenised:
            tf = {}
            for tok in tokens:
                tf[tok] = tf.get(tok, 0) + 1
            self.tf.append(tf)
            for tok in set(tokens):
                self.df[tok] = self.df.get(tok, 0) + 1

    def score(self, query_tokens: list[str]) -> list[tuple[int, float]]:
        scores = []
        for i, tf in enumerate(self.tf):
            dl = sum(tf.values())
            s = 0.0
            for tok in query_tokens:
                if tok not in tf:
                    continue
                freq = tf[tok]
                idf = math.log((self.N - self.df.get(tok, 0) + 0.5) /
                               (self.df.get(tok, 0) + 0.5) + 1)
                s += idf * (freq * (self.k1 + 1)) / (
                    freq + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            scores.append((self.doc_ids[i], s))
        return sorted(scores, key=lambda x: x[1], reverse=True)

    def retrieve(self, query: str, top_k: int = 10) -> list[int]:
        scored = self.score(query.lower().split())
        return [doc_id for doc_id, _ in scored[:top_k]]


# ---------------------------------------------------------------------------
# 4. VECTOR RETRIEVAL (sentence-transformers + numpy cosine)
# ---------------------------------------------------------------------------

class VectorIndex:
    """
    Thin wrapper around sentence-transformers + numpy for cosine similarity.
    Swap numpy search for faiss.IndexFlatIP in production.
    """

    def __init__(self, corpus: list[Document]):
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
            texts = [d.text for d in corpus]
            self.doc_ids = [d.id for d in corpus]
            embs = self.model.encode(texts, normalize_embeddings=True)
            self.embeddings = np.array(embs, dtype="float32")
            self._available = True
        except ImportError:
            print("  [VectorIndex] sentence-transformers not installed — "
                  "using random embeddings for demo.")
            self.doc_ids = [d.id for d in corpus]
            self.embeddings = np.random.randn(len(corpus), 64).astype("float32")
            self.embeddings /= np.linalg.norm(self.embeddings, axis=1, keepdims=True)
            self._available = False

    def retrieve(self, query: str, top_k: int = 10) -> list[int]:
        if self._available:
            q_emb = self.model.encode([query], normalize_embeddings=True)[0]
        else:
            q_emb = np.random.randn(self.embeddings.shape[1]).astype("float32")
            q_emb /= np.linalg.norm(q_emb)
        sims = self.embeddings @ q_emb
        ranked = np.argsort(-sims)[:top_k]
        return [self.doc_ids[i] for i in ranked]

    def embed(self, text: str) -> np.ndarray:
        if self._available:
            return self.model.encode([text], normalize_embeddings=True)[0]
        v = np.random.randn(self.embeddings.shape[1]).astype("float32")
        return v / np.linalg.norm(v)


# ---------------------------------------------------------------------------
# 5. RECIPROCAL RANK FUSION
# ---------------------------------------------------------------------------

def rrf_fuse(*ranked_lists: list[int], k: int = 60) -> list[int]:
    """
    Merge any number of ranked lists using RRF.

    score(doc) = Σ_i  1 / (k + rank_i(doc))

    Documents not appearing in a list get no contribution from it.
    """
    scores: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda d: scores[d], reverse=True)


# ---------------------------------------------------------------------------
# 6. CROSS-ENCODER RERANKING
# ---------------------------------------------------------------------------

class CrossEncoderReranker:
    """
    Uses sentence-transformers CrossEncoder.
    Falls back to a trivial identity reranker if not installed.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(model_name)
            self._available = True
        except ImportError:
            print("  [CrossEncoder] sentence-transformers not installed — "
                  "returning input order unchanged.")
            self._available = False

    def rerank(self, query: str, doc_ids: list[int],
               id_to_text: dict[int, str], top_k: int = 5) -> list[int]:
        if not doc_ids:
            return []
        if not self._available:
            return doc_ids[:top_k]
        pairs = [(query, id_to_text[i]) for i in doc_ids]
        scores = self.model.predict(pairs)
        ranked = sorted(range(len(doc_ids)),
                        key=lambda i: scores[i], reverse=True)
        return [doc_ids[i] for i in ranked[:top_k]]


# ---------------------------------------------------------------------------
# 7. MAXIMAL MARGINAL RELEVANCE
# ---------------------------------------------------------------------------

def mmr_select(query_emb: np.ndarray,
               doc_ids: list[int],
               id_to_emb: dict[int, np.ndarray],
               k: int = 5,
               lam: float = 0.5) -> list[int]:
    """
    lam=1.0 → pure relevance (same as sorting by cosine sim)
    lam=0.0 → pure diversity
    lam=0.5 → balanced (default)
    """
    selected: list[int] = []
    remaining = list(doc_ids)

    while len(selected) < k and remaining:
        # Relevance scores
        rel = np.array([id_to_emb[i] @ query_emb for i in remaining])
        # Redundancy scores
        if selected:
            sel_embs = np.stack([id_to_emb[i] for i in selected])
            red = np.array([id_to_emb[i] @ sel_embs.T for i in remaining]).max(axis=1)
        else:
            red = np.zeros(len(remaining))

        mmr_scores = lam * rel - (1 - lam) * red
        best_idx = int(np.argmax(mmr_scores))
        selected.append(remaining.pop(best_idx))

    return selected


# ---------------------------------------------------------------------------
# 8. HyDE — Hypothetical Document Embedding
# ---------------------------------------------------------------------------

def hyde_query_embedding(query: str,
                         vector_index: VectorIndex,
                         api_key: Optional[str] = None) -> np.ndarray:
    """
    Generate a hypothetical answer with Claude, then embed it.
    Falls back to plain query embedding if Anthropic key is not set.
    """
    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        print("  [HyDE] No ANTHROPIC_API_KEY — using raw query embedding.")
        return vector_index.embed(query)

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=key)
        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=150,
            messages=[{
                "role": "user",
                "content": (f"Write one short paragraph (2–3 sentences) "
                            f"that directly answers this question:\n\n{query}")
            }]
        )
        hypothetical_doc = resp.content[0].text.strip()
        print(f"  [HyDE] Hypothetical doc: {hypothetical_doc[:80]}…")
        return vector_index.embed(hypothetical_doc)
    except Exception as exc:
        print(f"  [HyDE] API error ({exc}) — falling back to raw query embedding.")
        return vector_index.embed(query)


# ---------------------------------------------------------------------------
# 9. FULL HYBRID RAG PIPELINE
# ---------------------------------------------------------------------------

def hybrid_rag_pipeline(
    query: str,
    corpus: list[Document],
    metadata_filters: Optional[dict] = None,
    use_hyde: bool = False,
    use_mmr: bool = True,
    rrf_k: int = 60,
    retrieve_top_n: int = 10,
    rerank_top_k: int = 5,
    final_k: int = 3,
    mmr_lambda: float = 0.5,
    verbose: bool = True,
) -> list[Document]:
    """
    Full pipeline:
      metadata filter → HyDE (opt) → BM25 + vector → RRF → cross-encoder → MMR (opt)

    Returns the final list of selected Documents.
    """
    sep = "─" * 60

    # ── Step 1: Metadata filter ──────────────────────────────────────
    if metadata_filters:
        filtered = metadata_filter(corpus, metadata_filters)
        if verbose:
            print(f"\n{sep}")
            print(f"[1] Metadata filter {metadata_filters}")
            print(f"    {len(corpus)} docs → {len(filtered)} docs")
    else:
        filtered = corpus
        if verbose:
            print(f"\n{sep}\n[1] No metadata filter — using full corpus ({len(corpus)} docs)")

    if not filtered:
        print("    No documents matched the filter.")
        return []

    id_to_doc  = {d.id: d for d in filtered}
    id_to_text = {d.id: d.text for d in filtered}

    # ── Step 2: Build BM25 + vector index ───────────────────────────
    if verbose:
        print(f"\n{sep}\n[2] Building indexes over {len(filtered)} documents…")
    bm25  = BM25(filtered)
    vidx  = VectorIndex(filtered)
    id_to_emb = {filtered[i].id: vidx.embeddings[i] for i in range(len(filtered))}

    # ── Step 3: HyDE (optional) ──────────────────────────────────────
    if use_hyde:
        if verbose:
            print(f"\n{sep}\n[3] HyDE — generating hypothetical document…")
        q_emb = hyde_query_embedding(query, vidx)
    else:
        q_emb = vidx.embed(query)
        if verbose:
            print(f"\n{sep}\n[3] HyDE skipped — using raw query embedding")

    # ── Step 4: BM25 retrieval ────────────────────────────────────────
    bm25_ids = bm25.retrieve(query, top_k=retrieve_top_n)
    if verbose:
        print(f"\n{sep}\n[4] BM25 top-{retrieve_top_n}: {bm25_ids}")

    # ── Step 5: Vector retrieval ──────────────────────────────────────
    sims = vidx.embeddings @ q_emb
    # restrict to filtered doc ids
    filtered_indices = [i for i, d in enumerate(filtered)]
    ranked_vec = np.argsort(-sims[filtered_indices])[:retrieve_top_n]
    vec_ids = [filtered[i].id for i in ranked_vec]
    if verbose:
        print(f"\n{sep}\n[5] Vector top-{retrieve_top_n}: {vec_ids}")

    # ── Step 6: RRF fusion ────────────────────────────────────────────
    fused = rrf_fuse(bm25_ids, vec_ids, k=rrf_k)
    if verbose:
        print(f"\n{sep}\n[6] RRF fused (k={rrf_k}): {fused}")

    # ── Step 7: Cross-encoder reranking ──────────────────────────────
    reranker = CrossEncoderReranker()
    reranked = reranker.rerank(query, fused, id_to_text, top_k=rerank_top_k)
    if verbose:
        print(f"\n{sep}\n[7] After cross-encoder reranking (top-{rerank_top_k}): {reranked}")

    # ── Step 8: MMR selection ─────────────────────────────────────────
    if use_mmr and len(reranked) > final_k:
        final_ids = mmr_select(q_emb, reranked, id_to_emb,
                               k=final_k, lam=mmr_lambda)
        if verbose:
            print(f"\n{sep}\n[8] MMR selection (λ={mmr_lambda}, k={final_k}): {final_ids}")
    else:
        final_ids = reranked[:final_k]
        if verbose:
            print(f"\n{sep}\n[8] MMR skipped — taking top-{final_k}: {final_ids}")

    final_docs = [id_to_doc[i] for i in final_ids if i in id_to_doc]

    if verbose:
        print(f"\n{sep}")
        print("FINAL RETRIEVED DOCUMENTS:")
        for rank, doc in enumerate(final_docs, 1):
            snippet = doc.text[:80].replace("\n", " ")
            print(f"  {rank}. [id={doc.id}] {snippet}…")
            print(f"     metadata: {doc.metadata}")
        print(sep)

    return final_docs


# ---------------------------------------------------------------------------
# 10. DEMO QUERIES
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Ph 225 — Retrieval Improvements Demo")
    print("=" * 60)

    # ── Query A: no filters ──────────────────────────────────────────
    print("\n\nQUERY A: 'how does hybrid search combine BM25 and vectors?'")
    hybrid_rag_pipeline(
        query="how does hybrid search combine BM25 and vectors",
        corpus=CORPUS,
        metadata_filters=None,
        use_hyde=False,
        use_mmr=True,
        final_k=3,
    )

    # ── Query B: with metadata filter ───────────────────────────────
    print("\n\nQUERY B: 'what is reranking?' (filter: source=paper, year=2023)")
    hybrid_rag_pipeline(
        query="what is reranking and why is it useful",
        corpus=CORPUS,
        metadata_filters={"source": "paper", "year": 2023},
        use_hyde=False,
        use_mmr=True,
        final_k=3,
    )

    # ── Query C: HyDE enabled ────────────────────────────────────────
    print("\n\nQUERY C: 'explain MMR for diverse results' (HyDE enabled)")
    hybrid_rag_pipeline(
        query="explain how to get diverse results when retrieving documents",
        corpus=CORPUS,
        metadata_filters=None,
        use_hyde=True,   # requires ANTHROPIC_API_KEY
        use_mmr=True,
        final_k=3,
    )

    # ── Standalone MMR demonstration ─────────────────────────────────
    print("\n\nSTANDALONE MMR DEMO — λ sweep")
    print("Same candidates, different λ values show relevance vs diversity trade-off\n")
    vidx = VectorIndex(CORPUS)
    q_emb = vidx.embed("retrieval augmented generation")
    id_to_emb = {CORPUS[i].id: vidx.embeddings[i] for i in range(len(CORPUS))}
    candidates = list(range(len(CORPUS)))  # all doc ids

    for lam in [1.0, 0.7, 0.5, 0.3, 0.0]:
        selected = mmr_select(q_emb, candidates, id_to_emb, k=3, lam=lam)
        print(f"  λ={lam:.1f} → docs {selected}: "
              + " | ".join(CORPUS[i].text[:40] for i in selected))

    print("\nDone. Set ANTHROPIC_API_KEY to enable HyDE.\n")
