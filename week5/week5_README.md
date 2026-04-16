# Ph 224/225 — LLM Engineering

Personal notes, code, and projects from the Phase 2 course.

---

## Week 5 — Evals, Memory, Retrieval, and Agents

### What was built

| Project | Description | Files |
|---------|-------------|-------|
| **Research Assistant** | Agentic RAG: topic → web search → 3 docs → cited synthesis | `research_agent.py`, `ResearchAssistant.jsx` |
| **Eval suite** | 4 cases × 5–6 checks, pytest, CI gate at ≥ 80% | `test_eval_suite.py` |
| **LlamaIndex RAG lab** | Index types, query engines, routers, vs LangChain | `llamaindex_rag_lab.py` |
| **RAG retrieval demo** | BM25 + vector hybrid, RRF, cross-encoder reranking, HyDE, MMR | `rag_retrieval_demo.py` |
| **Memory chatbot** | Summarisation memory with rolling compression | `summarisation_memory.py` |

---

### Week 5 concepts

#### Evals
- Structural checks (regex, counts) vs semantic checks (LLM judge, heuristics)
- Threshold as a design choice: 80% means 1–2 failures per case are acceptable
- Eval-first development: write checks before seeing model output
- Separate retrieval evals from generation evals — different failure modes

#### Memory patterns
- **In-context**: exact recall, O(n) token cost — fine for short sessions
- **Summarisation**: ~50% token reduction, rolling compression — best default
- **Entity**: structured fact store, engineering overhead — for personal assistants
- **External (Redis/vector)**: unlimited history, infra cost — production standard

#### Retrieval improvements
- **Hybrid search** (BM25 + vector, RRF): 5–15% better than either alone
- **Cross-encoder reranking**: run on top-50, return top-5, adds 3–8% precision
- **HyDE**: generate hypothetical answer, embed that — bridges query/answer gap
- **MMR**: prevents near-duplicate chunks, λ=0.5 balances relevance/diversity
- **Metadata filtering**: apply before retrieval, reduces candidate set dramatically

#### LlamaIndex
- Five index types: `VectorStoreIndex`, `SummaryIndex`, `KeywordTableIndex`, `KnowledgeGraphIndex`, `TreeIndex`
- `RouterQueryEngine` dispatches queries to the right index — built-in intent routing
- `QueryFusionRetriever` with `mode="reciprocal_rerank"` — RRF in 5 lines
- LlamaIndex vs LangChain: RAG-first design vs broader ecosystem/observability

---

### Running the eval suite

```bash
pip install anthropic httpx beautifulsoup4 pytest
export ANTHROPIC_API_KEY=sk-ant-...

# Run all evals
pytest test_eval_suite.py -v

# Single case
pytest test_eval_suite.py -v -k E01

# CI gate (exit 0 = pass, 1 = fail)
pytest test_eval_suite.py -q
echo "Exit: $?"
```

Expected: **≥ 17/21 checks passing** (≥ 80%).

---

### Research assistant quick start

```bash
# Single query
python research_agent.py "quantum computing"

# Interactive REPL
python research_agent.py --interactive

# Run built-in eval
python research_agent.py --eval
```

---

### Eval philosophy

See [`eval_philosophy.docx`](./eval_philosophy.docx) for the full write-up. Key principles:

1. Test behaviour, not vibes — every check must be falsifiable
2. Separate retrieval evals from generation evals
3. The threshold is a contract — document why you chose it
4. One regression test per production bug
5. LLM-as-judge is a tool, not an oracle — calibrate against human labels
6. Write evals before seeing model output

---

### File structure

```
week5/
├── research_agent.py          # Python backend agent + CLI + eval runner
├── test_eval_suite.py         # pytest eval suite (target ≥80%)
├── ResearchAssistant.jsx      # React app (Claude-in-Claude)
├── llamaindex_rag_lab.py      # LlamaIndex deep dive lab
├── rag_retrieval_demo.py      # Hybrid search + reranking demo
├── eval_philosophy.docx       # Eval philosophy document
└── README.md                  # This file
```

---

### Week 5 commit history

```
feat: add research assistant agent with agentic RAG loop
feat: add pytest eval suite (4 cases, 21 checks, ≥80% gate)
feat: add LlamaIndex lab — VectorStoreIndex, SummaryIndex, RouterQueryEngine
feat: add hybrid retrieval demo (BM25 + vector + RRF + cross-encoder)
feat: add summarisation memory chatbot (rolling compression, configurable threshold)
docs: add eval philosophy document
docs: update README with week 5 summary
```

---

*Ph 224/225 · Week 5 · April 2026*
