# Research Assistant — Week 5 Build Day

Agentic RAG pipeline: topic → web search → 3 docs retrieved → synthesised answer with inline citations.

## Project structure

```
research-assistant/
├── ResearchAssistant.jsx   ← React app (Claude-in-Claude, runs in browser)
├── research_agent.py       ← Python backend agent + CLI + eval runner
├── test_eval_suite.py      ← pytest eval suite (target: ≥80%)
└── README.md
```

---

## Quick start

### React app (browser, no install)
Drop `ResearchAssistant.jsx` into any Claude artifact or Vite/CRA project.
The app calls the Anthropic API directly from the browser using the artifact's
built-in API proxy.

### Python agent
```bash
pip install anthropic httpx beautifulsoup4 rich
export ANTHROPIC_API_KEY=sk-ant-...

# Single query
python research_agent.py "quantum computing"

# Interactive REPL
python research_agent.py --interactive

# Run eval suite (exit code 0 = pass, 1 = fail)
python research_agent.py --eval

# Quiet mode (suppress trace)
python research_agent.py "climate change" --quiet
```

### Optional: real web search (recommended for production)
```bash
pip install tavily-python
export TAVILY_API_KEY=tvly-...
# Agent automatically uses Tavily when the key is present
```

---

## Architecture

```
User query
    │
    ▼
┌─────────────────────────────────────────────┐
│  Claude (claude-sonnet-4)                   │
│  + search_web tool                          │
│  + fetch_document tool                      │
│                                             │
│  Agentic loop:                              │
│  1. search_web(query) → 5 results          │
│  2. fetch_document(url1) → doc text        │
│  3. fetch_document(url2) → doc text        │
│  4. fetch_document(url3) → doc text        │
│  5. Synthesise with [1][2][3] citations    │
└─────────────────────────────────────────────┘
    │
    ▼
Structured answer:
  - Opening summary
  - Key findings (bullets)
  - Caveats / nuances
  - ## Sources
```

---

## Eval suite

4 test cases × 5–6 checks = 21 total checks.
Suite passes when ≥ 80% of all checks pass.

| Case | Topic                          | Checks |
|------|--------------------------------|--------|
| E01  | Quantum computing              | 5      |
| E02  | Climate change / carbon capture| 6      |
| E03  | Retrieval augmented generation | 5      |
| E04  | CRISPR gene editing            | 5      |

**Check categories:**
- Structural: `## Sources` section, inline `[N]` citations
- Length: minimum word counts (150–200 words)
- Content: domain-specific terminology present
- Quality: caveats/limitations acknowledged, no uncited statistics
- Format: bullets or headings present

### Run with pytest
```bash
pip install pytest
pytest test_eval_suite.py -v
```

### Suite-level gate
`TestSuiteOverall::test_overall_suite_passes_80_percent` aggregates all 21 checks
and asserts the overall percentage ≥ 80%. This is the CI gate check.

---

## Extending

**Add a new eval case:**
```python
# research_agent.py  →  make_eval_suite()
EvalCase(
    id="E05",
    topic="mRNA vaccines",
    checks=[
        EvalCheck("c01", "Has citations", lambda r: bool(re.search(r'\[\d+\]', r))),
        EvalCheck("c02", "Sources section", lambda r: bool(re.search(r'##\s*sources', r, re.I))),
        EvalCheck("c03", "≥150 words", lambda r: len(r.split()) >= 150),
        EvalCheck("c04", "Mentions mRNA", lambda r: bool(re.search(r'mRNA|vaccine|lipid', r, re.I))),
    ],
)
```

**Swap search backend:**
```python
# Set TAVILY_API_KEY for real web search
export TAVILY_API_KEY=tvly-...
```

**Change LLM:**
```python
# research_agent.py  →  run_agent()
model="claude-opus-4-6"   # more capable, slower
model="claude-haiku-4-5-20251001"  # faster, cheaper
```

---

## Design decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Retrieval count | 3 docs | Balances context window budget vs coverage |
| Citations | `[N]` inline | Easy to verify, standard academic style |
| Search | Tavily / mock | Tavily for production, mock for CI |
| Eval threshold | 80% | Allows 1–2 check failures per case |
| Model | claude-sonnet-4 | Best quality/speed tradeoff for synthesis |
