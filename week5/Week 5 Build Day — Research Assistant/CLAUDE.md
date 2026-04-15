# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Agentic RAG pipeline: topic → web search → 3 docs retrieved → synthesised answer with inline citations. Two entry points:
- **React app** (`frontend/ResearchAssistant.jsx`): Claude-in-Claude running in browser, calls Anthropic API directly
- **Python agent** (`backend/research_agent.py`): CLI tool with REPL and eval runner

## Commands

### Python agent
```bash
# Single query
python research_agent.py "quantum computing"

# Interactive REPL
python research_agent.py --interactive

# Run eval suite (exit 0 = pass, 1 = fail at <80%)
python research_agent.py --eval

# Quiet mode
python research_agent.py "climate change" --quiet
```

### Eval suite (pytest)
```bash
pytest test_eval_suite.py -v
pytest test_eval_suite.py -v -k E01       # single case
pytest test_eval_suite.py -v --tb=short   # compact output
```

### Dependencies
```bash
pip install anthropic httpx beautifulsoup4 rich pytest
# Optional (real web search): pip install tavily-python && export TAVILY_API_KEY=tvly-...
```

## Architecture

```
User query → search_web (5 results) → fetch_document × 3 → synthesise [1][2][3]
```

**Agent loop** (up to 12 turns in Python, 10 in JS):
1. `search_web(query)` → 5 results with titles, URLs, snippets
2. Select 3 most relevant, `fetch_document(url, title)` for each
3. Synthesise answer with `[N]` citations, `## Sources` section

**Search backends** (auto-selected):
- `TAVILY_API_KEY` set + `httpx` available → Tavily real search
- Otherwise → `search_mock()` with topic-aware snippet enrichment

**Model**: `claude-sonnet-4-20250514` (both frontend and backend)

## Key Implementation Notes

- **Frontend agent** (`ResearchAssistant.jsx`) is a separate implementation from the Python agent — both exist and work independently
- **Tool definitions** are duplicated between `research_agent.py:TOOLS` and `ResearchAssistant.jsx:AGENT_TOOLS`
- The Python agent imports `AgentResult`, `run_agent`, `make_eval_suite` from `research_agent.py`; test_eval_suite.py is a pytest wrapper
- **Eval suite**: 4 cases (E01–E04) × 5–6 checks = 21 total checks; suite passes at ≥80%
- Add new eval cases in `research_agent.py:make_eval_suite()` using `EvalCase(id, topic, checks)`
- **Mock fallback**: if `httpx`/`bs4` unavailable or fetch fails, returns mock content so the agent still produces a response

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Retrieval count | 3 docs | Context window budget vs coverage |
| Citations | `[N]` inline | Verifiable, academic style |
| Search | Tavily / mock | Tavily when key present, mock otherwise |
| Eval threshold | 80% | Allows 1–2 check failures per case |
| Model | claude-sonnet-4-20250514 | Quality/speed tradeoff |
