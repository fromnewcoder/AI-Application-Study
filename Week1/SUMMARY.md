# Week 1 Summary: Tokens & Context Windows

## Overview

Week 1 focuses on the fundamentals of **tokens** and **context windows** — the core resource constraints in working with LLMs.

---

## Projects

### 1. `tokens-context-study/` — Interactive Study App (React + Vite)

A self-contained 60-minute interactive learning app with 4 timed sections:

| # | Section | Duration | Topic |
|---|---------|----------|-------|
| 1 | Tokenisation | 15 min | What tokens are, BPE algorithm, token vs word |
| 2 | Install tiktoken | 10 min | OpenAI's Rust-based tokeniser, encoding families |
| 3 | Count Tokens | 15 min | Programmatic token counting, chat overhead, chunking |
| 4 | Context Windows | 20 min | Context sizes, lost-in-the-middle, architecture patterns |

**Key features:**
- Live token visualiser (type text → see token breakdown)
- Expandable code snippets with copy button
- Architecture decision tree diagram
- Multiple-choice quizzes per section
- Section progress tracker with timer

**To run:**
```bash
cd Week1/tokens-context-study
npm install
npm run dev
```

---

### 2. `OpenAIDemo.py` — OpenAI API Demo

Basic OpenAI SDK demo calling `gpt-3.5-turbo` with a simple prompt.

```bash
pip install openai
# Set OPENAI_API_KEY env var
python Week1/OpenAIDemo.py
```

---

### 3. `DeepSeekAPIDemo.py` — DeepSeek API Demo

Calls `deepseek-chat` model via DeepSeek's API endpoint using the OpenAI SDK.

```bash
pip install openai
# Set DEEPSEEK_API_KEY env var
python Week1/DeepSeekAPIDemo.py
```

---

## Core Concepts Covered

### Tokenisation
- Tokens ≠ words; ~4 characters = 1 token in English; 1 word ≈ 1.3 tokens
- BPE (Byte Pair Encoding) merges frequent character pairs into subword tokens
- Rare words, code, and non-English text tokenise into more tokens

### tiktoken
- OpenAI's fast Rust-based tokeniser (Python bindings available)
- Encoding families: `cl100k_base` (GPT-3.5/4), `o200k_base` (GPT-4o)
- `enc.encode(text)` → list of token IDs; `enc.decode(tokens)` → original text

### Token Counting in Production
- Count **before** every API call to guard against context window errors
- Chat messages have ~4 token overhead per message (role labels, separators)
- Chunk long documents at ~80% of context limit; decode token boundaries, not characters
- Cache the encoding object — don't recreate per call

### Context Windows
- GPT-3.5-turbo: 16K tokens | GPT-4: 128K | Claude 3 Opus: 200K | Gemini 1.5 Pro: 1M
- **Lost-in-the-Middle**: LLMs attend best to content at start/end of context
- Architecture patterns: direct send → long-context model → RAG/chunking → summarisation
