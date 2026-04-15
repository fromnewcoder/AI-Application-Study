"""
Research Assistant — Python Backend Agent
==========================================
Agentic RAG pipeline: web search → fetch 3 docs → synthesise with citations.

Install:
    pip install anthropic httpx beautifulsoup4 rich

Optional (real web search):
    pip install tavily-python        # set TAVILY_API_KEY
    # or
    pip install requests             # use Brave Search API (BRAVE_API_KEY)

Run:
    python research_agent.py "quantum computing"
    python research_agent.py --eval          # run eval suite
    python research_agent.py --interactive   # REPL mode
"""

import os
import sys
import json
import time
import asyncio
import textwrap
import re
from dataclasses import dataclass, field
from typing import Optional
import argparse

import anthropic

# ── Optional deps ──────────────────────────────────────────────────────────────
try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.markdown import Markdown
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich import print as rprint
    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    console = None

# ── Colour fallback (no rich) ──────────────────────────────────────────────────
def _c(code, text): return f"\033[{code}m{text}\033[0m"
def green(t): return _c(32, t)
def amber(t): return _c(33, t)
def red(t):   return _c(31, t)
def cyan(t):  return _c(36, t)
def bold(t):  return _c(1, t)
def dim(t):   return _c(2, t)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. SEARCH BACKENDS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    domain: str = ""

    def __post_init__(self):
        if not self.domain and self.url:
            try:
                from urllib.parse import urlparse
                self.domain = urlparse(self.url).netloc.replace("www.", "")
            except Exception:
                pass


def search_tavily(query: str, max_results: int = 5) -> list[SearchResult]:
    """Use Tavily Search API (recommended for production)."""
    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
        resp = client.search(query=query, max_results=max_results)
        return [
            SearchResult(
                title=r.get("title", ""),
                url=r.get("url", ""),
                snippet=r.get("content", "")[:300],
            )
            for r in resp.get("results", [])
        ]
    except Exception as e:
        print(amber(f"  Tavily error: {e} — falling back to mock"))
        return []


def search_mock(query: str, max_results: int = 5) -> list[SearchResult]:
    """Simulated search for demo / offline use."""
    base = [
        SearchResult(
            title=f"{query} — Overview and Key Concepts",
            url=f"https://en.wikipedia.org/wiki/{query.replace(' ', '_')}",
            snippet=(
                f"{query} is a significant area of study with wide-ranging applications. "
                "Researchers have identified core principles, mechanisms, and real-world "
                "implications. Recent work has expanded understanding of edge cases and "
                "practical deployment considerations."
            ),
        ),
        SearchResult(
            title=f"Recent Advances in {query}",
            url=f"https://www.nature.com/search?q={query.replace(' ', '+')}",
            snippet=(
                f"A peer-reviewed survey of the {query} literature from 2022–2024. "
                "The study synthesises findings from 140 primary sources. "
                "Key results show significant progress, though challenges remain in "
                "scalability, interpretability, and generalisation to novel domains."
            ),
        ),
        SearchResult(
            title=f"Practical Guide: {query}",
            url=f"https://arxiv.org/search/?query={query.replace(' ', '+')}",
            snippet=(
                f"A practitioner-oriented review of {query}. Covers state-of-the-art "
                "methods, implementation considerations, evaluation benchmarks, and "
                "open problems. Includes comparative analysis of leading approaches "
                "with reproducible benchmarks and ablation studies."
            ),
        ),
    ]
    # Topic-aware enrichment
    q = query.lower()
    if "climate" in q or "carbon" in q:
        base[0].snippet = (
            "Climate change refers to long-term shifts in global temperatures driven "
            "primarily by anthropogenic greenhouse gas emissions since industrialisation. "
            "The IPCC Sixth Assessment Report concludes that limiting warming to 1.5°C "
            "requires immediate, deep emissions reductions across all sectors."
        )
        base[1].snippet = (
            "Carbon capture and storage (CCS) has seen cost reductions of ~60% since 2015. "
            "Direct air capture (DAC) costs stood at ~$250/tonne CO₂ in 2024, down from $600 "
            "in 2020. Global installed capacity remains <1 MtCO₂/yr — a fraction of the "
            "1–10 GtCO₂/yr needed by 2050 under most 1.5°C scenarios."
        )
    if "rag" in q or "retrieval" in q or "language model" in q:
        base[0].snippet = (
            "Retrieval-Augmented Generation (RAG) combines neural language models with "
            "document retrieval to reduce hallucination and enable knowledge-intensive tasks. "
            "Lewis et al. (2020) introduced the foundational architecture; subsequent work "
            "has added hybrid search, reranking, and iterative retrieval."
        )
        base[1].snippet = (
            "Hybrid retrieval (BM25 + dense vectors) consistently outperforms either approach "
            "alone on BEIR and MS MARCO benchmarks, with relative MRR@10 gains of 5–15%. "
            "Cross-encoder reranking adds a further 3–8% at the cost of additional latency. "
            "Production systems balance quality against p95 latency budgets."
        )
    return base[:max_results]


def search_web(query: str, max_results: int = 5) -> list[SearchResult]:
    """Route to best available search backend."""
    if os.environ.get("TAVILY_API_KEY") and HTTPX_AVAILABLE:
        results = search_tavily(query, max_results)
        if results:
            return results
    return search_mock(query, max_results)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. DOCUMENT FETCHER
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class FetchedDocument:
    url: str
    title: str
    content: str
    word_count: int = 0

    def __post_init__(self):
        self.word_count = len(self.content.split())


def fetch_document(url: str, title: str) -> FetchedDocument:
    """Fetch and extract text from a URL. Falls back to mock if unavailable."""
    if HTTPX_AVAILABLE:
        try:
            with httpx.Client(timeout=8, follow_redirects=True) as client:
                resp = client.get(url, headers={"User-Agent": "Mozilla/5.0 ResearchBot/1.0"})
                resp.raise_for_status()
                if BS4_AVAILABLE:
                    soup = BeautifulSoup(resp.text, "html.parser")
                    for tag in soup(["script", "style", "nav", "footer", "header"]):
                        tag.decompose()
                    text = " ".join(soup.get_text(" ", strip=True).split())
                    return FetchedDocument(url=url, title=title, content=text[:3000])
                return FetchedDocument(url=url, title=title, content=resp.text[:3000])
        except Exception:
            pass  # fall through to mock

    # Mock content for offline/demo use
    mock = (
        f"[Retrieved content for: {title}]\n\n"
        f"This document from {url} provides detailed coverage of the research topic. "
        "Key findings include: (1) substantial evidence for the primary hypothesis, "
        "(2) methodological advances enabling new experimental approaches, "
        "(3) open questions motivating further research. "
        "The work builds on prior literature and has been cited extensively. "
        "Practical implications are discussed for both academic and industry contexts. "
        "Limitations include sample size, geographic scope, and temporal coverage. "
        "Future directions include multi-modal extensions and cross-domain validation."
    )
    return FetchedDocument(url=url, title=title, content=mock)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. TOOL DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════

TOOLS = [
    {
        "name": "search_web",
        "description": (
            "Search the web for information on a topic. "
            "Returns up to 5 results with titles, URLs, and snippets. "
            "Call this first to discover relevant sources."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query. Be specific. Include key terms.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Number of results to return (1–5). Default 5.",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "fetch_document",
        "description": (
            "Fetch and read the full text of a specific URL. "
            "Call this on the 3 most relevant URLs from search results. "
            "Returns the document text for citation in your synthesis."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "The URL to fetch"},
                "title": {"type": "string", "description": "Document title for citation"},
            },
            "required": ["url", "title"],
        },
    },
]

SYSTEM_PROMPT = """You are a rigorous research assistant. Your goal is to produce \
authoritative, well-cited answers on any topic.

MANDATORY WORKFLOW:
1. Call search_web to find relevant sources for the topic.
2. Select the 3 most relevant results and call fetch_document on each one.
3. After retrieving all 3 documents, write your synthesis.

SYNTHESIS REQUIREMENTS:
- Use inline citations [1], [2], [3] tied to the order you fetched documents.
- Structure: opening summary → key findings (bullet list) → nuances/caveats → conclusion.
- End with a "## Sources" section listing each source as: [N] Title — URL
- Be precise. Do not fabricate statistics. Note genuine uncertainty.
- Minimum length: 200 words in the synthesis body.

You MUST retrieve exactly 3 documents before synthesising. Do not skip this step."""


# ═══════════════════════════════════════════════════════════════════════════════
# 4. AGENT RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class AgentResult:
    topic: str
    synthesis: str
    sources: list[dict] = field(default_factory=list)
    steps: list[dict] = field(default_factory=list)
    latency_ms: float = 0.0
    error: Optional[str] = None

    @property
    def success(self): return self.error is None and bool(self.synthesis)


def _log(step_type: str, label: str, detail: str = "", verbose: bool = True):
    if not verbose:
        return
    icons = {"search": "[S]", "fetch": "[F]", "think": "[T]", "done": "[+]", "error": "[X]"}
    colors_fn = {"search": amber, "fetch": cyan, "think": dim, "done": green, "error": red}
    icon = icons.get(step_type, "·")
    color = colors_fn.get(step_type, lambda x: x)
    detail_str = f"  {dim(detail[:60])}" if detail else ""
    print(f"  {color(icon)} {label}{detail_str}")


def run_agent(topic: str, verbose: bool = True) -> AgentResult:
    """
    Agentic loop:
      search → fetch 3 docs → synthesise with citations
    """
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
    messages = [{"role": "user", "content": f"Research this topic thoroughly and synthesise a cited answer: {topic}"}]
    sources: list[dict] = []
    steps: list[dict] = []
    synthesis = ""
    t0 = time.perf_counter()

    def step(stype, label, detail=""):
        steps.append({"type": stype, "label": label, "detail": detail})
        _log(stype, label, detail, verbose)

    try:
        for turn in range(12):
            response = client.messages.create(
                model="MiniMax-M2.7",
                max_tokens=2048,
                system=SYSTEM_PROMPT,
                tools=TOOLS,
                messages=messages,
            )

            # Collect text and tool calls
            tool_calls = []
            for block in response.content:
                if block.type == "text" and block.text.strip():
                    synthesis = block.text
                    snippet = block.text[:50].replace("\n", " ")
                    step("think", "Generating synthesis...", snippet)
                if block.type == "tool_use":
                    tool_calls.append(block)

            if response.stop_reason == "end_turn":
                break
            if not tool_calls:
                break

            # Execute tool calls
            tool_results = []
            for tc in tool_calls:
                if tc.name == "search_web":
                    query = tc.input.get("query", topic)
                    max_r = tc.input.get("max_results", 5)
                    step("search", f"Searching: {query}", query)
                    results = search_web(query, max_r)
                    result_json = json.dumps([
                        {"title": r.title, "url": r.url, "snippet": r.snippet}
                        for r in results
                    ])
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tc.id,
                        "content": result_json,
                    })

                elif tc.name == "fetch_document":
                    url = tc.input.get("url", "")
                    title = tc.input.get("title", url)
                    step("fetch", f"Fetching document [{len(sources)+1}]", title)
                    doc = fetch_document(url, title)
                    sources.append({
                        "index": len(sources) + 1,
                        "title": title,
                        "url": url,
                        "domain": doc.url,
                        "word_count": doc.word_count,
                    })
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tc.id,
                        "content": doc.content,
                    })

            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})

        step("done", f"Complete — {len(sources)} sources, {len(synthesis.split())} words")
        return AgentResult(
            topic=topic,
            synthesis=synthesis,
            sources=sources,
            steps=steps,
            latency_ms=(time.perf_counter() - t0) * 1000,
        )

    except Exception as e:
        step("error", str(e))
        return AgentResult(
            topic=topic,
            synthesis=synthesis,
            sources=sources,
            steps=steps,
            latency_ms=(time.perf_counter() - t0) * 1000,
            error=str(e),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 5. EVAL SUITE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class EvalCheck:
    id: str
    description: str
    fn: object  # callable(str) -> bool
    weight: float = 1.0


@dataclass
class EvalCase:
    id: str
    topic: str
    checks: list[EvalCheck]


@dataclass
class EvalResult:
    case_id: str
    topic: str
    passed: int
    total: int
    details: list[dict]
    agent_result: AgentResult

    @property
    def pct(self): return round(self.passed / self.total * 100) if self.total else 0
    @property
    def grade(self): return "PASS" if self.pct >= 80 else "FAIL"


def make_eval_suite() -> list[EvalCase]:
    return [
        EvalCase(
            id="E01",
            topic="quantum computing",
            checks=[
                EvalCheck("c01", "Has ≥1 inline citation [N]",
                          lambda r: bool(re.search(r'\[\d+\]', r))),
                EvalCheck("c02", "Has ## Sources section",
                          lambda r: bool(re.search(r'##\s*sources', r, re.I))),
                EvalCheck("c03", "Response ≥ 150 words",
                          lambda r: len(r.split()) >= 150),
                EvalCheck("c04", "Contains 'qubit' or 'superposition'",
                          lambda r: bool(re.search(r'qubit|superposition|entanglement', r, re.I))),
                EvalCheck("c05", "No uncited numeric stats (all % have nearby citation)",
                          lambda r: _check_no_uncited_stats(r)),
            ],
        ),
        EvalCase(
            id="E02",
            topic="climate change and carbon capture",
            checks=[
                EvalCheck("c01", "Has ≥1 inline citation [N]",
                          lambda r: bool(re.search(r'\[\d+\]', r))),
                EvalCheck("c02", "Has ## Sources section",
                          lambda r: bool(re.search(r'##\s*sources', r, re.I))),
                EvalCheck("c03", "Response ≥ 200 words",
                          lambda r: len(r.split()) >= 200),
                EvalCheck("c04", "Mentions climate/emissions term",
                          lambda r: bool(re.search(r'emission|carbon|temperatur|greenhouse|warming', r, re.I))),
                EvalCheck("c05", "Includes caveat/limitation",
                          lambda r: bool(re.search(r'however|although|challenge|uncertain|limit|caveat|despite', r, re.I))),
                EvalCheck("c06", "Structured with bullets or headings",
                          lambda r: bool(re.search(r'##|^\s*[-•*]', r, re.M))),
            ],
        ),
        EvalCase(
            id="E03",
            topic="retrieval augmented generation",
            checks=[
                EvalCheck("c01", "Has ≥2 inline citations",
                          lambda r: len(re.findall(r'\[\d+\]', r)) >= 2),
                EvalCheck("c02", "Has ## Sources section",
                          lambda r: bool(re.search(r'##\s*sources', r, re.I))),
                EvalCheck("c03", "Response ≥ 150 words",
                          lambda r: len(r.split()) >= 150),
                EvalCheck("c04", "Mentions retrieval/embedding",
                          lambda r: bool(re.search(r'retrieval|embed|vector|BM25|semantic', r, re.I))),
                EvalCheck("c05", "Structured output (bullets/headings)",
                          lambda r: bool(re.search(r'##|^\s*[-•*]', r, re.M))),
            ],
        ),
        EvalCase(
            id="E04",
            topic="CRISPR gene editing",
            checks=[
                EvalCheck("c01", "Has ≥1 inline citation",
                          lambda r: bool(re.search(r'\[\d+\]', r))),
                EvalCheck("c02", "Has ## Sources section",
                          lambda r: bool(re.search(r'##\s*sources', r, re.I))),
                EvalCheck("c03", "Response ≥ 150 words",
                          lambda r: len(r.split()) >= 150),
                EvalCheck("c04", "Mentions CRISPR or Cas9",
                          lambda r: bool(re.search(r'CRISPR|Cas9|genome|gene', r, re.I))),
                EvalCheck("c05", "Mentions ethics or limitations",
                          lambda r: bool(re.search(r'ethic|limit|off.target|concern|risk', r, re.I))),
            ],
        ),
    ]


def _check_no_uncited_stats(text: str) -> bool:
    """Pass if every sentence containing % also contains a citation [N]."""
    sentences = re.split(r'[.!?]', text)
    for sent in sentences:
        if '%' in sent and not re.search(r'\[\d+\]', sent):
            return False
    return True


def run_eval_suite(
    cases: list[EvalCase],
    verbose: bool = True,
    stop_on_fail: bool = False,
) -> tuple[list[EvalResult], int]:
    """
    Run all eval cases and return results + overall percentage.
    Threshold for PASS: 80% of all checks.
    """
    results: list[EvalResult] = []
    all_passed = 0
    all_total = 0

    sep = "-" * 62
    if verbose:
        print(f"\n{bold('Eval Suite')}")
        print(f"{dim(f'{len(cases)} cases · threshold: 80%')}\n")

    for case in cases:
        if verbose:
            print(f"{cyan('-->')} {bold(case.id)}: {case.topic}")

        agent_result = run_agent(case.topic, verbose=verbose)
        text = agent_result.synthesis

        details = []
        case_passed = 0
        for check in case.checks:
            try:
                ok = bool(check.fn(text))
            except Exception:
                ok = False
            details.append({"id": check.id, "desc": check.description, "passed": ok})
            if ok:
                case_passed += 1
            if verbose:
                mark = green("[PASS]") if ok else red("[FAIL]")
                print(f"    {mark} {check.description}")

        case_total = len(case.checks)
        all_passed += case_passed
        all_total += case_total
        pct = round(case_passed / case_total * 100)
        grade_str = green("PASS") if pct >= 80 else red("FAIL")

        if verbose:
            print(f"  {grade_str}  {pct}% ({case_passed}/{case_total})\n")

        result = EvalResult(
            case_id=case.id,
            topic=case.topic,
            passed=case_passed,
            total=case_total,
            details=details,
            agent_result=agent_result,
        )
        results.append(result)

        if stop_on_fail and pct < 80:
            if verbose:
                print(red("  Stopping on first failure (--stop-on-fail)."))
            break

    overall = round(all_passed / all_total * 100) if all_total else 0

    if verbose:
        print(sep)
        suite_grade = bold(green("SUITE PASSED")) if overall >= 80 else bold(red("SUITE FAILED"))
        print(f"  {suite_grade}  {overall}%  ({all_passed}/{all_total} checks)")
        print(sep)

        # Summary table
        print(f"\n  {'Case':<6}  {'Topic':<38}  {'Score':>6}  {'Grade'}")
        print(f"  {'─'*6}  {'─'*38}  {'─'*6}  {'─'*5}")
        for r in results:
            grade = green("PASS") if r.pct >= 80 else red("FAIL")
            print(f"  {r.case_id:<6}  {r.topic[:38]:<38}  {r.pct:>5}%  {grade}")
        print()

    return results, overall


# ═══════════════════════════════════════════════════════════════════════════════
# 6. CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def print_result(result: AgentResult):
    """Pretty-print a completed research result."""
    sep = "=" * 62
    print(f"\n{bold(sep)}")
    print(f"  {bold('Research:')} {result.topic}")
    print(f"  {dim(f'{len(result.sources)} sources · {len(result.synthesis.split())} words · {result.latency_ms:.0f}ms')}")
    print(f"{bold(sep)}\n")

    if result.error:
        print(red(f"Error: {result.error}\n"))
        return

    # Wrap synthesis for terminal
    for line in result.synthesis.split("\n"):
        if line.startswith("## "):
            print(f"\n{bold(line)}")
        elif line.startswith("- ") or line.startswith("• "):
            print(f"  {cyan('-')} {line[2:]}")
        elif line.strip():
            print(textwrap.fill(line, width=70, subsequent_indent="  "))
        else:
            print()


def interactive_repl():
    """Simple REPL for interactive research sessions."""
    print(bold("\n Research Assistant  (type 'exit' to quit, 'eval' to run eval suite)\n"))
    cases = make_eval_suite()
    while True:
        try:
            topic = input(cyan("Topic > ")).strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye.")
            break
        if not topic:
            continue
        if topic.lower() == "exit":
            break
        if topic.lower() == "eval":
            results, pct = run_eval_suite(cases)
            continue
        result = run_agent(topic, verbose=True)
        print_result(result)


def main():
    parser = argparse.ArgumentParser(description="Research Assistant Agent")
    parser.add_argument("topic", nargs="?", help="Topic to research")
    parser.add_argument("--eval", action="store_true", help="Run eval suite")
    parser.add_argument("--interactive", "-i", action="store_true", help="REPL mode")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress agent trace")
    parser.add_argument("--stop-on-fail", action="store_true", help="Halt eval on first failure")
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(red("Error: ANTHROPIC_API_KEY not set."))
        print("  export ANTHROPIC_API_KEY=sk-ant-...")
        sys.exit(1)

    if args.interactive:
        interactive_repl()
        return

    if args.eval:
        cases = make_eval_suite()
        results, pct = run_eval_suite(cases, verbose=not args.quiet, stop_on_fail=args.stop_on_fail)
        sys.exit(0 if pct >= 80 else 1)

    if args.topic:
        result = run_agent(args.topic, verbose=not args.quiet)
        print_result(result)
        sys.exit(0 if result.success else 1)

    parser.print_help()


if __name__ == "__main__":
    main()
