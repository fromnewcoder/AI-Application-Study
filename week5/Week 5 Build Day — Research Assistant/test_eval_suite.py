"""
Research Assistant — Eval Suite (pytest)
==========================================
Runs all 4 eval cases, scores every check, and asserts overall >= 80%.

Run:
    pytest test_eval_suite.py -v
    pytest test_eval_suite.py -v --tb=short   # compact failures
    pytest test_eval_suite.py -v -k E01       # single case
    pytest test_eval_suite.py --eval-only      # skip if no API key

Requires:
    pip install pytest anthropic
    export ANTHROPIC_API_KEY=sk-ant-...
"""

import os
import re
import sys
import json
import time
import pytest
from dataclasses import dataclass, field
from typing import Callable

# Import the agent from the sibling module
sys.path.insert(0, os.path.dirname(__file__))
try:
    from research_agent import run_agent, AgentResult
except ImportError:
    pytest.skip("research_agent.py not found", allow_module_level=True)

# ── Skip entire module if no API key ──────────────────────────────────────────
if not os.environ.get("ANTHROPIC_API_KEY"):
    pytest.skip("ANTHROPIC_API_KEY not set", allow_module_level=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def has_citations(text: str, min_count: int = 1) -> bool:
    """Response contains at least `min_count` inline [N] citations."""
    return len(re.findall(r'\[\d+\]', text)) >= min_count


def has_sources_section(text: str) -> bool:
    """Response ends with a ## Sources section."""
    return bool(re.search(r'##\s*sources', text, re.I))


def word_count(text: str) -> int:
    return len(text.split())


def has_structure(text: str) -> bool:
    """Response contains headings or bullet lists."""
    return bool(re.search(r'##|^\s*[-•*]', text, re.M))


def has_caveat(text: str) -> bool:
    """Response acknowledges limitation or uncertainty."""
    return bool(re.search(
        r'however|although|challenge|uncertain|limit|caveat|despite|drawback|not|yet',
        text, re.I
    ))


def no_uncited_percentages(text: str) -> bool:
    """Every sentence containing % also has a citation [N]."""
    for sent in re.split(r'[.!?]', text):
        if '%' in sent and not re.search(r'\[\d+\]', sent):
            return False
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures — run each agent once, cache result
# ═══════════════════════════════════════════════════════════════════════════════

_cache: dict[str, AgentResult] = {}

def get_result(topic: str) -> AgentResult:
    if topic not in _cache:
        _cache[topic] = run_agent(topic, verbose=False)
    return _cache[topic]


@pytest.fixture(scope="module")
def result_quantum():
    return get_result("quantum computing")

@pytest.fixture(scope="module")
def result_climate():
    return get_result("climate change and carbon capture")

@pytest.fixture(scope="module")
def result_rag():
    return get_result("retrieval augmented generation")

@pytest.fixture(scope="module")
def result_crispr():
    return get_result("CRISPR gene editing")


# ═══════════════════════════════════════════════════════════════════════════════
# E01 — Quantum computing
# ═══════════════════════════════════════════════════════════════════════════════

class TestE01Quantum:
    """5 checks — all must average ≥80% for case to pass."""

    def test_has_citation(self, result_quantum):
        assert has_citations(result_quantum.synthesis, 1), \
            "Expected at least one inline citation [N]"

    def test_has_sources_section(self, result_quantum):
        assert has_sources_section(result_quantum.synthesis), \
            "Expected a ## Sources section at end of response"

    def test_minimum_length(self, result_quantum):
        wc = word_count(result_quantum.synthesis)
        assert wc >= 150, f"Expected ≥150 words, got {wc}"

    def test_quantum_terminology(self, result_quantum):
        assert re.search(r'qubit|superposition|entanglement|quantum', result_quantum.synthesis, re.I), \
            "Expected quantum computing terminology (qubit, superposition, entanglement)"

    def test_no_uncited_stats(self, result_quantum):
        assert no_uncited_percentages(result_quantum.synthesis), \
            "Found percentage statistic(s) without citation"


# ═══════════════════════════════════════════════════════════════════════════════
# E02 — Climate change
# ═══════════════════════════════════════════════════════════════════════════════

class TestE02Climate:
    """6 checks."""

    def test_has_citation(self, result_climate):
        assert has_citations(result_climate.synthesis, 1)

    def test_has_sources_section(self, result_climate):
        assert has_sources_section(result_climate.synthesis)

    def test_minimum_length(self, result_climate):
        wc = word_count(result_climate.synthesis)
        assert wc >= 200, f"Expected ≥200 words, got {wc}"

    def test_climate_terminology(self, result_climate):
        assert re.search(r'emission|carbon|temperatur|greenhouse|warming|CO₂|CO2',
                         result_climate.synthesis, re.I)

    def test_has_caveat(self, result_climate):
        assert has_caveat(result_climate.synthesis), \
            "Response should acknowledge limitations or uncertainty"

    def test_structured_output(self, result_climate):
        assert has_structure(result_climate.synthesis), \
            "Expected structured output with headings (##) or bullets"


# ═══════════════════════════════════════════════════════════════════════════════
# E03 — RAG
# ═══════════════════════════════════════════════════════════════════════════════

class TestE03RAG:
    """5 checks."""

    def test_has_multiple_citations(self, result_rag):
        count = len(re.findall(r'\[\d+\]', result_rag.synthesis))
        assert count >= 2, f"Expected ≥2 citations, found {count}"

    def test_has_sources_section(self, result_rag):
        assert has_sources_section(result_rag.synthesis)

    def test_minimum_length(self, result_rag):
        wc = word_count(result_rag.synthesis)
        assert wc >= 150, f"Expected ≥150 words, got {wc}"

    def test_rag_terminology(self, result_rag):
        assert re.search(r'retrieval|embed|vector|BM25|semantic|augment',
                         result_rag.synthesis, re.I)

    def test_structured_output(self, result_rag):
        assert has_structure(result_rag.synthesis)


# ═══════════════════════════════════════════════════════════════════════════════
# E04 — CRISPR
# ═══════════════════════════════════════════════════════════════════════════════

class TestE04CRISPR:
    """5 checks."""

    def test_has_citation(self, result_crispr):
        assert has_citations(result_crispr.synthesis, 1)

    def test_has_sources_section(self, result_crispr):
        assert has_sources_section(result_crispr.synthesis)

    def test_minimum_length(self, result_crispr):
        wc = word_count(result_crispr.synthesis)
        assert wc >= 150, f"Expected ≥150 words, got {wc}"

    def test_crispr_terminology(self, result_crispr):
        assert re.search(r'CRISPR|Cas9|genome|gene edit|DNA', result_crispr.synthesis, re.I)

    def test_ethical_consideration(self, result_crispr):
        assert re.search(r'ethic|limit|off.target|concern|risk|germline',
                         result_crispr.synthesis, re.I), \
            "Expected discussion of ethics or limitations"


# ═══════════════════════════════════════════════════════════════════════════════
# Suite-level assertion — overall ≥ 80%
# ═══════════════════════════════════════════════════════════════════════════════

class TestSuiteOverall:
    """
    Aggregates all individual checks and asserts the suite passes at 80%.
    This test runs last (alphabetically 'T' > most letters used above).
    """

    CHECKS = [
        # (topic, check_fn, description)
        ("quantum computing",
         lambda r: has_citations(r, 1), "E01 citations"),
        ("quantum computing",
         lambda r: has_sources_section(r), "E01 sources section"),
        ("quantum computing",
         lambda r: word_count(r) >= 150, "E01 word count"),
        ("quantum computing",
         lambda r: bool(re.search(r'qubit|superposition|entanglement', r, re.I)),
         "E01 terminology"),
        ("quantum computing",
         lambda r: no_uncited_percentages(r), "E01 no uncited stats"),
        ("climate change and carbon capture",
         lambda r: has_citations(r, 1), "E02 citations"),
        ("climate change and carbon capture",
         lambda r: has_sources_section(r), "E02 sources section"),
        ("climate change and carbon capture",
         lambda r: word_count(r) >= 200, "E02 word count"),
        ("climate change and carbon capture",
         lambda r: bool(re.search(r'emission|carbon|greenhouse', r, re.I)),
         "E02 terminology"),
        ("climate change and carbon capture",
         lambda r: has_caveat(r), "E02 caveats"),
        ("climate change and carbon capture",
         lambda r: has_structure(r), "E02 structure"),
        ("retrieval augmented generation",
         lambda r: len(re.findall(r'\[\d+\]', r)) >= 2, "E03 2+ citations"),
        ("retrieval augmented generation",
         lambda r: has_sources_section(r), "E03 sources section"),
        ("retrieval augmented generation",
         lambda r: word_count(r) >= 150, "E03 word count"),
        ("retrieval augmented generation",
         lambda r: bool(re.search(r'retrieval|embed|vector|BM25', r, re.I)),
         "E03 terminology"),
        ("retrieval augmented generation",
         lambda r: has_structure(r), "E03 structure"),
        ("CRISPR gene editing",
         lambda r: has_citations(r, 1), "E04 citations"),
        ("CRISPR gene editing",
         lambda r: has_sources_section(r), "E04 sources section"),
        ("CRISPR gene editing",
         lambda r: word_count(r) >= 150, "E04 word count"),
        ("CRISPR gene editing",
         lambda r: bool(re.search(r'CRISPR|Cas9|genome', r, re.I)),
         "E04 terminology"),
        ("CRISPR gene editing",
         lambda r: bool(re.search(r'ethic|limit|off.target|risk', r, re.I)),
         "E04 ethics"),
    ]

    def test_overall_suite_passes_80_percent(self):
        """Suite-level gate: ≥80% of all checks must pass."""
        passed = 0
        failures = []

        for topic, check_fn, desc in self.CHECKS:
            result = get_result(topic)
            try:
                ok = bool(check_fn(result.synthesis))
            except Exception as e:
                ok = False
            if ok:
                passed += 1
            else:
                failures.append(desc)

        total = len(self.CHECKS)
        pct = round(passed / total * 100)

        if failures:
            fail_list = "\n  ".join(failures)
            print(f"\nFailed checks ({len(failures)}):\n  {fail_list}")

        assert pct >= 80, (
            f"Suite scored {pct}% ({passed}/{total}) — below 80% threshold.\n"
            f"Failed: {failures}"
        )
        print(f"\n✓ Suite PASSED: {pct}% ({passed}/{total} checks)")
