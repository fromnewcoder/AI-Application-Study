import { useState, useRef, useEffect, useCallback } from "react";

// ─── Design tokens ────────────────────────────────────────────────────────────
const T = {
  ink:     "#0f1117",
  paper:   "#fafaf8",
  mist:    "#f2f1ed",
  rule:    "#e4e2db",
  ghost:   "#9b9890",
  accent:  "#1a4f3a",
  accentL: "#e8f0eb",
  accentM: "#5a9470",
  gold:    "#b8860b",
  goldL:   "#fdf6e3",
  danger:  "#8b2020",
  dangerL: "#fdf0f0",
  mono:    `"JetBrains Mono", "Fira Code", monospace`,
  serif:   `"Playfair Display", "Georgia", serif`,
  sans:    `"DM Sans", "Helvetica Neue", sans-serif`,
};

// ─── Anthropic API call ───────────────────────────────────────────────────────
async function callClaude({ system, messages, tools, max_tokens = 2048, streaming = false }) {
  const body = {
    model: "claude-sonnet-4-20250514",
    max_tokens,
    system,
    messages,
  };
  if (tools?.length) body.tools = tools;

  const res = await fetch("https://api.anthropic.com/v1/messages", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err?.error?.message || `API error ${res.status}`);
  }
  return res.json();
}

// ─── Web search simulation ────────────────────────────────────────────────────
// In production swap this for a real search API (Brave, Tavily, SerpAPI).
// The agent calls search_web tool which triggers this.
function simulateSearch(query) {
  const results = {
    default: [
      {
        url: `https://en.wikipedia.org/wiki/${encodeURIComponent(query.split(" ").slice(0,3).join("_"))}`,
        title: `${query} — Wikipedia`,
        snippet: `${query} is a subject with extensive documentation. Key aspects include historical context, modern applications, and ongoing research. Scholars have noted its importance in multiple domains including science, technology, and culture.`,
        domain: "wikipedia.org",
        date: "2024",
      },
      {
        url: `https://www.nature.com/search?q=${encodeURIComponent(query)}`,
        title: `Recent advances in ${query} — Nature`,
        snippet: `A comprehensive review of the latest peer-reviewed research on ${query}. This article covers primary findings from 2022–2024, methodology, experimental results, and implications for future study. Consensus across studies suggests significant progress.`,
        domain: "nature.com",
        date: "2024",
      },
      {
        url: `https://arxiv.org/search/?searchtype=all&query=${encodeURIComponent(query)}`,
        title: `arXiv preprints: ${query}`,
        snippet: `Preprint repository results for ${query}. Contains cutting-edge research not yet peer-reviewed. Topics range from theoretical frameworks to empirical studies. High citation counts indicate community interest in the core findings.`,
        domain: "arxiv.org",
        date: "2025",
      },
    ],
  };

  // Inject topic-aware snippets for common queries
  const q = query.toLowerCase();
  if (q.includes("climate") || q.includes("carbon")) {
    results.default[0].snippet = "Climate change refers to long-term shifts in global temperatures and weather patterns. Since the 1800s, human activities have been the main driver, primarily through burning fossil fuels which releases greenhouse gases. The IPCC reports consensus: limiting warming to 1.5°C requires rapid, far-reaching transitions in energy, land use, and industry.";
    results.default[1].snippet = "Carbon capture and storage (CCS) technologies have matured significantly. Direct air capture costs have fallen from $600/tonne in 2020 to approximately $250/tonne in 2024. Deployment is accelerating in North America and Europe, with 45Q tax credits driving investment. However, scale remains a challenge—current global capacity captures ~0.01% of annual emissions.";
  }
  if (q.includes("ai") || q.includes("llm") || q.includes("machine learning")) {
    results.default[0].snippet = "Large language models (LLMs) are neural networks trained on vast text corpora to predict and generate human-like text. GPT-4, Claude, Gemini, and Llama represent the current generation. They exhibit emergent capabilities including reasoning, code generation, and instruction following. Scaling laws suggest performance improves predictably with compute and data.";
    results.default[1].snippet = "Retrieval-augmented generation (RAG) addresses hallucination by grounding LLM responses in retrieved documents. Hybrid search (BM25 + vector) combined with cross-encoder reranking achieves state-of-the-art retrieval precision on BEIR benchmarks. Production systems increasingly use RAG over fine-tuning for knowledge-intensive tasks due to lower cost and easier updates.";
  }

  return results.default;
}

// ─── Agent tools definition ───────────────────────────────────────────────────
const AGENT_TOOLS = [
  {
    name: "search_web",
    description: "Search the web for information about a topic. Returns up to 3 relevant results with titles, URLs, and snippets.",
    input_schema: {
      type: "object",
      properties: {
        query: { type: "string", description: "The search query to look up" },
      },
      required: ["query"],
    },
  },
  {
    name: "fetch_document",
    description: "Fetch and read the content of a specific URL to get full document details.",
    input_schema: {
      type: "object",
      properties: {
        url: { type: "string", description: "The URL to fetch" },
        title: { type: "string", description: "The document title for citation purposes" },
      },
      required: ["url", "title"],
    },
  },
];

const SYSTEM_PROMPT = `You are a rigorous research assistant. Your role is to investigate topics thoroughly, retrieve credible sources, and synthesise findings with precise citations.

WORKFLOW:
1. When given a topic, first call search_web to find relevant sources.
2. Identify the 3 most relevant results and call fetch_document on each.
3. After retrieving all documents, synthesise a well-structured answer.

SYNTHESIS RULES:
- Always cite sources inline using [1], [2], [3] notation matching the order you fetched them.
- Structure: opening summary paragraph → key findings (2-4 bullets) → nuances/caveats → conclusion.
- Be factual and precise. Note areas of uncertainty or ongoing debate.
- Never fabricate statistics or claims not present in retrieved documents.
- End your response with a "## Sources" section listing each source with title and URL.

Your responses should be graduate-level quality: specific, well-cited, and intellectually honest.`;

// ─── Eval suite ──────────────────────────────────────────────────────────────
const EVAL_CASES = [
  {
    id: "E01",
    topic: "quantum computing basics",
    checks: [
      { id: "c1", desc: "Contains at least one citation [N]", fn: r => /\[\d+\]/.test(r) },
      { id: "c2", desc: "Sources section present", fn: r => /##\s*sources/i.test(r) },
      { id: "c3", desc: "Response ≥ 150 words", fn: r => r.split(/\s+/).length >= 150 },
      { id: "c4", desc: "No fabricated statistics (no unsourced %)", fn: r => {
        const pct = r.match(/\d+%/g) || [];
        const cited = r.match(/\[\d+\]/g) || [];
        return pct.length === 0 || cited.length >= pct.length;
      }},
      { id: "c5", desc: "Contains key term 'qubit' or 'superposition'", fn: r => /qubit|superposition/i.test(r) },
    ],
  },
  {
    id: "E02",
    topic: "climate change carbon capture",
    checks: [
      { id: "c1", desc: "Contains at least one citation [N]", fn: r => /\[\d+\]/.test(r) },
      { id: "c2", desc: "Sources section present", fn: r => /##\s*sources/i.test(r) },
      { id: "c3", desc: "Response ≥ 200 words", fn: r => r.split(/\s+/).length >= 200 },
      { id: "c4", desc: "Mentions climate-relevant term", fn: r => /carbon|emission|temperatur|greenhouse/i.test(r) },
      { id: "c5", desc: "Includes caveats or uncertainty", fn: r => /however|although|challenge|uncertain|limit|caveat/i.test(r) },
    ],
  },
  {
    id: "E03",
    topic: "retrieval augmented generation",
    checks: [
      { id: "c1", desc: "Contains at least two citations", fn: r => (r.match(/\[\d+\]/g)||[]).length >= 2 },
      { id: "c2", desc: "Sources section present", fn: r => /##\s*sources/i.test(r) },
      { id: "c3", desc: "Response ≥ 150 words", fn: r => r.split(/\s+/).length >= 150 },
      { id: "c4", desc: "Mentions RAG or retrieval", fn: r => /retrieval|rag|vector|embedding/i.test(r) },
      { id: "c5", desc: "Structured with headings or bullets", fn: r => /##|^\s*[-•*]/m.test(r) },
    ],
  },
];

// ─── Components ───────────────────────────────────────────────────────────────

function Badge({ children, variant = "default", style: s }) {
  const variants = {
    default: { bg: T.mist, color: T.ghost, border: T.rule },
    accent:  { bg: T.accentL, color: T.accent, border: T.accentM },
    gold:    { bg: T.goldL, color: T.gold, border: "#e0c060" },
    danger:  { bg: T.dangerL, color: T.danger, border: "#e08080" },
  };
  const v = variants[variant];
  return (
    <span style={{
      display: "inline-flex", alignItems: "center", gap: 4,
      padding: "2px 8px", borderRadius: 20,
      fontSize: 11, fontWeight: 600, letterSpacing: "0.04em",
      fontFamily: T.sans,
      background: v.bg, color: v.color,
      border: `0.5px solid ${v.border}`,
      ...s,
    }}>
      {children}
    </span>
  );
}

function SourceCard({ source, index }) {
  const [expanded, setExpanded] = useState(false);
  return (
    <div style={{
      border: `0.5px solid ${T.rule}`,
      borderRadius: 8, overflow: "hidden",
      transition: "border-color 0.15s",
    }}
      onMouseEnter={e => e.currentTarget.style.borderColor = T.accentM}
      onMouseLeave={e => e.currentTarget.style.borderColor = T.rule}
    >
      <button
        onClick={() => setExpanded(x => !x)}
        style={{
          width: "100%", textAlign: "left", background: "none",
          border: "none", cursor: "pointer", padding: "10px 14px",
          display: "flex", alignItems: "center", gap: 10,
        }}
      >
        <span style={{
          width: 20, height: 20, borderRadius: "50%",
          background: T.accentL, color: T.accent,
          fontSize: 11, fontWeight: 700, fontFamily: T.mono,
          display: "flex", alignItems: "center", justifyContent: "center",
          flexShrink: 0,
        }}>{index}</span>
        <span style={{ flex: 1, fontFamily: T.sans, fontSize: 13, fontWeight: 500, color: T.ink, lineHeight: 1.4 }}>
          {source.title}
        </span>
        <Badge variant="default">{source.domain}</Badge>
        <span style={{ color: T.ghost, fontSize: 12, transform: expanded ? "rotate(180deg)" : "none", transition: "transform 0.2s" }}>▾</span>
      </button>
      {expanded && (
        <div style={{ padding: "0 14px 12px 44px", borderTop: `0.5px solid ${T.rule}` }}>
          <p style={{ fontFamily: T.sans, fontSize: 12, color: T.ghost, lineHeight: 1.6, margin: "8px 0 6px" }}>
            {source.snippet}
          </p>
          <a href={source.url} target="_blank" rel="noopener noreferrer"
            style={{ fontFamily: T.mono, fontSize: 11, color: T.accentM, textDecoration: "none" }}
          >
            {source.url.slice(0, 60)}{source.url.length > 60 ? "…" : ""}
          </a>
        </div>
      )}
    </div>
  );
}

function AgentStep({ step, index }) {
  const icons = { search: "◎", fetch: "↓", think: "◈", done: "✓", error: "✕" };
  const colors = { search: T.gold, fetch: T.accent, think: "#7070c0", done: T.accentM, error: T.danger };
  const icon = icons[step.type] || "·";
  const color = colors[step.type] || T.ghost;

  return (
    <div style={{
      display: "flex", gap: 10, alignItems: "flex-start",
      padding: "6px 0",
      animation: "fadeSlideIn 0.2s ease forwards",
      opacity: 0,
      animationDelay: `${index * 0.05}s`,
    }}>
      <span style={{
        fontFamily: T.mono, fontSize: 14, color, minWidth: 16,
        lineHeight: 1.4, marginTop: 1,
      }}>{icon}</span>
      <div style={{ flex: 1 }}>
        <span style={{ fontFamily: T.sans, fontSize: 12, color: T.ghost }}>
          {step.label}
        </span>
        {step.detail && (
          <span style={{ fontFamily: T.mono, fontSize: 11, color: T.accentM, marginLeft: 6 }}>
            {step.detail.slice(0, 50)}{step.detail.length > 50 ? "…" : ""}
          </span>
        )}
      </div>
    </div>
  );
}

function MarkdownResponse({ text }) {
  // Simple markdown renderer for the synthesis output
  const lines = text.split("\n");
  const elements = [];
  let i = 0;

  while (i < lines.length) {
    const line = lines[i];
    if (line.startsWith("## ")) {
      elements.push(
        <h2 key={i} style={{
          fontFamily: T.serif, fontSize: 18, fontWeight: 700,
          color: T.ink, margin: "20px 0 8px", letterSpacing: "-0.02em",
          borderBottom: `1px solid ${T.rule}`, paddingBottom: 6,
        }}>
          {renderInline(line.slice(3))}
        </h2>
      );
    } else if (line.startsWith("# ")) {
      elements.push(
        <h1 key={i} style={{
          fontFamily: T.serif, fontSize: 22, fontWeight: 700,
          color: T.ink, margin: "0 0 12px", letterSpacing: "-0.03em",
        }}>
          {renderInline(line.slice(2))}
        </h1>
      );
    } else if (line.startsWith("- ") || line.startsWith("• ") || line.startsWith("* ")) {
      elements.push(
        <div key={i} style={{ display: "flex", gap: 8, margin: "4px 0", alignItems: "flex-start" }}>
          <span style={{ color: T.accentM, fontWeight: 700, marginTop: 2, flexShrink: 0 }}>—</span>
          <p style={{ fontFamily: T.sans, fontSize: 14, color: T.ink, lineHeight: 1.7, margin: 0 }}>
            {renderInline(line.slice(2))}
          </p>
        </div>
      );
    } else if (line.trim() === "") {
      elements.push(<div key={i} style={{ height: 8 }} />);
    } else {
      elements.push(
        <p key={i} style={{ fontFamily: T.sans, fontSize: 14, color: T.ink, lineHeight: 1.75, margin: "4px 0" }}>
          {renderInline(line)}
        </p>
      );
    }
    i++;
  }
  return <div>{elements}</div>;
}

function renderInline(text) {
  // Handle [N] citations, **bold**, and *italic*
  const parts = [];
  let remaining = text;
  let key = 0;

  while (remaining.length > 0) {
    const citMatch = remaining.match(/\[(\d+)\]/);
    const boldMatch = remaining.match(/\*\*(.+?)\*\*/);
    const matches = [citMatch, boldMatch].filter(Boolean);
    if (matches.length === 0) { parts.push(remaining); break; }

    const first = matches.reduce((a, b) => (a.index < b.index ? a : b));
    if (first.index > 0) parts.push(remaining.slice(0, first.index));

    if (first === citMatch) {
      parts.push(
        <sup key={key++} style={{
          display: "inline-flex", alignItems: "center", justifyContent: "center",
          width: 16, height: 16, borderRadius: "50%",
          background: T.accentL, color: T.accent,
          fontSize: 9, fontWeight: 700, fontFamily: T.mono,
          margin: "0 1px", verticalAlign: "super",
          cursor: "default",
        }}>{citMatch[1]}</sup>
      );
      remaining = remaining.slice(first.index + first[0].length);
    } else {
      parts.push(<strong key={key++} style={{ fontWeight: 600, color: T.ink }}>{boldMatch[1]}</strong>);
      remaining = remaining.slice(first.index + first[0].length);
    }
  }
  return parts;
}

// ─── Eval Suite Component ─────────────────────────────────────────────────────

function EvalSuite({ onClose }) {
  const [running, setRunning] = useState(false);
  const [results, setResults] = useState([]);
  const [currentCase, setCurrentCase] = useState(null);
  const [done, setDone] = useState(false);

  const runEval = useCallback(async () => {
    setRunning(true);
    setDone(false);
    setResults([]);
    const allResults = [];

    for (const evalCase of EVAL_CASES) {
      setCurrentCase(evalCase.id);
      // Init result
      const caseResult = { ...evalCase, response: "", checks: evalCase.checks.map(c => ({ ...c, passed: null })), running: true };
      setResults(prev => [...prev.filter(r => r.id !== evalCase.id), caseResult]);

      try {
        // Run the agent for this eval case
        const response = await runAgent(evalCase.topic, () => {});
        const responseText = response?.synthesis || "";

        // Score each check
        const scoredChecks = evalCase.checks.map(c => ({
          ...c,
          passed: (() => { try { return c.fn(responseText); } catch { return false; } })(),
        }));

        const passed = scoredChecks.filter(c => c.passed).length;
        const total = scoredChecks.length;
        const pct = Math.round((passed / total) * 100);

        allResults.push({ id: evalCase.id, passed, total, pct, scoredChecks, response: responseText });
        setResults(prev => [...prev.filter(r => r.id !== evalCase.id), {
          ...evalCase, response: responseText, checks: scoredChecks,
          passed, total, pct, running: false,
        }]);
      } catch (e) {
        allResults.push({ id: evalCase.id, passed: 0, total: evalCase.checks.length, pct: 0, error: e.message, running: false });
        setResults(prev => [...prev.filter(r => r.id !== evalCase.id), {
          ...evalCase, error: e.message, checks: evalCase.checks.map(c => ({ ...c, passed: false })),
          passed: 0, total: evalCase.checks.length, pct: 0, running: false,
        }]);
      }
    }

    setCurrentCase(null);
    setRunning(false);
    setDone(true);
  }, []);

  const totalChecks = results.reduce((a, r) => a + (r.total || 0), 0);
  const passedChecks = results.reduce((a, r) => a + (r.passed || 0), 0);
  const overallPct = totalChecks > 0 ? Math.round((passedChecks / totalChecks) * 100) : 0;
  const passSuite = overallPct >= 80;

  return (
    <div style={{
      position: "fixed", inset: 0, zIndex: 100,
      background: "rgba(15,17,23,0.7)",
      display: "flex", alignItems: "center", justifyContent: "center",
      padding: 24,
    }}>
      <div style={{
        background: T.paper, borderRadius: 16,
        border: `1px solid ${T.rule}`,
        width: "100%", maxWidth: 680,
        maxHeight: "90vh", overflow: "auto",
        padding: "28px 32px",
      }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 24 }}>
          <div>
            <h2 style={{ fontFamily: T.serif, fontSize: 22, color: T.ink, margin: 0 }}>Eval Suite</h2>
            <p style={{ fontFamily: T.sans, fontSize: 13, color: T.ghost, margin: "4px 0 0" }}>
              {EVAL_CASES.length} test cases · {EVAL_CASES.reduce((a, c) => a + c.checks.length, 0)} checks · target ≥ 80%
            </p>
          </div>
          <button onClick={onClose} style={{
            background: T.mist, border: `0.5px solid ${T.rule}`,
            borderRadius: 8, padding: "6px 14px", cursor: "pointer",
            fontFamily: T.sans, fontSize: 13, color: T.ghost,
          }}>Close</button>
        </div>

        {done && (
          <div style={{
            background: passSuite ? T.accentL : T.dangerL,
            border: `1px solid ${passSuite ? T.accentM : "#e08080"}`,
            borderRadius: 10, padding: "14px 18px", marginBottom: 20,
            display: "flex", alignItems: "center", gap: 14,
          }}>
            <span style={{ fontSize: 28 }}>{passSuite ? "✓" : "✕"}</span>
            <div>
              <div style={{ fontFamily: T.serif, fontSize: 18, fontWeight: 700, color: passSuite ? T.accent : T.danger }}>
                {overallPct}% passing — suite {passSuite ? "PASSED" : "FAILED"}
              </div>
              <div style={{ fontFamily: T.sans, fontSize: 13, color: passSuite ? T.accentM : T.danger, marginTop: 2 }}>
                {passedChecks}/{totalChecks} checks passed · threshold: 80%
              </div>
            </div>
          </div>
        )}

        {!running && !done && (
          <button onClick={runEval} style={{
            width: "100%", padding: "12px 0",
            background: T.accent, color: "#fff",
            border: "none", borderRadius: 8,
            fontFamily: T.sans, fontSize: 14, fontWeight: 600,
            cursor: "pointer", marginBottom: 20,
            letterSpacing: "0.02em",
          }}>
            Run eval suite
          </button>
        )}

        {running && !done && (
          <div style={{ marginBottom: 20, textAlign: "center" }}>
            <div style={{ fontFamily: T.mono, fontSize: 12, color: T.ghost }}>
              Running {currentCase}…
            </div>
            <div style={{
              height: 4, background: T.rule, borderRadius: 2, marginTop: 10,
              overflow: "hidden",
            }}>
              <div style={{
                height: "100%", background: T.accentM, borderRadius: 2,
                animation: "evalProgress 2s ease-in-out infinite",
              }} />
            </div>
          </div>
        )}

        {results.map(r => (
          <div key={r.id} style={{
            border: `0.5px solid ${T.rule}`, borderRadius: 10,
            marginBottom: 12, overflow: "hidden",
          }}>
            <div style={{
              padding: "10px 16px", background: T.mist,
              display: "flex", alignItems: "center", gap: 10,
            }}>
              <span style={{ fontFamily: T.mono, fontSize: 11, color: T.ghost }}>{r.id}</span>
              <span style={{ flex: 1, fontFamily: T.sans, fontSize: 13, fontWeight: 500, color: T.ink }}>
                {r.topic}
              </span>
              {r.running ? (
                <Badge>running…</Badge>
              ) : (
                <Badge variant={r.pct >= 80 ? "accent" : "danger"}>
                  {r.pct}%
                </Badge>
              )}
            </div>
            {!r.running && r.checks && (
              <div style={{ padding: "8px 16px 10px" }}>
                {r.checks.map(c => (
                  <div key={c.id} style={{
                    display: "flex", gap: 8, alignItems: "center",
                    padding: "3px 0",
                  }}>
                    <span style={{
                      fontSize: 12,
                      color: c.passed === null ? T.ghost : c.passed ? T.accentM : T.danger,
                    }}>
                      {c.passed === null ? "·" : c.passed ? "✓" : "✕"}
                    </span>
                    <span style={{ fontFamily: T.sans, fontSize: 12, color: c.passed ? T.ink : T.ghost }}>
                      {c.desc}
                    </span>
                  </div>
                ))}
                {r.error && (
                  <p style={{ fontFamily: T.mono, fontSize: 11, color: T.danger, marginTop: 6 }}>
                    Error: {r.error}
                  </p>
                )}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

// ─── Core agent runner ────────────────────────────────────────────────────────
async function runAgent(topic, onStep) {
  const messages = [{ role: "user", content: `Research this topic thoroughly: ${topic}` }];
  const sources = [];
  let synthesis = "";

  // Agentic loop
  for (let turn = 0; turn < 10; turn++) {
    const data = await callClaude({
      system: SYSTEM_PROMPT,
      messages,
      tools: AGENT_TOOLS,
      max_tokens: 2048,
    });

    // Process content blocks
    const toolUses = [];
    for (const block of data.content) {
      if (block.type === "text" && block.text?.trim()) {
        synthesis = block.text;
        onStep({ type: "think", label: "Synthesising answer…", detail: block.text.slice(0, 40) });
      }
      if (block.type === "tool_use") {
        toolUses.push(block);
      }
    }

    if (data.stop_reason === "end_turn") break;

    if (toolUses.length === 0) break;

    // Execute tool calls
    const toolResults = [];
    for (const tu of toolUses) {
      let result = "";
      if (tu.name === "search_web") {
        onStep({ type: "search", label: `Searching web…`, detail: tu.input.query });
        const searchResults = simulateSearch(tu.input.query);
        result = JSON.stringify(searchResults);
      } else if (tu.name === "fetch_document") {
        onStep({ type: "fetch", label: `Fetching document`, detail: tu.input.title });
        const idx = sources.length + 1;
        const mockContent = `Full document content for "${tu.input.title}". This represents the retrieved text from ${tu.input.url}. The document contains detailed information relevant to the research topic, including background context, methodology, findings, and implications. Key points are well-documented and traceable to authoritative sources.`;
        sources.push({
          index: idx,
          title: tu.input.title,
          url: tu.input.url,
          domain: new URL(tu.input.url).hostname.replace("www.", ""),
          snippet: mockContent.slice(0, 180),
        });
        result = mockContent;
      }
      toolResults.push({
        type: "tool_result",
        tool_use_id: tu.id,
        content: result,
      });
    }

    messages.push({ role: "assistant", content: data.content });
    messages.push({ role: "user", content: toolResults });
  }

  onStep({ type: "done", label: "Research complete" });
  return { synthesis, sources };
}

// ─── Main App ─────────────────────────────────────────────────────────────────
export default function ResearchAssistant() {
  const [topic, setTopic] = useState("");
  const [phase, setPhase] = useState("idle"); // idle | running | done | error
  const [steps, setSteps] = useState([]);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [showEval, setShowEval] = useState(false);
  const stepsEndRef = useRef(null);
  const inputRef = useRef(null);

  useEffect(() => {
    stepsEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [steps]);

  const addStep = useCallback((step) => {
    setSteps(prev => [...prev, step]);
  }, []);

  const handleResearch = useCallback(async () => {
    if (!topic.trim() || phase === "running") return;
    setPhase("running");
    setSteps([]);
    setResult(null);
    setError("");

    try {
      const res = await runAgent(topic.trim(), addStep);
      setResult(res);
      setPhase("done");
    } catch (e) {
      setError(e.message);
      setPhase("error");
    }
  }, [topic, phase, addStep]);

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleResearch();
    }
  };

  const reset = () => {
    setTopic("");
    setPhase("idle");
    setSteps([]);
    setResult(null);
    setError("");
    setTimeout(() => inputRef.current?.focus(), 50);
  };

  return (
    <>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=DM+Sans:wght@400;500;600&family=JetBrains+Mono:wght@400;700&display=swap');
        * { box-sizing: border-box; }
        body { margin: 0; background: ${T.paper}; }
        @keyframes fadeSlideIn {
          from { opacity: 0; transform: translateY(4px); }
          to   { opacity: 1; transform: translateY(0); }
        }
        @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }
        @keyframes evalProgress {
          0%   { transform: translateX(-100%); width: 60%; }
          50%  { transform: translateX(66%);   width: 60%; }
          100% { transform: translateX(200%);  width: 60%; }
        }
        .research-input:focus { outline: none; border-color: ${T.accent} !important; }
        .research-input::placeholder { color: ${T.ghost}; }
        .run-btn:hover { background: #0f3527 !important; }
        .run-btn:disabled { opacity: 0.5; cursor: not-allowed !important; }
        .source-link:hover { text-decoration: underline !important; }
        scrollbar-width: thin;
        scrollbar-color: ${T.rule} transparent;
      `}</style>

      {showEval && <EvalSuite onClose={() => setShowEval(false)} />}

      <div style={{
        minHeight: "100vh", background: T.paper,
        display: "flex", flexDirection: "column",
        fontFamily: T.sans,
      }}>
        {/* Header */}
        <header style={{
          borderBottom: `1px solid ${T.rule}`,
          padding: "16px 32px",
          display: "flex", alignItems: "center", justifyContent: "space-between",
          background: T.paper,
        }}>
          <div style={{ display: "flex", alignItems: "baseline", gap: 12 }}>
            <h1 style={{
              fontFamily: T.serif, fontSize: 20, fontWeight: 700,
              color: T.ink, margin: 0, letterSpacing: "-0.02em",
            }}>
              Research Assistant
            </h1>
            <Badge variant="accent">Ph 225</Badge>
          </div>
          <div style={{ display: "flex", gap: 8 }}>
            {phase === "done" && (
              <button onClick={reset} style={{
                background: "none", border: `0.5px solid ${T.rule}`,
                borderRadius: 8, padding: "6px 14px", cursor: "pointer",
                fontFamily: T.sans, fontSize: 13, color: T.ghost,
              }}>
                New research
              </button>
            )}
            <button onClick={() => setShowEval(true)} style={{
              background: T.mist, border: `0.5px solid ${T.rule}`,
              borderRadius: 8, padding: "6px 14px", cursor: "pointer",
              fontFamily: T.sans, fontSize: 13, color: T.ink,
              fontWeight: 500,
            }}>
              Eval suite
            </button>
          </div>
        </header>

        {/* Main */}
        <main style={{ flex: 1, maxWidth: 840, width: "100%", margin: "0 auto", padding: "32px 24px" }}>

          {/* Input area */}
          {(phase === "idle" || phase === "error") && (
            <div style={{
              animation: "fadeSlideIn 0.4s ease forwards",
              marginBottom: 40,
            }}>
              <h2 style={{
                fontFamily: T.serif, fontSize: 32, fontWeight: 700,
                color: T.ink, margin: "0 0 8px", letterSpacing: "-0.03em", lineHeight: 1.2,
              }}>
                What would you like<br />to research?
              </h2>
              <p style={{ fontFamily: T.sans, fontSize: 14, color: T.ghost, margin: "0 0 28px" }}>
                Enter any topic. The agent will search the web, retrieve 3 sources, and synthesise a cited answer.
              </p>

              <div style={{ display: "flex", gap: 10 }}>
                <input
                  ref={inputRef}
                  className="research-input"
                  value={topic}
                  onChange={e => setTopic(e.target.value)}
                  onKeyDown={handleKeyDown}
                  placeholder="e.g. retrieval augmented generation, climate change, quantum computing…"
                  autoFocus
                  style={{
                    flex: 1, padding: "14px 18px",
                    fontFamily: T.sans, fontSize: 15, color: T.ink,
                    background: T.paper, border: `1px solid ${T.rule}`,
                    borderRadius: 10, transition: "border-color 0.15s",
                  }}
                />
                <button
                  className="run-btn"
                  onClick={handleResearch}
                  disabled={!topic.trim()}
                  style={{
                    padding: "14px 24px", background: T.accent, color: "#fff",
                    border: "none", borderRadius: 10, cursor: "pointer",
                    fontFamily: T.sans, fontSize: 14, fontWeight: 600,
                    transition: "background 0.15s", whiteSpace: "nowrap",
                    letterSpacing: "0.02em",
                  }}
                >
                  Research →
                </button>
              </div>

              {error && (
                <div style={{
                  marginTop: 16, padding: "10px 14px", borderRadius: 8,
                  background: T.dangerL, border: `0.5px solid #e08080`,
                  fontFamily: T.mono, fontSize: 12, color: T.danger,
                }}>
                  {error}
                </div>
              )}

              {/* Example chips */}
              <div style={{ marginTop: 20, display: "flex", gap: 8, flexWrap: "wrap" }}>
                {["quantum computing", "climate change carbon capture", "retrieval augmented generation", "CRISPR gene editing"].map(ex => (
                  <button key={ex} onClick={() => setTopic(ex)} style={{
                    padding: "5px 12px", background: T.mist, border: `0.5px solid ${T.rule}`,
                    borderRadius: 20, cursor: "pointer", fontFamily: T.sans,
                    fontSize: 12, color: T.ghost, transition: "all 0.15s",
                  }}
                    onMouseEnter={e => { e.target.style.background = T.accentL; e.target.style.color = T.accent; }}
                    onMouseLeave={e => { e.target.style.background = T.mist; e.target.style.color = T.ghost; }}
                  >
                    {ex}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Running state */}
          {phase === "running" && (
            <div style={{ animation: "fadeSlideIn 0.3s ease forwards" }}>
              <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 24 }}>
                <div style={{
                  width: 8, height: 8, borderRadius: "50%",
                  background: T.accentM, animation: "pulse 1.2s ease-in-out infinite",
                }} />
                <h2 style={{
                  fontFamily: T.serif, fontSize: 20, color: T.ink, margin: 0,
                  fontWeight: 700, letterSpacing: "-0.02em",
                }}>
                  Researching: <em>{topic}</em>
                </h2>
              </div>

              <div style={{
                background: T.mist, borderRadius: 10,
                border: `0.5px solid ${T.rule}`,
                padding: "16px 20px",
              }}>
                <div style={{
                  fontFamily: T.mono, fontSize: 11, color: T.ghost,
                  letterSpacing: "0.06em", marginBottom: 12, textTransform: "uppercase",
                }}>
                  Agent trace
                </div>
                {steps.map((s, i) => <AgentStep key={i} step={s} index={i} />)}
                <div ref={stepsEndRef} />
              </div>
            </div>
          )}

          {/* Results */}
          {phase === "done" && result && (
            <div style={{ animation: "fadeSlideIn 0.4s ease forwards" }}>
              {/* Topic header */}
              <div style={{ marginBottom: 28 }}>
                <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 8 }}>
                  <Badge variant="accent">Research complete</Badge>
                  <Badge variant="default">{result.sources.length} sources</Badge>
                </div>
                <h2 style={{
                  fontFamily: T.serif, fontSize: 26, fontWeight: 700,
                  color: T.ink, margin: 0, letterSpacing: "-0.03em",
                }}>
                  {topic}
                </h2>
              </div>

              <div style={{ display: "grid", gridTemplateColumns: "1fr 280px", gap: 28, alignItems: "start" }}>
                {/* Synthesis */}
                <div>
                  <div style={{
                    background: T.paper, border: `1px solid ${T.rule}`,
                    borderRadius: 12, padding: "24px 28px",
                  }}>
                    <MarkdownResponse text={result.synthesis} />
                  </div>

                  {/* Agent trace (collapsed) */}
                  <details style={{ marginTop: 16 }}>
                    <summary style={{
                      fontFamily: T.mono, fontSize: 11, color: T.ghost,
                      cursor: "pointer", letterSpacing: "0.04em", userSelect: "none",
                    }}>
                      Agent trace ({steps.length} steps)
                    </summary>
                    <div style={{
                      marginTop: 8, padding: "12px 16px",
                      background: T.mist, borderRadius: 8,
                      border: `0.5px solid ${T.rule}`,
                    }}>
                      {steps.map((s, i) => <AgentStep key={i} step={s} index={i} />)}
                    </div>
                  </details>
                </div>

                {/* Sources sidebar */}
                <div>
                  <div style={{
                    fontFamily: T.mono, fontSize: 10, color: T.ghost,
                    letterSpacing: "0.08em", textTransform: "uppercase",
                    marginBottom: 10,
                  }}>
                    Sources retrieved
                  </div>
                  <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                    {result.sources.map(s => (
                      <SourceCard key={s.index} source={s} index={s.index} />
                    ))}
                  </div>

                  {result.sources.length === 0 && (
                    <div style={{
                      padding: "12px 14px", background: T.goldL,
                      border: `0.5px solid #e0c060`, borderRadius: 8,
                      fontFamily: T.sans, fontSize: 12, color: T.gold,
                    }}>
                      No sources retrieved. The agent may have answered from context.
                    </div>
                  )}

                  {/* Quality badges */}
                  <div style={{ marginTop: 16 }}>
                    <div style={{
                      fontFamily: T.mono, fontSize: 10, color: T.ghost,
                      letterSpacing: "0.08em", textTransform: "uppercase",
                      marginBottom: 8,
                    }}>
                      Quality signals
                    </div>
                    <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                      {[
                        { label: "Citations present", ok: /\[\d+\]/.test(result.synthesis) },
                        { label: "Sources listed", ok: /##\s*source/i.test(result.synthesis) },
                        { label: "150+ words", ok: result.synthesis.split(/\s+/).length >= 150 },
                        { label: "Structured output", ok: /##|^[-•]/m.test(result.synthesis) },
                      ].map(q => (
                        <div key={q.label} style={{ display: "flex", gap: 8, alignItems: "center" }}>
                          <span style={{ color: q.ok ? T.accentM : T.danger, fontSize: 12 }}>
                            {q.ok ? "✓" : "✕"}
                          </span>
                          <span style={{ fontFamily: T.sans, fontSize: 12, color: q.ok ? T.ink : T.ghost }}>
                            {q.label}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}
        </main>
      </div>
    </>
  );
}
