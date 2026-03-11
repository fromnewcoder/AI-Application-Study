import { useState, useEffect, useRef } from "react";

const TOTAL_MINUTES = 60;

const sections = [
  {
    id: "tokenisation",
    label: "Tokenisation",
    emoji: "🔤",
    minutes: 15,
    color: "#f97316",
    accent: "#fed7aa",
    content: {
      title: "Understanding Tokenisation",
      objective: "Understand how text is split into tokens and why it matters.",
      theory: [
        {
          heading: "What is a Token?",
          body: "A token is the basic unit of text that a language model processes. Tokens are NOT the same as words — they're chunks of characters that appear frequently in training data. Common words may be one token; rare words may be split into multiple tokens.",
        },
        {
          heading: "How BPE Works",
          body: "Most modern LLMs use Byte Pair Encoding (BPE). It starts with individual characters, then repeatedly merges the most frequent adjacent pairs into a new token. The result: common words/subwords become single tokens; rare words decompose into smaller subword units.",
        },
        {
          heading: "Token Rules of Thumb",
          body: "~4 characters = 1 token in English. 1 word ≈ 1.3 tokens. 100 tokens ≈ 75 words. Code and non-English text typically use more tokens per word. Whitespace, punctuation, and capitalisation all affect tokenisation.",
        },
      ],
      interactive: {
        type: "tokeniser",
        label: "Live Token Visualiser",
        description: "Type text below to see how it gets tokenised (approximation using GPT-style rules):",
      },
      keyPoints: [
        "Tokens ≠ words — a word can be 1 or many tokens",
        "Whitespace and punctuation each count as tokens",
        "Rare words, names, and code are tokenised differently",
        "Tokenisation affects cost — APIs charge per token",
        "Different models use different vocabularies (tiktoken for OpenAI/Anthropic)",
      ],
      quiz: [
        { q: "The word 'unhappiness' is likely tokenised as:", a: "Multiple tokens (un / happiness or un / happy / ness)", wrong: ["One token", "Exactly 2 tokens always"] },
        { q: "Why does code typically use more tokens than English prose?", a: "Code contains rare identifiers, symbols, and patterns not common in training text", wrong: ["Code is always longer", "Code uses a different language"] },
      ],
    },
  },
  {
    id: "tiktoken",
    label: "Install tiktoken",
    emoji: "⚙️",
    minutes: 10,
    color: "#8b5cf6",
    accent: "#ede9fe",
    content: {
      title: "Install & Use tiktoken",
      objective: "Install tiktoken and tokenise your first string programmatically.",
      theory: [
        {
          heading: "What is tiktoken?",
          body: "tiktoken is OpenAI's fast BPE tokeniser written in Rust (with Python bindings). It supports the same encodings used by GPT-3.5, GPT-4, and Claude-compatible models. It's the go-to tool for counting tokens before API calls.",
        },
        {
          heading: "Encoding Families",
          body: "cl100k_base — used by GPT-3.5-turbo, GPT-4, and text-embedding-ada-002. o200k_base — used by GPT-4o. p50k_base — used by older Davinci models. Always match encoding to the model you're calling.",
        },
      ],
      interactive: {
        type: "code",
        label: "Installation & Basic Usage",
        steps: [
          {
            title: "Install",
            code: `pip install tiktoken`,
            lang: "bash",
          },
          {
            title: "Load an encoding",
            code: `import tiktoken\n\n# Load by encoding name\nenc = tiktoken.get_encoding("cl100k_base")\n\n# Or load by model name (auto-selects encoding)\nenc = tiktoken.encoding_for_model("gpt-4")`,
            lang: "python",
          },
          {
            title: "Encode & decode",
            code: `text = "Hello, world! This is a token test."\n\n# Encode to token IDs\ntokens = enc.encode(text)\nprint(tokens)       # [9906, 11, 1917, 0, 1115, 374, 264, 4037, 1296, 13]\nprint(len(tokens))  # 10\n\n# Decode back to text\ndecoded = enc.decode(tokens)\nprint(decoded)      # "Hello, world! This is a token test."`,
            lang: "python",
          },
          {
            title: "Inspect individual tokens",
            code: `for token_id in tokens:\n    token_bytes = enc.decode_single_token_bytes(token_id)\n    print(f"{token_id:6d} -> {token_bytes}")`,
            lang: "python",
          },
        ],
      },
      keyPoints: [
        "pip install tiktoken — fast Rust-based implementation",
        "Always match the encoding to your model",
        "enc.encode(text) → list of integer token IDs",
        "enc.decode(tokens) → original text (lossless)",
        "len(enc.encode(text)) gives you the exact token count",
      ],
      quiz: [
        { q: "Which encoding should you use for GPT-4?", a: "cl100k_base", wrong: ["p50k_base", "byte_level_bpe"] },
        { q: "What does enc.encode() return?", a: "A list of integer token IDs", wrong: ["A list of strings", "A single integer count"] },
      ],
    },
  },
  {
    id: "counting",
    label: "Count Tokens",
    emoji: "🔢",
    minutes: 15,
    color: "#0ea5e9",
    accent: "#e0f2fe",
    content: {
      title: "Count Tokens Programmatically",
      objective: "Build reusable functions to count tokens for prompts, chat messages, and files.",
      theory: [
        {
          heading: "Why Count Before Sending?",
          body: "Sending a prompt that exceeds the context window raises an error. Counting tokens upfront lets you: truncate long documents intelligently, estimate API costs before a call, chunk large texts into safe batches, and validate that system + user + history fits within limits.",
        },
        {
          heading: "Chat Message Overhead",
          body: "For chat models, each message has formatting overhead beyond just the text. OpenAI's format adds ~4 tokens per message (for role labels and separators) plus 2–3 tokens for the reply primer. Always account for this overhead in production systems.",
        },
      ],
      interactive: {
        type: "code",
        label: "Production-Ready Token Counter",
        steps: [
          {
            title: "Count tokens in a string",
            code: `import tiktoken\n\ndef count_tokens(text: str, model: str = "gpt-4") -> int:\n    enc = tiktoken.encoding_for_model(model)\n    return len(enc.encode(text))\n\nprint(count_tokens("Hello world"))  # 2\nprint(count_tokens("def fibonacci(n): return n if n < 2 else fibonacci(n-1)+fibonacci(n-2)"))`,
            lang: "python",
          },
          {
            title: "Count chat message tokens (with overhead)",
            code: `def count_chat_tokens(messages: list[dict], model: str = "gpt-4") -> int:\n    enc = tiktoken.encoding_for_model(model)\n    tokens_per_message = 3  # every message adds <|start|>role<|end|>\n    tokens_per_name = 1     # if name field present\n    \n    total = 0\n    for msg in messages:\n        total += tokens_per_message\n        for key, value in msg.items():\n            total += len(enc.encode(value))\n            if key == "name":\n                total += tokens_per_name\n    total += 3  # reply primer\n    return total\n\nmessages = [\n    {"role": "system", "content": "You are a helpful assistant."},\n    {"role": "user", "content": "Explain quantum entanglement simply."}\n]\nprint(count_chat_tokens(messages))  # ~23 tokens`,
            lang: "python",
          },
          {
            title: "Chunk a long document",
            code: `def chunk_text(text: str, max_tokens: int = 2000, model: str = "gpt-4") -> list[str]:\n    enc = tiktoken.encoding_for_model(model)\n    tokens = enc.encode(text)\n    chunks = []\n    for i in range(0, len(tokens), max_tokens):\n        chunk_tokens = tokens[i:i + max_tokens]\n        chunks.append(enc.decode(chunk_tokens))\n    return chunks\n\ndoc = "..." * 10000  # your long document\nchunks = chunk_text(doc, max_tokens=1500)\nprint(f"{len(chunks)} chunks created")`,
            lang: "python",
          },
          {
            title: "Estimate API cost",
            code: `PRICING = {\n    "gpt-4":        {"input": 0.03,  "output": 0.06},   # per 1K tokens\n    "gpt-3.5-turbo":{"input": 0.001, "output": 0.002},\n    "gpt-4o":       {"input": 0.005, "output": 0.015},\n}\n\ndef estimate_cost(prompt: str, expected_output_tokens: int, model="gpt-4") -> float:\n    input_tokens = count_tokens(prompt, model)\n    rates = PRICING[model]\n    cost = (input_tokens / 1000 * rates["input"]) + \\\n           (expected_output_tokens / 1000 * rates["output"])\n    return round(cost, 6)\n\nprint(f"Cost: ${"{"}estimate_cost('Write a poem about AI', 200){"}"}")`,
            lang: "python",
          },
        ],
      },
      keyPoints: [
        "Always count tokens BEFORE calling the API in production",
        "Chat messages have ~4 token overhead per message",
        "Use enc.encode() then len() — never estimate with word count",
        "Chunk documents at ~80% of context limit for safety",
        "Cache enc = tiktoken.encoding_for_model() — don't recreate per call",
      ],
      quiz: [
        { q: "What's the safest way to split a long document for an LLM?", a: "Encode to tokens, split token list at max_tokens boundary, then decode each chunk", wrong: ["Split by word count", "Split at every 1000 characters"] },
        { q: "Why add ~4 tokens per message overhead for chat?", a: "Each message includes role labels and special separator tokens added by the chat format", wrong: ["OpenAI charges extra per message", "The model reads the role as extra words"] },
      ],
    },
  },
  {
    id: "context",
    label: "Context Windows",
    emoji: "🪟",
    minutes: 20,
    color: "#10b981",
    accent: "#d1fae5",
    content: {
      title: "Context Window Size & Architecture",
      objective: "Understand why context window size is a core architectural constraint.",
      theory: [
        {
          heading: "What is the Context Window?",
          body: "The context window is the maximum number of tokens a model can process in a single forward pass — including system prompt, conversation history, documents, and the model's own output. Exceeding it causes errors. Unlike human memory, LLMs have NO memory outside this window.",
        },
        {
          heading: "Context Window Sizes (2024–2025)",
          body: "GPT-3.5-turbo: 16K tokens. GPT-4: 128K tokens. GPT-4o: 128K. Claude 3 Sonnet/Opus: 200K tokens. Gemini 1.5 Pro: 1M tokens. Larger windows enable: full codebases as context, long documents in one shot, multi-turn conversations without summarisation.",
        },
        {
          heading: "Why Size Matters for Architecture",
          body: "Context size directly determines WHICH patterns you can use. Small contexts (4K–16K) force you to chunk, summarise, or use RAG. Large contexts (100K+) allow 'long-context retrieval' — just stuffing in the document. But large contexts are slower and costlier; architecture must balance cost vs capability.",
        },
        {
          heading: "The Lost-in-the-Middle Problem",
          body: "Research shows LLMs perform best on information at the START and END of the context window. Content buried in the middle is often 'forgotten'. This matters enormously — don't assume 200K tokens means 200K tokens of equal attention.",
        },
      ],
      interactive: {
        type: "diagram",
        label: "Architecture Decision Tree",
        description: "How context window size shapes your system design:",
      },
      keyPoints: [
        "Context = system prompt + history + documents + output — all counted together",
        "Exceeding context window raises an API error — guard against it",
        "Larger context ≠ better; it's slower, costlier, and has attention dilution",
        "Lost-in-the-Middle: place critical info at start or end of context",
        "Common patterns: RAG (retrieval), sliding window, summarisation, hierarchical chunking",
        "Architecture choice depends on: doc size, latency budget, cost, and accuracy needs",
      ],
      quiz: [
        { q: "If your context window is 128K tokens and your system prompt uses 5K, what's left for user content?", a: "123K tokens (minus any expected output)", wrong: ["128K tokens — system prompt doesn't count", "125K — there's always a 3K reserve"] },
        { q: "What does 'Lost in the Middle' mean for prompt engineering?", a: "LLMs attend less to content in the middle of the context, so put critical info at start/end", wrong: ["The middle tokens cost more", "Long contexts always forget the system prompt"] },
      ],
    },
  },
];

const TokenVisualiser = () => {
  const [input, setInput] = useState("Hello, world! This is a token test.");
  const COLORS = ["#f97316","#8b5cf6","#0ea5e9","#10b981","#f43f5e","#eab308","#06b6d4","#a855f7"];

  const tokenise = (text) => {
    const tokens = [];
    let i = 0;
    const words = text.split(/(\s+|[.,!?;:'"()\[\]{}])/);
    words.forEach((w) => {
      if (!w) return;
      if (w.length > 6 && /[a-z]/i.test(w)) {
        const mid = Math.ceil(w.length / 2);
        tokens.push(w.slice(0, mid));
        tokens.push(w.slice(mid));
      } else {
        tokens.push(w);
      }
    });
    return tokens.filter(t => t.length > 0);
  };

  const tokens = tokenise(input);

  return (
    <div>
      <textarea
        value={input}
        onChange={e => setInput(e.target.value)}
        rows={3}
        style={{
          width: "100%", padding: "10px 14px", borderRadius: 8,
          border: "1.5px solid #e5e7eb", fontFamily: "monospace",
          fontSize: 14, resize: "vertical", outline: "none",
          background: "#fafafa", boxSizing: "border-box", marginBottom: 12,
        }}
      />
      <div style={{ display: "flex", flexWrap: "wrap", gap: 4, marginBottom: 10 }}>
        {tokens.map((t, i) => (
          <span key={i} style={{
            background: COLORS[i % COLORS.length] + "22",
            border: `1.5px solid ${COLORS[i % COLORS.length]}66`,
            color: COLORS[i % COLORS.length],
            borderRadius: 5, padding: "2px 7px", fontFamily: "monospace",
            fontSize: 13, fontWeight: 600,
          }}>{t === " " ? "·" : t}</span>
        ))}
      </div>
      <div style={{ fontSize: 13, color: "#6b7280" }}>
        ≈ <strong style={{ color: "#111" }}>{tokens.length} tokens</strong> · {input.length} characters · ratio: {(input.length / Math.max(tokens.length, 1)).toFixed(1)} chars/token
      </div>
    </div>
  );
};

const CodeBlock = ({ code, lang }) => {
  const [copied, setCopied] = useState(false);
  return (
    <div style={{ position: "relative", marginBottom: 12 }}>
      <pre style={{
        background: "#0f172a", color: "#e2e8f0", borderRadius: 10,
        padding: "14px 16px", fontSize: 12.5, overflowX: "auto",
        margin: 0, lineHeight: 1.7, fontFamily: "'Fira Code', 'Cascadia Code', monospace",
      }}><code>{code}</code></pre>
      <button
        onClick={() => { navigator.clipboard.writeText(code); setCopied(true); setTimeout(() => setCopied(false), 1800); }}
        style={{
          position: "absolute", top: 8, right: 8, background: copied ? "#10b981" : "#334155",
          color: "#fff", border: "none", borderRadius: 6, padding: "3px 10px",
          fontSize: 11, cursor: "pointer", transition: "background 0.2s",
        }}
      >{copied ? "✓ Copied" : "Copy"}</button>
    </div>
  );
};

const ArchDiagram = () => {
  const nodes = [
    { label: "Your\nDocument", x: 50, y: 50, color: "#6b7280" },
    { label: "< 16K\ntokens?", x: 220, y: 50, color: "#0ea5e9", decision: true },
    { label: "✓ Send\ndirectly", x: 380, y: 10, color: "#10b981" },
    { label: "< 128K\ntokens?", x: 380, y: 90, color: "#0ea5e9", decision: true },
    { label: "✓ Long-context\nmodel (GPT-4 / Claude)", x: 560, y: 50, color: "#10b981" },
    { label: "Use RAG or\nchunking strategy", x: 560, y: 130, color: "#f97316" },
  ];
  return (
    <div style={{ overflowX: "auto", paddingBottom: 8 }}>
      <svg viewBox="0 0 720 180" style={{ width: "100%", minWidth: 480, height: "auto" }}>
        <defs>
          <marker id="arr" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
            <path d="M0,0 L0,6 L8,3 z" fill="#94a3b8" />
          </marker>
        </defs>
        {/* Arrows */}
        <line x1="120" y1="65" x2="195" y2="65" stroke="#94a3b8" strokeWidth="1.5" markerEnd="url(#arr)" />
        <line x1="295" y1="55" x2="355" y2="25" stroke="#10b981" strokeWidth="1.5" markerEnd="url(#arr)" />
        <text x="315" y="30" fontSize="9" fill="#10b981">Yes</text>
        <line x1="295" y1="75" x2="355" y2="105" stroke="#f97316" strokeWidth="1.5" markerEnd="url(#arr)" />
        <text x="313" y="102" fontSize="9" fill="#f97316">No</text>
        <line x1="455" y1="95" x2="530" y2="68" stroke="#10b981" strokeWidth="1.5" markerEnd="url(#arr)" />
        <text x="470" y="72" fontSize="9" fill="#10b981">Yes</text>
        <line x1="455" y1="110" x2="530" y2="135" stroke="#f97316" strokeWidth="1.5" markerEnd="url(#arr)" />
        <text x="460" y="133" fontSize="9" fill="#f97316">No</text>

        {/* Nodes */}
        {[
          { label: "Your\nDocument", x: 30, y: 50, w: 90, color: "#6b7280" },
          { label: "< 16K tokens?", x: 195, y: 50, w: 100, color: "#0ea5e9", diamond: true },
          { label: "✓ Send directly", x: 355, y: 10, w: 100, color: "#10b981" },
          { label: "< 128K tokens?", x: 355, y: 90, w: 105, color: "#0ea5e9", diamond: true },
          { label: "Long-context\nmodel", x: 530, y: 50, w: 100, color: "#10b981" },
          { label: "RAG / Chunking\nStrategy", x: 530, y: 120, w: 110, color: "#f97316" },
        ].map((n, i) => (
          <g key={i}>
            <rect x={n.x} y={n.y - 16} width={n.w} height={36}
              rx={n.diamond ? 0 : 8}
              fill={n.color + "18"} stroke={n.color} strokeWidth="1.5"
              transform={n.diamond ? `rotate(0)` : ""}
            />
            {n.label.split("\n").map((line, li) => (
              <text key={li} x={n.x + n.w / 2} y={n.y + (li * 12) + (n.label.includes("\n") ? -4 : 4)}
                textAnchor="middle" fontSize="9.5" fill={n.color} fontWeight="600">
                {line}
              </text>
            ))}
          </g>
        ))}
      </svg>
      <div style={{ fontSize: 12, color: "#6b7280", marginTop: 4 }}>
        💡 RAG = Retrieval-Augmented Generation: embed + retrieve only relevant chunks at query time.
      </div>
    </div>
  );
};

const QuizBlock = ({ questions }) => {
  const [answers, setAnswers] = useState({});
  const shuffledOptions = useRef(
    questions.map(q => [q.a, ...q.wrong].sort(() => Math.random() - 0.5))
  );
  return (
    <div>
      {questions.map((q, qi) => {
        const options = shuffledOptions.current[qi];
        const selected = answers[qi];
        return (
          <div key={qi} style={{ marginBottom: 16 }}>
            <div style={{ fontWeight: 600, fontSize: 13.5, marginBottom: 8, color: "#1e293b" }}>
              Q{qi + 1}: {q.q}
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
              {options.map((opt, oi) => {
                const isSelected = selected === opt;
                const isCorrect = opt === q.a;
                let bg = "#f8fafc", border = "#e2e8f0", color = "#374151";
                if (isSelected && isCorrect) { bg = "#d1fae5"; border = "#10b981"; color = "#065f46"; }
                else if (isSelected && !isCorrect) { bg = "#fee2e2"; border = "#ef4444"; color = "#991b1b"; }
                return (
                  <button key={oi} onClick={() => setAnswers(a => ({ ...a, [qi]: opt }))}
                    style={{
                      background: bg, border: `1.5px solid ${border}`, color,
                      borderRadius: 7, padding: "8px 12px", fontSize: 13,
                      textAlign: "left", cursor: "pointer", transition: "all 0.15s",
                    }}
                  >
                    {isSelected ? (isCorrect ? "✅ " : "❌ ") : "○ "}{opt}
                  </button>
                );
              })}
            </div>
            {selected && selected !== q.a && (
              <div style={{ fontSize: 12, color: "#10b981", marginTop: 6, padding: "6px 10px", background: "#d1fae5", borderRadius: 6 }}>
                ✓ Correct answer: {q.a}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
};

export default function App() {
  const [activeSection, setActiveSection] = useState(0);
  const [completedSections, setCompletedSections] = useState(new Set());
  const [elapsed, setElapsed] = useState(0);
  const [running, setRunning] = useState(false);
  const timerRef = useRef(null);
  const [expandedStep, setExpandedStep] = useState(null);

  useEffect(() => {
    if (running) {
      timerRef.current = setInterval(() => setElapsed(e => e + 1), 1000);
    } else {
      clearInterval(timerRef.current);
    }
    return () => clearInterval(timerRef.current);
  }, [running]);

  const fmt = (s) => `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;
  const sec = sections[activeSection];
  const totalTarget = TOTAL_MINUTES * 60;
  const progress = Math.min(elapsed / totalTarget, 1);

  const markDone = () => {
    setCompletedSections(s => new Set([...s, activeSection]));
    if (activeSection < sections.length - 1) setActiveSection(activeSection + 1);
  };

  return (
    <div style={{
      minHeight: "100vh", background: "#f1f5f9",
      fontFamily: "'DM Sans', 'Segoe UI', sans-serif",
      display: "flex", flexDirection: "column",
    }}>
      {/* Google Font */}
      <style>{`@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=DM+Mono:wght@400;500&display=swap');
      * { box-sizing: border-box; }
      textarea:focus { border-color: #6366f1 !important; box-shadow: 0 0 0 3px #6366f122; }
      `}</style>

      {/* Header */}
      <div style={{
        background: "#0f172a", color: "#f8fafc",
        padding: "18px 24px", display: "flex", alignItems: "center",
        justifyContent: "space-between", flexWrap: "wrap", gap: 12,
      }}>
        <div>
          <div style={{ fontSize: 11, color: "#94a3b8", textTransform: "uppercase", letterSpacing: 2, marginBottom: 2 }}>1-Hour Study Resource</div>
          <div style={{ fontSize: 20, fontWeight: 700 }}>Tokens & Context Windows</div>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
          {/* Timer */}
          <div style={{
            background: "#1e293b", borderRadius: 10, padding: "8px 16px",
            fontFamily: "'DM Mono', monospace", fontSize: 20, letterSpacing: 2,
            color: elapsed > totalTarget * 0.9 ? "#f87171" : "#34d399",
          }}>{fmt(elapsed)}</div>
          <button onClick={() => setRunning(r => !r)} style={{
            background: running ? "#334155" : "#6366f1", color: "#fff",
            border: "none", borderRadius: 8, padding: "8px 18px",
            fontSize: 13, fontWeight: 600, cursor: "pointer",
          }}>{running ? "⏸ Pause" : elapsed === 0 ? "▶ Start Timer" : "▶ Resume"}</button>
        </div>
      </div>

      {/* Progress Bar */}
      <div style={{ background: "#1e293b", height: 4 }}>
        <div style={{
          height: "100%", background: "linear-gradient(90deg, #6366f1, #06b6d4)",
          width: `${progress * 100}%`, transition: "width 1s linear",
        }} />
      </div>

      <div style={{ display: "flex", flex: 1, maxWidth: 1100, margin: "0 auto", width: "100%", padding: "20px 16px", gap: 20 }}>
        {/* Sidebar */}
        <div style={{ width: 210, flexShrink: 0 }}>
          <div style={{ fontSize: 11, color: "#94a3b8", textTransform: "uppercase", letterSpacing: 1.5, marginBottom: 10, fontWeight: 600 }}>Sections</div>
          {sections.map((s, i) => {
            const done = completedSections.has(i);
            const active = i === activeSection;
            return (
              <button key={i} onClick={() => setActiveSection(i)} style={{
                width: "100%", textAlign: "left", padding: "10px 14px",
                borderRadius: 10, marginBottom: 6, cursor: "pointer",
                background: active ? s.color : done ? "#f0fdf4" : "#fff",
                border: active ? `2px solid ${s.color}` : done ? "1.5px solid #bbf7d0" : "1.5px solid #e2e8f0",
                color: active ? "#fff" : done ? "#065f46" : "#374151",
                transition: "all 0.15s",
              }}>
                <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
                  <span style={{ fontSize: 15 }}>{s.emoji}</span>
                  {done && <span style={{ fontSize: 12 }}>✓</span>}
                </div>
                <div style={{ fontWeight: 700, fontSize: 13, marginTop: 4 }}>{s.label}</div>
                <div style={{ fontSize: 11, opacity: 0.75, marginTop: 2 }}>{s.minutes} min</div>
              </button>
            );
          })}

          {/* Progress summary */}
          <div style={{ marginTop: 16, background: "#fff", borderRadius: 10, padding: "12px 14px", border: "1.5px solid #e2e8f0" }}>
            <div style={{ fontSize: 11, color: "#94a3b8", fontWeight: 600, textTransform: "uppercase", letterSpacing: 1, marginBottom: 8 }}>Your Progress</div>
            {sections.map((s, i) => (
              <div key={i} style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6 }}>
                <div style={{ width: 8, height: 8, borderRadius: "50%", background: completedSections.has(i) ? "#10b981" : "#e2e8f0" }} />
                <div style={{ fontSize: 12, color: completedSections.has(i) ? "#065f46" : "#9ca3af" }}>{s.label}</div>
              </div>
            ))}
            <div style={{ marginTop: 10, fontSize: 12, color: "#6b7280" }}>
              {completedSections.size} / {sections.length} complete
            </div>
          </div>
        </div>

        {/* Main Content */}
        <div style={{ flex: 1, minWidth: 0 }}>
          {/* Section Header */}
          <div style={{
            background: `linear-gradient(135deg, ${sec.color}18, ${sec.color}08)`,
            border: `1.5px solid ${sec.color}33`,
            borderRadius: 14, padding: "18px 22px", marginBottom: 18,
          }}>
            <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
              <span style={{ fontSize: 32 }}>{sec.emoji}</span>
              <div>
                <div style={{ fontSize: 11, color: sec.color, fontWeight: 700, textTransform: "uppercase", letterSpacing: 1.5 }}>
                  Section {activeSection + 1} of {sections.length} · {sec.minutes} minutes
                </div>
                <div style={{ fontSize: 22, fontWeight: 700, color: "#0f172a", marginTop: 2 }}>{sec.content.title}</div>
              </div>
            </div>
            <div style={{
              marginTop: 12, background: sec.color + "18", borderRadius: 8,
              padding: "8px 14px", fontSize: 13, color: sec.color, fontWeight: 600,
            }}>
              🎯 Objective: {sec.content.objective}
            </div>
          </div>

          {/* Theory Blocks */}
          <div style={{ marginBottom: 18 }}>
            <div style={{ fontSize: 13, fontWeight: 700, color: "#475569", textTransform: "uppercase", letterSpacing: 1.2, marginBottom: 10 }}>📖 Core Concepts</div>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))", gap: 12 }}>
              {sec.content.theory.map((t, i) => (
                <div key={i} style={{
                  background: "#fff", borderRadius: 12, padding: "14px 16px",
                  border: "1.5px solid #e2e8f0",
                }}>
                  <div style={{ fontWeight: 700, fontSize: 14, color: "#1e293b", marginBottom: 6 }}>{t.heading}</div>
                  <div style={{ fontSize: 13, color: "#475569", lineHeight: 1.65 }}>{t.body}</div>
                </div>
              ))}
            </div>
          </div>

          {/* Interactive Section */}
          <div style={{ background: "#fff", borderRadius: 14, border: "1.5px solid #e2e8f0", padding: "18px 20px", marginBottom: 18 }}>
            <div style={{ fontSize: 13, fontWeight: 700, color: "#475569", textTransform: "uppercase", letterSpacing: 1.2, marginBottom: 4 }}>
              ⚡ {sec.content.interactive.label}
            </div>
            {sec.content.interactive.description && (
              <div style={{ fontSize: 13, color: "#6b7280", marginBottom: 14 }}>{sec.content.interactive.description}</div>
            )}
            {sec.content.interactive.type === "tokeniser" && <TokenVisualiser />}
            {sec.content.interactive.type === "diagram" && <ArchDiagram />}
            {sec.content.interactive.type === "code" && (
              <div>
                {sec.content.interactive.steps.map((step, si) => (
                  <div key={si} style={{ marginBottom: 4 }}>
                    <button onClick={() => setExpandedStep(expandedStep === `${activeSection}-${si}` ? null : `${activeSection}-${si}`)}
                      style={{
                        width: "100%", textAlign: "left", background: "#f8fafc",
                        border: "1.5px solid #e2e8f0", borderRadius: expandedStep === `${activeSection}-${si}` ? "8px 8px 0 0" : 8,
                        padding: "10px 14px", cursor: "pointer", fontSize: 13,
                        fontWeight: 600, color: "#1e293b", display: "flex", justifyContent: "space-between",
                      }}>
                      <span>Step {si + 1}: {step.title}</span>
                      <span style={{ color: "#94a3b8" }}>{expandedStep === `${activeSection}-${si}` ? "▲" : "▼"}</span>
                    </button>
                    {expandedStep === `${activeSection}-${si}` && (
                      <div style={{ border: "1.5px solid #e2e8f0", borderTop: "none", borderRadius: "0 0 8px 8px", overflow: "hidden" }}>
                        <CodeBlock code={step.code} lang={step.lang} />
                      </div>
                    )}
                  </div>
                ))}
                <div style={{ fontSize: 12, color: "#94a3b8", marginTop: 8 }}>💡 Click each step to expand the code</div>
              </div>
            )}
          </div>

          {/* Key Points */}
          <div style={{ background: "#fff", borderRadius: 14, border: "1.5px solid #e2e8f0", padding: "16px 20px", marginBottom: 18 }}>
            <div style={{ fontSize: 13, fontWeight: 700, color: "#475569", textTransform: "uppercase", letterSpacing: 1.2, marginBottom: 12 }}>✅ Key Takeaways</div>
            <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
              {sec.content.keyPoints.map((pt, i) => (
                <div key={i} style={{ display: "flex", alignItems: "flex-start", gap: 10 }}>
                  <div style={{
                    width: 22, height: 22, borderRadius: "50%", background: sec.color + "22",
                    border: `1.5px solid ${sec.color}44`, color: sec.color,
                    fontSize: 11, fontWeight: 700, display: "flex", alignItems: "center",
                    justifyContent: "center", flexShrink: 0, marginTop: 1,
                  }}>{i + 1}</div>
                  <div style={{ fontSize: 13.5, color: "#374151", lineHeight: 1.55 }}>{pt}</div>
                </div>
              ))}
            </div>
          </div>

          {/* Quiz */}
          <div style={{ background: "#fff", borderRadius: 14, border: "1.5px solid #e2e8f0", padding: "16px 20px", marginBottom: 20 }}>
            <div style={{ fontSize: 13, fontWeight: 700, color: "#475569", textTransform: "uppercase", letterSpacing: 1.2, marginBottom: 14 }}>🧠 Quick Check</div>
            <QuizBlock questions={sec.content.quiz} />
          </div>

          {/* Done Button */}
          <button onClick={markDone} style={{
            background: sec.color, color: "#fff", border: "none",
            borderRadius: 10, padding: "12px 28px", fontSize: 15,
            fontWeight: 700, cursor: "pointer", width: "100%",
            boxShadow: `0 4px 14px ${sec.color}44`,
          }}>
            {activeSection < sections.length - 1 ? `✓ Done — Next: ${sections[activeSection + 1].label} →` : "🎉 Complete! All sections done"}
          </button>
        </div>
      </div>
    </div>
  );
}
