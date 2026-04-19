"""
=============================================================
  LangSmith Observability — Full Demo
  Topics covered:
    1. Tracing setup
    2. Span attribution & run types
    3. Tagging runs (tags + metadata)
    4. Prompt versioning (push / pull / pin)
    5. Debugging failures
=============================================================

QUICK START
-----------
1. Install dependencies:
   pip install langsmith langchain langchain-openai langchainhub python-dotenv

2. Create a .env file (or export directly):
   LANGCHAIN_TRACING_V2=true
   LANGCHAIN_API_KEY=ls__<your-key>
   LANGCHAIN_PROJECT=langsmith-demo
   OPENAI_API_KEY=sk-<your-key>

3. Run:
   python langsmith_observability_demo.py
"""

import os
import uuid
import time
import random
from dotenv import load_dotenv

# ── Load environment ──────────────────────────────────────────────────────────
load_dotenv()

# Validate required env vars before importing heavy deps
REQUIRED = ["LANGCHAIN_API_KEY", "LANGCHAIN_TRACING_V2", "OPENAI_API_KEY"]
missing = [k for k in REQUIRED if not os.getenv(k)]
if missing:
    raise EnvironmentError(
        f"Missing required environment variables: {missing}\n"
        "See the QUICK START section at the top of this file."
    )

# ── Core imports ──────────────────────────────────────────────────────────────
from langsmith import Client, traceable, get_current_run_tree
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, SystemMessage


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def section(title: str) -> None:
    """Pretty-print a section header."""
    print(f"\n{'═' * 60}")
    print(f"  {title}")
    print(f"{'═' * 60}")


def step(label: str, msg: str = "") -> None:
    print(f"  ▸ {label}" + (f": {msg}" if msg else ""))


# ─────────────────────────────────────────────────────────────────────────────
# 1. TRACING SETUP
# ─────────────────────────────────────────────────────────────────────────────

section("1 · Tracing Setup")

# LangChain auto-traces whenever LANGCHAIN_TRACING_V2=true.
# The Client is used for programmatic access (feedback, datasets, prompts).
client = Client()
llm = ChatOpenAI(model="MiniMax-M2.7", temperature=0)

# from langchain_core.runnables import RunnableLambda, RunnableParallel

# def llm_for_lcel(prompt_value) -> str:
#     MINIMAX_API_KEY = os.getenv("MINIMAX_API_KEY", "YOUR_KEY_HERE")
#     MINIMAX_BASE_URL = os.getenv("MINIMAX_BASE_URL", "https://api.minimax.chat/v1")
#     MINIMAX_MODEL = os.getenv("MINIMAX_MODEL", "MiniMax-M2.7")

#     from openai import OpenAI

#     client = OpenAI(
#         api_key=MINIMAX_API_KEY,
#         base_url=MINIMAX_BASE_URL,
#     )
#     messages = prompt_value.to_messages()
   
#     formatted = [
#         {"role": "system" if m.type == "system" else "user", "content": m.content}
#         for m in messages
#     ]
#     #print(formatted)
#     response = client.chat.completions.create(
#         model=MINIMAX_MODEL,
#         messages=formatted,
#         temperature=0.7,
#     )
#     print(response.choices[0].message.content)
#     return response.choices[0].message.content

# llm = RunnableLambda(llm_for_lcel)



step("LangSmith client", f"endpoint={client._host_url}")
step("Project", os.getenv("LANGCHAIN_PROJECT", "default"))
step(
    "Auto-trace active",
    "yes — every LangChain invocation will emit runs automatically",
)


# ─────────────────────────────────────────────────────────────────────────────
# 2. SPAN ATTRIBUTION  — @traceable + run_type
# ─────────────────────────────────────────────────────────────────────────────

section("2 · Span Attribution")

# run_type controls the icon and category inside LangSmith UI:
#   "chain"     → generic multi-step logic
#   "retriever" → document retrieval
#   "tool"      → tool / function call
#   "llm"       → direct LLM call  (usually captured automatically)
#   "embedding" → embedding call


@traceable(run_type="retriever", name="fake_vector_search")
def fake_vector_search(query: str) -> list[str]:
    """Simulates a vector-store retrieval.  Returns mock documents."""
    step("retriever", f"searching for: '{query}'")
    time.sleep(0.1)   # simulate latency
    return [
        f"Document A: background on '{query}'",
        f"Document B: recent updates on '{query}'",
    ]


@traceable(run_type="tool", name="web_search_tool")
def web_search_tool(query: str) -> str:
    """Simulates an external web search tool."""
    step("tool", f"web search: '{query}'")
    time.sleep(0.05)
    return f"[Web result] Top result for '{query}': summary text here."


@traceable(run_type="chain", name="rag_chain")
def rag_chain(question: str) -> str:
    """
    RAG chain: retrieve → augment → generate.
    Each sub-call is automatically nested as a child span.
    """
    docs = fake_vector_search(question)
    web  = web_search_tool(question)
    context = "\n".join(docs) + "\n" + web

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer concisely using the context below.\n\nContext:\n{context}"),
        ("human",  "{question}"),
    ])
    chain   = prompt | llm | StrOutputParser()
    answer  = chain.invoke({"context": context, "question": question})
    step("chain", f"answer length = {len(answer)} chars")
    return answer


# Kick off the chain — creates a 4-level span tree in LangSmith:
#   rag_chain → fake_vector_search
#             → web_search_tool
#             → ChatOpenAI (LLM span, auto-captured)
answer = rag_chain("What is LangSmith?")
print(f"\n  Answer: {answer[:120]}…")


# ─────────────────────────────────────────────────────────────────────────────
# 3. TAGGING RUNS — tags, metadata, runtime updates
# ─────────────────────────────────────────────────────────────────────────────

section("3 · Tagging Runs")

# Generate a session ID to group multiple turns together.
# (LangSmith doesn't have a native session object; use metadata for grouping.)
SESSION_ID = str(uuid.uuid4())
USER_ID    = "demo_user_42"

step("session_id", SESSION_ID)
step("user_id",    USER_ID)


@traceable(
    run_type="chain",
    name="tagged_pipeline",
    # ── Static tags & metadata set at decoration time ──────────────────────
    tags=["rag", "production", "v2"],
    metadata={
        "user_id":    USER_ID,
        "session_id": SESSION_ID,
        "env":        "demo",
    },
)
def tagged_pipeline(query: str, ab_variant: str = "A") -> str:
    """
    Demonstrates attaching structured metadata and dynamic runtime tags.
    """
    # ── Runtime update — attach computed info mid-execution ─────────────────
    run = get_current_run_tree()
    run.add_metadata({
        "ab_variant": ab_variant,
        "query_len":  len(query),
        "timestamp":  time.time(),
    })
    run.add_tags([f"ab:{ab_variant}", "tagged-at-runtime"])

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Be concise. Variant={variant}."),
        ("human",  "{query}"),
    ])
    chain  = prompt | llm | StrOutputParser()
    result = chain.invoke({"variant": ab_variant, "query": query})

    # Attach token-level stats after the LLM call completes
    run.add_metadata({"output_len": len(result)})

    step("tagged_pipeline", f"variant={ab_variant}, output_len={len(result)}")
    return result


tagged_pipeline("Explain observability in one sentence.", ab_variant="A")
tagged_pipeline("Explain observability in one sentence.", ab_variant="B")


# ─────────────────────────────────────────────────────────────────────────────
# 4. PROMPT VERSIONING — push / pull / pin
# ─────────────────────────────────────────────────────────────────────────────

section("4 · Prompt Versioning")

PROMPT_NAME = f"{client.settings.tenant_handle}/demo-rag-prompt"

# ── 4a. Push a prompt to the Hub ──────────────────────────────────────────────
rag_prompt_v1 = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful assistant.\n\nContext:\n{context}\n\n"
     "Answer the question using only the context above."),
    ("human", "{question}"),
])

step("Pushing prompt v1 to Hub", PROMPT_NAME)
try:
    commit_v1 = client.push_prompt(PROMPT_NAME, object=rag_prompt_v1)
    step("Commit v1", str(commit_v1))
except Exception as exc:
    # Gracefully handle permission issues in restricted environments
    step("Push skipped (permission / quota)", str(exc)[:80])
    commit_v1 = "SKIPPED"

# ── 4b. Push an improved v2 ───────────────────────────────────────────────────
rag_prompt_v2 = ChatPromptTemplate.from_messages([
    ("system",
     "You are a precise assistant. Reply in ≤3 sentences.\n\n"
     "Context:\n{context}\n\nIf the context is insufficient, say so."),
    ("human", "{question}"),
])

step("Pushing prompt v2 to Hub")
try:
    commit_v2 = client.push_prompt(PROMPT_NAME, object=rag_prompt_v2)
    step("Commit v2", str(commit_v2))
except Exception as exc:
    step("Push skipped", str(exc)[:80])
    commit_v2 = "SKIPPED"

# ── 4c. Pull latest vs pinned ─────────────────────────────────────────────────
step("Pulling latest prompt (dev)")
try:
    from langchainhub import pull as hub_pull
    prompt_latest = hub_pull(PROMPT_NAME)
    step("Latest prompt variables", str(prompt_latest.input_variables))
except Exception as exc:
    step("Pull skipped", str(exc)[:80])
    prompt_latest = rag_prompt_v2       # fall back to local object

if commit_v1 != "SKIPPED":
    step(f"Pulling pinned prompt (commit={commit_v1[:8]}…) for production")
    try:
        prompt_pinned = hub_pull(f"{PROMPT_NAME}:{commit_v1}")
        step("Pinned prompt variables", str(prompt_pinned.input_variables))
    except Exception as exc:
        step("Pin pull skipped", str(exc)[:80])
        prompt_pinned = rag_prompt_v1
else:
    prompt_pinned = rag_prompt_v1


# ── 4d. Use pinned prompt in a traced chain (best practice for production) ────
@traceable(
    run_type="chain",
    name="versioned_rag",
    metadata={"prompt_commit": str(commit_v1)[:8]},   # attach commit to trace
)
def versioned_rag(question: str, context: str) -> str:
    chain  = prompt_pinned | llm | StrOutputParser()
    result = chain.invoke({"question": question, "context": context})
    step("versioned_rag", f"used pinned prompt, output={len(result)} chars")
    return result


versioned_rag(
    question="What is prompt versioning?",
    context="Prompt versioning tracks changes to LLM prompts over time.",
)


# ─────────────────────────────────────────────────────────────────────────────
# 5. DEBUGGING FAILURES
# ─────────────────────────────────────────────────────────────────────────────

section("5 · Debugging Failures")


# ── 5a. A chain that deliberately fails ───────────────────────────────────────
@traceable(run_type="chain", name="buggy_chain", tags=["debug-demo"])
def buggy_chain(input_text: str) -> str:
    """
    Simulates a pipeline that raises an error mid-execution.
    The exception is captured by LangSmith and attached to the run.
    """
    step("buggy_chain", f"processing '{input_text}'")
    # Simulate a flaky retrieval step
    if random.random() < 0.7:         # fail ~70 % of the time for demo
        raise ValueError(
            f"Retrieval timeout for input: '{input_text}' "
            "(simulated failure for demo)"
        )
    return f"Success: processed '{input_text}'"


failing_run_id: str | None = None

for attempt in range(1, 4):
    try:
        result = buggy_chain(f"attempt-{attempt}")
        step(f"Attempt {attempt}", f"OK → {result}")
        break
    except ValueError as exc:
        step(f"Attempt {attempt}", f"FAILED → {exc}")
        # Capture the run_id so we can attach feedback (see below)
        # In production you'd get this from the LangSmith run tree
        failing_run_id = None   # would be set by run.id in a real handler


# ── 5b. Programmatic feedback on a run ────────────────────────────────────────
# In production you'd collect failing_run_id from get_current_run_tree().id
# inside the exception handler.  For this demo we list the most recent run.

step("Looking up the most recent 'buggy_chain' run to attach feedback…")
try:
    runs = list(client.list_runs(
        project_name=os.getenv("LANGCHAIN_PROJECT", "default"),
        filter='eq(name, "buggy_chain")',
        limit=1,
    ))
    if runs:
        failing_run_id = str(runs[0].id)
        step("Found run", failing_run_id[:8] + "…")

        client.create_feedback(
            run_id=failing_run_id,
            key="correctness",
            score=0,
            comment="Simulated retrieval timeout — flagged for investigation",
        )
        step("Feedback attached", "score=0, key=correctness")
    else:
        step("No runs found yet (traces may still be flushing)")
except Exception as exc:
    step("Feedback step skipped", str(exc)[:80])


# ── 5c. Add failing example to a Dataset for regression testing ───────────────
DATASET_NAME = "langsmith-demo-failures"

step("Upserting Dataset", DATASET_NAME)
try:
    # create_dataset is idempotent with read_only=False via try/except
    try:
        dataset = client.create_dataset(
            DATASET_NAME,
            description="Auto-collected failure cases for regression testing",
        )
        step("Dataset created", str(dataset.id)[:8] + "…")
    except Exception:
        datasets = list(client.list_datasets(dataset_name=DATASET_NAME))
        dataset  = datasets[0]
        step("Dataset already exists", str(dataset.id)[:8] + "…")

    client.create_example(
        inputs={"input_text": "attempt-1"},
        outputs={"expected": "Success: processed 'attempt-1'"},
        dataset_id=dataset.id,
    )
    step("Example added to dataset")
except Exception as exc:
    step("Dataset step skipped", str(exc)[:80])


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

section("Summary")
print("""
  All five observability patterns demonstrated:

  1. Tracing setup    — LANGCHAIN_TRACING_V2 + Client initialisation
  2. Span attribution — @traceable with run_type (chain / retriever / tool)
  3. Tagging runs     — static tags/metadata + runtime add_metadata()
  4. Prompt versions  — push_prompt(), hub.pull(), pinned commit in metadata
  5. Debugging        — error capture, create_feedback(), Dataset creation

  Open your LangSmith project to inspect the traces:
  https://smith.langchain.com
""")
