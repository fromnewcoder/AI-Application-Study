"""
LangChain Core Abstractions — MiniMax-7B Live Demo
===================================================
Uses the OpenAI client (MiniMax API is OpenAI-compatible) wrapped in
LangChain Core Runnables to demonstrate:

  1. .invoke()   — single call, waits for full response
  2. .stream()  — token-by-token streaming
  3. .batch()   — parallel multi-input dispatch
  4. LCEL       — prompt | model | parser pipeline
"""

import os
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ── MiniMax client (OpenAI-compatible API) ────────────────────────────────────
# MiniMax Chat Completion API: https://www.minimaxi.com/document
#
# Set environment variables before running:
#   MINIMAX_API_KEY   — your MiniMax API key
#   MINIMAX_BASE_URL  — MiniMax API base (default below)
#   MINIMAX_MODEL     — model name, e.g. "MiniMax-7B" or "abab7"

MINIMAX_API_KEY = os.getenv("MINIMAX_API_KEY", "YOUR_KEY_HERE")
MINIMAX_BASE_URL = os.getenv("MINIMAX_BASE_URL", "https://api.minimax.chat/v1")
MINIMAX_MODEL = os.getenv("MINIMAX_MODEL", "MiniMax-M2.7")

from openai import OpenAI

client = OpenAI(
    api_key=MINIMAX_API_KEY,
    base_url=MINIMAX_BASE_URL,
)


# # ── 1. Wrap raw LLM call as a LangChain Runnable ─────────────────────────────
# def llm_invoke(messages: list[dict]) -> str:
#     """Single .invoke() — waits for the complete response."""
#     response = client.chat.completions.create(
#         model=MINIMAX_MODEL,
#         messages=messages,
#         temperature=0.7,
#     )
#     return response.choices[0].message.content


# def llm_stream(messages: list[dict]):
#     """Streaming generator — yields tokens as they arrive."""
#     stream = client.chat.completions.create(
#         model=MINIMAX_MODEL,
#         messages=messages,
#         temperature=0.7,
#         stream=True,
#     )
#     for chunk in stream:
#         if chunk.choices[0].delta.content:
#             yield chunk.choices[0].delta.content


# runnable = RunnableLambda(llm_invoke)
# streamable = RunnableLambda(llm_stream)


# # ── 2. .invoke() — single input, waits for full answer ───────────────────────
# print("=" * 60)
# print("1.  .invoke()  — single call, waits for full response")
# print("=" * 60)

# messages = [{"role": "user", "content": "What makes LCEL (LangChain Expression Language) powerful?"}]
# result = runnable.invoke(messages)
# print(result)


# # ── 3. .stream() — token-by-token as the model generates ────────────────────
# print("\n" + "=" * 60)
# print("2.  .stream()  — token-by-token streaming")
# print("=" * 60)

# messages = [{"role": "user", "content": "Explain why runnables are the core abstraction in LangChain in one short sentence."}]
# for token in streamable.stream(messages):
#     print(token, end="", flush=True)
# print()


# # ── 4. .batch() — parallel multi-input dispatch ──────────────────────────────
# print("\n" + "=" * 60)
# print("3.  .batch()  — parallel multi-input dispatch")
# print("=" * 60)

# batch_inputs = [
#     [{"role": "user", "content": "What is a Runnable in LangChain?"}],
#     [{"role": "user", "content": "What does LCEL stand for?"}],
#     [{"role": "user", "content": "Name the three main Runnable methods."}],
# ]

# results = runnable.batch(batch_inputs)
# for i, r in enumerate(results, 1):
#     print(f"\n[Q{i}] {batch_inputs[i-1][0]['content']}")
#     print(f"[A{i}] {r}")


# ── 5. LCEL — full pipeline with prompt template ─────────────────────────────
print("\n" + "=" * 60)
print("4.  LCEL pipeline — prompt | model | parser")
print("=" * 60)

MINIMAX_MODEL_FOR_LCEL = MINIMAX_MODEL  # reuse above

# 5a. Build a ChatPromptTemplate
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a concise tutor. Answer in at most 2 sentences."),
    ("user", "Explain this concept: {concept}"),
])

# 5b. Build a custom Runnable that calls MiniMax via the OpenAI client
def llm_for_lcel(prompt_value) -> str:
    messages = prompt_value.to_messages()
    formatted = [
        {"role": "system" if m.type == "system" else "user", "content": m.content}
        for m in messages
    ]
    response = client.chat.completions.create(
        model=MINIMAX_MODEL_FOR_LCEL,
        messages=formatted,
        temperature=0.7,
    )
    return response.choices[0].message.content

llm_runnable = RunnableLambda(llm_for_lcel)

# 5c. Compose with LCEL pipe operator
chain = prompt | llm_runnable | StrOutputParser()

# 5d. Invoke the chain
answer = chain.invoke({"concept": "RunnableParallel in LCEL"})
print(f"Concept: RunnableParallel in LCEL")
print(f"Answer:  {answer}")

# ── 6. LCEL Streaming ───────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("5.  LCEL streaming pipeline")
print("=" * 60)

for token in chain.stream({"concept": "StrOutputParser"}):
    print(token, end="", flush=True)
print()


# ── 7. RunnableParallel — parallel branches ───────────────────────────────────
print("\n" + "=" * 60)
print("6.  RunnableParallel — parallel branches")
print("=" * 60)

# Two independent prompt chains run concurrently
prompt_a = ChatPromptTemplate.from_messages([
    ("system", "You are a historian."),
    ("human", "{topic}"),
])
prompt_b = ChatPromptTemplate.from_messages([
    ("system", "You are a poet."),
    ("human", "Write about {topic} in 2 sentences."),
])

chain_a = prompt_a | llm_runnable | StrOutputParser()
chain_b = prompt_b | llm_runnable | StrOutputParser()

parallel = RunnableParallel(
    historical=chain_a,
    poetic=chain_b,
)

topic = "artificial intelligence"
merged = parallel.invoke({"topic": topic})
print(f"Topic: {topic}")
print(f"\nHistorian's view: {merged['historical']}")
print(f"Poet's view:      {merged['poetic']}")
