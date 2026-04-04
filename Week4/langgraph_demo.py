"""
LangGraph Intro — Python Demo
==============================
Topics covered:
  • State (TypedDict)
  • Nodes (plain functions)
  • Edges (normal + conditional)
  • Compile + invoke
  • Graph visualisation (Mermaid PNG)
  • Simple 2-node graph with intent-based routing

Run:
    pip install langgraph langchain-openai pillow
    export OPENAI_API_KEY=sk-...
    python langgraph_demo.py
"""

# ── 1. Imports ────────────────────────────────────────────────────────────────
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END

# ── 2. State ──────────────────────────────────────────────────────────────────
# State is the shared "memory" that every node reads from and writes to.
# We use TypedDict so every key/type is explicit and LangGraph can validate it.

class ChatState(TypedDict):
    user_input: str          # raw message from the user
    intent:     str          # detected intent: "math" | "weather" | "unknown"
    response:   str          # final text sent back to the user


# ── 3. Nodes ──────────────────────────────────────────────────────────────────
# A node is any callable that receives the current state dict and returns
# a *partial* dict of keys it wants to update. Untouched keys stay as-is.

def detect_intent(state: ChatState) -> dict:
    """Node 1 – classify the user's intent from their raw input."""
    text = state["user_input"].lower()
    if any(kw in text for kw in ["add", "subtract", "multiply", "divide",
                                  "+", "-", "*", "/", "calculate", "math"]):
        intent = "math"
    elif any(kw in text for kw in ["weather", "temperature", "rain",
                                    "forecast", "sunny", "cloudy"]):
        intent = "weather"
    else:
        intent = "unknown"

    print(f"[detect_intent] '{state['user_input']}' → intent={intent}")
    return {"intent": intent}


def handle_math(state: ChatState) -> dict:
    """Node 2a – respond to maths-related questions."""
    response = (
        f"Math handler here! You asked: '{state['user_input']}'. "
        "I'd use a calculator tool or LLM to solve this properly."
    )
    print(f"[handle_math] responding")
    return {"response": response}


def handle_weather(state: ChatState) -> dict:
    """Node 2b – respond to weather-related questions."""
    response = (
        f"Weather handler here! You asked: '{state['user_input']}'. "
        "I'd call a weather API to fetch a real forecast."
    )
    print(f"[handle_weather] responding")
    return {"response": response}


def handle_unknown(state: ChatState) -> dict:
    """Node 2c – fallback for unrecognised intents."""
    response = (
        f"I'm not sure how to handle: '{state['user_input']}'. "
        "Could you rephrase or be more specific?"
    )
    print(f"[handle_unknown] responding")
    return {"response": response}


# ── 4. Conditional Router ─────────────────────────────────────────────────────
# A router is a plain function that returns the *name* of the next node.
# LangGraph uses its return value to pick the outgoing edge.

def route_by_intent(state: ChatState) -> Literal["handle_math",
                                                   "handle_weather",
                                                   "handle_unknown"]:
    """Conditional edge – decide which handler to call based on intent."""
    return {
        "math":    "handle_math",
        "weather": "handle_weather",
    }.get(state["intent"], "handle_unknown")


# ── 5. Build + Compile the Graph ─────────────────────────────────────────────
def build_graph() -> StateGraph:
    graph = StateGraph(ChatState)

    # Register nodes
    graph.add_node("detect_intent",  detect_intent)
    graph.add_node("handle_math",    handle_math)
    graph.add_node("handle_weather", handle_weather)
    graph.add_node("handle_unknown", handle_unknown)

    # Entry point
    graph.set_entry_point("detect_intent")

    # Conditional edge from detect_intent ──► one of the three handlers
    graph.add_conditional_edges(
        source="detect_intent",
        path=route_by_intent,
        path_map={
            "handle_math":    "handle_math",
            "handle_weather": "handle_weather",
            "handle_unknown": "handle_unknown",
        },
    )

    # Each handler terminates the graph
    for handler in ("handle_math", "handle_weather", "handle_unknown"):
        graph.add_edge(handler, END)

    return graph.compile()


# ── 6. Graph Visualisation ───────────────────────────────────────────────────
def visualise(compiled_graph, filename: str = "graph.png") -> None:
    """
    Save a Mermaid-rendered PNG of the graph structure.
    Requires: pip install pillow
    """
    try:
        png_bytes = compiled_graph.get_graph().draw_mermaid_png()
        with open(filename, "wb") as f:
            f.write(png_bytes)
        print(f"[visualise] graph saved → {filename}")
    except Exception as exc:
        # Graceful fallback: print Mermaid source instead
        mermaid_src = compiled_graph.get_graph().draw_mermaid()
        print("[visualise] Could not render PNG. Mermaid source:\n")
        print(mermaid_src)
        print(f"\n(Error was: {exc})")


# ── 7. Run ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = build_graph()

    # Save / display the graph structure
    visualise(app, "langgraph_intent_router.png")

    # Test with several inputs
    test_inputs = [
        "Can you calculate 42 * 7 for me?",
        "Will it rain in London tomorrow?",
        "Tell me a joke.",
    ]

    print("\n" + "=" * 60)
    for user_msg in test_inputs:
        print(f"\nUSER: {user_msg}")
        initial_state: ChatState = {
            "user_input": user_msg,
            "intent":     "",
            "response":   "",
        }
        result = app.invoke(initial_state)
        print(f"BOT:  {result['response']}")
        print("-" * 60)


# ── BONUS: Streaming invocation ───────────────────────────────────────────────
# Uncomment to see step-by-step state updates as each node runs:
#
# for event in app.stream({"user_input": "add 3 and 5", "intent": "", "response": ""}):
#     for node_name, node_output in event.items():
#         print(f"[stream] {node_name}: {node_output}")
