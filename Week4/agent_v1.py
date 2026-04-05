"""
Week 4 Build Day — Agent v1
============================
Multi-tool LangGraph agent with:
  • Web search (via DuckDuckGo — no API key needed)
  • Calculator (safe expression evaluator)
  • File I/O (read / write / append / list)
  • Weather (Open-Meteo — free, no API key)
  • Proper state management (full message history + tool call log)
  • 10 real test tasks at the bottom

Install:
    pip install langgraph langchain-openai duckduckgo-search requests

Run:
    export OPENAI_API_KEY=sk-...
    python agent_v1.py
"""

# ── Imports ───────────────────────────────────────────────────────────────────
import json
import math
import operator
import os
import re
import ast
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, TypedDict

import requests
from ddgs import DDGS
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.graph.message import add_messages
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool


# ── 1. Agent State ────────────────────────────────────────────────────────────
# add_messages is a reducer — it appends rather than replaces the messages list.
# This gives us a full conversation history across all turns.

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]   # full chat history
    tool_calls_log: list[dict]                # structured audit trail
    task_label: str                           # human-readable task name


# ── 2. Tools ──────────────────────────────────────────────────────────────────

@tool
def web_search(query: str, max_results: int = 4) -> str:
    """Search the web using DuckDuckGo. Returns titles, URLs, and snippets."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        if not results:
            return "No results found."
        lines = []
        for i, r in enumerate(results, 1):
            lines.append(f"[{i}] {r.get('title','')}")
            lines.append(f"    URL: {r.get('href','')}")
            lines.append(f"    {r.get('body','')[:200]}")

        print("\n".join(lines))
        return "\n".join(lines)
    except Exception as e:
        return f"Search failed: {e}"


@tool
def calculator(expression: str) -> str:
    """
    Safely evaluate a mathematical expression.
    Supports: +, -, *, /, **, %, //, sqrt, abs, round, log, sin, cos, tan,
              pi, e, floor, ceil, factorial, gcd.
    Example: '2**10', 'sqrt(144)', 'round(3.14159, 2)', 'factorial(10)'
    """
    ALLOWED_NAMES = {
        "sqrt": math.sqrt, "abs": abs, "round": round,
        "log": math.log, "log10": math.log10, "log2": math.log2,
        "sin": math.sin, "cos": math.cos, "tan": math.tan,
        "pi": math.pi, "e": math.e,
        "floor": math.floor, "ceil": math.ceil,
        "factorial": math.factorial, "gcd": math.gcd,
        "pow": pow, "min": min, "max": max, "sum": sum,
        "True": True, "False": False,
    }
    try:
        # Basic safety: allow only safe token types
        tree = ast.parse(expression, mode="eval")
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id not in ALLOWED_NAMES:
                        return f"Function '{node.func.id}' is not allowed."
            elif isinstance(node, ast.Name):
                if node.id not in ALLOWED_NAMES and node.id not in ("None",):
                    return f"Name '{node.id}' is not allowed."
        result = eval(compile(tree, "<expr>", "eval"), {"__builtins__": {}}, ALLOWED_NAMES)
        return f"Result: {result}"
    except ZeroDivisionError:
        return "Error: Division by zero."
    except Exception as e:
        return f"Error evaluating '{expression}': {e}"


@tool
def file_read(path: str) -> str:
    """Read the contents of a file. Returns the text content or an error message."""
    try:
        p = Path(path)
        if not p.exists():
            return f"File not found: {path}"
        return p.read_text(encoding="utf-8")
    except Exception as e:
        return f"Read error: {e}"


@tool
def file_write(path: str, content: str, mode: str = "w") -> str:
    """
    Write content to a file.
    mode='w'  → overwrite (default)
    mode='a'  → append
    Creates parent directories automatically.
    """
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open(mode, encoding="utf-8") as f:
            f.write(content)
        action = "appended to" if mode == "a" else "written to"
        return f"Successfully {action} '{path}' ({len(content)} chars)."
    except Exception as e:
        return f"Write error: {e}"


@tool
def file_list(directory: str = ".") -> str:
    """List files and subdirectories in a directory."""
    try:
        p = Path(directory)
        if not p.exists():
            return f"Directory not found: {directory}"
        entries = sorted(p.iterdir(), key=lambda x: (x.is_file(), x.name))
        lines = [f"Contents of '{directory}':"]
        for entry in entries:
            size = f"  ({entry.stat().st_size} bytes)" if entry.is_file() else "/"
            lines.append(f"  {'[DIR]' if entry.is_dir() else '[FILE]'} {entry.name}{size}")
        return "\n".join(lines)
    except Exception as e:
        return f"List error: {e}"


@tool
def get_weather(city: str) -> str:
    """
    Get current weather for a city using the Open-Meteo API (free, no key needed).
    Uses Open-Meteo geocoding to resolve the city name first.
    """
    try:
        geo_url = "https://geocoding-api.open-meteo.com/v1/search"
        geo_r = requests.get(geo_url, params={"name": city, "count": 1}, timeout=8)
        geo_r.raise_for_status()
        geo = geo_r.json()
        if not geo.get("results"):
            return f"City not found: {city}"
        loc = geo["results"][0]
        lat, lon = loc["latitude"], loc["longitude"]
        name = loc["name"]

        wx_url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "current": [
                "temperature_2m", "relative_humidity_2m",
                "wind_speed_10m", "weather_code", "apparent_temperature",
            ],
            "temperature_unit": "celsius",
            "wind_speed_unit": "kmh",
        }
        wx_r = requests.get(wx_url, params=params, timeout=8)
        wx_r.raise_for_status()
        wx = wx_r.json()["current"]

        WMO = {
            0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
            45: "Foggy", 48: "Icy fog", 51: "Light drizzle", 61: "Light rain",
            63: "Moderate rain", 65: "Heavy rain", 71: "Light snow", 80: "Rain showers",
            95: "Thunderstorm",
        }
        code = wx.get("weather_code", 0)
        desc = WMO.get(code, f"Code {code}")

        return (
            f"Weather in {name}:\n"
            f"  Condition:   {desc}\n"
            f"  Temperature: {wx['temperature_2m']}°C "
            f"(feels like {wx['apparent_temperature']}°C)\n"
            f"  Humidity:    {wx['relative_humidity_2m']}%\n"
            f"  Wind:        {wx['wind_speed_10m']} km/h"
        )
    except Exception as e:
        return f"Weather error: {e}"


# ── 3. Tool Registry ──────────────────────────────────────────────────────────
TOOLS = [web_search, calculator, file_read, file_write, file_list, get_weather]
TOOL_MAP = {t.name: t for t in TOOLS}


# ── 4. LLM ───────────────────────────────────────────────────────────────────
MINIMAX_API_KEY = os.getenv("MINIMAX_API_KEY", "YOUR_KEY_HERE")
MINIMAX_BASE_URL = os.getenv("MINIMAX_BASE_URL", "https://api.minimax.chat/v1")
MINIMAX_MODEL = os.getenv("MINIMAX_MODEL", "MiniMax-M2.7")
llm = ChatOpenAI(
    temperature=0.7,
    model = MINIMAX_MODEL,
    api_key=MINIMAX_API_KEY,
    base_url=MINIMAX_BASE_URL,
 
)
llm_with_tools = llm.bind_tools(TOOLS)

SYSTEM_PROMPT = """You are Agent v1 — a capable assistant with access to tools.

Rules:
1. Think before acting. Choose the right tool for the job.
2. Use calculator for ALL numerical computation — do not do arithmetic in your head.
3. Use web_search for any fact that might be outdated or unknown to you.
4. After receiving tool results, synthesise a clear, direct answer.
5. If a task requires multiple tools, chain them in logical order.
6. Be concise. The user cares about results, not verbose explanations.
"""


# ── 5. Graph Nodes ────────────────────────────────────────────────────────────

def agent_node(state: AgentState) -> dict:
    """Call the LLM. It decides whether to call a tool or produce a final answer."""
    msgs = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = llm_with_tools.invoke(msgs)
    return {"messages": [response]}


def tool_node(state: AgentState) -> dict:
    """Execute every tool call the LLM requested; return ToolMessages."""
    last_msg = state["messages"][-1]
    tool_messages = []
    log_entries = list(state.get("tool_calls_log", []))

    for tc in last_msg.tool_calls:
        tool_fn = TOOL_MAP.get(tc["name"])
        if tool_fn is None:
            result = f"Unknown tool: {tc['name']}"
        else:
            try:
                result = tool_fn.invoke(tc["args"])
            except Exception as e:
                result = f"Tool error: {e}"

        print(f"[TOOL CALL] {tc['name']} | args: {tc['args']} | result: {str(result)[:100]}")

        tool_messages.append(
            ToolMessage(content=str(result), tool_call_id=tc["id"], name=tc["name"])
        )
        log_entries.append({
            "timestamp": datetime.now().isoformat(),
            "tool": tc["name"],
            "args": tc["args"],
            "result_preview": str(result)[:120],
        })

    return {"messages": tool_messages, "tool_calls_log": log_entries}


# ── 6. Router ─────────────────────────────────────────────────────────────────

def should_continue(state: AgentState) -> str:
    """Route: if the LLM produced tool calls → run tools; else → end."""
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END


# ── 7. Build + Compile ────────────────────────────────────────────────────────

def build_agent() -> any:
    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")   # after tools → back to agent to reason
    return graph.compile()


# ── 8. Runner Helper ──────────────────────────────────────────────────────────

def run_task(agent: CompiledStateGraph, task: str, label: str = "") -> dict:
    """Run a single task through the agent. Returns the final state."""
    print(f"\n{'='*60}")
    print(f"TASK: {label or task}")
    print(f"{'='*60}")

    initial: AgentState = {
        "messages": [HumanMessage(content=task)],
        "tool_calls_log": [],
        "task_label": label or task,
    }

    final_state = agent.invoke(initial, {"recursion_limit": 20})

    # Print tool audit trail
    if final_state.get("tool_calls_log"):
        print("\nTool calls:")
        for entry in final_state["tool_calls_log"]:
            print(f"  [{entry['tool']}] args={entry['args']} → {entry['result_preview'][:80]}…")

    # Print final answer
    last_msg = final_state["messages"][-1]
    print(f"\nAnswer:\n{last_msg.content}")

    return final_state


# ── 9. Ten Real Test Tasks ────────────────────────────────────────────────────

TEST_TASKS = [
    {
        "label": "T01 — Calculator: compound interest",
        "task": (
            "If I invest $5,000 at 7% annual interest compounded monthly for 10 years, "
            "what will my balance be? Show the formula and result."
        ),
    },
    {
        "label": "T02 — Calculator: prime factorisation helper",
        "task": "What is 2**32 and what is sqrt(2**32)? Use the calculator.",
    },
    {
        "label": "T03 — Web search: recent AI news",
        "task": "What are the three most significant AI research announcements from the past week?",
    },
    {
        "label": "T04 — Web search + calculator: currency conversion",
        "task": (
            "Search for the current EUR to JPY exchange rate, "
            "then calculate how many yen 750 euros would buy."
        ),
    },
    {
        "label": "T05 — Weather: multi-city comparison",
        "task": (
            "Get the current weather in Tokyo, London, and New York. "
            "Which city is warmest right now?"
        ),
    },
    {
        "label": "T06 — Weather + file write: weather report",
        "task": (
            "Get the weather in Sydney and Paris, then write a short weather briefing "
            "to 'weather_report.txt' — include both cities and today's date."
        ),
    },
    {
        "label": "T07 — File I/O: write then read",
        "task": (
            "Write a Python one-liner that prints the Fibonacci sequence up to 100 "
            "to 'fibonacci.py', then read it back and confirm the content."
        ),
    },
    {
        "label": "T08 — File I/O: append log",
        "task": (
            "Append a timestamped log entry to 'agent_log.txt': "
            "'Agent v1 completed Task T08 successfully.' "
            "Then read the file to confirm."
        ),
    },
    {
        "label": "T09 — Multi-tool chain: research + summarise + save",
        "task": (
            "Search for 'LangGraph vs LangChain differences', summarise the key differences "
            "in 5 bullet points, and save the summary to 'langgraph_vs_langchain.md'."
        ),
    },
    {
        "label": "T10 — Multi-tool chain: weather + calc + file",
        "task": (
            "Get the weather in Berlin. Calculate the temperature in Fahrenheit "
            "(formula: F = C * 9/5 + 32). Then write a one-line summary of "
            "Berlin's weather in both Celsius and Fahrenheit to 'berlin_weather.txt'."
        ),
    },
]


# ── 10. Main ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Building Agent v1…")
    agent = build_agent()

    # Optional: save graph visualisation
    try:
        png = agent.get_graph().draw_mermaid_png()
        Path("agent_v1_graph.png").write_bytes(png)
        print("Graph saved → agent_v1_graph.png")
    except Exception:
        print("(Skipping PNG — run: pip install playwright && playwright install)")

    # Run all 10 test tasks
    results = []
    for task_def in TEST_TASKS:
        state = run_task(agent, task_def["task"], task_def["label"])
        results.append({
            "label": task_def["label"],
            "tool_calls": len(state.get("tool_calls_log", [])),
            "turns": sum(1 for m in state["messages"] if isinstance(m, AIMessage)),
        })

    # Summary table
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"{'Task':<45} {'Tools':>6} {'Turns':>6}")
    print("-"*60)
    for r in results:
        print(f"{r['label']:<45} {r['tool_calls']:>6} {r['turns']:>6}")
    print("="*60)
    print(f"Total tool calls: {sum(r['tool_calls'] for r in results)}")


# ── Streaming variant ─────────────────────────────────────────────────────────
# Uncomment to stream step-by-step instead of waiting for final result:
#
# for event in agent.stream(initial_state, {"recursion_limit": 20}):
#     for node_name, node_output in event.items():
#         for msg in node_output.get("messages", []):
#             print(f"[{node_name}] {type(msg).__name__}: {str(msg.content)[:120]}")
