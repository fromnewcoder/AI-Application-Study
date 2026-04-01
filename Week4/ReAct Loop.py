"""
ReAct Loop — Reason → Act → Observe
Using Anthropic client with MiniMax model
"""

import re
import anthropic

# Configure client for MiniMax endpoint
client = anthropic.Anthropic(
    base_url="https://api.minimaxi.com/anthropic",
    api_key="YOUR_API_KEY"  # Replace with your API key or use env var
)

# ============================================================
# Tool definitions
# ============================================================
TOOLS = {
    "search": {
        "description": "Search the web for information. Use this for current facts, news, or any info that may change over time.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query"}
            },
            "required": ["query"]
        }
    },
    "calc": {
        "description": "Perform a mathematical calculation. Use this for any math operations.",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Math expression, e.g. '2 + 2' or '100 * 0.05'"}
            },
            "required": ["expression"]
        }
    },
    "lookup": {
        "description": "Look up factual information about an entity (person, company, place, etc.).",
        "input_schema": {
            "type": "object",
            "properties": {
                "entity": {"type": "string", "description": "Entity to look up"}
            },
            "required": ["entity"]
        }
    }
}

# ============================================================
# Tool implementations (mocked for demo)
# ============================================================
def search_impl(query: str) -> str:
    """Mock web search — replace with real API in production."""
    mock_results = {
        "tokyo population": "Tokyo has approximately 13.96 million residents (2024).",
        "japan gdp per capita": "Japan's GDP per capita is approximately $33,834 USD (2024).",
        "openai ceo": "Sam Altman is the current CEO of OpenAI.",
        "christopher nolan birthplace": "Christopher Nolan was born in Westminster, London, England.",
        "federal funds rate": "The US Federal funds rate is 5.25%-5.50% as of late 2024.",
        "apple stock price": "Apple (AAPL) is trading at approximately $189.30 USD.",
        "usd to gbp": "1 USD = 0.792 GBP (current exchange rate).",
    }
    query_lower = query.lower()
    for key, result in mock_results.items():
        if key in query_lower:
            return result
    return f"No results found for: {query}"

def calc_impl(expression: str) -> str:
    """Mock calculator — safe eval for demo purposes."""
    try:
        # Only allow safe math operations
        allowed_chars = set("0123456789.+-*/() **")
        if all(c in allowed_chars for c in expression):
            result = eval(expression)
            return str(result)
        return f"Error: invalid expression '{expression}'"
    except Exception as e:
        return f"Error calculating: {e}"

def lookup_impl(entity: str) -> str:
    """Mock lookup — replace with real knowledge base in production."""
    mock_db = {
        "sam altman": "Sam Altman is CEO of OpenAI. He was president of Y Combinator (2014-2019) before joining OpenAI.",
        "christopher nolan": "Christopher Nolan is a film director known for Inception, Interstellar, and Oppenheimer (2023). Born in Westminster, London.",
        "tokyo": "Tokyo is the capital of Japan, with population ~13.96 million.",
        "japan": "Japan is an island nation in East Asia. GDP per capita: ~$33,834 USD (2024).",
    }
    entity_lower = entity.lower()
    for key, info in mock_db.items():
        if key in entity_lower:
            return info
    return f"No information found for: {entity}"

TOOL_IMPLS = {
    "search": search_impl,
    "calc": calc_impl,
    "lookup": lookup_impl,
}

# ============================================================
# System prompt for ReAct agent
# ============================================================
SYSTEM_PROMPT = """You are a ReAct agent. You solve problems by interleaving reasoning with action.

You have access to these tools:
- search(query): Search the web for current information
- calc(expression): Perform mathematical calculations
- lookup(entity): Look up factual information about people, places, companies, etc.

Format each iteration EXACTLY as:
Thought: [your reasoning about what to do next]
Action: tool_name(argument)
Observation: [the result — will be filled in by the system]

Repeat Thought → Action → Observation until you can give a Final Answer.

Stop format:
Final Answer: [your complete answer to the user's question]

IMPORTANT:
- Always include the exact words 'Thought:', 'Action:', 'Observation:', and 'Final Answer:'
- Arguments must be in quotes if they are strings: Action: search("query here")
- After receiving an Observation, reason about it before deciding the next Action
- If a tool returns an error, try a different approach in your next Thought
- Max iterations: 15. If not done by then, return your best partial answer."""

# ============================================================
# ReAct Loop Implementation
# ============================================================
def react_agent(question: str, max_iter: int = 15) -> str:
    """
    Run a ReAct loop until Final Answer or max iterations.

    Args:
        question: The user's question
        max_iter: Maximum number of iterations (default 15)

    Returns:
        The agent's final answer
    """
    messages = [
        {"role": "user", "content": question}
    ]

    prev_action = None
    consecutive_errors = 0

    for iteration in range(max_iter):
        print(f"\n{'='*60}")
        print(f"ITERATION {iteration + 1}/{max_iter}")
        print('='*60)

        # 1. REASON — call the LLM
        response = client.messages.create(
            model="MiniMax-M2.7",
            system=SYSTEM_PROMPT,
            messages=messages,
            max_tokens=1024
        )

        text = response.content[0].text
        print(f"\n[MODEL RESPONSE]\n{text}")

        # 2. Check for Final Answer (STOP CONDITION)
        if "Final Answer:" in text:
            answer = text.split("Final Answer:")[1].strip()
            print(f"\n*** STOPPING: Final Answer found ***")
            return answer

        # 3. Parse the Action
        action_match = re.search(
            r"Action:\s*(\w+)\((.*?)\)", text, re.DOTALL
        )

        if not action_match:
            print(f"\n*** STOPPING: No valid Action found ***")
            return f"No valid action found. Partial reasoning:\n{text}"

        tool_name = action_match.group(1)
        tool_arg = action_match.group(2).strip().strip('"').strip("'")
        current_action = f"{tool_name}({tool_arg})"

        print(f"\n[ACTION] {current_action}")

        # 4. Repetition detection
        if current_action == prev_action:
            print(f"\n*** STOPPING: Repetition detected (same action twice) ***")
            return f"Agent stuck in repetition. Last action: {current_action}\n\nPartial reasoning:\n{text}"
        prev_action = current_action

        # 5. ACT — call the tool
        if tool_name not in TOOL_IMPLS:
            observation = f"Error: unknown tool '{tool_name}'"
            print(f"[TOOL ERROR] {observation}")
        else:
            try:
                observation = TOOL_IMPLS[tool_name](tool_arg)
                print(f"[OBSERVATION] {observation}")
            except Exception as e:
                observation = f"Error executing tool: {e}"
                print(f"[TOOL ERROR] {observation}")

        # 6. Error streak tracking
        if "Error" in observation:
            consecutive_errors += 1
            if consecutive_errors >= 3:
                print(f"\n*** STOPPING: Too many consecutive errors ***")
                return f"Too many consecutive tool errors.\n\nLast error: {observation}\n\nPartial reasoning:\n{text}"
        else:
            consecutive_errors = 0

        # 7. OBSERVE — inject result into context
        messages.append({"role": "assistant", "content": text})
        messages.append({
            "role": "user",
            "content": f"Observation: {observation}"
        })

    # MAX ITERATIONS REACHED
    print(f"\n*** STOPPING: Max iterations ({max_iter}) reached ***")
    return f"Reached maximum iterations ({max_iter}). Could not complete the task."


# ============================================================
# Demo problems (from study material)
# ============================================================
PROBLEMS = [
    {
        "name": "Serial Lookup",
        "question": "Who is the current CEO of OpenAI, and what was their role before?"
    },
    {
        "name": "Lookup → Compute",
        "question": "If I invest $10,000 at the current US Fed funds rate for 2 years (compounded annually), what do I get back?"
    },
    {
        "name": "Error Recovery",
        "question": "Find the birthplace of the director of the 2023 film Oppenheimer, then find a famous landmark there."
    },
    {
        "name": "Stock Price Conversion",
        "question": "What is Apple's current stock price in GBP?"
    }
]


def run_demo():
    print("ReAct Loop Demo — Using MiniMax-M2.7 with Anthropic client")
    print("=" * 60)

    for i, problem in enumerate(PROBLEMS):
        print(f"\n{'#'*60}")
        print(f"PROBLEM {i + 1}: {problem['name']}")
        print(f"{'#'*60}")
        print(f"\nQUESTION: {problem['question']}")
        print("=" * 60)

        answer = react_agent(problem["question"])

        print(f"\n{'='*60}")
        print(f"FINAL ANSWER:")
        print(f"{'='*60}")
        print(answer)
        print()


if __name__ == "__main__":
    run_demo()