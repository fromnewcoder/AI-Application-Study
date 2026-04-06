import os
import operator
from typing import Annotated, TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver # <--- NEW IMPORT

# --- 1. State Definition ---
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    score: int
    feedback: str
    attempts: int

# --- 2. LLM ---
MINIMAX_API_KEY = os.getenv("MINIMAX_API_KEY", "YOUR_KEY_HERE")
MINIMAX_BASE_URL = os.getenv("MINIMAX_BASE_URL", "https://api.minimax.chat/v1")
MINIMAX_MODEL = os.getenv("MINIMAX_MODEL", "MiniMax-M2.7")
llm = ChatOpenAI(
    temperature=0,
    model=MINIMAX_MODEL,
    api_key=MINIMAX_API_KEY,
    base_url=MINIMAX_BASE_URL,
)

def generate_node(state: AgentState):
    messages = state["messages"]
    print(f"[generate] input messages: {messages}")
    response = llm.invoke(messages)
    print(f"[generate] raw response: {response}")
    print(f"[generate] response.content: {repr(response.content)}")
    if not response.content:
        print("[generate] WARNING: empty response from LLM!")
    return {"messages": [response], "attempts": 1}

def grade_node(state: AgentState):
    """Scores the answer. If low, sets up feedback."""
    last_message = state["messages"][-1].content
    print(f"[grade] last_message content: {repr(last_message)}")
    if not last_message:
        print("[grade] WARNING: empty last_message, skipping grade")
        return {"score": 5, "feedback": "No content to grade."}
    judge_prompt = f"Rate this answer 1-5. Format: SCORE: <int> | FEEDBACK: <text>\n\nAnswer: {last_message}"
    # Use HumanMessage to avoid MiniMax API issue with standalone SystemMessage
    result = llm.invoke([HumanMessage(content=judge_prompt)])

    # Parsing Logic
    content = result.content
    try:
        score = int(content.split("|")[0].replace("SCORE:", "").strip())
    except:
        score = 1
    feedback = content.split("|")[1].replace("FEEDBACK:", "").strip()

    print(f"--- GRADING: Score {score}/5 ---")
    return {"score": score, "feedback": feedback}

def rewrite_node(state: AgentState):
    """This node now requires Human Approval to run."""
    print("--- REWRITING (Approved by Human) ---")
    messages = state["messages"]
    feedback = state["feedback"]
    improvement_prompt = HumanMessage(content=f"Refine this based on feedback: {feedback}")
    response = llm.invoke(messages + [improvement_prompt])
    return {"messages": [response], "attempts": state["attempts"] + 1}

# --- 3. Conditional Logic ---
def route_submission(state: AgentState):
    if state["score"] < 3 and state["attempts"] < 3:
        return "rewrite"
    return "end"

# --- 4. Graph Construction ---
workflow = StateGraph(AgentState)
workflow.add_node("generate", generate_node)
workflow.add_node("grade", grade_node)
workflow.add_node("rewrite", rewrite_node)

workflow.set_entry_point("generate")
workflow.add_edge("generate", "grade")
workflow.add_edge("rewrite", "grade")
workflow.add_conditional_edges("grade", route_submission, {"rewrite": "rewrite", "end": END})

# --- 5. Compile with Memory & Interrupt ---
memory = MemorySaver() # <--- Persists state between pauses

app = workflow.compile(
    checkpointer=memory, 
    interrupt_before=["rewrite"] # <--- The Breakpoint
)
