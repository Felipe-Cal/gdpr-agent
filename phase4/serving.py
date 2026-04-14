"""
Phase 4 — LangGraph agent using a self-hosted vLLM endpoint.

Key concept: vLLM exposes an OpenAI-compatible REST API (/v1/chat/completions).
LangChain's ChatOpenAI can point at any OpenAI-compatible server by overriding
`base_url` — so swapping from Gemini to a self-hosted model is a one-line change.

This file is identical in structure to phase2/graph.py.  The only difference is
the model backend: ChatVertexAI (Gemini) → ChatOpenAI (vLLM/Mistral).

Why this matters architecturally:
  - Vendor independence: the LangGraph graph, tools, and prompts are unchanged.
    You can benchmark Gemini vs Mistral vs Llama side-by-side.
  - Cost control: once GKE amortises the GPU cost over enough requests, per-query
    cost is lower than a pay-per-token managed API.
  - Data sovereignty: the model runs in your GCP project, in your EU region.
    No query text leaves your infrastructure — directly relevant to GDPR.
"""

from typing import Annotated, Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from config import settings
from phase2.tools import TOOLS
from phase2.graph import SYSTEM_PROMPT  # reuse the same prompt — agent is model-agnostic


# ---------------------------------------------------------------------------
# Agent state — identical to Phase 2
# ---------------------------------------------------------------------------
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ---------------------------------------------------------------------------
# Model — vLLM endpoint instead of Vertex AI
# ---------------------------------------------------------------------------
def build_vllm_model(endpoint_url: str | None = None) -> ChatOpenAI:
    """
    Creates a ChatOpenAI instance pointing at the vLLM server.

    ChatOpenAI works with any OpenAI-compatible server. vLLM's API is a drop-in
    replacement: the /v1/chat/completions endpoint accepts the same JSON schema
    as OpenAI's API, so no other code changes are needed.

    Args:
        endpoint_url: The base URL of the vLLM server.
                      Defaults to settings.vllm_endpoint (from .env).
                      In dev: http://localhost:8000
                      In GKE: http://<LoadBalancer-IP>
    """
    url = endpoint_url or settings.vllm_endpoint

    llm = ChatOpenAI(
        model=settings.vllm_model,           # matches --served-model-name in k8s/deployment.yaml
        base_url=f"{url}/v1",
        api_key="none",                       # vLLM doesn't require auth by default
        temperature=0,
        max_tokens=4096,
    )
    return llm.bind_tools(TOOLS)


# ---------------------------------------------------------------------------
# Nodes — identical logic to Phase 2, different model backend
# ---------------------------------------------------------------------------
def make_call_model_node(model):
    def call_model(state: AgentState) -> dict:
        messages = state["messages"]
        if not any(isinstance(m, SystemMessage) for m in messages):
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
        response = model.invoke(messages)
        return {"messages": [response]}
    return call_model


def run_tools(state: AgentState) -> dict:
    last_message: AIMessage = state["messages"][-1]
    tool_map = {t.name: t for t in TOOLS}
    tool_messages = []
    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_id = tool_call["id"]
        if tool_name not in tool_map:
            result = f"Unknown tool: {tool_name}"
        else:
            try:
                result = tool_map[tool_name].invoke(tool_args)
            except Exception as e:
                result = f"Tool '{tool_name}' failed: {e}"
        tool_messages.append(
            ToolMessage(content=str(result), tool_call_id=tool_id, name=tool_name)
        )
    return {"messages": tool_messages}


def should_continue(state: AgentState) -> Literal["run_tools", "__end__"]:
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "run_tools"
    return "__end__"


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------
def build_graph(endpoint_url: str | None = None):
    """Compiles the LangGraph agent using the vLLM backend."""
    model = build_vllm_model(endpoint_url)
    call_model = make_call_model_node(model)

    workflow = StateGraph(AgentState)
    workflow.add_node("call_model", call_model)
    workflow.add_node("run_tools", run_tools)
    workflow.add_edge(START, "call_model")
    workflow.add_conditional_edges(
        "call_model",
        should_continue,
        {"run_tools": "run_tools", "__end__": END},
    )
    workflow.add_edge("run_tools", "call_model")
    checkpointer = MemorySaver()
    return workflow.compile(checkpointer=checkpointer)


def ask_vllm(
    question: str,
    thread_id: str = "default",
    endpoint_url: str | None = None,
) -> tuple[str, list[str]]:
    """
    Ask the vLLM-backed agent a question.

    Returns (answer_text, list_of_tool_names_called).
    """
    graph = build_graph(endpoint_url)
    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke(
        {"messages": [HumanMessage(content=question)]},
        config=config,
    )

    answer = ""
    tools_used = []
    for msg in result["messages"]:
        if isinstance(msg, AIMessage):
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                tools_used.extend(tc["name"] for tc in msg.tool_calls)
            else:
                answer = msg.content
    return answer, tools_used
