"""
Phase 3 — Traced agent wrapper.

An additive layer that wraps the Phase 2 agent calls to inject
observability callbacks (LangSmith).

This demonstrates how to add observability to an existing system
without modifying its core logic.
"""

from typing import Optional
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from phase2.graph import graph
from phase3.callbacks import get_callbacks

def traced_ask(question: str, thread_id: str = "default") -> tuple[str, list[str]]:
    """
    Ask the agent a question with tracing enabled.

    Args:
        question:  The user's question.
        thread_id: Conversation ID for multi-turn state.

    Returns:
        (answer_text, list_of_tool_names_called)
    """
    config = RunnableConfig(
        configurable={"thread_id": thread_id},
        callbacks=get_callbacks()
    )

    result = graph.invoke(
        {"messages": [HumanMessage(content=question)]},
        config=config,
    )

    from langchain_core.messages import AIMessage
    answer = ""
    tools_used = []

    for msg in result["messages"]:
        if isinstance(msg, AIMessage):
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                tools_used.extend(tc["name"] for tc in msg.tool_calls)
            else:
                answer = msg.content

    return answer, tools_used

def traced_stream_ask(question: str, thread_id: str = "default"):
    """
    Streaming version of traced_ask(). Yields (chunk, event_type) tuples.
    """
    config = RunnableConfig(
        configurable={"thread_id": thread_id},
        callbacks=get_callbacks()
    )
    inputs = {"messages": [HumanMessage(content=question)]}

    from langchain_core.messages import AIMessage

    for event in graph.stream(inputs, config=config, stream_mode="updates"):
        for node_name, node_output in event.items():
            if node_name == "call_model":
                msgs = node_output.get("messages", [])
                for msg in msgs:
                    if isinstance(msg, AIMessage) and not (
                        hasattr(msg, "tool_calls") and msg.tool_calls
                    ):
                        yield msg.content, "answer"
            elif node_name == "run_tools":
                msgs = node_output.get("messages", [])
                for msg in msgs:
                    yield msg.name, "tool_used"
