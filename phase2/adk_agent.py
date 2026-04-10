"""
Phase 2 — Google ADK (Agent Development Kit) agent.

This is the same GDPR analyst agent rebuilt using Google's own framework.
The point is to compare paradigms:

  LangGraph:  you define the graph explicitly — nodes, edges, state, routing logic.
              Maximum control and transparency. You see exactly how the agent reasons.

  Google ADK: declarative — you describe the agent and its tools; the framework
              handles the ReAct loop internally. Less boilerplate, tighter Vertex AI
              integration, but less visibility into internal state.

Both frameworks call the same tools (defined in phase2/tools.py). The difference
is entirely in how the orchestration is structured.

Usage:
    python -m phase2.main --agent adk

Google ADK docs: https://google.github.io/adk-docs/
"""

import asyncio

from config import settings

SYSTEM_PROMPT = """You are an expert GDPR legal analyst. You have access to three tools:

1. search_gdpr_documents — searches your knowledge base of GDPR regulation text and EDPB guidelines
2. get_gdpr_article — quickly retrieves the key provisions of a specific GDPR article by number
3. web_search — searches the web for recent enforcement actions, new guidelines, or news

How to reason:
- For specific article questions, start with get_gdpr_article for speed
- For broader questions or when you need exact regulatory text with context, use search_gdpr_documents
- For recent developments (fines, new guidance, country-specific enforcement), use web_search
- Always cite your sources (article numbers, document names, URLs) in your final answer
- Distinguish between hard legal requirements and best-practice recommendations
- If information is insufficient, say so — do not speculate"""


def _build_adk_agent():
    """
    Builds and returns a Google ADK agent.

    ADK uses a declarative approach:
    - You provide tools as plain Python functions (ADK reads their docstrings)
    - The model is specified by name
    - The orchestration loop (think → call tool → think → answer) is internal

    Key difference from LangGraph: the graph is implicit. You don't define nodes
    or edges — you just describe what the agent can do and let ADK figure out how.
    """
    try:
        from google.adk.agents import LlmAgent
    except ImportError:
        raise ImportError(
            "google-adk is not installed. Run: pip install google-adk"
        )

    # Import plain function versions of tools (ADK accepts callables, not @tool decorated)
    from phase2.tools import _get_vector_store, GDPR_ARTICLES

    # Re-define as plain functions for ADK (same logic, no LangChain decorator)
    def search_gdpr_documents(query: str) -> str:
        """Search the GDPR knowledge base for relevant legal text and guidance.

        Use when you need specific provisions, definitions, or guidance from ingested
        GDPR documents. Returns the top matching passages with source citations.

        Args:
            query: Natural language search query, e.g. 'conditions for valid consent'.
        """
        store = _get_vector_store()
        docs = store.similarity_search(query, k=settings.retrieval_top_k)
        if not docs:
            return "No relevant documents found."
        sections = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source_file", "unknown")
            page = doc.metadata.get("page", "")
            label = f"[{i}] {source}" + (f", page {page}" if page else "")
            sections.append(f"{label}\n{doc.page_content}")
        return "\n\n---\n\n".join(sections)

    def web_search(query: str) -> str:
        """Search the web for recent GDPR news, enforcement actions, and regulatory updates.

        Use for recent DPA enforcement decisions, new EDPB guidelines, or news about fines.

        Args:
            query: Search query, e.g. 'GDPR fines 2024 largest'.
        """
        try:
            from langchain_community.tools import DuckDuckGoSearchRun
            return DuckDuckGoSearchRun().run(query)
        except Exception as e:
            return f"Web search failed: {e}"

    def get_gdpr_article(article_number: str) -> str:
        """Get the key provisions of a specific GDPR article by number.

        Covers Articles 4, 5, 6, 7, 9, 13, 17, 20, 21, 25, 28, 30, 32, 33, 34, 35, 37, 44.

        Args:
            article_number: Article number as string, e.g. '6' or '35'.
        """
        article_number = article_number.strip().lstrip("0") or "0"
        article = GDPR_ARTICLES.get(article_number)
        if article:
            return article
        available = ", ".join(sorted(GDPR_ARTICLES.keys(), key=int))
        return f"Article {article_number} not in index. Available: {available}. Use search_gdpr_documents instead."

    agent = LlmAgent(
        # ADK uses the Vertex AI model name directly
        model=settings.gemini_model,
        name="gdpr_analyst_adk",
        description="Expert GDPR legal analyst with access to GDPR documents and web search",
        instruction=SYSTEM_PROMPT,
        tools=[search_gdpr_documents, web_search, get_gdpr_article],
    )

    return agent


def ask_adk(question: str, session_id: str = "default") -> str:
    """
    Ask the ADK agent a question. Returns the answer as a string.

    ADK uses an async runner internally, so this function wraps the async call.
    In production you would use the async API directly.

    Args:
        question:   The user's question.
        session_id: Session identifier for conversation continuity.
    """
    return asyncio.run(_ask_adk_async(question, session_id))


async def _ask_adk_async(question: str, session_id: str) -> str:
    """Async implementation of ask_adk."""
    try:
        from google.adk.runners import Runner
        from google.adk.sessions import InMemorySessionService
        from google.genai.types import Content, Part
    except ImportError:
        raise ImportError(
            "google-adk is not installed. Run: pip install google-adk"
        )

    agent = _build_adk_agent()
    session_service = InMemorySessionService()
    runner = Runner(
        agent=agent,
        app_name="gdpr_agent",
        session_service=session_service,
    )

    # Create or resume session
    session = await session_service.create_session(
        app_name="gdpr_agent",
        user_id="user",
        session_id=session_id,
    )

    user_message = Content(role="user", parts=[Part(text=question)])

    answer_parts = []
    async for event in runner.run_async(
        user_id="user",
        session_id=session.id,
        new_message=user_message,
    ):
        if event.is_final_response() and event.content:
            for part in event.content.parts:
                if hasattr(part, "text") and part.text:
                    answer_parts.append(part.text)

    return "".join(answer_parts) or "No response generated."
