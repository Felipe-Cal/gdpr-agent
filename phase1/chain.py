"""
Phase 1 RAG chain: BigQuery Vector Search + Gemini via LangChain LCEL.

Architecture:
    user question
        → retriever (BigQuery VECTOR_SEARCH, top-k chunks)
        → prompt (question + retrieved context)
        → Gemini (generates answer with citations)
        → structured answer

Why LCEL (LangChain Expression Language)?
    - Pipes (|) make the data flow explicit and easy to reason about
    - Each step is a Runnable — you can swap components without rewriting the chain
    - This pattern carries over directly to Phase 2 when we add LangGraph nodes
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_google_community import BigQueryVectorStore
from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings

from config import settings

# ---------------------------------------------------------------------------
# System prompt — the heart of the GDPR analyst persona
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are an expert GDPR legal analyst with deep knowledge of:
- The EU General Data Protection Regulation (GDPR) text
- European Data Protection Board (EDPB) guidelines and recommendations
- National Data Protection Authority (DPA) decisions and enforcement actions
- Privacy by Design and Data Protection by Default principles

When answering questions:
1. Ground every statement in the provided context — do not rely on general knowledge alone
2. Cite the specific article, recital, or guideline you are referencing
3. Distinguish between hard legal requirements and best-practice recommendations
4. Flag any areas of ambiguity or where DPA interpretations differ across member states
5. If the context is insufficient to answer confidently, say so clearly

Context from GDPR documents:
{context}
"""

HUMAN_PROMPT = "{question}"


def build_chain(vector_store: BigQueryVectorStore):
    """
    Constructs the RAG chain using LCEL.

    The chain flow:
        {"context": retriever, "question": passthrough}
            → prompt
            → llm
            → output parser (strips to plain string)
    """
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": settings.retrieval_top_k},
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", HUMAN_PROMPT),
    ])

    # temperature=0: deterministic — important for legal work where consistency matters
    llm = ChatVertexAI(
        model_name=settings.gemini_model,
        project=settings.gcp_project_id,
        location=settings.llm_region,
        temperature=0,
        max_output_tokens=2048,
    )

    def format_docs(docs) -> str:
        """Formats retrieved chunks into a single context string with source labels."""
        sections = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source_file", "unknown")
            page = doc.metadata.get("page", "")
            label = f"[{i}] Source: {source}" + (f", page {page}" if page else "")
            sections.append(f"{label}\n{doc.page_content}")
        return "\n\n---\n\n".join(sections)

    # RunnableParallel runs retriever and passthrough simultaneously,
    # then merges results into {"context": ..., "question": ...} for the prompt
    chain = (
        RunnableParallel({"context": retriever | format_docs, "question": RunnablePassthrough()})
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


def get_chain():
    """Initialises GCP clients and returns a ready-to-use chain."""
    embeddings = VertexAIEmbeddings(
        model_name=settings.embedding_model,
        project=settings.gcp_project_id,
    )

    vector_store = BigQueryVectorStore(
        project_id=settings.gcp_project_id,
        dataset_name=settings.bq_dataset,
        table_name=settings.bq_table,
        location=settings.gcp_region,
        embedding=embeddings,
    )

    return build_chain(vector_store)
