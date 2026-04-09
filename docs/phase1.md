# Phase 1 — RAG Foundation

Phase 1 builds the core of the GDPR Legal Analyst Agent: a Retrieval Augmented Generation (RAG) pipeline that lets you ask natural-language questions about GDPR documents and receive grounded, cited answers.

**Frameworks learned:** BigQuery Vector Search, Vertex AI `text-embedding-004`, Gemini 2.5 Flash Lite, LangChain LCEL

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│  INGESTION  (one-time, run with: python -m phase1.ingest)               │
│                                                                         │
│  GDPR PDFs                                                              │
│      │                                                                  │
│      ▼                                                                  │
│  PyPDFLoader ──► RecursiveCharacterTextSplitter ──► chunks              │
│  (load pages)    (1000 chars, 200 overlap)           (list[Document])   │
│                                                          │              │
│                                                          ▼              │
│                                                 VertexAIEmbeddings      │
│                                                 (text-embedding-004)    │
│                                                 768-dim vectors         │
│                                                          │              │
│                                                          ▼              │
│                                                 BigQueryVectorStore     │
│                                                 (document_chunks table) │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│  QUERY  (every question, run with: python -m phase1.main)               │
│                                                                         │
│  User question (string)                                                 │
│      │                                                                  │
│      ├─────────────────────────────────────────────────────┐           │
│      │                                                      │           │
│      ▼                                                      │           │
│  VertexAIEmbeddings                                 RunnablePassthrough │
│  (embed the question)                               (question unchanged)│
│      │                                                      │           │
│      ▼                                                      │           │
│  BigQuery VECTOR_SEARCH()                                   │           │
│  top-6 most similar chunks                                  │           │
│      │                                                      │           │
│      ▼                                                      │           │
│  format_docs()  ◄────────────────────────────────── RunnableParallel   │
│  (numbered list with source citations)                      │           │
│      │                                                      │           │
│      └──────────────────┬───────────────────────────────────┘           │
│                         ▼                                               │
│                 ChatPromptTemplate                                      │
│                 (system: GDPR analyst persona + context                 │
│                  human:  the original question)                         │
│                         │                                               │
│                         ▼                                               │
│                 ChatVertexAI (Gemini 2.5 Flash Lite)                    │
│                 temperature=0, max_tokens=2048                          │
│                         │                                               │
│                         ▼                                               │
│                 StrOutputParser                                          │
│                 (strips to plain string / streams tokens)               │
│                         │                                               │
│                         ▼                                               │
│                 Answer with citations                                   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Core concepts

### RAG — Retrieval Augmented Generation

**The problem RAG solves:** Large language models like Gemini are trained on general internet text up to a cutoff date. They have no knowledge of your specific documents, internal policies, or recent regulatory updates. You could fine-tune the model on your documents, but that is expensive and the model still cannot cite where its knowledge came from.

**How RAG works:** Instead of baking knowledge into the model, you retrieve it at query time. When a user asks a question:

1. You convert the question to a vector (embedding).
2. You search your document store for chunks whose vectors are most similar to the question vector.
3. You inject those chunks into the prompt as context.
4. The model generates an answer grounded in those chunks — and can cite them.

The model becomes a reasoning engine over your documents, not a memoriser of them. You can update the document store without touching the model. You get citations because you know exactly which chunks were retrieved.

**Why this matters for GDPR work specifically:** Legal text has precise meaning. "Legitimate interest" has a specific definition in Article 6. If the model answers from general training data, it might be correct in spirit but imprecise about which article applies. RAG forces it to quote the actual regulatory text.

---

### Embeddings

**What they are:** An embedding model takes text and outputs a list of numbers — a vector. The key property is that semantically similar texts produce vectors that are close together in vector space (measured by dot product or cosine similarity).

**The model: `text-embedding-004`** — Google's current best-in-class embedding model. It produces 768-dimensional vectors and is multilingual, which matters for GDPR work where you might have guidance in German, French, or other EU languages. The embedding model is separate from the generative model — it is a smaller, encoder-only model optimised specifically for converting text to vectors.

**Why 768 dimensions?** More dimensions = more capacity to encode nuance, but also more storage and more compute per similarity search. 768 is a well-established sweet spot for sentence-level and paragraph-level embeddings. It is the same dimension as the original BERT model.

**Important:** The same model must be used at ingestion time (when you embed document chunks) and at query time (when you embed the user's question). If you switch embedding models, you must re-embed all your documents — the vector spaces are not compatible across models.

---

### Vector search

**The task:** Given a query vector (the embedded question), find the K document vectors most similar to it. In Phase 1, K=6.

**The math:** Similarity is measured by dot product distance. Vectors that point in the same direction (high dot product) are semantically similar. The `VECTOR_SEARCH()` function in BigQuery computes this across all stored vectors and returns the top K.

**Approximate vs. exact:** At small scale (thousands of chunks), BigQuery uses brute-force exact search. At larger scale it switches to approximate nearest-neighbour algorithms (like ScaNN) that trade a small amount of accuracy for much faster search. This switch is transparent — you do not change your query.

**What "top 6" means in practice:** The agent retrieves the 6 document chunks most relevant to the question, regardless of whether they are from the same document, the same article, or even the same topic. It is purely similarity-based. If the question is "what is legitimate interest?", the top chunks will be the passages in your GDPR documents that talk about legitimate interest.

---

### LangChain LCEL (Expression Language)

**The problem it solves:** Composing LLM pipelines used to mean writing glue code — call the retriever, format the results, build a prompt string, call the LLM, parse the output. LCEL replaces this with a pipe syntax where each step is a `Runnable` object that knows how to receive input and pass output to the next step.

**The pipe pattern:**

```python
chain = (
    RunnableParallel({"context": retriever | format_docs, "question": RunnablePassthrough()})
    | prompt
    | llm
    | StrOutputParser()
)
```

Reading left to right: the input (a question string) goes into `RunnableParallel`, which runs the retriever (to get context) and a passthrough (to preserve the question) simultaneously, merges them into a dict, passes that to the prompt template, the filled prompt to the LLM, and the LLM response to the output parser.

**Why `RunnableParallel`?** The prompt template needs two things: the retrieved context and the original question. `RunnableParallel` lets you run both branches — the retriever branch and the passthrough branch — and merge the results before handing off to the prompt.

**Why this matters architecturally:** Every step is swappable. You can replace `BigQueryVectorStore.as_retriever()` with a different retriever, swap Gemini for Claude or a self-hosted model, or add a re-ranking step between the retriever and the prompt — all without touching the rest of the chain. This same composability is why LCEL chains map cleanly to LangGraph nodes in Phase 2.

**Streaming:** LCEL chains support `.stream()` natively. When you call `chain.stream(question)`, each token from Gemini is yielded as it arrives rather than waiting for the full response. This is what enables the streaming CLI in `main.py`.

---

### BigQuery Vector Search

**What it is:** A SQL function (`VECTOR_SEARCH()`) built into BigQuery that finds the nearest neighbours of a query vector in a column of stored vectors. LangChain's `BigQueryVectorStore` wraps this — you call `.as_retriever()` and it handles the SQL.

**Under the hood:** Your document chunks are stored in a BigQuery table (`document_chunks`) with a column for the raw text, a column for the embedding vector, and columns for metadata (source file, page number). When you retrieve, BigQuery runs a `VECTOR_SEARCH()` query against that table.

**Why BigQuery instead of Vertex AI Vector Search?**

This is the most important architectural trade-off to understand for the interview.

| | BigQuery Vector Search | Vertex AI Vector Search |
|-|------------------------|------------------------|
| Pricing model | Pay per query (first 1 TB/month free) | Dedicated index server — hourly charge even when idle |
| Latency | ~1–3 seconds (cold query) | ~10–50ms (persistent index in memory) |
| Scale | Up to ~10M rows comfortably | Designed for hundreds of millions of vectors |
| Setup | Table created automatically | Requires creating an Index and IndexEndpoint |
| Idle cost | $0 | ~$65–$200/month |

For a learning project with occasional queries, BigQuery is unambiguously the right choice. For a production system with high query volume (thousands of requests per minute) where sub-100ms latency is required, you would migrate to Vertex AI Vector Search. Being able to articulate this trade-off — and when each makes sense — is exactly the kind of reasoning a senior architect is expected to demonstrate.

---

### Data sovereignty and GDPR

All GCP calls in this project are pinned to `europe-west4` (Netherlands):

- The BigQuery dataset is created in `europe-west4`
- Vertex AI embedding calls use the `europe-west4` regional endpoint (`GCP_REGION`)
- Gemini calls use `LLM_REGION`, also set to `europe-west4` — this is a separate setting because model availability varies by region. If a model isn't available in your preferred region you can point `LLM_REGION` at another EU region (e.g. `europe-west1`) without moving your data.
- Data processed by these services never leaves EU infrastructure

**Why this matters:** GDPR Chapter V (Articles 44–49) restricts the transfer of personal data to third countries. If your AI pipeline sends EU personal data to a US-region API endpoint, you have a potential cross-border transfer issue that requires legal justification (adequacy decision, SCCs, etc.). Keeping everything in `europe-west4` is the simplest baseline control — no transfer, no problem.

In Phase 6 this goes further: VPC Service Controls will create a security perimeter around your GCP resources, and CMEK will let you control the encryption keys for data at rest.

---

## File walk-through

### `config.py`

```python
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    gcp_project_id: str
    gcp_region: str = "europe-west4"   # BigQuery + embeddings
    llm_region: str = "europe-west4"   # Gemini — can differ if model unavailable in region
    bq_dataset: str = "gdpr_agent"
    bq_table: str = "document_chunks"
    gemini_model: str = "gemini-2.5-flash-lite"
    embedding_model: str = "text-embedding-004"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    retrieval_top_k: int = 6
```

`pydantic-settings` reads from `.env`, validates types, and raises a clear error at startup if required values are missing. The entire codebase imports `from config import settings` — one place to change any setting. Note that `gcp_project_id` has no default, which means the app fails fast if you forget to set it.

The chunking and retrieval parameters (`chunk_size`, `chunk_overlap`, `retrieval_top_k`) are surfaced here so you can tune them without touching code.

---

### `setup/create_dataset.py`

Creates the BigQuery dataset in `europe-west4`. The critical line:

```python
dataset.location = settings.gcp_region  # "europe-west4"
```

BigQuery dataset location is immutable after creation. Getting this right once means all data written to this dataset stays in the EU. You run this script once; after that it checks if the dataset already exists and exits cleanly.

---

### `phase1/ingest.py`

Three main functions:

**`load_documents()`** — Uses LangChain's `PyPDFLoader` for PDFs and `TextLoader` for `.txt` files. Enriches metadata with `source_file` and `doc_type` for every page, which flows through to the final answer as citations.

**`chunk_documents()`** — Uses `RecursiveCharacterTextSplitter` with:
- `chunk_size=1000` characters
- `chunk_overlap=200` characters
- `separators=["\n\n", "\n", ". ", " ", ""]` — tries paragraph breaks before sentence breaks before word breaks

The overlap is important for legal text. Consider a definition that spans two paragraphs. Without overlap, a chunk boundary in the middle of that definition would leave each chunk without the full context. With 200 characters of overlap, the end of one chunk and the start of the next share text, preserving cross-boundary context.

**`ingest()`** — The main command. Calls the above, then:
1. Initialises `VertexAIEmbeddings` (this authenticates to Vertex AI via ADC)
2. Gets a `BigQueryVectorStore` pointing at your table
3. Calls `vector_store.add_documents(chunks)` — LangChain calls the embedding API in batches and writes the results to BigQuery

The table schema is created automatically on the first `add_documents()` call. If you run ingest again with new documents, it appends — it does not replace.

---

### `phase1/chain.py`

This is the core of the agent. The system prompt is carefully constructed:

```python
SYSTEM_PROMPT = """You are an expert GDPR legal analyst...
When answering questions:
1. Ground every statement in the provided context
2. Cite the specific article, recital, or guideline
3. Distinguish between hard legal requirements and best-practice recommendations
4. Flag any areas of ambiguity or where DPA interpretations differ
5. If context is insufficient, say so clearly
"""
```

Point 5 is important. Without it, LLMs tend to hallucinate an answer rather than admit uncertainty. Explicitly instructing the model to say "I don't have enough context" when the retrieved chunks are not sufficient is a basic form of RAG quality control.

The `format_docs()` function labels each chunk with its source:

```
[1] Source: gdpr_text.pdf, page 23
...chunk text...

---

[2] Source: edpb_guidelines.pdf, page 7
...chunk text...
```

This formatted string goes into `{context}` in the system prompt, giving the model both the content and a citation label it can reference in its answer.

`temperature=0` for the LLM is deliberate. For legal work you want deterministic, conservative answers — not creative paraphrasing. A higher temperature would make answers more varied but less reliable.

---

### `phase1/main.py`

A Rich-powered CLI with two modes:

- **Single question mode:** `python -m phase1.main --question "..."` — answers one question and exits. Useful for scripting or testing.
- **Interactive loop:** `python -m phase1.main` — prompts for questions until you type `exit`. Prints example questions on startup.

The `_ask()` function uses `chain.stream()` when `--stream` is enabled (default). This means tokens appear in your terminal as Gemini generates them, rather than waiting for the full response — much better UX for answers that can be several paragraphs long.

---

## How to run it

### Prerequisites

- Python 3.11+
- GCP project with billing enabled
- APIs enabled: `aiplatform.googleapis.com`, `bigquery.googleapis.com`
- `gcloud auth application-default login` completed

### Step-by-step

```bash
# 1. Install dependencies
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 2. Configure
cp .env.example .env
# Edit .env: set GCP_PROJECT_ID at minimum

# 3. Create BigQuery dataset (one time)
python -m setup.create_dataset

# 4. Add GDPR documents
# Drop PDFs into data/gdpr_docs/
# Recommended: download GDPR text from EUR-Lex

# 5. Ingest
python -m phase1.ingest
# Expect: "Ingestion complete." with chunk count

# 6. Query
python -m phase1.main
```

### What to expect

Ingestion output:
```
Found 1 documents.
Loaded 88 pages/sections.
Split into 312 chunks (size=1000, overlap=200).
Initialising Vertex AI embeddings...
Connecting to BigQuery vector store...
Upserting 312 chunks into BigQuery...
Ingestion complete.
```

First query (there is a ~2 second cold-start while BigQuery warms up):
```
╭─────────────────────────────────────────────────────────╮
│ GDPR Legal Analyst                                       │
│ Phase 1 — RAG with Vertex AI Vector Search + Gemini      │
╰─────────────────────────────────────────────────────────╯
Initialising Vertex AI connection...
Ready.

Your question: What are the lawful bases for processing personal data?

Q: What are the lawful bases for processing personal data?

A: Under Article 6 of the GDPR, processing of personal data is lawful
only if at least one of the following conditions applies...
```

---

## Example questions

These are good starting points for testing the agent:

```
What are the lawful bases for processing personal data under GDPR?
What must a privacy notice include?
When is a Data Protection Impact Assessment (DPIA) required?
What are the rights of data subjects under GDPR?
What constitutes a personal data breach and what are the notification obligations?
What is the definition of "personal data" under GDPR?
How does GDPR define "special categories" of personal data?
What is the role and responsibility of a Data Protection Officer (DPO)?
What are the conditions for valid consent under GDPR?
How does the right to erasure ("right to be forgotten") work?
```

Good questions for stress-testing retrieval quality:
- Ask something very specific to a single article — the retrieved chunks should contain that article's text
- Ask something broad — the agent should retrieve chunks from multiple articles and synthesise
- Ask something not covered in your documents — the agent should say "I don't have sufficient context"

---

## What this teaches you

### What you can say in an interview

**"Walk me through your RAG implementation."**

> "I built a two-stage pipeline. At ingestion time, I use LangChain's `RecursiveCharacterTextSplitter` to chunk PDFs into 1000-character overlapping segments, embed them with Google's `text-embedding-004` (768 dimensions), and store them in BigQuery. At query time, I embed the user's question with the same model, run `VECTOR_SEARCH()` in BigQuery to retrieve the top-6 most similar chunks, inject those into a ChatPromptTemplate, and call Gemini 2.5 Flash Lite via Vertex AI. The whole chain is composed with LCEL using pipe syntax, which makes it easy to reason about data flow and swap components. I chose Flash Lite over Pro for this learning project because it's ~17x cheaper per token with acceptable quality for Q&A — in production I'd evaluate Flash and Pro against a golden dataset to pick the right cost/quality trade-off."

**"Why BigQuery and not Vertex AI Vector Search?"**

> "For a development or low-traffic project, BigQuery is strictly better — it's serverless, the first terabyte of queries per month is free, and there's no hourly charge for an idle endpoint. The trade-off is latency: BigQuery takes 1–3 seconds per vector search vs. sub-50ms for Vertex AI Vector Search, which keeps a persistent index hot in memory. I'd migrate to Vertex AI Vector Search when I need consistent low latency at scale — say, hundreds of queries per minute in a customer-facing application."

**"How do you handle GDPR compliance in an AI pipeline?"**

> "All GCP calls are pinned to `europe-west4`. I created the BigQuery dataset in that region, and Vertex AI calls use the regional endpoint. This ensures data processed by the pipeline never leaves EU infrastructure, which is the baseline control for GDPR Chapter V. In a production system I'd add VPC Service Controls to prevent exfiltration and CMEK for data at rest — I've planned Phase 6 of this project to cover exactly that."

**"What are the limitations of this Phase 1 design?"**

> "Several. First, it's a single-turn system — there's no memory of previous questions in the same session. Second, the retriever is purely similarity-based, with no re-ranking or query decomposition, so a complex multi-part question might not retrieve the right chunks for all parts. Third, there's no observability — I can't see which chunks were retrieved, how long the query took, or whether the answer was grounded. Fourth, if a question requires reasoning across multiple articles (e.g., 'what must I do when a data subject exercises their right to erasure AND there's a legitimate interest basis?'), the 6-chunk context window might not be enough. Phase 2 addresses the first two with LangGraph agents, Phase 3 addresses observability."

---

## Bridge to Phase 2

Phase 1 is a single-turn, single-step pipeline: one question → retrieve → answer. This design has fundamental limitations for real legal analysis work:

**Problem 1: No memory.** Each question starts fresh. A lawyer's workflow involves follow-up questions — "now apply that to a SaaS company" — that depend on the previous answer.

**Problem 2: No reasoning.** Some GDPR questions require multiple retrieval steps. "Does Company X need a DPO?" requires first understanding what conditions trigger a DPO requirement (Article 37), then reasoning about whether those conditions apply. A single retrieval pass might not get all the relevant information.

**Problem 3: No tool use.** The agent cannot, for example, look up whether a specific country's DPA has issued guidance on a topic, or search for recent enforcement actions.

**Phase 2** addresses these by introducing LangGraph — a framework for building stateful, graph-based agents. Instead of a fixed linear chain, you define a graph where:
- Nodes are actions (retrieve, reason, answer, reflect)
- Edges are transitions (potentially conditional — e.g., "if the answer is uncertain, retrieve more")
- State is carried across nodes, enabling multi-turn conversation and multi-step reasoning

The ReAct (Reason + Act) pattern implemented in Phase 2 lets the agent interleave reasoning steps and retrieval calls: think about what it needs, retrieve it, think about whether that's enough, retrieve more if not, then generate a final answer. This is qualitatively more powerful than single-shot RAG for complex legal analysis.
