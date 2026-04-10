"""
Phase 2 — Tool definitions.

Each tool is a plain Python function with a clear docstring. The docstring is
what the LLM reads to decide when and how to call the tool — write it like an
API contract, not a code comment.

Three tools:
  - search_gdpr_documents : BigQuery vector search over ingested GDPR docs
  - web_search            : DuckDuckGo for recent news, enforcement actions
  - get_gdpr_article      : Fast static lookup of key GDPR articles by number

The same functions are used by both the LangGraph agent (via @tool) and the
Google ADK agent (as plain callables) — keeping logic in one place.
"""

from langchain_core.tools import tool
from langchain_google_community import BigQueryVectorStore
from langchain_google_vertexai import VertexAIEmbeddings

from config import settings

# ---------------------------------------------------------------------------
# Lazy singletons — initialised once on first tool call, not at import time.
# This keeps startup fast and avoids GCP calls when just importing the module.
# ---------------------------------------------------------------------------
_vector_store: BigQueryVectorStore | None = None


def _get_vector_store() -> BigQueryVectorStore:
    global _vector_store
    if _vector_store is None:
        embeddings = VertexAIEmbeddings(
            model_name=settings.embedding_model,
            project=settings.gcp_project_id,
        )
        _vector_store = BigQueryVectorStore(
            project_id=settings.gcp_project_id,
            dataset_name=settings.bq_dataset,
            table_name=settings.bq_table,
            location=settings.gcp_region,
            embedding=embeddings,
        )
    return _vector_store


# ---------------------------------------------------------------------------
# GDPR article reference data
# Concise summaries of the most commonly referenced articles.
# The agent calls this tool when it needs to know what a specific article says
# before deciding whether to search the full document store.
# ---------------------------------------------------------------------------
GDPR_ARTICLES: dict[str, str] = {
    "4": """Article 4 — Definitions
Key definitions used throughout GDPR:
- Personal data: any information relating to an identified or identifiable natural person (data subject)
- Processing: any operation performed on personal data (collection, storage, use, disclosure, deletion, etc.)
- Controller: entity that determines the purposes and means of processing
- Processor: entity that processes data on behalf of a controller
- Consent: freely given, specific, informed and unambiguous indication of the data subject's wishes
- Personal data breach: breach of security leading to accidental or unlawful destruction, loss, alteration, or unauthorised disclosure
- Special categories: data revealing racial/ethnic origin, political opinions, religious beliefs, genetic/biometric data, health data, sex life/orientation""",

    "5": """Article 5 — Principles relating to processing of personal data
Processing must comply with all six principles:
1. Lawfulness, fairness and transparency — legal basis required, no deception
2. Purpose limitation — collected for specified, explicit, legitimate purposes; no further incompatible processing
3. Data minimisation — adequate, relevant and limited to what is necessary
4. Accuracy — kept accurate and up to date; inaccurate data must be erased or rectified
5. Storage limitation — kept no longer than necessary for the purpose
6. Integrity and confidentiality — appropriate security against unauthorised processing, loss or damage
Accountability: the controller is responsible for and must be able to demonstrate compliance with all principles.""",

    "6": """Article 6 — Lawfulness of processing
Processing is lawful only if at least one of these bases applies:
(a) Consent — data subject has given consent for one or more specific purposes
(b) Contract — processing necessary for a contract with the data subject, or pre-contractual steps at their request
(c) Legal obligation — processing necessary to comply with a legal obligation of the controller
(d) Vital interests — processing necessary to protect vital interests of the data subject or another person
(e) Public task — processing necessary for a task in the public interest or exercise of official authority
(f) Legitimate interests — necessary for legitimate interests of the controller or third party, unless overridden by the data subject's interests or rights
Note: (f) cannot be used by public authorities in the performance of their tasks.""",

    "7": """Article 7 — Conditions for consent
Requirements for consent to be valid:
- Controller must be able to demonstrate consent was given (accountability)
- If given in writing alongside other matters, consent request must be clearly distinguishable, intelligible and easily accessible
- Data subject has the right to withdraw consent at any time; withdrawal must be as easy as giving consent
- Withdrawing consent does not affect lawfulness of processing before withdrawal
- Consent is not freely given if there is a clear imbalance of power (e.g. employer-employee) or if consent is a condition of a contract when not necessary""",

    "9": """Article 9 — Processing of special categories of personal data
Processing of these categories is prohibited unless an exception applies:
Special categories: racial/ethnic origin, political opinions, religious/philosophical beliefs, trade union membership, genetic data, biometric data (for unique identification), health data, sex life or sexual orientation.

Exceptions include:
- Explicit consent (higher bar than Article 6 consent)
- Employment and social security law obligations
- Vital interests (when data subject is incapable of giving consent)
- Legitimate activities of non-profit bodies with political, philosophical, religious or trade union aims
- Data manifestly made public by the data subject
- Legal claims
- Substantial public interest (with proportionality safeguards)
- Medical purposes under professional secrecy
- Public health
- Archiving, research, statistics (with proportionate safeguards)""",

    "13": """Article 13 — Information to be provided where data collected from data subject
When collecting data directly from the data subject, the controller must provide at the time of collection:
- Controller identity and contact details
- DPO contact details (if applicable)
- Purposes and legal basis for processing
- Legitimate interests (if relying on Article 6(1)(f))
- Any recipients or categories of recipients
- Transfers to third countries and safeguards
- Retention period or criteria used to determine it
- Data subject rights (access, rectification, erasure, restriction, portability, objection)
- Right to withdraw consent (if relying on consent)
- Right to lodge a complaint with a supervisory authority
- Whether provision is statutory/contractual and consequences of not providing data
- Existence of automated decision-making including profiling""",

    "17": """Article 17 — Right to erasure ('right to be forgotten')
Data subjects have the right to obtain erasure of personal data without undue delay when:
- Data no longer necessary for the purpose it was collected
- Consent is withdrawn and no other legal basis exists
- Data subject objects under Article 21 and no overriding legitimate grounds exist
- Data has been unlawfully processed
- Erasure required by EU or member state law
- Data collected in relation to child information society services (Article 8)

Exceptions where erasure may be refused:
- Freedom of expression and information
- Legal obligation or public interest task
- Public health (Article 9(2)(h) and (i))
- Archiving, research, statistical purposes
- Establishment, exercise or defence of legal claims""",

    "20": """Article 20 — Right to data portability
Data subjects have the right to receive their personal data in a structured, commonly used, machine-readable format AND the right to transmit it to another controller, when:
- Processing is based on consent (Article 6(1)(a) or 9(2)(a)) OR on a contract (Article 6(1)(b))
- Processing is carried out by automated means

Right to have data transmitted directly between controllers where technically feasible.
Does not apply to processing necessary for a public interest task or exercise of official authority.
Must not adversely affect the rights and freedoms of others.""",

    "21": """Article 21 — Right to object
Data subjects have the right to object at any time to processing based on:
- Article 6(1)(e) public task, or
- Article 6(1)(f) legitimate interests

Controller must cease processing unless it can demonstrate compelling legitimate grounds which override the data subject's interests, rights and freedoms, or for the establishment, exercise or defence of legal claims.

Absolute right to object to processing for direct marketing purposes — no balancing test, must always stop.
Where research/statistics, the right to object may be restricted if processing is necessary for those tasks and proportionate.""",

    "25": """Article 25 — Data protection by design and by default
By design: Controllers must implement appropriate technical and organisational measures (e.g. pseudonymisation) designed to implement the data protection principles and protect data subject rights — both at the time of determining the means of processing AND at the time of the processing itself.

By default: Controllers must implement measures to ensure that by default only personal data which are necessary for each specific purpose of processing are processed. This applies to the amount of data collected, extent of processing, period of storage and accessibility.

Certification (Article 42) may be used to demonstrate compliance. This article cannot be contracted away.""",

    "28": """Article 28 — Processor
Controllers must only use processors providing sufficient guarantees that processing meets GDPR requirements and protects data subject rights.

Processing by a processor must be governed by a contract or legal act (Data Processing Agreement / DPA) that sets out:
- Subject-matter, duration, nature and purpose of processing
- Type of personal data and categories of data subjects
- Obligations and rights of the controller

Mandatory processor obligations in the contract:
- Process only on documented instructions from controller
- Ensure persons processing data are bound by confidentiality
- Implement security measures (Article 32)
- Respect conditions for engaging sub-processors
- Assist controller with data subject rights
- Assist with security, breach notification, DPIA obligations
- Delete or return all data at end of services
- Provide all information necessary to demonstrate compliance and allow audits""",

    "30": """Article 30 — Records of processing activities
Controllers must maintain a record of processing activities containing:
- Name and contact details of controller (and DPO)
- Purposes of processing
- Categories of data subjects and personal data
- Categories of recipients
- Third country transfers and safeguards
- Retention periods (where possible)
- Description of technical and organisational security measures (where possible)

Processors must also maintain records containing:
- Name and contact of processor and each controller
- Categories of processing for each controller
- Third country transfers and safeguards
- Security measures description

Records must be in writing (including electronic form) and made available to supervisory authority on request.

Exemption: organisations with fewer than 250 employees unless processing is likely to result in risk to rights and freedoms, processing is not occasional, or processing includes special categories or criminal conviction data.""",

    "32": """Article 32 — Security of processing
Controllers and processors must implement appropriate technical and organisational measures to ensure a level of security appropriate to the risk, including as appropriate:
(a) Pseudonymisation and encryption of personal data
(b) Ability to ensure ongoing confidentiality, integrity, availability and resilience of systems
(c) Ability to restore availability and access to data in the event of a physical or technical incident
(d) Process for regularly testing, assessing and evaluating effectiveness of technical and organisational measures

When assessing appropriate level of security, take into account: state of the art, costs of implementation, nature/scope/context/purposes of processing, risks of varying likelihood and severity.

Both controller and processor must take steps to ensure that any person acting under their authority who has access to personal data does not process it except on instructions.""",

    "33": """Article 33 — Notification of personal data breach to supervisory authority
In the event of a personal data breach, the controller must notify the competent supervisory authority without undue delay and, where feasible, not later than 72 hours after becoming aware of it.

Notification must include (at minimum):
- Nature of the breach including categories and approximate numbers of data subjects and records concerned
- Contact details of DPO or other contact point
- Likely consequences of the breach
- Measures taken or proposed to address the breach, including mitigation measures

If not possible within 72 hours, reasons for delay must be provided and information may be provided in phases.

Exception: Notification is NOT required if the breach is unlikely to result in a risk to the rights and freedoms of natural persons.

Processors must notify controllers without undue delay after becoming aware of a breach.""",

    "34": """Article 34 — Communication of personal data breach to data subjects
When a breach is likely to result in a HIGH risk to rights and freedoms, the controller must communicate the breach to the data subjects without undue delay.

Communication must describe in clear and plain language:
- Nature of the breach
- DPO or other contact point details
- Likely consequences
- Measures taken or proposed

Exceptions — communication to data subjects NOT required if:
- Controller implemented appropriate technical/organisational protection measures (e.g. encryption) that render the data unintelligible
- Controller has taken subsequent measures ensuring the high risk is no longer likely to materialise
- It would involve disproportionate effort — in which case a public communication or similar measure may be used instead

Supervisory authority can require the controller to communicate the breach to data subjects.""",

    "35": """Article 35 — Data Protection Impact Assessment (DPIA)
A DPIA is required BEFORE processing that is likely to result in a high risk to rights and freedoms, in particular when using new technologies. Required in particular for:
(a) Systematic and extensive profiling with significant effects on individuals
(b) Large-scale processing of special categories (Article 9) or criminal conviction data (Article 10)
(c) Systematic monitoring of publicly accessible areas on a large scale

DPIA must contain at minimum:
- Systematic description of envisaged processing operations and purposes
- Assessment of the necessity and proportionality of the processing
- Assessment of risks to rights and freedoms of data subjects
- Measures to address risks including safeguards and security measures

DPO (if designated) must be consulted. Where appropriate, seek views of data subjects or their representatives.

If residual risk is high after mitigation, prior consultation with supervisory authority required (Article 36).""",

    "37": """Article 37 — Designation of the Data Protection Officer (DPO)
A DPO must be designated when:
(a) Processing carried out by a public authority or body (except courts acting in judicial capacity)
(b) Core activities consist of processing operations which require regular and systematic monitoring of data subjects on a large scale
(c) Core activities consist of large-scale processing of special categories (Article 9) or criminal conviction data (Article 10)

A group of undertakings may appoint a single DPO provided they are easily accessible from each establishment. The DPO may be a staff member or contracted externally. Contact details must be published and communicated to supervisory authority.

DPO must have expert knowledge of data protection law and practice.""",

    "44": """Article 44 — General principle for transfers to third countries
Any transfer of personal data to a third country (outside EU/EEA) or international organisation may only take place if the conditions laid down in Chapter V (Articles 44–49) are complied with by both the controller and processor. This applies even to onward transfers from a third country to another third country.

The main transfer mechanisms (in order of preference):
1. Adequacy decision (Article 45) — Commission has decided the country provides adequate protection
2. Appropriate safeguards (Article 46) — e.g. Standard Contractual Clauses (SCCs), Binding Corporate Rules (BCRs)
3. Derogations for specific situations (Article 49) — e.g. explicit consent, vital interests, legal claims (last resort, cannot be used systematically)

This is directly relevant to AI systems: calling a non-EU AI API endpoint with EU personal data may constitute a third-country transfer.""",
}


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

@tool
def search_gdpr_documents(query: str) -> str:
    """Search the GDPR knowledge base for relevant legal text and guidance.

    Use this tool when you need to find specific provisions, definitions,
    or guidance from the ingested GDPR documents (regulation text, EDPB
    guidelines, DPA decisions). Returns the top matching passages with
    source citations.

    Args:
        query: A natural language search query, e.g. 'conditions for valid consent'
               or 'when is a DPIA required'. Be specific for better results.
    """
    store = _get_vector_store()
    docs = store.similarity_search(query, k=settings.retrieval_top_k)

    if not docs:
        return "No relevant documents found for this query."

    sections = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source_file", "unknown")
        page = doc.metadata.get("page", "")
        label = f"[{i}] {source}" + (f", page {page}" if page else "")
        sections.append(f"{label}\n{doc.page_content}")

    return "\n\n---\n\n".join(sections)


@tool
def web_search(query: str) -> str:
    """Search the web for recent GDPR news, enforcement actions, and regulatory updates.

    Use this tool when you need information that may not be in the ingested
    documents — recent DPA enforcement decisions, new EDPB guidelines,
    news about fines, or country-specific interpretations.

    Args:
        query: A search query, e.g. 'GDPR fines 2024 largest' or
               'EDPB guidance on AI data processing 2025'.
    """
    try:
        from langchain_community.tools import DuckDuckGoSearchRun
        search = DuckDuckGoSearchRun()
        return search.run(query)
    except Exception as e:
        return f"Web search failed: {e}. Try rephrasing the query."


@tool
def get_gdpr_article(article_number: str) -> str:
    """Get the key provisions of a specific GDPR article by its number.

    Use this tool when you know which article is relevant and want a quick
    structured summary of its requirements — faster than searching the full
    document store. Covers Articles 4, 5, 6, 7, 9, 13, 17, 20, 21, 25, 28,
    30, 32, 33, 34, 35, 37 and 44.

    Args:
        article_number: The article number as a string, e.g. '6' or '35'.
                        Do not include 'Article' in the string.
    """
    article_number = article_number.strip().lstrip("0") or "0"
    article = GDPR_ARTICLES.get(article_number)

    if article:
        return article

    available = ", ".join(sorted(GDPR_ARTICLES.keys(), key=int))
    return (
        f"Article {article_number} is not in the quick-reference index. "
        f"Available articles: {available}. "
        f"Use search_gdpr_documents to look up articles not in this index."
    )


# Convenience list for importing into agent files
TOOLS = [search_gdpr_documents, web_search, get_gdpr_article]
