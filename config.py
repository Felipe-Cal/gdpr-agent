from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # GCP
    gcp_project_id: str
    gcp_region: str = "europe-west4"       # BigQuery + embeddings (EU data sovereignty)
    llm_region: str = "europe-west1"       # Gemini — wider model availability, still EU

    # BigQuery — vector store (serverless, no persistent endpoint cost)
    bq_dataset: str = "gdpr_agent"
    bq_table: str = "document_chunks"

    # Models
    gemini_model: str = "gemini-2.0-flash-lite"
    embedding_model: str = "text-embedding-004"

    # RAG chunking
    chunk_size: int = 1000
    chunk_overlap: int = 200
    retrieval_top_k: int = 6

    # Phase 3 — Observability: LangSmith
    # Uses EU endpoint so query data stays in Europe
    langsmith_tracing: bool = False
    langsmith_endpoint: str = "https://eu.api.smith.langchain.com"
    langsmith_api_key: str | None = Field(default=None, validation_alias="LANGCHAIN_API_KEY")
    langsmith_project: str = "gdpr-agent"

    # Phase 3 — Observability: LangFuse
    # Self-hostable alternative — keeps traces inside your GCP perimeter (GDPR-safe)
    # In Phase 6 we deploy LangFuse on Cloud Run; set langfuse_host to that URL.
    langfuse_tracing: bool = False
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_host: str = "https://eu.cloud.langfuse.com"  # EU cloud; swap for self-hosted URL

    # Phase 3 — Evaluation
    eval_model: str = "gemini-2.0-flash-lite"  # Cheapest model sufficient for LLM-as-judge


settings = Settings()
