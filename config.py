from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # GCP
    gcp_project_id: str
    gcp_region: str = "europe-west4"       # BigQuery + embeddings (EU data sovereignty)
    llm_region: str = "europe-west1"       # Gemini — wider model availability, still EU

    # BigQuery — vector store (serverless, no persistent endpoint cost)
    bq_dataset: str = "gdpr_agent"
    bq_table: str = "document_chunks"

    # Models
    gemini_model: str = "gemini-1.5-pro"
    embedding_model: str = "text-embedding-004"

    # RAG chunking
    chunk_size: int = 1000
    chunk_overlap: int = 200
    retrieval_top_k: int = 6


settings = Settings()
