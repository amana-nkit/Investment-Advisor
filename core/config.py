"""
core/config.py
Typed, validated settings loaded from environment variables.
Never access os.getenv() directly elsewhere — use get_settings() instead.

extra="ignore" means unknown keys in .env (typos, tool vars, etc.)
are silently skipped instead of raising a ValidationError.
"""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        # Silently ignore extra keys present in .env that are not declared
        # as fields below (e.g. OPEN_API_KEY typos, unrelated tool vars).
        extra="ignore",
    )

    # LLM — reads OPENAI_API_KEY from .env
    openai_api_key: str
    llm_model: str = "gpt-4.1"
    embedding_model: str = "text-embedding-3-small"
    llm_temperature: float = 0.3

    # Database (L3 memory)
    database_url: str = "sqlite:///./investment_advisor.db"

    # ChromaDB (L4 memory)
    chroma_path: str = "./chroma_db"

    # Email (SendGrid)
    sendgrid_api_key: str = ""
    email_from: str = "advisor@yourdomain.com"

    # File upload limits
    max_pdf_mb: int = 10

    # RAG settings
    chunk_size: int = 1000
    chunk_overlap: int = 200
    retrieval_top_k: int = 5
    conversation_window_k: int = 10

    # Logging
    log_level: str = "INFO"
    log_file: str = "logs/app.log"


@lru_cache
def get_settings() -> Settings:
    return Settings()
