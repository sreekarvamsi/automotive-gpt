"""
Application configuration.
All settings are loaded from environment variables (or a .env file).
Import the singleton `settings` object anywhere in the codebase.
"""

import os
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # ── OpenAI ────────────────────────────────────────────────
    openai_api_key: str = Field(..., description="OpenAI API key")
    openai_model: str = Field(default="gpt-4-turbo", description="Generation model")
    openai_embedding_model: str = Field(
        default="text-embedding-3-large", description="Embedding model (3072-dim)"
    )

    # ── Pinecone ──────────────────────────────────────────────
    pinecone_api_key: str = Field(..., description="Pinecone API key")
    pinecone_environment: str = Field(default="us-east-1")
    pinecone_index_name: str = Field(default="automotive-manuals")

    # ── Cohere ────────────────────────────────────────────────
    cohere_api_key: str = Field(..., description="Cohere API key")
    cohere_rerank_model: str = Field(default="rerank-v3")

    # ── PostgreSQL ────────────────────────────────────────────
    postgres_host: str = Field(default="localhost")
    postgres_port: int = Field(default=5432)
    postgres_db: str = Field(default="automotive_gpt")
    postgres_user: str = Field(default="postgres")
    postgres_password: str = Field(..., description="PostgreSQL password")

    # ── Redis ─────────────────────────────────────────────────
    redis_url: str = Field(default="redis://localhost:6379/0")

    # ── LangSmith (optional) ─────────────────────────────────
    langsmith_api_key: str | None = Field(default=None)
    langchain_tracing_v2: bool = Field(default=False)

    # ── Chunking & Retrieval ──────────────────────────────────
    chunk_size: int = Field(default=512)
    chunk_overlap: int = Field(default=50)
    top_k_retrieval: int = Field(default=10)
    rerank_top_n: int = Field(default=5)

    # ── Derived ───────────────────────────────────────────────
    @property
    def postgres_url(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    model_config = {"env_file": ".env", "extra": "ignore"}


# Singleton — import this everywhere
settings = Settings()
