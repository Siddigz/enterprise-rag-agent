from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", env_ignore_empty=True)

    database_url: str = "postgresql+psycopg://recon:recon@localhost:5432/recon"
    data_dir: Path = Path("data/raw")
    reports_dir: Path = Path("reports")

    # LLM
    llm_provider: str = "anthropic"  # "anthropic" | "fake"
    agent_model: str = "claude-opus-5-5"
    agent_effort: str = "medium"
    judge_model: str = "claude-opus-5-5"
    judge_effort: str = "high"
    # Server-side refusal fallbacks ("default" routes by refusal category; "" disables)
    anthropic_fallbacks: str = "default"
    agent_max_steps: int = 8

    # Embeddings
    embedding_provider: str = "fastembed"  # "fastembed" | "hash"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384
    embedding_cache_dir: str | None = None

    # Retrieval
    retrieval_top_k: int = 8
    rrf_k: int = 60

    # Drift detection: column mappings scoring below this are sent to the LLM resolver
    drift_auto_accept: float = 0.75

    # Reconciliation tolerances
    amount_abs_tolerance: float = 0.01
    amount_rel_tolerance: float = 0.005

    # Eval
    eval_sample_size: int | None = None
    eval_concurrency: int = 4
    grounding_threshold: float = 0.95
    # Grounding alone can be gamed (quote evidence, answer nothing useful), so the gate also needs accuracy.
    answer_accuracy_threshold: float = 0.8


@lru_cache
def get_settings() -> Settings:
    return Settings()
