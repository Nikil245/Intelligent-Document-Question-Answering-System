"""
Application configuration.

All settings can be overridden with environment variables or a `.env` file
(see `.env.example`). Pydantic validates every value at startup, so a typo in
the configuration fails fast with a readable error.
"""

from functools import lru_cache
from pathlib import Path
from typing import List, Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict
from typing_extensions import Annotated

# Folder that contains the `app` package (i.e. the `backend/` directory).
BASE_DIR = Path(__file__).resolve().parent.parent

# The ONLY model used by this project (mandated by the assignment).
DEFAULT_MODEL_NAME = "distilbert/distilbert-base-cased-distilled-squad"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        # Allow a field called "model_name" without Pydantic's protected namespace warning.
        protected_namespaces=(),
    )

    # ---- General ---------------------------------------------------------
    app_name: str = "Intelligent Document Question Answering System"
    app_version: str = "1.0.0"
    host: str = "127.0.0.1"
    port: int = 8000
    reload: bool = False
    log_level: str = "INFO"

    # ---- Model -----------------------------------------------------------
    model_name: str = DEFAULT_MODEL_NAME
    # "auto" -> CUDA if available, otherwise CPU. "cpu"/"cuda" force a device.
    device: Literal["auto", "cpu", "cuda"] = "auto"
    # Set to false only for testing; the API then reports the model as not loaded.
    load_model_on_startup: bool = True

    # ---- Uploads ---------------------------------------------------------
    upload_dir: Path = BASE_DIR / "uploads"
    max_upload_size_mb: int = Field(default=20, ge=1, le=200)
    # Delete uploaded PDFs of registered documents when the server stops
    # (the registry is in-memory, so the files would be orphaned anyway).
    cleanup_uploads_on_shutdown: bool = True

    # ---- Chunking (in DistilBERT tokens) ---------------------------------
    # DistilBERT accepts at most 512 tokens for [CLS] question [SEP] context [SEP].
    model_max_tokens: int = 512
    chunk_size_tokens: int = Field(default=384, ge=32)
    chunk_overlap_tokens: int = Field(default=64, ge=0)
    max_question_tokens: int = Field(default=64, ge=8)

    # ---- Retrieval -------------------------------------------------------
    top_k_chunks: int = Field(default=5, ge=1, le=10)
    # Chunks whose cosine similarity is <= this value are treated as irrelevant.
    min_retrieval_similarity: float = Field(default=0.0, ge=0.0, lt=1.0)

    # ---- Question answering ----------------------------------------------
    # Answers whose model score is below this threshold are reported as
    # "no reliable answer" instead of being returned as facts.
    confidence_threshold: float = Field(default=0.1, ge=0.0, le=1.0)
    max_answer_length_tokens: int = Field(default=50, ge=1, le=100)
    # Characters of context shown on each side of the answer in the source passage.
    passage_window_chars: int = Field(default=300, ge=50)

    # ---- CORS ------------------------------------------------------------
    # NoDecode lets us accept a simple comma-separated string from .env.
    cors_origins: Annotated[List[str], NoDecode] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, value):
        if isinstance(value, str):
            value = value.strip()
            if value.startswith("["):  # also allow a JSON list
                import json

                return json.loads(value)
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value

    @field_validator("upload_dir", mode="after")
    @classmethod
    def _resolve_upload_dir(cls, value: Path) -> Path:
        # Relative paths are interpreted relative to the backend/ folder,
        # regardless of the directory the server is started from.
        return value if value.is_absolute() else (BASE_DIR / value).resolve()

    @model_validator(mode="after")
    def _check_token_budget(self):
        # [CLS] question [SEP] context [SEP]  -> 3 special tokens.
        needed = self.chunk_size_tokens + self.max_question_tokens + 3
        if needed > self.model_max_tokens:
            raise ValueError(
                f"chunk_size_tokens ({self.chunk_size_tokens}) + max_question_tokens "
                f"({self.max_question_tokens}) + 3 special tokens = {needed} exceeds the "
                f"model limit of {self.model_max_tokens} tokens."
            )
        if self.chunk_overlap_tokens >= self.chunk_size_tokens:
            raise ValueError("chunk_overlap_tokens must be smaller than chunk_size_tokens.")
        return self

    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance (read once per process)."""
    return Settings()
