"""Pydantic models that define (and validate) the API's requests and responses."""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Common
# ---------------------------------------------------------------------------
class ErrorResponse(BaseModel):
    success: bool = False
    error: str = Field(..., description="Machine-readable error code")
    message: str = Field(..., description="Human-readable explanation")
    details: Optional[List[str]] = None


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
class HealthResponse(BaseModel):
    status: str = Field(..., description="'ok' when the model is ready, otherwise 'degraded'")
    model_loaded: bool
    model_status: str = Field(..., description="not_loaded | loading | loaded | failed")
    model_name: str
    device: str = Field(..., description="'cuda' or 'cpu'")
    device_name: Optional[str] = None
    model_error: Optional[str] = None
    documents_in_memory: int


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------
class UploadResponse(BaseModel):
    success: bool = True
    document_id: str
    filename: str
    num_pages: int
    num_chunks: int
    pages_with_text: int
    empty_pages: List[int]
    status: str
    message: str


class DocumentDetailsResponse(BaseModel):
    success: bool = True
    document_id: str
    filename: str
    num_pages: int
    num_chunks: int
    pages_with_text: int
    empty_pages: List[int]
    total_characters: int
    file_size_bytes: int
    status: str
    uploaded_at: datetime


class DeleteResponse(BaseModel):
    success: bool = True
    document_id: str
    message: str


# ---------------------------------------------------------------------------
# Questions
# ---------------------------------------------------------------------------
class AskRequest(BaseModel):
    document_id: str = Field(..., min_length=1, max_length=100, examples=["3f2b6c1e-...-uuid"])
    question: str = Field(
        ...,
        max_length=500,
        examples=["What is the main topic of this document?"],
    )

    @field_validator("document_id")
    @classmethod
    def _strip_document_id(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("document_id must not be empty")
        return value

    @field_validator("question")
    @classmethod
    def _question_not_empty(cls, value: str) -> str:
        value = " ".join(value.split())  # trim and collapse whitespace
        if not value:
            raise ValueError("Question must not be empty")
        return value


class SourceInfo(BaseModel):
    page_number: int = Field(..., description="Original PDF page number (starting at 1)")
    passage: str = Field(..., description="Supporting passage from the PDF around the answer")
    answer_start: Optional[int] = Field(None, description="Start offset of the answer inside 'passage'")
    answer_end: Optional[int] = Field(None, description="End offset of the answer inside 'passage'")
    chunk_id: int
    retrieval_score: float = Field(..., description="TF-IDF cosine similarity of this chunk")
    answer_span: Optional[str] = Field(None, description="Extracted span inside this passage")
    confidence: Optional[float] = Field(None, description="Model confidence score for this passage's span")


class RetrievedChunkDebug(BaseModel):
    rank: int
    chunk_id: int
    page_number: int
    retrieval_score: float
    raw_tfidf_score: float
    concept_coverage: float
    matched_concepts: List[str] = Field(default_factory=list)
    num_tokens: int
    is_heading_only: bool
    candidate_answer: Optional[str] = None
    candidate_score: Optional[float] = None
    text_preview: str


class AskResponse(BaseModel):
    success: bool
    status: str = Field(
        ..., description="answered | low_confidence | no_relevant_content | no_answer"
    )
    document_id: str
    question: str
    question_type: str = Field("factual", description="factual | descriptive")
    answer: Optional[str] = None
    confidence: Optional[float] = Field(
        None, description="DistilBERT confidence score (0-1), not a calibrated probability"
    )
    source: Optional[SourceInfo] = None
    supporting_passages: List[SourceInfo] = Field(default_factory=list)
    retrieved_chunks: List[RetrievedChunkDebug] = Field(default_factory=list)
    message: str

