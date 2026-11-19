"""
REST API routes.

All endpoints are synchronous (`def`, not `async def`): FastAPI runs them in a
thread pool, so CPU-heavy work such as PDF parsing and model inference does
not block the server's event loop.
"""

from fastapi import APIRouter, File, Request, UploadFile

from app.config import Settings
from app.schemas.request_response import (
    AskRequest,
    AskResponse,
    DeleteResponse,
    DocumentDetailsResponse,
    ErrorResponse,
    HealthResponse,
    RetrievedChunkDebug,
    SourceInfo,
    UploadResponse,
)
from app.services.document_service import process_pdf_upload
from app.services.model_service import ModelService
from app.services.qa_service import QAService
from app.storage.document_store import DocumentStore

router = APIRouter(prefix="/api")


# Shared objects live on `app.state` (created in main.py). Small helpers keep
# the route code readable and let tests inject fakes.
def _settings(request: Request) -> Settings:
    return request.app.state.settings


def _model(request: Request) -> ModelService:
    return request.app.state.model_service


def _store(request: Request) -> DocumentStore:
    return request.app.state.document_store


def _qa(request: Request) -> QAService:
    return request.app.state.qa_service


def _errors(*codes: int) -> dict:
    return {code: {"model": ErrorResponse} for code in codes}


# ---------------------------------------------------------------------------
# A. Health check
# ---------------------------------------------------------------------------
@router.get("/health", response_model=HealthResponse, tags=["System"])
def health(request: Request) -> HealthResponse:
    """Report backend status, model loading status, model name and device."""
    info = _model(request).info()
    return HealthResponse(
        status="ok" if info["model_loaded"] else "degraded",
        documents_in_memory=len(_store(request)),
        **info,
    )


# ---------------------------------------------------------------------------
# B. Upload PDF
# ---------------------------------------------------------------------------
@router.post(
    "/documents/upload",
    response_model=UploadResponse,
    status_code=201,
    tags=["Documents"],
    responses=_errors(400, 413, 415, 422, 503),
)
def upload_document(request: Request, file: UploadFile = File(..., description="PDF file")) -> UploadResponse:
    """Upload a PDF, extract its text page by page, chunk it and build the TF-IDF index."""
    settings = _settings(request)
    # Read at most limit + 1 bytes: enough to detect an oversized file
    # without loading an arbitrarily large upload into memory.
    data = file.file.read(settings.max_upload_size_bytes + 1)
    try:
        document = process_pdf_upload(
            filename=file.filename,
            content_type=file.content_type,
            data=data,
            settings=settings,
            model_service=_model(request),
            store=_store(request),
        )
    finally:
        file.file.close()

    message = f"PDF processed successfully: {document.page_count} page(s), {document.num_chunks} chunk(s)."
    if document.empty_pages:
        message += f" {len(document.empty_pages)} page(s) contained no extractable text and were skipped."
    return UploadResponse(
        document_id=document.document_id,
        filename=document.filename,
        num_pages=document.page_count,
        num_chunks=document.num_chunks,
        pages_with_text=document.pages_with_text,
        empty_pages=document.empty_pages,
        status=document.status,
        message=message,
    )


# ---------------------------------------------------------------------------
# D. Document details
# ---------------------------------------------------------------------------
@router.get(
    "/documents/{document_id}",
    response_model=DocumentDetailsResponse,
    tags=["Documents"],
    responses=_errors(404),
)
def get_document(document_id: str, request: Request) -> DocumentDetailsResponse:
    """Return metadata, page count and processing status of an uploaded document."""
    document = _store(request).get(document_id)
    return DocumentDetailsResponse(
        document_id=document.document_id,
        filename=document.filename,
        num_pages=document.page_count,
        num_chunks=document.num_chunks,
        pages_with_text=document.pages_with_text,
        empty_pages=document.empty_pages,
        total_characters=document.total_characters,
        file_size_bytes=document.file_size_bytes,
        status=document.status,
        uploaded_at=document.uploaded_at,
    )


# ---------------------------------------------------------------------------
# E. Delete document
# ---------------------------------------------------------------------------
@router.delete(
    "/documents/{document_id}",
    response_model=DeleteResponse,
    tags=["Documents"],
    responses=_errors(404),
)
def delete_document(document_id: str, request: Request) -> DeleteResponse:
    """Remove a document from the registry and delete its uploaded file."""
    document = _store(request).delete(document_id)
    return DeleteResponse(
        document_id=document.document_id,
        message=f"Document '{document.filename}' was deleted.",
    )


# ---------------------------------------------------------------------------
# C. Ask a question
# ---------------------------------------------------------------------------
@router.post(
    "/questions/ask",
    response_model=AskResponse,
    tags=["Questions"],
    responses=_errors(400, 404, 422, 500, 503),
)
def ask_question(payload: AskRequest, request: Request) -> AskResponse:
    """
    Answer a question about an uploaded document.

    `success` is false (with HTTP 200) when the request was valid but no
    reliable answer exists in the document; `status` explains why.
    """
    document = _store(request).get(payload.document_id)
    result = _qa(request).answer(payload.question, document.retriever)

    source = None
    if result.source is not None:
        source = SourceInfo(**vars(result.source))
    return AskResponse(
        success=result.success,
        status=result.status,
        document_id=document.document_id,
        question=payload.question,
        question_type=result.question_type,
        answer=result.answer,
        confidence=result.confidence,
        source=source,
        supporting_passages=[SourceInfo(**vars(sp)) for sp in result.supporting_passages],
        retrieved_chunks=[RetrievedChunkDebug(**vars(rc)) for rc in result.retrieved_chunks],
        message=result.message,
    )

