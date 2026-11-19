"""
Document processing: turns an uploaded PDF into a searchable, question-ready document.

    upload bytes ─► validate ─► extract pages ─► token-aware chunks ─► TF-IDF index ─► store
"""

import logging
import uuid
from pathlib import Path

from app.config import Settings
from app.exceptions import NoExtractableTextError
from app.services.model_service import ModelService
from app.services.pdf_service import extract_text_from_pdf, validate_pdf_upload
from app.services.retrieval_service import TfidfRetriever
from app.storage.document_store import DocumentStore, StoredDocument
from app.utils.text_chunking import chunk_pages

logger = logging.getLogger(__name__)


def process_pdf_upload(
    filename: str | None,
    content_type: str | None,
    data: bytes,
    settings: Settings,
    model_service: ModelService,
    store: DocumentStore,
) -> StoredDocument:
    """Validate, extract, chunk, index and register an uploaded PDF."""
    # The tokenizer is needed for token-aware chunking.
    model_service.ensure_ready()

    validate_pdf_upload(filename, content_type, data, settings.max_upload_size_bytes)
    extracted = extract_text_from_pdf(data)

    chunks = chunk_pages(
        extracted.pages,
        tokenizer=model_service.tokenizer,
        chunk_size=settings.chunk_size_tokens,
        overlap=settings.chunk_overlap_tokens,
    )
    if not chunks:
        raise NoExtractableTextError()
    retriever = TfidfRetriever(chunks)

    # Save the original file under a generated name; the client's filename is
    # never used as a path, which prevents path-traversal attacks.
    document_id = str(uuid.uuid4())
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = upload_dir / f"{document_id}.pdf"
    file_path.write_bytes(data)

    # Keep only the base name of the client's filename (handles "C:\\x\\a.pdf" too).
    display_name = Path((filename or "document.pdf").replace("\\", "/")).name

    document = StoredDocument(
        document_id=document_id,
        filename=display_name,
        file_path=file_path,
        file_size_bytes=len(data),
        page_count=extracted.page_count,
        pages=extracted.pages,
        empty_pages=extracted.empty_pages,
        chunks=chunks,
        retriever=retriever,
        status="ready",
    )
    store.add(document)
    logger.info(
        "Document %s processed: %d pages (%d empty), %d chunks, %d TF-IDF terms",
        document_id, extracted.page_count, len(extracted.empty_pages), len(chunks), retriever.vocabulary_size,
    )
    return document
