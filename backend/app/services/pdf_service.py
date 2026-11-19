"""
PDF processing service.

Responsibilities:
  1. Validate an uploaded file (extension, content type, size, PDF signature).
  2. Open it with PyMuPDF and extract text page by page (page numbers from 1).
  3. Clean the extracted text and detect empty / scanned PDFs.

Chunking is done separately in `app/utils/text_chunking.py`.
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import pymupdf  # PyMuPDF (historically imported as "fitz", which is now deprecated)

from app.exceptions import (
    CorruptedPDFError,
    EmptyFileError,
    EmptyPDFError,
    EncryptedPDFError,
    FileTooLargeError,
    InvalidFileTypeError,
    NoExtractableTextError,
)
from app.utils.text_chunking import PageText

logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {".pdf"}
# Browsers send "application/pdf". Some HTTP clients send the generic
# "application/octet-stream"; that is accepted because the PDF signature check
# below verifies the real file content anyway.
ALLOWED_CONTENT_TYPES = {
    "application/pdf",
    "application/x-pdf",
    "application/acrobat",
    "application/octet-stream",
}
PDF_SIGNATURE = b"%PDF-"
# The PDF specification allows the header to appear within the first 1024 bytes.
SIGNATURE_SEARCH_BYTES = 1024


@dataclass
class ExtractedDocument:
    """Result of PDF text extraction."""

    page_count: int
    pages: List[PageText]  # one entry per PDF page, including empty pages
    empty_pages: List[int] = field(default_factory=list)

    @property
    def pages_with_text(self) -> List[PageText]:
        return [p for p in self.pages if p.text]

    @property
    def total_characters(self) -> int:
        return sum(len(p.text) for p in self.pages)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate_pdf_upload(
    filename: str | None,
    content_type: str | None,
    data: bytes,
    max_size_bytes: int,
) -> None:
    """Raise an AppError subclass if the upload is not an acceptable PDF."""
    if not filename:
        raise InvalidFileTypeError("No filename was provided. Please upload a .pdf file.")

    extension = Path(filename).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise InvalidFileTypeError(
            f"Invalid file type '{extension or 'none'}'. Only .pdf files are accepted."
        )

    # Content-Type may carry parameters, e.g. "application/pdf; charset=binary".
    media_type = (content_type or "").split(";")[0].strip().lower()
    if media_type and media_type not in ALLOWED_CONTENT_TYPES:
        raise InvalidFileTypeError(
            f"Invalid content type '{media_type}'. Only PDF files (application/pdf) are accepted."
        )

    if len(data) == 0:
        raise EmptyFileError()

    if len(data) > max_size_bytes:
        # Only (limit + 1) bytes are read from the upload, so the real size is unknown here.
        raise FileTooLargeError(
            f"The file exceeds the maximum allowed size of {max_size_bytes // (1024 * 1024)} MB."
        )

    if PDF_SIGNATURE not in data[:SIGNATURE_SEARCH_BYTES]:
        raise InvalidFileTypeError(
            "The file does not look like a real PDF (missing '%PDF-' signature)."
        )


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------
_HYPHEN_LINEBREAK = re.compile(r"(\w)-\n(\w)")
_WHITESPACE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """
    Normalise raw PDF text:
      * re-join words hyphenated across line breaks ("informa-\\ntion" -> "information")
      * collapse line breaks, tabs and repeated spaces into single spaces
      * remove NUL characters that some PDFs contain
    """
    if not text:
        return ""
    text = text.replace("\x00", "")
    text = _HYPHEN_LINEBREAK.sub(r"\1\2", text)
    text = _WHITESPACE.sub(" ", text)
    return text.strip()


def extract_text_from_pdf(data: bytes) -> ExtractedDocument:
    """
    Extract cleaned text from every page of a PDF.

    Raises:
        CorruptedPDFError: the bytes cannot be parsed as a PDF.
        EncryptedPDFError: the PDF requires a password.
        EmptyPDFError: the PDF has zero pages.
        NoExtractableTextError: no page contains selectable text (e.g. scanned PDF).
    """
    try:
        document = pymupdf.open(stream=data, filetype="pdf")
    except Exception as exc:  # PyMuPDF raises several different error types
        logger.warning("Failed to open PDF: %s", exc)
        raise CorruptedPDFError() from exc

    try:
        if document.needs_pass:
            raise EncryptedPDFError()

        page_count = document.page_count
        if page_count == 0:
            raise EmptyPDFError()

        pages: List[PageText] = []
        empty_pages: List[int] = []
        for index in range(page_count):
            page_number = index + 1  # human-friendly page numbers start at 1
            try:
                raw = document.load_page(index).get_text("text")
            except Exception as exc:
                # A single damaged page should not break the whole document.
                logger.warning("Could not read page %d: %s", page_number, exc)
                raw = ""
            text = clean_text(raw)
            if not text:
                empty_pages.append(page_number)
            pages.append(PageText(page_number=page_number, text=text))
    except (EncryptedPDFError, EmptyPDFError):
        raise
    except Exception as exc:
        logger.warning("Failed while reading PDF pages: %s", exc)
        raise CorruptedPDFError() from exc
    finally:
        document.close()

    extracted = ExtractedDocument(page_count=page_count, pages=pages, empty_pages=empty_pages)
    if not extracted.pages_with_text:
        raise NoExtractableTextError()
    return extracted
