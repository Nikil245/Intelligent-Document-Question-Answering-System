"""Tests for PDF validation and page-by-page text extraction."""

import pymupdf
import pytest

from app.exceptions import (
    CorruptedPDFError,
    EmptyFileError,
    EmptyPDFError,
    EncryptedPDFError,
    FileTooLargeError,
    InvalidFileTypeError,
    NoExtractableTextError,
)
from app.services.pdf_service import clean_text, extract_text_from_pdf, validate_pdf_upload
from tests.conftest import make_pdf

MB = 1024 * 1024


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
class TestValidation:
    def test_valid_pdf_passes(self, sample_pdf_bytes):
        validate_pdf_upload("notes.pdf", "application/pdf", sample_pdf_bytes, 20 * MB)

    def test_uppercase_extension_and_octet_stream_allowed(self, sample_pdf_bytes):
        validate_pdf_upload("NOTES.PDF", "application/octet-stream", sample_pdf_bytes, 20 * MB)

    @pytest.mark.parametrize("filename", ["notes.txt", "notes.docx", "notes", "notes.pdf.exe"])
    def test_wrong_extension_rejected(self, filename, sample_pdf_bytes):
        with pytest.raises(InvalidFileTypeError):
            validate_pdf_upload(filename, "application/pdf", sample_pdf_bytes, 20 * MB)

    def test_missing_filename_rejected(self, sample_pdf_bytes):
        with pytest.raises(InvalidFileTypeError):
            validate_pdf_upload(None, "application/pdf", sample_pdf_bytes, 20 * MB)

    @pytest.mark.parametrize("content_type", ["text/plain", "image/png", "application/msword"])
    def test_wrong_content_type_rejected(self, content_type, sample_pdf_bytes):
        with pytest.raises(InvalidFileTypeError):
            validate_pdf_upload("notes.pdf", content_type, sample_pdf_bytes, 20 * MB)

    def test_fake_pdf_without_signature_rejected(self):
        with pytest.raises(InvalidFileTypeError, match="signature"):
            validate_pdf_upload("notes.pdf", "application/pdf", b"just some text, not a pdf", 20 * MB)

    def test_empty_file_rejected(self):
        with pytest.raises(EmptyFileError):
            validate_pdf_upload("notes.pdf", "application/pdf", b"", 20 * MB)

    def test_file_too_large_rejected(self):
        data = b"%PDF-1.4\n" + b"0" * (2 * MB)
        with pytest.raises(FileTooLargeError):
            validate_pdf_upload("big.pdf", "application/pdf", data, 1 * MB)


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------
class TestExtraction:
    def test_extracts_text_from_every_page(self, sample_pdf_bytes):
        extracted = extract_text_from_pdf(sample_pdf_bytes)
        assert extracted.page_count == 3
        assert "centre of our solar system" in extracted.pages[0].text
        assert "capital of France is Paris" in extracted.pages[1].text
        assert "Photosynthesis" in extracted.pages[2].text

    def test_page_numbers_start_at_one_and_are_preserved(self):
        data = make_pdf(["alpha page", "beta page", "gamma page", "delta page"])
        extracted = extract_text_from_pdf(data)
        assert [p.page_number for p in extracted.pages] == [1, 2, 3, 4]
        assert "gamma" in extracted.pages[2].text and extracted.pages[2].page_number == 3

    def test_empty_pages_are_reported_and_numbering_is_kept(self):
        data = make_pdf(["first page text", None, "third page text"])
        extracted = extract_text_from_pdf(data)
        assert extracted.page_count == 3
        assert extracted.empty_pages == [2]
        assert [p.page_number for p in extracted.pages_with_text] == [1, 3]

    def test_pdf_without_text_raises_ocr_message(self):
        data = make_pdf([None, None])
        with pytest.raises(NoExtractableTextError, match="OCR"):
            extract_text_from_pdf(data)

    def test_corrupted_pdf_raises(self):
        with pytest.raises(CorruptedPDFError):
            extract_text_from_pdf(b"%PDF-1.4\nthis is not really a pdf structure at all")

    def test_truncated_pdf_raises_corrupted_or_no_text(self, sample_pdf_bytes):
        # A badly truncated file either cannot be opened or yields no text.
        with pytest.raises((CorruptedPDFError, NoExtractableTextError)):
            extract_text_from_pdf(sample_pdf_bytes[:200])

    def test_encrypted_pdf_raises(self):
        document = pymupdf.open()
        document.new_page().insert_text((72, 72), "secret text")
        data = document.tobytes(
            encryption=pymupdf.PDF_ENCRYPT_AES_256, owner_pw="owner", user_pw="user"
        )
        document.close()
        with pytest.raises(EncryptedPDFError):
            extract_text_from_pdf(data)

    def test_zero_page_pdf_raises_empty_pdf_error(self, monkeypatch):
        class _ZeroPageDoc:
            needs_pass = False
            page_count = 0

            def close(self) -> None:
                return None

        monkeypatch.setattr(pymupdf, "open", lambda *args, **kwargs: _ZeroPageDoc())
        with pytest.raises(EmptyPDFError):
            extract_text_from_pdf(b"%PDF-1.4\n")


def test_clean_text_normalises_whitespace_and_hyphenation():
    raw = "Natural lan-\nguage   processing\n\nis   fun.\x00"
    assert clean_text(raw) == "Natural language processing is fun."
    assert clean_text("") == ""

