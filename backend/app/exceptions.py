"""
Custom exceptions used across the backend.

Each exception carries an HTTP status code, a short machine-readable error code
and a human-readable message. A single exception handler in `main.py` turns
them into JSON responses, so services never need to know about HTTP details
and clients never see stack traces or server paths.
"""


class AppError(Exception):
    status_code: int = 500
    error_code: str = "internal_error"
    default_message: str = "An unexpected error occurred."

    def __init__(self, message: str | None = None):
        self.message = message or self.default_message
        super().__init__(self.message)


# ---- Upload / PDF errors ---------------------------------------------------
class InvalidFileTypeError(AppError):
    status_code = 415
    error_code = "invalid_file_type"
    default_message = "Only PDF files are accepted."


class FileTooLargeError(AppError):
    status_code = 413
    error_code = "file_too_large"
    default_message = "The uploaded file exceeds the maximum allowed size."


class EmptyFileError(AppError):
    status_code = 400
    error_code = "empty_file"
    default_message = "The uploaded file is empty."


class CorruptedPDFError(AppError):
    status_code = 422
    error_code = "corrupted_pdf"
    default_message = "The PDF file could not be read. It may be corrupted or incomplete."


class EncryptedPDFError(AppError):
    status_code = 422
    error_code = "encrypted_pdf"
    default_message = "The PDF is password-protected. Please upload an unprotected PDF."


class EmptyPDFError(AppError):
    status_code = 422
    error_code = "empty_pdf"
    default_message = "The PDF does not contain any pages."


class NoExtractableTextError(AppError):
    status_code = 422
    error_code = "no_extractable_text"
    default_message = (
        "No extractable text was found in this PDF. It may be a scanned document "
        "or contain only images. OCR (optical character recognition) is not "
        "supported in this version, so please upload a PDF with selectable text."
    )


# ---- Document / question errors --------------------------------------------
class DocumentNotFoundError(AppError):
    status_code = 404
    error_code = "document_not_found"
    default_message = "No document was found with the given ID. Upload the PDF again."


class InvalidQuestionError(AppError):
    status_code = 400
    error_code = "invalid_question"
    default_message = "The question is invalid."


# ---- Model errors ----------------------------------------------------------
class ModelNotReadyError(AppError):
    status_code = 503
    error_code = "model_not_ready"
    default_message = "The question-answering model is not loaded yet."


class ModelInferenceError(AppError):
    status_code = 500
    error_code = "model_inference_error"
    default_message = "The question-answering model failed while processing the request."

