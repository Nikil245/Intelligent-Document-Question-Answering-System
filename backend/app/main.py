"""
FastAPI application entry point.

`create_app()` wires everything together:
  * settings, the DistilBERT model service, the document store and QA service
    are created once and stored on `app.state`;
  * the model is loaded once at startup (not per request);
  * CORS is configured for the separately developed frontend;
  * every error is converted to a clean JSON response without stack traces.
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.routes import router
from app.config import Settings, get_settings
from app.exceptions import AppError
from app.services.model_service import ModelService
from app.services.qa_service import QAService
from app.storage.document_store import DocumentStore

logger = logging.getLogger("app")


def _error_body(error: str, message: str, details: Optional[list] = None) -> dict:
    body = {"success": False, "error": error, "message": message}
    if details:
        body["details"] = details
    return body


def _register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def handle_app_error(request: Request, exc: AppError):
        return JSONResponse(status_code=exc.status_code, content=_error_body(exc.error_code, exc.message))

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(request: Request, exc: RequestValidationError):
        # Turn Pydantic's error list into short readable messages such as
        # "question: Question must not be empty".
        details = []
        for err in exc.errors():
            location = ".".join(str(part) for part in err.get("loc", ()) if part not in ("body", "query", "path"))
            message = str(err.get("msg", "Invalid value")).removeprefix("Value error, ")
            details.append(f"{location}: {message}" if location else message)
        return JSONResponse(
            status_code=422,
            content=jsonable_encoder(
                _error_body("validation_error", "Invalid request: " + "; ".join(details), details)
            ),
        )

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_error(request: Request, exc: StarletteHTTPException):
        codes = {404: "not_found", 405: "method_not_allowed", 413: "file_too_large"}
        return JSONResponse(
            status_code=exc.status_code,
            content=_error_body(codes.get(exc.status_code, "http_error"), str(exc.detail)),
        )

    @app.exception_handler(Exception)
    async def handle_unexpected_error(request: Request, exc: Exception):
        # Log full details on the server, return a generic message to the client.
        logger.exception("Unhandled error while processing %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=500,
            content=_error_body("internal_error", "An unexpected server error occurred. Please try again."),
        )


def create_app(
    settings: Optional[Settings] = None,
    model_service: Optional[ModelService] = None,
) -> FastAPI:
    """Build the FastAPI application. Tests can pass custom settings or a fake model service."""
    settings = settings or get_settings()
    logging.basicConfig(
        level=settings.log_level.upper(),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )

    model_service = model_service or ModelService(
        model_name=settings.model_name,
        device_preference=settings.device,
        max_seq_len=settings.model_max_tokens,
    )
    document_store = DocumentStore()
    qa_service = QAService(
        model_service=model_service,
        confidence_threshold=settings.confidence_threshold,
        top_k=settings.top_k_chunks,
        min_similarity=settings.min_retrieval_similarity,
        max_answer_len=settings.max_answer_length_tokens,
        max_question_tokens=settings.max_question_tokens,
        passage_window_chars=settings.passage_window_chars,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)
        if settings.load_model_on_startup and not model_service.is_loaded:
            # Loaded exactly once for the whole lifetime of the server.
            model_service.load()
        yield
        document_store.clear(remove_files=settings.cleanup_uploads_on_shutdown)

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "Upload a PDF and ask questions about it. Relevant passages are found with "
            "TF-IDF + cosine similarity, and answers are EXTRACTED from the PDF text by the "
            f"pretrained `{settings.model_name}` model."
        ),
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.model_service = model_service
    app.state.document_store = document_store
    app.state.qa_service = qa_service

    @app.middleware("http")
    async def reject_oversized_uploads(request: Request, call_next):
        # Reject obviously oversized uploads early using the Content-Length
        # header (+1 MB allowance for multipart overhead). The exact file size
        # is checked again during PDF validation.
        content_length = request.headers.get("content-length")
        limit = settings.max_upload_size_bytes + 1024 * 1024
        if content_length and content_length.isdigit() and int(content_length) > limit:
            return JSONResponse(
                status_code=413,
                content=_error_body(
                    "file_too_large",
                    f"The upload exceeds the maximum allowed size of {settings.max_upload_size_mb} MB.",
                ),
            )
        return await call_next(request)

    # Added last so it is the OUTERMOST middleware: every response, including
    # early error responses, then carries the CORS headers the browser needs.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )

    _register_exception_handlers(app)
    app.include_router(router)

    @app.get("/", tags=["System"])
    def root():
        return {
            "name": settings.app_name,
            "docs": "/docs",
            "health": "/api/health",
        }

    return app


app = create_app()
