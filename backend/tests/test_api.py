"""API tests using FastAPI's TestClient and the mocked model service."""

from tests.conftest import make_pdf


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
def test_health_endpoint(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model_status"] == "loaded"
    assert body["model_name"] == "distilbert/distilbert-base-cased-distilled-squad"
    assert body["device"] in ("cpu", "cuda")


def test_health_reports_degraded_when_model_not_loaded(test_settings):
    from fastapi.testclient import TestClient

    from app.main import create_app
    from tests.conftest import FakeModelService

    app = create_app(settings=test_settings, model_service=FakeModelService(loaded=False))
    with TestClient(app) as client:
        body = client.get("/api/health").json()
    assert body["status"] == "degraded" and body["model_loaded"] is False


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------
def test_upload_valid_pdf(client, sample_pdf_bytes, test_settings):
    response = client.post(
        "/api/documents/upload",
        files={"file": ("lecture notes.pdf", sample_pdf_bytes, "application/pdf")},
    )
    assert response.status_code == 201
    body = response.json()
    assert body["success"] is True
    assert body["filename"] == "lecture notes.pdf"
    assert body["num_pages"] == 3
    assert body["num_chunks"] >= 3
    assert body["status"] == "ready"
    # The file is stored under the generated UUID, not the client filename.
    assert (test_settings.upload_dir / f"{body['document_id']}.pdf").exists()


def test_upload_reports_empty_pages(client):
    pdf = make_pdf(["Page one has text.", None])
    body = client.post("/api/documents/upload", files={"file": ("a.pdf", pdf, "application/pdf")}).json()
    assert body["empty_pages"] == [2] and body["pages_with_text"] == 1


def test_upload_rejects_non_pdf(client):
    response = client.post("/api/documents/upload", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert response.status_code == 415
    assert response.json()["error"] == "invalid_file_type"


def test_upload_rejects_renamed_non_pdf(client):
    response = client.post(
        "/api/documents/upload", files={"file": ("fake.pdf", b"not a pdf at all", "application/pdf")}
    )
    assert response.status_code == 415


def test_upload_rejects_corrupted_pdf(client):
    response = client.post(
        "/api/documents/upload",
        files={"file": ("broken.pdf", b"%PDF-1.7\ngarbage garbage", "application/pdf")},
    )
    assert response.status_code == 422
    assert response.json()["error"] == "corrupted_pdf"


def test_upload_rejects_pdf_without_text(client):
    response = client.post(
        "/api/documents/upload", files={"file": ("scan.pdf", make_pdf([None]), "application/pdf")}
    )
    assert response.status_code == 422
    body = response.json()
    assert body["error"] == "no_extractable_text"
    assert "OCR" in body["message"]


def test_upload_rejects_empty_file(client):
    response = client.post("/api/documents/upload", files={"file": ("empty.pdf", b"", "application/pdf")})
    assert response.status_code == 400
    assert response.json()["error"] == "empty_file"


def test_upload_rejects_large_file(client):
    big = b"%PDF-1.4\n" + b"0" * (1024 * 1024 + 10)  # test limit is 1 MB
    response = client.post("/api/documents/upload", files={"file": ("big.pdf", big, "application/pdf")})
    assert response.status_code == 413
    assert response.json()["error"] == "file_too_large"


def test_upload_without_file_is_validation_error(client):
    response = client.post("/api/documents/upload")
    assert response.status_code == 422
    assert response.json()["error"] == "validation_error"


def test_upload_returns_503_when_model_not_loaded(test_settings, sample_pdf_bytes):
    from fastapi.testclient import TestClient

    from app.main import create_app
    from tests.conftest import FakeModelService

    app = create_app(settings=test_settings, model_service=FakeModelService(loaded=False))
    with TestClient(app) as client:
        response = client.post(
            "/api/documents/upload", files={"file": ("a.pdf", sample_pdf_bytes, "application/pdf")}
        )
    assert response.status_code == 503
    assert response.json()["error"] == "model_not_ready"


# ---------------------------------------------------------------------------
# Document details / delete
# ---------------------------------------------------------------------------
def test_get_document_details(client, uploaded_document_id):
    response = client.get(f"/api/documents/{uploaded_document_id}")
    assert response.status_code == 200
    body = response.json()
    assert body["document_id"] == uploaded_document_id
    assert body["num_pages"] == 3
    assert body["status"] == "ready"
    assert body["total_characters"] > 0


def test_get_unknown_document_returns_404(client):
    response = client.get("/api/documents/00000000-0000-0000-0000-000000000000")
    assert response.status_code == 404
    assert response.json()["error"] == "document_not_found"


def test_delete_document_removes_it_and_its_file(client, uploaded_document_id, test_settings):
    file_path = test_settings.upload_dir / f"{uploaded_document_id}.pdf"
    assert file_path.exists()
    response = client.delete(f"/api/documents/{uploaded_document_id}")
    assert response.status_code == 200 and response.json()["success"] is True
    assert not file_path.exists()
    assert client.get(f"/api/documents/{uploaded_document_id}").status_code == 404
    assert client.delete(f"/api/documents/{uploaded_document_id}").status_code == 404


# ---------------------------------------------------------------------------
# Questions
# ---------------------------------------------------------------------------
def test_ask_question_returns_answer_with_page(client, uploaded_document_id):
    response = client.post(
        "/api/questions/ask",
        json={"document_id": uploaded_document_id, "question": "What is the capital of France?"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["status"] == "answered"
    assert body["answer"] == "Paris"
    assert body["confidence"] == 0.93
    assert body["source"]["page_number"] == 2
    assert "Paris" in body["source"]["passage"]
    assert body["message"] == "Answer retrieved successfully"


def test_multiple_questions_on_same_document(client, uploaded_document_id):
    first = client.post(
        "/api/questions/ask",
        json={"document_id": uploaded_document_id, "question": "What is the capital of France?"},
    ).json()
    second = client.post(
        "/api/questions/ask",
        json={"document_id": uploaded_document_id, "question": "What star is at the centre of the solar system?"},
    ).json()
    assert first["answer"] == "Paris" and first["source"]["page_number"] == 2
    assert second["answer"] == "The Sun" and second["source"]["page_number"] == 1


def test_ask_question_with_no_relevant_content(client, uploaded_document_id):
    body = client.post(
        "/api/questions/ask",
        json={"document_id": uploaded_document_id, "question": "Who won the cricket world cup?"},
    ).json()
    assert body["success"] is False
    assert body["status"] == "no_relevant_content"
    assert body["answer"] is None


def test_ask_question_low_confidence(client, uploaded_document_id):
    # The fake model only gives a 0.01-score guess for photosynthesis questions.
    body = client.post(
        "/api/questions/ask",
        json={"document_id": uploaded_document_id, "question": "What do plants make with photosynthesis?"},
    ).json()
    assert body["success"] is False
    assert body["status"] == "low_confidence"
    assert body["answer"] is None
    assert body["source"]["page_number"] == 3


def test_ask_empty_question_rejected(client, uploaded_document_id):
    response = client.post("/api/questions/ask", json={"document_id": uploaded_document_id, "question": "   "})
    assert response.status_code == 422
    body = response.json()
    assert body["error"] == "validation_error"
    assert "Question must not be empty" in body["message"]


def test_ask_missing_question_rejected(client, uploaded_document_id):
    response = client.post("/api/questions/ask", json={"document_id": uploaded_document_id})
    assert response.status_code == 422
    assert "question" in response.json()["message"]


def test_ask_too_long_question_rejected(client, uploaded_document_id):
    response = client.post(
        "/api/questions/ask", json={"document_id": uploaded_document_id, "question": "word " * 200}
    )
    assert response.status_code in (400, 422)
    assert response.json()["success"] is False


def test_ask_unknown_document_returns_404(client):
    response = client.post(
        "/api/questions/ask",
        json={"document_id": "11111111-2222-3333-4444-555555555555", "question": "What is this?"},
    )
    assert response.status_code == 404
    assert response.json()["error"] == "document_not_found"


def test_ask_invalid_json_rejected(client):
    response = client.post(
        "/api/questions/ask", content="not json", headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 422
    assert response.json()["success"] is False


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------
def test_cors_allows_frontend_origin(client):
    response = client.options(
        "/api/questions/ask",
        headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "POST"},
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_cors_rejects_unknown_origin(client):
    response = client.options(
        "/api/questions/ask",
        headers={"Origin": "http://evil.example.com", "Access-Control-Request-Method": "POST"},
    )
    assert "access-control-allow-origin" not in response.headers


def test_unknown_route_returns_json_404(client):
    response = client.get("/api/does-not-exist")
    assert response.status_code == 404
    assert response.json()["success"] is False


def test_openapi_docs_available(client):
    assert client.get("/docs").status_code == 200
    assert "/api/questions/ask" in client.get("/openapi.json").json()["paths"]


# ---------------------------------------------------------------------------
# Error handling & configuration validation
# ---------------------------------------------------------------------------
def test_model_load_failure_reported_without_leaking_paths(test_settings, sample_pdf_bytes, monkeypatch):
    import transformers
    from fastapi.testclient import TestClient

    from app.main import create_app
    from app.services.model_service import ModelService, ModelStatus

    def _fail_load(*args, **kwargs):
        raise OSError("cannot access /secret/server/path/model.bin")

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", _fail_load)

    model_service = ModelService(model_name=test_settings.model_name)
    model_service.load()

    assert model_service.status == ModelStatus.FAILED
    assert model_service.error is not None
    assert "/secret/server/path" not in model_service.error

    app = create_app(settings=test_settings, model_service=model_service)
    with TestClient(app) as client:
        health_body = client.get("/api/health").json()
        assert health_body["status"] == "degraded"
        assert health_body["model_loaded"] is False
        assert health_body["model_status"] == "failed"
        assert health_body["model_error"]
        assert "/secret/server/path" not in health_body["model_error"]

        upload_resp = client.post(
            "/api/documents/upload",
            files={"file": ("a.pdf", sample_pdf_bytes, "application/pdf")},
        )
        assert upload_resp.status_code == 503
        assert upload_resp.json()["error"] == "model_not_ready"


def test_model_inference_failure_returns_sanitized_500(test_settings, sample_pdf_bytes):
    from fastapi.testclient import TestClient

    from app.exceptions import ModelInferenceError
    from app.main import create_app
    from tests.conftest import FakeModelService

    class FailingInferenceModel(FakeModelService):
        def predict(self, question: str, context: str, max_answer_len: int = 30):
            raise ModelInferenceError()

    app = create_app(settings=test_settings, model_service=FailingInferenceModel())
    with TestClient(app) as client:
        doc_id = client.post(
            "/api/documents/upload",
            files={"file": ("a.pdf", sample_pdf_bytes, "application/pdf")},
        ).json()["document_id"]

        response = client.post(
            "/api/questions/ask",
            json={"document_id": doc_id, "question": "What is the capital of France?"},
        )
        assert response.status_code == 500
        body = response.json()
        assert body["success"] is False
        assert body["error"] == "model_inference_error"
        assert "Traceback" not in response.text


def test_unexpected_error_returns_sanitized_500(test_settings, fake_model, monkeypatch):
    from fastapi.testclient import TestClient

    from app.main import create_app

    app = create_app(settings=test_settings, model_service=fake_model)

    def _boom(_document_id: str):
        raise RuntimeError("boom /internal/secret/path")

    monkeypatch.setattr(app.state.document_store, "get", _boom)

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/api/documents/any-id")
        assert response.status_code == 500
        body = response.json()
        assert body["success"] is False
        assert body["error"] == "internal_error"
        assert "/internal/secret/path" not in response.text
        assert "Traceback" not in response.text


def test_settings_cors_parsing_and_token_budget_validation():
    import pytest
    from pydantic import ValidationError

    from app.config import Settings

    parsed = Settings(_env_file=None, cors_origins="http://a:3000, http://b:5173")
    assert parsed.cors_origins == ["http://a:3000", "http://b:5173"]

    with pytest.raises(ValidationError, match="exceeds the model limit"):
        Settings(_env_file=None, chunk_size_tokens=480, max_question_tokens=64)


def test_ask_response_includes_debug_and_supporting_passages(client, uploaded_document_id):
    response = client.post(
        "/api/questions/ask",
        json={
            "document_id": uploaded_document_id,
            "question": "Explain what the capital of France is.",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["question_type"] == "descriptive"
    assert len(body["supporting_passages"]) >= 1
    assert len(body["retrieved_chunks"]) >= 1
    first_debug = body["retrieved_chunks"][0]
    assert "raw_tfidf_score" in first_debug
    assert "concept_coverage" in first_debug
    assert "matched_concepts" in first_debug
