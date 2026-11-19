"""
Optional integration test with the REAL pretrained DistilBERT model.

Skipped by default. Run it with:

    RUN_INTEGRATION_TESTS=1 pytest -m integration -v

The first run downloads the model (~260 MB) from the Hugging Face Hub.
"""

import os

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from tests.conftest import make_pdf

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("RUN_INTEGRATION_TESTS") != "1",
        reason="set RUN_INTEGRATION_TESTS=1 to run tests with the real DistilBERT model",
    ),
]

PAGES = [
    "Natural Language Processing. Natural Language Processing (NLP) is a field of artificial "
    "intelligence that helps computers understand human language.",
    "The Transformer architecture was introduced in 2017 in the paper Attention Is All You Need. "
    "It relies entirely on a mechanism called self-attention instead of recurrence.",
    "DistilBERT was created by Hugging Face. It is 40 percent smaller than BERT and runs 60 percent "
    "faster while keeping 97 percent of its language understanding performance.",
    "The Eiffel Tower is located in Paris. It was completed in 1889 and is about 330 metres tall.",
]


@pytest.fixture(scope="module")
def real_client(tmp_path_factory):
    settings = Settings(
        _env_file=None,
        upload_dir=tmp_path_factory.mktemp("uploads"),
        load_model_on_startup=True,
    )
    app = create_app(settings=settings)  # real ModelService -> real DistilBERT
    with TestClient(app) as client:
        health = client.get("/api/health").json()
        assert health["model_loaded"], health
        yield client


@pytest.fixture(scope="module")
def document_id(real_client):
    response = real_client.post(
        "/api/documents/upload", files={"file": ("nlp.pdf", make_pdf(PAGES), "application/pdf")}
    )
    assert response.status_code == 201, response.text
    return response.json()["document_id"]


def _ask(client, document_id, question):
    response = client.post("/api/questions/ask", json={"document_id": document_id, "question": question})
    assert response.status_code == 200, response.text
    return response.json()


def test_health_uses_required_model(real_client):
    body = real_client.get("/api/health").json()
    assert body["model_name"] == "distilbert/distilbert-base-cased-distilled-squad"
    assert body["device"] in ("cpu", "cuda")


@pytest.mark.parametrize(
    "question,expected,page",
    [
        ("When was the Transformer architecture introduced?", "2017", 2),
        ("Who created DistilBERT?", "Hugging Face", 3),
        ("When was the Eiffel Tower completed?", "1889", 4),
        ("What mechanism does the Transformer rely on?", "self-attention", 2),
    ],
)
def test_real_model_extracts_correct_answers(real_client, document_id, question, expected, page):
    body = _ask(real_client, document_id, question)
    assert body["success"] is True, body
    assert expected.lower() in body["answer"].lower()
    assert body["source"]["page_number"] == page
    assert 0.0 < body["confidence"] <= 1.0
    # The answer is an exact span of the returned passage (extractive QA).
    passage = body["source"]["passage"]
    assert passage[body["source"]["answer_start"]:body["source"]["answer_end"]] == body["answer"]


def test_real_model_does_not_invent_answers(real_client, document_id):
    body = _ask(real_client, document_id, "Who won the football world cup in 2022?")
    assert body["success"] is False
    assert body["answer"] is None


def test_real_tokenizer_chunks_fit_model_limit(real_client):
    from app.utils.text_chunking import PageText, chunk_page

    settings = real_client.app.state.settings
    tokenizer = real_client.app.state.model_service.tokenizer
    long_text = " ".join(PAGES * 30)  # several thousand tokens on one page
    chunks = chunk_page(
        PageText(7, long_text), tokenizer, settings.chunk_size_tokens, settings.chunk_overlap_tokens
    )
    assert len(chunks) > 3
    for chunk in chunks:
        n_tokens = len(tokenizer(chunk.text, add_special_tokens=False)["input_ids"])
        assert n_tokens <= settings.chunk_size_tokens
        # chunk + longest allowed question + [CLS]/[SEP]/[SEP] fits into 512 tokens
        assert n_tokens + settings.max_question_tokens + 3 <= 512
        assert chunk.page_number == 7
