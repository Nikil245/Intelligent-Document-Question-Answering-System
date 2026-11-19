"""
Shared test fixtures.

Ordinary tests never download DistilBERT. Instead they use:
  * SimpleTokenizer   - a tiny regex tokenizer that provides the same
                        `offset_mapping` interface as a Hugging Face fast tokenizer;
  * FakeModelService  - a ModelService whose `predict` is mocked: it returns a
                        pre-configured answer span when that text occurs in the
                        context, and a low-confidence guess otherwise.

The real model is only used in tests/test_integration_real_model.py.
"""

import re
from typing import Dict, List, Optional, Sequence, Tuple

import pymupdf
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.services.model_service import ModelService, ModelStatus


# ---------------------------------------------------------------------------
# Fake tokenizer and model
# ---------------------------------------------------------------------------
class SimpleTokenizer:
    """Splits text into words and punctuation, returning character offsets."""

    _pattern = re.compile(r"\w+|[^\w\s]")

    def __call__(self, text, add_special_tokens=True, return_offsets_mapping=False, **kwargs):
        spans = [(m.start(), m.end()) for m in self._pattern.finditer(text)]
        encoding = {"input_ids": list(range(len(spans)))}
        if return_offsets_mapping:
            encoding["offset_mapping"] = spans
        return encoding


class FakeModelService(ModelService):
    """
    Mocked DistilBERT. `rules` maps an answer string to the score that should
    be returned when that string appears in the context.
    """

    def __init__(self, rules: Optional[Dict[str, float]] = None, loaded: bool = True):
        super().__init__(model_name="distilbert/distilbert-base-cased-distilled-squad")
        self.rules = rules or {}
        self.tokenizer = SimpleTokenizer()
        self.device = "cpu"
        self.status = ModelStatus.LOADED if loaded else ModelStatus.NOT_LOADED
        self.calls: List[Tuple[str, str]] = []

    def load(self) -> None:  # never download anything in unit tests
        self.status = ModelStatus.LOADED

    def predict(self, question: str, context: str, max_answer_len: int = 30):
        self.ensure_ready()
        self.calls.append((question, context))
        for answer, score in self.rules.items():
            start = context.find(answer)
            if start != -1:
                return {"answer": answer, "score": score, "start": start, "end": start + len(answer)}
        # No configured answer in this context: return a weak guess (first word).
        first_word = context.split()[0] if context.split() else ""
        return {"answer": first_word, "score": 0.01, "start": 0, "end": len(first_word)}


# ---------------------------------------------------------------------------
# PDF helpers
# ---------------------------------------------------------------------------
def make_pdf(pages: Sequence[Optional[str]]) -> bytes:
    """Create an in-memory PDF. `None` or "" creates a blank page."""
    document = pymupdf.open()
    for text in pages:
        page = document.new_page()  # A4-ish default size
        if text:
            rect = pymupdf.Rect(50, 50, page.rect.width - 50, page.rect.height - 50)
            page.insert_textbox(rect, text, fontsize=10)
    data = document.tobytes()
    document.close()
    return data


SAMPLE_PAGES = [
    "Introduction to the solar system. The Sun is the star at the centre of our solar system.",
    "France is a country in Western Europe. The capital of France is Paris, a city famous for the Eiffel Tower.",
    "Photosynthesis is the process by which green plants use sunlight to make food from carbon dioxide and water.",
]


@pytest.fixture
def sample_pdf_bytes() -> bytes:
    return make_pdf(SAMPLE_PAGES)


@pytest.fixture
def tokenizer() -> SimpleTokenizer:
    return SimpleTokenizer()


# ---------------------------------------------------------------------------
# Application fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def test_settings(tmp_path) -> Settings:
    return Settings(
        _env_file=None,  # ignore any local .env file
        load_model_on_startup=False,
        upload_dir=tmp_path / "uploads",
        max_upload_size_mb=1,
        chunk_size_tokens=64,
        chunk_overlap_tokens=16,
        confidence_threshold=0.1,
    )


@pytest.fixture
def fake_model() -> FakeModelService:
    return FakeModelService(rules={"Paris": 0.93, "The Sun": 0.81})


@pytest.fixture
def client(test_settings, fake_model):
    app = create_app(settings=test_settings, model_service=fake_model)
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def uploaded_document_id(client, sample_pdf_bytes) -> str:
    response = client.post(
        "/api/documents/upload",
        files={"file": ("sample.pdf", sample_pdf_bytes, "application/pdf")},
    )
    assert response.status_code == 201, response.text
    return response.json()["document_id"]
