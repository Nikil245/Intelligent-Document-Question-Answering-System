"""Tests for the QA service (with mocked DistilBERT inference)."""

import pytest

from app.exceptions import InvalidQuestionError, ModelNotReadyError
from app.services.qa_service import (
    AnswerStatus,
    QAService,
    build_supporting_passage,
    refine_answer_span,
)
from app.services.retrieval_service import TfidfRetriever
from app.utils.text_chunking import TextChunk
from tests.conftest import FakeModelService


def _chunk(chunk_id, page, text):
    return TextChunk(chunk_id, page, 0, text, 0, len(text), len(text.split()))


@pytest.fixture
def retriever():
    return TfidfRetriever(
        [
            _chunk(0, 1, "The Sun is the star at the centre of the solar system."),
            _chunk(1, 2, "France is in Europe. The capital of France is Paris."),
            _chunk(2, 3, "Paris is also a common name. France exports wine and cheese."),
        ]
    )


def _service(model, threshold=0.1, top_k=3, max_question_tokens=64):
    return QAService(
        model_service=model,
        confidence_threshold=threshold,
        top_k=top_k,
        min_similarity=0.0,
        max_answer_len=30,
        max_question_tokens=max_question_tokens,
        passage_window_chars=300,
    )


def test_returns_answer_with_page_and_passage(retriever):
    model = FakeModelService(rules={"Paris": 0.9})
    result = _service(model).answer("What is the capital of France?", retriever)
    assert result.success is True
    assert result.status == AnswerStatus.ANSWERED
    assert result.answer == "Paris"
    assert result.confidence == 0.9
    assert result.source.page_number == 2
    passage = result.source.passage
    assert passage[result.source.answer_start:result.source.answer_end] == "Paris"


def test_each_retrieved_chunk_is_processed_individually(retriever):
    model = FakeModelService(rules={"Paris": 0.9})
    _service(model, top_k=3).answer("capital of France", retriever)
    contexts = [context for _, context in model.calls]
    assert len(contexts) == 2  # two chunks mention France
    assert len(set(contexts)) == 2


def test_best_candidate_selected_by_score(retriever):
    model = FakeModelService(rules={"wine and cheese": 0.95, "Paris": 0.4})
    result = _service(model).answer("What does France export?", retriever)
    assert result.answer == "wine and cheese"
    assert result.source.page_number == 3


def test_low_confidence_returns_no_answer(retriever):
    model = FakeModelService(rules={"Paris": 0.05})
    result = _service(model, threshold=0.2).answer("What is the capital of France?", retriever)
    assert result.success is False
    assert result.status == AnswerStatus.LOW_CONFIDENCE
    assert result.answer is None  # weak guesses are never presented as answers
    assert result.confidence == 0.05
    assert result.source is not None  # closest passage still provided for reference


def test_no_relevant_chunks(retriever):
    model = FakeModelService(rules={"Paris": 0.9})
    result = _service(model).answer("Who invented the telephone?", retriever)
    assert result.success is False
    assert result.status == AnswerStatus.NO_RELEVANT_CONTENT
    assert result.answer is None and result.source is None
    assert model.calls == []  # the model is not run without relevant context


def test_empty_model_answers_are_skipped(retriever):
    class EmptyAnswerModel(FakeModelService):
        def predict(self, question, context, max_answer_len=30):
            return {"answer": "   ", "score": 0.99, "start": 0, "end": 0}

    result = _service(EmptyAnswerModel()).answer("capital of France", retriever)
    assert result.success is False
    assert result.status == AnswerStatus.NO_ANSWER
    assert result.answer is None


def test_empty_question_rejected(retriever):
    with pytest.raises(InvalidQuestionError):
        _service(FakeModelService()).answer("   ", retriever)


def test_too_long_question_rejected(retriever):
    question = " ".join(["France"] * 100)
    with pytest.raises(InvalidQuestionError, match="too long"):
        _service(FakeModelService(), max_question_tokens=64).answer(question, retriever)


def test_model_not_loaded_raises(retriever):
    with pytest.raises(ModelNotReadyError):
        _service(FakeModelService(loaded=False)).answer("capital of France", retriever)


def test_build_supporting_passage_offsets():
    context = ("filler " * 100) + "ANSWER" + (" filler" * 100)
    start = context.index("ANSWER")
    passage, a_start, a_end = build_supporting_passage(context, start, start + 6, window=40)
    assert passage[a_start:a_end] == "ANSWER"
    assert len(passage) < len(context)
    assert not passage.startswith(" ") and not passage.endswith(" ")


def test_descriptive_question_avoids_heading_and_populates_debug_and_passages():
    retriever = TfidfRetriever(
        [
            _chunk(0, 1, "Feedforward Neural Networks"),
            _chunk(1, 2, "Feedforward Neural Networks can also be called multi-layer perceptrons (or MLPs) for historical reasons."),
            _chunk(2, 3, "A two-layer feedforward network uses an input layer, hidden layer, and output layer."),
            _chunk(3, 4, "Word2Vec learns dense embeddings with skip-gram negative sampling."),
        ]
    )
    model = FakeModelService(
        rules={
            "Feedforward Neural Networks": 0.75,
            "multi-layer perceptrons (or MLPs) for historical reasons": 0.65,
            "input layer, hidden layer, and output layer": 0.55,
        }
    )
    result = _service(model, top_k=5).answer(
        "Explain feedforward neural networks and their layers. (06 Marks)", retriever
    )
    assert result.success is True
    assert result.question_type == "descriptive"
    assert result.source.page_number in (2, 3)
    assert result.source.passage[result.source.answer_start : result.source.answer_end] == result.answer
    assert len(result.supporting_passages) >= 2
    assert len(result.retrieved_chunks) >= 2
    assert all(rc.raw_tfidf_score >= 0.0 for rc in result.retrieved_chunks)


def test_refine_answer_span_shifts_past_title_echo_and_preserves_exact_offsets():
    context = "Computation Graphs A computation graph represents the process of computing a mathematical expression 81"
    st, en, refined = refine_answer_span(
        context, 0, len("Computation Graphs"), {"computation", "graph"}, is_descriptive=False
    )
    assert refined == "A computation graph represents the process of computing a mathematical expression"
    assert context[st:en] == refined

