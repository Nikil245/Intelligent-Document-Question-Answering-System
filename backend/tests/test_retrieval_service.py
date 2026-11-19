"""Tests for TF-IDF + cosine similarity retrieval."""

import pytest

from app.services.retrieval_service import TfidfRetriever
from app.utils.text_chunking import TextChunk


def _chunk(chunk_id: int, page: int, text: str) -> TextChunk:
    return TextChunk(
        chunk_id=chunk_id, page_number=page, chunk_index_in_page=0,
        text=text, start_char=0, end_char=len(text), num_tokens=len(text.split()),
    )


@pytest.fixture
def retriever() -> TfidfRetriever:
    chunks = [
        _chunk(0, 1, "The Sun is the star at the centre of the solar system."),
        _chunk(1, 2, "The capital of France is Paris. Paris hosts the Eiffel Tower."),
        _chunk(2, 5, "Photosynthesis converts sunlight, water and carbon dioxide into glucose."),
        _chunk(3, 7, "Neural networks learn representations from data using backpropagation."),
        _chunk(4, 9, "The Transformer architecture relies on self-attention instead of recurrence."),
    ]
    return TfidfRetriever(chunks)


def test_most_relevant_chunk_ranked_first(retriever):
    results = retriever.retrieve("What is the capital of France?", top_k=3)
    assert results[0].chunk.chunk_id == 1
    assert results[0].rank == 1


def test_original_page_numbers_preserved(retriever):
    results = retriever.retrieve("How does photosynthesis use sunlight?", top_k=3)
    assert results[0].chunk.page_number == 5


def test_top_k_respected_and_scores_descending(retriever):
    results = retriever.retrieve("the sun star solar system France photosynthesis networks", top_k=3)
    assert len(results) <= 3
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
    assert all(0.0 < s <= 1.0 for s in scores)


def test_only_matching_chunks_returned(retriever):
    # Only one chunk mentions "self-attention", so only one result is returned
    # even though top_k is larger.
    results = retriever.retrieve("What is self-attention?", top_k=5)
    assert [r.chunk.chunk_id for r in results] == [4]


def test_unrelated_question_returns_nothing(retriever):
    assert retriever.retrieve("Who won the football world cup?", top_k=3) == []


def test_stop_word_only_question_returns_nothing(retriever):
    assert retriever.retrieve("What is the?", top_k=3) == []


def test_min_similarity_filters_weak_matches(retriever):
    weak = retriever.retrieve("France", top_k=5, min_similarity=0.0)
    strict = retriever.retrieve("France", top_k=5, min_similarity=0.99)
    assert weak and strict == []


def test_retriever_requires_chunks():
    with pytest.raises(ValueError):
        TfidfRetriever([])


def test_document_of_only_stop_words_still_indexes():
    retriever = TfidfRetriever([_chunk(0, 1, "the and of is"), _chunk(1, 2, "what which who")])
    results = retriever.retrieve("who", top_k=3)
    assert results and results[0].chunk.page_number == 2


def test_exam_rubric_cleaned_and_question_bank_and_headings_downweighted():
    chunks = [
        _chunk(0, 1, "Simple Neural Networks Feedforward Neural Networks"),
        _chunk(1, 2, "Feedforward networks use an input layer, hidden layer, and output layer to compute predictions."),
        _chunk(2, 3, "QUESTION BANK 1. Explain feedforward neural networks. (06 Marks | L2 | CO3) 2. Explain input and hidden layers. (08 Marks | L2 | CO3)"),
        _chunk(3, 4, "Cosine similarity compares word vectors in vector semantics."),
    ]
    retriever = TfidfRetriever(chunks)
    results = retriever.retrieve(
        "Explain the architecture of a feedforward neural network, identifying input, hidden and output layers. (06 Marks)",
        top_k=5,
    )
    assert results[0].chunk.page_number == 2
    assert "layer" in results[0].matched_concepts
    assert results[0].concept_coverage > 0.0
    heading_items = [r for r in results if r.chunk.page_number == 1]
    qb_items = [r for r in results if r.chunk.page_number == 3]
    assert heading_items and heading_items[0].is_heading_only is True
    assert qb_items and qb_items[0].is_question_bank is True


def test_plural_and_singular_concepts_match():
    chunks = [
        _chunk(0, 1, "A two-layer network has an input layer and a hidden layer."),
        _chunk(1, 2, "Word embedding vectors capture semantic similarity."),
    ]
    retriever = TfidfRetriever(chunks)
    layer_res = retriever.retrieve("What are the hidden layers?", top_k=2)
    assert layer_res and layer_res[0].chunk.page_number == 1
    emb_res = retriever.retrieve("How do embeddings work?", top_k=2)
    assert emb_res and emb_res[0].chunk.page_number == 2

