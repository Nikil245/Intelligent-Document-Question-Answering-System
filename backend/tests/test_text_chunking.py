"""Tests for token-aware chunking."""

import pytest

from app.utils.text_chunking import PageText, chunk_page, chunk_pages


def _long_text(n_words: int) -> str:
    return " ".join(f"word{i}" for i in range(n_words))


def test_short_page_becomes_single_chunk(tokenizer):
    page = PageText(page_number=4, text="A short page about transformers.")
    chunks = chunk_page(page, tokenizer, chunk_size=50, overlap=10)
    assert len(chunks) == 1
    assert chunks[0].text == page.text
    assert chunks[0].page_number == 4


def test_long_page_split_within_token_limit(tokenizer):
    page = PageText(page_number=1, text=_long_text(250))
    chunks = chunk_page(page, tokenizer, chunk_size=64, overlap=16)
    assert len(chunks) > 1
    for chunk in chunks:
        n_tokens = len(tokenizer(chunk.text, add_special_tokens=False)["input_ids"])
        assert n_tokens <= 64
        assert chunk.num_tokens <= 64


def test_chunks_overlap(tokenizer):
    page = PageText(page_number=1, text=_long_text(200))
    chunks = chunk_page(page, tokenizer, chunk_size=50, overlap=10)
    for previous, current in zip(chunks, chunks[1:]):
        last_words = previous.text.split()[-10:]
        first_words = current.text.split()[:10]
        assert last_words == first_words  # exactly `overlap` tokens are shared


def test_all_text_is_covered(tokenizer):
    words = _long_text(300).split()
    chunks = chunk_page(PageText(1, " ".join(words)), tokenizer, chunk_size=40, overlap=8)
    covered = set()
    for chunk in chunks:
        covered.update(chunk.text.split())
    assert covered == set(words)


def test_chunk_text_matches_original_offsets(tokenizer):
    page = PageText(page_number=2, text=_long_text(120))
    for chunk in chunk_page(page, tokenizer, chunk_size=30, overlap=5):
        assert page.text[chunk.start_char:chunk.end_char].strip() == chunk.text


def test_words_are_not_split_across_chunk_edges(tokenizer):
    # Punctuation attached to words forms "continuation" tokens; chunks must
    # not start with a dangling "," or end in the middle of "word,".
    text = " ".join(f"item{i}," for i in range(100))
    chunks = chunk_page(PageText(1, text), tokenizer, chunk_size=15, overlap=4)
    for chunk in chunks:
        assert not chunk.text.startswith(",")
        assert chunk.text.endswith(",")


def test_page_numbers_and_ids_preserved_across_pages(tokenizer):
    pages = [
        PageText(1, _long_text(100)),
        PageText(2, ""),  # empty page produces no chunks
        PageText(3, "Short third page."),
    ]
    chunks = chunk_pages(pages, tokenizer, chunk_size=40, overlap=10)
    assert {c.page_number for c in chunks} == {1, 3}
    assert [c.chunk_id for c in chunks] == list(range(len(chunks)))
    page3 = [c for c in chunks if c.page_number == 3]
    assert len(page3) == 1 and page3[0].chunk_index_in_page == 0


def test_empty_page_returns_no_chunks(tokenizer):
    assert chunk_page(PageText(1, "   "), tokenizer, chunk_size=10, overlap=2) == []


@pytest.mark.parametrize("size,overlap", [(0, 0), (10, 10), (10, -1)])
def test_invalid_parameters_rejected(tokenizer, size, overlap):
    with pytest.raises(ValueError):
        chunk_page(PageText(1, "text"), tokenizer, chunk_size=size, overlap=overlap)

