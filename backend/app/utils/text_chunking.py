r"""
Token-aware text chunking.

DistilBERT can only read 512 tokens at a time (question + context + special
tokens). Long pages are therefore split into overlapping windows measured in
*DistilBERT tokens* (not characters or words), so every chunk is guaranteed to
fit into the model together with the question.

    page text:  [t0 t1 t2 ... t383][t320 ... t703][t640 ... ]
                 \___ chunk 1 ___/ \__ chunk 2 __/
                          overlap = 64 tokens

The overlap means an answer that falls on a chunk boundary still appears
complete in at least one chunk.

Chunks never cross page boundaries, so every chunk belongs to exactly one PDF
page and the page number of an answer is always known.
"""

from dataclasses import dataclass
from typing import Any, List, Sequence, Tuple


@dataclass
class PageText:
    """Text extracted from a single PDF page (page numbers start at 1)."""

    page_number: int
    text: str


@dataclass
class TextChunk:
    """A piece of a page that is small enough for the QA model."""

    chunk_id: int  # position of the chunk in the whole document (0-based)
    page_number: int  # original PDF page number (1-based)
    chunk_index_in_page: int  # 0-based index of the chunk within its page
    text: str
    start_char: int  # character offset of the chunk inside the page text
    end_char: int
    num_tokens: int


def _token_offsets(text: str, tokenizer: Any) -> List[Tuple[int, int]]:
    """
    Return the (start_char, end_char) span of every token in `text`.

    Works with any Hugging Face *fast* tokenizer (DistilBERT ships one).
    Special tokens are not added because they are added later by the QA pipeline.
    """
    encoding = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        truncation=False,
        verbose=False,
    )
    return [tuple(span) for span in encoding["offset_mapping"]]


def _is_continuation(offsets: Sequence[Tuple[int, int]], index: int) -> bool:
    """
    True when token `index` directly continues the previous token without a
    space in between (e.g. word pieces "token" + "##izer", or "word" + ",").
    Used to avoid starting or ending a chunk in the middle of a word.
    """
    return 0 < index < len(offsets) and offsets[index][0] == offsets[index - 1][1]


def chunk_page(
    page: PageText,
    tokenizer: Any,
    chunk_size: int,
    overlap: int,
    first_chunk_id: int = 0,
) -> List[TextChunk]:
    """Split a single page into overlapping, token-limited chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    text = page.text
    if not text or not text.strip():
        return []

    offsets = _token_offsets(text, tokenizer)
    if not offsets:
        return []

    chunks: List[TextChunk] = []
    n_tokens = len(offsets)
    stride = chunk_size - overlap
    start = 0

    while start < n_tokens:
        end = min(start + chunk_size, n_tokens)  # exclusive token index

        # Do not cut a word in half: move the window edges to word boundaries.
        # Moving inwards keeps the chunk within the token budget; the overlap
        # with neighbouring chunks makes sure no text is lost.
        chunk_start = start
        while chunk_start < end - 1 and _is_continuation(offsets, chunk_start):
            chunk_start += 1
        chunk_end = end
        if chunk_end < n_tokens:
            while chunk_end > chunk_start + 1 and _is_continuation(offsets, chunk_end):
                chunk_end -= 1

        start_char = offsets[chunk_start][0]
        end_char = offsets[chunk_end - 1][1]
        chunk_text = text[start_char:end_char].strip()

        if chunk_text:
            chunks.append(
                TextChunk(
                    chunk_id=first_chunk_id + len(chunks),
                    page_number=page.page_number,
                    chunk_index_in_page=len(chunks),
                    text=chunk_text,
                    start_char=start_char,
                    end_char=end_char,
                    num_tokens=chunk_end - chunk_start,
                )
            )

        if end >= n_tokens:
            break
        start += stride

    return chunks


def chunk_pages(
    pages: Sequence[PageText],
    tokenizer: Any,
    chunk_size: int,
    overlap: int,
) -> List[TextChunk]:
    """Chunk every page and number the chunks across the whole document."""
    all_chunks: List[TextChunk] = []
    for page in pages:
        all_chunks.extend(
            chunk_page(page, tokenizer, chunk_size, overlap, first_chunk_id=len(all_chunks))
        )
    return all_chunks
