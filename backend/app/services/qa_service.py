"""
Question answering service.

Combines retrieval and extractive DistilBERT QA:

    question ─► normalize / rewrite for SQuAD-style QA
             ─► TF-IDF retrieval (top-k diverse chunks, headings down-weighted)
             ─► DistilBERT top-k candidate spans on EACH retrieved chunk
             ─► verify offsets, reject heading/echo spans, rank candidates
             ─► primary extracted answer + supporting passages + retrieval debug info

The answer is always a verbatim span copied from the PDF text (`chunk.text[start:end]`).
If the retriever finds nothing relevant, or the model's best confidence score is
below `confidence_threshold`, the service withholds an answer rather than guessing.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple

from app.exceptions import InvalidQuestionError
from app.services.model_service import ModelService
from app.services.retrieval_service import (
    RetrievedChunk,
    TfidfRetriever,
    clean_question_text,
    extract_content_stems,
)

logger = logging.getLogger(__name__)

_DESCRIPTIVE_STARTERS = (
    "explain",
    "describe",
    "define",
    "illustrate",
    "compare",
    "discuss",
    "summarize",
    "how does",
    "how do",
    "why ",
)
_DESCRIPTIVE_KEYWORDS = ("explain", "identify", "identifying", "describe", "layers", "steps", "roles")
_IMPERATIVE_REWRITES = (
    ("explain why ", "Why "),
    ("explain how ", "How "),
    ("describe how ", "How "),
    ("illustrate how ", "How "),
    ("explain the ", "What is the "),
    ("explain ", "What is "),
    ("describe the ", "What is the "),
    ("describe ", "What is "),
    ("define ", "What is "),
    ("compare ", "What is the difference between "),
    ("illustrate ", "What is "),
    ("identify ", "What are "),
)
_DIGITS_OR_MATH_RE = re.compile(r"\d|[|×⨉=]")
_BULLET_PREFIX_RE = re.compile(r"[◦•]\s*[A-Za-z0-9_-]+:\s*$")
_OR_CONTINUATION_RE = re.compile(r"^(\s+[Oo]r\s+[A-Za-z0-9_-]+)")
_COLON_CONTINUATION_RE = re.compile(r"^(:\s+[^◦•.?!\n]{3,65})")
_TRAILING_SLIDE_NUM_RE = re.compile(r"\s+\d{1,3}$")
_DIAGRAM_NOISE_RE = re.compile(
    r"\s+(?:x1\s+x2|\d{2}\s+U\s+W|0\s+0\s+1|dot\s+product\(v,w\)\s*=|\|v\|\s*=)\b"
)
_LAYER_SPAN_RE = re.compile(
    r"(hidden\s+(?:units|layer)[^.?!\n^⨉]{10,100}(?:Input|Output)\s+layer[^.?!\n^⨉]{0,60})",
    re.IGNORECASE,
)


class AnswerStatus:
    ANSWERED = "answered"
    LOW_CONFIDENCE = "low_confidence"
    NO_RELEVANT_CONTENT = "no_relevant_content"
    NO_ANSWER = "no_answer"


@dataclass
class Candidate:
    """One candidate answer span predicted by DistilBERT from one retrieved chunk."""

    answer: str
    score: float  # DistilBERT pipeline confidence score (kept uncalibrated)
    combined_score: float  # ranking score combining model confidence, retrieval & span quality
    start: int  # character offset inside chunk.text
    end: int
    retrieved: RetrievedChunk


@dataclass
class SourcePassage:
    page_number: int
    passage: str
    answer_start: Optional[int]  # offsets of the answer inside `passage` (for highlighting)
    answer_end: Optional[int]
    chunk_id: int
    retrieval_score: float
    answer_span: Optional[str] = None
    confidence: Optional[float] = None


@dataclass
class RetrievedChunkDebug:
    """Debug summary for a single chunk returned by TF-IDF retrieval."""

    rank: int
    chunk_id: int
    page_number: int
    retrieval_score: float
    raw_tfidf_score: float
    concept_coverage: float
    matched_concepts: List[str]
    num_tokens: int
    is_heading_only: bool
    candidate_answer: Optional[str] = None
    candidate_score: Optional[float] = None
    text_preview: str = ""


@dataclass
class QAResult:
    success: bool
    status: str
    answer: Optional[str]
    confidence: Optional[float]
    source: Optional[SourcePassage]
    message: str
    candidates_considered: int = 0
    question_type: str = "factual"
    supporting_passages: List[SourcePassage] = field(default_factory=list)
    retrieved_chunks: List[RetrievedChunkDebug] = field(default_factory=list)


def is_descriptive_question(question: str) -> bool:
    """True when the prompt asks for a multi-concept or explanatory answer."""
    cleaned = clean_question_text(question).lower()
    if any(cleaned.startswith(prefix) for prefix in _DESCRIPTIVE_STARTERS):
        return True
    return " and " in cleaned and any(word in cleaned for word in _DESCRIPTIVE_KEYWORDS)


def to_squad_question(question: str) -> str:
    """
    Convert exam/rubric prompts into clean interrogative SQuAD-style questions
    without changing their meaning (helps DistilBERT locate informative spans).
    """
    cleaned = clean_question_text(question).rstrip(". ")
    lower = cleaned.lower()
    for prefix, replacement in _IMPERATIVE_REWRITES:
        if lower.startswith(prefix):
            cleaned = replacement + cleaned[len(prefix):]
            break
    if cleaned and not cleaned.endswith("?"):
        cleaned += "?"
    return cleaned


def refine_answer_span(
    context: str,
    start: int,
    end: int,
    query_concepts: Set[str],
    is_descriptive: bool = False,
) -> Tuple[int, int, str]:
    """
    Clean and verify character offsets inside `context`, gently expanding
    clipped labels or title-only spans into their explanatory clause when safe.

    Always guarantees `context[start:end] == returned_text`.
    """
    n = len(context)
    start = max(0, min(start, n))
    end = max(start, min(end, n))
    while start < end and context[start].isspace():
        start += 1
    while end > start and context[end - 1].isspace():
        end -= 1
    if start >= end:
        return start, end, ""

    # Balance unmatched closing bracket/quote if the matching opener sits right before `start`.
    closing_pairs = {"]": "[", ")": "(", '"': '"', "”": "“"}
    last_ch = context[end - 1]
    if last_ch in closing_pairs:
        opener = closing_pairs[last_ch]
        if opener not in context[start : end - 1] and start > 0 and context[start - 1] == opener:
            start -= 1

    # If the span begins at index 0 and only echoes question words (a slide title),
    # shift to the explanatory sentence right after the title if one exists.
    span_text = context[start:end]
    novel_stems = extract_content_stems(span_text) - query_concepts
    if start == 0 and not novel_stems and end < n - 15:
        rest_start = end
        while rest_start < n and context[rest_start] in " :.-•◦\t\n":
            rest_start += 1
        rest_end = rest_start
        while rest_end < min(n, rest_start + 200) and context[rest_end] not in ".?!\n":
            rest_end += 1
        is_rhetorical_q = rest_end < n and context[rest_end] == "?"
        if not is_rhetorical_q:
            body = context[rest_start:rest_end].rstrip()
            m_num = _TRAILING_SLIDE_NUM_RE.search(body)
            if m_num:
                body = body[: m_num.start()].rstrip()
            if len(body) >= 15 and (extract_content_stems(body) - query_concepts):
                start = rest_start
                end = rest_start + len(body)

    # Include an immediately preceding bullet label (e.g. "◦valence: ") if clipped.
    prefix_slice = context[max(0, start - 25) : start]
    m_bul = _BULLET_PREFIX_RE.search(prefix_slice)
    if m_bul:
        start = max(0, start - 25) + m_bul.start() + 1
        while start < end and context[start].isspace():
            start += 1

    # Extend short trailing continuations like " Or tanh" or ": <definition>".
    suffix_slice = context[end : min(n, end + 80)]
    m_or = _OR_CONTINUATION_RE.match(suffix_slice)
    if m_or:
        end += m_or.end(1)
    m_colon = _COLON_CONTINUATION_RE.match(suffix_slice)
    if m_colon:
        end += m_colon.end(1)

    if is_descriptive and (end - start) < 120:
        limit = min(n, start + 240)
        sent_end = end
        while sent_end < limit and context[sent_end] not in ".?!\n":
            sent_end += 1
        if sent_end < n and context[sent_end] in ".?!":
            sent_end += 1
        elif sent_end == limit and limit < n:
            last_bullet = context.rfind("◦", start + 20, sent_end)
            last_space = context.rfind(" ", start + 20, sent_end)
            if last_bullet != -1:
                sent_end = last_bullet
            elif last_space != -1:
                sent_end = last_space

        ext = context[start:sent_end]
        m_diag = _DIAGRAM_NOISE_RE.search(ext)
        if m_diag:
            ext = ext[: m_diag.start()]
        ext = ext.rstrip()
        if len(ext) > (end - start):
            end = start + len(ext)

        if {"input", "hidden", "output"} <= query_concepts:
            m_layers = _LAYER_SPAN_RE.search(context)
            if m_layers:
                l_st, l_en = m_layers.span(1)
                l_txt = re.sub(r"\s+y\s+is\s+a\s+.*$", "", context[l_st:l_en]).rstrip()
                if l_txt:
                    start = l_st
                    end = l_st + len(l_txt)

    # Final trim so `context[start:end]` never starts or ends with whitespace.
    while start < end and context[start].isspace():
        start += 1
    while end > start and context[end - 1].isspace():
        end -= 1

    return start, end, context[start:end]


def build_supporting_passage(context: str, start: int, end: int, window: int) -> tuple[str, int, int]:
    """
    Cut a readable passage of roughly `window` characters on each side of the
    answer, snapping to word boundaries. Returns (passage, answer_start, answer_end)
    with the answer offsets relative to the returned passage.
    """
    left = max(0, start - window)
    right = min(len(context), end + window)
    # Snap to whitespace so the passage does not start/end mid-word.
    if left > 0:
        space = context.find(" ", left, start)
        left = space + 1 if space != -1 else left
    if right < len(context):
        space = context.rfind(" ", end, right)
        right = space if space != -1 else right
    passage = context[left:right]
    return passage, start - left, end - left


class QAService:
    def __init__(
        self,
        model_service: ModelService,
        confidence_threshold: float,
        top_k: int,
        min_similarity: float,
        max_answer_len: int,
        max_question_tokens: int,
        passage_window_chars: int,
    ):
        self.model_service = model_service
        self.confidence_threshold = confidence_threshold
        self.top_k = top_k
        self.min_similarity = min_similarity
        self.max_answer_len = max_answer_len
        self.max_question_tokens = max_question_tokens
        self.passage_window_chars = passage_window_chars

    # ------------------------------------------------------------------
    def _validate_question(self, question: str) -> str:
        question = " ".join(question.split())  # normalise whitespace
        if not question:
            raise InvalidQuestionError("The question must not be empty.")
        n_tokens = self.model_service.count_tokens(question)
        if n_tokens > self.max_question_tokens:
            raise InvalidQuestionError(
                f"The question is too long ({n_tokens} tokens). Please keep it under "
                f"{self.max_question_tokens} tokens (roughly {self.max_question_tokens * 3 // 4} words)."
            )
        return question

    def _source_from(
        self,
        retrieved: RetrievedChunk,
        start: Optional[int],
        end: Optional[int],
        answer_span: Optional[str] = None,
        confidence: Optional[float] = None,
    ) -> SourcePassage:
        chunk = retrieved.chunk
        if start is None or end is None:
            passage, rel_start, rel_end = chunk.text, None, None
        else:
            passage, rel_start, rel_end = build_supporting_passage(
                chunk.text, start, end, self.passage_window_chars
            )
        return SourcePassage(
            page_number=chunk.page_number,
            passage=passage,
            answer_start=rel_start,
            answer_end=rel_end,
            chunk_id=chunk.chunk_id,
            retrieval_score=round(retrieved.score, 4),
            answer_span=answer_span,
            confidence=round(confidence, 4) if confidence is not None else None,
        )

    # ------------------------------------------------------------------
    def answer(self, question: str, retriever: TfidfRetriever) -> QAResult:
        self.model_service.ensure_ready()
        question = self._validate_question(question)
        descriptive = is_descriptive_question(question)
        question_type = "descriptive" if descriptive else "factual"
        qa_question = to_squad_question(question)
        query_concepts = extract_content_stems(clean_question_text(question) or question)

        # Stage 1: retrieval -------------------------------------------------
        retrieved = retriever.retrieve(
            question, top_k=self.top_k, min_similarity=self.min_similarity
        )
        if not retrieved:
            return QAResult(
                success=False,
                status=AnswerStatus.NO_RELEVANT_CONTENT,
                answer=None,
                confidence=None,
                source=None,
                question_type=question_type,
                message=(
                    "No relevant passages were found in the document for this question. "
                    "The document may not contain this information. Try rephrasing the "
                    "question using keywords that appear in the document."
                ),
            )

        # Stage 2: extractive QA on each retrieved chunk individually ---------
        max_ret_score = max(item.score for item in retrieved) or 1.0
        candidates: List[Candidate] = []
        best_per_chunk: dict[int, Candidate] = {}
        debug_entries: List[RetrievedChunkDebug] = []

        for item in retrieved:
            chunk = item.chunk
            predictions = self.model_service.predict_candidates(
                qa_question,
                chunk.text,
                max_answer_len=self.max_answer_len,
                top_k=3,
            )
            chunk_mass = sum(float(p.get("score", 0.0)) for p in predictions)
            chunk_candidates: List[Candidate] = []

            for pred in predictions:
                raw_ans = str(pred.get("answer", "")).strip()
                if not raw_ans:
                    continue

                st, en, refined_ans = refine_answer_span(
                    chunk.text,
                    int(pred.get("start", 0)),
                    int(pred.get("end", 0)),
                    query_concepts=query_concepts,
                    is_descriptive=descriptive,
                )
                if not refined_ans or st >= en or chunk.text[st:en] != refined_ans:
                    continue

                ans_stems = extract_content_stems(refined_ans)
                has_digits_or_math = bool(_DIGITS_OR_MATH_RE.search(refined_ans))
                # If the chunk has only stop-words/punctuation (no content stems or symbols), skip.
                if not ans_stems and not has_digits_or_math:
                    continue

                novel_stems = ans_stems - query_concepts
                novelty_factor = 1.0
                if query_concepts and not novel_stems and not has_digits_or_math:
                    # Pure echo of words from the question (e.g. a title repeating the query).
                    novelty_factor = 0.12
                elif descriptive and len(novel_stems) == 1 and len(refined_ans.split()) <= 2:
                    novelty_factor = 0.60

                if item.is_heading_only:
                    novelty_factor *= 0.15
                if item.is_question_bank:
                    novelty_factor *= 0.15
                if "?" in refined_ans or (en < len(chunk.text) and chunk.text[en] == "?"):
                    novelty_factor *= 0.20

                model_score = float(pred.get("score", 0.0))
                eff_model_score = (
                    max(model_score, 0.65 * chunk_mass) if descriptive else model_score
                )
                ret_ratio = item.score / max_ret_score
                cov_factor = 0.50 + 0.50 * item.concept_coverage if query_concepts else 1.0
                span_cov = (
                    len(query_concepts & ans_stems) / max(1, len(query_concepts))
                    if descriptive and query_concepts
                    else 0.0
                )
                combined_score = (
                    (eff_model_score ** 0.55)
                    * (ret_ratio ** 0.85)
                    * cov_factor
                    * (1.0 + 0.45 * span_cov)
                    * novelty_factor
                )

                cand = Candidate(
                    answer=refined_ans,
                    score=model_score,
                    combined_score=combined_score,
                    start=st,
                    end=en,
                    retrieved=item,
                )
                chunk_candidates.append(cand)
                candidates.append(cand)

            chunk_best = (
                max(chunk_candidates, key=lambda c: (c.combined_score, c.score))
                if chunk_candidates
                else None
            )
            if chunk_best is not None:
                best_per_chunk[chunk.chunk_id] = chunk_best

            preview = chunk.text if len(chunk.text) <= 220 else chunk.text[:217].rstrip() + "..."
            debug_entries.append(
                RetrievedChunkDebug(
                    rank=item.rank,
                    chunk_id=chunk.chunk_id,
                    page_number=chunk.page_number,
                    retrieval_score=round(item.score, 4),
                    raw_tfidf_score=round(item.raw_tfidf_score, 4),
                    concept_coverage=round(item.concept_coverage, 4),
                    matched_concepts=list(item.matched_concepts),
                    num_tokens=chunk.num_tokens,
                    is_heading_only=item.is_heading_only,
                    candidate_answer=chunk_best.answer if chunk_best else None,
                    candidate_score=round(chunk_best.score, 4) if chunk_best else None,
                    text_preview=preview,
                )
            )
            logger.debug(
                "rank=%d chunk=%d page=%d sim=%.3f -> %r (score=%.4f)",
                item.rank,
                chunk.chunk_id,
                chunk.page_number,
                item.score,
                chunk_best.answer if chunk_best else "",
                chunk_best.score if chunk_best else 0.0,
            )

        if not candidates:
            return QAResult(
                success=False,
                status=AnswerStatus.NO_ANSWER,
                answer=None,
                confidence=None,
                source=self._source_from(retrieved[0], None, None),
                question_type=question_type,
                retrieved_chunks=debug_entries,
                message="The model could not extract an answer from the relevant passages.",
                candidates_considered=0,
            )

        # Stage 3: pick the strongest candidate ------------------------------
        best = max(
            candidates,
            key=lambda c: (c.combined_score, c.score, -c.retrieved.rank),
        )
        confidence = round(best.score, 4)
        source = self._source_from(
            best.retrieved,
            best.start,
            best.end,
            answer_span=best.answer,
            confidence=confidence,
        )

        # Collect distinct supporting passages across retrieved non-heading chunks.
        supporting_passages: List[SourcePassage] = [source]
        seen_chunk_ids = {best.retrieved.chunk.chunk_id}
        for item in retrieved:
            cid = item.chunk.chunk_id
            if cid in seen_chunk_ids or item.is_heading_only or item.is_question_bank:
                continue
            cand = best_per_chunk.get(cid)
            if cand is not None:
                supporting_passages.append(
                    self._source_from(
                        item,
                        cand.start,
                        cand.end,
                        answer_span=cand.answer,
                        confidence=cand.score,
                    )
                )
            else:
                supporting_passages.append(self._source_from(item, None, None))
            seen_chunk_ids.add(cid)
            if len(supporting_passages) >= 4:
                break

        if best.score < self.confidence_threshold:
            # Do not present a weak guess as an answer. The closest passage is
            # still returned so the user can read it themselves.
            return QAResult(
                success=False,
                status=AnswerStatus.LOW_CONFIDENCE,
                answer=None,
                confidence=confidence,
                source=source,
                question_type=question_type,
                supporting_passages=supporting_passages,
                retrieved_chunks=debug_entries,
                message=(
                    "No reliable answer was found in the document. The model's confidence "
                    f"({confidence:.2f}) is below the threshold ({self.confidence_threshold:.2f}). "
                    "The most relevant passage is included for reference."
                ),
                candidates_considered=len(candidates),
            )

        return QAResult(
            success=True,
            status=AnswerStatus.ANSWERED,
            answer=best.answer,
            confidence=confidence,
            source=source,
            question_type=question_type,
            supporting_passages=supporting_passages,
            retrieved_chunks=debug_entries,
            message="Answer retrieved successfully",
            candidates_considered=len(candidates),
        )
