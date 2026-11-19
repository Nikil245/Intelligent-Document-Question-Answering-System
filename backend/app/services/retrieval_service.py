"""
Document retrieval with TF-IDF, morphological normalization, concept coverage,
and cosine similarity.

Why retrieval is needed:
    DistilBERT reads at most 512 tokens at once, but a PDF can contain
    tens of thousands. Running the QA model on every chunk would be slow, so
    we first select the top-k chunks most likely to contain the answer.

Improvements for multi-page academic PDFs (slides, textbooks, question banks):
    1. Strips exam rubric noise such as "(06 Marks)" or leading question numbers
       from queries so they do not match exercise lists.
    2. Applies lightweight morphological normalization (e.g., "layers" -> "layer",
       "networks" -> "network") alongside surface forms so singular/plural forms
       align.
    3. Combines TF-IDF cosine similarity with IDF-weighted concept coverage so
       multi-concept questions prefer chunks covering more query concepts.
    4. Down-weights heading-only divider slides and exam question-bank pages so
       informative content chunks are ranked above bare section titles.
    5. Encourages chunk diversity so near-identical slide builds do not consume
       all top-k slots.

This remains purely lexical (TF-IDF + cosine similarity); no neural embedding
model is used.
"""

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Sequence, Set

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.utils.text_chunking import TextChunk

# Lightweight stop-word list: question words, function words, and academic prompt
# verbs/labels that carry little topical signal (scikit-learn's built-in list
# is still avoided because it removes words like "system", "first", "two", "name").
STOP_WORDS = sorted(
    {
        "a", "an", "the", "and", "or", "but", "if", "then", "else", "of", "at",
        "by", "for", "with", "about", "to", "from", "in", "on", "into", "onto",
        "as", "is", "are", "was", "were", "be", "been", "being", "am", "do",
        "does", "did", "doing", "have", "has", "had", "having", "it", "its",
        "this", "that", "these", "those", "there", "here", "i", "me", "my",
        "we", "our", "you", "your", "he", "him", "his", "she", "her", "they",
        "them", "their", "what", "which", "who", "whom", "whose", "when",
        "where", "why", "how", "can", "could", "would", "should", "will",
        "shall", "may", "might", "must", "so", "than", "too", "very", "just",
        "also", "any", "some", "such", "no", "not", "nor", "only", "own",
        "same", "s", "t", "please", "tell", "explain", "describe", "give",
        "according", "mentioned", "mention",
        # Academic prompt framing words
        "identify", "identifying", "illustrate", "illustrating", "define", "defining",
        "compare", "comparing", "discuss", "discussing", "determine", "determining",
        "calculate", "calculating", "list", "state", "suitable", "example", "examples",
        "respect", "marks", "mark", "briefly", "detail", "details", "concept", "concepts",
    }
)
_STOP_SET = set(STOP_WORDS)

# Verbs indicating an explanatory sentence rather than a bare slide/section title.
_FINITE_VERBS = {
    "is", "are", "was", "were", "can", "could", "will", "would", "should",
    "has", "have", "had", "do", "does", "did", "means", "represents", "defines",
    "computes", "uses", "takes", "gives", "tells", "allows", "relies", "converts",
    "learns", "hosts", "exports", "prefer", "preferred", "called", "need", "vary",
    "outperform", "favors", "favor",
}

_TOKEN_RE = re.compile(r"(?u)\b\w+\b")
_MARKS_PAREN_RE = re.compile(r"\(\s*\d+\s*marks?[^)]*\)", re.IGNORECASE)
_MARKS_BRACKET_RE = re.compile(r"\[\s*\d+\s*marks?[^\]]*\]", re.IGNORECASE)
_MARKS_INLINE_RE = re.compile(r"\b\d+\s*marks?\b", re.IGNORECASE)
_LEADING_QNUM_RE = re.compile(r"^\s*(?:q(?:uestion)?\s*)?\d+[\.\)]\s*", re.IGNORECASE)
_QB_MARKS_RE = re.compile(r"\b\d+\s*marks?\b|\bL[1-6]\s*\|\s*CO\d+\b", re.IGNORECASE)
_QB_PROMPTS_RE = re.compile(
    r"\b\d+\.\s+(?:Explain|Define|Describe|Illustrate|Compare|Discuss|Identify|Determine|Calculate|Given|With)\b"
)


def clean_question_text(question: str) -> str:
    """Strip exam rubric tags like '(06 Marks)' and leading question numbers."""
    cleaned = _MARKS_PAREN_RE.sub(" ", question)
    cleaned = _MARKS_BRACKET_RE.sub(" ", cleaned)
    cleaned = _MARKS_INLINE_RE.sub(" ", cleaned)
    cleaned = _LEADING_QNUM_RE.sub("", cleaned)
    return " ".join(cleaned.split())


def stem_word(word: str) -> str:
    """Lightweight deterministic morphological normalizer for English plurals/suffixes."""
    w = word.lower()
    if len(w) <= 3:
        return w
    if w.endswith("atic") and len(w) > 6:
        return w[:-4]
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith("es") and len(w) > 4 and w[-3] in "sxz":
        return w[:-2]
    if w.endswith("s") and not w.endswith("ss") and len(w) > 3:
        return w[:-1]
    return w


def stem_normalized_text(text: str) -> str:
    """Return space-joined stem-normalized tokens preserving original word order."""
    tokens = _TOKEN_RE.findall(text.lower())
    return " ".join(stem_word(tok) for tok in tokens)


def extract_content_stems(text: str, include_stop_words: bool = False) -> Set[str]:
    """Return distinct stemmed tokens in `text` (filtering stop-words by default)."""
    stems: Set[str] = set()
    for tok in _TOKEN_RE.findall(text.lower()):
        if len(tok) <= 1 and not include_stop_words:
            continue
        s = stem_word(tok)
        if not include_stop_words and (tok in _STOP_SET or s in _STOP_SET):
            continue
        stems.add(s)
    return stems


def is_heading_only_chunk(text: str, total_chunks: int, duplicate_counts: Counter) -> bool:
    """
    Detect short section-divider slides or repeated bare titles that contain no
    explanatory prose. Disabled for tiny test documents (<= 3 chunks).
    """
    if total_chunks <= 3:
        return False
    words = text.split()
    n_words = len(words)
    norm = text.strip().lower()
    if duplicate_counts[norm] >= 2 and n_words <= 16:
        return True
    if n_words <= 14 and len(text) < 115:
        has_explanatory_punct = any(ch in text for ch in '.?!;"“”=•◦à')
        lower_words = {w.lower().strip(",:()") for w in words}
        has_verb = bool(lower_words & _FINITE_VERBS)
        if not has_explanatory_punct and not has_verb:
            return True
    return False


def is_question_bank_chunk(text: str) -> bool:
    """Detect chunks that list exam/assignment questions rather than explanations."""
    if "question bank" in text.lower():
        return True
    if len(_QB_MARKS_RE.findall(text)) >= 2:
        return True
    return len(_QB_PROMPTS_RE.findall(text)) >= 3


@dataclass
class RetrievedChunk:
    """A chunk selected by the retriever, with similarity and debug metadata."""

    chunk: TextChunk
    score: float  # combined retrieval score in [0, 1]
    rank: int  # 1 = most relevant
    raw_tfidf_score: float = 0.0
    concept_coverage: float = 0.0
    matched_concepts: List[str] = field(default_factory=list)
    is_heading_only: bool = False
    is_question_bank: bool = False


class TfidfRetriever:
    """TF-IDF index over the chunks of ONE document."""

    def __init__(self, chunks: Sequence[TextChunk]):
        if not chunks:
            raise ValueError("Cannot build a retriever without any chunks.")
        self.chunks: List[TextChunk] = list(chunks)
        normalized_texts = [stem_normalized_text(chunk.text) for chunk in self.chunks]

        self._using_stop_words = True
        self.vectorizer = self._make_vectorizer(STOP_WORDS)
        try:
            self.matrix = self.vectorizer.fit_transform(normalized_texts)
        except ValueError:
            # "empty vocabulary": the document only contains stop words or
            # punctuation. Retry without stop-word removal.
            self._using_stop_words = False
            self.vectorizer = self._make_vectorizer(None)
            self.matrix = self.vectorizer.fit_transform(
                [t if t.strip() else chunk.text for t, chunk in zip(normalized_texts, self.chunks)]
            )

        duplicate_counts = Counter(chunk.text.strip().lower() for chunk in self.chunks)
        total = len(self.chunks)
        self._heading_flags = [
            is_heading_only_chunk(chunk.text, total, duplicate_counts) for chunk in self.chunks
        ]
        self._question_bank_flags = [is_question_bank_chunk(chunk.text) for chunk in self.chunks]
        self._chunk_stems = [
            extract_content_stems(chunk.text, include_stop_words=not self._using_stop_words)
            for chunk in self.chunks
        ]

    @staticmethod
    def _make_vectorizer(stop_words) -> TfidfVectorizer:
        return TfidfVectorizer(
            lowercase=True,
            stop_words=stop_words,
            ngram_range=(1, 2),  # single words and two-word phrases
            sublinear_tf=True,  # use 1 + log(tf) so repeated words do not dominate
            token_pattern=r"(?u)\b\w+\b",  # keep 1-character tokens such as "C" or "5"
        )

    @property
    def vocabulary_size(self) -> int:
        return len(self.vectorizer.vocabulary_)

    def term_idf(self, term: str) -> float:
        """Return fitted IDF weight for a stemmed term (defaults to 1.0 if unseen)."""
        idx = self.vectorizer.vocabulary_.get(term)
        if idx is None:
            return 1.0
        return float(self.vectorizer.idf_[idx])

    def retrieve(
        self,
        question: str,
        top_k: int = 5,
        min_similarity: float = 0.0,
    ) -> List[RetrievedChunk]:
        """
        Return up to `top_k` diverse, relevant chunks for `question`, best first.

        An empty list means no chunk shares meaningful vocabulary with the
        question, i.e. the document probably does not discuss it.
        """
        cleaned_question = clean_question_text(question) or question
        norm_query = stem_normalized_text(cleaned_question) or cleaned_question.lower()
        question_vector = self.vectorizer.transform([norm_query])
        if question_vector.nnz == 0:
            # None of the question's indexed terms occur anywhere in the document.
            return []

        raw_similarities = cosine_similarity(question_vector, self.matrix).ravel()
        query_concepts = extract_content_stems(
            cleaned_question, include_stop_words=not self._using_stop_words
        )
        total_query_idf = sum(self.term_idf(t) for t in query_concepts) or 1.0

        candidates = []
        for idx, raw_val in enumerate(raw_similarities):
            raw_score = float(raw_val)
            if raw_score <= 0.0:
                continue

            matched = sorted(query_concepts & self._chunk_stems[idx]) if query_concepts else []
            idf_cov = (
                sum(self.term_idf(t) for t in matched) / total_query_idf
                if query_concepts
                else 1.0
            )

            # Multi-concept questions benefit from boosting chunks that cover more
            # distinct query concepts rather than repeating a single word.
            if len(query_concepts) >= 2:
                adjusted = raw_score * (0.45 + 0.85 * idf_cov)
            else:
                adjusted = raw_score

            is_heading = self._heading_flags[idx]
            is_qb = self._question_bank_flags[idx]
            if is_heading:
                adjusted *= 0.20
            if is_qb:
                adjusted *= 0.15

            adjusted = min(1.0, adjusted)
            if adjusted <= min_similarity:
                continue

            candidates.append(
                (adjusted, raw_score, idf_cov, matched, is_heading, is_qb, idx)
            )

        if not candidates:
            return []

        # Highest adjusted score first; break ties by original chunk order.
        candidates.sort(key=lambda item: (-item[0], item[6]))

        selected = []
        deferred = []
        for item in candidates:
            idx = item[6]
            stems_i = self._chunk_stems[idx]
            norm_i = self.chunks[idx].text.strip().lower()
            is_near_dup = False
            for sel in selected:
                sel_idx = sel[6]
                if norm_i == self.chunks[sel_idx].text.strip().lower():
                    is_near_dup = True
                    break
                stems_sel = self._chunk_stems[sel_idx]
                union_len = len(stems_i | stems_sel)
                if union_len > 0 and (len(stems_i & stems_sel) / union_len) > 0.75:
                    is_near_dup = True
                    break
            if is_near_dup:
                deferred.append(item)
            else:
                selected.append(item)
                if len(selected) >= top_k:
                    break

        if len(selected) < top_k and deferred:
            selected_texts = {self.chunks[s[6]].text.strip().lower() for s in selected}
            for item in deferred:
                norm_i = self.chunks[item[6]].text.strip().lower()
                if norm_i not in selected_texts:
                    selected.append(item)
                    selected_texts.add(norm_i)
                    if len(selected) >= top_k:
                        break
            if len(selected) < top_k:
                for item in deferred:
                    if item not in selected:
                        selected.append(item)
                        if len(selected) >= top_k:
                            break
            # Maintain strictly descending score order after any backfill.
            selected.sort(key=lambda item: (-item[0], item[6]))

        results: List[RetrievedChunk] = []
        for rank, (score, raw_score, idf_cov, matched, is_heading, is_qb, idx) in enumerate(
            selected, start=1
        ):
            results.append(
                RetrievedChunk(
                    chunk=self.chunks[idx],
                    score=score,
                    rank=rank,
                    raw_tfidf_score=raw_score,
                    concept_coverage=idf_cov,
                    matched_concepts=matched,
                    is_heading_only=is_heading,
                    is_question_bank=is_qb,
                )
            )
        return results
