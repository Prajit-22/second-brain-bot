"""Pure text-processing and ranking helpers for a local second brain.

Zero I/O, zero third-party dependencies - every function is plain Python.
Adapted from the second-brain-bot project (github.com/Prajit-22/second-brain-bot),
where these same helpers power the live Telegram bot's hybrid retrieval.
"""

import hashlib
import re
import unicodedata
from typing import Dict, List, Optional, Sequence, Tuple

_WS_RE = re.compile(r"\s+")


def tokenize(text: str) -> List[str]:
    """Unicode-aware lowercase tokenizer.

    Keeps letters/digits from every script AND Unicode combining marks
    (category M*), because scripts like Devanagari encode vowels as marks:
    a plain \\w+ regex would shred Hindi words into fragments."""
    keep = [
        ch if (ch.isalnum() or unicodedata.category(ch).startswith("M")) else " "
        for ch in text.lower()
    ]
    return "".join(keep).split()


def content_hash(text: str) -> str:
    """Stable fingerprint used to dedupe items saved twice."""
    return hashlib.sha256(_WS_RE.sub(" ", text.strip()).encode("utf-8")).hexdigest()


def chunk_text(text: str, max_chars: int = 900, overlap: int = 150) -> List[str]:
    """Split text into overlapping chunks, preferring word boundaries.

    - Every chunk is at most ``max_chars`` long.
    - Consecutive chunks share roughly ``overlap`` characters so a fact that
      straddles a boundary still appears whole in at least one chunk.
    - Guarantees progress on pathological input (a single 10k-char "word").
    """
    if overlap >= max_chars // 2:
        raise ValueError("overlap must be less than half of max_chars")
    text = _WS_RE.sub(" ", text.strip())
    if not text:
        return []
    chunks: List[str] = []
    start, n = 0, len(text)
    while start < n:
        end = min(start + max_chars, n)
        if end < n:
            # Snap back to a space so we do not cut a word in half, but never
            # snap into the first half of the window (no progress risk).
            space = text.rfind(" ", start + max_chars // 2, end)
            if space != -1:
                end = space
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= n:
            break
        start = max(end - overlap, start + 1)
        while start < n and text[start] == " ":
            start += 1
    return chunks


def fts_query(question: str, max_terms: int = 12) -> Optional[str]:
    """Build a safe SQLite FTS5 MATCH query from natural language.

    Each token is wrapped in double quotes (FTS5's escape hatch, so input
    can never inject query syntax) and terms are OR-ed so word order does
    not matter. Returns None when the question has no usable terms.
    """
    seen, terms = set(), []
    for tok in tokenize(question):
        if len(tok) < 2 or tok in seen:
            continue
        seen.add(tok)
        terms.append(tok)
        if len(terms) >= max_terms:
            break
    if not terms:
        return None
    return " OR ".join('"' + t.replace('"', '""') + '"' for t in terms)


# --- relevance gating ------------------------------------------------------

STOPWORDS = frozenset(
    {
        "who", "what", "when", "where", "why", "how", "which", "whose", "whom",
        "is", "are", "was", "were", "am", "do", "does", "did", "can", "could",
        "should", "would", "will", "shall", "may", "might",
        "the", "an", "of", "to", "in", "on", "for", "and", "or", "but", "my",
        "me", "it", "its", "this", "that", "these", "those", "with", "about",
        "from", "by", "at", "as", "be", "been", "being", "we", "you", "your",
        "our", "us", "them", "they", "he", "she", "his", "her", "their",
        "there", "here", "not", "no", "so", "if", "then", "than", "into",
        "over", "under", "between", "all", "any", "some", "more", "most",
        "much", "many", "own", "same", "just", "also", "very", "really",
        "get", "got", "have", "has", "had",
        # common Hinglish function words (roman script)
        "kya", "kaun", "kab", "kahan", "kyun", "kaise", "kitna", "kitne",
        "hai", "hain", "ka", "ki", "ke", "ko", "se", "mein", "par", "tha",
        "thi", "ho", "hoga", "hogi", "aur", "ya", "bhi", "nahi",
    }
)


def content_terms(text: str) -> List[str]:
    """The words that carry meaning: length >= 3 and not stopwords."""
    return [tok for tok in tokenize(text) if len(tok) >= 3 and tok not in STOPWORDS]


def _term_in_words(term: str, words) -> bool:
    if term in words:
        return True
    # cheap plural/singular tolerance: ranking/rankings, ideas/idea
    if term.endswith("s") and term[:-1] in words:
        return True
    return term + "s" in words


def keyword_relevance(query: str, text: str) -> float:
    """Fraction of the query's content terms that literally appear in text.

    Returns 0.0 when the query has no content terms at all - then only the
    vector score can vouch for a chunk."""
    terms = set(content_terms(query))
    if not terms:
        return 0.0
    words = set(tokenize(text))
    hits = sum(1 for term in terms if _term_in_words(term, words))
    return hits / len(terms)


# A chunk survives retrieval only when the keyword side covers at least a
# third of the query's content terms, or the vector side is this similar.
# RRF *ranks* candidates; these thresholds keep a stray note out of an
# unrelated answer just because it shared a stopword or floated into the
# unconditional topK.
MIN_KEYWORD_RELEVANCE = 0.34
MIN_VECTOR_SCORE = 0.55


def filter_relevant(
    query,
    contexts: Sequence[Dict[str, str]],
    vector_scores: Optional[Dict[str, float]] = None,
    min_keyword: float = MIN_KEYWORD_RELEVANCE,
    min_vector: float = MIN_VECTOR_SCORE,
) -> List[Dict[str, str]]:
    """Drop retrieved chunks that neither retrieval side actually vouches
    for. ``contexts`` are chunk dicts (id/title/text); best-first order is
    preserved. ``query`` may be one string or several: a chunk passes the
    keyword gate when it is relevant to ANY of them."""
    queries = [query] if isinstance(query, str) else list(query)
    scores = vector_scores or {}
    kept = []
    for ctx in contexts:
        vector_score = scores.get(ctx.get("id"))
        text = "{title} {text}".format(title=ctx.get("title", ""), text=ctx.get("text", ""))
        relevance = max((keyword_relevance(q, text) for q in queries), default=0.0)
        if relevance >= min_keyword or (vector_score is not None and vector_score >= min_vector):
            kept.append(ctx)
    return kept


def rrf_merge(
    rankings: Sequence[Sequence[str]], k: int = 60, limit: int = 10
) -> List[Tuple[str, float]]:
    """Reciprocal Rank Fusion over several ranked id lists.

    Each hit scores 1 / (k + rank) per list; ids present in both the keyword
    (BM25) and vector rankings bubble to the top without either score scale
    mattering. Ties break by id so the output order is deterministic.
    """
    scores: Dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return ordered[:limit]


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity of two equal-length vectors (0..1 for embeddings)."""
    import math

    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)
