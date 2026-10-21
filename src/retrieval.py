"""Pure text-processing and ranking helpers for the second-brain bot.

Everything here is plain Python with zero I/O, so each function is unit
tested and easy to explain line by line. The Cloudflare-facing glue lives in
store.py / entry.py; nothing in this file knows the Workers runtime exists.
"""

import hashlib
import html
import re
import unicodedata
from typing import Dict, List, Optional, Sequence, Tuple

_WS_RE = re.compile(r"\s+")
_TAG_RE = re.compile(r"<[^>]+>")
_SCRIPT_STYLE_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.I | re.S)
_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.I | re.S)


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
    """Stable fingerprint used to dedupe items the user saves twice."""
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
            # snap into the first half of the window (that would risk no
            # progress on text with very long words).
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
    """Build a safe SQLite FTS5 MATCH query from a natural-language question.

    Each token is wrapped in double quotes (FTS5's escape hatch, so user
    input can never inject query syntax) and terms are OR-ed so word order
    does not matter. Returns None when the question has no usable terms.
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


def extract_title(html_text: str, fallback: str = "") -> str:
    """Pull the <title> out of a fetched page."""
    match = _TITLE_RE.search(html_text)
    if not match:
        return fallback
    return html.unescape(_WS_RE.sub(" ", match.group(1)).strip()) or fallback


def html_to_text(html_text: str, max_chars: int = 20000) -> str:
    """Crude but dependency-free readability: drop script/style, strip tags,
    unescape entities and collapse whitespace. Good enough to index and
    summarize a page; not a full browser."""
    text = _SCRIPT_STYLE_RE.sub(" ", html_text)
    text = _TAG_RE.sub(" ", text)
    text = html.unescape(text)
    return _WS_RE.sub(" ", text).strip()[:max_chars]
