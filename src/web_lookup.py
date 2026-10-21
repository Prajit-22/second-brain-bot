"""Keyless web lookup, backed by Wikipedia's public APIs.

No API key and no paid service: the opensearch endpoint finds candidate
pages and the REST summary endpoint returns a short, sourced extract. This
file only builds URLs and parses JSON - the Worker (entry.py) does the
actual fetching, so every function here is unit tested offline.
"""

from typing import Dict, List, Optional, Tuple
from urllib.parse import quote, urlencode

_WIKI_API = "https://en.wikipedia.org/w/api.php"
_WIKI_SUMMARY = "https://en.wikipedia.org/api/rest_v1/page/summary/"
MAX_EXTRACT_CHARS = 1200
MAX_ARTICLE_EXTRACT_CHARS = 2500


def wikipedia_search_url(query: str, limit: int = 3) -> str:
    return _WIKI_API + "?" + urlencode(
        {
            "action": "opensearch",
            "search": query,
            "limit": str(limit),
            "namespace": "0",
            "format": "json",
        }
    )


def wikipedia_fulltext_search_url(query: str, limit: int = 3) -> str:
    """Fallback for queries where opensearch finds no title: MediaWiki's
    full-text search matches article body text, not just titles."""
    return _WIKI_API + "?" + urlencode(
        {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": str(limit),
            "srnamespace": "0",
            "format": "json",
        }
    )


def parse_fulltext_results(payload) -> List[str]:
    """action=query&list=search answers {"query": {"search": [{"title": ...}]}}."""
    if not isinstance(payload, dict):
        return []
    hits = (payload.get("query") or {}).get("search") or []
    return [
        str(hit["title"])
        for hit in hits
        if isinstance(hit, dict) and isinstance(hit.get("title"), str)
    ]


def wikipedia_summary_url(title: str) -> str:
    return _WIKI_SUMMARY + quote(title.replace(" ", "_"), safe="")


def parse_search_results(payload) -> List[Tuple[str, str]]:
    """opensearch answers [query, [titles], [descriptions], [urls]]."""
    if not isinstance(payload, list) or len(payload) < 4:
        return []
    titles, urls = payload[1], payload[3]
    if not isinstance(titles, list) or not isinstance(urls, list):
        return []
    return [
        (str(title), str(url))
        for title, url in zip(titles, urls)
        if isinstance(title, str) and isinstance(url, str)
    ]


def parse_summary(payload) -> Optional[Dict[str, str]]:
    """Keep only real articles ('standard' type) with a usable extract -
    disambiguation pages and missing extracts are skipped by the caller
    trying the next search hit."""
    if not isinstance(payload, dict):
        return None
    if payload.get("type") != "standard":
        return None
    extract = " ".join(str(payload.get("extract") or "").split())
    title = str(payload.get("title") or "").strip()
    urls = payload.get("content_urls") or {}
    desktop = urls.get("desktop") or {}
    page_url = str(desktop.get("page") or "").strip()
    if not extract or not title or not page_url:
        return None
    return {
        "title": title,
        "extract": extract[:MAX_EXTRACT_CHARS],
        "url": page_url,
    }


def format_web_reply(result: Dict[str, str]) -> str:
    """The reply is always labeled as coming from the web (Wikipedia), so
    it can never be confused with an answer grounded in saved notes."""
    return "{title}\n\n{extract}\n\nFrom Wikipedia (web, not your notes): {url}".format(
        title=result["title"], extract=result["extract"], url=result["url"]
    )


_WIKI_ARTICLE_URL = "https://en.wikipedia.org/wiki/"

# Leading question phrases stripped to turn a question into a title search:
# "who is the prime minister of india?" -> "the prime minister of india",
# which opensearch can match to the "Prime Minister of India" article.
_QUESTION_PREFIXES = (
    "who is", "who's", "who are", "who was", "who were",
    "what is", "what's", "what are", "what was", "what were",
    "when is", "when was", "when did", "when will",
    "where is", "where are", "where was",
    "which is", "which are", "which was",
    "how many", "how much", "how old", "how tall", "how far", "how long",
    "why is", "why are", "why did",
    "is", "are", "was", "were", "do", "does", "did", "can",
    "kya", "kaun", "kab", "kaise", "kitna", "kitne", "kahan", "kyu", "kyun",
)


def search_query(question: str) -> str:
    """Turn a natural-language question into a Wikipedia title-search query
    by dropping the question mark and leading question words. Returns the
    original (minus the mark) when stripping would leave nothing."""
    original = " ".join(question.split()).strip().rstrip("?").strip()
    query = original
    # Iterate: "how old is the Eiffel Tower?" sheds "how old", then "is".
    for _ in range(3):
        lower = query.lower()
        stripped = None
        for prefix in _QUESTION_PREFIXES:
            if lower.startswith(prefix + " "):
                stripped = query[len(prefix) + 1:]
                break
            if lower == prefix:
                stripped = ""
                break
        if stripped is None:
            break
        query = stripped.strip()
    return query or original


def wikipedia_extract_url(title: str) -> str:
    """Full plain-text lead of an article (prop=extracts, intro only). The
    lead is richer than the REST summary - leads of office-holder articles
    name the incumbent, which is what "who is the current X?" needs."""
    return _WIKI_API + "?" + urlencode(
        {
            "action": "query",
            "prop": "extracts|info",
            "inprop": "url",
            "exintro": "1",
            "explaintext": "1",
            "redirects": "1",
            "titles": title,
            "format": "json",
        }
    )


def parse_extract(payload) -> Optional[Dict[str, str]]:
    """action=query&prop=extracts|info answers {"query": {"pages": {<id>:
    {"title", "extract", "fullurl"}}}. Missing pages carry "missing": ""."""
    if not isinstance(payload, dict):
        return None
    pages = (payload.get("query") or {}).get("pages") or {}
    if not isinstance(pages, dict):
        return None
    for page in pages.values():
        if not isinstance(page, dict) or "missing" in page:
            continue
        extract = " ".join(str(page.get("extract") or "").split())
        title = str(page.get("title") or "").strip()
        url = str(page.get("fullurl") or "").strip()
        if extract and title and url:
            return {
                "title": title,
                "extract": extract[:MAX_ARTICLE_EXTRACT_CHARS],
                "url": url,
            }
    return None


def format_web_answer_reply(answer: str, article: Dict[str, str]) -> str:
    """Web-grounded answers (the /ask fallback) carry the same honesty
    label as /web: Wikipedia, not the user's notes, plus the source URL."""
    return "{answer}\n\nFrom Wikipedia (web, not your notes): {url}".format(
        answer=answer, url=article["url"]
    )
