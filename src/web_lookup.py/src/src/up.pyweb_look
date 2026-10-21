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
