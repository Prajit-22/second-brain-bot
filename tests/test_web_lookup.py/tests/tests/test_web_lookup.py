from web_lookup import (
    format_web_reply,
    parse_search_results,
    parse_summary,
    wikipedia_search_url,
    wikipedia_summary_url,
)


class TestUrls:
    def test_search_url_encodes_query(self):
        url = wikipedia_search_url("retrieval augmented generation")
        assert url.startswith("https://en.wikipedia.org/w/api.php?")
        assert "search=retrieval+augmented+generation" in url
        assert "format=json" in url

    def test_summary_url_underscores_spaces(self):
        url = wikipedia_summary_url("Retrieval-augmented generation")
        assert url.endswith("/page/summary/Retrieval-augmented_generation")


class TestParseSearchResults:
    def test_pairs_titles_with_urls(self):
        payload = ["rag", ["RAG", "Raga"], ["d1", "d2"], ["https://e/1", "https://e/2"]]
        assert parse_search_results(payload) == [("RAG", "https://e/1"), ("Raga", "https://e/2")]

    def test_malformed_payload_returns_empty(self):
        assert parse_search_results({"oops": 1}) == []
        assert parse_search_results(["q"]) == []
        assert parse_search_results(["q", "notalist", [], []]) == []


class TestParseSummary:
    def _payload(self, **overrides):
        payload = {
            "type": "standard",
            "title": "RAG",
            "extract": "  Retrieval-augmented   generation combines retrieval with generation. ",
            "content_urls": {"desktop": {"page": "https://en.wikipedia.org/wiki/RAG"}},
        }
        payload.update(overrides)
        return payload

    def test_standard_article_parses(self):
        result = parse_summary(self._payload())
        assert result["title"] == "RAG"
        assert result["extract"].startswith("Retrieval-augmented generation")
        assert "  " not in result["extract"]
        assert result["url"].endswith("/RAG")

    def test_disambiguation_is_skipped(self):
        assert parse_summary(self._payload(type="disambiguation")) is None

    def test_missing_extract_or_url_is_skipped(self):
        assert parse_summary(self._payload(extract="")) is None
        assert parse_summary(self._payload(content_urls={})) is None
        assert parse_summary("not a dict") is None


class TestFormatWebReply:
    def test_reply_is_labeled_as_web_not_notes(self):
        reply = format_web_reply(
            {"title": "RAG", "extract": "RAG combines retrieval with generation.", "url": "https://e/x"}
        )
        assert reply.startswith("RAG\n\n")
        assert "From Wikipedia" in reply
        assert "not your notes" in reply
        assert "https://e/x" in reply


class TestFulltextFallback:
    def test_url_uses_search_list(self):
        from web_lookup import wikipedia_fulltext_search_url

        url = wikipedia_fulltext_search_url("cloudflare workers")
        assert "list=search" in url
        assert "srsearch=cloudflare+workers" in url

    def test_parse_titles(self):
        from web_lookup import parse_fulltext_results

        payload = {"query": {"search": [{"title": "Cloudflare", "snippet": "..."}]}}
        assert parse_fulltext_results(payload) == ["Cloudflare"]
        assert parse_fulltext_results({"query": {}}) == []
        assert parse_fulltext_results("junk") == []
