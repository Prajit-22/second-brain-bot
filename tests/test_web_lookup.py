from web_lookup import (
    format_web_answer_reply,
    format_web_reply,
    parse_extract,
    parse_search_results,
    parse_summary,
    search_query,
    wikipedia_extract_url,
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


class TestSearchQuery:
    def test_strips_question_words_and_mark(self):
        assert search_query("who is the prime minister of india?") == "prime minister of india"

    def test_strips_other_question_forms(self):
        assert search_query("what is the capital of France") == "capital of France"
        assert search_query("how old is the Eiffel Tower?") == "Eiffel Tower"
        assert search_query("kya launch kal hai?") == "launch kal hai"

    def test_strips_leading_article_for_title_ranking(self):
        # "the prime minister of india" ranks the List article above the
        # office article on opensearch; the bare phrase ranks correctly.
        assert search_query("who is the president of the united states?") == "president of the united states"

    def test_non_question_is_unchanged(self):
        assert search_query("retrieval augmented generation") == "retrieval augmented generation"

    def test_empty_after_strip_falls_back_to_original(self):
        assert search_query("kya?") == "kya"


class TestExtract:
    def test_url_uses_extracts_intro(self):
        url = wikipedia_extract_url("Prime Minister of India")
        assert "prop=extracts" in url
        assert "exintro=1" in url
        assert "explaintext=1" in url
        assert "redirects=1" in url
        assert "titles=Prime+Minister+of+India" in url

    def _payload(self, **page):
        page.setdefault("title", "Prime Minister of India")
        page.setdefault("extract", "The prime minister of India is the head of government.")
        page.setdefault("fullurl", "https://en.wikipedia.org/wiki/Prime_Minister_of_India")
        return {"query": {"pages": {"12345": page}}}

    def test_parses_first_real_page(self):
        result = parse_extract(self._payload())
        assert result["title"] == "Prime Minister of India"
        assert result["extract"].startswith("The prime minister of India")
        assert result["url"].endswith("/Prime_Minister_of_India")

    def test_missing_page_is_skipped(self):
        assert parse_extract({"query": {"pages": {"-1": {"missing": ""}}}}) is None
        assert parse_extract(self._payload(extract="")) is None
        assert parse_extract("not a dict") is None

    def test_long_extract_is_capped(self):
        result = parse_extract(self._payload(extract="x " * 5000))
        assert len(result["extract"]) <= 4000


class TestFormatWebAnswerReply:
    def test_reply_is_labeled_as_web_not_notes(self):
        reply = format_web_answer_reply(
            "Narendra Modi is the current prime minister of India.",
            {"title": "Prime Minister of India", "url": "https://en.wikipedia.org/wiki/Prime_Minister_of_India"},
        )
        assert reply.startswith("Narendra Modi")
        assert "From Wikipedia" in reply
        assert "not your notes" in reply
        assert reply.endswith("https://en.wikipedia.org/wiki/Prime_Minister_of_India")
