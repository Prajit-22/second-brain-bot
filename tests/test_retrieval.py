import pytest

from retrieval import (
    chunk_text,
    content_hash,
    extract_title,
    filter_relevant,
    fts_query,
    html_to_text,
    keyword_relevance,
    looks_like_question,
    rrf_merge,
    tokenize,
)


class TestTokenize:
    def test_lowercases_and_splits(self):
        assert tokenize("Hello, World!") == ["hello", "world"]

    def test_unicode_hindi_and_accents_survive(self):
        # The original ragkit bug was an ASCII-only tokenizer dropping Hindi.
        assert tokenize("बैंगलोर में café") == ["बैंगलोर", "में", "café"]

    def test_empty(self):
        assert tokenize("") == []


class TestContentHash:
    def test_stable_and_whitespace_insensitive(self):
        assert content_hash("hello   world") == content_hash("hello world")

    def test_different_text_different_hash(self):
        assert content_hash("a") != content_hash("b")


class TestChunkText:
    def test_empty_and_whitespace(self):
        assert chunk_text("") == []
        assert chunk_text("   \n  ") == []

    def test_short_text_single_chunk(self):
        assert chunk_text("short note") == ["short note"]

    def test_long_text_multiple_chunks_within_limit(self):
        text = " ".join("word%d" % i for i in range(400))  # ~2400 chars
        chunks = chunk_text(text, max_chars=900, overlap=150)
        assert len(chunks) >= 3
        assert all(len(c) <= 900 for c in chunks)

    def test_overlap_preserves_boundary_content(self):
        text = " ".join("w%03d" % i for i in range(400))
        chunks = chunk_text(text, max_chars=900, overlap=150)
        # Consecutive chunks must share at least one whole word.
        for first, second in zip(chunks, chunks[1:]):
            first_words = set(first.split())
            assert any(w in first_words for w in second.split())

    def test_full_coverage_no_duplicate_tail(self):
        text = " ".join("w%03d" % i for i in range(400))
        chunks = chunk_text(text, max_chars=900, overlap=150)
        assert chunks[0].startswith("w000")
        assert chunks[-1].endswith("w399")
        assert len(set(chunks)) == len(chunks)  # no duplicated last chunk

    def test_single_giant_word_terminates(self):
        chunks = chunk_text("x" * 5000, max_chars=900, overlap=150)
        assert chunks
        assert all(len(c) <= 900 for c in chunks)

    def test_rejects_absurd_overlap(self):
        with pytest.raises(ValueError):
            chunk_text("text", max_chars=100, overlap=60)


class TestFtsQuery:
    def test_builds_quoted_or_query(self):
        assert fts_query("what did I save about RAG pipelines?") == (
            '"what" OR "did" OR "save" OR "about" OR "rag" OR "pipelines"'
        )

    def test_injection_syntax_cannot_escape_quotes(self):
        # The tokenizer keeps only word characters, so FTS operators and
        # quotes in the input can never break out of the quoted terms.
        query = fts_query('weird " OR text ) NEAR')
        assert query == '"weird" OR "or" OR "text" OR "near"' 

    def test_dedupes_and_drops_single_chars(self):
        assert fts_query("a a a bb bb") == '"bb"'

    def test_none_when_no_terms(self):
        assert fts_query("! ? ...") is None

    def test_caps_terms(self):
        assert fts_query(" ".join("term%d" % i for i in range(30))).count("OR") == 11


class TestRrfMerge:
    def test_intersection_wins(self):
        keyword = ["a", "b", "c"]
        vector = ["b", "a", "d"]
        merged = rrf_merge([keyword, vector], limit=3)
        ids = [doc_id for doc_id, _ in merged]
        # 'a' and 'b' appear in both lists and must outrank c/d.
        assert set(ids[:2]) == {"a", "b"}

    def test_single_list_preserves_order(self):
        merged = rrf_merge([["x", "y", "z"]], limit=3)
        assert [i for i, _ in merged] == ["x", "y", "z"]

    def test_scores_descend_and_limit_applies(self):
        merged = rrf_merge([["a", "b", "c", "d"]], limit=2)
        assert [i for i, _ in merged] == ["a", "b"]
        assert merged[0][1] > merged[1][1]

    def test_empty(self):
        assert rrf_merge([[], []]) == []


class TestHtml:
    def test_extract_title(self):
        assert extract_title("<html><head><title> My Page </title></head></html>") == "My Page"

    def test_extract_title_fallback(self):
        assert extract_title("<p>no title</p>", fallback="fb") == "fb"

    def test_html_to_text_strips_scripts_styles_and_tags(self):
        page = "<style>.a{color:red}</style><script>bad()</script><h1>Hi</h1><p>Some&nbsp;text</p>"
        text = html_to_text(page)
        assert "bad()" not in text and "color" not in text
        assert "Hi" in text and "Some text" in text

    def test_html_to_text_caps_length(self):
        assert len(html_to_text("x" * 50000, max_chars=100)) == 100


class TestLooksLikeQuestion:
    @pytest.mark.parametrize("text", [
        "who is the prime minister of india?",
        "what is RAG",                       # question word, no mark
        "How do I deploy this",
        "kya kal launch hai",                # Hinglish question word
        "remember to ask Rahul tomorrow?",   # ends with ?
    ])
    def test_questions_detected(self, text):
        assert looks_like_question(text)

    @pytest.mark.parametrize("text", [
        "Meeting with Rahul: launch is Friday",
        "prime minister speech notes from today",
        "idea: add query expansion",
        "",
        "   ",
    ])
    def test_statements_are_not_questions(self, text):
        assert not looks_like_question(text)


class TestKeywordRelevance:
    def test_full_overlap(self):
        assert keyword_relevance("launch pricing page", "the launch pricing page draft") == 1.0

    def test_no_overlap(self):
        assert keyword_relevance("launch pricing", "prime minister of india") == 0.0

    def test_plural_tolerance_both_directions(self):
        assert keyword_relevance("ranking", "vector rankings merged") == 1.0
        assert keyword_relevance("ideas", "one idea saved") == 1.0

    def test_stopwords_and_short_words_do_not_count(self):
        # Only 'capital' and 'france' are content terms here.
        assert keyword_relevance("what is the capital of france?", "capital of france notes") == 1.0

    def test_query_without_content_terms_scores_zero(self):
        assert keyword_relevance("what is the?", "anything at all") == 0.0


class TestFilterRelevant:
    CTX = [
        {"id": "a", "title": "Launch", "text": "launch is friday"},
        {"id": "b", "title": "PM question", "text": "who is the prime minister of india?"},
    ]

    def test_keyword_gate_keeps_relevant_drops_stray(self):
        kept = filter_relevant("when is the launch?", self.CTX, {})
        assert [c["id"] for c in kept] == ["a"]

    def test_vector_gate_keeps_despite_no_keyword_overlap(self):
        kept = filter_relevant("launch plans", self.CTX, {"b": 0.9})
        assert {c["id"] for c in kept} == {"a", "b"}

    def test_below_both_thresholds_drops(self):
        kept = filter_relevant("launch plans", self.CTX, {"b": 0.4})
        assert [c["id"] for c in kept] == ["a"]
