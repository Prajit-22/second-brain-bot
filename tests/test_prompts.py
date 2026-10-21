from prompts import (
    MAX_SUMMARY_INPUT_CHARS,
    answer_messages,
    digest_messages,
    summarize_messages,
)


class TestSummarize:
    def test_structure_and_content(self):
        messages = summarize_messages("my long note")
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "my long note"

    def test_input_is_capped(self):
        messages = summarize_messages("x" * (MAX_SUMMARY_INPUT_CHARS + 500))
        assert len(messages[1]["content"]) == MAX_SUMMARY_INPUT_CHARS


class TestAnswer:
    def test_contexts_are_numbered_and_question_included(self):
        contexts = [
            {"title": "RAG notes", "text": "chunk, embed, retrieve"},
            {"url": "https://a.dev/x", "text": "deploy to cloudflare"},
        ]
        messages = answer_messages("how does retrieval work?", contexts)
        user = messages[1]["content"]
        assert "[1] RAG notes" in user
        assert "[2] https://a.dev/x" in user
        assert "how does retrieval work?" in user
        assert "cite" in messages[0]["content"].lower()

    def test_no_contexts_still_builds(self):
        messages = answer_messages("anything?", [])
        assert "(nothing saved yet)" in messages[1]["content"]

    def test_budget_limits_total_context(self):
        contexts = [{"title": "t", "text": "x" * 10000} for _ in range(5)]
        user = answer_messages("q", contexts)[1]["content"]
        assert len(user) < 10000


class TestDigest:
    def test_includes_titles_and_summaries(self):
        items = [
            {"title": "Note A", "summary": "about A"},
            {"title": "Note B", "summary": "about B"},
        ]
        user = digest_messages(items)[1]["content"]
        assert "Note A" in user and "about B" in user
