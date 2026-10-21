from prompts import (
    MAX_SUMMARY_INPUT_CHARS,
    NO_MATCH_TOKEN,
    answer_messages,
    brainstorm_messages,
    detail_summary_messages,
    digest_messages,
    general_answer_messages,
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


class TestAnswerFallbackContract:
    def test_answer_system_defines_no_match_token(self):
        messages = answer_messages("q", [{"title": "t", "text": "x"}])
        assert NO_MATCH_TOKEN in messages[0]["content"]

    def test_history_is_included_when_present(self):
        history = [
            {"role": "user", "text": "how does ranking work?"},
            {"role": "assistant", "text": "via RRF [1]"},
        ]
        user = answer_messages("and the constant?", [{"title": "t", "text": "x"}], history)[1]["content"]
        assert "Recent conversation:" in user
        assert "you: how does ranking work?" in user
        assert "bot: via RRF [1]" in user

    def test_history_absent_by_default(self):
        user = answer_messages("q", [{"title": "t", "text": "x"}])[1]["content"]
        assert "Recent conversation:" not in user


class TestGeneralAnswer:
    def test_structure_and_labeling(self):
        messages = general_answer_messages("what is RRF?")
        assert messages[0]["role"] == "system"
        assert "general knowledge" in messages[0]["content"]
        assert "Never pretend" in messages[0]["content"]
        assert messages[1]["content"].endswith("Question: what is RRF?")


class TestBrainstorm:
    def test_topic_and_notes_included(self):
        contexts = [{"title": "RAG notes", "text": "hybrid retrieval"}]
        messages = brainstorm_messages("better retrieval", contexts)
        user = messages[1]["content"]
        assert "Topic: better retrieval" in user
        assert "[1] RAG notes" in user
        assert "numbered" in messages[0]["content"]

    def test_no_notes_still_builds(self):
        user = brainstorm_messages("anything", [])[1]["content"]
        assert "(nothing saved yet)" in user


class TestDetailSummary:
    def test_title_and_content_included(self):
        messages = detail_summary_messages("Deploy notes", "run pywrangler deploy")
        user = messages[1]["content"]
        assert "Title: Deploy notes" in user
        assert "run pywrangler deploy" in user
        assert "bullet" in messages[0]["content"]
