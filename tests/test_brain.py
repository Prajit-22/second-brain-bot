import pytest

from brain import Brain, HELP_TEXT, format_answer_reply
from fakes import (
    FakeSender,
    FakeStore,
    fake_fetch_url,
    fake_get_file,
    fake_web_article,
    fake_web_lookup,
)
from telegram_api import Update

CHAT = 42


def make_brain(store=None, sender=None, **kwargs):
    store = store or FakeStore()
    sender = sender or FakeSender()
    kwargs.setdefault("fetch_url", fake_fetch_url)
    kwargs.setdefault("get_file", fake_get_file)
    return Brain(store, sender, **kwargs), store, sender


def text_update(text, chat_id=CHAT):
    return Update(update_id=1, chat_id=chat_id, kind="text", text=text)


class TestAuth:
    @pytest.mark.asyncio
    async def test_unauthorized_chat_is_ignored_silently(self):
        brain, store, sender = make_brain(allowed_chat_ids={CHAT})
        action = await brain.handle_update(text_update("/help", chat_id=999))
        assert action == "ignored-unauthorized"
        assert sender.sent == []

    @pytest.mark.asyncio
    async def test_authorized_chat_passes(self):
        brain, store, sender = make_brain(allowed_chat_ids={CHAT})
        assert await brain.handle_update(text_update("/help")) == "help"
        assert len(sender.sent) == 1


class TestCommands:
    @pytest.mark.asyncio
    async def test_help(self):
        brain, _, sender = make_brain()
        await brain.handle_update(text_update("/help"))
        assert sender.sent[0][1] == HELP_TEXT

    @pytest.mark.asyncio
    async def test_unknown_command(self):
        brain, _, sender = make_brain()
        action = await brain.handle_update(text_update("/dance"))
        assert action == "unknown-command"
        assert "/help" in sender.sent[0][1]

    @pytest.mark.asyncio
    async def test_ask_without_question_shows_usage(self):
        brain, _, sender = make_brain()
        assert await brain.handle_update(text_update("/ask")) == "ask-empty"
        assert "Usage" in sender.sent[0][1]


class TestSaving:
    @pytest.mark.asyncio
    async def test_save_plain_text(self):
        brain, store, sender = make_brain()
        action = await brain.handle_update(text_update("RAG = chunk, embed, retrieve, generate"))
        assert action == "saved"
        assert len(store.items) == 1
        assert store.items[0]["kind"] == "text"
        assert store.chunks, "chunks should be indexed"
        reply = sender.sent[0][1]
        assert "Saved #1:" in reply and "Summary:" in reply
        assert "/ask" in reply

    @pytest.mark.asyncio
    async def test_duplicate_save_is_detected(self):
        brain, store, sender = make_brain()
        await brain.handle_update(text_update("same note"))
        action = await brain.handle_update(text_update("same   note"))
        assert action == "save-duplicate"
        assert len(store.items) == 1
        assert "Already saved" in sender.sent[1][1]

    @pytest.mark.asyncio
    async def test_link_save_fetches_page_and_keeps_user_note(self):
        brain, store, sender = make_brain()
        action = await brain.handle_update(
            text_update("must read https://developers.cloudflare.com/workers/languages/python/")
        )
        assert action == "saved"
        item = store.items[0]
        assert item["kind"] == "link"
        assert item["title"] == "Cloudflare Python Workers docs"
        assert "must read" in item["content"]  # user's own words are kept

    @pytest.mark.asyncio
    async def test_voice_note_is_transcribed_and_saved(self):
        brain, store, sender = make_brain()
        update = Update(update_id=1, chat_id=CHAT, kind="voice", file_id="fid", caption="idea")
        action = await brain.handle_update(update)
        assert action == "saved"
        item = store.items[0]
        assert item["kind"] == "voice"
        assert "ragkit hybrid retrieval" in item["content"]
        assert item["content"].startswith("idea")

    @pytest.mark.asyncio
    async def test_unsupported_attachment_gets_polite_reply(self):
        brain, _, sender = make_brain()
        update = Update(update_id=1, chat_id=CHAT, kind="unsupported")
        assert await brain.handle_update(update) == "unsupported"
        assert "not supported" in sender.sent[0][1]


class TestAsk:
    @pytest.mark.asyncio
    async def test_ask_with_nothing_saved_falls_back_to_general_answer(self):
        brain, _, sender = make_brain()
        action = await brain.handle_update(text_update("/ask what is RRF?"))
        assert action == "answered-general"
        reply = sender.sent[0][1]
        assert reply.startswith("Not from your notes")
        assert "general knowledge" in reply

    @pytest.mark.asyncio
    async def test_ask_falls_back_when_notes_do_not_cover_question(self):
        brain, store, sender = make_brain()
        await brain.handle_update(text_update("RRF merges keyword and vector rankings by rank"))
        action = await brain.handle_update(text_update("/ask what is the capital of France?"))
        assert action == "answered-general"
        reply = sender.sent[-1][1]
        assert reply.startswith("Not from your notes")
        assert "Paris" in reply
        # The relevance threshold filters the unrelated note before the
        # grounded attempt even runs, so only the general-answer call happens
        # (no NO_MATCH round-trip through the model).
        assert len(store.chat_calls) == 1

    @pytest.mark.asyncio
    async def test_ask_records_conversation_turns(self):
        brain, store, _ = make_brain()
        await brain.handle_update(text_update("RRF merges keyword and vector rankings by rank"))
        await brain.handle_update(text_update("/ask how does ranking work?"))
        roles = [t["role"] for t in store.turns]
        assert roles == ["user", "assistant"]
        assert store.turns[0]["text"] == "how does ranking work?"

    @pytest.mark.asyncio
    async def test_followup_question_sees_history_in_prompt_and_retrieval(self):
        brain, store, _ = make_brain()
        await brain.handle_update(text_update("RRF merges keyword and vector rankings by rank"))
        await brain.handle_update(text_update("/ask how does ranking work?"))
        await brain.handle_update(text_update("/ask and what about the merge constant?"))
        # The last answer prompt carries the earlier exchange as history.
        user_prompt = store.chat_calls[-1][1]["content"]
        assert "Recent conversation:" in user_prompt
        assert "how does ranking work?" in user_prompt
        # Retrieval also saw the earlier question, so the follow-up still
        # found the RRF note (cited answer, not the general fallback).
        assert "ranking" in store.last_embed_text
        assert "Sources:" in _.sent[-1][1]

    @pytest.mark.asyncio
    async def test_ask_retrieves_ranks_and_cites(self):
        brain, store, sender = make_brain()
        await brain.handle_update(text_update("RRF merges keyword and vector rankings by rank"))
        await brain.handle_update(text_update("Deploy with pywrangler, needs python_workers flag"))
        action = await brain.handle_update(text_update("/ask how does ranking work?"))
        assert action == "answered"
        reply = sender.sent[-1][1]
        assert "Sources:" in reply
        # The chat model received numbered context built from saved chunks.
        assert "[1]" in store.chat_calls[-1][1]["content"]

    @pytest.mark.asyncio
    async def test_hybrid_uses_both_keyword_and_vector_lists(self):
        brain, store, _ = make_brain()
        await brain.handle_update(text_update("python workers need the python_workers flag"))
        await brain.handle_update(text_update("/ask deployment flags"))
        # FakeStore ranked both ways; rrf_merge combined them without error
        # and an answer went out.
        assert any("Sources:" in text for _, text in _.sent)


class TestFormatAnswerReply:
    def test_duplicate_item_chunks_group_their_numbers(self):
        # Two chunks from the same item must not produce a Sources list that
        # skips the number the model cited.
        contexts = [
            {"title": "RAG notes", "text": "chunk one"},
            {"title": "RAG notes", "text": "chunk two"},
            {"title": "Deploy notes", "text": "chunk three"},
        ]
        reply = format_answer_reply("the answer [2]", contexts)
        assert "[1, 2] RAG notes" in reply
        assert "[3] Deploy notes" in reply

    def test_distinct_sources_keep_plain_numbering(self):
        contexts = [
            {"title": "A", "text": "x"},
            {"url": "https://a.dev/y", "text": "y"},
        ]
        reply = format_answer_reply("answer", contexts)
        assert "[1] A" in reply and "[2] https://a.dev/y" in reply

    def test_no_contexts_returns_answer_untouched(self):
        assert format_answer_reply("answer", []) == "answer"


class TestForget:
    @pytest.mark.asyncio
    async def test_forget_removes_item_and_its_chunks(self):
        brain, store, sender = make_brain()
        await brain.handle_update(text_update("temporary note about nothing"))
        assert store.items and store.chunks
        action = await brain.handle_update(text_update("/forget 1"))
        assert action == "forgot"
        assert store.items == [] and store.chunks == {}
        assert "Forgot #1" in sender.sent[-1][1]

    @pytest.mark.asyncio
    async def test_forgotten_item_no_longer_answerable(self):
        brain, store, sender = make_brain()
        await brain.handle_update(text_update("secret launch codes are blue"))
        await brain.handle_update(text_update("/forget 1"))
        action = await brain.handle_update(text_update("/ask what are the launch codes?"))
        # Falls back to general knowledge - crucially without leaking the
        # deleted note's content into the answer.
        assert action == "answered-general"
        assert "blue" not in sender.sent[-1][1]

    @pytest.mark.asyncio
    async def test_forget_unknown_id(self):
        brain, store, sender = make_brain()
        action = await brain.handle_update(text_update("/forget 99"))
        assert action == "forget-missing"
        assert "No item #99" in sender.sent[-1][1]

    @pytest.mark.asyncio
    async def test_forget_without_id_shows_usage(self):
        brain, store, sender = make_brain()
        action = await brain.handle_update(text_update("/forget"))
        assert action == "forget-invalid"
        assert "Usage" in sender.sent[-1][1]

    @pytest.mark.asyncio
    async def test_help_mentions_forget(self):
        assert "/forget" in HELP_TEXT


class TestBrowse:
    @pytest.mark.asyncio
    async def test_recent_empty(self):
        brain, _, sender = make_brain()
        await brain.handle_update(text_update("/recent"))
        assert "Nothing saved" in sender.sent[0][1]

    @pytest.mark.asyncio
    async def test_recent_lists_newest_first(self):
        brain, store, sender = make_brain()
        await brain.handle_update(text_update("first note"))
        await brain.handle_update(text_update("second note"))
        await brain.handle_update(text_update("/recent"))
        reply = sender.sent[-1][1]
        assert reply.index("second note") < reply.index("first note")

    @pytest.mark.asyncio
    async def test_stats_counts(self):
        brain, store, sender = make_brain()
        await brain.handle_update(text_update("one"))
        await brain.handle_update(text_update("two"))
        await brain.handle_update(text_update("/stats"))
        assert "2 item(s)" in sender.sent[-1][1]

    @pytest.mark.asyncio
    async def test_digest(self):
        brain, store, sender = make_brain()
        await brain.handle_update(text_update("cloudflare notes"))
        await brain.handle_update(text_update("/digest"))
        reply = sender.sent[-1][1]
        assert "Digest" in reply and "Themes" in reply


class TestBrainstorm:
    @pytest.mark.asyncio
    async def test_brainstorm_builds_on_notes_with_sources(self):
        brain, store, sender = make_brain()
        await brain.handle_update(text_update("RAG notes: hybrid retrieval = BM25 + vectors"))
        action = await brain.handle_update(text_update("/brainstorm improve my retrieval"))
        assert action == "brainstormed"
        reply = sender.sent[-1][1]
        assert "query expansion" in reply
        assert "Sources:" in reply

    @pytest.mark.asyncio
    async def test_brainstorm_without_notes_still_works(self):
        brain, _, sender = make_brain()
        action = await brain.handle_update(text_update("/brainstorm blog post topics"))
        assert action == "brainstormed"
        assert "1." in sender.sent[-1][1]

    @pytest.mark.asyncio
    async def test_brainstorm_without_topic_shows_usage(self):
        brain, _, sender = make_brain()
        assert await brain.handle_update(text_update("/brainstorm")) == "brainstorm-empty"
        assert "Usage" in sender.sent[0][1]


class TestSummarizeSaved:
    @pytest.mark.asyncio
    async def test_summarize_gives_fuller_summary(self):
        brain, _, sender = make_brain()
        await brain.handle_update(text_update("RAG notes: hybrid retrieval = BM25 + vectors"))
        action = await brain.handle_update(text_update("/summarize 1"))
        assert action == "summarized"
        reply = sender.sent[-1][1]
        assert reply.startswith("Summary of #1")
        assert "BM25" in reply

    @pytest.mark.asyncio
    async def test_summarize_unknown_id(self):
        brain, _, sender = make_brain()
        action = await brain.handle_update(text_update("/summarize 99"))
        assert action == "summarize-missing"
        assert "No item #99" in sender.sent[-1][1]

    @pytest.mark.asyncio
    async def test_summarize_without_id_shows_usage(self):
        brain, _, sender = make_brain()
        action = await brain.handle_update(text_update("/summarize"))
        assert action == "summarize-invalid"
        assert "Usage" in sender.sent[-1][1]


class TestWeb:
    @pytest.mark.asyncio
    async def test_web_lookup_is_labeled_as_web(self):
        brain, _, sender = make_brain(web_lookup_fn=fake_web_lookup)
        action = await brain.handle_update(text_update("/web retrieval augmented generation"))
        assert action == "web-answered"
        reply = sender.sent[-1][1]
        assert "Retrieval-augmented generation" in reply
        assert "From Wikipedia" in reply
        assert "not your notes" in reply

    @pytest.mark.asyncio
    async def test_web_lookup_no_result(self):
        brain, _, sender = make_brain(web_lookup_fn=fake_web_lookup)
        action = await brain.handle_update(text_update("/web zzzz nothing"))
        assert action == "web-no-results"
        assert "No web result" in sender.sent[-1][1]

    @pytest.mark.asyncio
    async def test_web_unavailable_without_lookup_fn(self):
        brain, _, sender = make_brain()
        action = await brain.handle_update(text_update("/web anything"))
        assert action == "web-unavailable"

    @pytest.mark.asyncio
    async def test_web_without_query_shows_usage(self):
        brain, _, sender = make_brain(web_lookup_fn=fake_web_lookup)
        assert await brain.handle_update(text_update("/web")) == "web-empty"
        assert "Usage" in sender.sent[-1][1]


class TestAskWebGrounding:
    """When the notes miss, /ask tries Wikipedia before general knowledge."""

    @pytest.mark.asyncio
    async def test_ask_uses_wikipedia_when_notes_miss(self):
        brain, _, sender = make_brain(web_article_fn=fake_web_article)
        action = await brain.handle_update(text_update("/ask who is the prime minister of india?"))
        assert action == "answered-web"
        reply = sender.sent[-1][1]
        assert "Narendra Modi" in reply
        assert "From Wikipedia" in reply
        assert "not your notes" in reply
        assert "https://en.wikipedia.org/wiki/Prime_Minister_of_India" in reply

    @pytest.mark.asyncio
    async def test_plain_question_also_uses_wikipedia(self):
        brain, store, sender = make_brain(web_article_fn=fake_web_article)
        action = await brain.handle_update(text_update("who is the prime minister of india?"))
        assert action == "answered-web"
        assert store.items == [], "a question must not become a saved note"

    @pytest.mark.asyncio
    async def test_web_miss_falls_back_to_general(self):
        brain, _, sender = make_brain(web_article_fn=fake_web_article)
        action = await brain.handle_update(text_update("/ask zzzz unknown thing?"))
        assert action == "answered-general"
        assert sender.sent[-1][1].startswith("Not from your notes")

    @pytest.mark.asyncio
    async def test_irrelevant_article_falls_back_to_general(self):
        # Wikipedia returned the PM article, but the question is about
        # something else: the model must emit NO_MATCH and the bot falls
        # back to the labeled general answer rather than citing a bad page.
        brain, _, sender = make_brain(web_article_fn=fake_web_article)
        action = await brain.handle_update(text_update("/ask what is the capital of France?"))
        assert action == "answered-general"
        assert sender.sent[-1][1].startswith("Not from your notes")

    @pytest.mark.asyncio
    async def test_without_web_fn_general_fallback_is_kept(self):
        brain, _, sender = make_brain()
        action = await brain.handle_update(text_update("/ask what is the capital of France?"))
        assert action == "answered-general"
        assert "Paris" in sender.sent[-1][1]

    @pytest.mark.asyncio
    async def test_web_answer_records_conversation_turns(self):
        brain, store, _ = make_brain(web_article_fn=fake_web_article)
        await brain.handle_update(text_update("/ask who is the prime minister of india?"))
        assert [t["role"] for t in store.turns] == ["user", "assistant"]
        assert "Modi" in store.turns[-1]["text"]

    @pytest.mark.asyncio
    async def test_notes_still_win_over_wikipedia(self):
        brain, _, sender = make_brain(web_article_fn=fake_web_article)
        await brain.handle_update(text_update("RRF merges keyword and vector rankings by rank"))
        action = await brain.handle_update(text_update("/ask how does ranking work?"))
        assert action == "answered"
        assert "Sources:" in sender.sent[-1][1]


class TestQuestionRouting:
    @pytest.mark.asyncio
    async def test_plain_question_is_answered_not_saved(self):
        brain, store, sender = make_brain()
        action = await brain.handle_update(text_update("who is the prime minister of india?"))
        assert action == "answered-general"
        assert store.items == [], "a question must not become a saved note"
        assert sender.sent[-1][1].startswith("Not from your notes")

    @pytest.mark.asyncio
    async def test_question_word_without_question_mark(self):
        brain, store, _ = make_brain()
        action = await brain.handle_update(text_update("what is the capital of France"))
        assert action == "answered-general"
        assert store.items == []

    @pytest.mark.asyncio
    async def test_hinglish_question_word_is_answered_not_saved(self):
        brain, store, _ = make_brain()
        action = await brain.handle_update(text_update("kya kal launch hai"))
        assert action == "answered-general"
        assert store.items == []

    @pytest.mark.asyncio
    async def test_routed_question_records_conversation_turns(self):
        brain, store, _ = make_brain()
        await brain.handle_update(text_update("who is the prime minister of india?"))
        assert [t["role"] for t in store.turns] == ["user", "assistant"]

    @pytest.mark.asyncio
    async def test_forwarded_question_still_saves(self):
        brain, store, _ = make_brain()
        update = Update(
            update_id=1, chat_id=CHAT, kind="text",
            text="who is the prime minister of india?", is_forward=True,
        )
        action = await brain.handle_update(update)
        assert action == "saved"
        assert len(store.items) == 1

    @pytest.mark.asyncio
    async def test_question_with_link_still_saves(self):
        brain, store, _ = make_brain()
        action = await brain.handle_update(
            text_update("what does this mean? https://developers.cloudflare.com/workers/languages/python/")
        )
        assert action == "saved"
        assert store.items[0]["kind"] == "link"

    @pytest.mark.asyncio
    async def test_statement_about_a_question_still_saves(self):
        brain, store, _ = make_brain()
        action = await brain.handle_update(text_update("prime minister speech notes from today"))
        assert action == "saved"
        assert len(store.items) == 1


class TestRelevanceThreshold:
    @staticmethod
    async def _save_stray_question_note(brain):
        """Simulates the pre-fix bug: a plain question got saved as a note.
        (A forwarded question still saves, which is exactly the documented
        escape hatch, so it doubles as the test fixture.)"""
        update = Update(
            update_id=2, chat_id=CHAT, kind="text",
            text="Who is the prime minister of india?", is_forward=True,
        )
        assert await brain.handle_update(update) == "saved"

    @pytest.mark.asyncio
    async def test_stray_note_not_cited_in_ask(self):
        brain, store, sender = make_brain()
        await brain.handle_update(text_update(
            "Meeting with Rahul: launch is Friday, we still need the pricing page and a demo video."
        ))
        await self._save_stray_question_note(brain)
        action = await brain.handle_update(
            text_update("/ask what did Rahul and I decide about the launch?")
        )
        assert action == "answered"
        reply = sender.sent[-1][1]
        assert "Meeting with Rahul" in reply
        assert "prime minister" not in reply.lower()

    @pytest.mark.asyncio
    async def test_stray_note_not_cited_in_brainstorm(self):
        brain, store, sender = make_brain()
        await brain.handle_update(text_update(
            "Idea: build a Cloudflare Worker that rotates between free LLM APIs so apps never hit a 429."
        ))
        await self._save_stray_question_note(brain)
        action = await brain.handle_update(
            text_update("/brainstorm launch ideas for the Worker project")
        )
        assert action == "brainstormed"
        reply = sender.sent[-1][1]
        assert "Sources:" in reply
        assert "prime minister" not in reply.lower()

    @pytest.mark.asyncio
    async def test_only_stray_notes_means_brainstorm_without_sources(self):
        brain, store, sender = make_brain()
        await self._save_stray_question_note(brain)
        action = await brain.handle_update(
            text_update("/brainstorm launch ideas for the Worker project")
        )
        assert action == "brainstormed"
        assert "Sources:" not in sender.sent[-1][1]


class TestSearch:
    @pytest.mark.asyncio
    async def test_search_without_words_shows_usage(self):
        brain, _, sender = make_brain()
        action = await brain.handle_update(text_update("/search"))
        assert action == "search-empty"
        assert "Usage: /search" in sender.sent[-1][1]

    @pytest.mark.asyncio
    async def test_search_with_only_tiny_words_shows_usage(self):
        brain, _, sender = make_brain()
        action = await brain.handle_update(text_update("/search a i"))
        assert action == "search-empty"

    @pytest.mark.asyncio
    async def test_search_lists_matching_items_only(self):
        brain, store, sender = make_brain()
        await brain.handle_update(text_update("RAG retrieval notes: hybrid BM25 plus vector search"))
        await brain.handle_update(text_update("Grocery list: milk, eggs, coffee beans"))
        action = await brain.handle_update(text_update("/search retrieval"))
        assert action == "search-results"
        reply = sender.sent[-1][1]
        assert "#1" in reply and "RAG retrieval notes" in reply
        assert "#2" not in reply

    @pytest.mark.asyncio
    async def test_search_matches_any_word(self):
        # OR semantics: a multi-word query finds an item matching one term.
        brain, store, sender = make_brain()
        await brain.handle_update(text_update("Grocery list: milk, eggs, coffee beans"))
        action = await brain.handle_update(text_update("/search coffee rag"))
        assert action == "search-results"
        assert "#1" in sender.sent[-1][1]

    @pytest.mark.asyncio
    async def test_search_without_matches_suggests_next_steps(self):
        brain, _, sender = make_brain()
        await brain.handle_update(text_update("Grocery list: milk, eggs, coffee beans"))
        action = await brain.handle_update(text_update("/search zanzibar"))
        assert action == "search-no-results"
        assert "/recent" in sender.sent[-1][1]

    @pytest.mark.asyncio
    async def test_find_alias_behaves_like_search(self):
        brain, store, sender = make_brain()
        await brain.handle_update(text_update("RAG retrieval notes: hybrid BM25 plus vector search"))
        action = await brain.handle_update(text_update("/find retrieval"))
        assert action == "search-results"
        assert "#1" in sender.sent[-1][1]

    @pytest.mark.asyncio
    async def test_help_mentions_search_command(self):
        assert "/search" in HELP_TEXT
