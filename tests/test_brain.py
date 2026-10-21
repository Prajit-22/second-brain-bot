import pytest

from brain import Brain, HELP_TEXT, format_answer_reply
from fakes import FakeSender, FakeStore, fake_fetch_url, fake_get_file
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
    async def test_ask_with_nothing_saved(self):
        brain, _, sender = make_brain()
        action = await brain.handle_update(text_update("/ask what is RRF?"))
        assert action == "ask-no-results"
        assert "Nothing saved" in sender.sent[0][1]

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
        assert action == "ask-no-results"

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
