from telegram_api import (
    build_send_payload,
    find_urls,
    is_authorized,
    parse_command,
    parse_update,
    split_message,
    verify_secret,
)


def make_message(text=None, **extra):
    message = {
        "message_id": 1,
        "chat": {"id": 42, "type": "private"},
        "from": {"first_name": "Prajit", "last_name": "C"},
    }
    if text is not None:
        message["text"] = text
    message.update(extra)
    return {"update_id": 100, "message": message}


class TestParseUpdate:
    def test_text_message(self):
        update = parse_update(make_message("hello there"))
        assert update.kind == "text"
        assert update.chat_id == 42
        assert update.text == "hello there"
        assert update.sender_name == "Prajit C"
        assert not update.is_forward

    def test_forwarded_message_flagged(self):
        update = parse_update(make_message("read this", forward_origin={"type": "hidden_user"}))
        assert update.is_forward

    def test_voice_message(self):
        update = parse_update(make_message(voice={"file_id": "abc", "duration": 3}))
        assert update.kind == "voice"
        assert update.file_id == "abc"

    def test_audio_also_counts_as_voice(self):
        update = parse_update(make_message(audio={"file_id": "xyz"}, caption="my note"))
        assert update.kind == "voice"
        assert update.caption == "my note"

    def test_edited_message_accepted(self):
        payload = make_message("fixed text")
        payload["edited_message"] = payload.pop("message")
        assert parse_update(payload).text == "fixed text"

    def test_photo_without_text_is_unsupported(self):
        update = parse_update(make_message(photo=[{"file_id": "p1"}]))
        assert update.kind == "unsupported"

    def test_non_message_updates_return_none(self):
        assert parse_update({"update_id": 1, "my_chat_member": {}}) is None
        assert parse_update({"update_id": 1}) is None


class TestParseCommand:
    def test_command_with_args(self):
        assert parse_command("/ask what is RRF?") == ("ask", "what is RRF?")

    def test_command_with_bot_suffix_and_multiline_args(self):
        assert parse_command("/ask@SecondBrainBot line one\nline two") == (
            "ask",
            "line one\nline two",
        )

    def test_bare_command(self):
        assert parse_command("/help") == ("help", "")

    def test_plain_text_is_not_a_command(self):
        assert parse_command("just a /note") is None
        assert parse_command("") is None


class TestFindUrls:
    def test_finds_urls_and_strips_trailing_punctuation(self):
        assert find_urls("read https://a.dev/x,(great) and http://b.dev/y.") == [
            "https://a.dev/x",
            "http://b.dev/y",
        ]

    def test_no_urls(self):
        assert find_urls("no links here") == []

    def test_balanced_parentheses_survive(self):
        # Wikipedia-style links carry meaningful parentheses.
        assert find_urls("see https://en.wikipedia.org/wiki/RAG_(retrieval) ok") == [
            "https://en.wikipedia.org/wiki/RAG_(retrieval)"
        ]

    def test_unbalanced_trailing_bracket_is_dropped(self):
        assert find_urls("(check https://x.dev/a)") == ["https://x.dev/a"]
        assert find_urls("wiki https://en.wikipedia.org/wiki/RRF_(cs)) done") == [
            "https://en.wikipedia.org/wiki/RRF_(cs)"
        ]


class TestAuth:
    def test_no_whitelist_allows_everyone(self):
        assert is_authorized(999, None)

    def test_whitelist_filters(self):
        assert is_authorized(42, {42})
        assert not is_authorized(7, {42})


class TestVerifySecret:
    def test_match(self):
        assert verify_secret("s3cret", "s3cret")

    def test_mismatch_and_missing(self):
        assert not verify_secret("nope", "s3cret")
        assert not verify_secret(None, "s3cret")
        assert not verify_secret("s3cret", None)


class TestSplitMessage:
    def test_short_message_untouched(self):
        assert split_message("hi") == ["hi"]

    def test_splits_on_newlines_when_possible(self):
        text = "\n".join("line %d" % i for i in range(2000))
        parts = split_message(text)
        assert all(len(p) <= 4096 for p in parts)
        assert "\n".join(parts) == text

    def test_hard_cut_without_newlines(self):
        parts = split_message("x" * 9000)
        assert all(len(p) <= 4096 for p in parts)
        assert "".join(parts) == "x" * 9000


class TestPayload:
    def test_build_send_payload(self):
        payload = build_send_payload(42, "hello")
        assert payload["chat_id"] == 42
        assert payload["text"] == "hello"
