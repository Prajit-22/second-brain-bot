#!/usr/bin/env python3
"""End-to-end demo of the bot's logic with in-memory fakes - no Cloudflare
account, no Telegram token, no network.

It drives the REAL Brain + retrieval + prompts code path that the Worker
runs, with FakeStore standing in for D1/Vectorize/Workers AI:

    python demo/demo_local.py           # print the simulated chat
    python demo/demo_local.py --check   # exit non-zero if anything breaks
    python demo/demo_local.py --json    # machine-readable transcript

Run from the repo root:  PYTHONPATH=src:tests python demo/demo_local.py
"""

import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))

from brain import Brain  # noqa: E402
from fakes import FakeSender, FakeStore, fake_fetch_url, fake_get_file  # noqa: E402
from telegram_api import Update  # noqa: E402

CHAT_ID = 42

SCRIPT = [
    ("user", "/start"),
    ("user", "RAG notes: hybrid retrieval = BM25 + vector search merged with RRF (k=60)"),
    ("user", "must read https://developers.cloudflare.com/workers/languages/python/"),
    ("user", "voice", "idea: add a /digest command"),
    ("user", "RAG notes: hybrid retrieval = BM25 + vector search merged with RRF (k=60)"),
    ("user", "scratch note: try the new samosa place"),
    ("user", "/forget 4"),
    ("user", "/ask how does hybrid retrieval work?"),
    ("user", "/recent"),
    ("user", "/digest"),
    ("user", "/stats"),
]


async def run():
    store, sender = FakeStore(), FakeSender()
    brain = Brain(store, sender, fetch_url=fake_fetch_url, get_file=fake_get_file)
    transcript = []
    for step in SCRIPT:
        kind = step[1]
        if kind == "voice":
            update = Update(update_id=1, chat_id=CHAT_ID, kind="voice", file_id="demo", caption=step[2])
            shown = "[voice note] " + step[2]
        else:
            update = Update(update_id=1, chat_id=CHAT_ID, kind="text", text=kind)
            shown = kind
        before = len(sender.sent)
        action = await brain.handle_update(update)
        transcript.append({"from": "you", "text": shown})
        for _, text in sender.sent[before:]:
            transcript.append({"from": "brain", "text": text})
        transcript[-1]["action"] = action
    return transcript


def render(transcript):
    lines = []
    for msg in transcript:
        prefix = "YOU  > " if msg["from"] == "you" else "BOT  > "
        body = msg["text"].replace("\n", "\n       ")
        lines.append(prefix + body)
    return "\n".join(lines)


def check(transcript):
    texts = [m["text"] for m in transcript if m["from"] == "brain"]
    assert any("second brain" in t for t in texts), "help reply missing"
    assert any(t.startswith("Saved as #") for t in texts), "save reply missing"
    assert any("Already saved" in t for t in texts), "dedupe reply missing"
    assert any("Forgot #4" in t for t in texts), "forget reply missing"
    assert any("Sources:" in t for t in texts), "cited answer missing"
    assert any("Last 3 saved items" in t for t in texts), "recent missing"
    assert any("Digest" in t for t in texts), "digest missing"
    assert any("3 item(s)" in t for t in texts), "stats wrong (dedupe or forget broken?)"
    print("demo check: OK -", len(transcript), "messages")


def main():
    transcript = asyncio.run(run())
    if "--json" in sys.argv:
        print(json.dumps(transcript, indent=2))
    elif "--check" in sys.argv:
        check(transcript)
    else:
        print(render(transcript))


if __name__ == "__main__":
    main()
