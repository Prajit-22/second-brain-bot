"""Orchestration: what the bot *does* with each update.

The Brain talks to three injected collaborators, so it runs identically
inside the Cloudflare Worker and in local tests/demo:
  - store:      BrainStore (D1 + Vectorize + Workers AI) or a fake
  - send:       async callable(chat_id, text) that delivers a reply
  - fetch_url / get_file: async callables for link pages and voice files

All ranking/chunking logic comes from retrieval.py (pure, unit tested)."""

from typing import Awaitable, Callable, Dict, List, Optional, Set

import prompts
from retrieval import chunk_text, content_hash, fts_query, rrf_merge
from telegram_api import Update, find_urls, is_authorized, parse_command

HELP_TEXT = (
    "Hey! I'm your second brain.\n\n"
    "Send me anything:\n"
    "- text or links -> I save and summarize them\n"
    "- voice notes -> I transcribe and save them\n"
    "- then ask questions over everything you saved\n\n"
    "Commands:\n"
    "/ask <question> - answer from your saved notes, with sources\n"
    "/recent - your last 10 saved items\n"
    "/digest - AI digest of your recent saves\n"
    "/forget <id> - delete a saved item for good\n"
    "/stats - how much you have saved\n"
    "/help - this message"
)

SendFn = Callable[[int, str], Awaitable[None]]
FetchUrlFn = Callable[[str], Awaitable[Optional[Dict[str, str]]]]
GetFileFn = Callable[[str], Awaitable[Optional[bytes]]]


def _short_title(text: str, limit: int = 60) -> str:
    title = " ".join(text.split())
    if len(title) <= limit:
        return title if title else "note"
    cut = title[:limit].rsplit(" ", 1)[0].rstrip(" ,;:-")
    return (cut + "...") if cut else "note"


def format_saved_reply(item_id: int, title: str, summary: str, n_chunks: int) -> str:
    return "Saved #{id}: {title}\n\n{summary}\n\nAsk me anything with /ask.".format(
        id=item_id, title=title, summary=summary
    )


def format_answer_reply(answer: str, contexts: List[Dict[str, str]]) -> str:
    """Append a Sources list whose numbers match the numbered context the
    model saw. Several chunks can come from the same item; their numbers are
    grouped onto one line ("[1, 3] Title") so no citation ever points at a
    number missing from the list."""
    numbers_by_label: Dict[str, List[int]] = {}
    order: List[str] = []
    for i, ctx in enumerate(contexts, start=1):
        label = ctx.get("title") or ctx.get("url") or "note"
        if label not in numbers_by_label:
            numbers_by_label[label] = []
            order.append(label)
        numbers_by_label[label].append(i)
    if not order:
        return answer
    sources = [
        "[{nums}] {label}".format(
            nums=", ".join(str(n) for n in numbers_by_label[label]), label=label
        )
        for label in order
    ]
    return answer + "\n\nSources:\n" + "\n".join(sources)


class Brain:
    def __init__(
        self,
        store,
        send: SendFn,
        allowed_chat_ids: Optional[Set[int]] = None,
        fetch_url: Optional[FetchUrlFn] = None,
        get_file: Optional[GetFileFn] = None,
        max_context_chunks: int = 6,
    ):
        self._store = store
        self._send = send
        self._allowed = allowed_chat_ids
        self._fetch_url = fetch_url
        self._get_file = get_file
        self._max_context_chunks = max_context_chunks

    async def handle_update(self, update: Update) -> str:
        """Route one update. Returns a short action tag (handy for logs and
        tests) and never raises on user input."""
        if not is_authorized(update.chat_id, self._allowed):
            return "ignored-unauthorized"
        command = parse_command(update.text) if update.kind == "text" else None
        if command:
            return await self._handle_command(update.chat_id, command[0], command[1])
        if update.kind == "text":
            return await self._save_text(update)
        if update.kind == "voice":
            return await self._save_voice(update)
        await self._send(
            update.chat_id,
            "I can save text, links and voice notes - other attachments are not supported yet.",
        )
        return "unsupported"

    async def _handle_command(self, chat_id: int, name: str, args: str) -> str:
        if name in ("start", "help"):
            await self._send(chat_id, HELP_TEXT)
            return "help"
        if name == "ask":
            return await self._ask(chat_id, args)
        if name in ("forget", "delete"):
            return await self._forget(chat_id, args)
        if name == "recent":
            return await self._recent(chat_id)
        if name == "digest":
            return await self._digest(chat_id)
        if name == "stats":
            return await self._stats(chat_id)
        await self._send(chat_id, "Unknown command. Try /help.")
        return "unknown-command"

    # ---- saving ---------------------------------------------------------

    async def _save_text(self, update: Update) -> str:
        urls = find_urls(update.text)
        page = None
        if urls and self._fetch_url is not None:
            page = await self._fetch_url(urls[0])
        if page:
            title = page.get("title") or urls[0]
            body = page.get("text", "")
            # Keep the user's own words alongside the page - their note is
            # often the most useful thing to retrieve later.
            content = "{note}\n\n{body}".format(note=update.text, body=body).strip()
            url = urls[0]
        else:
            title = _short_title(update.text)
            content = update.text
            url = urls[0] if urls else ""
        return await self._save_content(update.chat_id, "link" if page else "text", title, url, content)

    async def _save_voice(self, update: Update) -> str:
        if self._get_file is None or not update.file_id:
            await self._send(update.chat_id, "Voice notes are not available on this deployment.")
            return "voice-unavailable"
        audio = await self._get_file(update.file_id)
        if not audio:
            await self._send(update.chat_id, "Could not download that voice note - try again?")
            return "voice-failed"
        transcript = (await self._store.transcribe(audio)).strip()
        if not transcript:
            await self._send(update.chat_id, "I could not transcribe that voice note.")
            return "voice-empty"
        content = transcript
        if update.caption:
            content = update.caption + "\n\n" + transcript
        return await self._save_content(
            update.chat_id, "voice", _short_title(transcript), "", content
        )

    async def _save_content(self, chat_id: int, kind: str, title: str, url: str, content: str) -> str:
        digest = content_hash(content)
        existing = await self._store.find_item_by_hash(digest)
        if existing:
            await self._send(
                chat_id,
                "Already saved as #{id}: {title}".format(id=existing["id"], title=existing["title"]),
            )
            return "save-duplicate"
        summary = (await self._store.summarize(content)).strip()
        chunks = chunk_text(content)
        vectors = await self._store.embed(chunks)
        item_id = await self._store.create_item(chat_id, kind, title, url, content, summary, digest)
        await self._store.add_chunks(item_id, chunks, vectors)
        await self._send(chat_id, format_saved_reply(item_id, title, summary, len(chunks)))
        return "saved"

    # ---- asking ---------------------------------------------------------

    async def _ask(self, chat_id: int, question: str) -> str:
        if not question:
            await self._send(chat_id, "Usage: /ask <question> - e.g. /ask what did I save about RAG?")
            return "ask-empty"
        vector = (await self._store.embed([question]))[0]
        query = fts_query(question)
        keyword_ids = await self._store.fts_search(query, limit=25) if query else []
        vector_ids = await self._store.vector_search(vector, limit=25)
        merged = rrf_merge([keyword_ids, vector_ids], limit=self._max_context_chunks)
        if not merged:
            await self._send(
                chat_id,
                "Nothing saved matches that yet. Forward me a few notes/links first, then ask again.",
            )
            return "ask-no-results"
        contexts = await self._store.get_chunks([cid for cid, _score in merged])
        answer = (await self._store.chat(prompts.answer_messages(question, contexts))).strip()
        await self._send(chat_id, format_answer_reply(answer, contexts))
        return "answered"

    # ---- browsing your brain -------------------------------------------

    async def _forget(self, chat_id: int, args: str) -> str:
        try:
            item_id = int(args.strip())
        except (ValueError, AttributeError):
            await self._send(chat_id, "Usage: /forget <id> - e.g. /forget 3 (ids come from /recent)")
            return "forget-invalid"
        deleted = await self._store.delete_item(item_id)
        if not deleted:
            await self._send(chat_id, "No item #{id} in your second brain.".format(id=item_id))
            return "forget-missing"
        await self._send(
            chat_id,
            "Forgot #{id}: {title} - removed {n} indexed chunk(s).".format(
                id=deleted["id"], title=deleted["title"], n=deleted["chunks_removed"]
            ),
        )
        return "forgot"

    async def _recent(self, chat_id: int, limit: int = 10) -> str:
        items = await self._store.recent_items(limit)
        if not items:
            await self._send(chat_id, "Nothing saved yet. Forward me a note or a link!")
            return "recent-empty"
        lines = [
            "#{id} [{kind}] {title} ({date})".format(
                id=it["id"], kind=it["kind"], title=it["title"], date=(it.get("created_at") or "")[:10]
            )
            for it in items
        ]
        await self._send(chat_id, "Last {n} saved items:\n".format(n=len(items)) + "\n".join(lines))
        return "recent"

    async def _digest(self, chat_id: int, limit: int = 10) -> str:
        items = await self._store.item_summaries(limit)
        if not items:
            await self._send(chat_id, "Nothing saved yet - save a few notes first.")
            return "digest-empty"
        digest = (await self._store.chat(prompts.digest_messages(items))).strip()
        await self._send(chat_id, "Digest of your last {n} saves:\n\n{digest}".format(n=len(items), digest=digest))
        return "digest"

    async def _stats(self, chat_id: int) -> str:
        stats = await self._store.stats()
        await self._send(
            chat_id,
            "Your second brain holds {items} item(s) and {chunks} indexed chunk(s).".format(
                items=stats["items"], chunks=stats["chunks"]
            ),
        )
        return "stats"
