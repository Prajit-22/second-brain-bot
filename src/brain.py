"""Orchestration: what the bot *does* with each update.

The Brain talks to three injected collaborators, so it runs identically
inside the Cloudflare Worker and in local tests/demo:
  - store:      BrainStore (D1 + Vectorize + Workers AI) or a fake
  - send:       async callable(chat_id, text) that delivers a reply
  - fetch_url / get_file: async callables for link pages and voice files

All ranking/chunking logic comes from retrieval.py (pure, unit tested)."""

from typing import Awaitable, Callable, Dict, List, Optional, Sequence, Set, Tuple

import prompts
import web_lookup
from retrieval import (
    chunk_text,
    content_hash,
    filter_relevant,
    fts_query,
    looks_like_question,
    rrf_merge,
)
from telegram_api import Update, find_urls, is_authorized, parse_command

HELP_TEXT = (
    "Hey! I'm your second brain.\n\n"
    "Send me anything:\n"
    "- text or links -> I save and summarize them\n"
    "- voice notes -> I transcribe and save them\n"
    "- a question (ends with ? or starts with who/what/how...) -> I answer\n"
    "  it like /ask instead of saving it (forward a question to save it)\n"
    "- then ask questions over everything you saved\n\n"
    "Commands:\n"
    "/ask <question> - answer from your notes, with sources; if your notes\n"
    "  don't cover it, Wikipedia is checked (labeled, with link) before you\n"
    "  get a labeled general answer - so current-fact questions work too\n"
    "/brainstorm <topic> - fresh ideas that build on what you saved\n"
    "/summarize <id> - a fuller summary of one saved item\n"
    "/web <query> - quick web lookup (via Wikipedia) for things you never saved\n"
    "/recent - your last 10 saved items\n"
    "/digest - AI digest of your recent saves\n"
    "/forget <id> - delete a saved item for good\n"
    "/stats - how much you have saved\n"
    "/help - this message"
)

SendFn = Callable[[int, str], Awaitable[None]]
FetchUrlFn = Callable[[str], Awaitable[Optional[Dict[str, str]]]]
GetFileFn = Callable[[str], Awaitable[Optional[bytes]]]
WebLookupFn = Callable[[str], Awaitable[Optional[Dict[str, str]]]]


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
        web_lookup_fn: Optional[WebLookupFn] = None,
        web_article_fn: Optional[WebLookupFn] = None,
        max_context_chunks: int = 6,
    ):
        self._store = store
        self._send = send
        self._allowed = allowed_chat_ids
        self._fetch_url = fetch_url
        self._get_file = get_file
        self._web_lookup = web_lookup_fn
        self._web_article = web_article_fn
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
            # A plain message that is clearly a question is the user asking,
            # not jotting something down - answer it like /ask instead of
            # silently saving the question itself as a note. Forwards and
            # messages with links still save (they are content to keep).
            if (
                looks_like_question(update.text)
                and not update.is_forward
                and not find_urls(update.text)
            ):
                return await self._ask(update.chat_id, update.text)
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
        if name == "brainstorm":
            return await self._brainstorm(chat_id, args)
        if name == "summarize":
            return await self._summarize_saved(chat_id, args)
        if name == "web":
            return await self._web(chat_id, args)
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

    async def _retrieve(self, question: str, history: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
        """Hybrid keyword + vector retrieval over saved chunks, with a
        relevance threshold: RRF only *ranks* candidates, and both backends
        return something for every query, so a final gate (keyword coverage
        or vector similarity) is what keeps unrelated saved notes out of the
        context - and therefore out of answers, citations and brainstorms.
        The recent conversation joins the retrieval query so a follow-up
        like 'and the second one?' still pulls the chunks the chat is about."""
        history_text = " ".join(
            turn.get("text", "") for turn in history if turn.get("role") == "user"
        )
        retrieval_query = (history_text + " " + question).strip()
        vector = (await self._store.embed([retrieval_query]))[0]
        query = fts_query(retrieval_query)
        keyword_ids = await self._store.fts_search(query, limit=25) if query else []
        vector_hits = await self._store.vector_search(vector, limit=25)
        merged = rrf_merge(
            [keyword_ids, [cid for cid, _score in vector_hits]],
            limit=self._max_context_chunks,
        )
        if not merged:
            return []
        chunks = await self._store.get_chunks([cid for cid, _score in merged])
        return filter_relevant([question, retrieval_query], chunks, dict(vector_hits))

    async def _ask(self, chat_id: int, question: str) -> str:
        if not question:
            await self._send(chat_id, "Usage: /ask <question> - e.g. /ask what did I save about RAG?")
            return "ask-empty"
        history = await self._store.recent_turns(chat_id)
        contexts = await self._retrieve(question, history)
        answer = ""
        if contexts:
            answer = (await self._store.chat(prompts.answer_messages(question, contexts, history))).strip()
        if not contexts or answer.upper().startswith(prompts.NO_MATCH_TOKEN):
            # The notes do not cover this one. For factual / current-info
            # questions general knowledge hits its training cutoff ("who is
            # the current ..."), so try Wikipedia first: an answer grounded
            # in the fetched article, clearly labeled, with the source URL.
            # Only when that yields nothing useful do we fall back to the
            # labeled general answer.
            web_reply, web_answer = await self._answer_from_web(question)
            if web_reply is not None:
                answer = web_answer
                await self._send(chat_id, web_reply)
                action = "answered-web"
            else:
                answer = (await self._store.chat(prompts.general_answer_messages(question, history))).strip()
                await self._send(chat_id, "Not from your notes - general answer:\n\n" + answer)
                action = "answered-general"
        else:
            await self._send(chat_id, format_answer_reply(answer, contexts))
            action = "answered"
        await self._store.add_turn(chat_id, "user", question)
        await self._store.add_turn(chat_id, "assistant", answer)
        return action

    async def _answer_from_web(self, question: str) -> Tuple[Optional[str], str]:
        """Ground a notes-miss question in Wikipedia. Returns (reply, answer)
        or (None, "") when web lookup is unavailable, finds nothing, or the
        article does not actually answer the question - the caller then uses
        the labeled general-knowledge fallback."""
        if self._web_article is None:
            return None, ""
        try:
            article = await self._web_article(question)
        except Exception:
            return None, ""
        if not article:
            return None, ""
        answer = (await self._store.chat(prompts.web_answer_messages(question, article))).strip()
        if not answer or answer.upper().startswith(prompts.NO_MATCH_TOKEN):
            return None, ""
        return web_lookup.format_web_answer_reply(answer, article), answer

    async def _brainstorm(self, chat_id: int, topic: str) -> str:
        if not topic:
            await self._send(chat_id, "Usage: /brainstorm <topic> - e.g. /brainstorm side project ideas from my notes")
            return "brainstorm-empty"
        history = await self._store.recent_turns(chat_id)
        contexts = await self._retrieve(topic, history)
        ideas = (await self._store.chat(prompts.brainstorm_messages(topic, contexts, history))).strip()
        if contexts:
            await self._send(chat_id, format_answer_reply(ideas, contexts))
        else:
            await self._send(chat_id, ideas)
        await self._store.add_turn(chat_id, "user", "/brainstorm " + topic)
        await self._store.add_turn(chat_id, "assistant", ideas)
        return "brainstormed"

    async def _summarize_saved(self, chat_id: int, args: str) -> str:
        try:
            item_id = int(args.strip())
        except (ValueError, AttributeError):
            await self._send(chat_id, "Usage: /summarize <id> - e.g. /summarize 3 (ids come from /recent)")
            return "summarize-invalid"
        item = await self._store.get_item(item_id)
        if not item:
            await self._send(chat_id, "No item #{id} in your second brain.".format(id=item_id))
            return "summarize-missing"
        detail = (
            await self._store.chat(prompts.detail_summary_messages(item["title"], item["content"]))
        ).strip()
        await self._send(
            chat_id,
            "Summary of #{id}: {title}\n\n{detail}".format(
                id=item_id, title=item["title"], detail=detail
            ),
        )
        return "summarized"

    async def _web(self, chat_id: int, query: str) -> str:
        if not query:
            await self._send(chat_id, "Usage: /web <query> - e.g. /web retrieval augmented generation")
            return "web-empty"
        if self._web_lookup is None:
            await self._send(chat_id, "Web lookup is not available on this deployment.")
            return "web-unavailable"
        result = await self._web_lookup(query)
        if not result:
            await self._send(chat_id, "No web result found for that - try rephrasing the query.")
            return "web-no-results"
        await self._send(chat_id, web_lookup.format_web_reply(result))
        return "web-answered"

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
