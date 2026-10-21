"""Cloudflare Python Worker entry point.

This module is the ONLY one that touches the Workers runtime (js.fetch, the
workers SDK, Pyodide FFI). It wires the pure/testable modules together:

    Telegram webhook -> parse_update -> Brain -> BrainStore -> D1/Vectorize/AI

Run locally with `uv run pywrangler dev`; deploy with `uv run pywrangler
deploy`. Unit tests do not import this file (there is no Workers runtime in
CI) - they cover every module it wires up.
"""

import json

from js import Object, fetch
from pyodide.ffi import to_js
from workers import Response, WorkerEntrypoint

from brain import Brain
from retrieval import extract_title, html_to_text
from store import BrainStore
from web_lookup import (
    parse_extract,
    parse_fulltext_results,
    parse_search_results,
    parse_summary,
    search_query,
    wikipedia_extract_url,
    wikipedia_fulltext_search_url,
    wikipedia_search_url,
    wikipedia_summary_url,
)
from telegram_api import (
    api_url,
    build_send_payload,
    file_url,
    parse_update,
    split_message,
    verify_secret,
)

USER_AGENT = "second-brain-bot/0.1 (+https://github.com/Prajit-22/second-brain-bot)"


def _js(value):
    """Deep-convert a Python dict/list into real JS objects for the FFI."""
    return to_js(value, dict_converter=Object.fromEntries)


class Default(WorkerEntrypoint):
    async def fetch(self, request):
        path = request.url.split("?")[0].rstrip("/")
        if request.method == "GET":
            # Health check / "it deployed" page.
            return Response("second-brain-bot is running. Talk to it on Telegram.")
        if request.method != "POST" or not path.endswith("/webhook"):
            return Response("not found", status=404)

        # Telegram signs every webhook call with the secret we registered in
        # setWebhook; without it anyone could POST fake updates.
        provided = request.headers.get("X-Telegram-Bot-Api-Secret-Token")
        if not verify_secret(provided, self.env.TELEGRAM_WEBHOOK_SECRET):
            return Response("forbidden", status=403)

        try:
            body = await request.json()
            if hasattr(body, "to_py"):
                body = body.to_py()
        except Exception:
            # Not valid JSON; tell Telegram it was accepted so it does not
            # keep retrying a payload that will never parse.
            return Response("ok")
        update = parse_update(body) if isinstance(body, dict) else None
        if update is None:
            return Response("ok")

        # Ack immediately (Telegram retries slow webhooks) and finish the AI
        # work in the background; replies go out via sendMessage.
        self.ctx.waitUntil(self._guarded(update))
        return Response("ok")

    async def _guarded(self, update):
        """waitUntil failures are otherwise silent - log the error and tell
        the user their message did not land instead of leaving them hanging."""
        try:
            await self._handle(update)
        except Exception as exc:
            print("handle_update failed: %r" % (exc,))
            try:
                await self._send(
                    update.chat_id,
                    "Something went wrong on my end while handling that - please try again.",
                )
            except Exception:
                pass

    async def _handle(self, update):
        allowed = {
            int(part.strip())
            for part in (self.env.ALLOWED_CHAT_IDS or "").split(",")
            if part.strip().lstrip("-").isdigit()
        }
        brain = Brain(
            store=BrainStore(
                self.env.DB,
                self.env.VECTORIZE,
                self.env.AI,
                llm_model=self.env.LLM_MODEL,
                embed_model=self.env.EMBED_MODEL,
                whisper_model=self.env.WHISPER_MODEL,
            ),
            send=self._send,
            allowed_chat_ids=allowed or None,
            fetch_url=self._fetch_url,
            get_file=self._get_file,
            web_lookup_fn=self._web_lookup,
            web_article_fn=self._web_article,
        )
        await brain.handle_update(update)

    # ---- outbound network (Telegram + link fetching) ---------------------

    async def _post_telegram(self, method, payload):
        resp = await fetch(
            api_url(self.env.TELEGRAM_BOT_TOKEN, method),
            _js(
                {
                    "method": "POST",
                    "headers": {"content-type": "application/json"},
                    "body": json.dumps(payload),
                }
            ),
        )
        return (await resp.json()).to_py()

    async def _send(self, chat_id, text):
        for part in split_message(text):
            await self._post_telegram("sendMessage", build_send_payload(chat_id, part))

    async def _get_file(self, file_id):
        info = await self._post_telegram("getFile", {"file_id": file_id})
        if not info.get("ok"):
            return None
        path = info["result"]["file_path"]
        resp = await fetch(file_url(self.env.TELEGRAM_BOT_TOKEN, path))
        buffer = await resp.arrayBuffer()
        return bytes(buffer.to_py())

    async def _wiki_get_json(self, url):
        resp = await fetch(
            url,
            _js({"headers": {"user-agent": USER_AGENT}, "redirect": "follow"}),
        )
        if resp.status >= 400:
            return None
        return (await resp.json()).to_py()

    async def _wiki_titles(self, query):
        """Candidate article titles: opensearch (matches titles) first,
        full-text search as fallback so multi-word queries still find one."""
        payload = await self._wiki_get_json(wikipedia_search_url(query))
        titles = [title for title, _url in parse_search_results(payload)]
        if not titles:
            payload = await self._wiki_get_json(wikipedia_fulltext_search_url(query))
            titles = parse_fulltext_results(payload)
        return titles

    async def _web_lookup(self, query):
        """Keyless web lookup for /web: Wikipedia search -> first real
        article's REST summary. Returns None when nothing usable comes back."""
        try:
            titles = await self._wiki_titles(query)
            for title in titles:
                summary = parse_summary(await self._wiki_get_json(wikipedia_summary_url(title)))
                if summary:
                    return summary
        except Exception:
            return None
        return None

    async def _web_article(self, question):
        """Web grounding for /ask: search with the question stripped of its
        question words (opensearch matches titles, so "who is the prime
        minister of india?" must become "the prime minister of india"), then
        fetch the full article lead - leads of office-holder articles name
        the incumbent, which the short REST summary can omit."""
        cleaned = search_query(question)
        queries = [cleaned] if cleaned == question.strip().rstrip("?").strip() else [cleaned, question]
        for query in queries:
            try:
                titles = await self._wiki_titles(query)
            except Exception:
                continue
            for title in titles:
                try:
                    article = parse_extract(await self._wiki_get_json(wikipedia_extract_url(title)))
                except Exception:
                    continue
                if article:
                    return article
        return None

    async def _fetch_url(self, url):
        try:
            resp = await fetch(url, _js({"headers": {"user-agent": USER_AGENT}, "redirect": "follow"}))
            if resp.status >= 400:
                return None
            html_text = await resp.text()
        except Exception:
            return None
        return {"title": extract_title(html_text, fallback=url), "text": html_to_text(html_text)}
