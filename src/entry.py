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

        body = (await request.json()).to_py()
        update = parse_update(body)
        if update is None:
            return Response("ok")

        # Ack immediately (Telegram retries slow webhooks) and finish the AI
        # work in the background; replies go out via sendMessage.
        self.ctx.waitUntil(self._handle(update))
        return Response("ok")

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

    async def _fetch_url(self, url):
        try:
            resp = await fetch(url, _js({"headers": {"user-agent": USER_AGENT}, "redirect": "follow"}))
            if resp.status >= 400:
                return None
            html_text = await resp.text()
        except Exception:
            return None
        return {"title": extract_title(html_text, fallback=url), "text": html_to_text(html_text)}
