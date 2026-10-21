"""Thin adapters over the Cloudflare bindings (D1, Vectorize, Workers AI).

Deliberately boring: every ranking/chunking/prompt decision lives in the
pure modules (retrieval.py, prompts.py) and every orchestration decision in
brain.py. This file only translates Python calls into binding calls and
converts Pyodide JsProxy results back into plain Python via ``_py``."""

import uuid
from typing import Dict, List, Optional

# Workers AI model ids. Overridable through wrangler [vars] so the deployment
# can be retuned without a code change.
DEFAULT_LLM_MODEL = "@cf/meta/llama-3.1-8b-instruct"
DEFAULT_EMBED_MODEL = "@cf/baai/bge-base-en-v1.5"  # 768-dim vectors
DEFAULT_WHISPER_MODEL = "@cf/openai/whisper"


def _py(value):
    """Convert a Pyodide JsProxy into a native Python object when needed.
    Plain Python objects (as in tests/demo) pass straight through."""
    if hasattr(value, "to_py"):
        return value.to_py()
    return value


class BrainStore:
    def __init__(self, db, vectorize, ai, llm_model=None, embed_model=None, whisper_model=None):
        self._db = db
        self._vectorize = vectorize
        self._ai = ai
        self._llm_model = llm_model or DEFAULT_LLM_MODEL
        self._embed_model = embed_model or DEFAULT_EMBED_MODEL
        self._whisper_model = whisper_model or DEFAULT_WHISPER_MODEL

    # ---- Workers AI ------------------------------------------------------

    async def embed(self, texts: List[str]) -> List[List[float]]:
        """bge-base embeddings; returns one 768-dim vector per input text."""
        resp = _py(await self._ai.run(self._embed_model, {"text": texts}))
        return [list(map(float, row)) for row in resp["data"]]

    async def chat(self, messages: List[Dict[str, str]]) -> str:
        resp = _py(await self._ai.run(self._llm_model, {"messages": messages}))
        return resp.get("response", "")

    async def summarize(self, text: str) -> str:
        import prompts  # local import keeps this module dependency-light

        return await self.chat(prompts.summarize_messages(text))

    async def transcribe(self, audio: bytes) -> str:
        """Whisper wants the audio as an array of byte values."""
        resp = _py(await self._ai.run(self._whisper_model, {"audio": list(audio)}))
        return resp.get("text", "")

    # ---- D1: items, chunks, FTS ------------------------------------------

    async def find_item_by_hash(self, digest: str) -> Optional[Dict]:
        row = _py(
            await self._db.prepare("SELECT id, title FROM items WHERE content_hash = ?")
            .bind(digest)
            .first()
        )
        return dict(row) if row else None

    async def create_item(self, chat_id, kind, title, url, content, summary, digest) -> int:
        row = _py(
            await self._db.prepare(
                "INSERT INTO items (chat_id, kind, title, url, content, summary, content_hash,"
                " created_at) VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now')) RETURNING id"
            )
            .bind(chat_id, kind, title, url, content, summary, digest)
            .first()
        )
        return int(row["id"])

    async def add_chunks(self, item_id: int, chunks: List[str], vectors: List[List[float]]) -> List[str]:
        """Store chunks in D1 (+ the FTS5 index) and their vectors in
        Vectorize, keyed by the same generated chunk id."""
        chunk_ids = [uuid.uuid4().hex for _ in chunks]
        statements = []
        for cid, idx, text in zip(chunk_ids, range(len(chunks)), chunks):
            statements.append(
                self._db.prepare("INSERT INTO chunks (id, item_id, idx, text) VALUES (?, ?, ?, ?)").bind(
                    cid, item_id, idx, text
                )
            )
            statements.append(
                self._db.prepare("INSERT INTO chunks_fts (chunk_id, text) VALUES (?, ?)").bind(cid, text)
            )
        if statements:
            await self._db.batch(statements)
        await self._vectorize.upsert(
            [
                {"id": cid, "values": vector, "metadata": {"item_id": item_id}}
                for cid, vector in zip(chunk_ids, vectors)
            ]
        )
        return chunk_ids

    async def fts_search(self, match_query: str, limit: int = 25) -> List[str]:
        """BM25 keyword search via SQLite FTS5. Lower bm25() is better; we
        only need the ordering for RRF, not the raw scores."""
        rows = _py(
            await self._db.prepare(
                "SELECT chunk_id FROM chunks_fts WHERE chunks_fts MATCH ?"
                " ORDER BY bm25(chunks_fts) LIMIT ?"
            )
            .bind(match_query, limit)
            .all()
        )
        return [row["chunk_id"] for row in rows.get("results", [])]

    async def vector_search(self, vector: List[float], limit: int = 25) -> List[str]:
        matches = _py(await self._vectorize.query(vector, {"topK": limit}))
        return [match["id"] for match in matches.get("matches", [])]

    async def get_chunks(self, chunk_ids: List[str]) -> List[Dict[str, str]]:
        """Hydrate chunk ids with text + owning item title/url, preserving the
        caller's ranking order."""
        if not chunk_ids:
            return []
        placeholders = ", ".join("?" for _ in chunk_ids)
        rows = _py(
            await self._db.prepare(
                "SELECT c.id AS id, c.text AS text, i.title AS title, i.url AS url"
                " FROM chunks c JOIN items i ON i.id = c.item_id"
                " WHERE c.id IN ({placeholders})".format(placeholders=placeholders)
            )
            .bind(*chunk_ids)
            .all()
        )
        by_id = {row["id"]: row for row in rows.get("results", [])}
        return [dict(by_id[cid]) for cid in chunk_ids if cid in by_id]

    async def recent_items(self, limit: int = 10) -> List[Dict]:
        rows = _py(
            await self._db.prepare(
                "SELECT id, kind, title, url, created_at FROM items ORDER BY id DESC LIMIT ?"
            )
            .bind(limit)
            .all()
        )
        return [dict(row) for row in rows.get("results", [])]

    async def item_summaries(self, limit: int = 10) -> List[Dict[str, str]]:
        rows = _py(
            await self._db.prepare("SELECT title, summary FROM items ORDER BY id DESC LIMIT ?")
            .bind(limit)
            .all()
        )
        return [dict(row) for row in rows.get("results", [])]

    async def stats(self) -> Dict[str, int]:
        items = _py(await self._db.prepare("SELECT COUNT(*) AS n FROM items").first())
        chunks = _py(await self._db.prepare("SELECT COUNT(*) AS n FROM chunks").first())
        return {"items": int(items["n"]), "chunks": int(chunks["n"])}
