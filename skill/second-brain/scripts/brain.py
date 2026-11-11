#!/usr/bin/env python3
"""A local second brain: save notes/links, then retrieve the chunks that
answer a question. One SQLite file, Python standard library only.

The agent using this skill does the reading and writing; this script is the
memory. Typical loop:

    python3 brain.py save --title "Release checklist" --text "..."
    python3 brain.py retrieve "how do we cut a release?"
    -> prints the surviving chunks as numbered context; the agent answers
       from them with [1]/[2] citations, or says the notes do not cover it.

Optional vector side: set BRAIN_EMBED_URL (OpenAI-compatible /embeddings
endpoint), BRAIN_EMBED_KEY and BRAIN_EMBED_MODEL and retrieval goes hybrid
(BM25 + vectors merged with Reciprocal Rank Fusion). Unset, it is keyword
BM25 only - still gated, still honest.
"""

import argparse
import datetime
import json
import os
import sqlite3
import sys
import urllib.request
import uuid

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import retrieval

SCHEMA = """
CREATE TABLE IF NOT EXISTS items (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    kind         TEXT    NOT NULL,              -- 'text' | 'link'
    title        TEXT    NOT NULL,
    url          TEXT    NOT NULL DEFAULT '',
    content      TEXT    NOT NULL,
    summary      TEXT    NOT NULL DEFAULT '',
    content_hash TEXT    NOT NULL UNIQUE,
    created_at   TEXT    NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
    id      TEXT PRIMARY KEY,
    item_id INTEGER NOT NULL REFERENCES items(id) ON DELETE CASCADE,
    idx     INTEGER NOT NULL,
    text    TEXT    NOT NULL,
    vector  TEXT                             -- JSON list of floats, when embedded
);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(chunk_id UNINDEXED, text);
CREATE INDEX IF NOT EXISTS idx_chunks_item ON chunks(item_id);
"""

DEFAULT_DB = os.environ.get(
    "BRAIN_DB", os.path.join(os.path.expanduser("~"), ".second-brain", "brain.db")
)
EMBED_URL = os.environ.get("BRAIN_EMBED_URL", "").strip()
EMBED_KEY = os.environ.get("BRAIN_EMBED_KEY", "").strip()
EMBED_MODEL = os.environ.get("BRAIN_EMBED_MODEL", "text-embedding-3-small").strip()

FETCH_LIMIT = 20   # candidates pulled from each retrieval side before the gate
KEEP_LIMIT = 6     # chunks handed to the agent as numbered context


def connect(db_path: str) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    return conn


# --- optional embeddings ----------------------------------------------------


def embed(texts):
    """OpenAI-compatible embeddings; None when the vector side is not set up."""
    if not EMBED_URL:
        return None
    body = json.dumps({"model": EMBED_MODEL, "input": texts}).encode("utf-8")
    req = urllib.request.Request(EMBED_URL, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    if EMBED_KEY:
        req.add_header("Authorization", "Bearer " + EMBED_KEY)
    with urllib.request.urlopen(req, timeout=30) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    return [list(map(float, row["embedding"])) for row in payload["data"]]


# --- commands ---------------------------------------------------------------


def cmd_save(conn, args) -> int:
    text = args.text if args.text is not None else sys.stdin.read()
    text = " ".join(text.split())
    if not text:
        print("nothing to save", file=sys.stderr)
        return 2
    digest = retrieval.content_hash(text)
    existing = conn.execute(
        "SELECT id, title FROM items WHERE content_hash = ?", (digest,)
    ).fetchone()
    if existing:
        print("duplicate of #%d (%s) - not saved again" % (existing[0], existing[1]))
        return 0
    kind = "link" if args.url else "text"
    title = args.title or text[:60] + ("..." if len(text) > 60 else "")
    now = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    cur = conn.execute(
        "INSERT INTO items (kind, title, url, content, summary, content_hash, created_at)"
        " VALUES (?, ?, ?, ?, ?, ?, ?)",
        (kind, title, args.url or "", text, args.summary or "", digest, now),
    )
    item_id = cur.lastrowid
    pieces = retrieval.chunk_text(text)
    vectors = embed(pieces) if EMBED_URL and pieces else None
    for idx, piece in enumerate(pieces):
        chunk_id = uuid.uuid4().hex
        vector_json = json.dumps(vectors[idx]) if vectors else None
        conn.execute(
            "INSERT INTO chunks (id, item_id, idx, text, vector) VALUES (?, ?, ?, ?, ?)",
            (chunk_id, item_id, idx, piece, vector_json),
        )
        conn.execute(
            "INSERT INTO chunks_fts (chunk_id, text) VALUES (?, ?)", (chunk_id, piece)
        )
    conn.commit()
    print("saved #%d (%s, %d chunk%s%s)"
          % (item_id, title, len(pieces), "" if len(pieces) == 1 else "s",
             ", embedded" if vectors else ""))
    return 0


def _bm25_ranking(conn, match_query):
    rows = conn.execute(
        "SELECT chunk_id FROM chunks_fts WHERE chunks_fts MATCH ?"
        " ORDER BY bm25(chunks_fts) LIMIT ?",
        (match_query, FETCH_LIMIT),
    ).fetchall()
    return [row[0] for row in rows]


def _vector_ranking(conn, query):
    if not EMBED_URL:
        return [], {}
    query_vec = embed([query])
    if not query_vec:
        return [], {}
    query_vec = query_vec[0]
    scores = {}
    for chunk_id, vector_json in conn.execute(
        "SELECT id, vector FROM chunks WHERE vector IS NOT NULL"
    ):
        scores[chunk_id] = retrieval.cosine(query_vec, json.loads(vector_json))
    ranked = sorted(scores, key=lambda cid: (-scores[cid], cid))[:FETCH_LIMIT]
    return ranked, scores


def cmd_retrieve(conn, args) -> int:
    query = " ".join(args.query)
    rankings = []
    match = retrieval.fts_query(query)
    if match:
        rankings.append(_bm25_ranking(conn, match))
    vector_scores = {}
    if EMBED_URL:
        vec_ranking, vector_scores = _vector_ranking(conn, query)
        rankings.append(vec_ranking)
    if not rankings:
        print("NO_MATCH")
        return 1
    merged = retrieval.rrf_merge(rankings, limit=FETCH_LIMIT)
    if not merged:
        print("NO_MATCH")
        return 1
    ids = [cid for cid, _ in merged]
    rows = conn.execute(
        "SELECT c.id, c.item_id, c.text, i.title FROM chunks c"
        " JOIN items i ON i.id = c.item_id WHERE c.id IN (%s)"
        % ",".join("?" * len(ids)),
        ids,
    ).fetchall()
    by_id = {row[0]: row for row in rows}
    contexts = [
        {"id": cid, "item_id": by_id[cid][1], "text": by_id[cid][2], "title": by_id[cid][3]}
        for cid in ids
        if cid in by_id
    ]
    kept = retrieval.filter_relevant(query, contexts, vector_scores)[:KEEP_LIMIT]
    if not kept:
        print("NO_MATCH")
        return 1
    for n, ctx in enumerate(kept, 1):
        print("[%d] (note #%d, %s) %s" % (n, ctx["item_id"], ctx["title"], ctx["text"]))
        print()
    return 0


def cmd_get(conn, args) -> int:
    row = conn.execute(
        "SELECT id, kind, title, url, content, summary, created_at FROM items WHERE id = ?",
        (args.item_id,),
    ).fetchone()
    if not row:
        print("no item #%d" % args.item_id, file=sys.stderr)
        return 2
    print("#%d [%s] %s" % (row[0], row[1], row[2]))
    if row[3]:
        print(row[3])
    if row[5]:
        print("summary: " + row[5])
    print("saved:   " + row[6])
    print()
    print(row[4])
    return 0


def cmd_recent(conn, args) -> int:
    rows = conn.execute(
        "SELECT id, title, kind, created_at FROM items ORDER BY id DESC LIMIT ?",
        (args.limit,),
    ).fetchall()
    if not rows:
        print("nothing saved yet")
        return 0
    for row in rows:
        print("#%d [%s] %s (%s)" % (row[0], row[2], row[1], row[3][:10]))
    return 0


def cmd_forget(conn, args) -> int:
    row = conn.execute("SELECT title FROM items WHERE id = ?", (args.item_id,)).fetchone()
    if not row:
        print("no item #%d" % args.item_id, file=sys.stderr)
        return 2
    chunk_ids = [
        r[0]
        for r in conn.execute("SELECT id FROM chunks WHERE item_id = ?", (args.item_id,))
    ]
    conn.execute("DELETE FROM chunks WHERE item_id = ?", (args.item_id,))
    for cid in chunk_ids:
        conn.execute("DELETE FROM chunks_fts WHERE chunk_id = ?", (cid,))
    conn.execute("DELETE FROM items WHERE id = ?", (args.item_id,))
    conn.commit()
    print("forgot #%d (%s) and its %d chunks" % (args.item_id, row[0], len(chunk_ids)))
    return 0


def cmd_stats(conn, args) -> int:
    items = conn.execute("SELECT COUNT(*) FROM items").fetchone()[0]
    chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    embedded = conn.execute(
        "SELECT COUNT(*) FROM chunks WHERE vector IS NOT NULL"
    ).fetchone()[0]
    print("%d items, %d chunks, %d embedded" % (items, chunks, embedded))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Local second brain: save notes, retrieve answering chunks."
    )
    parser.add_argument("--db", default=DEFAULT_DB, help="SQLite file (default: %(default)s)")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("save", help="save a note or link (dedupes by content)")
    p.add_argument("--title", help="note title (default: first 60 chars)")
    p.add_argument("--url", help="source URL, marks the item as a link")
    p.add_argument("--summary", help="one-line summary the agent wrote")
    p.add_argument("--text", help="content; omit to read it from stdin")
    p.set_defaults(func=cmd_save)

    p = sub.add_parser("retrieve", help="print numbered chunks that answer a question")
    p.add_argument("query", nargs="+")
    p.set_defaults(func=cmd_retrieve)

    p = sub.add_parser("get", help="print one saved item in full")
    p.add_argument("item_id", type=int)
    p.set_defaults(func=cmd_get)

    p = sub.add_parser("recent", help="list the newest saved items")
    p.add_argument("--limit", type=int, default=10)
    p.set_defaults(func=cmd_recent)

    p = sub.add_parser("forget", help="delete an item, its chunks and embeddings")
    p.add_argument("item_id", type=int)
    p.set_defaults(func=cmd_forget)

    p = sub.add_parser("stats", help="item/chunk counts")
    p.set_defaults(func=cmd_stats)

    args = parser.parse_args(argv)
    conn = connect(args.db)
    try:
        return args.func(conn, args)
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
