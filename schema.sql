-- Second-brain bot schema (D1 / SQLite). Apply once after `wrangler d1 create`:
--   npx wrangler d1 execute second-brain-db --remote --file=schema.sql

-- One row per saved thing (a note, a link, a voice-note transcript).
CREATE TABLE IF NOT EXISTS items (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id      INTEGER NOT NULL,
    kind         TEXT    NOT NULL,              -- 'text' | 'link' | 'voice'
    title        TEXT    NOT NULL,
    url          TEXT    NOT NULL DEFAULT '',
    content      TEXT    NOT NULL,              -- full text (page body / transcript)
    summary      TEXT    NOT NULL DEFAULT '',   -- 2-3 sentence Workers AI summary
    content_hash TEXT    NOT NULL UNIQUE,       -- sha256; dedupes repeat saves
    created_at   TEXT    NOT NULL
);

-- Content is chunked for retrieval; each chunk is embedded into Vectorize
-- under the same generated id.
CREATE TABLE IF NOT EXISTS chunks (
    id      TEXT    PRIMARY KEY,                -- uuid hex, also the Vectorize id
    item_id INTEGER NOT NULL REFERENCES items(id) ON DELETE CASCADE,
    idx     INTEGER NOT NULL,                   -- position inside the item
    text    TEXT    NOT NULL
);

-- Keyword side of the hybrid search: SQLite FTS5 gives us BM25 for free.
-- chunk_id rides along UNINDEXED so MATCH results map straight back to chunks.
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    chunk_id UNINDEXED,
    text
);

CREATE INDEX IF NOT EXISTS idx_chunks_item ON chunks(item_id);
CREATE INDEX IF NOT EXISTS idx_items_chat  ON items(chat_id, id);

-- Short-term chat memory: the last few /ask (and /brainstorm) exchanges per
-- chat, so follow-up questions have context. Pruned to the newest 20 per
-- chat on every insert (see BrainStore.add_turn).
CREATE TABLE IF NOT EXISTS turns (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id    INTEGER NOT NULL,
    role       TEXT    NOT NULL,              -- 'user' | 'assistant'
    text       TEXT    NOT NULL,
    created_at TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_turns_chat ON turns(chat_id, id);
