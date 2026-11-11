# Shipping this as a Telegram bot on Cloudflare Workers

The local store in `scripts/brain.py` is the portable core. The same retrieval
design runs in production as a Telegram bot on Cloudflare's free tier - this
is the architecture, for when the user wants phone access to their brain.

Full source and deploy guide: https://github.com/Prajit-22/second-brain-bot
(MIT). The repo README has copy-paste Windows/PowerShell deploy steps.

## Architecture

- **Cloudflare Python Worker** (`src/entry.py`): Telegram webhook endpoint.
  Acknowledges instantly, processes in the background (`ctx.waitUntil`),
  replies via `sendMessage`.
- **D1 (SQLite)**: items, chunks, and an FTS5 table for the keyword side -
  the same schema shape as the local store here.
- **Vectorize**: 768-dim `@cf/baai/bge-base-en-v1.5` embeddings per chunk,
  the vector side of the hybrid search.
- **Workers AI**: Llama 3.1 8B (summaries, answers, digests, brainstorms),
  bge (embeddings), Whisper (voice-note transcription). Model ids are
  wrangler `[vars]`, swappable without a code change.
- **Retrieval** (`src/retrieval.py`): the same pure helpers this skill
  vendors - hybrid BM25 + vector, RRF merge, relevance gate.
- **Short-term chat memory** (D1 `turns` table): the last few Q&A exchanges
  join both the retrieval query and the answer prompt, so follow-ups like
  "and the second one?" resolve. Pruned to 20 turns per chat on every write.

## Features beyond the local core

- Voice notes transcribed with Whisper and saved with their caption.
- Links fetched and the page body saved, so retrieval searches content.
- `/brainstorm <topic>` - ideas that build on related saved notes, cited.
- `/summarize`, `/get`, `/rename`, `/edit`, `/search`, `/recent`, `/digest`,
  `/stats`, `/forget`.
- Web fallback when notes do not cover a question: Wikipedia opensearch +
  article lead, grounded and labeled with the source URL; general-knowledge
  answer under a "not from your notes" label only when the web has nothing.
- Access control: `ALLOWED_CHAT_IDS` in wrangler.toml so only the owner's
  Telegram id can use the bot.

## Deploy in one paragraph

Create the bot with @BotFather. `npx wrangler login`, `npx wrangler d1
create second-brain-db`, apply `schema.sql`, `npx wrangler vectorize create
second-brain-chunks --dimensions=768 --metric=cosine`, `wrangler secret put`
the bot token and an invented webhook secret, set `ALLOWED_CHAT_IDS`, then
`uv run pywrangler deploy` and point Telegram's `setWebhook` at the worker
URL. Free tier covers personal use comfortably.
