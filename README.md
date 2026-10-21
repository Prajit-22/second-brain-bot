# second-brain-bot

A Telegram "second brain" that runs entirely on a **self-deployed Cloudflare Python Worker** - no server, no Docker, $0 on the free tier.

Forward it anything - text, links, voice notes - and it stores, transcribes and summarizes them. Then ask questions in plain language and it answers **from your saved notes, with sources**, using hybrid retrieval (BM25 + vector search merged with Reciprocal Rank Fusion).

![demo chat](demo/demo-chat.svg)

## What it does

| You send | The bot |
| --- | --- |
| plain text | saves it, writes a 2-3 sentence AI summary, indexes it |
| a link | fetches the page, extracts the text, saves page + your note |
| a voice note | transcribes it with Whisper, saves the transcript (+ your caption) |
| the same thing twice | detects the duplicate (sha256) instead of saving again |
| `/ask <question>` | hybrid search over your notes, answer with numbered citations |
| `/recent` `/digest` `/stats` | browse, get an AI digest, see counts |
| `/forget <id>` | deletes a saved item for good - item, chunks and embeddings |

## Architecture

![architecture](demo/architecture.svg)

- **Cloudflare Python Worker** (`src/entry.py`) - Telegram webhook endpoint. Acknowledges instantly, processes in the background (`ctx.waitUntil`), replies via `sendMessage`.
- **D1 (SQLite)** - items, chunks, and an **FTS5** table for the keyword side of the search (`bm25()` ranking).
- **Vectorize** - 768-dim `@cf/baai/bge-base-en-v1.5` embeddings per chunk for the semantic side.
- **Workers AI** - Llama 3.1 8B (summaries, answers, digests), bge (embeddings), Whisper (voice transcription). Models are wrangler `[vars]`, so they can be swapped without a code change.
- **Hybrid retrieval** - both rankings merged with **Reciprocal Rank Fusion** (`src/retrieval.py`), then the top chunks become numbered context for the answer prompt.

Everything is Python: one language from webhook to retrieval to prompts.

## Deploy your own (Windows)

You need: a Cloudflare account (free), a Telegram account, Node.js LTS and [uv](https://docs.astral.sh/uv/) installed. All commands below are PowerShell.

**1. Create the bot.** Message [@BotFather](https://t.me/BotFather) on Telegram -> `/newbot` -> copy the token.

**2. Clone and log in.**

```powershell
git clone https://github.com/Prajit-22/second-brain-bot.git
cd second-brain-bot
npx wrangler login
```

**3. Create the database and vector index.**

```powershell
npx wrangler d1 create second-brain-db
# copy the printed database_id into wrangler.toml (database_id = "...")
npx wrangler d1 execute second-brain-db --remote --file=schema.sql
npx wrangler vectorize create second-brain-chunks --dimensions=768 --metric=cosine
```

**4. Set secrets and your chat id.**

```powershell
npx wrangler secret put TELEGRAM_BOT_TOKEN       # the BotFather token
npx wrangler secret put TELEGRAM_WEBHOOK_SECRET  # any long random string you invent
```

Message [@userinfobot](https://t.me/userinfobot) to get your Telegram id and put it in `wrangler.toml` -> `ALLOWED_CHAT_IDS = "123456789"` so only you can use the bot.

**5. Deploy.**

```powershell
uv run pywrangler deploy
# prints https://second-brain-bot.<you>.workers.dev
```

**6. Point Telegram at your worker.**

```powershell
Invoke-RestMethod -Uri "https://api.telegram.org/bot<TOKEN>/setWebhook" -Method Post -Body @{
  url = "https://second-brain-bot.<you>.workers.dev/webhook"
  secret_token = "<same TELEGRAM_WEBHOOK_SECRET as step 4>"
}
```

Done - forward the bot a link and `/ask` it something.

One-click deploy also works via the button (you still need steps 1, 3, 4 and 6 for the bot token, database, secrets and webhook):

[![Deploy to Cloudflare](https://deploy.workers.cloudflare.com/button)](https://deploy.workers.cloudflare.com/?url=https://github.com/Prajit-22/second-brain-bot)

## Cost

Runs on the Cloudflare **free tier**: Workers requests, D1, Vectorize and Workers AI (Llama/bge/Whisper) all have daily free allocations sized far beyond personal second-brain use. No credit card required for the free plan.

## Develop and test locally

No Cloudflare account needed for the tests or the demo - the bot's logic is plain Python with the Cloudflare bindings behind one thin adapter (`src/store.py`):

```powershell
pip install -r requirements-dev.txt
python -m pytest tests/ -v          # 79 unit tests
python demo/demo_local.py           # end-to-end demo with in-memory fakes
```

`demo/demo_local.py` drives the **real** Brain + retrieval + prompts code path against in-memory fakes of D1/Vectorize/Workers AI - the same fakes the unit tests use. CI (`.github/workflows/ci.yml`) runs both on Python 3.9-3.12.

## Project layout

```
src/
  entry.py         Worker entry: webhook auth, network calls, wiring (only Cloudflare-aware file)
  brain.py         orchestration: save/ask/recent/digest/stats flows
  store.py         thin D1 + Vectorize + Workers AI adapter
  retrieval.py     chunking, tokenizer, FTS5 query building, RRF merge, HTML -> text (pure)
  prompts.py       system/user prompt builders (pure)
  telegram_api.py  update parsing, commands, auth, message splitting (pure)
tests/             79 unit tests + in-memory fakes
demo/demo_local.py end-to-end demo harness (prints the simulated chat above)
schema.sql         D1 schema (items, chunks, FTS5)
wrangler.toml      bindings + model config
```

## Design notes (the "why")

- **Why hybrid retrieval?** Pure vector search misses exact keywords ("that Ragas metric", an error string); pure BM25 misses paraphrases ("how do I deploy it" vs "shipping steps"). RRF merges both rankings by rank position only, so neither score scale matters - and a chunk found by both ranks highest.
- **Why D1 FTS5 instead of a second service?** SQLite ships `bm25()` for free, colocated with the metadata, one `schema.sql` to understand.
- **Why acknowledge the webhook instantly?** Telegram retries slow webhooks, which would double-save notes. `ctx.waitUntil` finishes the AI work after the 200, replies go out via the Bot API.
- **Why a webhook secret + chat allow-list?** The worker URL is public; `X-Telegram-Bot-Api-Secret-Token` proves the POST came from Telegram, and `ALLOWED_CHAT_IDS` makes the bot answer only you.
- **Why is everything pure except `entry.py`/`store.py`?** So 79 unit tests and the demo exercise the exact production logic with no Workers runtime - and every line stays explainable.
