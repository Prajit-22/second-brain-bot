---
name: second-brain
description: Builds and queries a local personal "second brain" - a private knowledge store the user saves notes and links into, then questions in plain language and gets answers grounded in their own saved material with citations. Use when the user wants to remember things long-term ("save this", "note this down"), asks questions over what they saved ("what did I note about X?"), wants personal knowledge management, or wants to give an agent persistent memory with honest, source-cited retrieval instead of hallucinated recall.
license: MIT
compatibility: Python 3.9+ with sqlite3 FTS5 (bundled in CPython). No third-party packages. Network access optional - only needed for the optional vector-embedding side or for fetching link contents.
metadata:
  author: Prajit Chaturvedi
  version: "1.0"
  homepage: https://github.com/Prajit-22/second-brain-bot
---

# Second Brain

A personal knowledge store with retrieval that does not hallucinate. The user
saves notes and links; later they ask questions and get answers built only
from what they saved, with citations - or an honest "your notes don't cover
this" instead of an invented answer.

The store is one SQLite file. `scripts/brain.py` is the whole engine:
standard library only, no pip install, no API key required. You (the agent)
do the reading, summarizing and answering; the script is the memory.

This skill is the distilled core of a production Telegram bot
(github.com/Prajit-22/second-brain-bot) that runs the same retrieval design
on Cloudflare Workers - D1, Vectorize and Workers AI - serving real saved
notes daily.

## Quick start

```bash
# save (dedupes exact repeats automatically)
python3 scripts/brain.py save --title "Release checklist" --text "..."
cat article.txt | python3 scripts/brain.py save --title "That article" --url https://...

# ask: prints the chunks that survive retrieval as numbered context
python3 scripts/brain.py retrieve "how do we cut a release?"

# browse and manage
python3 scripts/brain.py recent
python3 scripts/brain.py get 3
python3 scripts/brain.py forget 3
python3 scripts/brain.py stats
```

The database lives at `~/.second-brain/brain.db` (override with `--db` or
`BRAIN_DB`). Saving needs no network and no key.

## The workflow you follow

### Saving

1. When the user wants to keep something ("save this", "remember that",
   forwarded link, voice-note transcript), write a one-line summary yourself,
   then call `save` with `--title`, `--summary`, the content, and `--url`
   when it came from a page.
2. Confirm with the assigned note number: "Saved #4". If the script reports a
   duplicate, say so instead of pretending it saved.
3. For a link, fetch the page and save the extracted text, not the URL alone -
   the content is what retrieval searches.

### Answering

1. Run `retrieve` with the user's question.
2. If the output is `NO_MATCH`, the notes do not cover it. Say so plainly,
   then - only if the user wants - answer from your own knowledge clearly
   labeled as NOT from their notes, or offer a web lookup. Never dress up a
   general answer as something they saved.
3. Otherwise answer ONLY from the numbered chunks, citing them like [1] or
   [2]. If the chunks cover part of the question, answer that part and say
   what is missing. Under 200 words unless the user asks for depth.
4. Offer `get <id>` when the user wants the full note behind a citation.

### Why you can trust the retrieval

`retrieve` never returns "whatever ranked highest": every returned chunk
passed a relevance gate (see `references/design.md`). A query the notes do
not cover comes back as `NO_MATCH`, so you never have to guess whether the
top result is actually relevant. This is the property that makes cited
answers safe - protect it. Do not bypass the gate by answering from chunks
the command did not return.

## Optional: hybrid retrieval with embeddings

Out of the box, retrieval is keyword BM25 (SQLite FTS5) plus the gate. To add
the vector side - worth it once the store grows past a few hundred notes -
point the script at any OpenAI-compatible embeddings endpoint:

```bash
export BRAIN_EMBED_URL="https://api.openai.com/v1/embeddings"
export BRAIN_EMBED_KEY="sk-..."
export BRAIN_EMBED_MODEL="text-embedding-3-small"
```

With these set, new saves are embedded and `retrieve` merges the BM25 and
vector rankings with Reciprocal Rank Fusion before gating - the same hybrid
design the production bot runs on Cloudflare Vectorize. Unset, everything
keeps working keyword-only.

## Files

- `scripts/brain.py` - the store: save / retrieve / get / recent / forget / stats.
- `scripts/retrieval.py` - pure helpers: tokenizer, chunker, FTS5 query
  builder, relevance gate, RRF merge, cosine. No I/O; read it to understand
  or tune the ranking.
- `scripts/selftest.py` - 8-check smoke test (save, dedupe, gated retrieval,
  forget). Run it after install or after tuning thresholds.
- `references/design.md` - why the retrieval is built this way: hybrid BM25 +
  vectors, RRF, the relevance gate, honest fallback. Read before tuning
  thresholds or explaining tradeoffs to the user.
- `references/telegram-worker.md` - how this same design ships as a Telegram
  bot on Cloudflare Workers, for a user who wants chat access from their phone.

## Tuning

Defaults live in `scripts/retrieval.py`: `MIN_KEYWORD_RELEVANCE` (0.34) and
`MIN_VECTOR_SCORE` (0.55). Raise them when answers cite loosely-related notes;
lower them when real notes get missed. Test changes against a handful of saved
notes with one related and one unrelated query before trusting them.
