# Retrieval design: why this store does not hallucinate

Read this before tuning thresholds, porting the design, or explaining
tradeoffs to the user.

## The problem with naive RAG

The classic failure of "vector search + LLM" notes apps: every query returns
*something*, because both retrieval backends always produce a ranking - even
when nothing in the store is relevant. The LLM then writes a fluent answer
from an irrelevant note, and the user cannot tell it is wrong. A second brain
that invents memories is worse than no second brain.

This design fixes that with three pieces that each do one job.

## 1. Two retrieval sides, not one

- **Keyword side: BM25 over SQLite FTS5.** Exact-term matching - names,
  commands, error messages, rare words. SQLite ships FTS5, so this costs zero
  dependencies. The natural-language query is turned into a safe MATCH
  expression by `retrieval.fts_query`: each token is double-quoted (FTS5's
  escape hatch, so user input can never inject query syntax) and OR-ed.
- **Vector side: embeddings + cosine similarity.** Semantic matching - the
  question "how do I roll back a deploy?" finds a note titled "migration
  runbook" that never says those words. Optional in this skill (any
  OpenAI-compatible endpoint); in the production bot it runs on Cloudflare
  Vectorize with bge-base embeddings.

Each side covers the other's blind spot. Keyword search misses paraphrases;
vector search drifts on rare exact terms.

## 2. Reciprocal Rank Fusion merges the rankings

`retrieval.rrf_merge` scores each candidate `1 / (k + rank)` per ranking
(k = 60) and sums. Chunks found by BOTH sides bubble to the top, and neither
side's raw score scale matters (BM25 scores and cosine similarities are not
comparable - ranks are). This is the standard, well-studied fusion method;
do not replace it with score normalization.

## 3. The relevance gate - the piece that prevents hallucination

RRF *ranks* candidates. It cannot say "none of these are relevant" - its top
result exists for every query. So after fusion, `retrieval.filter_relevant`
drops every chunk that neither side actually vouches for:

- the keyword side vouches when at least ~1/3 of the query's content words
  (stopwords removed, plural-tolerant) literally appear in the chunk;
- the vector side vouches when the cosine similarity clears 0.55.

If nothing survives, `retrieve` prints `NO_MATCH` and the agent answers
honestly instead of confidently wrong. When embeddings are off, the keyword
gate alone still blocks unrelated notes.

The tokenizer is Unicode-aware and keeps combining marks, so Hinglish and
Devanagari notes tokenize correctly instead of shredding into fragments.

## Supporting decisions

- **Chunking with overlap** (`chunk_text`, 900 chars / 150 overlap): a fact
  that straddles a boundary still appears whole in at least one chunk. Chunks
  are what get embedded, ranked and cited; items stay whole for reading.
- **Dedupe by content hash** (sha256 of normalized text): re-saving the same
  thing is a no-op, so forwarding something twice never pollutes retrieval.
- **Citations are note numbers the user can check** (`get <id>` prints the
  full item). Every claim in an answer traces to a saved note.
- **Honest fallback order** in the production bot: notes -> Wikipedia
  (labeled, with source link) -> general knowledge (labeled "not from your
  notes"). The label is the feature.

## What to tune, and how to test it

- `MIN_KEYWORD_RELEVANCE` (0.34): raise when answers cite loosely related
  notes; lower when obviously relevant notes get missed.
- `MIN_VECTOR_SCORE` (0.55): same tradeoff on the semantic side.
- Chunk size/overlap: only with a real reason (e.g. notes are mostly code).

Test any change the way the project's CI does: save a handful of notes, then
run one query a note covers and one it does not. The first must return the
note, the second must return `NO_MATCH`. If you cannot make both hold, revert.
