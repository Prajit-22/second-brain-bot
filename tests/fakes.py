"""In-memory fakes that mirror the BrainStore surface. The demo
(demo/demo_local.py) uses the same fakes, so the demo is a real end-to-end
run of the bot's logic, just without Cloudflare."""

import hashlib
from typing import Dict, List, Optional

import prompts
from retrieval import tokenize

DIMENSIONS = 64


def fake_embed_one(text: str) -> List[float]:
    """Deterministic bag-of-words vector: hashing each token into a bucket.
    Semantically similar texts really do land near each other, which keeps
    the demo's vector search honest."""
    vector = [0.0] * DIMENSIONS
    for token in tokenize(text):
        bucket = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % DIMENSIONS
        vector[bucket] += 1.0
    norm = sum(v * v for v in vector) ** 0.5 or 1.0
    return [v / norm for v in vector]


def _cosine(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


class FakeStore:
    def __init__(self):
        self.items: List[Dict] = []
        self.chunks: Dict[str, Dict] = {}
        self.turns: List[Dict] = []
        self.chat_calls: List[List[Dict[str, str]]] = []
        self._next_id = 1

    @staticmethod
    def _question_covered(notes_text: str, question: str) -> bool:
        """Stand-in for the model's relevance judgment: the notes 'cover'
        the question when any content word of the question appears in them."""
        notes_lower = notes_text.lower()
        for tok in tokenize(question):
            if len(tok) >= 4 and (tok in notes_lower or tok.rstrip("s") in notes_lower):
                return True
        return False

    # --- "Workers AI" ---
    async def embed(self, texts):
        self.last_embed_text = texts[0] if texts else ""
        return [fake_embed_one(t) for t in texts]

    async def chat(self, messages):
        self.chat_calls.append(messages)
        system = messages[0]["content"]
        user = messages[1]["content"]
        if system == prompts.ANSWER_SYSTEM:
            notes = user.split("Question:")[0]
            question = user.rsplit("Question:", 1)[-1]
            if "(nothing saved yet)" in notes or not self._question_covered(notes, question):
                return prompts.NO_MATCH_TOKEN
            return "Based on your saved notes, retrieval combines keyword and vector search [1]."
        if system == prompts.GENERAL_ANSWER_SYSTEM:
            return "Paris is the capital of France. (general knowledge, no notes used)"
        if system == prompts.BRAINSTORM_SYSTEM:
            return (
                "1. Add query expansion to retrieval [1]\n"
                "2. Log questions that miss every note\n"
                "3. Weekly digest of gaps\n"
                "4. Try HyDE-style hypothetical chunks\n"
                "5. Compare against a pure vector baseline"
            )
        if system == prompts.DETAIL_SUMMARY_SYSTEM:
            return "Overview: the note explains hybrid retrieval.\n- BM25 for keywords\n- vectors for meaning\n- RRF merges both"
        if system == prompts.DIGEST_SYSTEM:
            return "Themes: infra notes and reading list. Overall: mostly Cloudflare lately."
        return "Summary: " + user[:80]

    async def summarize(self, text):
        first = " ".join(text.split())[:120]
        return "Summary: " + first

    async def transcribe(self, audio: bytes) -> str:
        return "remember to review the ragkit hybrid retrieval notes before the interview"

    # --- "D1 + Vectorize" ---
    async def find_item_by_hash(self, digest) -> Optional[Dict]:
        for item in self.items:
            if item["content_hash"] == digest:
                return item
        return None

    async def create_item(self, chat_id, kind, title, url, content, summary, digest) -> int:
        item = {
            "id": self._next_id,
            "chat_id": chat_id,
            "kind": kind,
            "title": title,
            "url": url,
            "content": content,
            "summary": summary,
            "content_hash": digest,
            "created_at": "2026-10-21 07:00:00",
        }
        self._next_id += 1
        self.items.append(item)
        return item["id"]

    async def add_chunks(self, item_id, chunks, vectors) -> List[str]:
        ids = []
        for idx, (text, vector) in enumerate(zip(chunks, vectors)):
            cid = "chunk-%d-%d" % (item_id, idx)
            self.chunks[cid] = {"id": cid, "item_id": item_id, "text": text, "vector": vector}
            ids.append(cid)
        return ids

    async def fts_search(self, match_query, limit=25) -> List[str]:
        # Emulate FTS5: rank by shared-token count with the OR-ed terms.
        terms = [t.strip('"').replace('""', '"') for t in match_query.split(" OR ")]
        scored = []
        for cid, chunk in self.chunks.items():
            tokens = set(tokenize(chunk["text"]))
            overlap = sum(1 for t in terms if t in tokens)
            if overlap:
                scored.append((cid, overlap))
        scored.sort(key=lambda kv: (-kv[1], kv[0]))
        return [cid for cid, _ in scored[:limit]]

    async def vector_search(self, vector, limit=25):
        scored = sorted(
            self.chunks.values(),
            key=lambda c: _cosine(vector, c["vector"]),
            reverse=True,
        )
        # Real Vectorize returns the topK nearest neighbours unconditionally,
        # each with its similarity score.
        return [(c["id"], _cosine(vector, c["vector"])) for c in scored[:limit]]

    async def get_chunks(self, chunk_ids) -> List[Dict[str, str]]:
        out = []
        for cid in chunk_ids:
            chunk = self.chunks[cid]
            item = self.items[chunk["item_id"] - 1]
            out.append({"id": cid, "text": chunk["text"], "title": item["title"], "url": item["url"]})
        return out

    async def delete_item(self, item_id):
        for i, item in enumerate(self.items):
            if item["id"] == item_id:
                doomed = [cid for cid, c in self.chunks.items() if c["item_id"] == item_id]
                for cid in doomed:
                    del self.chunks[cid]
                del self.items[i]
                return {"id": item["id"], "title": item["title"], "chunks_removed": len(doomed)}
        return None

    async def get_item(self, item_id):
        for item in self.items:
            if item["id"] == item_id:
                return item
        return None

    async def add_turn(self, chat_id, role, text, keep=20):
        self.turns.append({"chat_id": chat_id, "role": role, "text": text})
        del self.turns[:-keep]

    async def recent_turns(self, chat_id, limit=6):
        return [
            {"role": t["role"], "text": t["text"]}
            for t in self.turns if t["chat_id"] == chat_id
        ][-limit:]

    async def recent_items(self, limit=10):
        return list(reversed(self.items))[:limit]

    async def item_summaries(self, limit=10):
        return [{"title": i["title"], "summary": i["summary"]} for i in reversed(self.items[:limit])]

    async def stats(self):
        return {"items": len(self.items), "chunks": len(self.chunks)}


class FakeSender:
    def __init__(self):
        self.sent: List = []

    async def __call__(self, chat_id, text):
        self.sent.append((chat_id, text))


async def fake_fetch_url(url):
    return {
        "title": "Cloudflare Python Workers docs",
        "text": "Python Workers run FastAPI-style Python on Cloudflare. "
        "You must add the python_workers compatibility flag. "
        "Bindings expose D1, KV, Vectorize and Workers AI to Python code.",
    }


async def fake_get_file(file_id):
    return b"fake-ogg-audio-bytes"


async def fake_web_lookup(query):
    """Stands in for the Wikipedia round-trip in entry.py."""
    if "zzzz" in query:
        return None
    return {
        "title": "Retrieval-augmented generation",
        "extract": "Retrieval-augmented generation (RAG) combines a retriever "
        "over a document store with a generative model.",
        "url": "https://en.wikipedia.org/wiki/Retrieval-augmented_generation",
    }
