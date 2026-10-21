"""Prompt construction for the Workers AI chat model. Pure functions so the
exact wording the model sees is unit tested and reviewable."""

from typing import Dict, List, Sequence

# Keep the LLM input comfortably inside small-model context windows.
MAX_SUMMARY_INPUT_CHARS = 12000
MAX_ANSWER_INPUT_CHARS = 6000

SUMMARIZE_SYSTEM = (
    "You summarize notes saved in a personal second-brain app. "
    "Reply with ONLY the summary: 1-2 short sentences, under 35 words, "
    "plain and specific - what it is and why it might be useful later. "
    "No preamble, no labels, no quotes, no 'this note'. Never invent "
    "facts that are not in the note."
)

ANSWER_SYSTEM = (
    "You are the user's second brain. Answer ONLY from the numbered notes "
    "below, and cite the note numbers you used like [1] or [2]. If the notes "
    "do not contain the answer, say you have nothing saved about it and "
    "suggest what to save. Keep the answer under 200 words."
)

DIGEST_SYSTEM = (
    "You summarize the recent activity of a personal second-brain app. "
    "Group the notes into 2-4 themes and give one line per theme, then a "
    "one-line overall take. Be concrete; use the note titles."
)


def summarize_messages(text: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": SUMMARIZE_SYSTEM},
        {"role": "user", "content": text[:MAX_SUMMARY_INPUT_CHARS]},
    ]


def answer_messages(question: str, contexts: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    """contexts: dicts with title/url/text, already ranked best-first."""
    blocks = []
    budget = MAX_ANSWER_INPUT_CHARS
    for i, ctx in enumerate(contexts, start=1):
        label = ctx.get("title") or ctx.get("url") or "note"
        block = "[{i}] {label}\n{text}".format(i=i, label=label, text=ctx.get("text", ""))
        if len(block) > budget:
            block = block[:budget]
        budget -= len(block)
        blocks.append(block)
        if budget <= 0:
            break
    notes = "\n\n".join(blocks) if blocks else "(nothing saved yet)"
    user = "Notes:\n{notes}\n\nQuestion: {question}".format(notes=notes, question=question)
    return [
        {"role": "system", "content": ANSWER_SYSTEM},
        {"role": "user", "content": user},
    ]


def digest_messages(items: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    lines = [
        "- {title}: {summary}".format(title=it.get("title", "note"), summary=it.get("summary", ""))
        for it in items
    ]
    return [
        {"role": "system", "content": DIGEST_SYSTEM},
        {"role": "user", "content": "Recent saved notes:\n" + "\n".join(lines)},
    ]
