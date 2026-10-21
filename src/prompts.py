"""Prompt construction for the Workers AI chat model. Pure functions so the
exact wording the model sees is unit tested and reviewable."""

from typing import Dict, List, Optional, Sequence

# Keep the LLM input comfortably inside small-model context windows.
MAX_SUMMARY_INPUT_CHARS = 12000
MAX_ANSWER_INPUT_CHARS = 6000
MAX_HISTORY_TURNS = 6
MAX_HISTORY_TURN_CHARS = 500

# The answer model emits exactly this token when the saved notes do not
# cover the question; brain.py then falls back to a labeled general answer.
NO_MATCH_TOKEN = "NO_MATCH"

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
    "do not contain the answer, reply with exactly " + NO_MATCH_TOKEN + " and "
    "nothing else - a different step will answer from general knowledge. "
    "Keep the answer under 200 words."
)

GENERAL_ANSWER_SYSTEM = (
    "You are a helpful assistant inside a personal second-brain Telegram bot. "
    "The user's saved notes did not cover this question, so answer from your "
    "own general knowledge. Be direct and practical, under 150 words, plain "
    "text, and say when you are unsure. Never pretend the answer came from "
    "the user's notes."
)

WEB_ANSWER_SYSTEM = (
    "You answer questions inside a personal second-brain Telegram bot. The "
    "user's saved notes did not cover this question, so you are given a "
    "Wikipedia article extract. Answer ONLY from that extract: direct, "
    "plain text, under 80 words. If the extract does not answer the "
    "question, reply with exactly " + NO_MATCH_TOKEN + " and nothing else - "
    "a different step will answer from general knowledge. When the question "
    "is about a current office holder or very recent events, add a short "
    "note that Wikipedia is usually current but can lag breaking news."
)

BRAINSTORM_SYSTEM = (
    "You are a brainstorming partner inside a personal second-brain app. "
    "The user gives you a topic; you may also see notes they saved. "
    "Reply with 4-6 sharp, concrete ideas as a numbered list, one line "
    "each - no preamble, no pep talk, nothing after the list. Use a saved "
    "note only when it is genuinely relevant to the topic, and cite it "
    "like [1] on the idea it informs. Never mention or cite a note that "
    "is not relevant."
)

DETAIL_SUMMARY_SYSTEM = (
    "You summarize one saved note in a personal second-brain app. Reply with "
    "a 2-3 sentence overview, then 3-6 bullet points covering the key facts, "
    "decisions or steps. Stay faithful to the text; never invent details."
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


def _format_history(history: Optional[Sequence[Dict[str, str]]]) -> str:
    """Render the last few conversation turns as 'you: ...' / 'bot: ...'
    lines so follow-up questions ('and the second one?') make sense."""
    if not history:
        return ""
    lines = []
    for turn in history[-MAX_HISTORY_TURNS:]:
        speaker = "you" if turn.get("role") == "user" else "bot"
        text = " ".join((turn.get("text") or "").split())[:MAX_HISTORY_TURN_CHARS]
        if text:
            lines.append("{speaker}: {text}".format(speaker=speaker, text=text))
    if not lines:
        return ""
    return "Recent conversation:\n" + "\n".join(lines) + "\n\n"


def _numbered_notes(contexts: Sequence[Dict[str, str]]) -> str:
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
    return "\n\n".join(blocks) if blocks else "(nothing saved yet)"


def answer_messages(
    question: str,
    contexts: Sequence[Dict[str, str]],
    history: Optional[Sequence[Dict[str, str]]] = None,
) -> List[Dict[str, str]]:
    user = "{history}Notes:\n{notes}\n\nQuestion: {question}".format(
        history=_format_history(history),
        notes=_numbered_notes(contexts),
        question=question,
    )
    return [
        {"role": "system", "content": ANSWER_SYSTEM},
        {"role": "user", "content": user},
    ]


def general_answer_messages(
    question: str,
    history: Optional[Sequence[Dict[str, str]]] = None,
) -> List[Dict[str, str]]:
    user = "{history}Question: {question}".format(
        history=_format_history(history), question=question
    )
    return [
        {"role": "system", "content": GENERAL_ANSWER_SYSTEM},
        {"role": "user", "content": user},
    ]


def web_answer_messages(question: str, article: Dict[str, str]) -> List[Dict[str, str]]:
    user = "Wikipedia article: {title}\n\n{extract}\n\nQuestion: {question}".format(
        title=article.get("title", ""),
        extract=article.get("extract", ""),
        question=question,
    )
    return [
        {"role": "system", "content": WEB_ANSWER_SYSTEM},
        {"role": "user", "content": user},
    ]


def brainstorm_messages(
    topic: str,
    contexts: Sequence[Dict[str, str]],
    history: Optional[Sequence[Dict[str, str]]] = None,
) -> List[Dict[str, str]]:
    user = "{history}Related saved notes:\n{notes}\n\nTopic: {topic}".format(
        history=_format_history(history),
        notes=_numbered_notes(contexts),
        topic=topic,
    )
    return [
        {"role": "system", "content": BRAINSTORM_SYSTEM},
        {"role": "user", "content": user},
    ]


def detail_summary_messages(title: str, content: str) -> List[Dict[str, str]]:
    user = "Title: {title}\n\n{text}".format(
        title=title or "note", text=content[:MAX_SUMMARY_INPUT_CHARS]
    )
    return [
        {"role": "system", "content": DETAIL_SUMMARY_SYSTEM},
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
