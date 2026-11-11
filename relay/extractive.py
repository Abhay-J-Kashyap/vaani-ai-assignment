"""The zero-token answer: quoted passages with citations, built entirely by local code (ADR: extractive answers use
no Groq call). Used at L3/L4, when the wait would outlast the person's patience, and when a draft fails the gate
twice. Every sentence is a verbatim quote, so it passes the same citation gate as a drafted answer.
"""
from __future__ import annotations

import re

WORD = re.compile(r"[a-z0-9]+(?:[._-][a-z0-9]+)*")
SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[])")
STOP = set("a an the of to in on for and or is are was were be by with as at from that this these those what which "
           "how why does do did it its their there than then into about between each both use used using paper "
           "papers explain describe compare report following according".split())

NOT_COVERED = "The three papers in this library do not appear to cover this question, so I won't guess."


def content_words(s: str) -> set[str]:
    return {w for w in WORD.findall(s.lower()) if w not in STOP and len(w) > 1}


def best_quote(question: str, chunk: dict) -> str:
    q = content_words(question)
    if chunk["kind"] == "table":
        lines = [l for l in chunk["text"].split("\n") if l.strip()]
        head, rows = lines[0], lines[1:]
        best = sorted(rows, key=lambda r: len(q & content_words(r)), reverse=True)[:2]
        return f"{head.split(' (rows part')[0]} {'; '.join(r.strip() for r in best)}".strip()
    sents = [s.strip().replace("\n", " ") for s in SENT.split(chunk["text"]) if len(s.strip()) > 30]
    if not sents:
        return chunk["text"].strip().replace("\n", " ")[:400]
    return max(sents, key=lambda s: len(q & content_words(s)))


def extractive_sentences(question: str, hits: list, confident: bool, n: int = 3) -> list[str]:
    if not confident or not hits:
        return [NOT_COVERED]
    out, seen = [], set()
    for h in hits:
        c = h.chunk
        if c["cite"] in seen:
            continue
        seen.add(c["cite"])
        quote = best_quote(question, c).rstrip(".")
        out.append(f"“{quote}.” [{c['cite']}]")
        if len(out) == n:
            break
    return out


def human_wait(s: float) -> str:
    """For people, not machines: '40 seconds', 'about 22 minutes', 'about 3 hours' (not Groq's '21m31.629s')."""
    if s < 60:
        return f"{max(1, round(s))} seconds"
    if s < 90 * 60:
        return f"about {round(s / 60)} minutes"
    return f"about {round(s / 3600)} hours"


def extractive_notice(reason: str, wait_s: float | None) -> str:
    when = f" The budget refills continuously; the next full answer should be possible in {human_wait(wait_s)}." \
        if wait_s and wait_s < 86400 else ""
    lead = {
        "daily": "Full answers are paused: today's model budget is used up." + when,
        "busy": "The service is busy right now, so instead of making you wait" +
                (f" {human_wait(wait_s)}" if wait_s else "") + ", here is a quick answer.",
        "unverified": "I couldn't produce a written answer that checks out against the cited passages.",
        "error": "The answer service is having trouble right now.",
    }.get(reason, "")
    return f"{lead} These are the most relevant passages, quoted from the papers:".strip()
