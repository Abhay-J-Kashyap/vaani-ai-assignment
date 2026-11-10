"""Stage 5: the per-sentence citation gate (ADR Option C). Deterministic; costs no tokens.

`SentenceSplitter` cuts the streamed draft into sentences as soon as each is complete. A citation written after
the full stop ("... gradients. [AIAYN §3.2.1]") is attached to the sentence before it, so the splitter waits to see
the next non-space character before it cuts.

`check_citations(sentence, passages, question)` is the code tool. A sentence passes when:
  1. it carries at least one citation, and every citation is the label of a passage that was retrieved for this
     request; and
  2. every number in it appears in a cited passage, or in the question, or is the sum, difference, product or
     ratio of two numbers in the cited passages (so "58.2 - 55.5 = 2.7" can pass, and is recorded as derived).
A sentence saying the papers don't cover something needs no citation (it is the refusal path), but any number in it
must come from the question or a cited passage.
What it does NOT check: whether a correctly cited sentence reads its source correctly. That gap is measured
offline with the gold set (ADR "The cost, stated plainly").
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

PAPER_CODES = ("AIAYN", "MOSHI", "HNET")
ABBREV = {"e.g", "i.e", "et al", "al", "fig", "figs", "eq", "eqs", "sec", "vs", "cf", "approx", "resp", "no", "tab",
          "etc", "ca", "appx"}
BRACKET = re.compile(r"\[([^\[\]]{2,120})\]")
NUM = re.compile(r"(?<![\w.])[-−]?\d[\d,]*(?:\.\d+)?(?:[eE][-−]?\d+)?(?:[KMB](?![a-zA-Z]))?")
REF_NUM = re.compile(r"(?:§\s*|\b(?:Table|Tables|Figure|Figures|Fig\.|Section|Sections|Sec\.|Eq\.|Equation|Appendix|"
                     r"Algorithm|Q|arXiv:?)\s*)[A-Z]?\d+(?:\.\d+)*(?:\s*(?:,|and|–|-)\s*[A-Z]?\d+(?:\.\d+)*)*", re.I)
NOT_COVERED = re.compile(
    r"\b(?:papers?|passages?|sources?|library|provided (?:text|material)|they|it)\b[^.]{0,80}?"
    r"\b(?:do(?:es)?\s+not|don't|doesn't|never|did\s+not)\b[^.]{0,60}?"
    r"\b(?:say|state|report|cover|mention|provide|give|establish|specify|include|describe|contain|discuss|address|"
    r"disclose|list|measure|evaluate|present)"
    r"|\bnot\s+(?:covered|reported|stated|specified|mentioned|provided|established|given|addressed|disclosed)\b"
    r"|\bno\s+(?:information|data|figures?|mention|reported|such)\b", re.I)
MARKDOWN = re.compile(r"^\s*(?:[-*•]\s+|\d+[.)]\s+|#+\s+)|\*\*|__|`")


def normalise_cite(raw: str, last_paper: str | None) -> tuple[str | None, str | None]:
    s = re.sub(r"\s+", " ", raw.strip()).replace("Section ", "§").replace("Sec. ", "§").replace("§ ", "§")
    for code in PAPER_CODES:
        if s.upper().startswith(code):
            return f"{code} {s[len(code):].strip()}", code
    if last_paper and re.match(r"^(§|Table|Figure|Abstract|Algorithm)", s):
        return f"{last_paper} {s}", last_paper
    return None, last_paper


def find_cites(sentence: str) -> tuple[list[str], list[str]]:
    """Citations in a sentence -> (labels, unparseable bracket texts). Handles [A §1; A Table 2] and [A §1, §2]."""
    labels, bad = [], []
    for m in BRACKET.finditer(sentence):
        inner = m.group(1)
        if not re.search(r"\b(AIAYN|MOSHI|HNET)\b", inner, re.I):
            continue   # a bracket that isn't a citation, e.g. "[0, 1]"
        last = None
        for part in re.split(r"[;,]", inner):
            if not part.strip():
                continue
            label, last = normalise_cite(part, last)
            (labels if label else bad).append(label or part.strip())
    return labels, bad


def strip_cites(sentence: str) -> str:
    return BRACKET.sub(lambda m: "" if re.search(r"\b(AIAYN|MOSHI|HNET)\b", m.group(1), re.I) else m.group(0),
                       sentence)


def num_value(tok: str) -> float | None:
    t = tok.replace(",", "").replace("−", "-")
    mult = {"K": 1e3, "M": 1e6, "B": 1e9}.get(t[-1:], 1)
    if mult != 1:
        t = t[:-1]
    try:
        return float(t) * mult
    except ValueError:
        return None


def numbers(text: str, drop_refs: bool = True) -> list[tuple[str, float]]:
    if drop_refs:
        text = REF_NUM.sub(" ", text)
    out = []
    for m in NUM.finditer(text):
        v = num_value(m.group(0))
        if v is not None:
            out.append((m.group(0), v))
    return out


def decimals(tok: str) -> int:
    t = tok.rstrip("KMB").replace(",", "")
    return len(t.split(".")[1]) if "." in t else 0


def close(a: float, b: float) -> bool:
    return abs(a - b) <= 1e-9 * max(1.0, abs(a), abs(b))


def derived(tok: str, v: float, pool: list[float]) -> str | None:
    """Is v (rounded as written) the sum, difference, product or ratio of two distinct numbers that appear both in
    the cited passages and in the sentence itself? Requiring the operands in the sentence ("2.7 (58.2 - 55.5)")
    stops chance matches such as 12 = 6 + 6."""
    d = decimals(tok)
    pool = sorted(set(pool))[:200]
    for i, a in enumerate(pool):
        for b in pool[i + 1:]:
            for name, r in (("diff", abs(a - b)), ("sum", a + b), ("prod", a * b),
                            ("ratio", a / b if b else None), ("ratio", b / a if a else None)):
                if r is not None and close(round(r, d), abs(v)):
                    return f"{name}({a:g},{b:g})"
    return None


@dataclass
class GateResult:
    ok: bool
    kind: str                       # answer | not_covered | skip (no content) | fail
    reason: str = ""
    cites: list[str] = field(default_factory=list)
    unmatched: list[str] = field(default_factory=list)
    derived: list[str] = field(default_factory=list)


def check_citations(sentence: str, passages: dict[str, str], question: str) -> GateResult:
    """The deterministic gate. `passages` maps each retrieved cite label to its retrieved text (joined if several
    pieces share a label)."""
    plain = MARKDOWN.sub("", sentence).strip()
    if not re.search(r"[A-Za-z]{2}", strip_cites(plain)):
        return GateResult(True, "skip", "no content")
    if re.fullmatch(r"[^.!?]{0,60}:", plain):
        return GateResult(True, "skip", "heading")
    labels, bad = find_cites(plain)
    q_nums = [v for _, v in numbers(question, drop_refs=False)]
    body = strip_cites(plain)
    if not labels:
        if NOT_COVERED.search(body):
            stray = [t for t, v in numbers(body) if not any(close(v, q) for q in q_nums)]
            if stray:
                return GateResult(False, "fail", "uncited number in a not-covered sentence", unmatched=stray)
            return GateResult(True, "not_covered")
        return GateResult(False, "fail", "no citation" + (f" (unparsed: {bad})" if bad else ""))
    unknown = [c for c in labels if c not in passages]
    if unknown:
        return GateResult(False, "fail", f"cites not retrieved: {unknown}", cites=labels)
    cited_text = "\n".join(passages[c] for c in labels)
    pool = [v for _, v in numbers(cited_text, drop_refs=False)]
    unmatched, how = [], []
    body_nums = numbers(body)
    in_both = [p for p in pool if any(close(p, v) for _, v in body_nums)]
    for tok, v in body_nums:
        if any(close(abs(v), abs(p)) for p in pool) or any(close(v, q) for q in q_nums):
            continue
        d = derived(tok, v, in_both)
        if d:
            how.append(f"{tok}={d}")
        else:
            unmatched.append(tok)
    if unmatched:
        return GateResult(False, "fail", f"numbers not in cited passages: {unmatched}", labels, unmatched, how)
    return GateResult(True, "answer", cites=labels, derived=how)


class SentenceSplitter:
    """Feed streamed text; get back each sentence once it is complete (with any trailing citation attached)."""

    def __init__(self):
        self.buf = ""

    def feed(self, text: str) -> list[str]:
        self.buf += text
        out = []
        while True:
            cut = self._find_cut()
            if cut is None:
                break
            s, self.buf = self.buf[:cut].strip(), self.buf[cut:]
            if s:
                out.append(s)
        return out

    def flush(self) -> list[str]:
        s, self.buf = self.buf.strip(), ""
        return [s] if s else []

    def _find_cut(self) -> int | None:
        b, depth, i = self.buf, 0, 0
        while i < len(b):
            ch = b[i]
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth = max(0, depth - 1)
            elif ch == "\n" and depth == 0 and b[:i].strip():
                return i + 1
            elif ch in ".!?" and depth == 0:
                j = i + 1
                if j >= len(b):
                    return None                     # need to see what follows
                if not b[j].isspace():
                    i += 1
                    continue                        # 0.5, e.g., §3.2.1
                word = re.search(r"([A-Za-z][A-Za-z.]*)$", b[:i])
                if word and word.group(1).lower().rstrip(".") in ABBREV:
                    i += 1
                    continue
                k = j
                while k < len(b) and b[k] in " \t":
                    k += 1
                if k >= len(b):
                    return None
                e = k
                while e < len(b) and b[e] == "[":   # citations after the full stop belong to this sentence
                    if "]" not in b[e:]:
                        return None                 # citation still arriving
                    m = BRACKET.match(b, e)
                    if not (m and re.search(r"\b(AIAYN|MOSHI|HNET)\b", m.group(1), re.I)):
                        break
                    e = m.end()
                    while e < len(b) and b[e] in " .":
                        e += 1
                if e >= len(b) and e > k:
                    return None                     # might be followed by another citation
                return e
            i += 1
        return None
