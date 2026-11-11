"""Stage 1: route the question. A Groq tool-calling request on the small model (the brief's "tool call"):
the model picks `search_papers(query, papers?, sub_questions?)` or `refuse_out_of_scope(reason)`.

The heuristic router (no LLM call) is used when the 20b can't be spent: degradation level L1 and below, the
extractive tier, or a failed router call. It detects which papers a question is about from their vocabulary.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from relay.settings import PAPERS

TOOLS = [
    {"type": "function", "function": {
        "name": "search_papers",
        "description": "Search three papers: AIAYN = 'Attention Is All You Need' (the original Transformer); "
                       "MOSHI = Moshi, a speech-text dialogue model with the Mimi codec; HNET = 'Dynamic Chunking' "
                       "(H-Net). Use for any question about these papers or their topics, even if the exact "
                       "detail may be missing.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "the question rewritten as a search query with its key "
                                                       "technical terms"},
            "papers": {"type": "array", "items": {"type": "string", "enum": list(PAPERS)},
                       "description": "the papers the question is about; omit if unsure"},
            "sub_questions": {"type": "array", "items": {"type": "string"}, "maxItems": 3,
                              "description": "only if the question needs facts from more than one paper or has "
                                             "separate parts: one self-contained search query per part"}},
            "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "refuse_out_of_scope",
        "description": "Only for questions clearly unrelated to these papers (e.g. weather, cooking, unrelated "
                       "software). Never for questions about the papers' topics.",
        "parameters": {"type": "object", "properties": {"reason": {"type": "string"}}, "required": ["reason"]}}},
]

SYSTEM = ("You route engineers' questions about three research papers. Call exactly one tool. Prefer "
          "search_papers; name the papers involved, and split questions that compare papers into one sub-question "
          "per paper.")

VOCAB = {
    # whole words: "transformer base" once matched "BPE Transformer baseline" in an H-Net question (Q12)
    "AIAYN": r"attention is all you need|original transformer|transformer base\b|transformer big\b|base model\b|"
             r"scaled dot.product|multi.head|positional encoding|sinusoid|encoder.decoder attention|\bbleu\b|\bwmt\b|"
             r"label smoothing|\bd_?model\b|\bd_?ff\b|\bd_k\b|vaswani|machine translation|constituency",
    "MOSHI": r"\bmoshi\b|\bmimi\b|inner monologue|full.duplex|depth transformer|temporal transformer|\bhelium\b|"
             r"spoken dialogue|speech.to.speech|\brvq\b|semantic token|acoustic (?:token|delay)|\bfisher\b",
    "HNET": r"\bh.net\b|dynamic chunking|dechunk|routing module|downsampler|byte.level|tokeniz|\bchunk",
}


@dataclass
class Route:
    action: str                          # search | refuse
    query: str
    papers: list[str] = field(default_factory=list)
    sub_questions: list[str] = field(default_factory=list)
    reason: str = ""
    by: str = "heuristic"                # llm | heuristic


def detect_papers(q: str) -> list[str]:
    ql = q.lower()
    return [p for p in PAPERS if re.search(VOCAB[p], ql)]


def heuristic_route(question: str) -> Route:
    return Route("search", question, detect_papers(question), [], by="heuristic")


def parse_tool_call(question: str, tool_calls: list) -> Route | None:
    """Turn the model's tool call into a Route. Streamed tool calls may arrive split; join argument fragments."""
    if not tool_calls:
        return None
    name, args = None, ""
    for tc in tool_calls:
        fn = tc.get("function") or {}
        name = name or fn.get("name")
        args += fn.get("arguments") or ""
    try:
        a = json.loads(args or "{}")
    except json.JSONDecodeError:
        return None
    if name == "refuse_out_of_scope":
        return Route("refuse", question, reason=str(a.get("reason", ""))[:300], by="llm")
    if name != "search_papers":
        return None
    papers = [p for p in (a.get("papers") or []) if p in PAPERS]
    subs = [s for s in (a.get("sub_questions") or []) if isinstance(s, str) and s.strip()][:3]
    q = a.get("query") if isinstance(a.get("query"), str) and a["query"].strip() else question
    return Route("search", q, papers, subs, by="llm")


def router_messages(question: str) -> list[dict]:
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]
