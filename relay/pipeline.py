"""The Relay pipeline (ADR-0001, Option C). One request:

  [0 Admit]    answer cache -> tier from the budget ledger (Full 120b / Lean 20b / Extractive) -> ladder level L1
  [1 Route]    Groq tool call on the 20b: search_papers(query, papers, sub_questions) | refuse_out_of_scope
               (heuristic router when the 20b can't be spent)
  [2 Retrieve] local hybrid search, in a worker thread; LOOP: low confidence -> reformulate, at most 2 more searches
  [3 Decide]   BRANCH: REFUSE (confidence below threshold) | DECOMPOSE (one search per paper / sub-question,
               merged; sub-questions drive retrieval only) | ANSWER
  [4 Draft]    streamed; FIFO admission per model; retries per the ADR policy; downgrade on limits
  [5 Gate]     per sentence: check_citations (deterministic tool); failed sentences are withheld; a draft with
               nothing that passes is regenerated once (<= 2 drafter calls), then falls back to quoted passages
  [6 Emit/log] one row (the 12 results.csv columns) + per-phase trace events

  python -m relay.pipeline "question" [--source stand-in|real] [--setting h256k5] [--no-cache] [--dry-run]

Status recorded per request (recommended by the assistant; pending the author's decision):
  ok            the person got the answer the tier planned for them, with no limit forcing anything
  rate_limited  a limit (a 429, or the ledger predicting one) forced a weaker answer or no answer
  timeout       a timeout or the person's deadline cut the answer short (including giving up)
  error         a server error that wasn't recovered by the retry
"""
from __future__ import annotations

import argparse
import asyncio
import math
import re
import sys
import time
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from relay import settings as S
from relay.extractive import NOT_COVERED, extractive_notice, extractive_sentences, human_wait
from relay.gate import SentenceSplitter, check_citations
from relay.ledger import Ledger
from relay.llm import LLM, CallResult
from relay.retrieval import Hit, full_tables, get_index
from relay.router import TOOLS, Route, detect_papers, heuristic_route, parse_tool_call, router_messages
from relay.runtime import AnswerCache, Recorder, RetryBudget

ROOT = Path(__file__).resolve().parents[1]

DRAFT_SYSTEM = """You answer engineers' questions using ONLY the passages provided. Rules:
1. End every sentence, including any summary, with at least one citation in square brackets, copying a passage label exactly, e.g. [AIAYN §3.2.1].
2. Use only facts and numbers stated in the passage you cite, written exactly as the passage writes them. If you compute a number (e.g. a difference), write the two source numbers in the same sentence, e.g. "2.7 points (58.2 vs 55.5)". When reading a table, take each value from its own row and column.
3. Answer every part of the question. If it asks for a list of items, give each item, each with its citation.
4. If the passages do not contain the answer, reply with one sentence: "The papers do not say ..." naming what is missing. Never guess or use outside knowledge.
5. If the passages answer only part of the question, answer that part and say which part the papers do not cover.
6. If two passages disagree (e.g. prose versus a table), report both and attribute each.
7. Plain sentences only: no headings, no tables, no bullet symbols. At most 12 sentences."""

REGEN_NOTE = ("Your previous answer was rejected because its sentences lacked valid citations or used numbers not in "
              "the cited passage. Every sentence must end with one of these labels: {labels}.")


def ntok(s: str) -> int:
    from corpus.build import ntok as _n
    return _n(s)


@dataclass
class Req:
    id: str
    question: str
    scenario: str
    patience_s: float
    t0: float = field(default_factory=time.perf_counter)
    start_iso: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="milliseconds"))
    tier: str = "full"
    branch: str = ""
    cause: str | None = None          # rate_limited | timeout | error, when something forced the outcome
    retries: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    first_token_ms: float | None = None
    sentences: list[str] = field(default_factory=list)
    notices: list[str] = field(default_factory=list)
    withheld: int = 0
    drafter_calls: int = 0
    truncated: str = ""               # why the answer was cut short, if it was
    on_event: object = None
    use_cache: bool = True            # False: this request neither reads nor writes the answer cache
    cached_in: int = 0                # prompt tokens Groq served from its prefix cache (subset of tokens_in)

    @property
    def deadline(self) -> float:
        return self.t0 + self.patience_s

    def left(self) -> float:
        return self.deadline - time.perf_counter()

    def ms(self) -> float:
        return round((time.perf_counter() - self.t0) * 1000, 1)


class Relay:
    def __init__(self, source: str = "stand-in", setting: str = S.DEFAULT_SETTING, scenario: str = "dev",
                 ledger: Ledger | None = None, base_url: str | None = None, use_cache: bool = True,
                 allow_downgrade: bool = True, retry_mode: str = "policy", llm_router: bool = True,
                 rows_path: Path | None = None, trace_path: Path | None = None, cpu_workers: int = 2):
        self.source, self.scenario = source, scenario
        self.setting = S.parse_setting(setting)
        self.llm = LLM(source, ledger, base_url)
        self.ledger = self.llm.ledger
        self.cache = AnswerCache(self.ledger.name, persist=self.ledger.persist) if use_cache else None
        self.allow_downgrade, self.llm_router = allow_downgrade, llm_router
        self.retry = RetryBudget(retry_mode)
        self.rec = Recorder(rows_path, trace_path)
        self.cpu = ThreadPoolExecutor(max_workers=cpu_workers, thread_name_prefix="retrieval")
        self.locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self.queued: dict[str, list[int]] = defaultdict(list)     # per model: token needs of admitted-but-waiting
        self.pending_full = 0     # requests past admission that will want a Full draft but haven't reserved it yet
        # Warm up before the first request: index, embedding model (and re-ranker), tokenizer. Cold, the first
        # search took ~1.1 s instead of ~0.05 s, which would land on whichever request came first.
        get_index(self.setting.size).search("warm up", 1, self.setting.rerank, late=self.setting.late)
        ntok("warm up")

    async def close(self) -> None:
        await self.llm.close()
        self.cpu.shutdown(wait=False)
        self.rec.close()

    # ------------------------------------------------------------------ public entry point
    async def ask(self, question: str, *, request_id: str | None = None, patience_s: float | None = None,
                  on_event=None, use_cache: bool = True, scenario: str | None = None) -> dict:
        """use_cache=False models a person whose wording differs from every earlier asker: the exact-match cache
        can't serve them (the load generator uses it to set a stated repeat rate). `scenario` labels this one
        request (the drain run switches from 'drain' to 'daily_exhausted' when the drafter's budget is gone)."""
        r = Req(request_id or uuid.uuid4().hex[:12], question, scenario or self.scenario,
                patience_s or S.DEFAULT_PATIENCE_S, on_event=on_event, use_cache=use_cache)
        self.retry.note_request()
        self.trace(r, "queued", patience_s=r.patience_s)
        try:
            await self._run(r)
        except asyncio.CancelledError:
            r.cause = "timeout"
            self.trace(r, "gave_up")
            raise
        except Exception as e:                                       # noqa: BLE001 -- logged, never swallowed silently
            r.cause = "error"
            self.trace(r, "exception", error=f"{type(e).__name__}: {e}"[:300])
            raise
        finally:
            self._finish(r)
        return self.result(r)

    # ------------------------------------------------------------------ the stages
    async def _run(self, r: Req) -> None:
        s = self.setting
        # [0] admission: cache, tier, ladder level
        if self.cache and r.use_cache and (hit := self.cache.get(r.question, s.name)):
            r.tier, r.branch = "cached", "cache"
            self.trace(r, "cache_hit")
            for sent in hit["sentences"]:
                self.emit(r, "sentence", sent)
            return
        k, rerank = s.k, s.rerank
        full_cost = self.est_prompt(r.question, k) + S.FULL_MAX_TOKENS
        lean_cost = self.est_prompt(r.question, max(2, k - 2)) + S.LEAN_MAX_TOKENS
        r.tier = self.ledger.tier(full_cost, lean_cost) if self.allow_downgrade else "full"
        if r.tier != "full":
            r.cause = "rate_limited"
        l1 = self.l1(full_cost)
        if l1:
            k, rerank = max(2, k - 1), False
        if r.tier == "lean":
            k = max(2, k - 2)
        self.trace(r, "admitted", tier=r.tier, l1=l1, k=k, rerank=rerank, pending_full=self.pending_full,
                   drafter_tpd=round(self.ledger.acct(S.FULL_MODEL).tpd.level()))

        self.pending_full += r.tier == "full"
        try:
            # [1] route
            route = await self.route(r, use_llm=self.llm_router and r.tier != "extractive" and not l1)
            papers = sorted(set(route.papers) | set(detect_papers(r.question)))
            # [2] retrieve, with the reformulation loop
            hits, conf = await self.retrieve_loop(r, route, papers, k, rerank)
        finally:
            self.pending_full -= r.tier == "full"

        # [3] decide
        tau = S.TAU["rerank" if rerank else "cos"]
        parts = self.plan_parts(r.question, route, papers)
        if route.action == "refuse" and conf < tau["loop"]:
            r.branch = "refuse"
        elif conf < tau["refuse"]:
            r.branch = "refuse"
        elif parts:
            r.branch = "decompose"
            hits = await self.decompose(r, parts, hits, k, rerank)
        else:
            r.branch = "answer"
        self.trace(r, "decide", branch=r.branch, confidence=round(conf, 3), papers=papers,
                   router=route.by, router_action=route.action, n_parts=len(parts))
        if r.branch == "refuse":
            self.emit(r, "sentence", NOT_COVERED)
            return
        hits, named = self.named_tables(r.question, papers, hits)
        if named:
            self.trace(r, "named_tables", added=named)
        hits = self.cap_passages(self.expand_tables(hits))

        # [4]+[5] draft through the gate, by tier
        if r.tier == "lean":
            # Say so whenever the person gets the weaker answer, not only when a draft is downgraded mid-way.
            if self.daily_gone():
                wait, _ = self.ledger.wait_s(S.FULL_MODEL, self.est_prompt(r.question, self.setting.k),
                                             S.FULL_MAX_TOKENS)
                self.emit(r, "notice", "Shorter answer: today's budget for full answers is used up; the next full "
                                       f"answer should be possible in {human_wait(wait)}.")
            else:
                self.emit(r, "notice", "Shorter answer: full answers are being rationed right now.")
        if r.tier == "extractive":
            await self.extractive(r, hits, conf >= tau["refuse"], "daily" if self.daily_gone() else "busy")
            return
        await self.draft(r, hits)

    # ------------------------------------------------------------------ [1] route
    async def route(self, r: Req, use_llm: bool) -> Route:
        if not use_llm:
            self.trace(r, "route_end", by="heuristic")
            return heuristic_route(r.question)
        msgs = router_messages(r.question)
        prompt = S.CHAT_OVERHEAD + S.TOOLS_TOKENS + sum(ntok(m["content"]) for m in msgs)
        tpm = self.ledger.acct(S.ROUTER_MODEL).tpm
        if tpm.level() - (prompt + S.ROUTER_MAX_TOKENS) < S.L1_HEADROOM * tpm.capacity:
            # L1 on the 20b: keep its per-minute headroom for Lean drafts. Measured on the stand-in: a burst of 12
            # LLM-routed requests used the 20b's whole TPM, so every Lean fallback was shed to quoted passages.
            self.trace(r, "route_end", by="heuristic", why="router tpm headroom")
            return heuristic_route(r.question)
        if not await self.admit(r, S.ROUTER_MODEL, prompt, S.ROUTER_MAX_TOKENS, max_wait=0.5):
            self.trace(r, "route_end", by="heuristic", why="router budget")
            return heuristic_route(r.question)
        self.trace(r, "route_start", prompt_est=prompt)
        res = await self.llm.chat(model=S.ROUTER_MODEL, messages=msgs, max_tokens=S.ROUTER_MAX_TOKENS,
                                  prompt_tokens=prompt, tools=TOOLS, tool_choice="required", stream=False,
                                  first_token_s=S.ROUTER_TIMEOUT_S, total_s=S.ROUTER_TIMEOUT_S, charge_start=False)
        self.count(r, res)
        route = parse_tool_call(r.question, res.tool_calls) if res.status == "ok" else None
        self.trace(r, "route_end", by="llm" if route else "heuristic", status=res.status, ms=res.total_ms, usage=res.usage,
                   tool=route and route.action, papers=route and route.papers,
                   subs=route and route.sub_questions, err=None if route else str(res.error)[:200])
        # A failed router call is not retried: the heuristic router is free and nearly as good for retrieval.
        return route or heuristic_route(r.question)

    # ------------------------------------------------------------------ [2] retrieve
    async def search(self, query: str, k: int, rerank: bool, papers: list[str] | None = None, r: Req | None = None):
        """The retrieval tool, with a deadline (the brief: "tools time out"). On timeout it degrades to BM25 alone,
        which needs no embedding model; its confidence is NaN, which the decide branch treats as 'unknown' (it
        neither refuses nor loops on it)."""
        idx = get_index(self.setting.size)
        loop = asyncio.get_running_loop()
        fut = loop.run_in_executor(self.cpu, lambda: idx.search(query, k, rerank, pool=30, papers=papers,
                                                                 dense=self.setting.dense, late=self.setting.late))
        limit = S.RETRIEVAL_TIMEOUT_S * (6 if rerank else 1)
        try:
            return await asyncio.wait_for(fut, timeout=limit)
        except asyncio.TimeoutError:
            if r:
                self.trace(r, "retrieval_timeout", limit_s=limit, query=query[:200])
            hits = idx.bm25_only(query, k, papers)
            return hits, {"retrieve_ms": limit * 1000, "fuse_rerank_ms": 0.0, "top_dense_cos": float("nan")}

    @staticmethod
    def confidence(hits, timing, rerank: bool) -> float:
        if rerank and hits and hits[0].rerank is not None:
            return hits[0].rerank
        return timing["top_dense_cos"]

    @staticmethod
    def merge(lists: list) -> list:
        """Round-robin merge of ranked hit lists, dropping repeats, so every list's best hits are kept."""
        merged, seen = [], set()
        for i in range(max((len(x) for x in lists), default=0)):
            for lst in lists:
                if i < len(lst) and lst[i].chunk["id"] not in seen:
                    seen.add(lst[i].chunk["id"])
                    merged.append(lst[i])
        return merged

    async def retrieve_loop(self, r: Req, route: Route, papers: list[str], k: int, rerank: bool):
        tau = S.TAU["rerank" if rerank else "cos"]["loop"]
        names = {"AIAYN": "Transformer attention", "MOSHI": "Moshi Mimi speech", "HNET": "H-Net dynamic chunking"}
        # The person's own question is always searched alongside the router's rewrite and the two are merged
        # (real eval, 2026-11-11: the rewrite alone missed AIAYN Table 3 for Q4, and both h512k4 and h256k8 then
        # refused a question the papers answer; its confidence was high, so the loop never searched again).
        first = [route.query] + ([r.question] if route.query.strip() != r.question.strip() else [])
        queries = [first]
        if papers:
            queries.append([f"{r.question} {' '.join(names[p] for p in papers)}"])
        best = None
        for i, qs in enumerate(queries[:1 + S.MAX_REFORMULATIONS]):
            self.trace(r, "retrieve_start", iteration=i, query=" || ".join(q[:200] for q in qs))
            results = [await self.search(q, k, rerank, r=r) for q in qs]
            hits = self.merge([h for h, _ in results])[:k]
            confs = [self.confidence(h, t, rerank) for h, t in results]
            conf = max(confs) if all(c == c for c in confs) else tau        # NaN (timed out) -> unknown
            timing = {"retrieve_ms": sum(t["retrieve_ms"] for _, t in results),
                      "fuse_rerank_ms": sum(t["fuse_rerank_ms"] for _, t in results)}
            self.trace(r, "retrieve_end", iteration=i, confidence=round(conf, 3), ms=timing["retrieve_ms"],
                       rerank_ms=timing["fuse_rerank_ms"], cites=[h.chunk["cite"] for h in hits])
            if best is None or conf > best[1]:
                best = (hits, conf)
            if conf >= tau:
                break
        return best

    @staticmethod
    def plan_parts(question: str, route: Route, papers: list[str]) -> list[str]:
        """Searches for the DECOMPOSE branch, as 'query ||PAPER' (PAPER empty = all papers).
        * The router's sub-questions first, each restricted to the paper it names. On real Groq (2026-11-10) the
          router wrote focused ones ("H-Net encoder dynamic chunking main network ... problem") that reach more of
          the method sections than the whole comparison question does.
        * Then the whole question, restricted to each paper that no sub-question covered: a single search over a
          cross-paper question returns one paper only (gold set Q13-Q15)."""
        parts, covered = [], set()
        for s in route.sub_questions[:3]:
            named = detect_papers(s)
            p = named[0] if len(named) == 1 else ""
            parts.append(f"{s} ||{p}")
            covered.update(named)
        if len(papers) >= 2:
            parts += [f"{question} ||{p}" for p in papers if p not in covered]
        return parts[:4]

    async def decompose(self, r: Req, parts: list[str], hits, k: int, rerank: bool):
        """Sub-questions drive retrieval only (author, 2026-11-10): one search per part, merged round-robin so each
        part is represented, then one draft."""
        per = max(2, math.ceil(k / len(parts)) + 1)
        lists = [hits]
        for p in parts:
            q, _, paper = p.partition(" ||")
            self.trace(r, "retrieve_start", part=p[:200])
            h, timing = await self.search(q, per, rerank, papers=[paper] if paper else None, r=r)
            self.trace(r, "retrieve_end", part=p[:200], ms=timing["retrieve_ms"], rerank_ms=timing["fuse_rerank_ms"],
                       cites=[x.chunk["cite"] for x in h])
            lists.append(h)
        return self.merge(lists[1:] + lists[:1])

    def named_tables(self, question: str, papers: list[str], hits):
        """A question that names a table ("According to Table 1") gets that table looked up directly, in the paper
        the question is about. Every paper has a Table 1-3, so retrieval alone fetched the wrong paper's table
        (real eval, 2026-11-11: Q9 at h256k8 never got MOSHI Table 1; Q4 got HNET Table 1 for AIAYN Table 3).
        With no identifiable paper, nothing is added: guessing the paper would recreate the same trap."""
        nums = sorted(set(re.findall(r"\bTable\s+(\d+)\b", question)))
        if not nums or not papers:
            return hits, []
        idx = get_index(self.setting.size)
        have = {h.chunk["cite"] for h in hits}
        added = []
        for p in papers:
            for n in nums:
                label = f"{p} Table {n}"
                if label in have:
                    continue
                part = next((c for c in idx.chunks if c["cite"] == label), None)
                if part:
                    added.append(Hit(part, float("inf"), None, None, float("nan")))
        return added + hits, [h.chunk["cite"] for h in added]

    @staticmethod
    def expand_tables(hits):
        """Replace a retrieved table part with the whole table (once), when the table fits TABLE_EXPAND_MAX."""
        from dataclasses import replace
        tables, out, seen = full_tables(), [], set()
        for h in hits:
            c = h.chunk
            if c["kind"] == "table" and c["cite"] in tables:
                if c["cite"] in seen:
                    continue
                seen.add(c["cite"])
                text = tables[c["cite"]]
                n = ntok(text)
                if n <= S.TABLE_EXPAND_MAX:
                    h = replace(h, chunk=c | {"text": text, "tokens": n, "expanded": True})
            out.append(h)
        return out

    @staticmethod
    def cap_passages(hits):
        out, total = [], 0
        for h in hits:
            if total + h.chunk["tokens"] > S.MAX_PASSAGE_TOKENS and out:
                continue
            out.append(h)
            total += h.chunk["tokens"]
        return out

    # ------------------------------------------------------------------ [4]+[5] draft through the gate
    @staticmethod
    def passages_of(hits) -> dict[str, str]:
        """cite label -> all retrieved text under that label (several pieces can share one label)."""
        out: dict[str, str] = {}
        for h in hits:
            out[h.chunk["cite"]] = (out.get(h.chunk["cite"], "") + "\n" + h.chunk["text"]).strip()
        return out

    @staticmethod
    def draft_messages(question: str, hits, regen_note: str = "") -> list[dict]:
        body = "\n".join(f"[{h.chunk['cite']}] {h.chunk['text']}" for h in hits)
        user = f"PASSAGES:\n{body}\nQUESTION: {question}"
        system = DRAFT_SYSTEM + ("\n" + regen_note if regen_note else "")
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    async def draft(self, r: Req, hits) -> None:
        regen = ""
        while r.drafter_calls < S.MAX_DRAFTER_CALLS:
            lean = r.tier == "lean"
            model = S.LEAN_MODEL if lean else S.FULL_MODEL
            max_tokens = S.LEAN_MAX_TOKENS if lean else S.FULL_MAX_TOKENS
            use = hits[:max(2, self.setting.k - 2)] if lean else hits      # Lean: fewer passages
            passages = self.passages_of(use)
            msgs = self.draft_messages(r.question, use, regen)
            prompt = S.CHAT_OVERHEAD + sum(ntok(m["content"]) for m in msgs)
            outcome = await self.call_drafter(r, model, msgs, prompt, max_tokens, passages)
            if outcome in ("downgrade", "fallback"):
                if outcome == "downgrade" and not lean and self.allow_downgrade and self.lean_ok(prompt):
                    r.tier = "lean"
                    self.emit(r, "notice", "Shorter answer: full answers are being rationed right now.")
                    continue
                # Evaluation runs never downgrade (author, 2026-11-09 (d)): the row records the limit, no answer.
                if not r.sentences and self.allow_downgrade:
                    reason = "daily" if self.daily_gone() else ("error" if r.cause == "error" else "busy")
                    await self.extractive(r, hits, True, reason)
                return
            if r.sentences:
                if r.withheld:
                    self.emit(r, "notice", f"{r.withheld} sentence(s) withheld because they could not be verified "
                                           f"against the cited passages.")
                if r.truncated:
                    self.emit(r, "notice", f"Answer cut short ({r.truncated}); the sentences above are complete "
                                           f"and verified.")
                if not r.withheld and not r.truncated and self.cache and r.use_cache and r.tier == "full":
                    self.cache.put(r.question, self.setting.name, {"sentences": r.sentences})
                return
            if r.truncated and r.cause == "timeout":
                break                                             # out of time: no second draft
            regen = REGEN_NOTE.format(labels=", ".join(f"[{c}]" for c in passages))
            self.trace(r, "regenerate", withheld=r.withheld)
        if self.allow_downgrade:
            await self.extractive(r, hits, True, "unverified")

    async def call_drafter(self, r: Req, model: str, msgs, prompt: int, max_tokens: int,
                           passages: dict[str, str]) -> str:
        """One drafter call with the ADR retry policy. Returns 'done', 'downgrade' (a limit) or 'fallback'."""
        attempt = 0
        while True:
            if not await self.admit(r, model, prompt, max_tokens, max_wait=r.left() - S.DRAFT_MARGIN_S,
                                    naive=self.retry.mode == "naive" and attempt > 0):
                r.cause = r.cause or "rate_limited"
                return "downgrade"
            r.drafter_calls += 1
            self.trace(r, "call_start", model=model, attempt=attempt, prompt_est=prompt, max_tokens=max_tokens)
            splitter, emitted_before = SentenceSplitter(), len(r.sentences)
            res: CallResult | None = None
            async for kind, val in self.llm.stream_chat(
                    model=model, messages=msgs, max_tokens=max_tokens, prompt_tokens=prompt,
                    first_token_s=min(S.FIRST_TOKEN_TIMEOUT_S, max(0.5, r.left())),
                    total_s=min(S.CALL_TOTAL_TIMEOUT_S, max(1.0, r.left())), charge_start=False):
                if kind == "delta":
                    # gpt-oss often writes citations in its native full-width brackets 【…】 (real Groq, 2026-11-10:
                    # this alone failed a whole draft). Normalise the bracket style only; the label is still checked.
                    val = val.replace("【", "[").replace("】", "]")
                    for sent in splitter.feed(val):
                        self.gate(r, sent, passages)
                else:
                    res = val
            self.count(r, res)
            self.trace(r, "call_end", model=model, status=res.status, http=res.http_status, ms=res.total_ms,
                       first_byte_ms=res.first_token_ms, usage=res.usage, err=str(res.error)[:200] if res.error else None,
                       queue_time=res.usage.get("queue_time") if res.usage else None, finish=res.finish_reason)
            if res.status == "ok" and res.finish_reason != "length":
                for sent in splitter.flush():
                    self.gate(r, sent, passages)
                return "done"
            if res.first_token_ms is not None:
                # Cut short after the answer started (max_tokens, total timeout, or a stream error). The unfinished
                # tail is dropped, never sent: a sentence cut mid-thought is the "embarrassing output" the brief
                # forbids. No retry after the first token (ADR retry policy).
                tail = splitter.flush()
                r.truncated = {"ok": "length limit", "timeout": "took too long"}.get(res.status, "stream error")
                if res.status != "ok":
                    r.cause = r.cause or res.status
                self.trace(r, "truncated", why=r.truncated, dropped_tail=tail[0][:200] if tail else None)
                return "done"
            if res.status == "ok":
                # 200 with no answer text: gpt-oss spent max_tokens on reasoning (measured Day 1). Not a server
                # fault, so no retry; draft() regenerates once.
                self.trace(r, "empty_answer", finish=res.finish_reason)
                return "done"
            # failed before any answer token: decide whether to retry
            assert len(r.sentences) == emitted_before
            retry, wait = self.should_retry(r, res, attempt)
            if not retry:
                if res.status == "rate_limited":
                    r.cause = r.cause or "rate_limited"
                    return "downgrade"
                r.cause = r.cause or ("timeout" if res.status == "timeout" else "error")
                return "fallback"
            r.retries += 1
            self.retry.take()
            self.trace(r, "retry_wait_start", wait_s=round(wait, 3), why=res.status, attempt=attempt)
            await asyncio.sleep(wait)
            self.trace(r, "retry_wait_end")
            attempt += 1

    def should_retry(self, r: Req, res: CallResult, attempt: int) -> tuple[bool, float]:
        if self.retry.mode == "naive":
            return attempt < 2, 0.0                                   # 3 immediate attempts, no budget
        if self.retry.mode == "none" or not self.retry.allow():
            return False, 0.0
        if res.status == "rate_limited":
            lim = res.limit or {}
            if lim.get("daily"):
                return False, 0.0                                     # never retry a daily limit
            wait = float(lim.get("wait_s") or res.headers.get("retry-after") or 1.0)
            return attempt < 2 and wait < r.left() - S.DRAFT_MARGIN_S, wait
        if res.status == "timeout":                                   # no first token in time
            return attempt < 1 and r.left() >= 2 * S.FIRST_TOKEN_P50_S + S.DRAFT_MARGIN_S, 0.0
        return attempt < 1 and r.left() > S.DRAFT_MARGIN_S + 1, self.retry.backoff(attempt)   # 5xx / connection

    def gate(self, r: Req, sentence: str, passages: dict[str, str]) -> None:
        g = check_citations(sentence, passages, r.question)
        self.trace(r, "gate_pass" if g.ok else "gate_fail", kind=g.kind, reason=g.reason, cites=g.cites,
                   derived=g.derived or None, text=sentence[:2000])   # whole sentence: the ungated answer is rebuilt from these
        if not g.ok:
            r.withheld += 1
        elif g.kind != "skip":
            self.emit(r, "sentence", sentence)

    async def extractive(self, r: Req, hits, confident: bool, reason: str) -> None:
        if r.tier != "extractive":
            self.trace(r, "fallback_extractive", reason=reason)
        r.tier = "extractive"
        wait = None
        if reason in ("daily", "busy"):
            wait, _ = self.ledger.wait_s(S.FULL_MODEL, self.est_prompt(r.question, self.setting.k), S.FULL_MAX_TOKENS)
        sents = extractive_sentences(r.question, hits, confident)
        if sents != [NOT_COVERED]:
            self.emit(r, "notice", extractive_notice(reason, wait))
        passages = self.passages_of(hits)
        for sent in sents:
            g = check_citations(sent, passages, r.question)
            self.trace(r, "gate_pass" if g.ok else "gate_fail", kind=g.kind, reason=g.reason, extractive=True)
            if g.ok:
                self.emit(r, "sentence", sent)

    # ------------------------------------------------------------------ admission against the ledger
    async def admit(self, r: Req, model: str, prompt: int, max_tokens: int, max_wait: float, naive: bool = False) -> bool:
        """FIFO per model. Sheds at once when the estimated wait (including requests queued ahead) exceeds
        max_wait. On success the call is reserved in the ledger; the caller must start it immediately."""
        need = prompt + max_tokens
        if naive:
            self.ledger.start(model, prompt, max_tokens)
            return True
        a = self.ledger.acct(model)
        q = self.queued[model]
        est = max(a.tpm.wait_for(need + sum(q)), a.tpd.wait_for(need + sum(q)),
                  a.rpm.wait_for(1 + len(q)), a.rpd.wait_for(1 + len(q)))
        if est > max_wait:
            self.trace(r, "shed", model=model, est_wait_s=round(est, 2), max_wait_s=round(max_wait, 2))
            return False
        q.append(need)
        give_up_at = time.perf_counter() + max(0.0, max_wait)
        self.trace(r, "admit_wait_start", model=model, est_wait_s=round(est, 2), ahead=len(q) - 1)
        lock = self.locks[model]
        try:
            try:
                await asyncio.wait_for(lock.acquire(), timeout=max(0.001, give_up_at - time.perf_counter()))
            except asyncio.TimeoutError:
                self.trace(r, "shed", model=model, why="queue wait")
                return False
            try:
                while True:
                    wait, which = self.ledger.wait_s(model, prompt, max_tokens)
                    if wait <= 0:
                        self.ledger.start(model, prompt, max_tokens)
                        self.trace(r, "admit_wait_end", model=model)
                        return True
                    if time.perf_counter() + wait > give_up_at:
                        self.trace(r, "shed", model=model, est_wait_s=round(wait, 2), which=which)
                        return False
                    await asyncio.sleep(wait + 0.01)
            finally:
                lock.release()
        finally:
            q.remove(need)

    def lean_ok(self, prompt: int) -> bool:
        r = self.ledger.acct(S.LEAN_MODEL).tpd
        return r.level() - (prompt + S.LEAN_MAX_TOKENS) >= 0.25 * r.capacity

    def daily_gone(self) -> bool:
        a = self.ledger.acct(S.FULL_MODEL)
        need = self.est_prompt("", self.setting.k) + S.FULL_MAX_TOKENS
        return a.tpd.level() < need or a.rpd.level() < 1

    def l1(self, full_cost: int) -> bool:
        """Ladder L1 (tight per-minute): the drafter's per-minute headroom, after the Full drafts already on their
        way (routing or retrieving) and those queued for admission, is below 30%. Counting requests in flight
        matters in a burst: at t=0 every bucket is full, but twelve requests are about to want it."""
        a = self.ledger.acct(S.FULL_MODEL)
        demand = self.pending_full * full_cost + sum(self.queued[S.FULL_MODEL])
        n = self.pending_full + len(self.queued[S.FULL_MODEL])
        return (a.tpm.level() - demand < S.L1_HEADROOM * a.tpm.capacity
                or a.rpm.level() - n < S.L1_HEADROOM * a.rpm.capacity)

    def est_prompt(self, question: str, k: int) -> int:
        return S.CHAT_OVERHEAD + ntok(DRAFT_SYSTEM) + ntok(question) + min(S.MAX_PASSAGE_TOKENS,
                                                                            int(k * self.setting.size * 0.85))

    # ------------------------------------------------------------------ [6] emit and log
    def emit(self, r: Req, kind: str, text: str) -> None:
        if kind == "sentence":
            if r.first_token_ms is None:
                r.first_token_ms = r.ms()
                self.trace(r, "first_answer_token")
            r.sentences.append(text)
        else:
            r.notices.append(text)
            self.trace(r, "notice", text=text[:1000])       # what the person was told (heading 7 evidence)
        if r.on_event:
            r.on_event(kind, text)

    def count(self, r: Req, res: CallResult | None) -> None:
        if res and res.usage:
            r.tokens_in += int(res.usage.get("prompt_tokens") or 0)
            r.tokens_out += int(res.usage.get("completion_tokens") or 0)
            r.cached_in += int((res.usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)

    def trace(self, r: Req, ev: str, **kw) -> None:
        self.rec.trace({"rid": r.id, "t_ms": r.ms(), "ev": ev, **{k: v for k, v in kw.items() if v is not None}})

    def _finish(self, r: Req) -> None:
        status = r.cause or "ok"
        self.trace(r, "done", status=status, tier=r.tier, branch=r.branch, sentences=len(r.sentences),
                   withheld=r.withheld, drafter_calls=r.drafter_calls, retries=r.retries,
                   tokens_in=r.tokens_in, cached_in=r.cached_in)
        self.rec.row({
            "request_id": r.id, "scenario": r.scenario, "source": self.source, "start_time": r.start_iso,
            "first_token_ms": "" if r.first_token_ms is None else r.first_token_ms, "total_ms": r.ms(),
            "status": status, "retries": r.retries, "tokens_in": r.tokens_in, "tokens_out": r.tokens_out,
            "retrieval_setting": S.setting_label(self.setting.name, r.tier), "answer_grade": "",
        })

    def result(self, r: Req) -> dict:
        return {"request_id": r.id, "status": r.cause or "ok", "tier": r.tier, "branch": r.branch,
                "first_token_ms": r.first_token_ms, "total_ms": r.ms(), "retries": r.retries,
                "tokens_in": r.tokens_in, "tokens_out": r.tokens_out, "withheld": r.withheld,
                "drafter_calls": r.drafter_calls, "notices": r.notices, "sentences": list(r.sentences),
                "answer": " ".join(r.sentences)}


# ---------------------------------------------------------------------- CLI
def log_paths(source: str, scenario: str) -> tuple[Path, Path]:
    """Real calls always log to results/ (every real request is kept). Stand-in development runs go to data/dev/."""
    if source == "real" or scenario != "dev":
        return ROOT / "results" / f"pipeline_rows_{scenario}.csv", ROOT / "results" / "trace.jsonl"
    return ROOT / "data" / "dev" / "pipeline_rows_dev.csv", ROOT / "data" / "dev" / "trace_dev.jsonl"


async def dry_run(question: str, setting: str) -> None:
    """Route heuristically and retrieve locally; print what a real request would send and cost. No model call."""
    s = S.parse_setting(setting)
    idx = get_index(s.size)
    route = heuristic_route(question)
    hits, timing = idx.search(question, s.k, s.rerank, dense=s.dense, late=s.late)
    msgs = Relay.draft_messages(question, Relay.cap_passages(hits))
    prompt = S.CHAT_OVERHEAD + sum(ntok(m["content"]) for m in msgs)
    router = S.CHAT_OVERHEAD + S.TOOLS_TOKENS + sum(ntok(m["content"]) for m in router_messages(question))
    print(f"papers={route.papers} confidence={Relay.confidence(hits, timing, s.rerank):.3f} "
          f"cites={[h.chunk['cite'] for h in hits]}")
    print(f"router call ({S.ROUTER_MODEL}): ~{router} prompt + {S.ROUTER_MAX_TOKENS} max "
          f"-> TPM charge ~{router + S.ROUTER_MAX_TOKENS}")
    print(f"draft call  ({S.FULL_MODEL}): ~{prompt} prompt + {S.FULL_MAX_TOKENS} max "
          f"-> TPM charge ~{prompt + S.FULL_MAX_TOKENS}; TPD charge = actual usage (~{prompt} + completion)")


async def main_async(a) -> None:
    if a.dry_run:
        await dry_run(a.question, a.setting)
        return
    rows, trace = log_paths(a.source, a.scenario)
    relay = Relay(a.source, a.setting, a.scenario, base_url=a.base_url, use_cache=not a.no_cache,
                  rows_path=rows, trace_path=trace)

    def show(kind, text):
        print(("  " if kind == "sentence" else "  (") + text + ("" if kind == "sentence" else ")"), flush=True)

    try:
        res = await relay.ask(a.question, on_event=show)
    finally:
        await relay.close()
    print({k: v for k, v in res.items() if k not in ("answer", "notices")})
    print(f"row -> {rows.relative_to(ROOT)}; trace -> {trace.relative_to(ROOT)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--source", choices=["stand-in", "real"], default="stand-in")
    ap.add_argument("--setting", default=S.DEFAULT_SETTING)
    ap.add_argument("--scenario", default="dev")
    ap.add_argument("--base-url")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="route + retrieve locally and print the token cost")
    ap.add_argument("--confirm-real", action="store_true", help="required with --source real (spends daily budget)")
    a = ap.parse_args()
    if a.source == "real" and not (a.dry_run or a.confirm_real):
        sys.exit("--source real spends shared daily budget: run with --dry-run first, then --confirm-real")
    asyncio.run(main_async(a))


if __name__ == "__main__":
    main()
