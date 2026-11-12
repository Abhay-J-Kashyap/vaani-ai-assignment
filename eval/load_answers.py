"""Rebuild the text each real load request showed the person, from results/trace.jsonl, for grading.

  python -m eval.load_answers            ->  results/load_answers.jsonl (one line per real load request)

Real load scenarios: steady, burst, drain (+ its daily_exhausted tail), real_smoke.

Which question each request asked: the trace logs only the first 200 characters of the retrieval query (with the
router's keywords in front when the router ran), and nothing at all for a cache hit. So the question is found by
replaying the load generator's seeded draws (load/run_load.py, seed 7): arrivals are sorted before questions are
drawn, so the i-th request queued in a run asked the i-th drawn question. The replay is checked against every
request whose query the trace shows; any mismatch stops the script. real_smoke (5 requests, not from run_load)
is matched from its query text alone.

Each record: question id and text, tier, branch, notices, the sentences shown (with citations), the sentences
withheld, and for a cache hit the earlier Full answer it replayed (the latest cacheable Full answer to the same
question before it, in trace order).
"""
from __future__ import annotations

import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path

from eval.retrieval_scan import load_questions
from load import run_load as L

ROOT = Path(__file__).resolve().parents[1]
SEED = 7
# real_smoke (2026-11-10, older pipeline): these three logged only the router's rewritten query, never the question.
# Assigned by hand from that query's topic (assistant, 2026-11-12):
SMOKE = {"2c815bc21203": "Q1",   # "Why divide query-key dot products by sqrt of key dimension before softmax ..."
         "236d753ccc91": "Q1",   # same wording
         "261de3570169": "Q15",  # "Moshi Temporal Transformer Depth Transformer decomposition hierarchy problem"
         "0992b0c8c8c9": "Q15"}  # same router query; its decompose part restates Q15


def replay_steady_burst(name: str, qids: list[str]) -> list[str]:
    sc = L.SCENARIOS[name]
    rng = random.Random(SEED)
    times = L.arrivals(sc, rng)
    return [rng.choice(qids) for _ in times]


def replay_drain(n_drain: int, n_exhausted: int, qids: list[str]) -> list[str]:
    sc = L.SCENARIOS["drain"]
    rng = random.Random(SEED)
    out = []

    def launch():
        out.append(rng.choice(qids))
        rng.lognormvariate(math.log(L.PATIENCE_MEDIAN_S), L.PATIENCE_SIGMA)
        rng.random()

    for _ in range(n_drain):
        launch()
        rng.expovariate(sc.rate_per_min / 60)
    for _ in range(n_exhausted):
        launch()
        rng.expovariate(sc.exhausted_rate / 60)
    return out


def main() -> None:
    qs = load_questions()                                  # qid -> question, in the order run_load draws from
    qids = list(qs)
    rows = {r["request_id"]: r for r in csv.DictReader((ROOT / "results.csv").open(encoding="utf-8"))
            if r["source"] == "real" and r["scenario"] in ("steady", "burst", "drain", "daily_exhausted", "real_smoke")}

    ev, order = defaultdict(list), []
    with (ROOT / "results" / "trace.jsonl").open(encoding="utf-8") as f:
        for line in f:
            e = json.loads(line)
            rid = e.get("rid")
            if rid is None:
                continue
            if rid not in ev:
                order.append(rid)
            ev[rid].append(e)

    def seen_qid(es) -> str | None:
        """Question id from the logged query (text after '||' if the router prefixed keywords), by prefix."""
        texts = [e.get("query") or e.get("part") for e in es if e["ev"] == "retrieve_start"]
        for q in texts:
            if not q:
                continue
            q = q.split("||")[-1].strip()[:150]
            hits = [k for k, v in qs.items() if v.strip().startswith(q) or q.startswith(v.strip()[:150])]
            if len(hits) == 1:
                return hits[0]
        return SMOKE.get(es[0]["rid"])

    # replay per run, in queued (trace) order
    qid_of: dict[str, str] = {}
    for name in ("steady", "burst"):
        rids = [r for r in order if r in rows and rows[r]["scenario"] == name]
        for rid, q in zip(rids, replay_steady_burst(name, qids)):
            qid_of[rid] = q
    d = [r for r in order if r in rows and rows[r]["scenario"] == "drain"]
    x = [r for r in order if r in rows and rows[r]["scenario"] == "daily_exhausted"]
    for rid, q in zip(d + x, replay_drain(len(d), len(x), qids)):
        qid_of[rid] = q
    for rid in (r for r in order if r in rows and rows[r]["scenario"] == "real_smoke"):
        qid_of[rid] = seen_qid(ev[rid])

    checked, bad = 0, []
    for rid, q in qid_of.items():
        s = seen_qid(ev[rid])
        if s is not None:
            checked += 1
            if s != q:
                bad.append((rid, rows[rid]["scenario"], q, s))
    if bad or any(v is None for v in qid_of.values()):
        raise SystemExit(f"replay disagrees with the trace: {bad[:10]}; unresolved: "
                         f"{[k for k, v in qid_of.items() if v is None]}")

    def shown(es):
        return [{"text": e.get("text"), "cites": e.get("cites", []), "extractive": e.get("extractive", False)}
                for e in es if e["ev"] == "gate_pass" and e.get("kind") == "answer"]

    # cache hits: replay the latest cacheable Full answer to the same question before them (any scenario, real)
    last_full: dict[str, list] = {}
    out = []
    for rid in order:
        es = ev[rid]
        done = next((e for e in es if e["ev"] == "done"), {})
        q = qid_of.get(rid) or seen_qid(es)
        if rid in rows:
            rec = {"request_id": rid, "scenario": rows[rid]["scenario"], "qid": q, "question": qs[q],
                   "tier": done.get("tier"), "branch": done.get("branch"), "status": rows[rid]["status"],
                   "notices": [e["text"] for e in es if e["ev"] == "notice"],
                   "sentences": shown(es),
                   "not_covered": [e.get("text") for e in es if e["ev"] == "gate_pass" and e.get("kind") == "not_covered"],
                   "withheld": [{"text": e.get("text"), "reason": e.get("reason")} for e in es if e["ev"] == "gate_fail"]}
            if done.get("tier") == "cached":
                rec["cached_from"] = last_full.get(q)
            out.append(rec)
        if q and done.get("tier") == "full" and not done.get("withheld") and not done.get("truncated"):
            last_full[q] = {"request_id": rid, "sentences": shown(es)}

    path = ROOT / "results" / "load_answers.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for rec in out:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"{len(out)} real load requests -> {path.relative_to(ROOT)}; replayed question checked against the trace "
          f"for {checked}, all agree")
    print("cache hits without a source answer:", [r["request_id"] for r in out if r["tier"] == "cached" and not r.get("cached_from")])
    print("extractive sentences without text:", sum(1 for r in out for s in r["sentences"] if s["text"] is None))


if __name__ == "__main__":
    main()
