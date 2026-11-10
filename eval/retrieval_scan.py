"""Shortlist retrieval settings for Table 3 without Groq.

  python -m eval.retrieval_scan [--settings bm256k8 h256k5 ...]

For every gold question and setting, runs the pipeline's own retrieval path (Relay.retrieve_loop, the decide
branch, Relay.decompose, Relay.cap_passages) with the heuristic router, since the LLM router would cost Groq calls.
Per answerable question it records which required points are *reachable*: at least one of the point's listed cites
is among the retrieved passage labels. That is an upper bound on what a perfect drafter could answer and a
conservative one (the key's cite lists are a guide; a fact may also be stated elsewhere). Per unanswerable question
it records whether the decide branch refuses before drafting.

Writes results/retrieval_scan.csv (one row per setting x question) and prints a summary per setting:
  points_reachable   share of required points across the 15 answerable questions
  all_reachable      questions with every required point reachable (the ceiling on strict 'correct')
  refused_unans      unanswerable questions refused by the decide branch (the rest rely on the drafter)
  refused_ans        answerable questions wrongly refused by the decide branch
  prompt_tok p50/max drafter prompt size (passages + instructions + question)
  retr_ms p50/max    retrieval time, including re-rank, all searches of a request
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import re
import statistics
import time
from pathlib import Path

from relay import settings as S
from relay.ledger import Ledger
from relay.pipeline import Relay, Req, ntok
from relay.router import detect_papers, heuristic_route

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ["bm256k8", "h128k8", "h256k5", "h256k8", "h512k4", "h512k6", "h256k5rr", "h512k4rr"]


def load_questions() -> dict[str, str]:
    txt = (ROOT / "eval" / "the_relay_20_evaluation_questions.txt").read_text(encoding="utf-8")
    out = {}
    for b in re.split(r"\n(?=Q\d+\. )", txt):
        m = re.match(r"Q(\d+)\. [^\n]*\n(.+)", b, flags=re.S)
        if m:
            out[f"Q{m.group(1)}"] = " ".join(m.group(2).split("\nPART ")[0].split())
    return out


async def scan_one(relay: Relay, qid: str, question: str, key: dict) -> dict:
    s = relay.setting
    r = Req(f"scan-{s.name}-{qid}", question, "scan", 60.0)
    t0 = time.perf_counter()
    route = heuristic_route(question)
    papers = sorted(set(route.papers) | set(detect_papers(question)))
    hits, conf = await relay.retrieve_loop(r, route, papers, s.k, s.rerank)
    tau = S.TAU["rerank" if s.rerank else "cos"]
    parts = relay.plan_parts(question, route, papers)
    if conf < tau["refuse"]:
        branch = "refuse"
    elif parts:
        branch = "decompose"
        hits = await relay.decompose(r, parts, hits, s.k, s.rerank)
    else:
        branch = "answer"
    hits = relay.cap_passages(hits)
    ms = (time.perf_counter() - t0) * 1000
    cites = [h.chunk["cite"] for h in hits]
    prompt = S.CHAT_OVERHEAD + sum(ntok(m["content"]) for m in relay.draft_messages(question, hits))
    req = key.get("required", [])
    reach = [bool(set(p["cites"]) & set(cites)) for p in req] if branch != "refuse" else [False] * len(req)
    return {"setting": s.name, "qid": qid, "answerable": key["answerable"], "branch": branch,
            "confidence": round(conf, 3), "points": len(req), "reachable": sum(reach),
            "missing": "; ".join(p["point"][:60] for p, ok in zip(req, reach) if not ok),
            "prompt_tokens": prompt, "retrieval_ms": round(ms, 1), "cites": " | ".join(cites)}


async def main_async(settings: list[str]) -> None:
    qs = load_questions()
    key = {q["id"]: q for q in json.loads((ROOT / "eval" / "answer_key.json").read_text(encoding="utf-8"))["questions"]}
    rows = []
    for name in settings:
        relay = Relay("stand-in", name, "scan", ledger=Ledger("scan", persist=False), use_cache=False,
                      rows_path=None, trace_path=None, cpu_workers=1)
        for qid, q in qs.items():
            rows.append(await scan_one(relay, qid, q, key[qid]))
        await relay.close()
        print(f"  scanned {name}", flush=True)

    out = ROOT / "results" / "retrieval_scan.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"\n{'setting':<10} {'points_reach':>12} {'all_reach':>9} {'ref_unans':>9} {'ref_ans':>7} "
          f"{'prompt p50/max':>15} {'retr_ms p50/max':>16}")
    for name in settings:
        rs = [r for r in rows if r["setting"] == name]
        ans = [r for r in rs if r["answerable"]]
        una = [r for r in rs if not r["answerable"]]
        pts = sum(r["reachable"] for r in ans) / sum(r["points"] for r in ans)
        allr = sum(r["reachable"] == r["points"] for r in ans)
        pt = sorted(r["prompt_tokens"] for r in rs if r["branch"] != "refuse")
        ms = sorted(r["retrieval_ms"] for r in rs)
        print(f"{name:<10} {pts * 100:>11.1f}% {allr:>5}/{len(ans):<3} {sum(r['branch'] == 'refuse' for r in una):>5}/{len(una):<3}"
              f" {sum(r['branch'] == 'refuse' for r in ans):>3}/{len(ans):<3} {statistics.median(pt):>7.0f}/{pt[-1]:<7}"
              f" {statistics.median(ms):>8.0f}/{ms[-1]:<7.0f}")
    print(f"\nper-question rows -> {out.relative_to(ROOT)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings", nargs="+", default=DEFAULT)
    asyncio.run(main_async(ap.parse_args().settings))


if __name__ == "__main__":
    main()
