"""Answer the 20 gold questions on real Groq under each retrieval setting (Table 3).

  python -m eval.run_eval --dry-run                      # projected tokens per model vs the ledger; no Groq
  python -m eval.run_eval --confirm-real                 # spends daily budget (author approved 2026-11-11)
  python -m eval.run_eval --confirm-real --questions Q1 Q12 --settings h512k4

Rules (author's decisions):
  * Full answers only: no tier downgrade, no answer cache, no extractive fallback (Budget plan (d)). A request
    the budget can't serve waits (patience 180 s); if it still can't, its row says rate_limited and it is retried
    on the next run (resume skips (setting, question) pairs already answered ok).
  * Questions are interleaved across settings (Q1 under each setting, then Q2, ...), so Groq's time-of-day latency
    drift doesn't land on one setting.
  * One shared ledger for all settings, persisted, so admission sees every call.
Rows: results/pipeline_rows_eval.csv (scenario 'eval', retrieval_setting = the setting). Answers for grading:
results/eval_answers.jsonl (append-only), including the sentences the gate withheld (the ungated Option D answer).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from eval.retrieval_scan import load_questions
from relay import settings as S
from relay.ledger import Ledger
from relay.pipeline import Relay, log_paths

ROOT = Path(__file__).resolve().parents[1]
ANSWERS = ROOT / "results" / "eval_answers.jsonl"
SCAN = ROOT / "results" / "retrieval_scan.csv"
SETTINGS = ["h512k4", "h256k8", "h512k4rr"]
PATIENCE_S = 180.0
ROUTER_COST, COMPLETION_EST = 480, 450        # measured on real_smoke: router ~400 in + ~80 out; draft out 117-652
SLACK = 0.15                                  # keep 15% of the day's budget for reruns (Budget plan)


def done_pairs(scenario: str) -> set[tuple[str, str]]:
    if not ANSWERS.exists():
        return set()
    out = set()
    for line in ANSWERS.read_text(encoding="utf-8").splitlines():
        a = json.loads(line)
        if a.get("scenario") == scenario and a["status"] == "ok" and a["sentences"]:
            out.add((a["setting"], a["qid"]))
    return out


def scan_prompts() -> dict[tuple[str, str], int]:
    import csv
    return {(r["setting"], r["qid"]): int(r["prompt_tokens"]) for r in csv.DictReader(SCAN.open(encoding="utf-8"))}


async def wait_for_budget(ledger: Ledger, prompt: int) -> float:
    """Sleep until both the router and a drafter call of about this size would be admitted at once. The margin
    covers the gap between the scan's prompt estimate and the real prompt (the LLM router's query differs)."""
    t = time.perf_counter()
    while True:
        w = max(ledger.wait_s(S.FULL_MODEL, prompt + 400, S.FULL_MAX_TOKENS)[0],
                ledger.wait_s(S.ROUTER_MODEL, ROUTER_COST, S.ROUTER_MAX_TOKENS)[0])
        if w <= 0:
            return time.perf_counter() - t
        await asyncio.sleep(w + 0.05)


def projected(settings: list[str], pairs: list[tuple[str, str]]) -> dict[str, int]:
    prompt = scan_prompts()
    drafter = sum(prompt.get(p, 2500) + COMPLETION_EST for p in pairs)
    return {S.FULL_MODEL: drafter, S.ROUTER_MODEL: ROUTER_COST * len(pairs)}


def withheld_sentences(rid: str) -> list[str]:
    trace = ROOT / "results" / "trace.jsonl"
    out = []
    with trace.open(encoding="utf-8") as f:
        for line in f:
            if rid in line:
                e = json.loads(line)
                if e["rid"] == rid and e["ev"] == "gate_fail":
                    out.append(e.get("text", ""))
    return out


async def main_async(a) -> None:
    qs = load_questions()
    qids = a.questions or list(qs)
    prompt_est = scan_prompts()
    have = done_pairs(a.scenario)
    pairs = [(s, q) for q in qids for s in a.settings if (s, q) not in have]
    ledger = Ledger("real")
    need = projected(a.settings, pairs)
    print(f"{len(pairs)} answers to run ({len(have)} already answered ok), settings={a.settings}")
    ok = True
    for m, n in need.items():
        lvl = ledger.acct(m).tpd.level()
        room = lvl - SLACK * ledger.acct(m).tpd.capacity
        print(f"  {m}: projected ~{n:,} tokens; ledger level ~{lvl:,.0f}; room after 15% slack ~{room:,.0f}")
        ok &= n <= room
    if a.dry_run:
        return
    if not ok and not a.force:
        raise SystemExit("projected spend exceeds today's room; wait for refill or pass --force")

    rows, trace = log_paths("real", a.scenario)
    relays = {s: Relay("real", s, a.scenario, ledger=ledger, use_cache=False, allow_downgrade=False,
                       rows_path=rows, trace_path=trace) for s in a.settings}
    stamp = datetime.now(timezone.utc).strftime("%m%d%H%M")
    t0 = time.perf_counter()
    try:
        for i, (s, qid) in enumerate(pairs, 1):
            # Wait for per-minute budget BEFORE the request starts, so its first-token time measures the pipeline
            # and not this runner's own pacing (the first eval run, 2026-11-11, sent back to back and recorded
            # 10-25 s first tokens that were almost all admission wait).
            waited = await wait_for_budget(ledger, prompt_est.get((s, qid), 2500))
            rid = f"{a.scenario}-{s}-{qid}-{stamp}"
            res = await relays[s].ask(qs[qid], request_id=rid, patience_s=PATIENCE_S)
            rec = {"scenario": a.scenario, "setting": s, "qid": qid, "request_id": rid, "question": qs[qid],
                   "pre_wait_s": round(waited, 2),
                   "sentences": res["sentences"], "notices": res["notices"],
                   "withheld_sentences": withheld_sentences(rid)} | {k: res[k] for k in (
                       "status", "branch", "first_token_ms", "total_ms", "tokens_in", "tokens_out", "withheld",
                       "drafter_calls", "retries")}
            with ANSWERS.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"[{i:>2}/{len(pairs)} {time.perf_counter() - t0:6.0f}s pre-wait {waited:4.1f}s] {s:<9} {qid:<4} "
                  f"{res['status']:<12} "
                  f"{res['branch']:<9} ft={res['first_token_ms']} total={res['total_ms']} "
                  f"tok={res['tokens_in']}/{res['tokens_out']} withheld={res['withheld']} calls={res['drafter_calls']}",
                  flush=True)
    finally:
        for r in relays.values():
            await r.close()
        print("ledger now:", {m: round(ledger.acct(m).tpd.level()) for m in (S.FULL_MODEL, S.ROUTER_MODEL)})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings", nargs="+", default=SETTINGS)
    ap.add_argument("--questions", nargs="+")
    ap.add_argument("--scenario", default="eval", help="results.csv scenario; also names the rows file")    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--confirm-real", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if not (a.dry_run or a.confirm_real):
        raise SystemExit("spends real budget: --dry-run first, then --confirm-real")
    asyncio.run(main_async(a))


if __name__ == "__main__":
    main()
