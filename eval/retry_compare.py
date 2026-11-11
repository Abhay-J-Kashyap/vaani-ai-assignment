"""Heading 6: naive retries vs the ADR retry policy, same arrivals/questions/patience (seed), from results.csv rows
plus the trace for what each drafter call did. No Groq.

  python -m eval.retry_compare [retry_naive retry_policy]
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

from tools.tables import pct

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    scs = sys.argv[1:] or ["retry_naive", "retry_policy"]
    rows = [r for r in csv.DictReader((ROOT / "results.csv").open(encoding="utf-8")) if r["scenario"] in scs]
    rid_sc = {r["request_id"]: r["scenario"] for r in rows}
    calls = {s: Counter() for s in scs}
    tok_failed = Counter()
    with (ROOT / "results" / "trace.jsonl").open(encoding="utf-8") as f:
        for line in f:
            e = json.loads(line)
            s = rid_sc.get(e["rid"])
            if not s:
                continue
            if e["ev"] == "call_start" and "120b" in e["model"]:
                calls[s]["drafter attempts"] += 1
                calls[s]["drafter tokens reserved"] += e["prompt_est"] + e["max_tokens"]
            if e["ev"] == "call_end" and "120b" in e["model"]:
                calls[s][f"drafter {e['status']}"] += 1
                if e["status"] != "ok":
                    tok_failed[s] += 1
            if e["ev"] == "retry_wait_start":
                calls[s]["retries"] += 1
    for s in scs:
        g = [r for r in rows if r["scenario"] == s]
        tiers = Counter(r["retrieval_setting"].partition("|")[2] or "full" for r in g)
        tt = [float(r["total_ms"]) for r in g]
        ft = [float(r["first_token_ms"]) for r in g if r["first_token_ms"]]
        st = Counter(r["status"] for r in g)
        print(f"{s}: n={len(g)} status={dict(st)} answers by tier={dict(tiers)}")
        print(f"   first token p50/p95 {pct(ft, 50):,.0f} / {pct(ft, 95):,.0f} ms; total p50/p95 {pct(tt, 50):,.0f} / "
              f"{pct(tt, 95):,.0f} ms; retries/request {sum(int(r['retries']) for r in g) / len(g):.2f}")
        print(f"   drafter calls: {dict(calls[s])}")


if __name__ == "__main__":
    main()
