"""Recompute the write-up's tables from results.csv and nothing else (the brief: every figure must be computable from it).

  python -m tools.tables [--table 3]

Table 3 (retrieval settings): scenario 'eval' rows, grouped by retrieval_setting.
  quality      share of answers graded correct or refused_correctly (n = graded rows)
  correct      answerable questions graded correct / answerable questions
  tokens/answer  mean tokens_in + tokens_out (router + drafter calls of the request)
  first token  p50 / p95 of first_token_ms, total p50 / p95 of total_ms
Times in ms, percentages to one decimal place; "not measured" where a column has no data.
"""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from relay import settings as S

ROOT = Path(__file__).resolve().parents[1]


def pct(xs: list[float], p: float) -> float | None:
    """Nearest-rank percentile (no interpolation), so any reader gets the same number by sorting the column."""
    if not xs:
        return None
    xs = sorted(xs)
    return xs[max(0, math.ceil(p / 100 * len(xs)) - 1)]


def fmt_ms(v: float | None) -> str:
    return "not measured" if v is None else f"{v:,.0f}"


def rows() -> list[dict]:
    return list(csv.DictReader((ROOT / "results.csv").open(newline="", encoding="utf-8")))


def table3(rs: list[dict], scenario: str = "eval") -> None:
    by = defaultdict(list)
    for r in rs:
        if r["scenario"] == scenario:
            by[r["retrieval_setting"]].append(r)
    print(f"Table 3 - retrieval settings (scenario '{scenario}', source real). Percentages are of questions asked;\n"
          f"'Wrong' includes wrong answers and wrong refusals, so the three outcome columns sum to 100%.\n")
    print("Setting (piece size, pieces fetched, re-rank) | Questions asked | Correct and cited % | Wrong % | "
          "Correctly refused % | Tokens per answer | First token p50 / p95 (ms) | Total time p50 / p95 (ms)")
    for s, g in by.items():
        st = S.parse_setting(s.partition("|")[0])
        graded = [r for r in g if r["answer_grade"]]
        n = len(graded)
        c = sum(r["answer_grade"] == "correct" for r in graded)
        w = sum(r["answer_grade"] in ("wrong", "refused_wrongly") for r in graded)
        rc = sum(r["answer_grade"] == "refused_correctly" for r in graded)
        toks = [int(r["tokens_in"]) + int(r["tokens_out"]) for r in g if r["status"] == "ok"]
        ft = [float(r["first_token_ms"]) for r in g if r["first_token_ms"]]
        tt = [float(r["total_ms"]) for r in g if r["total_ms"]]
        desc = f"{s}: {st.size} tokens, {st.k} pieces, re-rank {'yes' if st.rerank else 'no'}"
        print(" | ".join([
            desc, str(len(g)),
            *(f"{100 * x / n:.1f}%" if n else "not measured" for x in (c, w, rc)),
            f"{sum(toks) / len(toks):,.0f}" if toks else "not measured",
            f"{fmt_ms(pct(ft, 50))} / {fmt_ms(pct(ft, 95))}",
            f"{fmt_ms(pct(tt, 50))} / {fmt_ms(pct(tt, 95))}",
        ]))


LOAD_SCENARIOS = ["steady", "burst", "overload", "tpm_half", "daily_exhausted"]
# Groq list prices, USD per 1M tokens (input, output). The free tier bills nothing; these give "the bill" at scale.
PRICE = {"120b": (0.15, 0.60), "20b": (0.075, 0.30)}


def ts(r: dict) -> float:
    return datetime.fromisoformat(r["start_time"]).timestamp() * 1000


def peak_in_system(g: list[dict]) -> int:
    """Peak queue length, as defined for Table 2b: the most requests in the system at once (waiting or being
    served), from the [start_time, start_time + total_ms] intervals in results.csv."""
    ev = sorted([(ts(r), 1) for r in g] + [(ts(r) + float(r["total_ms"]), -1) for r in g], key=lambda e: (e[0], e[1]))
    cur = peak = 0
    for _, d in ev:
        cur += d
        peak = max(peak, cur)
    return peak


def cost_usd(r: dict) -> float:
    """Upper bound: a Full row's tokens (router + drafter) at the 120b price, any other tier at the 20b price.
    The router's ~0.5K tokens per request really cost the 20b price, so Full rows are overstated by < 3%."""
    suffix = r["retrieval_setting"].partition("|")[2]
    pin, pout = PRICE["120b" if not suffix and r["retrieval_setting"] != "cache" else "20b"]
    return (int(r["tokens_in"] or 0) * pin + int(r["tokens_out"] or 0) * pout) / 1e6


def table2(rs: list[dict]) -> None:
    by = defaultdict(list)
    for r in rs:
        if r["scenario"] in LOAD_SCENARIOS:
            by[(r["scenario"], r["source"])].append(r)
    keys = sorted(by, key=lambda k: (LOAD_SCENARIOS.index(k[0]), k[1]))
    print("Table 2a - latency (nearest-rank percentiles; first token over requests that got one, total over all)\n")
    print("Scenario | Real or stand-in | Requests sent | Duration (s) | Offered rate (per min) | "
          "First token p50 / p95 / p99 (ms) | Total time p50 / p95 / p99 (ms)")
    for k in keys:
        g = by[k]
        starts = sorted(ts(r) for r in g)
        end = max(ts(r) + float(r["total_ms"]) for r in g)
        span = (starts[-1] - starts[0]) / 1000
        ft = [float(r["first_token_ms"]) for r in g if r["first_token_ms"]]
        tt = [float(r["total_ms"]) for r in g]
        flag = " (p99 from n<100)" if len(tt) < 100 else ""
        print(f"{k[0]} | {k[1]} | {len(g)} | {(end - starts[0]) / 1000:,.0f} | "
              f"{len(g) / span * 60 if span else float('nan'):.1f} | "
              f"{' / '.join(fmt_ms(pct(ft, p)) for p in (50, 95, 99))} (n={len(ft)}) | "
              f"{' / '.join(fmt_ms(pct(tt, p)) for p in (50, 95, 99))} (n={len(tt)}){flag}")
    print("\nTable 2b - outcomes (an answer = a request that got any answer text; error % is the remainder)\n")
    print("Scenario | Succeeded % | Rate-limited % | Timed out or given up % | Retries per request | "
          "Peak queue length | Tokens per answer | Cost per 1,000 answers (USD, upper bound)")
    for k in keys:
        g = by[k]
        n = len(g)
        share = {s: 100 * sum(r["status"] == s for r in g) / n for s in ("ok", "rate_limited", "timeout", "error")}
        answered = [r for r in g if r["first_token_ms"]]
        toks = [int(r["tokens_in"] or 0) + int(r["tokens_out"] or 0) for r in answered]
        cost = sum(cost_usd(r) for r in answered)
        print(f"{k[0]} ({k[1]}) | {share['ok']:.1f}% | {share['rate_limited']:.1f}% | {share['timeout']:.1f}% | "
              f"{sum(int(r['retries'] or 0) for r in g) / n:.2f} | {peak_in_system(g)} | "
              f"{sum(toks) / len(toks):,.0f} | "
              f"{1000 * cost / len(answered):.3f}" + (f" | error {share['error']:.1f}%" if share["error"] else ""))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default="3")
    ap.add_argument("--scenario", default="eval")
    a = ap.parse_args()
    if a.table == "3":
        table3(rows(), a.scenario)
    elif a.table == "2":
        table2(rows())


if __name__ == "__main__":
    main()
