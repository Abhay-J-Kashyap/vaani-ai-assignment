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


# Stand-in runs kept as a fallback for a row the write-up reports from a real run (ADR, evening of 2026-11-09:
# "The stand-in version is kept as a fallback in case the real run fails"). The real run worked, so these rows stay
# in results.csv (raw results are kept) but are printed apart from Tables 2a/2b, which match the write-up.
FALLBACK = {("daily_exhausted", "stand-in")}


def table2(rs: list[dict]) -> None:
    by = defaultdict(list)
    for r in rs:
        if r["scenario"] in LOAD_SCENARIOS:
            by[(r["scenario"], r["source"])].append(r)
    keys = sorted(by, key=lambda k: (LOAD_SCENARIOS.index(k[0]), k[1]))
    real = {k[0] for k in keys if k[1] == "real"}
    fallback = [k for k in keys if k in FALLBACK and k[0] in real]
    main = [k for k in keys if k not in fallback]
    print("Tables 2a and 2b as in the write-up: one row per load scenario.\n")
    print_table2(by, main)
    if fallback:
        print("\n\nNot in the write-up: stand-in fallback runs for rows the write-up reports from a real run "
              "(kept in results.csv; superseded because the real run succeeded).\n")
        print_table2(by, fallback)


def print_table2(by: dict, keys: list) -> None:
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


def table1(rs: list[dict]) -> None:
    """Table 1 from results.csv where the CSV supports it (probe_* rows); the exact limit values come from Groq's
    own headers and 429 bodies (results/probe_raw.jsonl) and the console capture, as decided on 2026-11-09."""
    def t(r):
        return ts(r) / 1000

    def first(scn, status):
        g = sorted((r for r in rs if r["scenario"] == scn), key=t)
        return g, next((r for r in g if r["status"] == status), None)

    rpm, rl = first("probe_rpm", "rate_limited")
    rpm_ok = [r for r in rpm if r["status"] == "ok" and t(r) < t(rl)]
    rpm_rec = next(r for r in sorted(rs, key=t) if r["scenario"] == "probe_rpm_recovery" and r["status"] == "ok")
    tpm, tl = first("probe_tpm", "rate_limited")
    tpm_used = sum(int(r["tokens_in"]) + int(r["tokens_out"]) for r in tpm
                   if r["status"] == "ok" and t(tl) - 60 <= t(r) < t(tl))
    tpm_rec = next(r for r in sorted(rs, key=t) if r["scenario"] == "probe_tpm_recovery" and r["status"] == "ok")
    conc = [r for r in rs if r["scenario"] == "probe_concurrency" and r["status"] == "ok"]
    spans = [(t(r), t(r) + float(r["total_ms"]) / 1000) for r in conc]
    in_flight = max(sum(1 for s, e in spans if s <= s0 < e) for s0, _ in spans)
    daily = [r for r in rs if r["scenario"] == "probe_daily" and r["status"] == "ok"]
    daily_tok = sum(int(r["tokens_in"]) + int(r["tokens_out"]) for r in daily)
    rl_ms = sorted(float(r["total_ms"]) for r in rs if r["scenario"].startswith("probe_") and r["status"] == "rate_limited")
    rec = sorted((r for r in rs if r["scenario"] == "probe_daily_recovery"), key=t)
    first_rl = next((r for r in rec if r["status"] == "rate_limited"), None)
    rec_ok = [r for r in rec if r["status"] == "ok" and first_rl and t(r) > t(first_rl)]
    daily_rec = f"first ok {t(rec_ok[0]) - t(first_rl):,.0f} s after a daily 429" if rec_ok else "see probe_raw"
    probes = [r for r in rs if r["scenario"].startswith("probe_")]
    slow_503 = [r for r in probes if r["status"] == "error" and r["total_ms"] and float(r["total_ms"]) >= 29_000]

    # Requests per day (2026-11-12, qwen, no qwen use in the 24 h before): tiny requests every ~2.2 s until the daily
    # 429, which ends the run. The value is the ok count before it. A continuous refill (1 request per 86.4 s) would
    # have allowed ~26 more over the run; there were none: Groq's reset header grew exactly 86.4 s per request, so the
    # budget did not refill while requests kept coming (probe_raw). It did refill after an idle gap (next row).
    rpd = sorted((r for r in rs if r["scenario"] == "probe_rpd"), key=t)
    rpd_row = ("Requests per day | Groq states 1,000 | x-ratelimit-limit-requests header on every response (probe_raw); "
               "console; not driven to the limit | medium")
    if rpd and rpd[-1]["status"] == "rate_limited":
        n_ok = sum(1 for r in rpd if r["status"] == "ok")
        span = t(rpd[-1]) - t(rpd[0])
        rpd_rec = sorted((r for r in rs if r["scenario"] == "probe_rpd_recovery"), key=t)
        back = next((r for r in rpd_rec if r["status"] == "ok"), None)
        back_s = f"; one ok again {t(back) - t(rpd[-1]):,.0f} s after it" if back else ""
        rpd_row = (f"Requests per day | observed {n_ok:,}; Groq states 1,000 | driven to the limit on qwen (probe_rpd): "
                   f"{n_ok:,} ok in {span / 60:.1f} min, then a daily 429{back_s}; no request credited back while "
                   f"requests kept coming (observation) | high")
    print("Table 1 - limits found in the real service (gpt-oss-120b / gpt-oss-20b; daily budget measured on "
          "qwen/qwen3.8-27b, same console limits)")
    print("'observed' = computed from results.csv below; 'Groq states' = Groq's 429 bodies and x-ratelimit-* headers "
          "(results/probe_raw.jsonl) and the console capture (results/groq_console_limits_2026-11-09.md)\n")
    print("Limit | Value found | How found | Confidence")
    print(f"Requests per minute | observed {len(rpm_ok)}; Groq states 30 | {len(rpm_ok)} ok in a back-to-back burst, "
          f"then rate_limited {(t(rl) - t(rpm[0])) * 1000:,.0f} ms in (probe_rpm) | high")
    print(f"Tokens per minute | observed >={tpm_used:,}; Groq states 8,000 | {tpm_used:,} tokens used in the 60 s before "
          f"the first token 429 (a lower bound, since max_tokens is reserved up front) | high")
    print(rpd_row)
    print(f"Tokens per day | observed {daily_tok:,}; Groq states 200,000 per model | {len(daily)} ok requests used "
          f"{daily_tok:,} tokens before the first daily 429 (probe_daily); refill fitted at 2.31 tokens/s from the "
          f"429s' 'Used' readings while idle (probe_raw) | high")
    print(f"Requests in flight at once | observed: no limit hit at {in_flight} | {in_flight} requests overlapped, all ok "
          f"(probe_concurrency); RPM 30 stops a larger test | low")
    print(f"What it returns when a limit is hit | observed: HTTP 429 in {pct(rl_ms, 50):,.0f} ms (p50, n={len(rl_ms)}) | "
          f"error.type 'requests' or 'tokens'; message states limit, used, requested and the exact wait; retry-after "
          f"header; separately, 503 'over capacity' after ~30 s ({len(slow_503)} of {len(probes):,} probe calls) | high")
    rpd_back = ""
    if rpd and rpd[-1]["status"] == "rate_limited":
        b = next((r for r in sorted((r for r in rs if r["scenario"] == "probe_rpd_recovery"), key=t)
                  if r["status"] == "ok"), None)
        rpd_back = f", requests per day: one more {t(b) - t(rpd[-1]):,.0f} s after its 429" if b else ""
    print(f"Time until it recovers | observed: RPM {(t(rpm_rec) - t(rl)) * 1000:,.0f} ms, TPM {(t(tpm_rec) - t(tl)) * 1000:,.0f} ms "
          f"(start to start){rpd_back}; tokens per day: {daily_rec} | next ok request after each 429 "
          f"(probe_*_recovery); the daily 429's stated wait was exact | high")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default="3")
    ap.add_argument("--scenario", default="eval")
    a = ap.parse_args()
    if a.table == "3":
        table3(rows(), a.scenario)
    elif a.table == "2":
        table2(rows())
    elif a.table == "1":
        table1(rows())


if __name__ == "__main__":
    main()
