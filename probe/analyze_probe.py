"""Summarise results/probe_raw.jsonl into the numbers for Table 1 and the stand-in calibration.

  python -m probe.analyze_probe              # every model and phase
  python -m probe.analyze_probe --model openai/gpt-oss-120b

Everything printed here comes from the raw file only, so a reviewer can rerun it.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "probe_raw.jsonl"


def pct(xs: list[float], p: float) -> float | None:
    """Nearest-rank percentile, so a reviewer can recompute it by hand from a sorted column."""
    if not xs:
        return None
    xs = sorted(xs)
    return xs[max(0, math.ceil(p / 100 * len(xs)) - 1)]


def fmt(xs: list[float]) -> str:
    if not xs:
        return "n=0"
    return (f"n={len(xs)} p50={pct(xs, 50):.0f} p95={pct(xs, 95):.0f} p99={pct(xs, 99):.0f} "
            f"min={min(xs):.0f} max={max(xs):.0f}")


def lognormal_fit(xs: list[float]) -> str:
    """Mean and standard deviation of ln(x), used to parameterise the stand-in's latency sampler."""
    xs = [x for x in xs if x and x > 0]
    if len(xs) < 3:
        return "too few samples"
    logs = [math.log(x) for x in xs]
    return f"mu={st.mean(logs):.3f} sigma={st.pstdev(logs):.3f} (ms; median≈{math.exp(st.mean(logs)):.0f})"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    a = ap.parse_args()
    recs = [json.loads(line) for line in RAW.read_text(encoding="utf-8").splitlines() if line.strip()]
    recs = [r for r in recs if not r.get("dry_run") and (not a.model or r["model"] == a.model)]
    print(f"{len(recs)} raw records from {RAW.relative_to(ROOT)}\n")

    # ---- limits as reported by the headers
    print("== Limits from response headers (latest record per model)")
    latest = {}
    for r in recs:
        if (r.get("headers") or {}).get("x-ratelimit-limit-requests"):
            latest[r["model"]] = r
    for m, r in sorted(latest.items()):
        h = r["headers"]
        print(f"  {m:<28} RPD={h['x-ratelimit-limit-requests']:>6} TPM={h['x-ratelimit-limit-tokens']:>6} "
              f"region={h.get('x-groq-region')}")

    # ---- refill model: is the daily bucket continuous? reset-requests should equal used/RPD*86400
    print("\n== Daily refill check: reset-requests vs (RPD-remaining) * 86400/RPD")
    for r in recs:
        h = r.get("headers") or {}
        if not h.get("x-ratelimit-reset-requests"):
            continue
        from probe.groq_probe import parse_duration_s
        lim, rem = int(h["x-ratelimit-limit-requests"]), int(h["x-ratelimit-remaining-requests"])
        predicted = (lim - rem) * 86400 / lim
        observed = parse_duration_s(h["x-ratelimit-reset-requests"])
        if r["phase"] in ("headers",) or r.get("seq", 0) % 15 == 0:
            print(f"  {r['model']:<28} used={lim - rem:>4} observed={observed:9.1f}s predicted={predicted:9.1f}s")

    # ---- latency by model and size (phase 'latency' only, successful streamed calls)
    print("\n== Latency (phase=latency, status 200). Times in ms, client-measured")
    groups = defaultdict(list)
    for r in recs:
        if r["phase"] == "latency" and r.get("status_code") == 200:
            groups[(r["model"], r["size"])].append(r)
    for (m, size), rs in sorted(groups.items()):
        ttft = [r["first_content_ms"] for r in rs if r.get("first_content_ms")]
        first_any = [min(x for x in (r.get("first_reasoning_ms"), r.get("first_content_ms")) if x) for r in rs
                     if r.get("first_reasoning_ms") or r.get("first_content_ms")]
        total = [r["total_ms"] for r in rs]
        u = [r.get("usage") or {} for r in rs]
        queue = [x["queue_time"] * 1000 for x in u if x.get("queue_time") is not None]
        server = [x["total_time"] * 1000 for x in u if x.get("total_time") is not None]
        tps = [x["completion_tokens"] / x["completion_time"] for x in u
               if x.get("completion_time") and x.get("completion_tokens")]
        reasoning = [(x.get("completion_tokens_details") or {}).get("reasoning_tokens", 0) for x in u]
        p_in = [x.get("prompt_tokens") for x in u if x.get("prompt_tokens")]
        finish = Counter(r.get("finish_reason") for r in rs)
        print(f"  {m} | {size} | prompt tokens p50={pct(p_in, 50)}")
        print(f"     first answer token  {fmt(ttft)}")
        print(f"     first any token     {fmt(first_any)}")
        print(f"     total               {fmt(total)}")
        print(f"     groq queue_time     {fmt(queue)}")
        print(f"     groq server time    {fmt(server)}")
        print(f"     output tokens/s     {fmt(tps)}")
        print(f"     reasoning tokens    {fmt(reasoning)}   finish={dict(finish)}")
        print(f"     lognormal(first answer token) {lognormal_fit(ttft)}")
        gaps = [r for r in rs if r.get("max_event_gap_ms") is not None]
        if gaps:
            stalls = [r for r in gaps if r["max_event_gap_ms"] >= 1000]
            after_first = [r for r in stalls if r.get("first_content_ms")
                           and r["max_event_gap_after_ms"] >= r["first_content_ms"]]
            print(f"     longest SSE gap     {fmt([r['max_event_gap_ms'] for r in gaps])}")
            print(f"     stalls >=1s: {len(stalls)}/{len(gaps)} ({100 * len(stalls) / len(gaps):.1f}%), "
                  f"of which after first answer token: {len(after_first)}")

    # ---- every non-200: what the service returns at a limit
    print("\n== Non-200 responses")
    seen = set()
    for r in recs:
        if r.get("status_code") in (200, None):
            continue
        h = r.get("headers") or {}
        body = json.dumps(r.get("error_body"))[:300]
        key = (r["model"], r["status_code"], body[:120])
        if key in seen:
            continue
        seen.add(key)
        print(f"  [{r['phase']}] {r['model']} status={r['status_code']} retry-after={h.get('retry-after')} "
              f"limit={r.get('limit_hit')}\n     body={body}")
    exc = Counter(r.get("exception") for r in recs if r.get("exception"))
    if exc:
        print("  client exceptions:", dict(exc))

    # ---- recovery after a limit
    print("\n== Recovery after the first 429")
    for r in recs:
        if r["phase"].endswith("_recovery") and r.get("status_code") == 200:
            print(f"  [{r['phase']}] {r['model']} recovered {r['since_first_429_ms']} ms after first 429 (poll {r['poll']})")

    # ---- daily runs: exhaustion, reserved-vs-used check, drain, and the refill rate from 429 readings
    for run in sorted({r["run_id"] for r in recs if r["phase"].startswith("daily_")}):
        rs = [r for r in recs if r["run_id"] == run]
        ex = [r for r in rs if r["phase"] in ("daily_tokens", "daily_requests")]
        if not ex:
            continue
        ok = [r for r in ex if r.get("status_code") == 200]
        hit = next((r for r in ex if r.get("limit_hit") == "daily"), None)
        actual = sum((r.get("usage") or {}).get("total_tokens", 0) for r in ok)
        reserved = sum((r.get("usage") or {}).get("prompt_tokens", 0) + r["max_tokens"] for r in ok)
        print(f"\n== Daily run {run} on {ex[0]['model']}")
        print(f"  exhaust: {len(ok)} successful requests; tokens actually used={actual:,}; "
              f"reserved (prompt + max_tokens)={reserved:,}")
        void = [r for r in ok if r.get("task") == "ok"
                and (r.get("usage") or {}).get("completion_tokens", 0) >= 0.9 * r["max_tokens"]]
        tests = [r for r in ok if r.get("task") == "ok"]
        if tests:
            print(f"  reserve-test requests: {len(tests)}, of which {len(void)} filled max_tokens "
                  f"(those don't separate reserved from used)")
        watch = [r for r in rs if r["phase"] in ("daily_drain", "daily_watch")]
        pts = [(r["since_first_daily_429_ms"] / 1000, r["limit_used"]) for r in watch
               if r.get("limit_hit") == "daily" and r.get("limit_used") is not None and r.get("kind") == "large"]
        rate = None
        if len(pts) >= 3:
            mx, my = st.mean(x for x, _ in pts), st.mean(y for _, y in pts)
            sxx = sum((x - mx) ** 2 for x, _ in pts)
            if sxx:
                rate = -sum((x - mx) * (y - my) for x, y in pts) / sxx
        if hit:
            mins = hit.get("since_daily_start_ms", 0) / 60000
            print(f"  first daily 429 after {mins:.1f} min: Limit={hit.get('limit_limit')} Used={hit.get('limit_used')} "
                  f"Requested={hit.get('limit_requested')} stated wait={hit.get('stated_wait_s')}s")
            print(f"     message: {json.dumps(hit.get('error_body'))[:400]}")
            if hit.get("limit_used") is not None and hit.get("limit_limit"):
                r_ = rate if rate is not None else hit["limit_limit"] / 86400
                refilled = r_ * mins * 60
                print(f"  Used at the 429 vs our totals, after subtracting ~{refilled:,.0f} tokens refilled during the run "
                      f"({'fitted' if rate is not None else 'assumed'} {r_:.2f} tokens/s):")
                print(f"     if charged actual use: expect Used ≈ {actual - refilled:,.0f}")
                print(f"     if charged max_tokens: expect Used ≈ {reserved - refilled:,.0f}")
        for ph in ("daily_check_before", "daily_check_after"):
            for r in (x for x in rs if x["phase"] == ph):
                h = r.get("headers") or {}
                print(f"  [{ph}] {r['model']} status={r.get('status_code')} "
                      f"remaining-requests={h.get('x-ratelimit-remaining-requests')} limit={r.get('limit_hit')}")
        drained = [r for r in watch if r["phase"] == "daily_drain"]
        if drained:
            print(f"  drain: {sum(1 for r in drained if r.get('status_code') == 200)} tiny requests still got through "
                  f"after the first daily 429")
        for kind in ("tiny", "large"):
            first = next((r for r in watch if r["phase"] == "daily_watch" and r.get("kind") == kind
                          and r.get("status_code") == 200), None)
            when = (f"{first['since_first_daily_429_ms'] / 1000:.0f} s after the first daily 429"
                    if first else "none")
            print(f"  first {kind} success in the watch: {when}")
        if rate is not None:
            print(f"  refill rate fitted from {len(pts)} 'Used' readings on refused large polls: {rate:.2f} tokens/s "
                  f"(× 86,400 = {rate * 86400:,.0f} tokens/day)")
        stated = [(r["since_first_daily_429_ms"] / 1000, r["stated_wait_s"]) for r in watch
                  if r.get("kind") == "large" and r.get("stated_wait_s") is not None]
        if stated:
            print("  stated waits on large polls (s after first 429 → stated wait s): "
                  + ", ".join(f"{t:.0f}→{w:.0f}" for t, w in stated[:12]) + (" …" if len(stated) > 12 else ""))

    # ---- concurrency
    conc = defaultdict(list)
    for r in recs:
        if r["phase"] == "concurrency":
            conc[(r["model"], r["level"])].append(r)
    if conc:
        print("\n== Concurrency")
        for (m, lvl), rs in sorted(conc.items()):
            ok = [r for r in rs if r.get("status_code") == 200]
            print(f"  {m} level={lvl:>2} ok={len(ok)}/{len(rs)} statuses={dict(Counter(r.get('status_code') for r in rs))} "
                  f"first answer token {fmt([r['first_content_ms'] for r in ok if r.get('first_content_ms')])}")


if __name__ == "__main__":
    main()
