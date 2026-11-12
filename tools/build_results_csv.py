"""Build results.csv (the brief's raw file) from the raw logs. Exactly the 12 columns the brief specifies.

  python -m tools.build_results_csv            # writes results.csv at the repo root
  python -m tools.build_results_csv --check    # also recomputes the Table 1 figures the CSV can support

Sources, all kept unchanged (results.csv is derived, so rebuilding it is safe):
  results/probe_raw.jsonl        every real Groq probe request -> one row each, scenario probe_*
  results/pipeline_rows*.csv     pipeline requests (real and stand-in), written by the pipeline with the same header
  results/grades_*.csv           answer_grade per graded request_id (graded against eval/answer_key.json)

Probe rows have no answer, so answer_grade is left empty and retrieval_setting is 'none' (decision recorded in the
ADR, 2026-11-09). What a 429 says and the exact limit values live in the raw headers/bodies in probe_raw.jsonl.
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "probe_raw.jsonl"
OUT = ROOT / "results.csv"
COLUMNS = ["request_id", "scenario", "source", "start_time", "first_token_ms", "total_ms", "status", "retries",
           "tokens_in", "tokens_out", "retrieval_setting", "answer_grade"]

SCENARIO = {
    "headers": "probe_headers", "latency": "probe_latency",
    "rpm_burst": "probe_rpm", "rpm_recovery": "probe_rpm_recovery",
    "tpm_burst": "probe_tpm", "tpm_recovery": "probe_tpm_recovery",
    "concurrency": "probe_concurrency",
    "daily_tokens": "probe_daily", "daily_requests": "probe_rpd",
    "daily_drain": "probe_daily_drain", "daily_watch": "probe_daily_recovery",
    "daily_check_before": "probe_daily_check", "daily_check_after": "probe_daily_check",
}


def status_of(r: dict) -> str:
    code, exc = r.get("status_code"), r.get("exception") or ""
    if exc.startswith("timeout"):
        return "timeout"
    if code == 200 and not r.get("stream_error"):
        return "ok"
    if code == 429:
        return "rate_limited"
    return "error"


def probe_rows() -> list[dict]:
    rows = []
    raws = [json.loads(line) for line in RAW.read_text(encoding="utf-8").splitlines() if line.strip()]
    # The requests-per-day run (2026-11-12) has its own watch phase; keep it apart from the Nov 9 token test's.
    rpd_runs = {r.get("run_id") for r in raws if r["phase"] == "daily_requests"}
    for r in raws:
        if r.get("dry_run"):
            continue
        u = r.get("usage") or {}
        st = status_of(r)
        scenario = SCENARIO.get(r["phase"], f"probe_{r['phase']}")
        if r["phase"] == "daily_watch" and r.get("run_id") in rpd_runs:
            scenario = "probe_rpd_recovery"
        rows.append({
            "request_id": f"probe-{r['probe_id']}",
            "scenario": scenario,
            "source": "real",
            "start_time": r["start_time"],
            # first answer token; empty when the call wasn't streamed or produced no answer text
            "first_token_ms": r.get("first_content_ms") if r.get("first_content_ms") is not None else "",
            "total_ms": r.get("total_ms", ""),
            "status": st,
            "retries": 0,  # probes never retry; a recovery poll is its own request and its own row
            # rejected calls are not processed or charged, so their token cells are left empty
            "tokens_in": u.get("prompt_tokens", "") if st == "ok" else "",
            "tokens_out": u.get("completion_tokens", "") if st == "ok" else "",
            "retrieval_setting": "none",
            "answer_grade": "",
        })
    return rows


def pipeline_rows() -> list[dict]:
    rows = []
    for f in sorted((ROOT / "results").glob("pipeline_rows*.csv")):
        with f.open(newline="", encoding="utf-8") as fh:
            rd = csv.DictReader(fh)
            if rd.fieldnames != COLUMNS:
                raise SystemExit(f"{f.name}: header must be exactly {COLUMNS}, got {rd.fieldnames}")
            rows.extend(rd)
    return rows


def check(rows: list[dict]) -> None:
    """Recompute the Table 1 figures that the CSV alone supports, the way a reviewer would."""
    def t(row):
        return datetime.fromisoformat(row["start_time"]).timestamp()

    rpm = sorted((r for r in rows if r["scenario"] == "probe_rpm"), key=t)
    first_rl = next((r for r in rpm if r["status"] == "rate_limited"), None)
    if first_rl:
        ok_before = [r for r in rpm if r["status"] == "ok" and t(r) < t(first_rl)]
        span = t(first_rl) - t(rpm[0])
        print(f"RPM: {len(ok_before)} ok requests, then rate_limited at {span * 1000:.0f} ms into the burst")
        rec = next((r for r in rows if r["scenario"] == "probe_rpm_recovery" and r["status"] == "ok"), None)
        if rec:
            print(f"     recovered: next ok request started {(t(rec) - t(first_rl)) * 1000:.0f} ms after the 429")
    tpm = sorted((r for r in rows if r["scenario"] == "probe_tpm"), key=t)
    first_rl = next((r for r in tpm if r["status"] == "rate_limited"), None)
    if first_rl:
        ok = [r for r in tpm if r["status"] == "ok" and t(first_rl) - 60 <= t(r) < t(first_rl)]
        used = sum(int(r["tokens_in"]) + int(r["tokens_out"]) for r in ok)
        print(f"TPM: {used:,} tokens accepted in the 60 s before the first token 429 (a lower bound on TPM)")
        rec = next((r for r in rows if r["scenario"] == "probe_tpm_recovery" and r["status"] == "ok"), None)
        if rec:
            print(f"     recovered: next ok request started {(t(rec) - t(first_rl)) * 1000:.0f} ms after the 429")
    conc = [r for r in rows if r["scenario"] == "probe_concurrency" and r["status"] == "ok"]
    if conc:
        spans = [(t(r), t(r) + float(r["total_ms"]) / 1000) for r in conc]
        peak = max(sum(1 for s, e in spans if s <= s0 < e) for s0, _ in spans)
        print(f"In flight at once: {peak} requests overlapped, all ok ({len(conc)} concurrency rows)")
    daily = [r for r in rows if r["scenario"] == "probe_daily"]
    if daily:
        ok = [r for r in daily if r["status"] == "ok"]
        print(f"Daily: {len(ok)} ok rows, {sum(int(r['tokens_in']) + int(r['tokens_out']) for r in ok):,} tokens "
              f"before the first rate_limited row")
    else:
        print("Daily: no probe_daily rows yet (night test pending)")


GRADES = {"correct", "wrong", "refused_correctly", "refused_wrongly"}


def apply_grades(rows: list[dict]) -> int:
    """answer_grade comes only from results/grades_*.csv (request_id, answer_grade, ...), written by grading against
    eval/answer_key.json. A grade for an unknown request or with an unknown value is an error, not skipped."""
    by_id = {r["request_id"]: r for r in rows}
    n = 0
    for f in sorted((ROOT / "results").glob("grades_*.csv")):
        for g in csv.DictReader(f.open(newline="", encoding="utf-8")):
            if g["answer_grade"] not in GRADES:
                raise SystemExit(f"{f.name}: bad answer_grade {g['answer_grade']!r} for {g['request_id']}")
            if g["request_id"] not in by_id:
                raise SystemExit(f"{f.name}: grade for unknown request_id {g['request_id']}")
            by_id[g["request_id"]]["answer_grade"] = g["answer_grade"]
            n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    rows = probe_rows() + pipeline_rows()
    graded = apply_grades(rows)
    if graded:
        print(f"{graded} grades applied from results/grades_*.csv")
    rows.sort(key=lambda r: r["start_time"])
    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    by = {}
    for r in rows:
        by[(r["source"], r["scenario"], r["status"])] = by.get((r["source"], r["scenario"], r["status"]), 0) + 1
    print(f"{len(rows)} rows -> {OUT.relative_to(ROOT)}")
    for k, n in sorted(by.items()):
        print(f"  {k[0]:<8} {k[1]:<24} {k[2]:<13} {n}")
    if a.check:
        print()
        check(rows)


if __name__ == "__main__":
    main()
