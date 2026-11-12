"""Answer quality under load, by answer tier (Full 120b / Lean 20b / cached / quoted passages), from the graded real
load answers. 0 tokens.

  python -m eval.load_quality

Reads results.csv (answer_grade, status, scenario) and results/load_answers.jsonl (the tier of each request, which
results.csv also carries in retrieval_setting: 'h256k8' Full, '...|lean' Lean, 'cache', 'extractive').
"""
from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOAD = ("steady", "burst", "drain", "daily_exhausted")


def tier_of(setting: str) -> str:
    if setting == "cache":
        return "cached"
    if setting == "extractive" or setting.endswith("|extractive"):
        return "quoted passages"
    if setting.endswith("|lean"):
        return "lean (20b)"
    return "full (120b)"


def main() -> None:
    rows = [r for r in csv.DictReader((ROOT / "results.csv").open(encoding="utf-8"))
            if r["source"] == "real" and r["scenario"] in LOAD]
    missing = [r["request_id"] for r in rows if not r["answer_grade"]]
    if missing:
        raise SystemExit(f"{len(missing)} real load rows have no grade; run eval.grades_load_2026-11-12 and rebuild")
    by = defaultdict(Counter)
    for r in rows:
        by[tier_of(r["retrieval_setting"])][r["answer_grade"]] += 1
        by["all real load"][r["answer_grade"]] += 1
    print("Real load answers (steady, burst, drain, daily_exhausted), graded against the frozen key.")
    print("Answerable questions: correct % of answerable; unanswerable: refused correctly % of unanswerable.\n")
    print(f"{'tier':<17} {'n':>4} {'answerable':>10} {'correct':>8} {'wrong':>6} {'ref_wrong':>9} {'correct %':>9} "
          f"{'unanswerable':>12} {'ref_ok %':>8}")
    for t in ("full (120b)", "lean (20b)", "cached", "quoted passages", "all real load"):
        c = by.get(t, Counter())
        ans = c["correct"] + c["wrong"] + c["refused_wrongly"]
        una = c["refused_correctly"] + 0
        n = sum(c.values())
        pc = f"{100 * c['correct'] / ans:.1f}%" if ans else "-"
        pr = f"{100 * c['refused_correctly'] / una:.1f}%" if una else "-"
        print(f"{t:<17} {n:>4} {ans:>10} {c['correct']:>8} {c['wrong']:>6} {c['refused_wrongly']:>9} {pc:>9} "
              f"{una:>12} {pr:>8}")


if __name__ == "__main__":
    main()
