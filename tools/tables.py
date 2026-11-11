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
from pathlib import Path

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
    print(f"Table 3 - retrieval settings (scenario '{scenario}', source real; n per setting shown)\n")
    head = ["Setting", "n", "Good %", "Correct (answerable)", "Refused correctly", "Tokens/answer (in+out)",
            "First token p50 / p95 ms", "Total p50 / p95 ms"]
    print(" | ".join(head))
    for s, g in by.items():
        graded = [r for r in g if r["answer_grade"]]
        good = sum(r["answer_grade"] in ("correct", "refused_correctly") for r in graded)
        # A refusal grade says which kind of question it was; 'wrong' can be either, so answerable = graded minus
        # unanswerable, where unanswerable rows are those graded refused_correctly or wrong-on-a-refusal-question.
        # In this eval every unanswerable question was refused, so unanswerable = refused_correctly rows.
        una = [r for r in graded if r["answer_grade"] == "refused_correctly"]
        ans = [r for r in graded if r not in una]
        toks = [int(r["tokens_in"]) + int(r["tokens_out"]) for r in g if r["status"] == "ok"]
        ft = [float(r["first_token_ms"]) for r in g if r["first_token_ms"]]
        tt = [float(r["total_ms"]) for r in g if r["total_ms"]]
        print(" | ".join([
            s, str(len(g)),
            f"{100 * good / len(graded):.1f}%" if graded else "not measured",
            f"{sum(r['answer_grade'] == 'correct' for r in ans)}/{len(ans)}",
            f"{len(una)}/{len(graded) - len(ans)}",
            f"{sum(toks) / len(toks):,.0f}" if toks else "not measured",
            f"{fmt_ms(pct(ft, 50))} / {fmt_ms(pct(ft, 95))}",
            f"{fmt_ms(pct(tt, 50))} / {fmt_ms(pct(tt, 95))}",
        ]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default="3")
    ap.add_argument("--scenario", default="eval")
    a = ap.parse_args()
    if a.table == "3":
        table3(rows(), a.scenario)


if __name__ == "__main__":
    main()
