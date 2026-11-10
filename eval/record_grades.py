"""Check a grades file against the eval answers, print Table 3's quality columns, and record point coverage.

  python -m eval.record_grades results/grades_eval.csv

Grades file columns: request_id, setting, qid, answer_grade, points_covered, points_required, notes.
  * every graded request_id must exist in results/eval_answers.jsonl, and an unanswerable question can only be
    refused_correctly or wrong; an answerable one correct, wrong or refused_wrongly (the grading rule);
  * point coverage is a diagnostic, not a grade (author, 2026-11-10): it is appended to results/trace.jsonl as a
    'grade' event, once per request_id, and never written to results.csv.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRACE = ROOT / "results" / "trace.jsonl"


def main() -> None:
    path = Path(sys.argv[1])
    key = {q["id"]: q for q in json.loads((ROOT / "eval" / "answer_key.json").read_text(encoding="utf-8"))["questions"]}
    answers = {}
    for line in (ROOT / "results" / "eval_answers.jsonl").read_text(encoding="utf-8").splitlines():
        a = json.loads(line)
        answers[a["request_id"]] = a
    grades = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    for g in grades:
        a = answers.get(g["request_id"])
        if not a:
            raise SystemExit(f"unknown request_id {g['request_id']}")
        allowed = {"correct", "wrong", "refused_wrongly"} if key[a["qid"]]["answerable"] else {"refused_correctly", "wrong"}
        if g["answer_grade"] not in allowed:
            raise SystemExit(f"{g['request_id']} ({a['qid']}): {g['answer_grade']} not allowed for this question")

    have = set()
    if TRACE.exists():
        for line in TRACE.read_text(encoding="utf-8").splitlines():
            if '"ev": "grade"' in line:
                have.add(json.loads(line)["rid"])
    with TRACE.open("a", encoding="utf-8") as f:
        for g in grades:
            if g["request_id"] not in have and g["points_required"]:
                f.write(json.dumps({"rid": g["request_id"], "ev": "grade", "grade": g["answer_grade"],
                                    "points_covered": int(g["points_covered"]),
                                    "points_required": int(g["points_required"])}) + "\n")

    by = defaultdict(lambda: defaultdict(int))
    cov = defaultdict(lambda: [0, 0])
    for g in grades:
        s = g["setting"]
        by[s][g["answer_grade"]] += 1
        by[s]["n"] += 1
        if g["points_required"]:
            cov[s][0] += int(g["points_covered"])
            cov[s][1] += int(g["points_required"])
    print(f"{'setting':<10} {'n':>3} {'correct':>8} {'wrong':>6} {'ref_ok':>7} {'ref_wrong':>9} {'good %':>7} {'points (trace)':>15}")
    for s, d in by.items():
        good = d["correct"] + d["refused_correctly"]
        print(f"{s:<10} {d['n']:>3} {d['correct']:>8} {d['wrong']:>6} {d['refused_correctly']:>7} "
              f"{d['refused_wrongly']:>9} {100 * good / d['n']:>6.1f}% {cov[s][0]:>7}/{cov[s][1]:<7}")


if __name__ == "__main__":
    main()
