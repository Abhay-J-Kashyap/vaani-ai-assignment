"""Option C (stream sentence by sentence through the gate) vs Option B (buffer the whole answer, check, then send),
from the eval run alone; no extra Groq calls (author, 2026-11-11: no Groq judge, the assistant judges).

  python -m eval.c_vs_b

* Latency: B sends nothing until the draft is complete, so B's first token is C's total time for the same request
  (the deterministic check adds ~1 ms). Both columns are in results.csv, so this is computable from it. A runtime LLM
  judge would add its own call on top of B's first token; that part is not measured.
* Quality: the assistant's grading is the judge. "Misread but passed the gate" = answered questions whose note
  records a wrong fact or an unsupported citation (the gate's known blind spot).
* Gate-fail rate: withheld sentences / all drafted sentences, from results/eval_answers.jsonl.
Pre-registered triggers for switching to B (ADR "Decision record", 2026-11-09): misread > 10% of answered;
gate-fail > 25% of sentences; C's first-token p50 not at least 30% lower than B's.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

from tools.tables import pct

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    rows = [r for r in csv.DictReader((ROOT / "results.csv").open(encoding="utf-8")) if r["scenario"] == "eval"]
    notes = {g["request_id"]: g for g in csv.DictReader((ROOT / "results" / "grades_eval.csv").open(encoding="utf-8"))}
    sent = defaultdict(lambda: [0, 0])
    for line in (ROOT / "results" / "eval_answers.jsonl").read_text(encoding="utf-8").splitlines():
        a = json.loads(line)
        if a.get("scenario") == "eval":
            sent[a["setting"]][0] += len(a["sentences"])
            sent[a["setting"]][1] += len(a["withheld_sentences"])
    by = defaultdict(list)
    for r in rows:
        by[r["retrieval_setting"]].append(r)
    print("setting  | C first token p50/p95 | B first token (= C total) p50/p95 | C lower at p50 / p95 | "
          "misread but passed | gate-fail rate")
    for s, g in by.items():
        c = [float(r["first_token_ms"]) for r in g if r["first_token_ms"]]
        b = [float(r["total_ms"]) for r in g if r["first_token_ms"]]
        answered = [r for r in g if r["answer_grade"] in ("correct", "wrong")]
        misread = [r for r in answered if "[wrong-fact" in notes[r["request_id"]]["notes"]
                   or "unsupported citation" in notes[r["request_id"]]["notes"]]
        ok, wh = sent[s]
        d50 = 1 - pct(c, 50) / pct(b, 50)
        d95 = 1 - pct(c, 95) / pct(b, 95)
        print(f"{s:<8} | {pct(c, 50):,.0f} / {pct(c, 95):,.0f} ms | {pct(b, 50):,.0f} / {pct(b, 95):,.0f} ms | "
              f"{100 * d50:.1f}% / {100 * d95:.1f}% | {len(misread)}/{len(answered)} = "
              f"{100 * len(misread) / len(answered):.1f}% | {wh}/{ok + wh} = {100 * wh / (ok + wh):.1f}%")


if __name__ == "__main__":
    main()
