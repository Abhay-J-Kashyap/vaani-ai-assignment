"""Write a review sheet for the author's spot-check of the assistant's grades.

  python -m eval.spot_check [--out eval/spot_check_2026-11-11.md] [Q3:h256k8 Q7:h256k8 ...]

For each picked answer: the question, the key's required points (with acceptable cites), exactly what was sent,
what the gate withheld, the assistant's grade and note, and a blank line for the author's verdict.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from eval.retrieval_scan import load_questions

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ["Q3:h256k8", "Q7:h256k8", "Q10:h256k8", "Q13:h256k8", "Q13:h512k4", "Q8:h512k4rr", "Q6:h256k8",
           "Q14:h512k4", "Q15:h512k4rr", "Q16:h512k4rr"]
WHY = {
    "Q3:h256k8": "Grade changed under the less-strict rule (wavelength progression not given). Tests the rule.",
    "Q7:h256k8": "Grade changed under the rule (codec numbers 24 kHz / Q=8 / 2048 not given).",
    "Q10:h256k8": "Grade changed under the rule; every claim is cited to HNET §1 (the introduction).",
    "Q13:h256k8": "Grade changed under the rule (does not say only chunking changes sequence length).",
    "Q13:h512k4": "Wrong only because one contrast sentence cites HNET §1 for a claim about sinusoids.",
    "Q8:h512k4rr": "Correct with padding tokens but no Whisper timestamps: is the main idea there?",
    "Q6:h256k8": "Wrong with 4/5 points: only 'user stream replaced by real audio at inference' is missing.",
    "Q14:h512k4": "Wrong for a wrong fact: 'tau = 2 corresponds to a 2-second audio delay'.",
    "Q15:h512k4rr": "Wrong, partly for 'the decoder network finally restores the original resolution'.",
    "Q16:h512k4rr": "A template refusal from the decide branch (no drafter call): counted as refused correctly.",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("picks", nargs="*", default=DEFAULT)
    ap.add_argument("--out", default="eval/spot_check_2026-11-11.md")
    ap.add_argument("--scenario", default="eval")
    a = ap.parse_args()
    qs = load_questions()
    key = {q["id"]: q for q in json.loads((ROOT / "eval" / "answer_key.json").read_text(encoding="utf-8"))["questions"]}
    answers = {(x["qid"], x["setting"]): x for x in map(json.loads, (ROOT / "results" / "eval_answers.jsonl")
                                                         .read_text(encoding="utf-8").splitlines())
               if x.get("scenario") == a.scenario}
    grades = {g["request_id"]: g for g in csv.DictReader((ROOT / "results" / "grades_eval.csv").open(encoding="utf-8"))}
    out = ["# Spot-check of the assistant's grades",
           "",
           "Rule in force (author, 2026-11-11): a required point counts when its main idea is stated correctly; specific "
           "details are required only when the question asks for them. Wrong facts and citations that don't support "
           "their claim make the answer wrong. Grades are of what was **sent**; withheld sentences don't count.",
           "",
           "For each item write **agree** or **disagree + your grade**. Disagreements are recorded as overrules.", ""]
    for i, pick in enumerate(a.picks, 1):
        qid, setting = pick.split(":")
        ans = answers[(qid, setting)]
        g = grades[ans["request_id"]]
        k = key[qid]
        out += [f"## {i}. {qid} under {setting}", "", f"*Why check this one:* {WHY.get(pick, '')}", "",
                f"**Question:** {qs[qid]}", "", "**Key: required points**", ""]
        for j, p in enumerate(k.get("required", []), 1):
            out.append(f"{j}. {p['point']} — cites: {', '.join(p['cites'])}")
        if k.get("expected"):
            out.append(f"- Expected: {k['expected']}")
        out += ["", "**Answer as sent:**", ""]
        out += [f"> {s}" for s in ans["sentences"]] or ["> (nothing)"]
        if ans["withheld_sentences"]:
            out += ["", "**Withheld by the gate (not part of the answer):**", ""]
            out += [f"> ~~{w}~~" for w in ans["withheld_sentences"]]
        cov = f" ({g['points_covered']}/{g['points_required']} points)" if g["points_required"] else ""
        out += ["", f"**Assistant's grade:** `{g['answer_grade']}`{cov}. {g['notes']}", "",
                "**Your verdict:** ", "", "---", ""]
    Path(ROOT / a.out).write_text("\n".join(out), encoding="utf-8")
    print(f"{len(a.picks)} items -> {a.out}")


if __name__ == "__main__":
    main()
