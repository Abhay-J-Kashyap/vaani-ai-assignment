"""Spot-check of the real load grades (author's request, 2026-11-12): print each sampled answer with, for every
sentence, the passages its citations point to (closest sentences from the cited section in the corpus), so the grade
and every citation can be checked against the paper text. 0 tokens.

  python -m eval.spot_check_load > results/spot_check_load_evidence_2026-11-12.txt

Sample (fixed, seed 12):
  1. every non-cached answer graded wrong;
  2. every refusal: quoted-passage answers (refused_wrongly by rule) and a check that each refused_correctly answer
     is a pure "the papers do not say" with nothing else shown;
  3. correct answers: a random 12 of the non-cached ones, plus at least one cross-paper (Q5, Q13) and one numerical
     (Q4, Q9, Q12) answer from each tier present.
"""
from __future__ import annotations

import csv
import json
import random
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STOP = set("the a an of and or to in on for by with is are was were be as that this it its from at which their "
           "these those each both into than then so not can".split())
CROSS, NUMERIC = {"Q5", "Q13", "Q15"}, {"Q4", "Q9", "Q12"}


def passages() -> dict[str, str]:
    by = defaultdict(list)
    for line in (ROOT / "data" / "corpus" / "chunks_256.jsonl").open(encoding="utf-8"):
        c = json.loads(line)
        by[c["cite"]].append(c["text"])
    return {k: " ".join(dict.fromkeys(v)) for k, v in by.items()}


def lookup(cite: str, P: dict[str, str]) -> str:
    cite = cite.strip().rstrip(".").replace("H‑Net", "HNET")
    if cite in P:
        return P[cite]
    sub = [v for k, v in P.items() if k.startswith(cite + ".")]          # "MOSHI §3.3" -> its subsections
    return " ".join(sub)


def words(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9.]+", s.lower()) if w not in STOP and len(w) > 2}


def best(claim: str, text: str, n: int = 2) -> list[str]:
    sents = re.split(r"(?<=[.!?])\s+", text)
    cw = words(claim)
    nums = set(re.findall(r"\d+(?:\.\d+)?", claim))
    scored = sorted(sents, key=lambda s: -(len(cw & words(s)) + 3 * len(nums & set(re.findall(r"\d+(?:\.\d+)?", s)))))
    return [s[:360] for s in scored[:n] if s.strip()]


def main() -> None:
    P = passages()
    recs = {json.loads(l)["request_id"]: json.loads(l) for l in (ROOT / "results" / "load_answers.jsonl").open(encoding="utf-8")}
    grades = {g["request_id"]: g for g in csv.DictReader((ROOT / "results" / "grades_load.csv").open(encoding="utf-8"))}
    key = {q["id"]: q for q in json.loads((ROOT / "eval" / "answer_key.json").read_text(encoding="utf-8"))["questions"]}

    wrong = [r for r, g in grades.items() if g["answer_grade"] == "wrong" and recs[r]["tier"] != "cached"]
    refused = [r for r, g in grades.items() if g["answer_grade"].startswith("refused")]
    correct = [r for r, g in grades.items() if g["answer_grade"] == "correct" and recs[r]["tier"] != "cached"]
    rng = random.Random(12)
    sample = rng.sample(sorted(correct), 12)
    for tier in sorted({recs[r]["tier"] for r in correct}):
        for group in (CROSS, NUMERIC):
            if not any(recs[r]["qid"] in group and recs[r]["tier"] == tier for r in sample):
                pool = sorted(r for r in correct if recs[r]["qid"] in group and recs[r]["tier"] == tier)
                if pool:
                    sample.append(rng.choice(pool))

    def show(rid: str) -> None:
        r, g, k = recs[rid], grades[rid], key[recs[rid]["qid"]]
        print(f"\n=== {rid} {r['scenario']} {r['tier']} {r['qid']} grade={g['answer_grade']} "
              f"points={g['points_covered']}/{g['points_required']}  note: {g['notes']}")
        for i, p in enumerate(k.get("required", []), 1):
            print(f"  R{i} L{p.get('level')}: {p['point']}")
        for s in r["sentences"]:
            print(f"  >> {s['text']}")
            for c in dict.fromkeys(s["cites"]):
                text = lookup(c, P)
                if not text:
                    print(f"     [{c}] NOT FOUND IN CORPUS")
                    continue
                for b in best(s["text"] or "", text):
                    print(f"     [{c}] {b}")
        for w in r["withheld"]:
            print(f"  (withheld: {w['reason']})")

    print(f"# 1. wrong, non-cached: {len(wrong)}")
    for rid in sorted(wrong, key=lambda x: (int(recs[x]["qid"][1:]), x)):
        show(rid)
    print(f"\n# 2. refusals: {len(refused)}")
    for rid in sorted(refused, key=lambda x: (int(recs[x]["qid"][1:]), x)):
        r, g = recs[rid], grades[rid]
        ok = (r["tier"] in ("extractive", "cached")) or (not r["sentences"] and r["not_covered"])
        print(f"  {rid} {r['qid']} answerable={key[r['qid']]['answerable']} tier={r['tier']} grade={g['answer_grade']} "
              f"{'pure refusal' if ok else 'MAKES CLAIMS'}: {(r['not_covered'] or [''])[0][:110]}")
    print(f"\n# 3. correct sample: {len(sample)} of {len(correct)} non-cached correct")
    for rid in sorted(sample, key=lambda x: (int(recs[x]["qid"][1:]), x)):
        show(rid)


if __name__ == "__main__":
    main()
