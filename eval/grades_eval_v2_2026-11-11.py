"""Grades for eval_v2 (2026-11-11 17:15, h256k8 on the shipped pipeline), by the assistant, with the two-level rubric
frozen in eval/answer_key.json before any of these answers was read (commit c64ccf8).

  python -m eval.grades_eval_v2_2026-11-11   ->  results/grades_eval_v2.csv

points_covered counts required points whose Level 1 main idea is present (Level 2 points count when present).
Tags as in the first run's grades: [gate-FP], [gate catch], [retrieval], [wrong-fact], [variance] (a Level 1 point
the same setting covered in the first run and dropped here: the drafter is sampled, temperature 0.2).
"""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAMP = "11111147"

G = {
    "Q1": ("correct", 3, 3, "all Level 1 points, cited 3.2.1"),
    "Q2": ("correct", 4, 4, "all four, incl. masking to -inf and the offset"),
    "Q3": ("correct", 4, 4, "all, incl. the wavelength progression and learned-embedding result"),
    "Q4": ("correct", 6, 6, "all six base values from AIAYN Table 3 (first run: refused wrongly; fixed by the "
                            "original-question search and named-table lookup)"),
    "Q5": ("correct", 6, 6, "three limitations and how Moshi addresses each (first run: withheld by the list-marker "
                            "gate bug; fixed)"),
    "Q6": ("wrong", 3, 5, "R2 missing: says the streams are 'processed together' as parallel streams but not that they "
                          "are combined into one jointly modelled sequence; R4 (Level 2) missing; credits overlap "
                          "handling partly to the text prefix [variance: the first run's h256k8 answer had R2]"),
    "Q7": ("wrong", 3, 4, "R3 missing: describes plain RVQ layering (first level semantic, later acoustic), never the "
                          "split RVQ that is how Mimi combines them [variance: the first run's h256k8 answer had it]"),
    "Q8": ("correct", 4, 4, "time-aligned text stream as the prefix sub-sequence; scaffolding; delay -> ASR/TTS"),
    "Q9": ("correct", 6, 6, "all six from MOSHI Table 1 (first run: retrieval missed the table; fixed by named-table "
                            "lookup)"),
    "Q10": ("correct", 5, 5, "every stage's role, cited to HNET 1 and Figure 1, which state them"),
    "Q11": ("correct", 3, 4, "routing (cosine, threshold), downsampler keeps b=1 and discards b=0; R4 is Level 2"),
    "Q12": ("wrong", 2, 3, "both scales and both differences right (2.2 and 2.7) and 3.1's 2.2/2.6 attributed, but the "
                           "explicitly asked 'is there any discrepancy?' is not answered: that sentence was withheld for "
                           "having no citation [gate catch] and its reasoning was off ('absolute vs percentage-point')"),
    "Q13": ("correct", 3, 3, "both purposes and the fixed/position vs learned/content contrast, each cited to its paper"),
    "Q14": ("wrong", 1, 4, "acoustic delay described as holding back frames for limited context, not the semantic-to-"
                           "acoustic offset; text-audio delay calls text tokens 'semantic (text) tokens' [wrong-fact] and "
                           "misses ASR/TTS [variance: the first run's h256k8 answer was correct]"),
    "Q15": ("wrong", 3, 6, "Moshi's problem without streaming (R2), the decoder credited with restoring resolution (R4), "
                           "H-Net's problem without replacing tokenization (R5); R1, R3 and the axis contrast (R6) right"),
}
for q in ("Q16", "Q17", "Q18", "Q19", "Q20"):
    G[q] = ("refused_correctly", "", "", "refused; no value given")

out = ROOT / "results" / "grades_eval_v2.csv"
with out.open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["request_id", "setting", "qid", "answer_grade", "points_covered", "points_required", "notes"])
    for qid, (grade, cov, req, note) in G.items():
        w.writerow([f"eval_v2-h256k8-{qid}-{STAMP}", "h256k8", qid, grade, cov, req, note])
print(f"{len(G)} grades -> {out.relative_to(ROOT)}")
