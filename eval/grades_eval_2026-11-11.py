"""Grades for the 2026-11-11 eval run (scenario 'eval'), by the assistant, against eval/answer_key.json.

  python -m eval.grades_eval_2026-11-11   ->  results/grades_eval.csv

Rule applied (author, 2026-11-11: "drop down the strictness a bit"): an answerable question is `correct` when every
required point's MAIN IDEA is stated correctly and every claim cites a passage that supports it. Specific details
inside a point (numbers, names, formulas, secondary clauses) are required only when the question asks for them.
A wrong fact or an unsupported citation still makes the answer wrong; refusal grading is unchanged.
The first grading (same day) read every distinct claim in a point as required; answers whose grade changed between
the two readings say so in their note ("strict: wrong"). Grades are of what the person was SENT; sentences the gate
withheld are not part of the answer (tagged [gate-FP] where the withheld sentence was correct and cost the grade).

Tags: [gate-FP] gate withheld a correct sentence; [retrieval] the needed passage was not retrieved;
[wrong-fact] the answer states something false; [borderline] see above.
"""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAMP = "11110706"
S = ["h512k4", "h256k8", "h512k4rr"]

# qid: [(grade, covered, required, note) for h512k4, h256k8, h512k4rr]
G = {
    "Q1": [("correct", 3, 3, "all points, cited 3.2.1")] * 3,
    "Q2": [("correct", 4, 4, "all points; 'offset by one' detail omitted (subordinate)")] * 3,
    "Q3": [("correct", 4, 4, "all points incl. formula and wavelength progression"),
           ("correct", 4, 4, "all main ideas; wavelength progression not given (not asked by Q3; strict: wrong); R3 cited 6.2, which supports it"),
           ("correct", 4, 4, "all main ideas with formula; wavelength progression not given (not asked; strict: wrong)")],
    "Q4": [("refused_wrongly", 0, 6, "said the papers don't give the values [retrieval: AIAYN Table 3 not retrieved for the router's query]"),
           ("refused_wrongly", 0, 6, "same refusal [retrieval: got HNET Table 1 x5 instead of AIAYN Table 3, the 'Table N in every paper' trap]"),
           ("wrong", 5, 6, "R6 100K steps withheld: model wrote '100 K', gate read 100 [gate-FP]")],
    "Q5": [("wrong", 3, 6, "R1-R3 (the three limitations) withheld: list markers (1)(2)(3) read as numbers [gate-FP]"),
           ("wrong", 3, 6, "R1-R3 withheld, same list-marker cause [gate-FP]"),
           ("wrong", 2, 6, "R1-R3 withheld [gate-FP]; R5 160/200 ms latency missing")],
    "Q6": [("wrong", 3, 5, "R2 delay+concatenation and R4 inference stream replacement missing; calls the multi-stream model 'a Depth Transformer' [wrong-fact]"),
           ("wrong", 4, 5, "R4 (user stream replaced by real audio at inference) missing; a second summary sentence withheld for '2' [gate-FP, no grade effect]"),
           ("wrong", 3, 5, "R2 and R4 missing; 'processed jointly by a Depth-Transformer architecture' [wrong-fact]")],
    "Q7": [("wrong", 1, 4, "R1 lacks Q=8/2048 [borderline]; R2 says WavLM distilled 'into the causal encoder', not the first quantizer level; R3 split RVQ named but described as a plain RVQ [wrong-fact]"),
           ("correct", 4, 4, "all main ideas incl. split RVQ; codec numbers not given (not asked by Q7; strict: wrong)"),
           ("correct", 4, 4, "all main ideas; codec numbers not given (not asked; strict: wrong)")],
    "Q8": [("wrong", 3, 4, "R2 alignment (Whisper timestamps, PAD/EPAD) missing"),
           ("wrong", 3, 4, "R2 alignment missing"),
           ("correct", 4, 4, "R2 main idea present (text aligned to the frame rate with padding tokens); Whisper not named (strict: wrong)")],
    "Q9": [("correct", 6, 6, "all six from MOSHI Table 1, no Temporal values"),
           ("wrong", 3, 6, "dim/heads/layers right; says the papers don't give MLP dim, cardinality, frame rate [retrieval: MOSHI Table 1 not retrieved]"),
           ("correct", 6, 6, "all six from MOSHI Table 1")],
    "Q10": [("wrong", 4, 5, "R2 chunking sentence withheld: '0' from the superscript in p^0 read as a number [gate-FP]"),
            ("correct", 5, 5, "every stage's main idea, cited to HNET 1 (intro), which states them; recursion and residual Linear(x) not given (not asked; strict: wrong)"),
            ("wrong", 4, 5, "says the decoder upsamples to byte resolution (dechunking does) [wrong-fact]")],
    "Q11": [("correct", 4, 4, "routing (cosine, 0.5 threshold) and downsampler (keep b=1, discard rest); probabilities-to-dechunking not given (not asked by Q11; strict: wrong)")] * 3,
    "Q12": [("wrong", 0, 3, "only the 3.1 prose figures sent; table sentences withheld and were wrong (58.2 placed at Large) [retrieval: table split into row parts]"),
            ("wrong", 0, 3, "H-Net averages and 3.1 figures sent, no baselines or differences; withheld sentences wrong (53.3 at both scales)"),
            ("wrong", 0, 3, "3.1 figures sent; table reading confused by row-part chunks; withheld sentences wrong (gain 0%)")],
    "Q13": [("wrong", 3, 3, "the contrast sentence's claims about sinusoids are cited only to HNET 1, which doesn't support that half (unsupported citation)"),
            ("correct", 3, 3, "both purposes and the fixed/position vs learned/content contrast, each cited to its paper; length difference not stated (strict: wrong)"),
            ("wrong", 2, 3, "both contrast sentences withheld for missing citations (genuine gate catch)")],
    "Q14": [("wrong", 1, 4, "'tau = 2 corresponds to a 2-second audio delay' [wrong-fact: 2 steps = 160 ms]; R3 lacks ASR/TTS; R4 missing"),
            ("correct", 4, 4, "causal mask; acoustic delay and Temporal Transformer; text-audio delay -> TTS/ASR; autoregressive"),
            ("wrong", 1, 4, "acoustic delay described as latency only; text-audio delay lacks ASR/TTS; R4 missing")],
    "Q15": [("wrong", 3, 6, "R2 (why flattening fails), R5 (H-Net's problem), R6 (explicit contrast of axes) missing"),
            ("wrong", 2, 6, "R2, R4 (decoder combines coarse output with residuals), R5, R6 missing"),
            ("wrong", 2, 6, "R2 only partly (S vs K*S steps, no streaming); R4 says the decoder restores resolution [wrong-fact]; R5, R6 missing")],
}
for q in ("Q16", "Q17", "Q18", "Q19", "Q20"):
    G[q] = [("refused_correctly", "", "", "refused; no value given")] * 3
G["Q16"] = [G["Q16"][0], G["Q16"][1], ("refused_correctly", "", "", "refused at the decide branch (re-rank score below threshold), no drafter call")]
G["Q18"] = [G["Q18"][0], G["Q18"][1], ("refused_correctly", "", "", "refused at the decide branch, no drafter call")]
G["Q20"] = [G["Q20"][0], G["Q20"][1], ("refused_correctly", "", "", "refused at the decide branch, no drafter call")]

out = ROOT / "results" / "grades_eval.csv"
with out.open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["request_id", "setting", "qid", "answer_grade", "points_covered", "points_required", "notes"])
    for qid, per in G.items():
        for s, (grade, cov, req, note) in zip(S, per):
            w.writerow([f"eval-{s}-{qid}-{STAMP}", s, qid, grade, cov, req, note])
print(f"{sum(len(v) for v in G.values())} grades -> {out.relative_to(ROOT)}")
