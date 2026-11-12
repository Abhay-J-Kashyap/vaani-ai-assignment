"""Grades for the 172 real load answers (steady, burst, drain, daily_exhausted, real_smoke), by the assistant, against
the two-level rubric frozen in eval/answer_key.json (commit c64ccf8), applying the same precedents as eval_v2.
Asked for by the author on 2026-11-12, to measure what a weaker answer under load costs in quality.

  python -m eval.load_answers                 # first: rebuild results/load_answers.jsonl from the trace
  python -m eval.grades_load_2026-11-12       ->  results/grades_load.csv

Only what the person saw is graded: sentences withheld by the gate do not count. Rules applied by this script:
  * a cache hit replays an earlier Full answer, so it gets that answer's grade (points and notes copied);
  * a quoted-passages answer (tier 'extractive') makes no claim, so it is a refusal (author, 2026-11-12):
    refused_correctly for an unanswerable question, refused_wrongly for an answerable one;
  * an unanswerable question's answer must be only a "the papers do not say" statement; anything else stops the
    script for a manual grade (none did).
Every other answer is graded by hand below: (grade, points covered, points required, note).
Precedents carried over from eval_v2: Q6 needs the streams combined into one jointly modelled sequence (R2); Q7 needs
the split RVQ (R3), plain "first level semantic, later levels acoustic" is the plain-RVQ reading; Q12 needs the
discrepancy actually stated, showing 2.6 and 2.7 side by side without saying they differ is not enough; Q14 needs the
semantic-to-acoustic offset (R2) and the ASR/TTS direction (R3); Q15 needs H-Net's problem as replacing tokenization
(R5) and the axis contrast (R6).
"""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

C = "correct"
W = "wrong"
G = {
    # Q1 (3 points)
    "2c815bc21203": (C, 3, 3, "growth with d_k, small softmax gradients, 1/sqrt(d_k)"),
    "236d753ccc91": (C, 3, 3, "all three"),
    "8b63b7859007": (C, 3, 3, "all three"),
    "541b554bab1a": (C, 3, 3, "all three"),
    "ecfc8291e123": (C, 3, 3, "all three, plus the footnote's variance argument"),
    "d723ef794854": (C, 3, 3, "all three"),
    "22a9500f6a34": (C, 3, 3, "all three; 'variance becomes 1' follows from the cited footnote"),
    # Q2 (4)
    "e26ccf5aa767": (C, 4, 4, "all four, incl. -inf masking"),
    "d1994817e6ba": (C, 4, 4, "all four"),
    "671e43c6ad8a": (C, 4, 4, "all four"),
    "37a4c8433f95": (C, 4, 4, "all four"),
    "c9c546d22ae2": (C, 4, 4, "all four"),
    # Q3 (4; R3 Level 2)
    "a60b004c4906": (C, 4, 4, "all four"),
    "da143e4237d5": (C, 4, 4, "all four"),
    "5756f51636e1": (C, 3, 4, "R3 (learned embeddings, Level 2) withheld by the gate [gate catch: 25.7/25.8 not in cited passage]"),
    "ae1e7bf1d7a6": (C, 4, 4, "all four"),
    "b52c1f2b0fdc": (C, 4, 4, "all four"),
    "d60285c75f1c": (C, 4, 4, "all four"),
    "02ddd44f2ae6": (C, 4, 4, "all four"),
    "d3deac22cbaa": (C, 4, 4, "all four"),
    "db5586b69ddc": (C, 3, 4, "R3 (Level 2) withheld by the gate (cite to a table row not retrieved)"),
    "a078a2bfb3b1": (C, 4, 4, "all four"),
    # Q4 (6)
    "bb71ee733f04": (C, 6, 6, "all six base values from AIAYN Table 3"),
    "1d3a338eef7e": (C, 6, 6, "all six"),
    "fa711bf25d7c": (C, 6, 6, "all six"),
    "2ef48469e2ea": (C, 6, 6, "all six"),
    "8400206fc2ab": (C, 6, 6, "all six"),
    "5d2bd5b277c6": (C, 6, 6, "all six"),
    "8e0ab0a00271": (W, 2, 6, "d_ff, d_k/d_v and steps withheld: cited §3.1 does not hold them [gate catch, retrieval]; "
                              "the person saw only N, d_model, h, and 'eight heads' cites §3.1, which does not state it "
                              "(spot-check 2026-11-12: points 3 -> 2)"),
    "6bced2bfe080": (C, 6, 6, "all six (a first, uncited copy was withheld; the cited copy passed)"),
    # Q5 (6)
    "789595165023": (C, 6, 6, "all six; the Inner Monologue link is §3.1's own ('jointly model text and audio tokens ... "
                              "while remaining a speech-to-speech system'), quoted with slightly changed wording "
                              "(spot-check 2026-11-12: note corrected)"),
    "c60c039fd4f0": (C, 6, 6, "all six"),
    "cf6863ca16f2": (C, 6, 6, "all six"),
    "b48eb42cd0b2": (C, 6, 6, "all six"),
    "671df73597a2": (C, 6, 6, "all six; same Inner Monologue attribution as 789595165023, not material"),
    "72fb4420f660": (C, 6, 6, "all six"),
    "eaa15a801e02": (C, 6, 6, "all six"),
    # Q6 (5; R4 Level 2)
    "3a1526c55198": (C, 4, 5, "streams, acoustic delay + concatenation into one sequence, no turns, overlap; R4 (L2) missing"),
    "195aac6a3152": (C, 4, 5, "as above; R4 (L2) missing"),
    "05eb6acc393f": (C, 4, 5, "as above; R4 (L2) missing"),
    "e6c3574929c3": (W, 3, 5, "R2 missing: 'processed together ... separate autoregressive sequences', never combined into one "
                              "jointly modelled sequence (eval_v2 precedent); R4 missing"),
    "53f703db0f25": (C, 4, 5, "as 3a1526c55198; R4 (L2) missing"),
    # Q7 (4)
    "0211dc5ccc5a": (W, 3, 4, "R3 missing: plain RVQ layering (first level semantic, later acoustic), no split RVQ "
                              "(eval_v2 precedent)"),
    "ae3d062576b1": (W, 3, 4, "R3 missing (no split RVQ)"),
    "48e3fa205432": (W, 3, 4, "R3 missing (no split RVQ)"),
    "5a793cc7e134": (W, 3, 4, "R3 missing (no split RVQ)"),
    "ca5a40680ad7": (W, 3, 4, "R3 missing (no split RVQ)"),
    "1b2fdc7987d4": (W, 3, 4, "R3 missing: follows MOSHI §1's own summary ('distilling semantic information into the "
                              "first level of acoustic tokens') and never mentions the split RVQ of §3.3.2 / Figure 2; "
                              "incomplete, not contrary to the paper (spot-check 2026-11-12: note corrected)"),
    "770782502d85": (W, 3, 4, "R3 missing (no split RVQ)"),
    "48ff2584fc41": (W, 3, 4, "R3 missing (no split RVQ); the 'transposed causal convolutions' sentence cites §3.3, "
                              "which does not state it (spot-check 2026-11-12)"),
    # Q8 (4)
    "d00f10eecfc4": (C, 4, 4, "prefix text stream, per-timestep alignment, scaffolding, delay -> TTS/ASR in the right direction"),
    "a624032cdbf7": (W, 3, 4, "R4 reversed in one sentence: 'delaying audio tokens (audio ahead of text) ... streaming ASR'; "
                              "a later sentence gives the right direction, so the answer contradicts itself [wrong-fact]"),
    "d6405e39771a": (C, 4, 4, "all four, directions right"),
    "adf1ed7816b4": (C, 4, 4, "all four, directions right"),
    "d28b19f00ce8": (C, 4, 4, "all four, directions right; 'factuality' is from MOSHI §1 but cited to §3.4.4, which "
                              "supports only the linguistic-quality half (minor; spot-check 2026-11-12)"),
    "8c9de9f1ec50": (C, 4, 4, "direction sentence withheld [gate catch], but the TTS = audio-delayed / ASR = text-delayed "
                              "configuration states it"),
    # Q9 (6)
    "230ba5b0d882": (C, 6, 6, "all six from MOSHI Table 1"),
    "2046245587e3": (C, 6, 6, "all six"),
    "98c990d690f1": (C, 6, 6, "all six"),
    "df785cfad4bd": (C, 6, 6, "all six"),
    "dca6af4dd4cc": (C, 6, 6, "all six"),
    "5ebbdcfacdcf": (C, 6, 6, "all six"),
    "054ae6f1cc28": (C, 6, 6, "all six"),
    "5de5b23efa3d": (C, 6, 6, "all six"),
    # Q10 (5)
    "4a94f0f134da": (C, 5, 5, "every stage's role (HNET §1 puts smoothing in DC; consistent with its cite)"),
    "cf53f524ee0e": (W, 3, 5, "encoder said to compress into a lower-resolution representation [wrong-fact]; dechunking "
                              "conflated with the decoder"),
    "f33d5dbaa555": (C, 5, 5, "every stage; 'encoder first compresses the input vectors' is loose but claims no change of "
                              "resolution"),
    "0543d499aa5b": (C, 5, 5, "every stage's role"),
    "95814bd3bd10": (W, 3, 5, "encoder said to compress into a lower-resolution representation [wrong-fact]; dechunking "
                              "'implemented by a decoder network'"),
    "83a520c29985": (W, 4, 5, "'the dechunking (or decoder) network': two distinct stages conflated [wrong-fact]"),
    # Q11 (4; R4 Level 2)
    "2e0e0d015220": (C, 3, 4, "cosine routing, p>=0.5 boundary, direct selection; R4 (L2) missing"),
    "dbd134f3af5a": (C, 3, 4, "as above"),
    "e2acddf73914": (C, 3, 4, "as above"),
    "f7174b0a7504": (C, 3, 4, "as above"),
    "6c5664186dd0": (C, 4, 4, "all four, incl. compressing the boundary probabilities"),
    "34bc3276b058": (C, 3, 4, "formula + 0.5 threshold state R1-R2; direct selection; R4 (L2) missing"),
    "c01cf83ca08d": (C, 3, 4, "R1-R3 (formula drops the norms in the denominator, a typesetting slip); R4 missing"),
    # Q12 (3)
    "054ee9d79b1e": (W, 2, 3, "both scales and differences right, §3.1's 2.2/2.6 attributed, but the asked discrepancy "
                              "sentence was withheld (no citation) [gate catch] (eval_v2 precedent)"),
    "b99d82f6f4d6": (W, 2, 3, "as above; the withheld sentence's reasoning was also off"),
    "20d3e54b05c3": (W, 2, 3, "as above; discrepancy sentences withheld"),
    "e228214e1365": (W, 2, 3, "as above"),
    "0712702e40e6": (W, 2, 3, "as above"),
    "b5c88870075b": (C, 3, 3, "states 2.7 from the table 'whereas §3.1 states 2.6', each attributed; Large matches"),
    # Q13 (3)
    "905e28ad16a2": (C, 3, 3, "both purposes and the fixed/position vs learned/content contrast; each fact cited to its paper"),
    "fd7fbdcbc76c": (C, 3, 3, "both purposes and the contrast, each cited to its paper"),
    "74e4cfec28b5": (C, 3, 3, "both purposes and the contrast"),
    "4fbe181512f7": (C, 3, 3, "both purposes and the contrast (one restating sentence withheld)"),
    "bfac6217ced9": (C, 3, 3, "both purposes and the contrast"),
    "f0b0d030de22": (C, 3, 3, "purposes in sentences 1-2, contrast framed by 'In contrast' (two summary sentences withheld)"),
    # Q14 (4; R4 Level 2)
    "9c09ead8007f": (W, 1, 4, "acoustic delay as frames held back for look-ahead, not the semantic-to-acoustic offset; "
                              "text-audio delay direction muddled, no ASR/TTS (eval_v2 precedent)"),
    "31c7829e76c0": (W, 1, 4, "R3 missing: paraphrases MOSHI §2 (a delay allows TTS and ASR) without saying which "
                              "direction gives which; its acoustic-delay sentence cites §2, which does not mention the "
                              "acoustic delay, so R2 does not count (spot-check 2026-11-12: points 2 -> 1, note corrected)"),
    "a390f57a6be0": (W, 1, 4, "conflates Figure 9's TTS-mode setting ('audio delayed by 2 seconds' and 'acoustic token "
                              "delay tau=2') into 'acoustic delay tau = 2 s' relative to text [wrong-fact]; text-audio delay "
                              "described as the pre-training randomisation (§4.4) (spot-check 2026-11-12: note sharpened)"),
    "c2a6c5729ede": (W, 1, 4, "acoustic delay vague; text 'provided later than acoustic tokens' yet conditions acoustic "
                              "generation [wrong-fact]; no ASR/TTS"),
    "301fc011a7c6": (W, 1, 4, "same pattern as c2a6c5729ede"),
    "f5aae1bd9485": (W, 1, 4, "same pattern as c2a6c5729ede"),
    "bc29f68ed587": (W, 1, 4, "acoustic delay as audio frame pacing [wrong-fact]; no ASR/TTS"),
    # Q15 (6)
    "0992b0c8c8c9": (W, 2, 6, "Moshi side only; decoder said to use the smoothing module; the contrast was withheld"),
    "261de3570169": (W, 3, 6, "H-Net order wrong (DC before the encoder) [wrong-fact]; no contrast (R6); R5 roughly "
                              "present via §1's 'replacing handcrafted heuristics' (spot-check 2026-11-12: points 2 -> 3)"),
    "b619594abc73": (W, 3, 6, "R5 given as compute on long sequences, not replacing tokenization; DC sentence withheld; "
                              "no axis contrast (R6)"),
    "5019f4142497": (W, 4, 6, "R5 and R6 missing (eval_v2 precedent)"),
    "d61740852d04": (W, 3, 6, "R6 missing; R5 only half there: cites HNET §2.2's 'compressing low-information regions' "
                              "but not replacing tokenization (spot-check 2026-11-12: note corrected)"),
    "1306035b1856": (W, 4, 6, "R5 withheld, R6 missing"),
    "0c49d461abba": (W, 2, 6, "every H-Net sentence withheld (no citation): the person saw only Moshi"),
    "db6e2fe1c9b6": (W, 4, 6, "R5 and R6 missing"),
    "b6083aac6564": (W, 4, 6, "R5 and R6 missing"),
    "41b524b3d8a8": (W, 3, 6, "R5 not stated as replacing tokenization; R6 missing"),
}


def main() -> None:
    key = {q["id"]: q for q in json.loads((ROOT / "eval" / "answer_key.json").read_text(encoding="utf-8"))["questions"]}
    recs = [json.loads(l) for l in (ROOT / "results" / "load_answers.jsonl").open(encoding="utf-8")]
    by_rid = {r["request_id"]: r for r in recs}
    out, problems = {}, []

    def grade(r):
        rid, k = r["request_id"], key[r["qid"]]
        if rid in out:
            return out[rid]
        if r["tier"] == "cached":
            src = r["cached_from"]["request_id"]
            if src in by_rid:
                g = grade(by_rid[src])
            elif src in G:
                g = G[src]
            else:
                problems.append(f"{rid}: cache source {src} not graded")
                return None
            g = (g[0], g[1], g[2], f"cache hit: replay of {src}")
        elif r["tier"] == "extractive":
            g = ("refused_wrongly" if k["answerable"] else "refused_correctly", "", "",
                 "quoted passages only (no claim): counted as a refusal")
        elif not k["answerable"]:
            if r["sentences"] or not r["not_covered"]:
                problems.append(f"{rid}: unanswerable {r['qid']} but the answer makes claims; grade by hand")
                return None
            g = ("refused_correctly", "", "", "states the papers do not say; no value given")
        else:
            if rid not in G:
                problems.append(f"{rid}: answerable {r['qid']} not graded")
                return None
            g = G[rid]
        allowed = {"correct", "wrong", "refused_wrongly"} if k["answerable"] else {"refused_correctly", "wrong"}
        if g[0] not in allowed:
            problems.append(f"{rid}: {g[0]} not allowed for {r['qid']}")
        out[rid] = g
        return g

    for r in recs:
        grade(r)
    extra = set(G) - set(by_rid)
    if problems or extra:
        raise SystemExit("\n".join(problems + [f"graded but not a load answer: {sorted(extra)}"]))
    path = ROOT / "results" / "grades_load.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["request_id", "setting", "qid", "answer_grade", "points_covered", "points_required", "notes"])
        for r in recs:
            g = out[r["request_id"]]
            w.writerow([r["request_id"], r["tier"], r["qid"], g[0], g[1], g[2], g[3]])
    print(f"{len(out)} grades -> {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
