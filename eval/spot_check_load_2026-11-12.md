# Spot-check of the 172 real load grades (2026-11-12)

Requested by the author: prioritise wrong and refusal grades, then randomly sample correct answers (including
cross-paper and numerical questions), and check every citation against the actual passages. Done by the assistant;
0 tokens.

- Evidence, one block per answer, with each sentence next to the closest sentences of every passage it cites:
  `results/spot_check_load_evidence_2026-11-12.txt`, from `python -m eval.spot_check_load` (fixed sample, seed 12).
- Grades: `results/grades_load.csv`, written by `eval/grades_load_2026-11-12.py`. The rubric is frozen in `eval/answer_key.json`.

## What was checked

| Group | Checked | Result |
|---|---|---|
| Graded `wrong`, non-cached | all 36 | all 36 confirmed |
| Refusals (`refused_correctly`, `refused_wrongly`) | all 54 | all 54 confirmed; every one is a pure "the papers do not say", quoted passages, or a cache replay of one |
| Graded `correct`, non-cached | 12 of 70, random (seed 12); covers cross-paper Q5/Q13 and numerical Q9 in both tiers | all 12 confirmed |

**Grade changes: none.** `results.csv`, Tables 2 and 3, and the quality-by-tier figures are unchanged (regenerated to confirm).

## What changed: grading notes and diagnostic points (not grades)

| Request | Question | Change | Why, from the passages |
|---|---|---|---|
| 8e0ab0a00271 | Q4 | points 3 -> 2 | "eight heads" cites AIAYN §3.1, which does not state it (heads are in §3.2.2) |
| 31c7829e76c0 | Q14 | points 2 -> 1; note | its delay sentence paraphrases MOSHI §2 ("allows deriving streaming TTS and ASR") without the direction, so R3 is missing (an omission, not the "wrong fact" first noted); its acoustic-delay sentence cites §2, which never mentions the acoustic delay, so R2 does not count |
| 261de3570169 | Q15 | points 2 -> 3 | §1's "replacing handcrafted heuristics with ... segmentation learned from data" covers R5; still wrong (stage order, no contrast) |
| 1b2fdc7987d4 | Q7 | note | it follows MOSHI §1's own summary ("distilling semantic information into the first level of acoustic tokens"); incomplete (no split RVQ, §3.3.2 / Figure 2), not contrary to the paper |
| a390f57a6be0 | Q14 | note | Figure 9 says "audio delayed by 2 seconds" and "acoustic token delay τ=2"; the answer merged them into "τ = 2 s" |
| d61740852d04 | Q15 | note | R5 half there: it cites HNET §2.2's "compressing low-information regions" |
| 48ff2584fc41 | Q7 | note | "transposed causal convolutions" cites §3.3, which does not state it |
| 789595165023 | Q5 | note | the Inner Monologue link is MOSHI §3.1's own ("jointly model text and audio tokens ... while remaining a speech-to-speech system"); quoted with slightly changed wording |
| d28b19f00ce8 | Q8 | note | "factuality" comes from MOSHI §1 but is cited to §3.4.4, which supports only the linguistic-quality half; minor |

## Observations worth knowing

- **Q7 is wrong in all 8 non-cached answers for one reason.** The papers' own §1 summary says semantic information goes into "the first level of acoustic tokens". The drafter repeats that summary and never reaches the split RVQ in §3.3.2. The grade follows the frozen rubric (R3 is Level 1). The answers are incomplete, not invented.
- **The citation gate does what it claims, and no more.** Every sentence's citation named a retrieved passage, and every number in a shown sentence appears in its cited passage. But two sentences cite a passage that does not support the specific claim: Q4 heads cited to §3.1, and Q14 acoustic delay cited to §2. That is the gate's documented blind spot: it checks names and numbers, not meaning.
- **The quoted-passage answers have no logged text** (the trace records only the event), so their grade rests on the author's rule (quoted passages are a refusal), not on reading them.
