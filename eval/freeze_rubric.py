"""Freeze the two-level rubric into eval/answer_key.json (author approved 2026-11-11 with two edits).

  python -m eval.freeze_rubric

Applies the author's edits to the assistant's draft (eval/rubric_levels_draft.json, kept as history):
  * Q8 point 2: Level 1 = text tokens are temporally aligned with the audio frames and incorporated into the generated
    multi-stream sequence; Level 2 = Whisper word-level timestamps, the PAD/EPAD names, 12.5 Hz, and the specific
    padding mechanism between words.
  * Q15 point 5: Level 1 = H-Net learns content-dependent segmentation instead of fixed tokenization, which lets
    computation be allocated adaptively across the sequence; Level 2 = explicitly, that low-information regions are
    compressed while informative regions keep finer granularity.
Refuses to run twice: the rubric is frozen once, and is never changed after seeing results.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KEY = ROOT / "eval" / "answer_key.json"

EDITS = {
    ("Q8", 2): {"level": 1,
                "level1_main": "Text tokens are temporally aligned with the audio frames and incorporated into the "
                               "generated multi-stream sequence.",
                "level2_parts": "Whisper word-level timestamps; the PAD and EPAD names; the exact 12.5 Hz alignment; "
                                "the specific padding mechanism between words",
                "why": "asks 'how time-aligned text tokens are incorporated into audio generation' (author's edit)"},
    ("Q15", 5): {"level": 1,
                 "level1_main": "H-Net learns content-dependent segmentation instead of relying on fixed tokenization, "
                                "which lets computation be allocated adaptively across the sequence.",
                 "level2_parts": "explicitly: low-information regions are compressed while informative regions keep "
                                 "finer granularity",
                 "why": "asks 'what computational or modelling problem is each hierarchy intended to address' "
                        "(author's edit)"},
}

RUBRIC = {
    "version": "two-level rubric, frozen 2026-11-11 by the author (assistant's draft, approved with two edits)",
    "level1": "Must pass; decides correct or wrong. The answer (a) answers the central question, (b) covers the main "
              "idea of every Level 1 point (the explicitly requested components), (c) contains no material factual "
              "error, (d) cites passages that actually support its claims, (e) invents nothing absent from the papers.",
    "level2": "Partial credit only, recorded as a diagnostic (point coverage in the trace); it never changes the grade: "
              "exact numbers, extra architectural detail, equations and implementation specifics, secondary "
              "experimental observations, appendix context. Listed per point as 'level2_parts', or a whole point "
              "tagged level 2.",
    "material_error": "A factual error about something the question asks for, or one that changes the meaning.",
    "no_rescue": "Level 2 never rescues an answer with a material factual error: nine right details plus 'tau = 2 means "
                 "two seconds' is still wrong (author, 2026-11-11). Equally, omitting Level 2 names (e.g. PAD/EPAD) "
                 "never fails an answer whose core concepts are right.",
    "unanswerable": "Q16-Q20: Level 1 = says the papers do not establish the asked information and gives no value for "
                    "it; adjacent facts only if cited and framed as not answering.",
    "rule": "The same frozen rubric grades both evaluation runs (eval and eval_v2). It is not modified after seeing "
            "the second run's results.",
}


def main() -> None:
    key = json.loads(KEY.read_text(encoding="utf-8"))
    if "rubric" in key:
        raise SystemExit("rubric already frozen in answer_key.json; it is not changed after freezing")
    draft = json.loads((ROOT / "eval" / "rubric_levels_draft.json").read_text(encoding="utf-8"))["levels"]
    n = 0
    for q in key["questions"]:
        if not q["answerable"]:
            continue
        tags = {t["point"]: t for t in draft[q["id"]]}
        for i, p in enumerate(q["required"], 1):
            t = dict(tags[i])
            t.update(EDITS.get((q["id"], i), {}))
            p["level"] = t["level"]
            p["level_why"] = t["why"]
            if t.get("level1_main"):
                p["level1_main"] = t["level1_main"]
            if t.get("level2_parts"):
                p["level2_parts"] = t["level2_parts"]
            n += 1
    key["rubric"] = RUBRIC
    KEY.write_text(json.dumps(key, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"froze levels on {n} required points; rubric written to {KEY.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
