"""Unit checks for the citation gate and the streaming sentence splitter. No network.

  python -m tests.test_gate
"""
from __future__ import annotations

import random

from relay.gate import SentenceSplitter, check_citations, find_cites

P = {
    "AIAYN §3.2.1": "We suspect that for large values of d_k, the dot products grow large in magnitude. "
                    "To counteract this effect, we scale the dot products by 1/sqrt(d_k).",
    "AIAYN Table 3": "Table 3: Variations on the Transformer architecture.\nbase | 6 | 512 | 2048 | 8 | 64 | 64 | 100K",
    "HNET Table 2": "Table 2: ... Transformer (Token) 760M 53.3 ... H-Net (2-stage) 55.5 ... XL 55.5 58.2",
}
Q = "What is the power draw on an RTX 4090?"


def splits(text: str, step: int) -> list[str]:
    sp, out = SentenceSplitter(), []
    for i in range(0, len(text), step):
        out += sp.feed(text[i:i + step])
    return out + sp.flush()


def main() -> None:
    text = ("Dot products grow large for large d_k, e.g. when d_k is 64 [AIAYN §3.2.1]. They are scaled by "
            "1/sqrt(d_k). [AIAYN §3.2.1] [AIAYN Table 3]\nThe base model uses 6 layers and d_model 512 "
            "[AIAYN Table 3]. The papers do not report power draw.")
    expect = ["Dot products grow large for large d_k, e.g. when d_k is 64 [AIAYN §3.2.1].",
              "They are scaled by 1/sqrt(d_k). [AIAYN §3.2.1] [AIAYN Table 3]",
              "The base model uses 6 layers and d_model 512 [AIAYN Table 3].",
              "The papers do not report power draw."]
    for step in (1, 2, 3, 7, 50, 1000):
        got = splits(text, step)
        assert got == expect, (step, got)
    rng = random.Random(1)
    for _ in range(200):
        cuts = sorted(rng.sample(range(1, len(text)), 12))
        sp, out, prev = SentenceSplitter(), [], 0
        for c in cuts + [len(text)]:
            out += sp.feed(text[prev:c])
            prev = c
        assert out + sp.flush() == expect
    print("splitter: ok (fixed and random chunkings)")

    assert find_cites("x [AIAYN §3.2.1; AIAYN Table 3] y")[0] == ["AIAYN §3.2.1", "AIAYN Table 3"]
    assert find_cites("x [HNET §2.1.1, §2.1.2]")[0] == ["HNET §2.1.1", "HNET §2.1.2"]
    assert find_cites("range [0, 1] only")[0] == []

    cases = [
        ("The base model has 6 layers and d_model = 512 [AIAYN Table 3].", True, "answer"),
        ("The base model trains for 100K steps [AIAYN Table 3].", True, "answer"),
        ("The base model has 12 layers [AIAYN Table 3].", False, "fail"),                      # made-up number
        ("Scaling by 1/sqrt(d_k) fixes this [AIAYN §3.2.2].", False, "fail"),                  # not retrieved
        ("Scaling fixes the softmax gradients.", False, "fail"),                               # no cite
        ("At XL the gap is 2.7 points (58.2 vs 55.5) [HNET Table 2].", True, "answer"),        # derived number
        ("At Large the gap is 2.2 points [HNET Table 2].", False, "fail"),                     # operands not shown
        ("At Large the gap is 2.2 points (55.5 vs 53.3) [HNET Table 2].", True, "answer"),
        ("The papers do not report the power draw on an RTX 4090.", True, "not_covered"),
        ("The papers do not report it, but it is about 450 W.", False, "fail"),                # guess smuggled in
        ("As Table 3 shows, the base model uses 8 heads [AIAYN Table 3].", True, "answer"),    # 'Table 3' is a ref
        ("**Encoder:**", True, "skip"),
        # false positives seen in the real eval (2026-11-11): each withheld a correct sentence
        ("The model was trained for 100 K steps (train = 100K) [AIAYN Table 3].", True, "answer"),
        ("Pipelines suffer from (1) latency, (2) a text bottleneck and (3) turns [AIAYN §3.2.1].", True, "answer"),
        ("The chunks (x^1, p^0) and z^{s+1} go to the next stage [AIAYN §3.2.1].", True, "answer"),
        ("The Large (760 M) Transformer scores 53.3 [HNET Table 2].", True, "answer"),
        ("The base model trains for 300 K steps [AIAYN Table 3].", False, "fail"),           # still caught
        ("Pipelines suffer from 4 problems [AIAYN §3.2.1].", False, "fail"),                  # a real count, still checked
    ]
    P["HNET Table 2"] += " Large 760M"
    for s, ok, kind in cases:
        r = check_citations(s, P, Q)
        assert (r.ok, r.kind) == (ok, kind), (s, r)
    print("gate: ok", [check_citations(c[0], P, Q).derived for c in cases if "gap" in c[0]])


if __name__ == "__main__":
    main()
