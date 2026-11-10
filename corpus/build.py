"""Parse the three LaTeX sources into structured blocks: data/corpus/blocks.jsonl.

  python -m corpus.build            # parse + print a structure summary to check against the PDFs

Each block is a paragraph, a table (caption + rows), a figure caption or an algorithm, carrying the paper key,
the LaTeX-style section number and path, and the float number ("Table 3"), so every citation can name a place a
reader can find in the PDF.
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import tiktoken

from corpus.fetch import PAPERS, RAW
from corpus.latex import parse

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "corpus" / "blocks.jsonl"
MAIN = {"AIAYN": "ms.tex", "MOSHI": "main.tex", "HNET": "main.tex"}
ENC = tiktoken.get_encoding("o200k_base")   # the base of gpt-oss's o200k_harmony tokenizer


def ntok(s: str) -> int:
    return len(ENC.encode(s, disallowed_special=()))   # H-Net's text contains a literal "<|endoftext|>"


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for key, (aid, ver, title) in PAPERS.items():
        blocks = parse(RAW / f"{aid}v{ver}" / "src", MAIN[key])
        kinds = Counter(b.kind for b in blocks)
        toks = sum(ntok(b.text + b.caption) for b in blocks)
        print(f"\n=== {key} ({aid}v{ver}): {len(blocks)} blocks {dict(kinds)}, ~{toks:,} tokens")
        seen = set()
        for b in blocks:
            if b.sec_path and tuple(b.sec_path) not in seen:
                seen.add(tuple(b.sec_path))
                print(f"   {'  ' * (len(b.sec_path) - 1)}{b.sec_path[-1]}{'  [appendix]' if b.appendix else ''}")
        for b in blocks:
            if b.float_num:
                print(f"      {b.float_num:<11} ({b.sec_num or '-'}) {b.caption[:90]}")
        for i, b in enumerate(blocks):
            rows.append({"paper": key, "arxiv": f"{aid}v{ver}", "block": i, **asdict(b),
                         "tokens": ntok(b.text + b.caption)})
    with OUT.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n{len(rows)} blocks -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
