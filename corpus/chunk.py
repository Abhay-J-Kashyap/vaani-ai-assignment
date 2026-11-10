"""Cut the parsed blocks into retrievable pieces at a target size: data/corpus/chunks_<size>.jsonl.

  python -m corpus.chunk --sizes 128 256 512

Rules (ADR: children-only index, carrying their section path):
  * A piece never crosses a section boundary. Consecutive paragraphs of one section are packed up to the target.
  * A paragraph longer than the target is split at sentence boundaries.
  * A table is its own piece. A large table is split into row groups, and each group repeats the caption and the
    header rows, so a number is never separated from what it measures.
  * Figure captions and the abstract are their own pieces.
Every piece carries `cite`, the label the drafter must use and the citation gate checks: "AIAYN §3.2.1",
"AIAYN Table 3", "MOSHI Figure 4", "HNET §A.1", "AIAYN Abstract".
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from corpus.build import OUT as BLOCKS, ntok

ROOT = Path(__file__).resolve().parents[1]
SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\"'])")


def cite_of(b: dict) -> str:
    if b["kind"] == "abstract":
        return f"{b['paper']} Abstract"
    if b["float_num"]:
        return f"{b['paper']} {b['float_num']}"
    if b["sec_num"]:
        return f"{b['paper']} §{b['sec_num']}"
    # Unnumbered heading (e.g. H-Net's run-in headings inside §1): cite the nearest numbered parent section, as a
    # reader of the PDF would. Unnumbered top-level matter keeps its title; text before any heading is front matter.
    for part in reversed(b["sec_path"]):
        m = re.match(r"([A-Z]|\d+)(\.\d+)*\s", part)
        if m:
            return f"{b['paper']} §{part.split()[0]}"
    if b["sec_path"]:
        return f"{b['paper']} {b['sec_path'][-1]}"
    return f"{b['paper']} front matter"


def split_long(text: str, size: int) -> list[str]:
    out, cur = [], ""
    for s in SENT.split(text):
        if cur and ntok(cur + " " + s) > size:
            out.append(cur)
            cur = s
        else:
            cur = f"{cur} {s}".strip()
    if cur:
        out.append(cur)
    return out


def table_pieces(b: dict, size: int) -> list[str]:
    rows = [r for r in b["text"].split("\n") if r.strip()]
    head = f"{b['float_num']}: {b['caption']}"
    if ntok(head + "\n" + "\n".join(rows)) <= max(size, 200):
        return [head + "\n" + "\n".join(rows)]
    header_rows = rows[:2]
    pieces, cur = [], []
    budget = max(size - ntok(head + "\n" + "\n".join(header_rows)), 60)
    for r in rows[2:]:
        if cur and ntok("\n".join(cur + [r])) > budget:
            pieces.append(cur)
            cur = []
        cur.append(r)
    if cur:
        pieces.append(cur)
    n = len(pieces)
    return [f"{head} (rows part {i + 1}/{n})\n" + "\n".join(header_rows + p) for i, p in enumerate(pieces)]


def chunk(blocks: list[dict], size: int) -> list[dict]:
    pieces, buf = [], []

    def emit(kind: str, b: dict, text: str):
        pieces.append({"id": f"{b['paper']}-{size}-{len(pieces):04d}", "paper": b["paper"], "arxiv": b["arxiv"],
                       "cite": cite_of(b), "kind": kind, "sec_num": b["sec_num"], "sec_path": b["sec_path"],
                       "appendix": b["appendix"], "text": text, "tokens": ntok(text)})

    def flush():
        if buf:
            emit("text", buf[0], "\n".join(x["text"] for x in buf))
            buf.clear()

    for b in blocks:
        if b["kind"] == "text":
            if buf and (buf[0]["paper"], buf[0]["sec_path"]) != (b["paper"], b["sec_path"]):
                flush()
            if b["tokens"] > size:
                flush()
                for part in split_long(b["text"], size):
                    emit("text", b, part)
                continue
            if buf and ntok("\n".join(x["text"] for x in buf + [b])) > size:
                flush()
            buf.append(b)
        else:
            flush()
            if b["kind"] == "table":
                for t in table_pieces(b, size):
                    emit("table", b, t)
            elif b["kind"] in ("figure", "algorithm"):
                emit(b["kind"], b, f"{b['float_num']}: {b['caption']}" + (f"\n{b['text']}" if b["text"] else ""))
            else:   # abstract
                for part in split_long(b["text"], size) if b["tokens"] > size else [b["text"]]:
                    emit("abstract", b, part)
    flush()
    return pieces


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", nargs="+", type=int, default=[128, 256, 512])
    a = ap.parse_args()
    blocks = [json.loads(l) for l in BLOCKS.read_text(encoding="utf-8").splitlines()]
    for size in a.sizes:
        pieces = chunk(blocks, size)
        out = BLOCKS.parent / f"chunks_{size}.jsonl"
        with out.open("w", encoding="utf-8") as f:
            for p in pieces:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        toks = sorted(p["tokens"] for p in pieces)
        print(f"size {size:>4}: {len(pieces):>4} pieces, tokens p50={toks[len(toks) // 2]} max={toks[-1]} "
              f"total={sum(toks):,} -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
