"""How this library misleads a naive retriever: what plain BM25 (256-token pieces, top 8, no table lookup, no
per-paper search, no whole tables) returns for one example question per failure mode. No Groq.

  python -m eval.naive_retriever
Also checks the raw LaTeX: an ingester that indexed every .tex file would also index text the papers never publish.
"""
from __future__ import annotations

import re
from pathlib import Path

from relay.retrieval import get_index

ROOT = Path(__file__).resolve().parents[1]
CASES = [
    ("1. Every paper has a Table 1-3",
     "According to Table 3, what are the hyperparameters of the original Transformer base model?",
     "want AIAYN Table 3"),
    ("1. Every paper has a Table 1-3",
     "According to Table 1, what are the Depth Transformer's model dimension, MLP dimension, heads and layers?",
     "want MOSHI Table 1"),
    ("2. Introductions restate the method in the same words",
     "Describe the H-Net pipeline: encoder, dynamic chunking, main network, dechunking and decoder. What does each do?",
     "want HNET §2.1.1 (method), not HNET §1 (introduction)"),
    ("3. The papers contradict themselves",
     "What BLEU score does the Transformer reach on WMT 2014 English-to-French?",
     "Abstract and Table 2 say 41.8; §6.1 prose says 41.0"),
    ("3. The papers contradict themselves",
     "By how much does H-Net (2-stage) beat the BPE Transformer at XL scale?",
     "§3.1 prose says +2.6; Table 2 gives 58.2 - 55.5 = 2.7"),
]


def dense_only(idx, q: str, k: int = 8) -> list[str]:
    from relay.retrieval import embedder
    import numpy as np
    v = np.array(next(iter(embedder().query_embed([q]))), dtype=np.float32)
    order = np.argsort(-(idx.emb @ (v / np.linalg.norm(v))))[:k]
    return [idx.chunks[int(i)]["cite"] for i in order]


def main() -> None:
    idx = get_index(256)
    for mode, q, want in CASES:
        hits, _ = idx.search(q, 8, dense=False)
        print(f"{mode}\n  Q: {q}\n  {want}\n  BM25 top 8: {[h.chunk['cite'] for h in hits]}")
        print(f"  dense-only top 8: {dense_only(idx, q)}")
        for h in hits[:8]:
            for num in ("41.8", "41.0", "2.6", "2.7"):
                if mode.startswith("3") and num in h.chunk["text"]:
                    print(f"     {h.chunk['cite']} contains {num}")
        print()
    src = next((ROOT / "data" / "raw" / "1706.03762v7").rglob("ms.tex"), None)
    if not src:
        print("4. (raw LaTeX not fetched: run python -m corpus.fetch)")
        return
    live = src.read_text(encoding="utf-8", errors="replace")
    for f in sorted(src.parent.glob("*.tex")):
        if re.search(rf"^\s*%\s*\\input\{{{f.stem}\}}", live, flags=re.M):
            text = f.read_text(encoding="utf-8", errors="replace")
            body = " ".join(re.sub(r"\\[a-zA-Z]+|[{}$%]", " ", text).split())
            print(f"4. Unpublished source: {f.relative_to(ROOT)} ({len(text.split())} words) ships in the arXiv "
                  f"source, but its \\input is commented out of ms.tex, so it is not in the published paper.\n"
                  f"   It begins: {body[:220]}...")


if __name__ == "__main__":
    main()
