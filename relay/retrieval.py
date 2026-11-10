"""Local hybrid retrieval over the chunked papers. No Groq calls.

  python -m relay.retrieval "how many attention heads does the base model use" --size 256 --k 5 [--rerank]

  * BM25 (rank_bm25) keeps exact tokens a dense model blurs: d_model, Mimi, "Table 3", 28.4.
  * Dense: FastEmbed BAAI/bge-small-en-v1.5 (ONNX, 384-d), cosine similarity.
  * Fused with reciprocal-rank fusion (RRF, k=60), so neither score scale dominates.
  * Optional re-rank of the fused pool with FastEmbed's cross-encoder Xenova/ms-marco-MiniLM-L-6-v2.
Each hit reports its ranks and raw scores; the top dense cosine and the re-rank score are the confidence signals
the extractive path uses to decide "the papers don't cover this" without a model call.
Embeddings are cached in data/index/ (git-ignored), keyed by a hash of the chunk file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "corpus"
INDEX = ROOT / "data" / "index"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"
TOKEN = re.compile(r"[a-z0-9]+(?:[._][a-z0-9]+)*")
RRF_K = 60


def tokens(s: str) -> list[str]:
    toks = TOKEN.findall(s.lower())
    # also index the parts of compound tokens, so "d_model" matches "d model" and "h-net" style splits
    return toks + [p for t in toks if ("_" in t or "." in t) and not t.replace(".", "").isdigit()
                   for p in re.split(r"[._]", t) if p]


def passage(c: dict) -> str:
    """What gets embedded and BM25-indexed: the citation and section path give each piece its context."""
    return f"{c['cite']} | {' > '.join(c['sec_path'])}\n{c['text']}"


@lru_cache(maxsize=1)
def embedder():
    from fastembed import TextEmbedding
    return TextEmbedding(EMBED_MODEL, cache_dir=str(INDEX / "models"))


@lru_cache(maxsize=1)
def reranker():
    from fastembed.rerank.cross_encoder import TextCrossEncoder
    return TextCrossEncoder(RERANK_MODEL, cache_dir=str(INDEX / "models"))


@dataclass
class Hit:
    chunk: dict
    score: float          # final ordering score (RRF, or re-rank score when re-ranked)
    bm25_rank: int | None
    dense_rank: int | None
    dense_cos: float
    rerank: float | None = None


class Index:
    def __init__(self, size: int):
        self.size = size
        path = CORPUS / f"chunks_{size}.jsonl"
        raw = path.read_bytes()
        self.chunks = [json.loads(l) for l in raw.decode("utf-8").splitlines() if l.strip()]
        texts = [passage(c) for c in self.chunks]
        self.bm25 = BM25Okapi([tokens(t) for t in texts])
        key = hashlib.sha256(raw + EMBED_MODEL.encode()).hexdigest()[:16]
        cache = INDEX / f"emb_{size}_{key}.npy"
        if cache.exists():
            self.emb = np.load(cache)
        else:
            INDEX.mkdir(parents=True, exist_ok=True)
            self.emb = np.array(list(embedder().passage_embed(texts)), dtype=np.float32)
            self.emb /= np.linalg.norm(self.emb, axis=1, keepdims=True)
            np.save(cache, self.emb)

    def search(self, query: str, k: int = 5, rerank: bool = False, pool: int = 30,
               papers: list[str] | None = None, dense: bool = True) -> tuple[list[Hit], dict]:
        """`papers` restricts the search to those papers (used by the decompose branch: one sub-query per paper).
        `dense=False` is BM25 only (the naive baseline); the dense cosine is still computed for the confidence signal."""
        t0 = time.perf_counter()
        bm = self.bm25.get_scores(tokens(query))
        q = np.array(next(iter(embedder().query_embed([query]))), dtype=np.float32)
        q /= np.linalg.norm(q)
        cos = self.emb @ q
        if papers:
            keep = np.array([c["paper"] in papers for c in self.chunks])
            bm = np.where(keep, bm, -np.inf)
            cos = np.where(keep, cos, -np.inf)
        bm_order = np.argsort(-bm)[:pool]
        dn_order = np.argsort(-cos)[:pool]
        t1 = time.perf_counter()
        bm_rank = {int(i): r for r, i in enumerate(bm_order) if bm[i] > 0}
        dn_rank = {int(i): r for r, i in enumerate(dn_order) if np.isfinite(cos[i])} if dense else {}
        fused = {}
        for i in set(bm_rank) | set(dn_rank):
            fused[i] = sum(1.0 / (RRF_K + 1 + rk[i]) for rk in (bm_rank, dn_rank) if i in rk)
        cand = sorted(fused, key=fused.get, reverse=True)[:pool]
        hits = [Hit(self.chunks[i], fused[i], bm_rank.get(i), dn_rank.get(i), float(cos[i])) for i in cand]
        if rerank and hits:
            scores = list(reranker().rerank(query, [passage(h.chunk) for h in hits]))
            for h, s in zip(hits, scores):
                h.rerank = float(s)
                h.score = float(s)
            hits.sort(key=lambda h: h.score, reverse=True)
        t2 = time.perf_counter()
        timing = {"retrieve_ms": round((t1 - t0) * 1000, 1), "fuse_rerank_ms": round((t2 - t1) * 1000, 1),
                  "top_dense_cos": float(cos.max())}
        return hits[:k], timing


@lru_cache(maxsize=4)
def get_index(size: int) -> Index:
    return Index(size)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--rerank", action="store_true")
    a = ap.parse_args()
    t0 = time.perf_counter()
    idx = get_index(a.size)
    print(f"index size={a.size}: {len(idx.chunks)} pieces, loaded in {(time.perf_counter() - t0) * 1000:.0f} ms")
    hits, timing = idx.search(a.query, a.k, a.rerank)
    hits, timing = idx.search(a.query, a.k, a.rerank)   # second call: warm timing, as the server would see it
    print(f"timing (warm): {timing}")
    for h in hits:
        c = h.chunk
        print(f"\n[{c['cite']}] score={h.score:.4f} bm25#{h.bm25_rank} dense#{h.dense_rank} cos={h.dense_cos:.3f}"
              f"{'' if h.rerank is None else f' rerank={h.rerank:.2f}'} ({c['tokens']} tok)")
        print("   " + c["text"][:260].replace("\n", " "))


if __name__ == "__main__":
    main()
