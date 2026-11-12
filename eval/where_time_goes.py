"""Heading 4: split one slow request's latency into waiting, working and retrying, before and after its first token.

  python -m eval.where_time_goes                         # slowest real load request (steady/burst/drain) by total_ms
  python -m eval.where_time_goes --scenario eval_v2      # slowest in another scenario
  python -m eval.where_time_goes --rid <request_id>

From results/trace.jsonl (times are ms since the request arrived):
  waiting  = our admission queue (admit_wait_start -> admit_wait_end) + Groq's own queue inside each call
             (usage.queue_time, reported by Groq) + time before the request was admitted at all
  working  = routing, retrieval and decide (our CPU), the model's prompt processing and generation (call time minus
             Groq's queue), and the gate (between the last token and the sentence leaving)
  retrying = retry_wait_start -> retry_wait_end, plus the whole duration of any call attempt that failed
Each interval is split at the first answer token, so we can say whether the delay came before or after it.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def events(rid: str) -> list[dict]:
    out = []
    with (ROOT / "results" / "trace.jsonl").open(encoding="utf-8") as f:
        for line in f:
            if rid in line:
                e = json.loads(line)
                if e["rid"] == rid:
                    out.append(e)
    return out


def split(a: float, b: float, cut: float | None) -> tuple[float, float]:
    """Milliseconds of [a, b] before and after the first token."""
    if cut is None:
        return b - a, 0.0
    return max(0.0, min(b, cut) - a), max(0.0, b - max(a, cut))


def analyse(rid: str) -> None:
    ev = events(rid)
    if not ev:
        raise SystemExit(f"no trace events for {rid}")
    ft = next((e["t_ms"] for e in ev if e["ev"] == "first_answer_token"), None)
    done = next(e for e in ev if e["ev"] == "done")
    parts: list[tuple[str, str, float, float]] = []      # (kind, what, start, end)
    open_ = {}
    for e in ev:
        k = e["ev"]
        if k == "admit_wait_start":
            open_[("admit", e.get("model"))] = e["t_ms"]
        elif k == "admit_wait_end" and ("admit", e.get("model")) in open_:
            parts.append(("waiting", f"admission queue ({e['model'].split('/')[-1]})",
                          open_.pop(("admit", e.get("model"))), e["t_ms"]))
        elif k == "route_start":
            open_["route"] = e["t_ms"]
        elif k == "route_end" and "route" in open_:
            s = open_.pop("route")
            q = 1000 * float((e.get("usage") or {}).get("queue_time") or 0)
            parts += [("waiting", "Groq queue (router call)", s, s + q), ("working", "router tool call", s + q, e["t_ms"])]
        elif k == "retrieve_start":
            open_.setdefault("retrieve", e["t_ms"])
        elif k == "decide" and "retrieve" in open_:
            parts.append(("working", "retrieval loop + decide (local)", open_.pop("retrieve"), e["t_ms"]))
        elif k == "call_start":
            open_["call"] = (e["t_ms"], e.get("attempt", 0))
        elif k == "call_end" and "call" in open_:
            s, _ = open_.pop("call")
            if e.get("status") != "ok":
                parts.append(("retrying", f"failed call ({e.get('status')})", s, e["t_ms"]))
                continue
            q = 1000 * float((e.get("usage") or {}).get("queue_time") or 0)
            fb = e.get("first_byte_ms")
            parts.append(("waiting", "Groq queue (drafter call)", s, s + q))
            if fb is not None:
                parts += [("working", "drafter: prompt processing to first byte", s + q, s + fb),
                          ("working", "drafter: generating the answer", s + fb, e["t_ms"])]
            else:
                parts.append(("working", "drafter call", s + q, e["t_ms"]))
        elif k == "retry_wait_start":
            open_["retry"] = e["t_ms"]
        elif k == "retry_wait_end" and "retry" in open_:
            parts.append(("retrying", "retry backoff wait", open_.pop("retry"), e["t_ms"]))
    total = done["t_ms"]
    print(f"Request {rid}: status={done.get('status')} tier={done.get('tier')} branch={done.get('branch')} "
          f"retries={done.get('retries')}; first answer token at {ft} ms, done at {total} ms\n")
    print(f"{'kind':<9} {'what':<44} {'from':>8} {'to':>8} {'before 1st token':>17} {'after':>8}")
    sums = {}
    for kind, what, a, b in sorted(parts, key=lambda p: p[2]):
        pre, post = split(a, b, ft)
        sums.setdefault(kind, [0.0, 0.0])
        sums[kind][0] += pre
        sums[kind][1] += post
        print(f"{kind:<9} {what:<44} {a:>8.0f} {b:>8.0f} {pre:>17.0f} {post:>8.0f}")
    covered = sum(v[0] + v[1] for v in sums.values())
    print(f"\n{'total':<54} {'':>8} {'':>8} {sum(v[0] for v in sums.values()):>17.0f} "
          f"{sum(v[1] for v in sums.values()):>8.0f}   (unattributed: {total - covered:.0f} ms)")
    for kind, (pre, post) in sums.items():
        print(f"  {kind:<9} {pre + post:>7.0f} ms = {100 * (pre + post) / total:5.1f}% of the request "
              f"({pre:.0f} ms before the first token, {post:.0f} ms after)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rid")
    ap.add_argument("--scenario", nargs="+", default=["steady", "burst", "drain", "daily_exhausted"])
    a = ap.parse_args()
    rid = a.rid
    if not rid:
        rows = [r for r in csv.DictReader((ROOT / "results.csv").open(encoding="utf-8"))
                if r["scenario"] in a.scenario and r["source"] == "real" and r["total_ms"]]
        if not rows:
            raise SystemExit(f"no real rows in {a.scenario} yet")
        rid = max(rows, key=lambda r: float(r["total_ms"]))["request_id"]
    analyse(rid)


if __name__ == "__main__":
    main()
