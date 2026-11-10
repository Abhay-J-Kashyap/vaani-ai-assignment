"""Derive the stand-in's parameters from the raw probe file: standin/calibration.json.

  python -m probe.calibrate

Every number the stand-in uses comes from here, so "calibrated from what you measured" is checkable: rerun this on
results/probe_raw.jsonl and you get the same file. Only real (non-dry-run) records are used.

Model of one successful streamed call (all times client-side, ms):
  first answer token = groq queue_time + rest
    queue_time: a "calm" lognormal, plus correlated "episodes" (queue_time > 1 s) that hit requests close in time
    rest: lognormal whose median grows linearly with prompt tokens (prefill + reasoning + network)
  after the first token: tokens stream at the server rate, except a "trickle" share that streams at ~70 tok/s
"""
from __future__ import annotations

import json
import math
import statistics as st
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "probe_raw.jsonl"
OUT = ROOT / "standin" / "calibration.json"
EPISODE_MS = 1000


def lognorm(xs):
    logs = [math.log(x) for x in xs if x and x > 0]
    return {"mu": round(st.mean(logs), 4), "sigma": round(st.pstdev(logs), 4), "n": len(logs),
            "median_ms": round(math.exp(st.mean(logs)), 1)}


def linfit(pts):
    mx, my = st.mean(x for x, _ in pts), st.mean(y for _, y in pts)
    sxx = sum((x - mx) ** 2 for x, _ in pts)
    b = sum((x - mx) * (y - my) for x, y in pts) / sxx if sxx else 0.0
    return my - b * mx, b


def main() -> None:
    recs = [json.loads(l) for l in RAW.read_text(encoding="utf-8").splitlines() if l.strip()]
    recs = [r for r in recs if not r.get("dry_run")]
    cal = {"source": str(RAW.relative_to(ROOT)), "models": {}}
    by_model = defaultdict(list)
    for r in recs:
        if r["phase"] in ("latency", "concurrency") and r.get("status_code") == 200 and r.get("first_content_ms"):
            by_model[r["model"]].append(r)
    for model, rs in sorted(by_model.items()):
        q = [(r["usage"] or {}).get("queue_time", 0) * 1000 for r in rs if r.get("usage")]
        calm_q = [x for x in q if x < EPISODE_MS]
        episodes = [x for x in q if x >= EPISODE_MS]
        rest_pts, rest_resid = [], []
        for r in rs:
            u = r.get("usage") or {}
            if u.get("queue_time") is None or u["queue_time"] * 1000 >= EPISODE_MS:
                continue
            rest = r["first_content_ms"] - u["queue_time"] * 1000
            if rest > 0:
                rest_pts.append((u.get("prompt_tokens", 0), math.log(rest)))
        a, b = linfit(rest_pts)
        resid = [y - (a + b * x) for x, y in rest_pts]
        tps = [u["completion_tokens"] / u["completion_time"] for u in (r.get("usage") or {} for r in rs)
               if u.get("completion_time") and u.get("completion_tokens")]
        lat = [r for r in rs if r["phase"] == "latency"]
        trickle = [r for r in lat if r["total_ms"] - r["first_content_ms"] > 1000]
        trickle_rate = [((r.get("usage") or {}).get("completion_tokens", 0)) / ((r["total_ms"] - r["first_content_ms"]) / 1000)
                        for r in trickle]
        reasoning = [((r.get("usage") or {}).get("completion_tokens_details") or {}).get("reasoning_tokens", 0) for r in rs]
        cal["models"][model] = {
            "n_calls": len(rs),
            "queue_calm_ms": lognorm(calm_q),
            "queue_episode": {"threshold_ms": EPISODE_MS, "share": round(len(episodes) / len(q), 4) if q else 0,
                              "extra_ms": lognorm(episodes) if len(episodes) >= 3 else None,
                              "note": "episodes are correlated: requests arriving together are delayed together"},
            "rest_first_token": {"log_ms_intercept": round(a, 4), "log_ms_per_prompt_token": round(b, 7),
                                 "sigma": round(st.pstdev(resid), 4), "n": len(rest_pts)},
            "server_tokens_per_s": {"median": round(st.median(tps), 1), "p10": round(sorted(tps)[len(tps) // 10], 1)},
            "trickle": {"share": round(len(trickle) / len(lat), 4) if lat else 0, "n": len(trickle),
                        "tokens_per_s": round(st.median(trickle_rate), 1) if trickle_rate else None},
            "reasoning_tokens_median": st.median(reasoning) if reasoning else 0,
        }
    # Episode length is fitted on both gpt-oss models pooled (the 120b alone has too few episodes);
    # each model keeps its own episode share.
    pooled = [(r["usage"] or {}).get("queue_time", 0) * 1000 for rs in by_model.values() for r in rs
              if r.get("usage") and (r["usage"].get("queue_time") or 0) * 1000 >= EPISODE_MS]
    for m in cal["models"].values():
        m["queue_episode"]["extra_ms_pooled"] = lognorm(pooled) if len(pooled) >= 3 else None
    r429 = [r["total_ms"] for r in recs if r.get("status_code") == 429]
    r503 = [r for r in recs if r.get("status_code") == 503]
    # 503s were only seen on qwen (2 of ~80); gpt-oss had none in ~200 calls. Use the rate over all chat calls.
    chat = [r for r in recs if r.get("status_code") in (200, 503)]
    cal["rate_limited_response_ms"] = lognorm(r429)
    cal["over_capacity_503"] = {"share": round(len(r503) / len(chat), 4) if chat else 0, "n": len(r503),
                                "delay_ms": round(st.median(r["total_ms"] for r in r503), 1) if r503 else None,
                                "seen_on": sorted({r["model"] for r in r503})}
    cal["limits"] = {"rpm": 30, "tpm": 8000, "rpd": 1000, "tpd": 200000,
                     "refill": "continuous token buckets; TPM reserves prompt + max_tokens; TPD charges actual usage",
                     "evidence": "probe_raw.jsonl headers + 429 bodies; results/groq_console_limits_2026-11-09.md"}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(cal, indent=2), encoding="utf-8")
    print(json.dumps(cal, indent=2))


if __name__ == "__main__":
    main()
