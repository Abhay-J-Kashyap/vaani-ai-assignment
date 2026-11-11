"""Load generator: run one Table 2 scenario through the pipeline, on the stand-in or on real Groq.

  python -m load.run_load overload                       # stand-in (starts its own on a spare port)
  python -m load.run_load steady --source real --dry-run # projected spend; then --confirm-real
  python -m load.run_load burst_retry_naive              # heading 6 experiment (stand-in)

Simulated people:
  * arrive as a Poisson process at the scenario's rate (a burst is many arrivals in a few seconds);
  * ask one of the 20 gold questions, drawn with replacement. A stated REPEAT_SHARE (25%) word it exactly as an
    earlier person might, so the exact-match answer cache can serve them; everyone else is treated as differently
    worded and bypasses the cache. (With 20 questions, unrestricted repeats were 83% of an overload run, and cache
    hits made overload look easy: first-token p50 48 ms. Those rows are kept as overload_hirepeat.);
  * each has a patience drawn from a lognormal with median 12 s (about 10% give up within 6 s, 10% wait over 26 s).
    The pipeline is told it, and sheds to quoted passages rather than make someone wait past it. If the answer
    still isn't complete when patience runs out, the person gives up: the request is cancelled and its row says
    'timeout'. This distribution is an assumption, not a measurement (write-up heading 8).
Rows go to results/pipeline_rows_<scenario>.csv (source stand-in or real), events to results/trace.jsonl.
"""
from __future__ import annotations

import argparse
import asyncio
import random
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from eval.retrieval_scan import load_questions
from relay import settings as S
from relay.ledger import DRAFTER, ROUTER, Ledger
from relay.pipeline import Relay, log_paths

ROOT = Path(__file__).resolve().parents[1]
PATIENCE_MEDIAN_S, PATIENCE_SIGMA = 12.0, 0.6
REPEAT_SHARE = 0.25


@dataclass
class Scenario:
    name: str
    rate_per_min: float          # mean arrival rate outside bursts
    duration_s: float
    burst: int = 0               # extra arrivals packed into the first burst_window_s
    burst_window_s: float = 5.0
    tpm_scale: float = 1.0       # 0.5 = per-minute limit halved (stand-in only)
    tpd_exhausted: bool = False  # drafter's daily budget at 0 when the run starts (stand-in)
    fault_503: float | None = None
    retry_mode: str = "policy"
    stand_in_only: bool = False
    notes: list[str] = field(default_factory=list)


SCENARIOS = {
    # within capacity: ~2.7 Full answers/min fit the drafter's 8K TPM at ~3K reserved per call
    "steady": Scenario("steady", rate_per_min=2.0, duration_s=600),
    # an incident: 20 people within 5 s, then a trickle
    "burst": Scenario("burst", rate_per_min=1.0, duration_s=180, burst=20),
    # sustained overload: ~7x the Full-answer capacity for 5 minutes
    "overload": Scenario("overload", rate_per_min=20.0, duration_s=300, stand_in_only=True),
    # the steady load again with the per-minute limit halved (TPM 4,000 on both models)
    "tpm_half": Scenario("tpm_half", rate_per_min=4.0, duration_s=300, tpm_scale=0.5, stand_in_only=True),
    # the drafter's daily budget already gone (the real version runs on the last evening; this is the fallback)
    "daily_exhausted": Scenario("daily_exhausted", rate_per_min=4.0, duration_s=180, tpd_exhausted=True,
                                stand_in_only=True),
    # heading 6, first attempt (kept): a burst under 15% faults. Admission shed most of it before any model call,
    # so only 13 drafter calls ran and retries barely happened (naive 2, policy 0) - no difference to measure.
    "burst_retry_naive": Scenario("burst_retry_naive", rate_per_min=2.0, duration_s=120, burst=24,
                                  fault_503=0.15, retry_mode="naive", stand_in_only=True),
    "burst_retry_policy": Scenario("burst_retry_policy", rate_per_min=2.0, duration_s=120, burst=24,
                                   fault_503=0.15, retry_mode="policy", stand_in_only=True),
    # heading 6: a load the per-minute budget can mostly serve, with 25% of calls hitting the measured 503
    # "over capacity" fault. A failed call has already reserved its per-minute tokens, so retries compete with
    # other people's first attempts. Same seed for both, so both get the same arrivals, questions and patience.
    "retry_naive": Scenario("retry_naive", rate_per_min=3.0, duration_s=300, fault_503=0.25, retry_mode="naive",
                            stand_in_only=True),
    "retry_policy": Scenario("retry_policy", rate_per_min=3.0, duration_s=300, fault_503=0.25, retry_mode="policy",
                             stand_in_only=True),
}


def arrivals(sc: Scenario, rng: random.Random) -> list[float]:
    t, out = 0.0, sorted(rng.uniform(0, sc.burst_window_s) for _ in range(sc.burst))
    while True:
        t += rng.expovariate(sc.rate_per_min / 60)
        if t >= sc.duration_s:
            return sorted(out)
        out.append(t)


def start_standin(port: int, sc: Scenario, seed: int) -> subprocess.Popen:
    args = [sys.executable, "-m", "standin.server", "--port", str(port), "--seed", str(seed)]
    if sc.fault_503 is not None:
        args += ["--fault-503", str(sc.fault_503)]
    proc = subprocess.Popen(args, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(60):
        try:
            httpx.get(f"http://127.0.0.1:{port}/admin/state", timeout=0.5)
            break
        except httpx.HTTPError:
            time.sleep(0.25)
    cfg = {"reset": True, "tpm_scale": sc.tpm_scale}
    if sc.tpd_exhausted:
        cfg["tpd_level"] = {DRAFTER: 0}
    httpx.post(f"http://127.0.0.1:{port}/admin/config", json=cfg, timeout=5)
    return proc


async def person(relay: Relay, q: str, patience: float, delay: float, cache_ok: bool, out: list) -> None:
    await asyncio.sleep(delay)
    try:
        res = await asyncio.wait_for(relay.ask(q, patience_s=patience, use_cache=cache_ok), timeout=patience)
        out.append(res)
    except asyncio.TimeoutError:
        out.append({"status": "gave_up"})
    except Exception as e:                                           # noqa: BLE001 -- counted, run continues
        out.append({"status": "exception", "error": f"{type(e).__name__}: {e}"[:200]})


async def run(sc: Scenario, source: str, seed: int, port: int, dev: bool) -> None:
    rng = random.Random(seed)
    qs = list(load_questions().values())
    times = arrivals(sc, rng)
    picks = [rng.choice(qs) for _ in times]
    patience = [min(120.0, rng.lognormvariate(__import__("math").log(PATIENCE_MEDIAN_S), PATIENCE_SIGMA))
                for _ in times]
    cache_ok = [rng.random() < REPEAT_SHARE for _ in times]
    print(f"{sc.name}: {len(times)} requests over {sc.duration_s:.0f} s (burst {sc.burst}), source={source}, "
          f"cache-eligible wording {sum(cache_ok)} ({100 * sum(cache_ok) / max(1, len(times)):.1f}%), "
          f"retry={sc.retry_mode}", flush=True)

    proc = None
    if source == "stand-in":
        proc = start_standin(port, sc, seed)
        ledger = Ledger(f"load_{sc.name}", persist=False)
        for m in (DRAFTER, ROUTER):
            a = ledger.acct(m)
            a.tpm.capacity, a.tpm.per_s = 8000 * sc.tpm_scale, 8000 * sc.tpm_scale / 60
            a.tpm.set_level(a.tpm.capacity)
        if sc.tpd_exhausted:
            ledger.acct(DRAFTER).tpd.set_level(0)
        base = f"http://127.0.0.1:{port}/v1"
    else:
        ledger, base = Ledger("real"), None
    rows, trace = log_paths(source, "dev" if dev else sc.name)
    relay = Relay(source, S.DEFAULT_SETTING, sc.name, ledger=ledger, base_url=base, use_cache=True,
                  retry_mode=sc.retry_mode, rows_path=rows, trace_path=trace)
    if source == "stand-in":
        relay.cache.data.clear()                                     # each stand-in scenario starts cold
    out: list = []
    t0 = time.perf_counter()
    try:
        await asyncio.gather(*(person(relay, q, p, d, c, out) for q, p, d, c in zip(picks, patience, times, cache_ok)))
    finally:
        await relay.close()
        if proc:
            proc.terminate()
    took = time.perf_counter() - t0
    by = {}
    for o in out:
        k = o["status"] if o.get("status") in ("gave_up", "exception") else f"{o['status']}/{o['tier']}"
        by[k] = by.get(k, 0) + 1
    print(f"done in {took:.0f} s: {dict(sorted(by.items()))}")
    print(f"rows -> {rows.relative_to(ROOT)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("scenario", choices=list(SCENARIOS))
    ap.add_argument("--source", choices=["stand-in", "real"], default="stand-in")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--port", type=int, default=8091)
    ap.add_argument("--dev", action="store_true", help="log to data/dev/ instead of results/ (rehearsal)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--confirm-real", action="store_true")
    a = ap.parse_args()
    sc = SCENARIOS[a.scenario]
    if a.source == "real":
        if sc.stand_in_only:
            sys.exit(f"{sc.name} runs on the stand-in only (author's decision, 2026-11-09)")
        n = len(arrivals(sc, random.Random(a.seed)))
        led = Ledger("real")
        print(f"{sc.name} on real Groq: ~{n} requests; worst case ~{n * 3000:,} drafter tokens if all Full "
              f"(the ledger downgrades or sheds beyond the budget). Drafter level now ~{led.acct(DRAFTER).tpd.level():,.0f}.")
        if a.dry_run or not a.confirm_real:
            sys.exit(0 if a.dry_run else "spends real budget: --dry-run first, then --confirm-real")
    asyncio.run(run(sc, a.source, a.seed, a.port, a.dev))


if __name__ == "__main__":
    main()
