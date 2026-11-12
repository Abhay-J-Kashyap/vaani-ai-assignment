"""Code-review fixes of 2026-11-13 (ADR): the daily-token reservation and the archive path check. No Groq.

    python -m tests.test_budget

1. Ledger: a call reserves prompt + max_tokens against TPD when it starts; finish() swaps that for the actual
   usage; a failed or rejected call gets it all back.
2. Tier: with room above the reserve floor for one Full answer, two requests decided before either has drafted
   get one Full and one Lean (before the fix, both were promised Full).
3. The same through the real pipeline on the stand-in: two concurrent requests near the floor, one Full.
4. corpus.fetch.safe_extract: a member that resolves to a sibling folder sharing the prefix is skipped.
"""
from __future__ import annotations

import asyncio
import io
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

import httpx

from corpus.fetch import safe_extract
from relay import settings as S
from relay.ledger import DRAFTER, ROUTER, Ledger
from relay.limits import error_429
from relay.pipeline import Relay

PORT = 8097
URL = f"http://127.0.0.1:{PORT}"
OUT = Path(__file__).resolve().parents[1] / "data" / "dev"
Q1 = "Why are the query-key dot products divided by the square root of the key dimension?"
Q2 = "How many attention heads does the base Transformer use, and what is the dimension of each head?"


def test_reservation() -> None:
    L = Ledger("t_res", persist=False)
    tpd = L.acct(DRAFTER).tpd
    t = time.time()
    tpd.set_level(100_000, t)
    assert L.start(DRAFTER, 2_000, 1_000, now=t) == 3_000
    assert abs(tpd.level(t) - 97_000) < 1, "in flight: the whole prompt + max_tokens is held against TPD"
    L.finish(DRAFTER, {"total_tokens": 2_400, "prompt_tokens_details": {"cached_tokens": 400}}, None, now=t,
             reserved=3_000)
    assert abs(tpd.level(t) - 98_000) < 1, "finished: only the uncached tokens actually used are charged"

    L.start(DRAFTER, 2_000, 1_000, now=t)
    L.finish(DRAFTER, None, None, now=t, reserved=3_000)            # timeout / 5xx: no usage reported
    assert abs(tpd.level(t) - 98_000) < 1, "a failed call gets its reservation back"

    L.start(DRAFTER, 2_000, 1_000, now=t)
    L.on_429(DRAFTER, error_429(DRAFTER, "tpm", 8_000, 7_000, 3_000, 7.5), now=t, reserved=3_000)
    assert abs(tpd.level(t) - 98_000) < 1, "a rejected call is not counted (measured), so it is refunded"
    print("ledger  : ok (reserve at start, swap for usage at finish, refund on failure and on a 429)")


def test_tier_committed() -> None:
    L = Ledger("t_tier", persist=False)
    full, lean = 3_000, 1_500
    floor = L.reserve_floor()
    L.acct(DRAFTER).tpd.set_level(floor + 1.5 * full)                  # room above the floor for one Full answer
    first = L.tier(full, lean, committed=0)
    second_old = L.tier(full, lean)                                     # the old rule: nothing promised is counted
    second = L.tier(full, lean, committed=full)                         # the fix: the first request's promise counts
    assert (first, second_old, second) == ("full", "full", "lean"), (first, second_old, second)
    print(f"tier    : ok (one Full of room: old rule gave {first}+{second_old}, now {first}+{second})")


async def test_pipeline_concurrent() -> None:
    proc = subprocess.Popen([sys.executable, "-m", "standin.server", "--port", str(PORT), "--fault-503", "0"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(50):
            try:
                httpx.get(f"{URL}/admin/state", timeout=0.5)
                break
            except httpx.HTTPError:
                time.sleep(0.2)
        OUT.mkdir(parents=True, exist_ok=True)
        rl = Relay("stand-in", "h256k8", "test_budget", ledger=Ledger("t_pipe", persist=False),
                   base_url=f"{URL}/v1", rows_path=OUT / "test_budget_rows.csv",
                   trace_path=OUT / "test_budget_trace.jsonl", use_cache=False)
        full_cost = max(rl.est_prompt(q, rl.setting.k) for q in (Q1, Q2)) + S.FULL_MAX_TOKENS
        rl.ledger.acct(DRAFTER).tpd.set_level(rl.ledger.reserve_floor() + 1.5 * full_cost)
        rl.ledger.acct(ROUTER).tpd.set_level(200_000)
        res = await asyncio.gather(rl.ask(Q1, patience_s=20), rl.ask(Q2, patience_s=20))
        await rl.close()
        tiers = sorted(x["tier"] for x in res)
        assert tiers == ["full", "lean"], tiers
        assert rl.full_held == 0, "every promise is released when its request ends"
        assert all(x["answer"] for x in res), "nobody turned away"
        print(f"pipeline: ok (two concurrent requests, room for one Full: tiers {tiers})")
    finally:
        proc.terminate()


def test_safe_extract() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        dest = Path(tmp) / "src"
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tf:
            for name, data in (("main.tex", b"ok"), ("../src_evil/payload.tex", b"bad"), ("../../up.tex", b"bad")):
                info = tarfile.TarInfo(name)
                info.size = len(data)
                tf.addfile(info, io.BytesIO(data))
        written = safe_extract(buf.getvalue(), dest)
        assert (dest / "main.tex").exists() and written == ["main.tex"], written
        assert not (Path(tmp) / "src_evil").exists(), "a sibling folder sharing the prefix must not be written"
        assert not (Path(tmp).parent / "up.tex").exists()
    print("extract : ok (sibling-prefix and parent paths skipped)")


if __name__ == "__main__":
    test_reservation()
    test_tier_committed()
    test_safe_extract()
    asyncio.run(test_pipeline_concurrent())
    print("\nall budget and extraction checks passed")
