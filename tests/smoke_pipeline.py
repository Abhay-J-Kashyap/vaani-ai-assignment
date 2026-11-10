"""Pipeline smoke test against the stand-in (no Groq). Starts its own stand-in on a spare port.

  python -m tests.smoke_pipeline

Covers each path once: single-paper answer; cross-paper DECOMPOSE; low-confidence REFUSE; cache hit; a burst of
concurrent requests through FIFO admission; per-minute limit halved (shedding / Lean / extractive); daily budget
exhausted (the 3pm answer: extractive, zero tokens, honest wait); a client giving up (cancellation -> timeout row).
"""
from __future__ import annotations

import asyncio
import csv
import subprocess
import sys
import time
from pathlib import Path

import httpx

from relay.ledger import DRAFTER, ROUTER, Ledger
from relay.pipeline import Relay

PORT = 8098
URL = f"http://127.0.0.1:{PORT}"
OUT = Path(__file__).resolve().parents[1] / "data" / "dev"
ROWS, TRACE = OUT / "smoke_rows.csv", OUT / "smoke_trace.jsonl"

Q_SINGLE = "Why are the query-key dot products divided by the square root of the key dimension?"
Q_CROSS = ("Compare the sinusoidal positional encoding in Attention Is All You Need with the dynamic chunking "
           "mechanism in H-Net. What problem does each solve?")
Q_OUT = "What was the exact training expenditure in US dollars for Moshi, including electricity and storage?"


def admin(**cfg):
    httpx.post(f"{URL}/admin/config", json=cfg, timeout=5)


async def main() -> None:
    for p in (ROWS, TRACE):
        p.unlink(missing_ok=True)
    proc = subprocess.Popen([sys.executable, "-m", "standin.server", "--port", str(PORT), "--fault-503", "0"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(50):
            try:
                httpx.get(f"{URL}/admin/state", timeout=0.5)
                break
            except httpx.HTTPError:
                time.sleep(0.2)

        def relay(name: str, **kw) -> Relay:
            return Relay("stand-in", "h256k5", "smoke", ledger=Ledger(name, persist=False), base_url=f"{URL}/v1",
                         rows_path=ROWS, trace_path=TRACE, **kw)

        rl = relay("smoke1")
        a = await rl.ask(Q_SINGLE)
        print(f"1 single : {a['status']} {a['tier']} {a['branch']} ft={a['first_token_ms']} total={a['total_ms']} "
              f"tok={a['tokens_in']}/{a['tokens_out']} withheld={a['withheld']}")
        assert a["status"] == "ok" and a["branch"] == "answer" and a["answer"]

        b = await rl.ask(Q_CROSS)
        print(f"2 cross  : {b['status']} {b['branch']} answer={b['answer'][:140]}...")
        assert b["branch"] == "decompose" and "AIAYN" in b["answer"] and "HNET" in b["answer"]

        c = await rl.ask(Q_SINGLE)
        print(f"3 cache  : {c['tier']} ft={c['first_token_ms']} tok={c['tokens_in']}/{c['tokens_out']}")
        assert c["tier"] == "cached" and c["tokens_in"] == 0

        d = await rl.ask(Q_OUT)
        print(f"4 out    : {d['branch']} {d['answer'][:120]}")
        await rl.close()

        # burst of 12 concurrent, distinct-enough questions on a fresh ledger; FIFO admission under TPM 8K.
        # Reset the stand-in so its buckets match the fresh client ledger (each section starts from full limits).
        admin(reset=True)
        rl = relay("smoke2", use_cache=False)
        qs = [f"{Q_SINGLE} (variant {i})" for i in range(12)]
        t0 = time.perf_counter()
        res = await asyncio.gather(*(rl.ask(q, patience_s=20) for q in qs))
        tiers = [x["tier"] for x in res]
        print(f"5 burst12: {time.perf_counter() - t0:.1f}s statuses={[x['status'] for x in res]}")
        print(f"           tiers={tiers}")
        print(f"           first_token_ms={sorted(round(x['first_token_ms'] or -1) for x in res)}")
        assert all(x["answer"] for x in res), "nobody turned away"
        await rl.close()

        # per-minute limit halved on the stand-in; the client ledger is told the same limit
        admin(reset=True, tpm_scale=0.5)
        rl = relay("smoke3", use_cache=False)
        for m in (DRAFTER, ROUTER):
            rl.ledger.acct(m).tpm.capacity = 4000
            rl.ledger.acct(m).tpm.per_s = 4000 / 60
            rl.ledger.acct(m).tpm.set_level(4000)
        res = await asyncio.gather(*(rl.ask(q, patience_s=8) for q in qs[:8]))
        print(f"6 tpm/2  : statuses={[x['status'] for x in res]} tiers={[x['tier'] for x in res]}")
        assert all(x["answer"] for x in res)
        await rl.close()

        # daily budget exhausted on both models: the 3pm answer
        admin(reset=True, tpd_level={DRAFTER: 0, ROUTER: 0})
        rl = relay("smoke4", use_cache=False)
        for m in (DRAFTER, ROUTER):
            rl.ledger.acct(m).tpd.set_level(0)
        e = await rl.ask("How does Mimi combine semantic and acoustic information?")
        print(f"7 daily  : {e['status']} {e['tier']} ft={e['first_token_ms']} tok={e['tokens_in']}/{e['tokens_out']}")
        print(f"           notice: {e['notices'][0][:200]}")
        print(f"           answer: {e['answer'][:200]}")
        assert e["tier"] == "extractive" and e["tokens_in"] == 0 and e["status"] == "rate_limited"

        # a person who gives up after 0.3 s: the request is cancelled and logged as a timeout
        admin(reset=True)
        rl2 = relay("smoke5", use_cache=False)
        try:
            await asyncio.wait_for(rl2.ask(Q_CROSS), timeout=0.3)
        except asyncio.TimeoutError:
            pass
        await rl.close()
        await rl2.close()

        rows = list(csv.DictReader(ROWS.open(encoding="utf-8")))
        print(f"8 rows   : {len(rows)} rows; last status={rows[-1]['status']} setting={rows[-1]['retrieval_setting']}")
        assert rows[-1]["status"] == "timeout"
        print("\nall pipeline smoke checks passed")
    finally:
        proc.terminate()


if __name__ == "__main__":
    asyncio.run(main())
