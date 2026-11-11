"""Heading 7: the exact reply a waiting person receives at 3pm when the daily budget is gone. Stand-in, no Groq.

  python -m eval.show_3pm  ->  results/reply_at_3pm_2026-11-11.txt

Two states, because the fallback has two steps:
  A. the 120b drafter's daily budget is gone, the 20b still has budget -> a shorter (Lean) answer from the 20b;
  B. both daily budgets are gone -> quoted passages built by local code, with the honest wait, no model call.
The wall clock is pinned to 15:00 so the 20% drafter reserve is in its afternoon state (released evenly to 18:00).
"""
from __future__ import annotations

import asyncio
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx

from relay import ledger as ledger_mod
from relay.ledger import DRAFTER, ROUTER, Ledger
from relay.pipeline import Relay

ROOT = Path(__file__).resolve().parents[1]
PORT = 8096
Q = "How does Mimi combine semantic and acoustic information in its tokens?"


async def ask(state: str, tpd: dict) -> list[str]:
    httpx.post(f"http://127.0.0.1:{PORT}/admin/config", json={"reset": True, "tpd_level": tpd}, timeout=5)
    led = Ledger(f"pm3_{state}", persist=False)
    for m, lvl in tpd.items():
        led.acct(m).tpd.set_level(lvl)
    relay = Relay("stand-in", "h256k8", "dev", ledger=led, base_url=f"http://127.0.0.1:{PORT}/v1", use_cache=False,
                  rows_path=None, trace_path=None)
    shown: list[str] = []
    t0 = time.perf_counter()
    res = await relay.ask(Q, on_event=lambda kind, text: shown.append(("    " if kind == "sentence" else "  [") + text
                                                                          + ("" if kind == "sentence" else "]")))
    await relay.close()
    return [f"State {state}: status={res['status']} tier={res['tier']} first text after {res['first_token_ms']} ms, "
            f"done after {(time.perf_counter() - t0) * 1000:.0f} ms, tokens {res['tokens_in']}/{res['tokens_out']}",
            f"  Person asks: {Q}"] + shown + [""]


async def main() -> None:
    ledger_mod.datetime = type("FixedClock", (datetime,), {"now": classmethod(lambda cls: cls(2026, 11, 12, 15, 0))})
    proc = subprocess.Popen([sys.executable, "-m", "standin.server", "--port", str(PORT), "--fault-503", "0"], cwd=ROOT,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(60):
            try:
                httpx.get(f"http://127.0.0.1:{PORT}/admin/state", timeout=0.5)
                break
            except httpx.HTTPError:
                time.sleep(0.25)
        out = ["What the person sees at 3pm with the daily budget gone (stand-in; replies as rendered to the person)", ""]
        out += await ask("A (120b daily budget gone, 20b has budget)", {DRAFTER: 0})
        out += await ask("B (both daily budgets gone)", {DRAFTER: 0, ROUTER: 0})
    finally:
        proc.terminate()
    path = ROOT / "results" / "reply_at_3pm_2026-11-11.txt"
    path.write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    asyncio.run(main())
