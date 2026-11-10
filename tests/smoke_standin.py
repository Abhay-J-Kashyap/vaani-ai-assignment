"""Smoke test: the client, ledger and stand-in together. Starts the stand-in on a spare port; no Groq calls.

  python -m tests.smoke_standin

Checks: streamed answer with citations and a first token; router tool call; per-minute 429 in Groq format with a
stated wait the ledger adopts; header sync of the ledger; daily-exhausted 429 names TPD.
"""
from __future__ import annotations

import asyncio
import subprocess
import sys
import time

import httpx

from relay.ledger import Ledger
from relay.llm import LLM

PORT = 8099
URL = f"http://127.0.0.1:{PORT}/v1"
PROMPT = [{"role": "system", "content": "Answer only from the passages. Cite each sentence."},
          {"role": "user", "content": "PASSAGES:\n[AIAYN §3.2.1] We suspect that for large values of d_k, the dot "
                                      "products grow large in magnitude, pushing the softmax function into regions "
                                      "where it has extremely small gradients. To counteract this effect, we scale "
                                      "the dot products by 1/sqrt(d_k).\n"
                                      "QUESTION: why are dot products scaled by sqrt(d_k)?"}]


async def main() -> None:
    proc = subprocess.Popen([sys.executable, "-m", "standin.server", "--port", str(PORT), "--fault-503", "0"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(50):
            try:
                httpx.get(f"http://127.0.0.1:{PORT}/admin/state", timeout=0.5)
                break
            except httpx.HTTPError:
                time.sleep(0.2)
        llm = LLM("stand-in", ledger=Ledger("smoke", persist=False), base_url=URL)
        m = "openai/gpt-oss-120b"

        deltas, res = [], None
        async for kind, val in llm.stream_chat(model=m, messages=PROMPT, max_tokens=300, prompt_tokens=200):
            (deltas.append(val) if kind == "delta" else None)
            res = val if kind == "done" else res
        print(f"1. stream: status={res.status} first_token={res.first_token_ms} ms total={res.total_ms} ms "
              f"chunks={len(deltas)} usage={res.usage.get('prompt_tokens')}/{res.usage.get('completion_tokens')}")
        print(f"   answer: {res.content[:160]}")
        assert res.status == "ok" and "[AIAYN §3.2.1]" in res.content and res.first_token_ms

        tools = [{"type": "function", "function": {"name": "search_papers", "parameters": {"type": "object"}}}]
        r = await llm.chat(model="openai/gpt-oss-20b", messages=PROMPT, max_tokens=150, prompt_tokens=200,
                           tools=tools, stream=False, total_s=5)
        print(f"2. router: status={r.status} tool={r.tool_calls[0]['function']['name'] if r.tool_calls else None} "
              f"in {r.total_ms} ms")
        assert r.tool_calls

        httpx.post(f"http://127.0.0.1:{PORT}/admin/config", json={"tpm_scale": 0.5})   # TPM 4,000
        big = [{"role": "user", "content": "PASSAGES:\n[X §1] " + "token budget " * 900 + "\nQUESTION: q"}]
        outs, levels = [], []
        for _ in range(3):
            st = httpx.get(f"http://127.0.0.1:{PORT}/admin/state").json()["models"].get(m, {})
            levels.append(st.get("tpm"))
            outs.append(await llm.chat(model=m, messages=big, max_tokens=200, prompt_tokens=2000))
        rl = next((o for o in outs if o.status == "rate_limited"), None)
        print(f"3. halved TPM: statuses={[o.status for o in outs]} tpm level before each={levels} "
              f"call ms={[o.total_ms for o in outs]}")
        print(f"   429: {rl.error['error']['message'][:150]}... retry-after={rl.headers.get('retry-after')}")
        print(f"   parsed: {rl.limit}  ledger tpm level={llm.ledger.acct(m).tpm.level():.0f}")
        assert rl and rl.limit["which"] == "tpm" and rl.limit["wait_s"] > 0

        httpx.post(f"http://127.0.0.1:{PORT}/admin/config", json={"reset": True, "tpd_level": {m: 0}})
        d = await llm.chat(model=m, messages=PROMPT, max_tokens=300, prompt_tokens=200)
        print(f"4. daily exhausted: status={d.status} which={d.limit and d.limit['which']} "
              f"wait={d.limit and round(d.limit['wait_s'])} s ({d.total_ms} ms to refuse)")
        assert d.status == "rate_limited" and d.limit["daily"]
        await llm.close()
        print("\nall smoke checks passed")
    finally:
        proc.terminate()


if __name__ == "__main__":
    asyncio.run(main())
