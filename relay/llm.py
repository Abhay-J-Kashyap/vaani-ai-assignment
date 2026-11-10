"""Async OpenAI-compatible chat client for Groq and the stand-in (same code path; only base_url differs).

  REAL     https://api.groq.com/openai/v1   (key from .env, never logged)
  STANDIN  http://127.0.0.1:8088/v1

`stream_chat` yields ("delta", text) as answer tokens arrive and finally ("done", CallResult).
Two timeouts (ADR "Timeouts"):
  * first_token_s: no answer token by then -> timeout. Measured first-token p99 ~2.4 s, and a 503 "over capacity"
    can hang ~30 s before any header, so this must sit well below 30 s.
  * total_s: the whole call. A trickle (~70 tok/s after the first token) is slow but useful, so this is generous.
Every call goes through the ledger: admission is checked by the caller; this module charges start/finish/429.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx
from dotenv import load_dotenv

from relay.ledger import Ledger

ROOT = Path(__file__).resolve().parents[1]
REAL_URL = "https://api.groq.com/openai/v1"
STANDIN_URL = "http://127.0.0.1:8088/v1"


@dataclass
class CallResult:
    model: str
    status: str                      # ok | rate_limited | timeout | error
    http_status: int | None = None
    first_token_ms: float | None = None
    total_ms: float | None = None
    content: str = ""
    tool_calls: list = field(default_factory=list)
    usage: dict = field(default_factory=dict)
    headers: dict = field(default_factory=dict)
    error: dict | str | None = None
    limit: dict | None = None        # parsed 429: which, daily, limit, used, requested, wait_s
    finish_reason: str | None = None


class LLM:
    def __init__(self, source: str = "stand-in", ledger: Ledger | None = None, base_url: str | None = None):
        self.source = source
        self.base_url = base_url or (REAL_URL if source == "real" else STANDIN_URL)
        key = ""
        if source == "real":
            load_dotenv(ROOT / ".env")
            key = os.environ.get("GROQ_API_KEY", "").strip()
            if not key:
                raise SystemExit("GROQ_API_KEY is empty (.env)")
        self.headers = {"Authorization": f"Bearer {key or 'standin'}"}
        self.ledger = ledger or Ledger("real" if source == "real" else "standin", persist=source == "real")
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=5.0),
                                        limits=httpx.Limits(max_connections=128))

    async def close(self):
        await self.client.aclose()

    async def stream_chat(self, *, model: str, messages: list[dict], max_tokens: int, prompt_tokens: int,
                          tools: list | None = None, tool_choice=None, stream: bool = True,
                          reasoning_effort: str | None = "low", temperature: float = 0.2,
                          first_token_s: float = 4.0, total_s: float = 20.0, charge_start: bool = True):
        """charge_start=False when the caller's admission step has already reserved this call in the ledger."""
        body = {"model": model, "messages": messages, "max_tokens": max_tokens, "stream": stream,
                "temperature": temperature}
        if reasoning_effort and model.startswith("openai/gpt-oss"):
            body["reasoning_effort"] = reasoning_effort
        if tools:
            body["tools"], body["tool_choice"] = tools, tool_choice or "auto"
        res = CallResult(model=model, status="error")
        t0 = time.perf_counter()
        if charge_start:
            self.ledger.start(model, prompt_tokens, max_tokens)
        queue: asyncio.Queue = asyncio.Queue()

        async def run():
            try:
                async with self.client.stream("POST", f"{self.base_url}/chat/completions", json=body,
                                              headers=self.headers) as r:
                    res.http_status = r.status_code
                    res.headers = {k.lower(): v for k, v in r.headers.items() if k.lower() != "set-cookie"}
                    if r.status_code != 200:
                        raw = (await r.aread()).decode("utf-8", "replace")
                        try:
                            res.error = json.loads(raw)
                        except json.JSONDecodeError:
                            res.error = raw[:500]
                        res.status = "rate_limited" if r.status_code == 429 else "error"
                        return
                    if not stream:
                        payload = json.loads(await r.aread())
                        msg = (payload.get("choices") or [{}])[0].get("message") or {}
                        res.content = msg.get("content") or ""
                        res.tool_calls = msg.get("tool_calls") or []
                        res.usage = payload.get("usage") or {}
                        res.finish_reason = (payload.get("choices") or [{}])[0].get("finish_reason")
                        res.first_token_ms = round((time.perf_counter() - t0) * 1000, 1)
                        res.status = "ok"
                        return
                    async for line in r.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            break
                        chunk = json.loads(data)
                        if "error" in chunk:
                            res.error = chunk["error"]
                            return
                        for ch in chunk.get("choices") or []:
                            d = ch.get("delta") or {}
                            if d.get("content"):
                                if res.first_token_ms is None:
                                    res.first_token_ms = round((time.perf_counter() - t0) * 1000, 1)
                                res.content += d["content"]
                                await queue.put(d["content"])
                            if d.get("tool_calls"):
                                res.tool_calls.extend(d["tool_calls"])
                            if ch.get("finish_reason"):
                                res.finish_reason = ch["finish_reason"]
                        u = chunk.get("usage") or (chunk.get("x_groq") or {}).get("usage")
                        if u:
                            res.usage = u
                    res.status = "ok"
            except httpx.HTTPError as e:
                res.error = f"{type(e).__name__}: {e}"[:300]
            finally:
                await queue.put(None)

        task = asyncio.create_task(run())
        deadline_first = t0 + first_token_s
        deadline_total = t0 + total_s
        try:
            while True:
                now = time.perf_counter()
                limit = deadline_first if res.first_token_ms is None and stream else deadline_total
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=max(0.0, limit - now))
                except asyncio.TimeoutError:
                    task.cancel()
                    res.status = "timeout"
                    res.error = "first_token_timeout" if res.first_token_ms is None else "total_timeout"
                    break
                if item is None:
                    break
                yield ("delta", item)
        finally:
            if not task.done():
                task.cancel()
            res.total_ms = round((time.perf_counter() - t0) * 1000, 1)
            if res.status == "rate_limited":
                res.limit = self.ledger.on_429(model, res.error)
            else:
                self.ledger.finish(model, res.usage if res.status == "ok" else None, res.headers)
        yield ("done", res)

    async def chat(self, **kw) -> CallResult:
        res = None
        async for kind, val in self.stream_chat(**kw):
            if kind == "done":
                res = val
        return res
