"""Stand-in model service: an OpenAI/Groq-compatible /v1/chat/completions we control, calibrated from Groq probes.

  python -m standin.server --port 8088 [--tpm-scale 0.5] [--tpd-start 0] [--seed 1]

Behaviour (numbers from standin/calibration.json, made by `python -m probe.calibrate` from results/probe_raw.jsonl):
  * Limits: per-model continuous buckets RPM 30 / TPM 8,000 / RPD 1,000 / TPD 200,000. TPM and TPD admission
    check prompt + max_tokens; TPM is charged that, TPD the actual usage. A refusal is a Groq-format 429 with the
    exact "Please try again in ..." wait and a rounded-up retry-after header, returned after ~100 ms. Rejected
    calls are free. Groq-style x-ratelimit-* headers on every response.
  * Slow: first answer token = queue (calm lognormal, or a correlated episode adding ~1.8 s) + rest (lognormal,
    median growing with prompt tokens). Episodes hit every request that arrives while one is active.
  * Slow success: a "trickle" share streams at ~67 tok/s instead of ~476 tok/s after the first token.
  * Fail: a 503 "over capacity" share answers only after ~30 s, like Groq's.
  * Slower under load: queue time is multiplied by 1 + max(0, in_flight - 16) / 16. ASSUMED, not measured:
    Groq showed no concurrency effect up to 16 and RPM 30 stops us measuring beyond (ADR: least like the real thing).
  * Streams: SSE chunks with reasoning deltas first, then content, then usage in x_groq, then [DONE].
  * Content: extractive. It quotes sentences from the passages in the prompt with their [cite] labels, so the
    citation gate and the grader run unchanged; quality numbers from stand-in runs are therefore not model quality.
  * Router calls (tools present): always calls search_papers with the user's question.
Admin: GET /admin/state, POST /admin/config {"tpm_scale":0.5,"tpd_level":{"model":0},"fault_503":0.0,...}.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import re
import time
import uuid
from pathlib import Path

import tiktoken
from aiohttp import web

from relay.limits import Bucket, error_429

ROOT = Path(__file__).resolve().parents[1]
CAL = json.loads((ROOT / "standin" / "calibration.json").read_text(encoding="utf-8"))
ENC = tiktoken.get_encoding("o200k_base")
CHAT_OVERHEAD = 78      # gpt-oss chat template tokens: a 20-token prompt was billed ~98 prompt tokens on Groq
EPISODE_S = 8.0         # ASSUMED episode length: an 8-request batch was delayed together, one 30 s later was not
LOAD_KNEE = 16          # no slowdown measured up to 16 in flight
SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")
WORD = re.compile(r"[a-z0-9]+")


def ntok(s: str) -> int:
    return len(ENC.encode(s, disallowed_special=()))


class Model:
    def __init__(self, name: str, cfg: "Config"):
        c = CAL["models"].get(name) or CAL["models"]["openai/gpt-oss-120b"]
        self.name, self.c, self.cfg = name, c, cfg
        lim = CAL["limits"]
        self.rpm = Bucket(lim["rpm"], lim["rpm"] / 60)
        self.tpm = Bucket(lim["tpm"] * cfg.tpm_scale, lim["tpm"] * cfg.tpm_scale / 60)
        self.rpd = Bucket(lim["rpd"], lim["rpd"] / 86400)
        self.tpd = Bucket(lim["tpd"], lim["tpd"] / 86400)
        share = c["queue_episode"]["share"]
        self.episode_rate = share / (EPISODE_S * (1 - share)) if share < 1 else 1.0
        self.episode_until = 0.0
        self.next_onset = time.time() + cfg.rng.expovariate(self.episode_rate) if self.episode_rate else math.inf

    def in_episode(self, now: float) -> bool:
        while now >= self.next_onset:
            self.episode_until = self.next_onset + EPISODE_S
            self.next_onset = self.episode_until + self.cfg.rng.expovariate(self.episode_rate)
        return now < self.episode_until

    def first_token_ms(self, prompt_tokens: int, in_flight: int, now: float) -> tuple[float, float]:
        rng = self.cfg.rng
        q = self.c["queue_calm_ms"]
        queue = rng.lognormvariate(q["mu"], q["sigma"])
        ep = self.c["queue_episode"].get("extra_ms") or self.c["queue_episode"].get("extra_ms_pooled")
        if ep and self.in_episode(now):
            queue += rng.lognormvariate(ep["mu"], ep["sigma"])
        queue *= 1 + max(0, in_flight - LOAD_KNEE) / LOAD_KNEE
        r = self.c["rest_first_token"]
        rest = math.exp(r["log_ms_intercept"] + r["log_ms_per_prompt_token"] * prompt_tokens + rng.gauss(0, r["sigma"]))
        return queue, rest


class Config:
    def __init__(self, a):
        self.tpm_scale = a.tpm_scale
        self.fault_503 = a.fault_503
        self.rng = random.Random(a.seed)
        self.models: dict[str, Model] = {}
        self.in_flight = 0
        self.tpd_start = a.tpd_start

    def model(self, name: str) -> Model:
        if name not in self.models:
            m = Model(name, self)
            if self.tpd_start is not None:
                m.tpd.set_level(self.tpd_start)
            self.models[name] = m
        return self.models[name]


def rl_headers(m: Model, now: float) -> dict:
    used_req = m.rpd.capacity - m.rpd.level(now)
    used_tok = m.tpm.capacity - m.tpm.level(now)
    return {"x-ratelimit-limit-requests": str(int(m.rpd.capacity)),
            "x-ratelimit-remaining-requests": str(int(m.rpd.level(now))),
            "x-ratelimit-reset-requests": f"{used_req / m.rpd.per_s:.2f}s",
            "x-ratelimit-limit-tokens": str(int(m.tpm.capacity)),
            "x-ratelimit-remaining-tokens": str(int(m.tpm.level(now))),
            "x-ratelimit-reset-tokens": f"{used_tok / m.tpm.per_s:.2f}s",
            "x-groq-region": "standin"}


def extract_answer(messages: list[dict], max_tokens: int) -> str:
    """Quote the best-matching sentence from up to three passages in the prompt, each with its [cite]."""
    user = "\n".join(m.get("content") or "" for m in messages if m.get("role") == "user")
    q = re.search(r"QUESTION:\s*(.+)", user, flags=re.S)
    qwords = set(WORD.findall((q.group(1) if q else user).lower()))
    passages = re.findall(r"^\[([^\]]+)\]\s*(.+?)(?=^\[[^\]]+\]|\Z|^QUESTION:)", user, flags=re.M | re.S)
    out = []
    for cite, text in passages[:3]:
        sents = [s.strip().replace("\n", " ") for s in SENT.split(text) if len(s.strip()) > 30]
        if not sents:
            continue
        best = max(sents, key=lambda s: len(qwords & set(WORD.findall(s.lower()))))
        out.append(f"{best.rstrip('.')} [{cite}].")
    ans = " ".join(out) or "The provided passages do not cover this question."
    toks = ENC.encode(ans, disallowed_special=())
    return ENC.decode(toks[:max(1, max_tokens - 10)])


async def chat(request: web.Request) -> web.StreamResponse:
    cfg: Config = request.app["cfg"]
    body = await request.json()
    m = cfg.model(body.get("model", "openai/gpt-oss-120b"))
    messages = body.get("messages") or []
    max_tokens = int(body.get("max_tokens") or 512)
    prompt = CHAT_OVERHEAD + sum(ntok(x.get("content") or "") for x in messages)
    if body.get("tools"):
        prompt += ntok(json.dumps(body["tools"]))
    requested = prompt + max_tokens
    now = time.time()
    rl = CAL["rate_limited_response_ms"]

    # limits, in Groq's order of reporting: per-day first, then per-minute
    for which, b, need in (("rpd", m.rpd, 1), ("tpd", m.tpd, requested), ("rpm", m.rpm, 1), ("tpm", m.tpm, requested)):
        wait = b.wait_for(need, now)
        if wait > 0:
            await asyncio.sleep(cfg.rng.lognormvariate(rl["mu"], rl["sigma"]) / 1000)
            used = int(round(b.capacity - b.level(now)))
            wait = min(wait, 86400.0)
            hdr = rl_headers(m, now) | {"retry-after": str(math.ceil(wait))}
            return web.json_response(error_429(m.name, which, int(b.capacity), used, need if which in ("tpm", "tpd") else 1,
                                               wait), status=429, headers=hdr)
    m.rpm.take(1, now)
    m.rpd.take(1, now)
    m.tpm.take(requested, now)

    if cfg.rng.random() < cfg.fault_503:
        await asyncio.sleep(CAL["over_capacity_503"]["delay_ms"] / 1000)
        return web.json_response({"error": {"message": f"{m.name} is currently over capacity. Please try again and "
                                            f"back off exponentially. (stand-in)", "type": "internal_server_error"}},
                                 status=503, headers=rl_headers(m, time.time()))

    cfg.in_flight += 1
    try:
        queue_ms, rest_ms = m.first_token_ms(prompt, cfg.in_flight, now)
        trickle = cfg.rng.random() < m.c["trickle"]["share"]
        rate = (m.c["trickle"]["tokens_per_s"] or 67.0) if trickle else m.c["server_tokens_per_s"]["median"]
        reasoning = int(m.c.get("reasoning_tokens_median") or 0)
        if body.get("tools"):
            text = ""
            q = next((x.get("content") for x in reversed(messages) if x.get("role") == "user"), "")
            tool_calls = [{"index": 0, "id": f"call_{uuid.uuid4().hex[:8]}", "type": "function",
                           "function": {"name": "search_papers", "arguments": json.dumps({"query": q[:500]})}}]
        else:
            text, tool_calls = extract_answer(messages, max_tokens - reasoning), []
        completion = reasoning + ntok(text) + (20 if tool_calls else 0)
        usage = {"queue_time": round(queue_ms / 1000, 4), "prompt_tokens": prompt, "completion_tokens": completion,
                 "total_tokens": prompt + completion, "completion_time": round(completion / rate, 4),
                 "prompt_time": round(rest_ms / 2000, 4),
                 "completion_tokens_details": {"reasoning_tokens": reasoning}}
        cid = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        await asyncio.sleep((queue_ms + rest_ms) / 1000)

        if not body.get("stream"):
            m.tpd.take(prompt + completion, time.time())
            msg = {"role": "assistant", "content": text or None}
            if tool_calls:
                msg["tool_calls"] = [{k: v for k, v in t.items() if k != "index"} for t in tool_calls]
            return web.json_response({"id": cid, "object": "chat.completion", "model": m.name,
                                      "choices": [{"index": 0, "message": msg,
                                                   "finish_reason": "tool_calls" if tool_calls else "stop"}],
                                      "usage": usage}, headers=rl_headers(m, time.time()))

        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream", **rl_headers(m, time.time())})
        await resp.prepare(request)

        async def send(delta: dict, finish=None, extra=None):
            chunk = {"id": cid, "object": "chat.completion.chunk", "model": m.name,
                     "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
            if extra:
                chunk.update(extra)
            await resp.write(f"data: {json.dumps(chunk)}\n\n".encode())

        await send({"role": "assistant", "reasoning": "…"})
        if tool_calls:
            await send({"tool_calls": tool_calls})
        toks = ENC.encode(text, disallowed_special=())
        step = 3
        # Pace against an absolute schedule. On Windows, asyncio treats sleeps shorter than its ~15.6 ms clock
        # resolution as already due, so per-chunk sleeps of ~6 ms would stream everything at once.
        t_stream = time.perf_counter()
        for i in range(0, len(toks), step):
            await send({"content": ENC.decode(toks[i:i + step])})
            ahead = t_stream + (i + step) / rate - time.perf_counter()
            if ahead > 0.02:
                await asyncio.sleep(ahead)
        m.tpd.take(prompt + completion, time.time())
        await send({}, finish="tool_calls" if tool_calls else ("length" if completion >= max_tokens else "stop"),
                   extra={"x_groq": {"usage": usage}})
        await resp.write(b"data: [DONE]\n\n")
        return resp
    finally:
        cfg.in_flight -= 1


async def admin_state(request: web.Request) -> web.Response:
    cfg: Config = request.app["cfg"]
    now = time.time()
    return web.json_response({"in_flight": cfg.in_flight, "tpm_scale": cfg.tpm_scale, "fault_503": cfg.fault_503,
                              "models": {n: {k: round(b.level(now), 1) for k, b in
                                             (("rpm", m.rpm), ("tpm", m.tpm), ("rpd", m.rpd), ("tpd", m.tpd))}
                                         for n, m in cfg.models.items()}})


async def admin_config(request: web.Request) -> web.Response:
    cfg: Config = request.app["cfg"]
    body = await request.json()
    if body.get("reset"):
        cfg.models.clear()
    if "tpm_scale" in body:
        cfg.tpm_scale = float(body["tpm_scale"])
        for m in cfg.models.values():
            cap = CAL["limits"]["tpm"] * cfg.tpm_scale
            m.tpm = Bucket(cap, cap / 60)
    if "fault_503" in body:
        cfg.fault_503 = float(body["fault_503"])
    for name, level in (body.get("tpd_level") or {}).items():
        cfg.model(name).tpd.set_level(float(level))
    return await admin_state(request)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8088)
    ap.add_argument("--tpm-scale", type=float, default=1.0, help="0.5 = per-minute limit halved")
    ap.add_argument("--tpd-start", type=float, default=None, help="initial TPD level (0 = daily budget exhausted)")
    ap.add_argument("--fault-503", type=float, default=CAL["over_capacity_503"]["share"])
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    app = web.Application(client_max_size=4 * 1024 * 1024)
    app["cfg"] = Config(a)
    app.router.add_post("/v1/chat/completions", chat)
    app.router.add_get("/admin/state", admin_state)
    app.router.add_post("/admin/config", admin_config)
    print(f"stand-in on http://127.0.0.1:{a.port}/v1  tpm_scale={a.tpm_scale} fault_503={a.fault_503}")
    web.run_app(app, host="127.0.0.1", port=a.port, print=None)


if __name__ == "__main__":
    main()
