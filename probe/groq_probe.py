"""Probe Groq's real service: how its latency is spread, what its limits are, and what it returns when one is hit.

Every request is appended as one JSON line to results/probe_raw.jsonl. Each line keeps the full
response headers, the error body, Groq's own usage timings and our client-side timestamps,
so that Table 1 and the stand-in calibration can be recomputed from the raw file.

Run from the repo root with the venv active. Always do --dry-run first; it sends nothing.

  python -m probe.groq_probe headers     --models openai/gpt-oss-20b openai/gpt-oss-120b
  python -m probe.groq_probe latency     --model openai/gpt-oss-20b --per-size 15 --rpm 30 --tpm 8000
  python -m probe.groq_probe rpm         --model openai/gpt-oss-20b --burst 35
  python -m probe.groq_probe tpm         --model openai/gpt-oss-20b --size large --tpm 8000
  python -m probe.groq_probe concurrency --model openai/gpt-oss-20b --levels 1 2 4 8 16
  python -m probe.groq_probe daily       --model <sacrificial-model> --target tokens --confirm-exhaust

The API key is read from .env (GROQ_API_KEY). It is never printed or logged.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "probe_raw.jsonl"
URL = "https://api.groq.com/openai/v1/chat/completions"
RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

# Words of filler per size. One word is about 1.0-1.3 tokens; the real counts come from `usage`.
SIZES = {"tiny": 0, "small": 150, "medium": 1150, "large": 2700}
_WORDS = ("signal latency token budget queue retry window burst model stream paper section "
          "table figure encoder decoder attention chunk router limit replica shard cache").split()


# --------------------------------------------------------------------------- helpers
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def ms_since(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 1)


def parse_duration_s(v: str | None) -> float | None:
    """Parse Groq reset/retry values: '7.66s', '2m59.56s', '1h2m3s', '120ms', or plain seconds '12'."""
    if not v:
        return None
    v = v.strip()
    try:
        return float(v)
    except ValueError:
        pass
    total, matched = 0.0, False
    for num, unit in re.findall(r"([\d.]+)(ms|h|m|s)", v):
        matched = True
        total += float(num) * {"h": 3600, "m": 60, "s": 1, "ms": 0.001}[unit]
    return total if matched else None


def build_messages(size: str, seed: int, task: str = "explain") -> list[dict]:
    """A unique nonce goes first in every prompt. Groq says cached tokens don't count toward limits,
    so a repeated prompt would make the latency and limit numbers look better than they are.
    task='ok' asks for a one-word reply. With a large max_tokens it makes the reserved and actually used
    token counts far apart, which tells us which of the two the daily budget is charged."""
    nonce = uuid.uuid4().hex
    rng = random.Random(seed)
    filler = " ".join(rng.choice(_WORDS) for _ in range(SIZES[size]))
    if size == "tiny":
        user = f"[{nonce}] Reply with the single word OK."
    elif task == "ok":
        user = f"[{nonce}]\n{filler}\n\nIgnore the text above. Reply with the single word OK and nothing else."
    else:
        user = (f"[{nonce}]\n{filler}\n\nIgnore the text above. In about 80 words, explain why an API "
                f"might limit requests per minute.")
    return [{"role": "user", "content": user}]


def is_gpt_oss(model: str) -> bool:
    return model.startswith("openai/gpt-oss")


def write(rec: dict) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def classify_429(rec: dict) -> str | None:
    """Return which limit a 429 names, based on the error text (e.g. 'requests per minute')."""
    if rec.get("status_code") != 429:
        return None
    text = json.dumps(rec.get("error_body", "")).lower()
    for key, label in (("per day", "daily"), ("rpd", "daily"), ("tpd", "daily"),
                       ("per minute", "minute"), ("rpm", "minute"), ("tpm", "minute")):
        if key in text:
            return label
    return "unknown"


def error_message(rec: dict) -> str:
    b = rec.get("error_body")
    if isinstance(b, dict):
        return str((b.get("error") or {}).get("message") or b)
    return str(b or "")


def limit_numbers(rec: dict) -> dict:
    """Pull 'Limit L, Used U, Requested R' and 'try again in X' out of a 429 message.
    On a daily 429 'Used' is a free reading of the daily bucket, which is how the refill rate is measured."""
    msg = error_message(rec)
    out = {}
    for name in ("Limit", "Used", "Requested"):
        m = re.search(rf"{name} (\d+)", msg)
        if m:
            out[f"limit_{name.lower()}"] = int(m.group(1))
    m = re.search(r"try again in ([0-9hms.]+)", msg)
    if m:
        out["stated_wait_s"] = parse_duration_s(m.group(1).rstrip("."))
    return out


# --------------------------------------------------------------------------- one request
class Prober:
    def __init__(self, key: str, reasoning_effort: str | None, dry_run: bool):
        self.key = key
        self.reasoning_effort = reasoning_effort
        self.dry_run = dry_run
        self.sent = 0
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(90.0, connect=10.0),
                                        limits=httpx.Limits(max_connections=64))

    async def close(self):
        await self.client.aclose()

    async def call(self, *, phase: str, model: str, size: str, max_tokens: int,
                   stream: bool = True, meta: dict | None = None, task: str = "explain") -> dict:
        seed = random.randrange(1 << 30)
        body = {"model": model, "messages": build_messages(size, seed, task),
                "max_tokens": max_tokens, "stream": stream, "temperature": 0.2}
        if is_gpt_oss(model) and self.reasoning_effort:
            body["reasoning_effort"] = self.reasoning_effort
        rec: dict = {"run_id": RUN_ID, "probe_id": uuid.uuid4().hex[:12], "phase": phase,
                     "model": model, "size": size, "task": task, "max_tokens": max_tokens, "stream": stream,
                     "reasoning_effort": body.get("reasoning_effort"), "start_time": now_iso(),
                     **(meta or {})}
        if self.dry_run:
            rec["dry_run"] = True
            return rec

        self.sent += 1
        t0 = time.perf_counter()
        first_reasoning = first_content = None
        chunks, parts = 0, []
        # Gap between consecutive SSE events; locates stalls in delivery after the first token
        last_event = None
        max_gap = max_gap_at = None
        try:
            async with self.client.stream("POST", URL, json=body,
                                          headers={"Authorization": f"Bearer {self.key}"}) as r:
                rec["status_code"] = r.status_code
                rec["headers_ms"] = ms_since(t0)
                rec["headers"] = {k.lower(): v for k, v in r.headers.items()
                                  if k.lower() not in ("set-cookie",)}
                if r.status_code != 200:
                    raw = (await r.aread()).decode("utf-8", "replace")
                    try:
                        rec["error_body"] = json.loads(raw)
                    except json.JSONDecodeError:
                        rec["error_body"] = raw[:2000]
                elif stream:
                    async for line in r.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        t_evt = ms_since(t0)
                        if last_event is not None and (max_gap is None or t_evt - last_event > max_gap):
                            max_gap, max_gap_at = round(t_evt - last_event, 1), last_event
                        last_event = t_evt
                        if data == "[DONE]":
                            break
                        chunk = json.loads(data)
                        if "error" in chunk:
                            rec["stream_error"] = chunk["error"]
                            break
                        for ch in chunk.get("choices") or []:
                            d = ch.get("delta") or {}
                            if d.get("reasoning") and first_reasoning is None:
                                first_reasoning = ms_since(t0)
                            if d.get("content"):
                                if first_content is None:
                                    first_content = ms_since(t0)
                                chunks += 1
                                parts.append(d["content"])
                            if ch.get("finish_reason"):
                                rec["finish_reason"] = ch["finish_reason"]
                        usage = chunk.get("usage") or (chunk.get("x_groq") or {}).get("usage")
                        if usage:
                            rec["usage"] = usage
                else:
                    payload = json.loads(await r.aread())
                    rec["usage"] = payload.get("usage")
                    msg = (payload.get("choices") or [{}])[0].get("message") or {}
                    parts.append(msg.get("content") or "")
                    rec["finish_reason"] = (payload.get("choices") or [{}])[0].get("finish_reason")
        except httpx.TimeoutException as e:
            rec["exception"] = f"timeout:{type(e).__name__}"
        except httpx.HTTPError as e:
            rec["exception"] = f"http:{type(e).__name__}:{e}"[:300]
        rec["total_ms"] = ms_since(t0)
        rec["first_reasoning_ms"] = first_reasoning
        rec["first_content_ms"] = first_content
        rec["content_chunks"] = chunks
        rec["max_event_gap_ms"] = max_gap          # longest silence between two SSE events
        rec["max_event_gap_after_ms"] = max_gap_at  # when that silence started, measured from send
        rec["output_preview"] = "".join(parts)[:160]
        rec["limit_hit"] = classify_429(rec)
        if rec.get("status_code") == 429:
            rec.update(limit_numbers(rec))
        write(rec)
        return rec


def summary(rec: dict) -> str:
    h = rec.get("headers") or {}
    u = rec.get("usage") or {}
    return (f"[{rec['phase']}] {rec['model']} {rec['size']:<6} status={rec.get('status_code')} "
            f"ttft={rec.get('first_content_ms')} total={rec.get('total_ms')} "
            f"tok={u.get('prompt_tokens')}/{u.get('completion_tokens')} "
            f"rem_req(day)={h.get('x-ratelimit-remaining-requests')} "
            f"rem_tok(min)={h.get('x-ratelimit-remaining-tokens')} "
            f"retry_after={h.get('retry-after')} limit={rec.get('limit_hit')}")


def est_tokens(size: str, max_tokens: int) -> int:
    return int(SIZES[size] * 1.3) + 40 + max_tokens


# --------------------------------------------------------------------------- phases
async def phase_headers(p: Prober, a) -> None:
    for m in a.models:
        rec = await p.call(phase="headers", model=m, size="tiny", max_tokens=16, stream=False)
        if p.dry_run:
            print(f"would send 1 tiny request to {m}")
            continue
        print(summary(rec))
        h = rec.get("headers") or {}
        print("   limit-requests (RPD):", h.get("x-ratelimit-limit-requests"),
              "| limit-tokens (TPM):", h.get("x-ratelimit-limit-tokens"),
              "| reset-requests:", h.get("x-ratelimit-reset-requests"),
              "| reset-tokens:", h.get("x-ratelimit-reset-tokens"))


async def phase_latency(p: Prober, a) -> None:
    """Sequential and paced below both RPM and TPM, so we measure latency, not queueing at a limit.
    Sizes are interleaved so that drift over time affects every size equally."""
    order = [s for _ in range(a.per_size) for s in a.sizes]
    rpm_gap = 60.0 / (a.rpm * a.safety)
    if p.dry_run:
        toks = sum(est_tokens(s, a.max_tokens) for s in order)
        mins = sum(max(rpm_gap, est_tokens(s, a.max_tokens) / (a.tpm * a.safety) * 60) for s in order) / 60
        print(f"would send {len(order)} streamed requests to {a.model}; ~{toks:,} tokens; ~{mins:.1f} min")
        return
    for i, size in enumerate(order):
        rec = await p.call(phase="latency", model=a.model, size=size, max_tokens=a.max_tokens,
                           meta={"seq": i})
        print(summary(rec))
        if rec.get("status_code") == 429:
            wait = parse_duration_s((rec.get("headers") or {}).get("retry-after")) or 10
            print(f"   429 during latency phase; sleeping {wait:.1f}s (this sample is logged, not retried)")
            await asyncio.sleep(wait)
            continue
        used = (rec.get("usage") or {}).get("total_tokens") or est_tokens(size, a.max_tokens)
        await asyncio.sleep(max(rpm_gap, used / (a.tpm * a.safety) * 60))


async def recovery_poll(p: Prober, a, model: str, size: str, max_tokens: int, phase: str, t_hit: float,
                        first_wait: float | None) -> None:
    """After a 429, wait for the retry-after hint, then poll until a request succeeds. Recovery time is
    measured from the first 429. Polls count toward RPM themselves, so the interval is kept coarse."""
    await asyncio.sleep(first_wait if first_wait is not None else a.poll_interval)
    for k in range(a.max_polls):
        rec = await p.call(phase=f"{phase}_recovery", model=model, size=size, max_tokens=max_tokens,
                           meta={"poll": k, "since_first_429_ms": round((time.perf_counter() - t_hit) * 1000)})
        print(summary(rec))
        if rec.get("status_code") == 200:
            print(f"   recovered {rec['since_first_429_ms']} ms after the first 429 (poll {k})")
            return
        await asyncio.sleep(a.poll_interval)
    print("   did not recover within max polls")


async def phase_rpm(p: Prober, a) -> None:
    """Send tiny requests back to back, with no pacing, until a 429 or until `burst` requests have gone."""
    if p.dry_run:
        print(f"would send up to {a.burst} tiny requests back to back to {a.model}, then up to "
              f"{a.max_polls} recovery polls every {a.poll_interval}s")
        return
    t_start = time.perf_counter()
    for i in range(a.burst):
        rec = await p.call(phase="rpm_burst", model=a.model, size="tiny", max_tokens=8, stream=False,
                           meta={"seq": i, "since_burst_start_ms": round((time.perf_counter() - t_start) * 1000)})
        print(summary(rec))
        if rec.get("status_code") == 429:
            print(f"   first 429 at request #{i + 1} after {rec['since_burst_start_ms']} ms")
            ra = parse_duration_s((rec.get("headers") or {}).get("retry-after"))
            await recovery_poll(p, a, a.model, "tiny", 8, "rpm", time.perf_counter(), ra)
            return
    print(f"   no 429 after {a.burst} requests; RPM limit is above {a.burst} or the window is not 60s")


async def phase_tpm(p: Prober, a) -> None:
    """Send large prompts back to back until a 429 that names the token limit."""
    if p.dry_run:
        print(f"would send up to {a.max_requests} '{a.size}' requests (~{est_tokens(a.size, 64):,} tokens each) "
              f"back to back to {a.model}, then recovery polls of the same size")
        return
    for i in range(a.max_requests):
        rec = await p.call(phase="tpm_burst", model=a.model, size=a.size, max_tokens=64, stream=False,
                           meta={"seq": i})
        print(summary(rec))
        if rec.get("status_code") == 429:
            ra = parse_duration_s((rec.get("headers") or {}).get("retry-after"))
            await recovery_poll(p, a, a.model, a.size, 64, "tpm", time.perf_counter(), ra)
            return
    print(f"   no 429 after {a.max_requests} requests")


async def phase_concurrency(p: Prober, a) -> None:
    """Fire N streamed requests at the same moment for each level, waiting `gap` seconds between levels so
    the per-minute limit doesn't confound the result. Watch whether latency rises or errors appear as N grows."""
    if p.dry_run:
        print(f"would send {sum(a.levels)} requests to {a.model} in levels {a.levels}, {a.gap}s apart")
        return
    for lvl in a.levels:
        recs = await asyncio.gather(*[
            p.call(phase="concurrency", model=a.model, size="small", max_tokens=128, meta={"level": lvl, "slot": j})
            for j in range(lvl)])
        ok = [r for r in recs if r.get("status_code") == 200]
        ttfts = sorted(r["first_content_ms"] for r in ok if r.get("first_content_ms"))
        print(f"level {lvl:>2}: ok={len(ok)}/{lvl} statuses={sorted({r.get('status_code') for r in recs}, key=str)} "
              f"ttft_min={ttfts[0] if ttfts else None} ttft_max={ttfts[-1] if ttfts else None}")
        await asyncio.sleep(a.gap)


# Prompt tokens Groq counted for these sizes on Day 1 (usage.prompt_tokens); used only for dry-run estimates.
MEASURED_PROMPT_TOKENS = {"tiny": 98, "large": 2813}


def watch_line(rec: dict) -> str:
    return (f"   +{rec.get('since_first_daily_429_ms', 0) / 1000:7.1f}s [{rec['phase']}/{rec.get('watch_stage')}] "
            f"{rec['size']:<5} status={rec.get('status_code')} limit={rec.get('limit_hit')} "
            f"used={rec.get('limit_used')} requested={rec.get('limit_requested')} "
            f"stated_wait={rec.get('stated_wait_s')}")


async def phase_daily(p: Prober, a) -> None:
    """Run the model's daily budget down until a 429 that names a per-day limit, then watch it come back.
    Use a model the pipeline does NOT use.

    1. Exhaust: requests paced just under the per-minute limits, so the 429 we get is the daily one.
       With --target tokens, every second request asks for one word with a large max_tokens. The daily 429's
       "Used N" then tells us whether the daily budget is charged max_tokens up front or actual output.
    2. Drain: after the first daily 429 there are usually still a few thousand tokens left, enough for small
       requests. Tiny requests use them up, so recovery is timed from a truly empty bucket.
    3. Watch: a large request every --watch-every seconds (or sooner, at the wait the last 429 stated), and a
       tiny one in between until a tiny gets through. Refused calls are free and each one's "Used N" is a reading
       of the bucket, so the readings give the refill rate. After --watch-minutes, poll every
       --slow-every-minutes, up to --watch-hours, so a fixed-time reset or a rolling 24 h window is caught too.
       Stops at the first large success.
    --check-model sends one tiny request to that model before and after, to show whether daily budgets are
    per model (its counter shouldn't move) or shared."""
    if not a.confirm_exhaust:
        sys.exit("refusing to run: pass --confirm-exhaust. This spends the whole daily budget of " + a.model)
    tokens = a.target == "tokens"
    size, base_mt = ("large", 64) if tokens else ("tiny", 8)
    min_gap = 60.0 / (a.rpm * a.safety)

    def plan(i: int) -> tuple[str, int]:
        if tokens and a.reserve_test_tokens and i % 2 == 1:
            return "ok", a.reserve_test_tokens
        return "explain", base_mt

    def pace(charged: float) -> float:
        return max(min_gap, charged / (a.tpm * a.safety) * 60)

    if p.dry_run:
        prompt = MEASURED_PROMPT_TOKENS[size]
        charges = [prompt + plan(i)[1] for i in range(2)]
        per_req = sum(charges) / 2
        n = int(a.expected_tpd / per_req) if tokens else int(a.expected_rpd)
        mins = n * sum(pace(c) for c in charges) / 2 / 60
        print(f"model {a.model}, target {a.target}, pacing at {a.safety:.0%} of RPM {a.rpm:g} / TPM {a.tpm:g}")
        print(f"1. exhaust: ~{n} '{size}' requests (if TPD={a.expected_tpd:,}), ~{mins:.0f} min, "
              f"~{n * per_req:,.0f} tokens charged (all of {a.model}'s daily budget)")
        if tokens and a.reserve_test_tokens:
            print(f"   every 2nd request: one-word reply with max_tokens={a.reserve_test_tokens} "
                  f"(reserved vs used differ by ~{a.reserve_test_tokens - 5} tokens each)")
        if tokens:
            print(f"2. drain: up to {a.drain_max} tiny requests ({min_gap:.1f}s apart) until a tiny gets a daily 429;"
                  f" spends only the leftover (< ~{prompt + base_mt:,} tokens)")
        close_polls = int(a.watch_minutes * 60 / a.watch_every)
        slow_polls = int(max(0, a.watch_hours * 60 - a.watch_minutes) / a.slow_every_minutes)
        print(f"3. watch: up to {close_polls} large + {close_polls} tiny polls in the first {a.watch_minutes:g} min, "
              f"then up to {slow_polls} of each every {a.slow_every_minutes:g} min, up to {a.watch_hours:g} h. "
              f"Refused polls cost nothing; at most one tiny and one large succeed (~{prompt + base_mt + 110:,} tokens)")
        if a.check_model:
            print(f"   plus 2 tiny requests to {a.check_model} (before and after), ~200 tokens of its budget")
        return

    if a.check_model:
        print(summary(await p.call(phase="daily_check_before", model=a.check_model, size="tiny", max_tokens=8,
                                   stream=False)))
    t_start = time.perf_counter()
    deadline = t_start + a.max_minutes * 60
    i, hit = 0, None
    while time.perf_counter() < deadline:
        task, mt = plan(i)
        rec = await p.call(phase=f"daily_{a.target}", model=a.model, size=size, max_tokens=mt, stream=False,
                           task=task, meta={"seq": i, "since_daily_start_ms": ms_since(t_start)})
        i += 1
        if i % 10 == 0 or rec.get("status_code") != 200:
            print(summary(rec))
        if rec.get("limit_hit") == "daily":
            hit = rec
            break
        if rec.get("status_code") == 429:
            await asyncio.sleep(rec.get("stated_wait_s") or parse_duration_s(
                (rec.get("headers") or {}).get("retry-after")) or 10)
            continue
        prompt = (rec.get("usage") or {}).get("prompt_tokens") or est_tokens(size, 0)
        await asyncio.sleep(pace(prompt + mt))
    if hit is None:
        print(f"   stopped at --max-minutes after {i} requests without a daily 429")
        return
    t_hit = time.perf_counter()
    print(f"   daily limit hit after {i} requests, {(t_hit - t_start) / 60:.1f} min in.\n   message: "
          f"{error_message(hit)[:400]}")
    if a.check_model:
        print(summary(await p.call(phase="daily_check_after", model=a.check_model, size="tiny", max_tokens=8,
                                   stream=False)))
    await daily_watch(p, a, size, base_mt, t_hit, drain=tokens)


async def daily_watch(p: Prober, a, big_size: str, big_mt: int, t_hit: float, drain: bool) -> None:
    def el() -> float:
        return time.perf_counter() - t_hit

    def meta(stage: str, k: int, kind: str) -> dict:
        return {"watch_stage": stage, "poll": k, "kind": kind, "since_first_daily_429_ms": round(el() * 1000)}

    if drain:
        for k in range(a.drain_max):
            rec = await p.call(phase="daily_drain", model=a.model, size="tiny", max_tokens=8, stream=False,
                               meta=meta("drain", k, "tiny"))
            print(watch_line(rec))
            if rec.get("limit_hit") == "daily":
                print(f"   bucket empty: {k} tiny requests still got through after the large one was refused")
                break
            if rec.get("status_code") == 429:
                await asyncio.sleep(rec.get("stated_wait_s") or 2)
                continue
            await asyncio.sleep(60.0 / (a.rpm * a.safety))
        else:
            print(f"   tiny requests still pass after {a.drain_max}; the watch starts with tokens left over")

    primary = "large" if big_size != "tiny" else "tiny"
    kinds = [primary] + (["tiny"] if primary == "large" else [])
    next_at = {"large": el() + a.watch_every, "tiny": el() + a.watch_every / 2}
    done: set[str] = set()
    k = 0
    while el() < a.watch_hours * 3600 and primary not in done:
        kind, t = min(((kd, next_at[kd]) for kd in kinds if kd not in done), key=lambda x: x[1])
        await asyncio.sleep(max(0.0, t - el()))
        stage = "close" if el() < a.watch_minutes * 60 else "slow"
        size, mt = (big_size, big_mt) if kind == "large" else ("tiny", 8)
        rec = await p.call(phase="daily_watch", model=a.model, size=size, max_tokens=mt, stream=False,
                           meta=meta(stage, k, kind))
        k += 1
        print(watch_line(rec))
        if rec.get("status_code") == 200:
            done.add(kind)
            print(f"   {kind} request succeeded {el():.0f}s after the first daily 429")
            continue
        step = a.watch_every if el() < a.watch_minutes * 60 else a.slow_every_minutes * 60
        stated = rec.get("stated_wait_s")
        # Probe again exactly when the 429 said to, if that's sooner: this checks whether the stated wait is true.
        next_at[kind] = el() + (min(step, stated + 1) if stated is not None else step)
    if primary not in done:
        print(f"   no {primary} request succeeded within {a.watch_hours:g} h")


# --------------------------------------------------------------------------- cli
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="print the plan; send nothing")
    ap.add_argument("--reasoning-effort", default="low", help="for gpt-oss models only; '' to omit")
    ap.add_argument("--rpm", type=float, default=30, help="assumed requests/min, used for pacing")
    ap.add_argument("--tpm", type=float, default=8000, help="assumed tokens/min, used for pacing")
    ap.add_argument("--safety", type=float, default=0.8, help="pace at this fraction of rpm/tpm")
    ap.add_argument("--poll-interval", type=float, default=5.0)
    ap.add_argument("--max-polls", type=int, default=24)
    sub = ap.add_subparsers(dest="phase", required=True)

    s = sub.add_parser("headers"); s.add_argument("--models", nargs="+", required=True)
    s = sub.add_parser("latency"); s.add_argument("--model", required=True)
    s.add_argument("--per-size", type=int, default=15)
    s.add_argument("--sizes", nargs="+", default=["small", "medium", "large"], choices=list(SIZES))
    s.add_argument("--max-tokens", type=int, default=200)
    s = sub.add_parser("rpm"); s.add_argument("--model", required=True); s.add_argument("--burst", type=int, default=35)
    s = sub.add_parser("tpm"); s.add_argument("--model", required=True)
    s.add_argument("--size", default="large", choices=list(SIZES)); s.add_argument("--max-requests", type=int, default=6)
    s = sub.add_parser("concurrency"); s.add_argument("--model", required=True)
    s.add_argument("--levels", nargs="+", type=int, default=[1, 2, 4, 8, 16]); s.add_argument("--gap", type=float, default=30)
    s = sub.add_parser("daily"); s.add_argument("--model", required=True)
    s.add_argument("--target", choices=["requests", "tokens"], default="tokens")
    s.add_argument("--max-minutes", type=float, default=120, help="cap on the exhaust stage only")
    s.add_argument("--confirm-exhaust", action="store_true")
    s.add_argument("--check-model", help="send one tiny request to this model before and after (per-model budgets?)")
    s.add_argument("--reserve-test-tokens", type=int, default=600,
                   help="max_tokens for every 2nd exhaust request (one-word reply); 0 disables")
    s.add_argument("--drain-max", type=int, default=60)
    s.add_argument("--watch-every", type=float, default=60, help="seconds between polls in the close watch")
    s.add_argument("--watch-minutes", type=float, default=40)
    s.add_argument("--slow-every-minutes", type=float, default=15)
    s.add_argument("--watch-hours", type=float, default=6)
    s.add_argument("--expected-tpd", type=int, default=200_000, help="dry-run estimate only (console value)")
    s.add_argument("--expected-rpd", type=int, default=1000, help="dry-run estimate only")
    a = ap.parse_args()

    load_dotenv(ROOT / ".env")
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key and not a.dry_run:
        sys.exit("GROQ_API_KEY is empty. Put it in .env (never commit that file).")

    async def run():
        p = Prober(key, a.reasoning_effort or None, a.dry_run)
        try:
            await {"headers": phase_headers, "latency": phase_latency, "rpm": phase_rpm, "tpm": phase_tpm,
                   "concurrency": phase_concurrency, "daily": phase_daily}[a.phase](p, a)
        finally:
            await p.close()
            if not a.dry_run:
                print(f"\n{p.sent} requests sent; raw lines appended to {OUT.relative_to(ROOT)} (run {RUN_ID})")

    asyncio.run(run())


if __name__ == "__main__":
    main()
