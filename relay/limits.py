"""Continuous token buckets and Groq's rate-limit wire format, shared by the budget ledger and the stand-in.

Measured on Groq (ADR, probe findings): every limit refills continuously, with no fixed windows.
  RPM 30 -> 0.5 req/s; TPM 8,000 -> 133 tok/s; RPD 1,000 -> 1 req / 86.4 s; TPD 200,000 -> 2.31 tok/s.
TPM and TPD admission both check prompt + max_tokens ("Requested 2821" = prompt 2757 + max_tokens 64 on the
daily 429). TPM is charged that reservation; TPD is charged the tokens actually used.
"""
from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field


@dataclass
class Bucket:
    capacity: float
    per_s: float
    level_at: float = field(default=None)   # level at time `t`
    t: float = field(default=None)

    def __post_init__(self):
        if self.level_at is None:
            self.level_at = self.capacity
        if self.t is None:
            self.t = time.time()

    def level(self, now: float | None = None) -> float:
        now = time.time() if now is None else now
        return min(self.capacity, self.level_at + max(0.0, now - self.t) * self.per_s)

    def set_level(self, level: float, now: float | None = None) -> None:
        self.t = time.time() if now is None else now
        self.level_at = max(-self.capacity, min(self.capacity, level))

    def take(self, n: float, now: float | None = None) -> None:
        now = time.time() if now is None else now
        self.set_level(self.level(now) - n, now)

    def wait_for(self, n: float, now: float | None = None) -> float:
        """Seconds until `n` fits (0 if it fits now); inf if it can never fit."""
        if n > self.capacity:
            return math.inf
        lvl = self.level(now)
        return 0.0 if lvl >= n else (n - lvl) / self.per_s


def fmt_wait(s: float) -> str:
    """Groq style: '2s', '3.4125s', '19m39.792s', '1h2m3s'."""
    s = max(s, 0.0)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    out = (f"{int(h)}h" if h else "") + (f"{int(m)}m" if (h or m) else "")
    return out + (f"{sec:.3f}".rstrip("0").rstrip(".") + "s")


LIMIT_NAMES = {"rpm": ("requests per minute (RPM)", "requests"), "tpm": ("tokens per minute (TPM)", "tokens"),
               "rpd": ("requests per day (RPD)", "requests"), "tpd": ("tokens per day (TPD)", "tokens")}


def error_429(model: str, which: str, limit: int, used: int, requested: int, wait_s: float) -> dict:
    name, typ = LIMIT_NAMES[which]
    msg = (f"Rate limit reached for model `{model}` in organization `org_standin` service tier `on_demand` on {name}: "
           f"Limit {limit}, Used {used}, Requested {requested}. Please try again in {fmt_wait(wait_s)}. "
           f"Need more tokens? Upgrade to Dev Tier today at https://console.groq.com/settings/billing")
    return {"error": {"message": msg, "type": typ, "code": "rate_limit_exceeded"}}


def parse_429(body: dict | str | None) -> dict:
    """Which limit, its numbers and the stated wait, from a Groq (or stand-in) 429 body."""
    msg = str((body.get("error") or {}).get("message") if isinstance(body, dict) else body or "")
    low = msg.lower()
    which = ("tpd" if "tokens per day" in low else "rpd" if "requests per day" in low else
             "tpm" if "tokens per minute" in low else "rpm" if "requests per minute" in low else None)
    out = {"which": which, "daily": which in ("tpd", "rpd")}
    for name in ("Limit", "Used", "Requested"):
        m = re.search(rf"{name} (\d+)", msg)
        if m:
            out[name.lower()] = int(m.group(1))
    m = re.search(r"try again in ([0-9hms.]+)", msg)
    if m:
        total = 0.0
        for num, unit in re.findall(r"([\d.]+)(ms|h|m|s)", m.group(1).rstrip(".")):
            total += float(num) * {"h": 3600, "m": 60, "s": 1, "ms": 0.001}[unit]
        out["wait_s"] = total
    return out
