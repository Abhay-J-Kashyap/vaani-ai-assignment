"""Budget ledger: one account per model, plus the tier policy (ADR "Budget plan", author's decisions 2026-11-09/10).

Accounts track RPM, TPM, RPD and TPD as continuous buckets (relay.limits.Bucket).
  * RPD and TPM are re-synced from every response's x-ratelimit-* headers (Groq reports those two).
  * TPD is not in Groq's headers. Each call reserves prompt + max_tokens against TPD when it starts (so calls in
    flight count against the daily budget), then finish() swaps the reservation for the actual usage; a call
    that fails or is rejected gets it all back. The ledger also corrects itself from a daily 429's "Used N".
    (Before 2026-11-13, TPD was charged only at finish(); see the ADR. Every measured run used that version.)
  * A 429's stated wait is exact (measured), so it sets the bucket so that the wait matches.
State persists to data/state/ledger_<name>.json with wall-clock times, so the daily budget carries across runs.

Tier policy:
  Full (gpt-oss-120b) while its TPD level, minus the Full answers already promised but not yet reserved
    (`committed`), minus this answer's cost, stays above the reserve floor.
    Reserve = 20% of TPD, released evenly from 00:00 to 18:00 local: floor(t) = R * (1 - t/18h), 0 after 18:00.
  Lean (gpt-oss-20b draft) when Full can't, as long as the 20b stays above 25% (routing has priority there).
  Extractive (no Groq call) otherwise. Cached answers are decided before the ledger is consulted.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from relay.limits import Bucket, parse_429

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "data" / "state"
DRAFTER, ROUTER = "openai/gpt-oss-120b", "openai/gpt-oss-20b"
DEFAULT_LIMITS = {"rpm": 30, "tpm": 8000, "rpd": 1000, "tpd": 200_000}
RESERVE_SHARE = 0.20          # author, 2026-11-10
RESERVE_UNTIL_HOUR = 18       # author, 2026-11-10
ROUTER_FLOOR_SHARE = 0.25     # author approved (c), 2026-11-09


@dataclass
class Account:
    model: str
    rpm: Bucket
    tpm: Bucket
    rpd: Bucket
    tpd: Bucket

    @classmethod
    def new(cls, model: str, limits: dict) -> "Account":
        return cls(model, Bucket(limits["rpm"], limits["rpm"] / 60), Bucket(limits["tpm"], limits["tpm"] / 60),
                   Bucket(limits["rpd"], limits["rpd"] / 86400), Bucket(limits["tpd"], limits["tpd"] / 86400))

    def buckets(self):
        return {"rpm": self.rpm, "tpm": self.tpm, "rpd": self.rpd, "tpd": self.tpd}


class Ledger:
    def __init__(self, name: str = "real", limits: dict | None = None, persist: bool = True):
        self.name, self.persist = name, persist
        self.limits = dict(limits or DEFAULT_LIMITS)
        self.accounts: dict[str, Account] = {}
        self.path = STATE / f"ledger_{name}.json"
        if persist and self.path.exists():
            self._load()

    # ------------------------------------------------------------------ accounts and persistence
    def acct(self, model: str) -> Account:
        if model not in self.accounts:
            self.accounts[model] = Account.new(model, self.limits)
        return self.accounts[model]

    def _load(self) -> None:
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        for model, bs in raw.get("accounts", {}).items():
            a = self.acct(model)
            for k, b in bs.items():
                bucket = a.buckets()[k]
                bucket.capacity, bucket.per_s, bucket.level_at, bucket.t = b["capacity"], b["per_s"], b["level"], b["t"]

    def save(self) -> None:
        if not self.persist:
            return
        STATE.mkdir(parents=True, exist_ok=True)
        data = {"saved_at": time.time(), "accounts": {
            m: {k: {"capacity": b.capacity, "per_s": b.per_s, "level": b.level_at, "t": b.t} for k, b in a.buckets().items()}
            for m, a in self.accounts.items()}}
        self.path.write_text(json.dumps(data, indent=1), encoding="utf-8")

    # ------------------------------------------------------------------ admission and charging
    def wait_s(self, model: str, prompt_tokens: int, max_tokens: int, now: float | None = None) -> tuple[float, str]:
        """Seconds until a call of this size is allowed by every limit, and which limit binds."""
        a = self.acct(model)
        need = prompt_tokens + max_tokens
        waits = {"rpm": a.rpm.wait_for(1, now), "tpm": a.tpm.wait_for(need, now),
                 "rpd": a.rpd.wait_for(1, now), "tpd": a.tpd.wait_for(need, now)}
        which = max(waits, key=waits.get)
        return waits[which], which

    def start(self, model: str, prompt_tokens: int, max_tokens: int, now: float | None = None) -> int:
        """Reserve one call: RPM, RPD, and prompt + max_tokens against both TPM and TPD. Returns the TPD
        reservation, which finish() or on_429() must get back as `reserved`."""
        a = self.acct(model)
        need = prompt_tokens + max_tokens
        a.rpm.take(1, now)
        a.rpd.take(1, now)
        a.tpm.take(need, now)
        a.tpd.take(need, now)
        return need

    def finish(self, model: str, usage: dict | None, headers: dict | None, now: float | None = None,
               reserved: int = 0) -> None:
        """Swap the call's TPD reservation for its actual usage (all of it back if no usage was reported);
        re-sync RPD and TPM from Groq's headers when present."""
        a = self.acct(model)
        used = 0
        if usage:
            total = usage.get("total_tokens") or (usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0))
            # Groq: cached prompt tokens (automatic prefix caching on gpt-oss) don't count toward rate limits
            cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
            used = max(0, total - cached)
        if used or reserved:
            a.tpd.take(used - reserved, now)
        h = {k.lower(): v for k, v in (headers or {}).items()}
        if h.get("x-ratelimit-remaining-requests") is not None:
            a.rpd.set_level(float(h["x-ratelimit-remaining-requests"]), now)
        if h.get("x-ratelimit-remaining-tokens") is not None:
            a.tpm.set_level(float(h["x-ratelimit-remaining-tokens"]), now)
        self.save()

    def on_429(self, model: str, body, now: float | None = None, reserved: int = 0) -> dict:
        """Give back the call's TPD reservation (rejected calls aren't counted, measured), then set the named
        bucket so its wait equals the 429's stated wait (measured to be exact)."""
        if reserved:
            self.acct(model).tpd.take(-reserved, now)
        info = parse_429(body)
        which = info.get("which")
        if which:
            b = self.acct(model).buckets()[which]
            if info.get("limit"):
                b.capacity, b.per_s = info["limit"], info["limit"] / (86400 if info["daily"] else 60)
            need = info.get("requested", 1)
            wait = info.get("wait_s")
            if wait is not None:
                b.set_level(need - wait * b.per_s, now)
            elif info.get("used") is not None and info.get("limit"):
                b.set_level(info["limit"] - info["used"], now)
            self.save()
        return info

    # ------------------------------------------------------------------ tier policy
    def reserve_floor(self, when: datetime | None = None) -> float:
        when = when or datetime.now()
        frac = min(1.0, (when.hour + when.minute / 60 + when.second / 3600) / RESERVE_UNTIL_HOUR)
        return RESERVE_SHARE * self.acct(DRAFTER).tpd.capacity * (1 - frac)

    def tier(self, full_cost: int, lean_cost: int, now: float | None = None, when: datetime | None = None,
             committed: float = 0.0) -> str:
        """`committed`: TPD already promised to Full requests that haven't reserved their draft call yet (still
        routing, retrieving or queued). Calls in flight are already reserved in the TPD level itself."""
        d, r = self.acct(DRAFTER).tpd, self.acct(ROUTER).tpd
        if d.level(now) - committed - full_cost >= self.reserve_floor(when):
            return "full"
        if r.level(now) - lean_cost >= ROUTER_FLOOR_SHARE * r.capacity:
            return "lean"
        return "extractive"

    def snapshot(self, now: float | None = None) -> dict:
        return {m: {k: round(b.level(now), 1) for k, b in a.buckets().items()} for m, a in self.accounts.items()}
