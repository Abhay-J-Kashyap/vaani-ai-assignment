"""Runtime helpers for the pipeline: the answer cache, the global retry budget, and the row/trace recorder."""
from __future__ import annotations

import csv
import json
import random
import re
import time
from collections import deque
from pathlib import Path

from relay.settings import RETRY_BASE_S, RETRY_BUDGET_SHARE, RETRY_CAP_S

ROOT = Path(__file__).resolve().parents[1]
COLUMNS = ["request_id", "scenario", "source", "start_time", "first_token_ms", "total_ms", "status", "retries",
           "tokens_in", "tokens_out", "retrieval_setting", "answer_grade"]


class AnswerCache:
    """Exact-match cache on the normalised question (author's decision 4). Not semantic: two differently worded
    questions can need different answers, and a wrong cache hit would be a wrong answer. Only answers that passed
    the gate whole (nothing withheld, not truncated) are stored."""

    def __init__(self, name: str, persist: bool):
        self.path = ROOT / "data" / "state" / f"cache_{name}.json"
        self.persist = persist
        self.data: dict = {}
        if persist and self.path.exists():
            self.data = json.loads(self.path.read_text(encoding="utf-8"))

    @staticmethod
    def key(question: str, setting: str) -> str:
        q = re.sub(r"[^a-z0-9 ]+", " ", question.lower())
        return f"{setting}::{' '.join(q.split())}"

    def get(self, question: str, setting: str) -> dict | None:
        return self.data.get(self.key(question, setting))

    def put(self, question: str, setting: str, entry: dict) -> None:
        self.data[self.key(question, setting)] = entry | {"stored_at": time.time()}
        if self.persist:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.data, ensure_ascii=False, indent=1), encoding="utf-8")


class RetryBudget:
    """ADR retry policy: retries may be at most 10% of requests in a rolling 60 s window (at least 1).
    mode 'naive' (for the retry experiment) allows every retry; 'none' allows none."""

    def __init__(self, mode: str = "policy"):
        self.mode = mode
        self.requests: deque = deque()
        self.retries: deque = deque()

    def _prune(self, now: float) -> None:
        for d in (self.requests, self.retries):
            while d and now - d[0] > 60:
                d.popleft()

    def note_request(self) -> None:
        self.requests.append(time.monotonic())

    def allow(self) -> bool:
        if self.mode == "naive":
            return True
        if self.mode == "none":
            return False
        now = time.monotonic()
        self._prune(now)
        return len(self.retries) < max(1, int(RETRY_BUDGET_SHARE * len(self.requests)))

    def take(self) -> None:
        self.retries.append(time.monotonic())

    @staticmethod
    def backoff(attempt: int) -> float:
        """Full-jitter exponential backoff: uniform(0, min(cap, base * 2^attempt))."""
        return random.uniform(0, min(RETRY_CAP_S, RETRY_BASE_S * 2 ** attempt))


class Recorder:
    """Appends one row per request (the 12 results.csv columns) and per-phase trace events. Never overwrites."""

    def __init__(self, rows_path: Path | None, trace_path: Path | None):
        self.rows_path, self.trace_path = rows_path, trace_path
        for p in (rows_path, trace_path):
            if p:
                p.parent.mkdir(parents=True, exist_ok=True)
        if rows_path and not rows_path.exists():
            with rows_path.open("w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=COLUMNS).writeheader()
        self._trace = trace_path.open("a", encoding="utf-8") if trace_path else None

    def row(self, r: dict) -> None:
        if not self.rows_path:
            return
        with self.rows_path.open("a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=COLUMNS).writerow({k: r.get(k, "") for k in COLUMNS})

    def trace(self, ev: dict) -> None:
        if self._trace:
            self._trace.write(json.dumps(ev, ensure_ascii=False) + "\n")
            self._trace.flush()

    def close(self) -> None:
        if self._trace:
            self._trace.close()
