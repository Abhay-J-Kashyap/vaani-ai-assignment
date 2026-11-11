"""Pipeline constants and retrieval-setting names. One place, so the write-up can quote them.

Retrieval settings are named compactly and recorded verbatim in results.csv's `retrieval_setting`:
  h256k5     hybrid (BM25 + dense, RRF), 256-token pieces, 5 pieces
  h512k4rr   hybrid, 512-token pieces, 4 pieces, cross-encoder re-rank
  bm256k8    BM25 only (the naive baseline), 256-token pieces, 8 pieces
The answer tier is appended when it isn't Full (see `setting_label`); a cache hit is just "cache" (author, 2026-11-09).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from relay.ledger import DRAFTER, ROUTER

# --- models and token allowances. max_tokens is charged to TPM up front and includes gpt-oss reasoning tokens;
# too small and gpt-oss returns an empty answer (measured Day 1), so these are tight but not starved.
FULL_MODEL, LEAN_MODEL, ROUTER_MODEL = DRAFTER, ROUTER, ROUTER
FULL_MAX_TOKENS = 900
LEAN_MAX_TOKENS = 600
ROUTER_MAX_TOKENS = 300
CHAT_OVERHEAD = 78           # chat-template tokens Groq bills on top of the message text (measured)
TOOLS_TOKENS = 260            # the two tool schemas, as billed (estimate; corrected by usage on every call)
TABLE_EXPAND_MAX = 1200      # give the drafter a whole table when any part is retrieved, up to this size
RETRIEVAL_TIMEOUT_S = 2.0     # the retrieval tool's deadline (warm h256k8: ~66 ms p50); re-rank settings get 12 s
MAX_PASSAGE_TOKENS = 2600    # cap on retrieved text in one drafter prompt (decompose merges several queries)

# --- deadlines (ADR "fast" targets: first-token p95 <= 4,000 ms, total p95 <= 8,000 ms)
FIRST_TOKEN_TIMEOUT_S = 4.0   # per call: no answer token by then -> timeout (measured p99 ~2.4 s; a 503 hangs ~30 s)
CALL_TOTAL_TIMEOUT_S = 20.0   # per call: a trickle (~70 tok/s) is slow but useful
ROUTER_TIMEOUT_S = 3.0
DEFAULT_PATIENCE_S = 30.0     # how long a person waits before giving up, unless the load generator samples one
DRAFT_MARGIN_S = 2.0          # don't start a full draft with less than this left before the deadline

# --- retrieval loop and decide branch. Confidence is the top re-rank score when re-ranked, else the top dense
# cosine. Provisional, from the 20 gold questions at h256k5 (2026-11-10): re-rank scores of the 3 clearly
# unanswerable questions were -1.4..-6.6, every answerable one >= -0.06; dense cosine does not separate them
# (unanswerable 0.66-0.83, answerable 0.77-0.93), so without re-rank the refuse threshold only catches the extreme.
TAU = {"rerank": {"loop": 1.0, "refuse": -1.0}, "cos": {"loop": 0.80, "refuse": 0.70}}
MAX_REFORMULATIONS = 2        # retrieval loop: at most 2 extra searches
MAX_DRAFTER_CALLS = 2         # author, 2026-11-10: draft plus one regeneration

# --- degradation ladder L1: per-minute headroom below this share -> heuristic router, k-1, no re-rank
L1_HEADROOM = 0.30

# --- retry policy (ADR "Retry policy")
RETRY_BASE_S, RETRY_CAP_S = 0.25, 2.0
RETRY_BUDGET_SHARE = 0.10     # retries <= 10% of requests in a rolling 60 s window
FIRST_TOKEN_P50_S = 0.6       # measured drafter first-token p50; a first-token-timeout retry needs 2x this left

PAPERS = ("AIAYN", "MOSHI", "HNET")


@dataclass(frozen=True)
class Setting:
    name: str
    size: int
    k: int
    rerank: bool
    dense: bool = True
    late: bool = False           # 'cb' suffix: ColBERT late-interaction re-rank (experiment, 2026-11-11)


SETTING_RE = re.compile(r"^(h|bm)(\d+)k(\d+)(rr|cb)?$")


def parse_setting(name: str) -> Setting:
    m = SETTING_RE.match(name)
    if not m:
        raise ValueError(f"retrieval setting must look like h256k5, h512k4rr, h256k8cb or bm256k8; got {name!r}")
    return Setting(name, int(m.group(2)), int(m.group(3)), m.group(4) == "rr", m.group(1) == "h", m.group(4) == "cb")


DEFAULT_SETTING = "h256k8"    # chosen from Table 3 by the author, 2026-11-11 (65.0% good, 0 misreads past the gate)


def setting_label(setting: str, tier: str) -> str:
    """What goes in results.csv's retrieval_setting: the setting, plus the tier when it isn't Full."""
    if tier == "cached":
        return "cache"
    return setting if tier == "full" else f"{setting}|{tier}"
