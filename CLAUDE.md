# The Relay: Vaani take-home

A Python pipeline that answers engineers' questions using **only** three arXiv papers, on Groq's free tier. It must hold up under bursty load. A stand-in model service covers loads Groq can't take.

**The brief is the spec.** `The_Relay_Assignment_Brief.pdf` should be kept in the repo root. If it isn't there, ask the user for it. Check every plan against it, and point out anything it requires that we're skipping. The user wants this done strictly, without being asked.

**Design and decisions:** [docs/adr/0001-relay-pipeline-architecture.md](docs/adr/0001-relay-pipeline-architecture.md). Read it before you change the design. Add each new decision to it as it's made, with the date and who decided.

## Requirements checklist (condensed from the brief; the PDF wins if they differ)

- **Pipeline:** Python. ≥4 steps, ≥1 tool call, ≥1 branch or loop, and one step retrieves from the library.
- **Library:** arXiv 1706.03762**v7** (Attention), 2410.00037**v1** (Moshi), 2507.07955**v2** (Dynamic Chunking / H-Net), at exactly those versions.
  - Every answer cites where in the papers it came from.
  - If the papers don't say, the answer says so instead of guessing.
- **Stand-in service:** can be slow, fail, refuse at a limit, slow down under load, and stream. It is calibrated from real measurements, and the write-up says where it differs most from Groq.
- **Real probes first:** a modest number of requests to learn the latency spread, the limits, and the response at a limit.
- **Load runs:** ≥3 shapes. Table 2a requires 5 rows: steady, burst, sustained overload, per-minute limit halved, daily budget exhausted. Mark each row real or stand-in.
- **Answer quality:** at least 15 self-written questions, including some the papers don't answer. Report tokens per answer.
- **Retrieval:** at least 3 settings (piece size, pieces fetched, re-rank) with quality, tokens and first-token time. Pick one and say why. Say what retrieval does when the limit is close.
- **Raw file `results.csv`:** one row per request, with exactly these columns: `request_id, scenario, source (real|stand-in), start_time, first_token_ms, total_ms, status (ok|rate_limited|timeout|error), retries, tokens_in, tokens_out, retrieval_setting, answer_grade (correct|wrong|refused_correctly|refused_wrongly)`. **Every figure in every table must be computable from it.**
- **Tables:**
  - Table 1: limits, with value, how found and confidence.
  - Table 2a: latency, p50/p95/p99 for first token and total time.
  - Table 2b: outcomes, cost per 1,000 answers.
  - Table 3: retrieval settings.
  - Times in ms, percentages to one decimal place. Write "not measured" where something wasn't measured.
- **Write-up:** at most 2 pages, with these headings in this order:
  1. What I built and why
  2. The real service and my stand-in
  3. Results (include one number nobody could guess without running it)
  4. Where the time goes (waiting / working / retrying for one slow request; before or after the first token)
  5. When a limit is hit
  - **Retrieval: quality versus cost**
  - **How this library will mislead a naive retriever** (3 ways, each with an example question)
  6. Retries (the policy, plus one case where retrying made things worse)
  7. What the person sees (3pm, daily budget gone)
  8. Unsure / left out / learned
  - **Your feedback on Vaani** (vaani.dev)
  - **How I worked with the assistant**: the raw conversation history; the 3 prompts that mattered most; one thing the assistant got wrong that the user caught; one decision where the user overruled it; what in the submission is distinctly the user's
- **Ground rules:**
  - Submit within 4 days.
  - Never commit the key.
  - One free account. No tricks to get around limits.
  - The daily limit is shared across everything, so plan requests.

## Measured facts (Day 1, 2026-11-09; full table in the ADR; raw data in `results/probe_raw.jsonl`)

- **Models on this account:** `openai/gpt-oss-20b` is the router and `openai/gpt-oss-120b` the drafter. Each has RPD 1,000, TPM 8,000 and RPM 30. Llama models aren't available (404). The sacrificial model is `qwen/qwen3.8-27b`, with the same limits.
- **Refill:** both the daily and the per-minute budgets refill **continuously**. Daily: one request per 86.4 s. Per-minute: 133 tokens per second.
- **`max_tokens` is charged against TPM up front**, and it includes reasoning tokens. Keep it tight. If it's too small, gpt-oss returns an empty answer.
- **A 429** returns in ~55 ms. The body's `error.type` is `requests` or `tokens`, and the message gives the exact wait. Rejected calls aren't counted.
- **Latency:** first answer token p50 ~0.5–0.7 s. Groq's `usage.queue_time` is often half or more of it, and it spikes in correlated episodes. On the 120b, 5 of 120 responses (4.2%) trickled after the first token at ~70 tokens/s, all in the afternoon window. In the evening window the tail came from queue delays *before* the first token instead.
- **Tokens per day:** 200K for all three models, according to the console (`results/groq_console_limits_2026-11-09.md`). The console also confirms every measured RPM, RPD and TPM value.
- **The qwen daily test confirmed it** (run 20261109T180359Z):
  - TPD is 200,000, and the daily 429 states the exact wait.
  - It refills **continuously at 2.31 tokens/s**; there's no fixed reset.
  - The daily budget charges **tokens actually used**, not `max_tokens`. `max_tokens` costs only TPM.
  - Daily budgets are **per model**.
  - New failure mode: **503 "over capacity" after ~30 s** (2 of ~80 calls).
- **TPD is the binding daily limit: about 55 full answers per day on the drafter**, against an RPD of 1,000. Day 1 probing used ~82K of the 120b budget and ~70K of the 20b budget. Plan real calls per day before any eval or load run.
- **Budget plan and product budget manager:** see the "Budget plan" section of the ADR. It covers the 20% drafter reserve (released evenly until 18:00 local time), the answer tiers (Full on 120b, Lean on 20b, Extractive, Cached), and the answer cache.
- **`results.csv` is derived:** `tools/build_results_csv.py` rebuilds it from `results/probe_raw.jsonl` plus `results/pipeline_rows*.csv`. Probe rows use `scenario=probe_*` and leave `answer_grade` empty. Table 1 numbers must come from the CSV where it can supply them. For example, recovery is measured start-to-start: 2,051 ms (RPM) and 4,068 ms (TPM).

## Working rules

- **Never print, log or commit `GROQ_API_KEY`.** It lives only in `.env`, which is git-ignored. To check it, print its length and prefix only.
- **Every real Groq call costs shared daily budget.** Run `--dry-run` first, state the request and token cost, and get the user's OK before large runs. Don't exhaust a pipeline model's daily budget; it takes 24 h to refill fully.
- **Keep every raw result.** Append it; never overwrite `results/*.jsonl`. The reviewers will read them.
- **Calls about the design belong to the user.** Recommend, then record what they decide in the ADR. The user also has to report where they overruled the assistant, so don't quietly change a decision they made.
- **Environment:** Windows. Use `.\.venv\Scripts\python.exe` (Python 3.11). Inline `python -c` with quotes breaks in PowerShell, so write a script file instead.

## Commands

```
.\.venv\Scripts\python.exe -m probe.groq_probe --dry-run <phase> ...   # phases: headers latency rpm tpm concurrency daily
.\.venv\Scripts\python.exe -m probe.analyze_probe [--model M]          # Table 1 + calibration numbers from the raw file
.\.venv\Scripts\python.exe -m probe.list_models                        # models this key can use (not a chat call)
.\.venv\Scripts\python.exe -m tools.build_results_csv --check          # rebuild results.csv (12 columns) from raw logs; recompute Table 1 from it
.\.venv\Scripts\python.exe -m corpus.fetch                             # arXiv LaTeX sources at exact versions -> data/raw/ (+ manifest with sha256)
.\.venv\Scripts\python.exe -m corpus.build                             # parse -> data/corpus/blocks.jsonl; prints section/table/figure numbering to check against the PDFs
.\.venv\Scripts\python.exe -m corpus.chunk --sizes 128 256 512         # pieces per size -> data/corpus/chunks_<size>.jsonl
.\.venv\Scripts\python.exe -m relay.retrieval "question" --size 256 --k 5 [--rerank]   # local hybrid search; no Groq
.\.venv\Scripts\python.exe -m probe.calibrate                          # probe_raw.jsonl -> standin/calibration.json
.\.venv\Scripts\python.exe -m standin.server --port 8088 [--tpm-scale 0.5] [--tpd-start 0]   # stand-in Groq; admin at /admin/state, /admin/config
.\.venv\Scripts\python.exe -m tests.smoke_standin                      # client + ledger + stand-in end to end; no Groq
.\.venv\Scripts\python.exe -m eval.check_key                           # every answer-key cite resolves to a real corpus place
.\.venv\Scripts\python.exe -m relay.pipeline "question" [--setting h256k5] [--no-cache]   # one request on the stand-in (start it first); logs to data/dev/
.\.venv\Scripts\python.exe -m relay.pipeline "question" --source real --dry-run           # route + retrieve locally, print the token cost; no Groq
.\.venv\Scripts\python.exe -m tests.test_gate                          # citation gate + streaming sentence splitter
.\.venv\Scripts\python.exe -m tests.smoke_pipeline                     # every pipeline path on its own stand-in (burst, TPM halved, daily gone, give-up)
# Night daily test (spends qwen's whole daily budget; exhaust ~26 min, then watches recovery up to 6 h):
.\.venv\Scripts\python.exe -m probe.groq_probe --safety 0.95 daily --model qwen/qwen3.8-27b --target tokens --max-minutes 180 --confirm-exhaust --check-model openai/gpt-oss-20b
```
