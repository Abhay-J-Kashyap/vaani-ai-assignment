# The Relay

A Python pipeline that answers engineers' questions using only three arXiv papers (Attention Is All You Need v7, Moshi v1, Dynamic Chunking / H-Net v2), on Groq's free tier, built to hold up under bursty load. The write-up is [WRITEUP.md](WRITEUP.md); every design decision, with its date and who made it, is in [docs/adr/0001-relay-pipeline-architecture.md](docs/adr/0001-relay-pipeline-architecture.md).

## Recompute the tables from the raw file

Every figure in the write-up's tables comes from `results.csv` (one row per request, the brief's 12 columns):

```
python -m tools.build_results_csv --check   # rebuild results.csv from the raw logs; recompute Table 1 figures
python -m tools.tables --table 1            # Table 1: limits
python -m tools.tables --table 2            # Tables 2a and 2b: latency and outcomes per load scenario
python -m tools.tables --table 3            # Table 3: retrieval settings (first eval run)
python -m tools.tables --table 3 --scenario eval_v2   # the chosen setting on the shipped pipeline
```

Definitions used by `tools/tables.py`:
- Percentiles are nearest-rank.
- First-token percentiles cover requests that got any answer text; total-time percentiles cover all requests.
- Peak queue length is the most requests in the system at once, from `start_time` and `total_ms` overlaps.
- Cost per 1,000 answers is an upper bound at Groq list prices: Full rows at the 120b price, other tiers at the 20b price.
- In Table 3, "Wrong" includes wrong refusals, so the three outcome columns sum to 100%.

## What is where

| Path | What it holds |
|---|---|
| `results.csv` | The raw file the brief asks for. Derived: rebuilt from the files below, never edited by hand |
| `results/probe_raw.jsonl` | Every real probe request to Groq, with headers and bodies (Table 1 evidence) |
| `results/pipeline_rows_*.csv` | Every pipeline request, one file per scenario; real and stand-in, told apart by `source` |
| `results/trace.jsonl` | Per-phase events for every pipeline request: waits, calls, gate decisions, notices (heading 4) |
| `results/eval_answers.jsonl`, `results/grades_eval*.csv` | The gold-set answers as sent (plus what the gate withheld) and their grades with notes |
| `eval/` | The author's 20 questions and answer key with the frozen two-level rubric, the eval runner and grading tools |
| `relay/` | The pipeline: admission and budget ledger, router (tool call), retrieval, decide, draft, citation gate |
| `standin/` | The stand-in model service and its calibration from the probe data |
| `load/` | The load generator for the Table 2 scenarios and the retry experiment |
| `corpus/` | Fetches the arXiv LaTeX at the exact versions, parses it and chunks it |

## Run it

Needs Python 3.11, `pip install -r requirements.txt`, and a Groq key in `.env` as `GROQ_API_KEY=...` (the key is never committed).

```
python -m corpus.fetch && python -m corpus.build && python -m corpus.chunk     # the library, at exact versions
python -m standin.server --port 8088                                         # the stand-in, in another terminal
python -m relay.pipeline "Why are dot products scaled by sqrt(d_k)?"           # one request on the stand-in
python -m relay.pipeline "..." --source real --dry-run                        # token cost of a real request, no call
python -m tests.test_gate && python -m tests.smoke_pipeline                    # tests, no Groq
python -m load.run_load overload                                              # a stand-in load scenario
```

Real Groq calls spend a shared daily budget (200K tokens per model), so every real run has a `--dry-run` and needs `--confirm-real`.
