# The Relay: write-up

> **DRAFT, 2026-11-12.** Every table is recomputed from `results.csv` by `python -m tools.tables --table 1|2|3`. **[AUTHOR]** = the author's own words. Detail behind each claim: `docs/adr/0001-relay-pipeline-architecture.md`.

## 1. What I built and why
Seven steps:
1. admit (cache, budget ledger);
2. route: a gpt-oss-20b **tool call**;
3. retrieve: local BM25 + dense search, with a **loop** that rewrites low-confidence queries;
4. decide: a **branch** to answer, refuse, or decompose per paper;
5. draft on gpt-oss-120b, streamed;
6. gate;
7. log.

**The decision that shaped it most** is a deterministic per-sentence citation gate. A sentence is sent only if its citations name retrieved passages and its numbers appear in them. It costs no tokens and keeps streaming. Its blind spot, a real number on the wrong table row, is measured, not hidden.

**Second decision:** budget the *day*. 200K tokens per model per day binds long before the request limits, so a ledger rations the 120b, then falls back to shorter 20b answers, then to quoted passages.

**We trade *good* for *fast* and *nobody turned away* under load,** never the reverse: a cut-off answer never shows its half-sentence.

## 2. The real service and my stand-in
**Table 1** comes from 341 real probe requests, Groq's 429 bodies and headers, and the console. A final-day recheck matched.

| Limit | Value found | How found | Confidence |
|---|---|---|---|
| Requests per minute | 30 | 30 ok in a burst, then 429 at 13,009 ms; 429 body; console | high |
| Tokens per minute | 8,000 | ≥5,755 accepted in the 60 s before a 429 (`max_tokens` is reserved up front); 429 body; console | high |
| Requests per day | 1,000 | `x-ratelimit-limit-requests` header; console; not driven to the limit | medium |
| Tokens per day | 200,000 per model | 201,298 tokens in 72 requests before the daily 429; refills continuously at 2.31 tokens/s | high |
| Requests in flight at once | no limit at 16 | 16 overlapping requests all ok; RPM 30 stops a larger test | low |
| What it returns at a limit | HTTP 429 in 99 ms (p50) | `error.type`; message states limit, used and exact wait; `retry-after`. Also 503 "over capacity" after ~30 s | high |
| Time until it recovers | RPM 2,051 ms, TPM 4,068 ms; daily continuous | next ok request after each 429; the stated wait was exact | high |

**The stand-in** speaks Groq's API, 429 format and SSE streaming over continuous token buckets. Its latency is sampled from fits to our probes: queue time and its episodes, token rate, a 4.2% trickle mode, and 503s after 30 s.

**It differs most from Groq in two ways:**
- It quotes rather than writes, so its rows measure the system, not answer quality.
- Its slowdown above 16 requests in flight is our guess; RPM 30 stops us measuring Groq there.

## 3. Results
**Table 2a: latency** (nearest-rank; first token over requests that got one; p99 from n<100 in every row)

| Scenario | Real or stand-in | Requests sent | Duration (s) | Offered rate (per min) | First token p50 / p95 / p99 (ms) | Total time p50 / p95 / p99 (ms) |
|---|---|---|---|---|---|---|
| Steady | real | 27 | 571 | 2.9 | 1,512 / 3,713 / 7,352 | 1,780 / 4,264 / 7,352 |
| Burst | real | 20 | 16 | 264.8 | 48 / 7,710 / 15,557 | 49 / 7,852 / 16,237 |
| Sustained overload | stand-in | 113 | 301 | 22.8 | 69 / 19,910 / 25,630 | 69 / 20,943 / 26,927 |
| Per-minute limit halved | stand-in | 27 | 282 | 5.7 | 582 / 11,344 / 14,242 | 695 / 11,474 / 14,534 |
| Daily budget exhausted | real | 20 | 328 | 3.7 | 1,637 / 3,374 / 5,713 | 1,808 / 3,487 / 5,995 |

**Table 2b: outcomes** ("rate-limited" = a limit forced a weaker answer; peak queue = most requests in the system at once; cost = upper bound at Groq list prices, the free tier bills $0)

| Scenario | Succeeded % | Rate-limited % | Timed out or given up % | Retries per request | Peak queue length | Tokens per answer | Cost per 1,000 answers |
|---|---|---|---|---|---|---|---|
| Steady | 88.9% | 11.1% | 0.0% | 0.00 | 2 | 2,836 | $0.546 |
| Burst | 30.0% | 70.0% | 0.0% | 0.00 | 6 | 492 | $0.065 |
| Sustained overload | 15.0% | 84.1% | 0.9% | 0.00 | 4 | 670 | $0.103 |
| Per-minute limit halved | 25.9% | 74.1% | 0.0% | 0.00 | 2 | 1,228 | $0.201 |
| Daily budget exhausted | 10.0% | 90.0% | 0.0% | 0.00 | 2 | 2,378 | $0.238 |

**Daily budget exhausted, for real:** 100 requests (`drain`) spent the 120b's day in 31 min, then 20 were sent with it gone: 18 shorter 20b answers, 2 from the cache. It ran at 17:10, when ~1.9K reserve tokens (under one answer) were still unreleased.

**Answer quality:** 20 questions written by the author, 5 of them unanswerable from the papers. The assistant graded them against the author's key with a two-level rubric frozen before the final run, and the author spot-checked 10 grades. In **Table 3** below, the ceiling for "correct" is 75%, and all 15 refusals were correct.

| Setting (piece size, pieces fetched, re-rank) | Questions asked | Correct and cited % | Wrong % | Correctly refused % | Tokens per answer | First token p50 / p95 (ms) | Total time p50 / p95 (ms) |
|---|---|---|---|---|---|---|---|
| 512, 4, no | 20 | 30.0% | 45.0% | 25.0% | 2,602 | 1,318 / 2,268 | 1,740 / 2,568 |
| **256, 8, no (chosen)** | 20 | 50.0% | 25.0% | 25.0% | 2,988 | 1,388 / 2,542 | 1,788 / 3,447 |
| 512, 4, yes | 20 | 35.0% | 40.0% | 25.0% | 2,431 | 5,583 / 14,883 | 5,865 / 15,229 |
| 256, 8, no: shipped pipeline | 20 | 50.0% | 25.0% | 25.0% | 3,065 | 1,399 / 3,265 | 1,721 / 3,535 |

**The number we couldn't have guessed [AUTHOR picks]:** the slowest real requests spent 91–95% of their time waiting in *our own* queue, not in Groq (the slowest: 34.7 s queued, estimated to within 0.1 s). *(Alternatives: the same setting scored 10/15 twice, but a different 10; naive retries doubled the median first token.)*

## 4. Where the time goes
The slowest real request (drain, `fd7fbdcbc76c`, a two-paper comparison) took **36,571 ms**.
- **Waiting: 34,883 ms (95.4%), all before the first token.**
  - 34,724 ms was in our admission queue for the 120b's per-minute budget. We estimated 34.7 s, and their patience was 38.2 s, so they were queued for a full answer, not downgraded.
  - 159 ms was in Groq's queue.
- **Working: 1,686 ms.** Retrieval and decide took 160 ms, prompt processing 925 ms and generation 600 ms, of which 472 ms came after the first token.
- **Retrying: 0 ms.**

**How we know:** per-phase trace timestamps plus Groq's own `usage.queue_time` (`eval/where_time_goes.py`). The slowest burst and steady requests were 90.6% and 94.5% waiting, also in our queue.

## 5. When a limit is hit
**Per-minute.** Admission is first in, first out against a ledger that mirrors Groq's buckets.
- **Delayed:** anyone whose estimated wait fits their patience.
- **A weaker answer at once:** everyone else, shorter 20b answers or quoted passages.
- **Dropped:** nobody. In the real burst, 14 of 20 got a weaker answer in about 50 ms and none gave up.

**Per-day.** A 20% reserve is released evenly until 18:00, then 20b answers labelled as shorter, then quoted passages.

## Retrieval: quality versus cost
**Chosen: 256-token pieces × 8, no re-rank.** It had the best correct-and-cited rate, no misread sentence past the gate (0 of 14), and a first token inside our 4 s target.

**Re-ranking lost twice:** the cross-encoder quadrupled first-token time for no gain, and ColBERT (local test) cut evidence retrieved from 88.1% to 79.1%.

**Near the limit:** a free heuristic router and one fewer piece; shorter answers use 6 pieces; quoted passages use none.

## How this library will mislead a naive retriever
1. **Every paper has a Table 1–3.** *"According to Table 1, what are the Depth Transformer's dimensions?"* Dense retrieval's top 8 has no Moshi Table 1. Our fix looks up a named table in the paper the question names.
2. **Introductions restate the method.** *"Describe the H-Net pipeline: encoder … decoder."* §1 outranks the method in §2.1.1.
3. **The papers contradict themselves.** *"What BLEU does the Transformer reach on WMT14 English-to-French?"* Both retrievers rank §6.1's "41.0" above Table 2's "41.8". Our drafter must report both and attribute each.

## 6. Retries
**Policy:**
- per-minute 429: retry only if the stated wait fits the person's patience;
- daily 429: never;
- 5xx: once, with jittered backoff;
- after the first token: never;
- at most 10% of requests retry per 60 s.

**Naive retries made things worse** (stand-in, same people, 25% of calls failing with 503). They reserved 81% more per-minute budget, caused 10 rate-limit rejections, and doubled the median first token (2,191 → 4,711 ms), for one extra full answer.

## 7. What the person sees
At 3pm with the daily budget gone:
- **While the 20b has budget:** a shorter answer, headed *"Shorter answer: today's budget for full answers is used up; the next full answer should be possible in about 21 minutes."* In the real run the stated wait was 4–8 minutes, because the budget refills continuously and had only just run out.
- **Once both budgets are gone:** in ~30 ms with no model call, *"Full answers are paused… the next full answer should be possible in about 22 minutes. These are the most relevant passages, quoted from the papers:"*, followed by 2–3 cited quotes or *"…do not appear to cover this question, so I won't guess."*

## 8. What I'm unsure about, what I left out, what I learned
**[AUTHOR]**
- **Least certain:** patience, assumed 12 s median with no data; if wrong, we downgrade too early or lose more people.
- **Not built:** HyDE, whose invented answers would pull unanswerable questions toward plausible passages; a runtime LLM judge, on token cost.
- **Learned:**

## Your feedback on Vaani
**[AUTHOR]**

## How I worked with the assistant
**[AUTHOR]** Conversation history: `[file]`. Cover: the 3 prompts that mattered most, one thing the assistant got wrong that you caught, one decision where you overruled it, and what is distinctly yours. *Candidates from the ADR:* overrules (keeping Q1–Q3; not recording the stopped first eval; no Groq judge; the two-level rubric); yours (the 20 questions and key, the spot-check, the reserve, option C).
