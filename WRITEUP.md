# The Relay: write-up

> **DRAFT, 2026-11-11.** Numbers are recomputed from `results.csv` by `python -m tools.tables --table 1|2|3`. **TBD** marks what tomorrow's real runs fill in. **[AUTHOR]** marks text that must be the author's own. Target: 2 pages; cut to fit once the tables are final.

## 1. What I built and why

A seven-step Python pipeline. Every step is a stage in `relay/pipeline.py`:
1. **Admit:** answer cache and budget ledger.
2. **Route:** a gpt-oss-20b *tool call*, `search_papers` or `refuse_out_of_scope`.
3. **Retrieve:** local BM25 + dense hybrid search, with a *loop* that reformulates when confidence is low.
4. **Decide:** a *branch* that answers, refuses, or decomposes per paper.
5. **Draft:** gpt-oss-120b, streamed.
6. **Gate:** checks each sentence's citations.
7. **Log:** one row per request plus a trace.

**The decision that shaped it most: a deterministic per-sentence citation gate.**
- A sentence reaches the person only if every citation names a passage we retrieved, and every number appears in a cited passage (or is a stated difference of two that do).
- The gate costs no tokens and streams sentence by sentence.
- It catches made-up citations and made-up numbers. It cannot catch a real number attached to the wrong row; we measured that gap rather than hide it.
- **Second shaping decision:** budget the *day*, not the minute. Groq's daily token budget (200K per model) binds long before its request limits. A ledger rations the 120b and falls back to shorter 20b answers, then to quoted passages.
- **Goals given up:** under overload most people get a weaker answer quickly (*fast* and *nobody turned away* over *good*). A cut-off answer never shows its half-sentence (*good* over *fast*).

## 2. The real service and my stand-in

**Limits** (Table 1) came from 339 deliberate real probe requests (one row each in `results.csv`), Groq's own 429 bodies and headers, and the console:

| Limit | Value found | How found | Confidence |
|---|---|---|---|
| Requests per minute | 30 | 30 ok in a burst, then rate-limited 13,009 ms in; 429 body; console | high |
| Tokens per minute | 8,000 | 5,755 tokens used in the 60 s before the first 429 (a lower bound: `max_tokens` is reserved up front); 429 body; console | high |
| Requests per day | 1,000 | `x-ratelimit-limit-requests` header; console; not driven to the limit | medium |
| Tokens per day | 200,000 per model | 72 requests used 201,298 tokens before the daily 429; refill fitted at 2.31 tokens/s; console | high |
| Requests in flight at once | no limit hit at 16 | 16 overlapping requests all ok; RPM 30 stops a larger test | low |
| What it returns at a limit | HTTP 429 in 99 ms (p50, n=24) | `error.type` requests/tokens; message gives limit, used, requested and the exact wait; `retry-after`. Separately, 503 "over capacity" after ~30 s | high |
| Time until it recovers | RPM 2,051 ms, TPM 4,068 ms; daily: continuous refill, no reset | next ok request after each 429; the daily 429's stated wait was exact | high |

**Stand-in** (`standin/server.py`):
- Groq's API shape, its 429 format and headers, continuous token buckets and SSE streaming.
- Latency sampled from fits to the probe data: queue time, queue "episodes", per-token rate, a 4.2% trickle mode, and 503s after 30 s.

**Where it differs most from Groq:**
- Its answers are quoted sentences, so it cannot refuse or misread a paper: stand-in rows measure the system, not answer quality.
- Its slowdown above 16 requests in flight is a curve we chose, because RPM 30 stops us measuring Groq there.
- It charges per-minute tokens for calls that then fail with a 503. That is unverified on Groq.

## 3. Results

**Table 2a: latency** (nearest-rank percentiles; n per cell in `tools.tables`)

| Scenario | Real or stand-in | Requests sent | Duration (s) | Offered rate (per min) | First token p50 / p95 / p99 (ms) | Total time p50 / p95 / p99 (ms) |
|---|---|---|---|---|---|---|
| Steady | real | TBD | TBD | TBD | TBD | TBD |
| Burst | real | TBD | TBD | TBD | TBD | TBD |
| Sustained overload | stand-in | 113 | 301 | 22.8 | 69 / 19,910 / 25,630 | 69 / 20,943 / 26,927 |
| Per-minute limit halved | stand-in | 27 | 282 | 5.7 | 582 / 11,344 / 14,242 | 695 / 11,474 / 14,534 |
| Daily budget exhausted | real | TBD | TBD | TBD | TBD | TBD |

**Table 2b: outcomes**

| Scenario | Succeeded % | Rate-limited % | Timed out or given up % | Retries per request | Peak queue length | Tokens per answer | Cost per 1,000 answers |
|---|---|---|---|---|---|---|---|
| Steady | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Burst | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Sustained overload | 15.0% | 84.1% | 0.9% | 0.00 | 4 | 670 | $0.103 |
| Per-minute limit halved | 25.9% | 74.1% | 0.0% | 0.00 | 2 | 1,228 | $0.201 |
| Daily budget exhausted | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

How to read Table 2:
- **"Rate-limited"** means a limit forced a weaker answer (shorter, or quoted passages) or no answer.
- **"Peak queue length"** is the most requests in the system at once.
- **Cost** is an upper bound at Groq list prices (the free tier bills $0): Full rows at the 120b price, others at the 20b price.
- The stand-in "daily budget exhausted" run is kept as a fallback row: 11 of 19 got Lean answers, 8 quoted passages.

**Answer quality: 20 questions written by the author,** 15 answerable and 5 the papers don't answer. Graded by the assistant against the author's key with a two-level rubric, frozen before the final run. The author spot-checked 10 grades.

**Table 3**

| Setting (piece size, pieces fetched, re-rank) | Questions asked | Correct and cited % | Wrong % | Correctly refused % | Tokens per answer | First token p50 / p95 (ms) | Total time p50 / p95 (ms) |
|---|---|---|---|---|---|---|---|
| 512, 4, no | 20 | 30.0% | 45.0% | 25.0% | 2,602 | 1,318 / 2,268 | 1,740 / 2,568 |
| **256, 8, no (chosen)** | 20 | 50.0% | 25.0% | 25.0% | 2,988 | 1,388 / 2,542 | 1,788 / 3,447 |
| 512, 4, yes | 20 | 35.0% | 40.0% | 25.0% | 2,431 | 5,583 / 14,883 | 5,865 / 15,229 |
| 256, 8, no, shipped pipeline | 20 | 50.0% | 25.0% | 25.0% | 3,065 | 1,399 / 3,265 | 1,721 / 3,535 |

- Every unanswerable question was refused correctly, under every setting.
- The maximum "correct" here is 75%, because 5 of the 20 questions should be refused.

**A number we could not have guessed [AUTHOR picks one]:**
- (a) The same setting on the same 15 questions scored 10/15 twice, but a *different* 10: run-to-run variance is about ±3 questions.
- (b) Naive retries more than doubled the median first token (2,191 → 4,711 ms).
- (c) 6.5% of prompt tokens were served from Groq's prefix cache, for free.

## 4. Where the time goes

**TBD** from the slowest real request in tomorrow's runs. The trace (`results/trace.jsonl`) times each phase: queued and admission wait (waiting); router, retrieval and drafting (working, with Groq's `usage.queue_time` separating Groq-side waiting); retry waits (retrying). It also records first byte versus first gated sentence, so we can say whether the delay came before or after the first token.

## 5. When a limit is hit

**Per-minute limit.** Admission is first in, first out per model, against a ledger that mirrors Groq's buckets and corrects itself from every response header.
- **Who is delayed:** anyone whose estimated wait, including requests queued ahead, fits within their patience.
- **Who gets a weaker answer:** anyone whose wait wouldn't fit. They are moved at once to a shorter answer from the 20b, or to quoted passages, rather than kept waiting.
- **Who is dropped:** nobody is refused outright. In stand-in overload, 84% got a weaker answer in about 70 ms and 0.9% gave up.
- Under a burst the router steps aside for a free heuristic, to keep the 20b's minute budget for shorter answers.

**Per-day limit.** A 20% reserve of the 120b's day is released evenly until 18:00, so a morning burst can't spend the afternoon. When the 120b's day is gone, answers come from the 20b, labelled as shorter; when the 20b's day is also low, quoted passages. Evaluation runs never downgrade.

## Retrieval: quality versus cost

Table 3. **We chose 256-token pieces × 8, no re-rank:**
- It had the highest correct-and-cited rate (50.0%).
- It is the only setting with no misread sentence getting past the gate (0 of 14 answered).
- Its first token stays far inside our 4 s target.

**Re-ranking failed twice.** The cross-encoder quadrupled first-token time (p95 14.9 s) without gaining quality. A ColBERT late-interaction re-ranker, tested locally, cut the share of required evidence retrieved from 88.1% to 79.1%.

**What retrieval does when the limit is close:**
- Under per-minute pressure (L1): the free heuristic router and one fewer piece.
- Shorter answers send 6 pieces, not 8.
- Quoted passages send none to a model.

## How this library will mislead a naive retriever

(Plain BM25 and dense-only top 8: `results/naive_retriever_2026-11-11.txt`.)
1. **Every paper has a Table 1–3.** *"According to Table 1, what are the Depth Transformer's dimensions?"* Dense retrieval's top 8 contains no Moshi Table 1. In our own eval, H-Net's Table 1 was retrieved for AIAYN's "Table 3". Fix: look up a named table in the paper the question is about.
2. **Introductions restate the method in the same words.** *"Describe the H-Net pipeline: encoder, dynamic chunking, main network, dechunking, decoder."* The introduction (§1) outranks the method (§2.1.1). Token-level ColBERT made it total: all 8 passages came from the overview.
3. **The papers contradict themselves.** *"What BLEU does the Transformer reach on WMT14 English-to-French?"* Both retrievers rank §6.1's prose ("41.0") above the Abstract and Table 2 ("41.8"). Likewise H-Net §3.1 says "+2.6" where Table 2 gives 2.7. Our drafter is told to report both and attribute each.

*(A fourth: AIAYN's arXiv source ships `sqrt_d_trick.tex`, "Justification of the Scaling Factor", which is commented out of the paper. Indexing every `.tex` file would cite text the paper never published.)*

## 6. Retries

**Policy:**
- Per-minute 429: wait the stated time only if it fits within the person's patience.
- Daily 429: never retry.
- 5xx: one retry with full-jitter backoff.
- No first token: one retry if there's time.
- After the first token: never retry.
- Overall, at most 10% of requests may retry in any 60 s.

**Retrying made things worse** (stand-in, same people and questions, 25% of calls hitting the measured 503 "over capacity"). Naive immediate retries:
- reserved 81% more per-minute budget (68,556 vs 37,892 tokens);
- caused 10 rate-limit rejections where the policy caused none;
- more than doubled the median first token (2,191 → 4,711 ms);
- for one extra full answer.

## 7. What the person sees

At 3pm with the daily budget gone:
- **While the 20b still has budget**, the person gets a shorter written answer in under a second, headed: *"Shorter answer: today's budget for full answers is used up; the next full answer should be possible in about 21 minutes."*
- **Once both are gone**, in about 30 ms, with no model call: *"Full answers are paused: today's model budget is used up. The budget refills continuously; the next full answer should be possible in about 22 minutes. These are the most relevant passages, quoted from the papers:"* That is followed by 2–3 cited quotes, or, if retrieval finds nothing relevant, *"The three papers… do not appear to cover this question, so I won't guess."*

Groq's daily budget refills continuously (we measured it), so the message gives a wait, not "tomorrow".

## 8. What I'm unsure about, what I left out, what I learned

- **Least certain assumption [AUTHOR to confirm]:** how long people wait. We used a lognormal with a 12 s median and no data behind it. If people wait longer, we move them to weaker answers too early. If they wait less, more give up before even the quoted passages arrive.
- **Deliberately not built [AUTHOR to confirm]:**
  - HyDE, because in a narrow library its invented "hypothetical answer" would pull unanswerable questions toward plausible passages, and we refuse 15/15 today.
  - A runtime LLM judge (Option B), on token cost; our grading measured its blind spot instead.
- **Learned [AUTHOR, in your own words]:** e.g. Groq's daily limit is a continuously refilling bucket, not a midnight reset, and `max_tokens` is reserved against the per-minute budget up front.

## Your feedback on Vaani

**[AUTHOR]** What stood out, what confused you, one thing you would change (https://www.vaani.dev/).

## How I worked with the assistant

**[AUTHOR]** Raw conversation history attached (`[file name]`). Half a page:
- **Three prompts that mattered most,** and why.
- **One thing it got wrong that you caught,** and how.
- **One decision where you overruled it.**
- **What is distinctly yours.**

*Candidates recorded in the ADR, for you to choose from, not text to paste:*
- *Overrules:* keeping Q1–Q3 against the assistant's advice; not recording the first stopped eval run; ruling out a Groq judge ("you be the judge"); the 20% reserve released to 18:00; the two-level rubric with your two edits.
- *Yours:* the 20 questions and the answer key verified against the PDFs; the strict-then-two-level grading; the 10-grade spot-check; the 20b/120b budget split; option C for eval_v2.
