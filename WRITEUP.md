# The Relay: write-up

## 1. What I built and why
Seven steps: **admit** (answer cache, daily budget ledger) → **route** (a gpt-oss-20b **tool call**) → **retrieve** (local BM25 + dense search, with a **loop** that rewrites weak queries) → **decide** (a **branch**: answer, refuse, or decompose per paper) → **draft** (gpt-oss-120b, streamed) → **gate** → **log**.

**The key choice** was a deterministic citation gate on each streamed sentence: send it only when citations point to retrieved passages and its numbers appear there. It costs no tokens and preserves streaming; it can still miss a correct number from the wrong table row, which we measure. A daily ledger rations the 200K tokens/model: 120B first, then shorter 20B answers, then quoted passages. Under load, we trade answer quality for speed and availability, never by sending a cut-off sentence.

## 2. The real service and my stand-in
**Table 1:** limits found with 341 real probe requests, Groq's 429 bodies and headers, and the console. A final-day recheck matched.

| Limit | Value found: observed (*Groq states*) | How found | Confidence |
|---|---|---|---|
| Requests per minute | 30 (*Groq states 30*) | 30 ok in a burst, then a 429 at 13,009 ms | high |
| Tokens per minute | ≥5,755 (*Groq states 8,000*) | used in the 60 s before a 429; a lower bound, as `max_tokens` is reserved | high |
| Requests per day | 1,000 (*Groq states 1,000*) | 1,000 ok in 38.3 min on a spare model, then a daily 429 | high |
| Tokens per day | 201,298 (*Groq states 200,000*) | used on a spare model before its daily 429; refill 2.31 tokens/s while idle | high |
| Requests in flight at once | no limit at 16 | 16 overlapping requests, all ok; RPM 30 blocks a bigger test | low |
| What it returns at a limit | HTTP 429 in 99 ms (p50) | `error.type`; message gives the exact wait; `retry-after`; 503s after ~30 s | high |
| Time until it recovers | 2,051 ms (RPM), 4,068 ms (TPM), 90 s (RPD) | next ok after each 429; the stated wait was exact | high |

**Evidence sources:** Observed values are recomputed from `results.csv`; "Groq states" values come from 429 bodies, `x-ratelimit-*` headers (`results/probe_raw.jsonl`) and console. In the 38-min RPD run (one request every ~2.2 s), no request credit returned; continuous refill would predict ~26. One request succeeded after a 90-s idle gap. We infer—but cannot establish—that credits return only after idle gaps of ~86.4 s.

**The stand-in** speaks Groq's API, 429 bodies and SSE streaming over continuous token buckets. Its latency is sampled from fits to our probes: queue time with its correlated episodes, token rate, a 4.2% trickle mode, and 503s after 30 s. **Least like Groq:** it quotes instead of writing, so its rows measure the system, not answer quality; and its slowdown above 16 requests in flight is a guess, because RPM 30 stopped us measuring Groq there.

## 3. Results
**Table 2a: latency** (nearest-rank percentiles; first token over requests that got one; p99 from fewer than 100 requests in every row)

| Scenario | Real or stand-in | Requests sent | Duration (s) | Offered rate (per min) | First token p50 / p95 / p99 (ms) | Total time p50 / p95 / p99 (ms) |
|---|---|---|---|---|---|---|
| Steady | real | 27 | 571 | 2.9 | 1,512 / 3,713 / 7,352 | 1,780 / 4,264 / 7,352 |
| Burst | real | 20 | 16 | 264.8 | 48 / 7,710 / 15,557 | 49 / 7,852 / 16,237 |
| Sustained overload | stand-in | 113 | 301 | 22.8 | 69 / 19,910 / 25,630 | 69 / 20,943 / 26,927 |
| Per-minute limit halved | stand-in | 27 | 282 | 5.7 | 582 / 11,344 / 14,242 | 695 / 11,474 / 14,534 |
| Daily budget exhausted | real | 20 | 328 | 3.7 | 1,637 / 3,374 / 5,713 | 1,808 / 3,487 / 5,995 |

**Table 2b: outcomes** (rate-limited = a limit forced a weaker answer; peak queue = most requests in the system at once; cost = upper bound at Groq list prices, the free tier bills $0)

| Scenario | Succeeded % | Rate-limited % | Timed out or given up % | Retries per request | Peak queue length | Tokens per answer | Cost per 1,000 answers |
|---|---|---|---|---|---|---|---|
| Steady | 88.9% | 11.1% | 0.0% | 0.00 | 2 | 2,836 | $0.546 |
| Burst | 30.0% | 70.0% | 0.0% | 0.00 | 6 | 492 | $0.065 |
| Sustained overload | 15.0% | 84.1% | 0.9% | 0.00 | 4 | 670 | $0.103 |
| Per-minute limit halved | 25.9% | 74.1% | 0.0% | 0.00 | 2 | 1,228 | $0.201 |
| Daily budget exhausted | 10.0% | 90.0% | 0.0% | 0.00 | 2 | 2,378 | $0.238 |

**Daily budget exhausted** ran for real: 100 `drain` requests spent the 120b's day in 31 min, then 20 more were sent. It started at 17:10, when ~1.9K reserve tokens (under one answer) were still unreleased.

**Answer quality:** 20 questions by the author (5 unanswerable). The assistant graded against the author's key with a rubric frozen before the final run; the author re-graded 10 answers independently and cross-checked many more by hand. Shipped pipeline: 50.0% correct and cited, 25.0% wrong, 25.0% correctly refused, 3,065 tokens per answer (Table 3).

**The number we couldn't have guessed:** The surprising number: 95.4% of the slowest real request's 36,571 ms was waiting, almost entirely in our admission queue; our 34.7-s estimate was within 0.1 s.

## 4. Where the time goes
The slowest real request (drain, `fd7fbdcbc76c`, a two-paper comparison) took **36,571 ms**:
- **Waiting: 34,883 ms (95.4%), all before the first token.** 34,724 ms in our admission queue for the 120b's per-minute budget (we estimated 34.7 s; their patience was 38.2 s, so they were queued for a full answer, not downgraded) and 159 ms in Groq's queue.
- **Working: 1,686 ms.** Retrieval and decide 160 ms, prompt processing 925 ms, generation 600 ms (472 ms after the first token).
- **Retrying: 0 ms.**

**How we know:** per-phase trace timestamps plus Groq's own `usage.queue_time` (`eval/where_time_goes.py`). The slowest burst and steady requests were 90.6% and 94.5% waiting, also in our queue.

## 5. When a limit is hit
**Per-minute.** Admission is first in, first out against a ledger that mirrors Groq's buckets.
- *Delayed:* anyone whose estimated wait fits their patience, less a 2 s margin (one person waited 34.7 s).
- *A weaker answer at once:* everyone else: a shorter 20b answer, or quoted passages if the 20b is busy (14 of 20 in the real burst, ~50 ms).
- *Dropped:* nobody is refused, but people can give up: none in the real runs, 1 of 113 (0.9%) under stand-in overload.

**Per-day.** The 120b's 200K tokens are the scarce resource.
- *Delayed:* nobody, since a daily wait outlasts anyone's patience.
- *A weaker answer:* 20% of the day is released evenly until 18:00, so a morning burst cannot spend the evening's full answers. After that, a 20b answer labelled as shorter, with the expected wait; when the 20b's day is gone too, quoted passages with no model call.
- *Dropped:* nobody (real: of the 20 sent after the drain, 18 got shorter answers and 2 cached ones).

**What "weaker" cost** (167 real load answers, graded like Table 3): on answerable questions the 20b was correct as often as the 120b (20/30 against 48/72, 66.7% each), but it failed the H-Net pipeline question 3 times out of 3, which the 120b passed 3 out of 3. Quoted passages answered none of 8 answerable questions. Every unanswerable question was refused.

## Retrieval: quality versus cost
**Table 3: retrieval settings** (20 questions each; the ceiling for "correct" is 75%; all 15 refusals were correct)

| Setting (piece size, pieces fetched, re-rank) | Questions asked | Correct and cited % | Wrong % | Correctly refused % | Tokens per answer | First token p50 / p95 (ms) | Total time p50 / p95 (ms) |
|---|---|---|---|---|---|---|---|
| 512, 4, no | 20 | 30.0% | 45.0% | 25.0% | 2,602 | 1,318 / 2,268 | 1,740 / 2,568 |
| **256, 8, no (chosen)** | 20 | 50.0% | 25.0% | 25.0% | 2,988 | 1,388 / 2,542 | 1,788 / 3,447 |
| 512, 4, yes | 20 | 35.0% | 40.0% | 25.0% | 2,431 | 5,583 / 14,883 | 5,865 / 15,229 |
| 256, 8, no: shipped pipeline | 20 | 50.0% | 25.0% | 25.0% | 3,065 | 1,399 / 3,265 | 1,721 / 3,535 |

**Chosen: 256-token pieces × 8, no re-rank.** It had the best correct-and-cited rate, let no misread sentence past the gate (0 of 14), and kept the first token inside our 4 s target. **Re-ranking lost twice:** the cross-encoder quadrupled first-token time for no gain, and ColBERT (local test) cut evidence retrieved from 88.1% to 79.1%. **Near the limit:** the free heuristic router and one fewer piece; shorter answers use 6 pieces, quoted passages none.

## How this library will mislead a naive retriever
1. **Every paper has a Table 1–3.** *"According to Table 1, what are the Depth Transformer's dimensions?"* Dense retrieval's top 8 has no Moshi Table 1. Our fix looks up a named table in the paper the question names.
2. **Introductions restate the method.** *"Describe the H-Net pipeline: encoder … decoder."* §1 outranks the method in §2.1.1.
3. **The papers contradict themselves.** *"What BLEU does the Transformer reach on WMT14 English-to-French?"* Both retrievers rank §6.1's "41.0" above Table 2's "41.8". Our drafter must report both and attribute each.

## 6. Retries
**Policy:** on a per-minute 429, retry only if the stated wait fits the person's patience; on a daily 429, never; on a 5xx, once, with jittered backoff; never after the first token; at most 10% of requests retry per 60 s. **Retrying made things worse** (stand-in, same people, 25% of calls failing with 503): naive retries reserved 81% more per-minute budget, caused 10 rate-limit rejections, and doubled the median first token (2,191 → 4,711 ms), for one extra full answer.

## 7. What the person sees
At 3pm with the daily budget gone, while the 20b still has budget: a shorter answer headed *"Shorter answer: today's budget for full answers is used up; the next full answer should be possible in about 21 minutes."* (Real run: 4–8 minutes, as the budget refills continuously.) Once both budgets are gone, in ~30 ms with no model call: *"Full answers are paused… the next full answer should be possible in about 22 minutes. These are the most relevant passages, quoted from the papers:"*, then 2–3 cited quotes, or *"…do not appear to cover this question, so I won't guess."*

## 8. What I'm unsure about, what I left out, what I learned
 **Least certain:** patience, assumed 12 s median with no data as it wasn't measured using actual user behaviour; if wrong, we downgrade too early or lose more people. **Not built:** HyDE, because hypothetical generated answers could pull retrieval towards unsupported passages, and a runtime LLM judge on every request because of its additional token and latency cost. **Learned:** I learned that a daily request quota cannot always be modelled as a continuously replenishing bucket. During our 38.3-minute Qwen experiment, no requests were credited back under steady traffic, but one succeeded after a 90-second idle gap. This was an observed behaviour, not proof of Groq's internal algorithm.

## Your feedback on Vaani
Vaani stood out for its end to end voice agent approach, especially interruptions, language switching and workflow integrations. I couldn't distinguish core features from newer launches. I'd clarify feature availability and publish language benchmarks and real-world latency.


## How I worked with the assistant
Conversation history: Attached in repo. 
Three prompts had the biggest impact on the project. First, I asked it to recheck the entire assignment brief against the implementation and identify anything remaining. This exposed gaps in per day degradation handling, report requirements and submission readiness. Second, I asked, "Right now, you're grading it, but what can be used as grading in production?" This shifted the discussion beyond offline evaluation towards practical monitoring signals such as citation-gate failures, refusal rates and user feedback. Third, I asked it to grade the 172 real load answers and investigate Groq's daily request limit using Qwen. This helped measure the quality trade-off between models under actual load and revealed unexpected quota behaviour.

One mistake I caught was that 172 real user-facing load requests had been left without an answer_grade in results.csv, even though they could be evaluated against our question set. I noticed the empty fields, questioned the omission and had the responses extracted from traces and graded against the answer key.

I overruled the assistant's initial approach of running an LLM-based judge on every request. Instead, I chose the deterministic sentence level citation gate to avoid additional model calls and token consumption while preserving streaming. Its semantic limitations were documented and measured separately.

What was distinctly mine was the system's engineering trade offs: the 20% daily reserve, fallback priorities, Option C verification strategy, evaluation rubric and manual validation. I directed the implementation, reviewed the questions against the papers and made the final decisions based on experimental evidence.
