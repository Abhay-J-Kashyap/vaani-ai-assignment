# ADR-0001: Pipeline architecture for The Relay

**Status:** Accepted (Option C), with a measured fallback to Option B. See "Decision record".
**Date:** 2026-11-09
**Deciders:** Abhay K Kashyap (author, owns every call below)
**Source of truth:** `The_Relay_Assignment_Brief.pdf`. Where this ADR and the brief disagree, the brief wins.

---

## Context

We are building a small Python assistant. It answers engineers' questions using only three arXiv papers at pinned versions:

- 1706.03762v7, Attention Is All You Need
- 2410.00037v1, Moshi
- 2507.07955v2, Dynamic Chunking / H-Net

It runs on Groq's free tier, and it has to keep working when load arrives in bursts.

### Hard requirements from the brief

| # | Requirement | Where this ADR satisfies it |
|---|---|---|
| R1 | ≥4 steps, ≥1 tool call, ≥1 branch or loop, one step retrieves from the library | Pipeline stages 1–6; router tool call; decide branch; capped retrieval loop |
| R2 | Every answer cites where it came from; says so when the papers don't cover it | Chunk IDs carry the section path; citation gate; refuse branch |
| R3 | Real Groq free-tier model, own key, limits discovered from docs **and** from responses | Probe plan (Action Items §A) |
| R4 | Stand-in service that can be slow, fail, refuse at a limit, slow down under load, and stream | `standin/` service, calibrated from probe data |
| R5 | ≥3 load shapes. Table 2a names 5: steady, burst, sustained overload, per-minute limit halved, daily budget exhausted | Load generator scenarios S1–S5 |
| R6 | `results.csv` with the exact columns; every table figure recomputable from it | Telemetry design |
| R7 | ≥3 retrieval settings (piece size, pieces fetched, re-rank) with quality, tokens, TTFT | Retrieval settings RS1–RS4 |
| R8 | Quality on ≥15 self-written questions; tokens per answer | Gold set G (≥20 questions) |
| R9 | Split one slow request into waiting / working / retrying, before or after first token | Per-phase trace events |
| R10 | Behaviour at the per-minute and per-day limits; retry policy plus one case where retrying hurt; the reply a person sees at 3pm with the daily budget gone | Degradation ladder; retry policy; extractive fallback |
| R11 | Write-up of ≤2 pages with fixed headings; Vaani feedback; raw conversation history | Action Items §E |
| R12 | No key in the submission; no extra accounts or tricks to get around limits | `.env` + `.gitignore`; one account |

### Forces in tension

The company wants four things: fast, good, cheap, and nobody turned away. The brief says it can't have all four. On top of that, it must **never send a wrong answer just to be fast**. So correctness is not something we trade away. We trade the other three against each other, inside a hard correctness floor.

- **Free-tier limits.** Groq limits RPM, TPM, RPD and TPD per model and per organisation. Each user request makes 2–6 model calls, so every limit bites 2–6× sooner than a per-request count suggests. TPM is expected to bind before RPM, because retrieved context is large. This is a hypothesis; the probes must confirm it.
- **Verify before sending vs streaming.** If the full answer is checked before anything is sent, TTFT includes the whole generation, and streaming gains nothing.
- **Users give up.** A request whose user has left should stop spending tokens.

### Definition of "fast" (provisional; revisit after probe Phase 2)

| Metric | Target at steady load | Notes |
|---|---|---|
| First token p95 | ≤ 4,000 ms (revised 2026-11-10 from 3,000) | "First token" means the first token of answer text. Status events don't count. |
| Total time p95 | ≤ 8,000 ms (revised 2026-11-10 from 15,000) | |
| Client patience (give-up) | Drawn per simulated user, median 20 s, range 8–45 s | The brief says users give up "sooner than you expect", so we test short patience too |
| Server deadline | 25 s per request | Past this we stop and send an honest fallback |

---

## Decision

Adopt **Option C: a staged pipeline with an admission controller, a model-driven tool call that routes the question, a capped retrieval loop, a decide branch, a streamed per-sentence citation gate, and a zero-token extractive fallback.**

```
client ──► [0 Admission]  token-aware budget check · deadline · priority queue · shed when estimated wait > deadline
              │
              ▼
         [1 Route]  small model, Groq tool calling: search_papers(query, paper?) | refuse_out_of_scope(reason)
              │      (degraded: heuristic router, no LLM call)
              ▼
         [2 Retrieve]  hybrid BM25 + local embeddings ► optional local cross-encoder re-rank
              │  ▲
              │  └── loop: top score < τ  → reformulate and retry (max 2 iterations)
              ▼
         [3 Decide]  branch: ANSWER | REFUSE (not in papers) | DECOMPOSE (≤3 sub-questions → back to 2)
              │
              ▼
         [4 Draft]  large model, streamed, must cite chunk IDs like [AIAYN §3.2.2]
              │
              ▼
         [5 Gate]  per sentence: deterministic check_citations tool
              │      pass → send the sentence · fail → drop it and regenerate once, else refuse
              ▼
         [6 Emit + log]  one row to results.csv, phase events to trace.jsonl
```

**The decision that shapes the design most (heading 1 of the write-up):** the per-sentence citation gate in stage 5. The draft is buffered one sentence at a time. Each sentence goes out only after a deterministic check passes:

- it carries at least one citation,
- every cited chunk ID is in the retrieved set, and
- every number in the sentence appears in a cited chunk.

This keeps TTFT close to the time to the first finished sentence, while making it impossible to send an uncited sentence or a citation that was never retrieved.

**The cost, stated plainly:** the gate catches made-up citations and made-up numbers. It does **not** catch a correctly cited sentence that misreads its source. We accept that gap at runtime and measure it offline with the gold set.

### Decision record (2026-11-09, author's call)

The author accepted Option C: every sentence is checked for a citation before it goes out, mainly to keep time to first token low. Option B (buffer the whole answer, one LLM judge, then send) is the planned fallback if C doesn't hold up.

**Fallback triggers.** These are provisional thresholds, set before any data, so the switch is decided by measurement and not by feel. Any one of them, measured on the gold set with the chosen retrieval setting, sends us to Option B:

| Trigger | Threshold | Why it matters |
|---|---|---|
| Wrong %: answers that pass the gate but misread their source | > 10% of answered questions | The gate checks citations, not meaning. This is its known blind spot. |
| Gate-fail rate: sentences dropped or regenerated | > 25% of sentences | Regenerating erases the first-token advantage, so B would be as fast and safer |
| First-token p50 under C | Not at least 30% lower than under B, on the same questions | C exists to win on first-token time. If it doesn't, its complexity isn't worth it. |
| Embarrassing output | Any sentence that passed the gate but is cut mid-thought, or a partial answer that reads as a full one | The brief: never send a wrong or embarrassing answer |

**How we find out:** in the retrieval runs, answer the same gold-set questions under both C and B and compare. This costs one extra judge call per question, so plan it into the daily request budget. The comparison is itself a number for the write-up that couldn't have been guessed.

---

## Options Considered

### Option A: the original plan (linear pipeline, section-level parent chunks, RAGAS check, then stream)

`request manager → understand → retrieve (parent = section, child = subsection) → generate → RAGAS faithfulness / quality → stream`

| Dimension | Assessment |
|---|---|
| Complexity | Medium |
| Cost (tokens / calls) | **High.** RAGAS faithfulness splits an answer into claims and verifies each with another LLM call, adding ~3–10 calls per request. Section-level parents in Moshi and H-Net can run to thousands of tokens each. |
| Behaviour under limits | **Poor.** It multiplies RPM and TPM use exactly when budget is scarce. |
| Latency | TTFT ≈ full generation + full evaluation. Streaming at the end adds nothing. |
| Brief compliance | **Fails R1** (no branch or loop) and **R2** (no refusal path). Doesn't cover R4–R6 or R9. RAGAS scores don't map to the required `answer_grade` labels. |

**Pros:** easy to explain. RAGAS is a recognised quality metric.
**Cons:** see the table. The most serious one: the expensive check sits on the hot path at the worst possible time.

### Option B: buffer the whole answer, verify with one LLM judge, then send

| Dimension | Assessment |
|---|---|
| Complexity | Low–Medium |
| Cost | Medium: +1 judge call per request |
| Behaviour under limits | Medium: the judge competes with drafting for TPM |
| Latency | TTFT = draft time + judge time, typically several seconds |
| Brief compliance | Can be made to comply, but the "first token" numbers will look bad and streaming has no purpose |

**Pros:** the strongest runtime check of meaning; simple control flow.
**Cons:** the worst TTFT; the judge can itself be wrong or rate-limited; the judge fails exactly when we need it.

### Option C (chosen): staged pipeline with per-sentence deterministic gate, conditional judge, and extractive fallback

| Dimension | Assessment |
|---|---|
| Complexity | Medium–High: admission controller, gate, fallback ladder |
| Cost | Low: the gate costs no tokens. A judge call is made only when the gate is inconclusive, e.g. a sentence with no number and a weak overlap with its cited chunk. |
| Behaviour under limits | Good: a clear degradation ladder, ending in a zero-token answer |
| Latency | TTFT ≈ route + retrieve + time to the first finished sentence |
| Brief compliance | Covers R1–R12 by design |

**Pros:** correctness floor enforced without tokens; a clear story for each degradation step; the "3pm" reply never needs Groq.
**Cons:** checks citations, not meaning; sentence splitting has edge cases (equations, "e.g."); more code to defend in the live session.

### Option D: no gate, stream the draft immediately (baseline only)

Rejected for production because it breaks "never send a wrong answer". **Kept as a measured baseline** in the retrieval runs, to show what the gate costs and what it catches. That gives a number nobody could guess without running it.

---

## Trade-off Analysis

| Goal given up | For | Where and how much (fill in from measurements) |
|---|---|---|
| Fast (some users) | Good | The gate holds each sentence back by its check time (~ms) plus regeneration time when a check fails. Measure the gate-fail rate. |
| Nobody turned away | Fast, for those already admitted | We shed new arrivals when estimated wait > deadline, and send an honest "busy, retry in N s" right away instead of a 25 s wait that times out |
| Good (some users) | Nobody turned away | When the daily budget is low, the answer is a set of cited extracted passages, not a written-up reply. Weaker, but never wrong. |
| Cheap | Good | Optional re-rank and parent expansion cost tokens or CPU. Kept only if Table 3 shows the quality gain. |

### Sub-decisions

| Area | Decision | Rejected alternative and why |
|---|---|---|
| **Corpus source** | arXiv LaTeX source (`/e-print/<id>v<N>`), parsed for `\section`/`\subsection`, table captions and the appendix flag | PDF text extraction: loses section boundaries, garbles tables and equations |
| **Chunking** | Index **children** only: paragraph- or subsection-sized pieces carrying their section path. Expanding to the parent is an option, capped at 1,500 tokens. Long sections are split. | Uncapped section-level parents (original plan). In Moshi and H-Net, two or three of them can use a whole minute's TPM in one request. |
| **Retrieval** | Hybrid BM25 + small local embedding model; optional local cross-encoder re-rank. References or bibliography sections are excluded or down-weighted. | Dense only: misses exact tokens such as `d_model`, "Mimi" and table labels. Groq embeddings: as far as we know, Groq serves no embedding models (verify). |
| **Tool call** | The routing step is a Groq tool-calling request: the model chooses between `search_papers` and `refuse_out_of_scope`. `check_citations` is a code tool used by the gate. | Retrieval as a plain function call: weak evidence for "the model uses a tool". |
| **Model routing** | Small model for routing; large model for drafting. Groq limits are per model, so this spreads load across separate quotas. **Stated openly in the write-up.** | One model for everything: simpler, but runs into its own limits sooner. Model names are chosen after checking the console (e.g. llama-3.1-8b-instant / llama-3.3-70b-versatile, if still offered). |
| **Rate limiting** | Client-side token buckets per model for RPM, TPM and daily budget. Each bucket starts at the documented values and is **corrected from live `x-ratelimit-*` headers** on every response. A 429 with `retry-after` freezes that bucket. | Fixed limits in config: the brief warns limits "can change without notice". |
| **Timeouts** | Two per call: time to first token (catches hung calls) and total time (catches slow successes). | One total timeout: can't tell "not started" from "slow but working". |
| **Cancellation** | When the client gives up, the request and its in-flight Groq stream are cancelled. | Letting it finish: burns tokens nobody reads. |
| **Quality eval** | Hand-written gold set (≥20 questions; ≥5 unanswerable, ≥3 cross-paper, ≥3 table or number lookups), each with an expected answer, an expected section, and an answerable flag. Grades map directly to `answer_grade`. RAGAS is offline and optional at most. | RAGAS on the hot path (see Option A). |

### Degradation ladder (answers heading 5 and heading 7)

| Level | Trigger (from live buckets) | Behaviour | Who is affected |
|---|---|---|---|
| L0 Normal | Headroom ≥ 30% | Full pipeline: LLM router, k=4, re-rank | — |
| L1 Tight per-minute | RPM or TPM headroom < 30% | Heuristic router (saves one call); k=3; no re-rank; no parent expansion | Everyone gets slightly lower quality; the draft keeps its budget |
| L2 Per-minute exhausted | Bucket empty or 429 received | Queue with deadline. Order: finish in-flight requests first, then oldest first. Shed new arrivals whose estimated wait > deadline. | New arrivals wait or are told "busy, retry in N s" |
| L3 Daily budget low | Remaining daily < reserve (**20% of the drafter's TPD**, author's decision 2026-11-09; see Budget plan for how the reserve is paced and the Lean tier on gpt-oss-20b) | Reserve the remainder for in-flight drafts. New requests get the **extractive answer**: top passages, each cited, plus "the papers say…", with no model synthesis. | New arrivals get a weaker but never-wrong answer |
| L4 Daily budget gone | RPD or TPD exhausted (429 with daily-reset semantics) | Extractive answer only, plus an honest note that full answers resume at the reset time, taken from headers or the error body | Everyone until reset |

**At 3pm with the daily budget gone (heading 7),** the user gets this in under ~500 ms:
- a plain-language note: "Full answers are paused until <reset time>",
- the 2–3 most relevant cited passages from local retrieval, which costs no Groq tokens, and
- if retrieval confidence is low, a clear "the papers don't appear to cover this".

---

## Retry policy (heading 6)

- **429 (per-minute):** retry only if `retry-after` fits inside the request's remaining deadline. Otherwise drop to L2 or L3 immediately.
- **429 (daily):** never retry; switch to L4.
- **5xx / connection error:** one retry with full-jitter exponential backoff (base 250 ms, cap 2 s).
- **First-token timeout:** one retry, only if at least 2 × the p50 first-token time is left before the deadline.
- **Total timeout after the first token:** no retry. Send what passed the gate, plus "answer truncated".
- **Global retry budget:** retries may be at most 10% of requests in a rolling 60 s window. This prevents retry storms.
- **Planned "retrying made things worse" experiment:** run the burst scenario on the stand-in twice. First with naive retries (3 immediate attempts, no budget), then with the policy above. Compare 429 % and p99 total time. Expected (to be confirmed): naive retries make the 429 cascade bigger and lengthen recovery.

---

## Stand-in model service

- **Interface:** an OpenAI-compatible `POST /v1/chat/completions` with SSE streaming, so the pipeline switches between real and stand-in only by `base_url`. It sends Groq-style `x-ratelimit-*` and `retry-after` headers and the same 429 body shape.
- **Latency:** first-token time and token rate are sampled from distributions **fitted to probe Phase 2 data** (e.g. lognormal for first-token time, normal for tokens per second). A separate heavy-tail probability models "slow for no visible reason".
- **Load slowdown:** latency multiplier = f(in-flight count), with the curve shaped from probe Phase 5 data.
- **Faults:** set rates for 5xx errors, hangs before the first token, and slow successes.
- **Limits:** its own token buckets (RPM, TPM, RPD, TPD), set by configuration. This supports the "per-minute limit halved" and "daily budget exhausted" scenarios.
- **Content:** sentences extracted from the provided context, with citations, so the gate and the grader run unchanged.
- **Where it is least like Groq (pre-registered guesses, to be checked):** its slowdown is a curve we chose, while Groq's queueing is hidden; its answers are extractive, so quality numbers from stand-in runs are not real model quality.

---

### As built (2026-11-10)

- **Calibration is a file, derived by a script.** `python -m probe.calibrate` turns `results/probe_raw.jsonl` into `standin/calibration.json`, so every stand-in number can be recomputed. Per model:
  - calm `queue_time` lognormal (120b median 278 ms);
  - share of queue episodes (≥1 s): 120b 1.7%, 20b 10.5%. Episode length is pooled across both models (median 1.8 s, n=10) because the 120b alone had only 2;
  - first-token remainder: lognormal whose log-median grows 0.0003 per prompt token (~170 ms at 264 tokens, ~360 ms at 2,813);
  - server output rate: 120b 475 tokens/s, 20b 895;
  - trickle: 4.2% at 67 tokens/s (120b);
  - 429 response time: median 102 ms;
  - 503 "over capacity": 0.64% of all chat calls (2 of ~310, both on qwen), answered after 30 s.
- **Shared rules.** `relay/limits.py` (continuous buckets plus Groq's 429 format and parser) is used by both the stand-in and the ledger, so they apply the same rules. TPM and TPD admission check prompt + `max_tokens`; TPM charges that reservation, TPD charges actual usage. The stand-in answers its 429s with Groq's exact message format, `retry-after` rounded up, and `x-ratelimit-*` headers.
- **Assumptions, not measurements.** These belong under "where it differs most" in heading 2:
  - Load slowdown: queue time × (1 + max(0, in_flight − 16)/16). Nothing was observed up to 16 in flight, and RPM 30 stops us measuring beyond that.
  - Episode length: fixed at 8 s, inferred from one 8-request batch delayed together and a 16-request batch 30 s later that was not.
  - Content: extractive. It quotes the best-matching sentence of up to 3 prompt passages with their `[cite]`, so its "quality" is not model quality.
  - The router always calls `search_papers`.
- **Smoke test** (`python -m tests.smoke_standin`, no Groq): a streamed, cited answer; a router tool call; a halved-TPM 429 whose stated wait the ledger adopts (and a correct admission after refill during a 4 s episode-delayed call); daily-exhausted refusal naming TPD with a 201 s wait (465 tokens ÷ 2.31/s), in 90 ms.
- **Bug found and fixed:** on Windows, asyncio treats sleeps shorter than its ~15.6 ms clock resolution as already due, so per-chunk pacing streamed everything at once. That would have made every stand-in first token equal its last token. The stand-in now paces against an absolute schedule; measured spread is 132 ms for 72 tokens, against ~150 ms expected.

### Pipeline as built (2026-11-10, developed on the stand-in; no Groq calls)

Modules: `relay/pipeline.py` (stages 0–6, CLI), `relay/gate.py` (sentence splitter + `check_citations`), `relay/router.py` (tool schemas, heuristic router), `relay/extractive.py` (zero-token answer), `relay/runtime.py` (answer cache, retry budget, row/trace recorder), `relay/settings.py` (every constant). Tests: `tests/test_gate.py`, `tests/smoke_pipeline.py`.

- **Brief's structural requirements:** ≥4 steps (admit, route, retrieve, decide, draft, gate, log); tool call (the 20b chooses `search_papers` or `refuse_out_of_scope`; the gate is a code tool); loop (retrieval reformulates up to 2 more times while confidence < τ_loop); branch (decide: answer / refuse / decompose); retrieval step (local hybrid search).
- **Decompose is per paper.** Measured on the gold set (h256k5 + re-rank, no Groq): a single search for a cross-paper question returns one paper only (Q13, Q15 → H-Net only; Q14 mostly Moshi). When a question names ≥2 papers (router's `papers` ∪ vocabulary match), the pipeline runs one search per paper, restricted to that paper, and merges round-robin. The router's `sub_questions` are used only for single-paper questions.
- **Refusal threshold.** Confidence is the top re-rank score when re-ranking, else the top dense cosine. Dense cosine does not separate answerable from unanswerable (0.77–0.93 vs 0.66–0.83); re-rank does for the clear cases (answerable ≥ −0.06; Q16, Q19, Q20 at −1.4 to −6.6; the near-misses Q17 and Q18 score like answerable ones). So the decide branch refuses only the clear cases, and the near-miss refusals depend on the drafter following rule 3 of its prompt. Thresholds are provisional (`relay/settings.py: TAU`), to be refit when Table 3 picks a setting.
- **The gate, as built.** A sentence passes when every citation is the label of a retrieved passage and every number appears in a cited passage or the question, or is the sum, difference, product or ratio of two numbers that appear **both in the cited passages and in the sentence** (e.g. "2.7 points (58.2 vs 55.5)"). The first version accepted any such combination and let "12 layers" pass as 6 + 6; requiring the operands in the sentence fixed it, and the drafter prompt asks for exactly that form. A "the papers do not say…" sentence needs no citation, but any number in it must come from the question.
- **Cut-short answers never show a half sentence.** On `max_tokens`, a total timeout or a stream error after the first token, the unfinished tail is dropped and logged, and the person sees the complete, verified sentences plus "Answer cut short".
- **Admission.** FIFO per model; a request is shed at once (not after waiting) when the estimated wait, including requests queued ahead, exceeds its remaining patience minus a 2 s drafting margin. Shed requests drop to Lean, then to quoted passages.
- **Burst finding (stand-in, 12 simultaneous requests, 20 s patience):** first version 4 Full / 1 Lean / 7 quoted; the 12 router calls had used the 20b's whole TPM, starving Lean. Fix: ladder L1 now counts requests already in flight (a burst is "tight" before any bucket has drained), and the router steps aside for the heuristic router when the 20b's headroom is < 30%. Result: 4 Full / 5 Lean / 3 quoted, everyone answered, first token 0.4–0.8 s for all but the two queued Full answers.
- **Cold start.** The first search took ~1.1 s (embedding model load) against ~0.03 s warm, so the pipeline warms the index, embedder and tokenizer on start.
- **Retrieval runs in a 2-thread pool** so CPU work (re-rank especially) doesn't block streaming for other requests.
- **Logging:** stand-in development runs log to `data/dev/` (git-ignored). Real calls and named scenarios always log to `results/pipeline_rows_<scenario>.csv` and `results/trace.jsonl`, append-only.
- **Corpus fix:** pieces under unnumbered headings now cite the nearest numbered section ("HNET §1" for H-Net's run-in headings in the introduction), not the heading title; text before any heading is "AIAYN front matter". Answer key still resolves 109/109.

**First real requests (2026-11-10, scenario `real_smoke`, 5 rows; author approved the spend).** Q1, Q15 and Q16 of the gold set, run one after another; Q1 and Q15 again after fixes.
- **Cost per Full answer: 1.9–2.7K tokens** (prompt 1.4–2.1K for the drafter plus ~0.4K router; output 117–652 including only 14–75 reasoning tokens at `reasoning_effort=low`). The Budget plan's 3.6K estimate was pessimistic: ~200K / 2.3K ≈ 85 Full answers per day, not 55. Recompute from the eval runs. Longest output was 559 tokens, so `max_tokens=900` has margin.
- **First answer sentence 1.0–2.1 s**, inside the 4 s target: router 0.3–0.6 s, retrieval ~0.01 s, drafter first byte 0.46–1.15 s (Groq `queue_time` 0.17–0.92 s of it). A one-sentence answer has first token ≈ total, because the gate releases a sentence only when it is complete.
- **Q16 (unanswerable) refused correctly** with "The papers do not say…"; the near-miss refusal path works on the real model (the stand-in cannot test it).
- **gpt-oss writes citations in full-width brackets 【AIAYN §3.2.1】.** The gate saw no citation and withheld the whole first draft of Q1, costing a second drafter call (+1.7K tokens, +0.6 s). Fix: normalise the bracket characters before the gate; the label is still checked. After the fix Q1 took one call, 1.0 s to first sentence, 2.2K tokens.
- **An uncited closing summary sentence** in Q15 was withheld, as designed. The prompt now says "every sentence, including any summary".
- **Decompose now uses the router's sub-questions first,** each restricted to the paper it names, then the whole question for any paper not yet covered. The router wrote focused sub-questions; the Moshi half of Q15 gained the RQ-Transformer detail.
- **A retriever trap found on the way (candidate for "How this library will mislead a naive retriever"):** H-Net's introduction lists the same stage names (encoder, chunking, main network, decoder) as its method section, so at 256-token pieces the introduction (§1) outranks §2.1.1 for Q10/Q15-style questions, with or without re-rank. At 512-token pieces, §2.1.1 and §2.1.2 rank first or second without re-rank (< 1 ms, against 1.6–4.2 s for re-rank). To be measured in Table 3.

**Local retrieval scan (2026-11-11, `python -m eval.retrieval_scan`, no Groq; rows in `results/retrieval_scan.csv`).** The pipeline's own retrieval path (loop, decide, per-paper decompose, passage cap) with the heuristic router, scored by whether each required point's listed cite is among the retrieved labels (an upper bound for a perfect drafter, and conservative because cite lists are a guide).

| Setting | Points reachable | All points reachable | Unanswerable refused before drafting | Answerable wrongly refused | Prompt tokens p50 / max | Retrieval ms p50 / max |
|---|---|---|---|---|---|---|
| bm256k8 (BM25 only) | 86.6% | 11/15 | 0/5 | 0/15 | 2,123 / 2,961 | 21 / 90 |
| h128k8 | 79.1% | 9/15 | 0/5 | 0/15 | 1,347 / 2,686 | 126 / 312 |
| h256k5 (current default) | 73.1% | 8/15 | 0/5 | 0/15 | 1,432 / 2,679 | 66 / 281 |
| h256k8 | 86.6% | 10/15 | 0/5 | 0/15 | 2,078 / 3,022 | 66 / 249 |
| h512k4 | 86.6% | 10/15 | 0/5 | 0/15 | 1,878 / 2,962 | 105 / 392 |
| h512k6 | 86.6% | 10/15 | 0/5 | 0/15 | 2,660 / 3,013 | 101 / 368 |
| h256k5rr | 76.1% | 9/15 | 3/5 | 0/15 | 1,399 / 2,562 | 5,119 / 15,152 |
| h512k4rr | 74.6% | 9/15 | 3/5 | 0/15 | 1,911 / 2,908 | 5,139 / 16,076 |

- **Re-rank lowers reachability and costs 5–16 s per request** (it runs on every search: loop iterations and decompose parts). The cross-encoder (`ms-marco-MiniLM`, trained on web search) drops H-Net §2.1.1 for Q10 entirely at 512. Its one strength: its score lets the decide branch refuse Q16, Q18 and Q20 before any drafter call, with no answerable question wrongly refused.
- **The current default h256k5 is the weakest hybrid.** Bigger pieces (512) or more pieces (k=8) both reach 86.6%.
- **BM25 alone does as well as hybrid on reachability** here, because the gold questions use the papers' own vocabulary.
- **Points no setting reaches** (with the heuristic router): Q6 (acoustic delay on both streams, inference sampling), Q7 (WavLM distillation, split RVQ), Q14 (Moshi's acoustic and text-audio delays), Q15 (H-Net's problem statement). These are Moshi §3.3.2 / §3.4.2–§3.4.4 details; the LLM router's sub-questions may reach them in real runs.

**Decisions on the evaluation (author, 2026-11-11):**
- **Table 3 settings: h512k4, h256k8, h512k4rr** (assistant's shortlist, accepted). Each pair isolates one dimension the brief names: piece size and pieces fetched (h512k4 vs h256k8, at a similar prompt size) and re-rank (h512k4 vs h512k4rr). The naive BM25 baseline's behaviour is shown from the local scan, not from real answers.
- **Grading: the assistant grades all 60 answers against the author's key**, by the amended strict rule, with a written note per required point (`results/grades_eval.csv`). No LLM judge on Groq. The author reviews a sample and overrules where they disagree; overrules are recorded as evidence for the write-up.
- **Option C vs B comparison: run it** (author: "no harm in running it"). It runs on the setting Table 3 picks, so after the main eval. The ungated baseline (Option D) costs nothing extra: the trace keeps every withheld sentence, so D's answer is C's answer plus what the gate withheld.

**Eval run, graded (2026-11-11; 60 real answers, `results/grades_eval.csv`; Table 3 from `python -m tools.tables --table 3`).**

| Setting | Good % | Correct (answerable) | Refused correctly | Tokens/answer | First token p50 / p95 ms | Total p50 / p95 ms |
|---|---|---|---|---|---|---|
| h512k4 | 45.0% | 4/15 | 5/5 | 2,602 | 1,318 / 2,268 | 1,740 / 2,568 |
| h256k8 | 40.0% | 3/15 | 5/5 | 2,988 | 1,388 / 2,542 | 1,788 / 3,447 |
| h512k4rr | 40.0% | 3/15 | 5/5 | 2,431 | 5,583 / 14,883 | 5,865 / 15,229 |

- **Every unanswerable question was refused correctly under every setting** (15/15); re-rank refused 3 of them at the decide branch with no drafter call, which is why its tokens/answer is lowest.
- **Why strict correctness is low (causes tagged per answer in the notes):**
  - *Gate false positives* (the gate withheld correct sentences): list markers "(1) (2) (3)" read as numbers (Q5, all three settings); "100 K" with a space vs "100K" (Q4 rr); a superscript in "p^0" read as 0 (Q10). These alone cost Q5 ×3 and Q4 rr.
  - *Retrieval misses*: the LLM router's rewritten query missed AIAYN Table 3 for Q4 at h512k4/h256k8 (both refused wrongly; h256k8 pulled HNET Table 1, the "Table N exists in every paper" trap); MOSHI Table 1 missed for Q9 at h256k8; H-Net Table 2 split into row-part pieces confused Q12 under every setting.
  - *Drafter omissions and wrong facts*: Q6, Q8, Q14, Q15 detail; "τ = 2 is a 2-second delay" (Q14 h512k4); "the decoder upsamples" (Q10/Q15 rr).
  - *Borderline points* (strict reading requires detail the question doesn't ask for): Q3 wavelength progression, Q7 codec numbers, Q10 recursion, Q11 probabilities to dechunking, Q13 length contrast. Pending the author's ruling; a lenient ruling would flip h256k8 to 7/15 and h512k4rr to 6/15, h512k4 to 5/15.
- **Re-rank breaks the first-token target** (p95 14.9 s against 4 s) without improving quality.

**Decisions after the graded eval (author, 2026-11-11):**
1. **Setting: h512k4** (on the assistant's recommendation from the strict grades). *Reopened the same day:* under the less-strict rule below, h256k8 leads (65.0% vs 50.0%); the assistant asked the author to confirm or switch.
2. **Fix the gate's three false positives before step 3** (accepted). List markers "(1) (2) (3)", a space before K/M/B ("100 K"), and super/subscripts ("p^0") are no longer read as data values; tests added; real counts and made-up numbers are still caught ("4 problems", "300 K steps"). Replaying the fixed gate on the eval's withheld sentences: all six false positives now pass, and three *wrong* Q12 sentences ("53.3 % at both scales") also pass. They had been withheld only by accident. That is the gate's known blind spot (a real number attached to the wrong row), shown on real output. Trace events now keep the whole sentence (they were cut at 300 characters, which hid the citations of long sentences).
3. **Grading made less strict (author: "right now the results make it seem as if we have a useless RAG system").** A required point counts when its main idea is stated correctly. Specific details inside it (numbers, names, formulas, secondary clauses) are required only when the question asks for them. Wrong facts and unsupported citations still make an answer wrong; refusal grading is unchanged. The first, stricter grades are kept in each changed answer's note ("strict: wrong"). Table 3 under each rule:

   | Setting | Strict: good % (correct) | Less strict: good % (correct) |
   |---|---|---|
   | h512k4 | 45.0% (4/15) | 50.0% (5/15) |
   | h256k8 | 40.0% (3/15) | 65.0% (8/15) |
   | h512k4rr | 40.0% (3/15) | 60.0% (7/15) |
4. **No Groq judge (author overruled the planned Option B run: "you be the judge cause we don't have a lot of tokens").** The Option B mode (buffer + one LLM judge on the 20b) that had been added for the comparison was removed unrun. The comparison is computed from the eval run alone (`python -m eval.c_vs_b`, 0 tokens): B sends nothing until the draft is complete, so B's first token is C's `total_ms`; the assistant's grading is the judge of "misread but passed the gate".

   | Setting | C first token p50 / p95 | B first token p50 / p95 | C lower by (p50 / p95) | Misread but passed | Gate-fail rate |
   |---|---|---|---|---|---|
   | h512k4 | 1,318 / 2,268 ms | 1,740 / 2,568 ms | 24.3% / 11.7% | 4/14 = 28.6% | 5/53 = 9.4% |
   | h256k8 | 1,388 / 2,542 ms | 1,788 / 3,447 ms | 22.3% / 26.3% | 0/14 = 0.0% | 6/57 = 10.5% |
   | h512k4rr | 5,583 / 14,883 ms | 5,865 / 15,229 ms | 4.8% / 2.3% | 3/15 = 20.0% | 8/56 = 14.3% |

   Against the pre-registered triggers: gate-fail stays under 25% everywhere. C's first-token p50 is 22–24% below a *judge-free* B, short of the 30% bar; but any runtime judge call adds to B's first token, and the bar is met once that call takes ≥ 143 ms at h512k4 (1,318 ≤ 0.7 × (1,740 + 143)). No Groq call measured on Day 1 or since returned that fast. The misread trigger (> 10%) fires at h512k4 (28.6%) and not at h256k8 (0%): this is the gate's blind spot, which only a meaning check (a judge) would catch, at a token cost the author ruled out.

**Decisions on the pipeline (author, 2026-11-10):** all three accepted as recommended below.

1. **`status` for a downgraded answer (author: accepted).** Recommended: `rate_limited` whenever a limit (a 429, or the ledger predicting one) gave the person a weaker answer than Full, including Lean and quoted passages; `ok` only when nothing forced a downgrade. Otherwise Table 2b would show 100% `ok` in the daily-exhausted row and hide the event.
2. **Tier in `retrieval_setting` (author: accepted).** Recommended: `h256k5` for Full, `h256k5|lean`, `h256k5|extractive`, and `cache` (already decided). The tier changes retrieval (Lean uses k−2 passages; extractive quotes passages), and this keeps Table 2b's tier breakdown computable from `results.csv` alone.
3. **After two drafts fail the gate (author: quoted passages, not a bare refusal).** The ADR said "refuse"; built as: quoted passages with "I couldn't produce a written answer that checks out against the cited passages". Still never wrong, more useful than a bare refusal. Evaluation runs never fall back (author's decision (d)), so this cannot inflate graded results.

## Telemetry

- **`results.csv`:** one row per request (pipeline requests, real and stand-in, plus every Groq probe request; see the gap 4 decision), with exactly these columns: `request_id, scenario, source, start_time, first_token_ms, total_ms, status, retries, tokens_in, tokens_out, retrieval_setting, answer_grade`.
  - `source` ∈ {real, stand-in}.
  - `status` ∈ {ok, rate_limited, timeout, error}. A client give-up is recorded as `timeout`, and this is stated in the write-up.
- **`trace.jsonl`:** per-phase events for each request: `queued, admitted, route_start/end, retrieve_start/end, call_start, first_byte, call_end, retry_wait_start/end, gate_pass/fail, first_answer_token, done`. This is what heading 4 (waiting, working, retrying) is computed from.
- **`probe_raw.jsonl`:** every raw Groq probe response, headers included (heading 2, Table 1).
- **`tables.py`:** recomputes every number in Tables 2a, 2b and 3 from `results.csv` and nothing else. Reviewers will do the same.

---

## Consequences

**Easier**
- Every degradation step has a named trigger and a named affected group, which answers headings 5 and 7 directly.
- The correctness floor costs no tokens, so it holds even at L4.
- The real and stand-in runs share one code path, so their results are comparable.

**Harder**
- More moving parts to defend live: the admission controller, the gate and the ladder.
- Sentence splitting around equations and abbreviations needs tests.
- Deadline-aware cancellation through async Groq streams needs care.

**To revisit after measurement**
- The fast targets (first-token and total-time p95), once probe Phase 2 data exists.
- Whether TTFT binds before RPM.
- The re-rank and parent-expansion defaults, once Table 3 results exist.
- The value of the L3 reserve.
- The routing threshold τ and the loop cap.

### Open questions

1. What should `answer_grade` be for stand-in rows? Proposal: grade them, but state that their content is extractive and the quality is not real model quality. The alternative is to grade them `not measured`, which the column's allowed values don't include.
2. Which Groq models are on the free tier today, and what are their per-model limits? Answered by probe Phases 0–1.

---

## Action Items

### A. Repo and safety (before any code)
1. [x] `git init` inside `vaani-ai-assignment`. The current git repository is the whole home folder.
2. [x] Add `.gitignore` with `.env`, `data/raw/`, `__pycache__/`, `.venv/`. Keep the Groq key only in `.env`.
3. [x] Save `The_Relay_Assignment_Brief.pdf` into the repo (copied from Downloads, 2026-11-09).
4. [ ] Start exporting this assistant conversation as you go. It's a required submission item.

### Schedule (author's decision, 2026-11-09)

1. Probe first, in stages.
2. End of Day 1: exhaust the daily limit on a separate model the pipeline doesn't use.
3. Build the stand-in and the pipeline from the probe data.
4. Run the gold-set questions and the real load scenarios.
5. End of that day: exhaust the daily limit again, this time to collect the real "daily budget exhausted" scenario.

**Risk to confirm on Day 1:** Groq reports a `reset-requests` duration, not a calendar reset. If the daily budget refills gradually over 24 hours instead of all at once, step 5 also eats into the next day's budget. Before step 5, read the reset behaviour from the Day 1 headers.

**Model choice (author's decision, 2026-11-09):**
- Router: `openai/gpt-oss-20b`.
- Drafter: `openai/gpt-oss-120b`. Both have public prices, needed for cost per 1,000 answers.
- Sacrificial model for the Day 1 daily-limit test: `llama-3.1-8b-instant`. The pipeline doesn't use it. **Superseded the same day:** the model returned 404 on this account, so the sacrificial model is `qwen/qwen3.8-27b`, which has the same limits as the pipeline models.
- **Daily test timing (author's decision, 2026-11-09):** the assistant suggested running the qwen daily test straight away, because qwen's budget is separate from the pipeline models and watching recovery can take hours. The author kept it for the night, as in the original schedule.
- Accepted risk: the reasoning tokens gpt-oss produces may push back the first answer token. Probe phase 2 measures this. If it's too slow, `reasoning_effort` and the Llama models are the levers.

### Decisions, evening of 2026-11-09 (author)

- **Table 2a sources:** steady and burst are run on **real Groq**. Sustained overload and per-minute-limit-halved run on the **stand-in**. Groq's limit can't be halved, and overload would only burn the daily budget. The daily-exhausted row's source is still open (see the budget plan).
- **Latency samples (gap 5):** no extra probe requests. Table 2a's real rows and the retrieval-evaluation answers supply the samples. Every percentile is reported with its n, and a p99 from fewer than 100 samples is flagged.
- **Second time window (gap 6):** before the night daily test, repeat the afternoon latency run: gpt-oss-120b, 15 each of small, medium and large, plus gpt-oss-20b, 15 small. Same design as the afternoon run, so the two windows compare like for like. The cost (~70–89K on 120b, ~6.5K on 20b) is refilled before the next morning. The bucket caps at 200K, so leaving tokens unspent tonight would waste them.
- **Limit recheck (gap 8):** on the final day, one `headers` request per model, compared with `results/groq_console_limits_2026-11-09.md`.
- **The budget plan covers the product, not just our own runs** (author's correction to the assistant's framing). See "Budget plan" below.
- **Daily-exhausted row: real**, folded into the steady and burst runs on the last evening (Nov 12). Those runs keep going until the drafter's daily budget is gone. The stand-in version is kept as a fallback in case the real run fails.
- **Per-minute behaviour on the 120b (gap 7):** no dedicated test. The real burst run on the 120b will hit its per-minute limit, and Table 1 cites that.
- **`results.csv` (gap 4):**
  1. Exactly the 12 columns the brief specifies.
  2. Every Groq probe request is a row, labelled by scenario (`probe_rpm`, `probe_tpm`, `probe_daily_recovery`, …).
  3. Real pipeline rows and stand-in rows go in the same file, told apart by `source`.
  4. `answer_grade` is empty on probe rows, because a probe has no answer to grade. The report says so.
  5. For limits the CSV can't reconstruct (the exact RPD and TPM values, and what a 429 says), the evidence is the raw headers and bodies in `results/probe_raw.jsonl` and the console capture in `results/groq_console_limits_2026-11-09.md`.

### Budget plan (author's decisions, 2026-11-09)

The budget is planned in **tokens per model per day**. TPD (200K) binds long before RPD (1,000). One component, the **budget ledger**, serves both our own real runs and the product at runtime.

**Capacity, from the console limits and estimated costs** (recompute once the real cost per answer is measured):

| | Value |
|---|---|
| Full answers per full drafter budget | ~55 (200K ÷ ~3.6K) |
| Burst ceiling | ~2.2 full answers/min (TPM 8K) |
| Steady state once the stored budget is spent | ~2.3 full answers/hour (refill 2.3 tokens/s) |
| Router calls per day (gpt-oss-20b, separate 200K) | ~260 |

**What the product's budget manager does (all six accepted by the author):**
1. **Live budget view.** TPM and RPD come from the `x-ratelimit-*` headers on every response. TPD isn't in the headers, so the ledger keeps its own estimate: it charges each call, refills at TPD/86,400 per second, and corrects itself whenever a daily 429 reports "Used N". Limits are read live, not hard-coded, because the brief says they "can change without notice".
2. **Answer tiers by budget.** Each request gets a token allowance:
   - *Full:* gpt-oss-120b drafts, using the retrieval setting chosen in Table 3.
   - *Lean:* **gpt-oss-20b drafts** (author: yes), labelled to the user as a shorter answer, with fewer passages and a smaller `max_tokens`.
   - *Extractive:* cited passages, no drafter tokens.
   - *Cached:* see 4.

   This answers heading 5 (who gets a weaker answer) and the retrieval question (what retrieval does when the limit is close).
3. **Daily reserve: 20% of the drafter's TPD** (author, 2026-11-09). That is 40K tokens, about 11 full answers. A burst gets ~44 full answers before answers switch to Lean. The assistant suggested 20%; the author agreed. An intermediate message said 40%, which the author later confirmed was a typo; 30% was briefly recorded before that correction. Full answers draw freely while the drafter's level is above 20%. Below it, the reserve is released at a paced rate, so a morning burst can't take the afternoon's answers.
4. **Answer cache** (author: yes; the author also thought of it independently). Bursts follow incidents, so many people ask the same question. A cached answer has already passed the citation gate, costs 0 tokens, and returns in milliseconds.
5. **Nobody turned away.** If a person's estimated wait exceeds the patience deadline, they get the extractive answer straight away instead of being dropped.
6. **Cost per 1,000 answers (Table 2b).** Computed from `tokens_in` and `tokens_out` at Groq's list prices (gpt-oss-120b $0.15/$0.60, gpt-oss-20b $0.075/$0.30 per 1M tokens). The first estimate is ~$0.70 per 1,000 full answers.

**Details proposed by the assistant, all approved by the author on 2026-11-09:**
- (a) **Pacing the reserve.** Below 20%, release the reserve evenly over the hours left in the working day (end of day configurable, default 18:00 local), on top of the refill. By evening the whole reserve is available, which is also what lets the last-evening real run reach a truly empty budget for the daily-exhausted row.
- (b) **Marking cache hits in `results.csv`.** Within the 12 columns, a hit is recorded as `retrieval_setting=cache`, with `tokens_in` and `tokens_out` at 0. Hit rate is then countable. Load runs use a question mix with a stated repeat rate, because an all-identical burst would make the cache look better than it is.
- (c) **The router has priority on gpt-oss-20b.** Lean drafts compete with routing for the 20b's 200K. Lean drafting stops while the 20b's level is below 25%, so routing (and with it the extractive answer) keeps working.
- (d) **Evaluation runs don't use tiers.** Table 3 compares retrieval settings, so every evaluation answer must be a Full answer. An evaluation run checks the ledger first and refuses to start (or pauses) if the budget won't cover it, instead of quietly downgrading answers.
- (e) **L4's message changes.** The daily budget refills continuously, so there is no single reset time. The 3pm message gives the estimated time until the next full answer ((cost − level) ÷ refill rate), not "resumes at midnight".

**Ledger accounts (clarified 2026-11-10).** One account per model, because Groq limits each model separately. Each account tracks RPM, TPM, RPD and TPD:
- `gpt-oss-120b`: Full drafts only, with the 20% reserve.
- `gpt-oss-20b`: router plus Lean drafts. The router has priority, and Lean drafting stops below 25%.
- qwen is not used by the product.

**Extractive answers use no Groq call (assistant's recommendation, 2026-11-10).** They are built entirely by local code: retrieval, sentence selection, citations from section paths, and an out-of-scope check by retrieval-score threshold instead of the 20b router. The fallback therefore costs 0 tokens, works when both models are exhausted or Groq is down, and returns in < ~500 ms. Cost: refusal accuracy in extractive mode depends on the score threshold. It is measured on the gold set's unanswerable questions, with no Groq calls.

**Our own real runs** go through the same ledger. Every real run's dry run prints its projected tokens per model, and the run refuses to start if that exceeds the day's plan (keeping 15% slack for reruns).

### Decisions, 2026-11-10 (author)

- **Gold questions:** written by the author (≥15; aim ~20, including unanswerable, cross-paper, number/table and retriever-trap questions). The assistant may review them for coverage but does not write them.
- **Embedding and re-rank stack:** FastEmbed (ONNX). The alternative, sentence-transformers, pulls in all of PyTorch.
- **Decompose drives retrieval only.** Sub-questions retrieve, their pieces are merged, and there is one draft. At most **2 drafter calls per request** (draft plus one regeneration).
- **"Fast" targets, revised from measurements:** first-token p95 ≤ 4,000 ms; total p95 ≤ 8,000 ms. These replace the provisional 3,000 / 15,000 ms. The basis: p50 ≈ router ~0.4 s + retrieval ~0.1–0.3 s + drafter first token ~0.6 s, with Groq queue episodes adding ~2 s at the tail, and the trickle setting total time.
- **The Option C vs B comparison uses both p50 and p95.** Measurement shows the per-sentence gate's advantage is concentrated in the slow tail: a 300-token answer completes ~0.6 s after its first token at 476 tokens/s, but ~4.3 s after it during a 70 tokens/s trickle.
- **Build order:** corpus ingestion and retrieval start now. Development stays independent of Groq wherever possible: the stand-in serves development, and real calls are for budgeted runs only.

### Corpus and retrieval findings (2026-11-10, no Groq calls)

- **The sources contain text the published papers don't.** AIAYN v7 ships `parameter_attention.tex` and `sqrt_d_trick.tex`, but their `\input`s are commented out of `ms.tex`. H-Net defines `\new{…}` to print nothing, which hides edit-marked text. The parser follows only live `\input`s after stripping comments, and it expands the papers' own macros. A naive "index every .tex file" ingester would retrieve unpublished text. This is a candidate for "How this library will mislead a naive retriever".
- **AIAYN v7 contradicts itself on EN-FR BLEU:** the abstract and Table 2 say 41.8, while the §6.1 text says 41.0.
- **Structure parsed** (numbering computed the way LaTeX does, then checked against the published structure):
  - AIAYN: §1–7, Tables 1–4, Figures 1–5.
  - Moshi: §1–7 plus appendices A–F, Tables 1–19, Figures 1–11.
  - H-Net: §1–5 plus appendices A–F, Tables 1–6, Figures 1–14.

  Every piece carries a `cite` label ("AIAYN §3.2.1", "MOSHI Table 7"). Totals: ~71K tokens of content across 433 blocks.
- **Piece sizes built:** 128 (803 pieces), 256 (462), 512 (240) target tokens. Tables are split by row groups, and each group repeats the caption and header.
- **Re-ranking is the expensive part of retrieval, and it costs CPU time, not tokens.** Warm, on this laptop: hybrid BM25 + dense search takes 11–50 ms. The cross-encoder re-rank takes 0.8–1.7 s for a pool of 10 and 1.6–3.6 s for a pool of 30, growing with piece size. That is up to ~90% of the 4 s first-token p95 budget, it runs on the shared CPU, and it would serialise under a burst. Table 3 must report it. L1 of the degradation ladder (drop re-rank) is justified by this measurement.

- **Numbering check (author, against the PDF):** Moshi Table 3 (§5) and Table 17 (appendix A) really do share the caption "Ablation study on hyperparameters of the Mimi codec". The parser's float numbering is correct. Two tables with the same caption are also a retrieval trap.
- **Gold set received (author, 2026-11-10):** 20 questions in `eval/the_relay_20_evaluation_questions.txt`: 12 single-paper (4 AIAYN, 5 Moshi, 3 H-Net), 3 cross-paper, 5 unanswerable. The questions are the author's; the assistant reviewed coverage only.

- **Grading rule: strict (author, 2026-11-10).** `correct` means every required point is present and right, with each claim cited to a supporting passage. Any missing point, wrong fact or unsupported citation is `wrong`. Full rule in `eval/answer_key.json`. (Citation rule amended later the same day; see "Grading rule amended" below.)
- **Answer key:** drafted by the assistant from the parsed text (`eval/answer_key.json`, 108 citations, all checked by `python -m eval.check_key`). Every question stays `verified: false` until the author checks it against the PDF.
- **Gold set verified by the author against the exact-version PDFs (2026-11-10).** Canonical files:
  - `eval/the_relay_20_evaluation_questions.txt`
  - `eval/answer_key.json` (20/20 verified, 109 cites, all resolving)
  - `eval/question_verification_report.txt`

  History is kept: `eval/answer_key_draft.json` (the assistant's draft, later flagged verified by the author without the refinements) and `eval/the_relay_20_evaluation_questions_original.txt`.

  The author tightened Q6, Q11, Q12, Q15 and Q19, and also reworded Q8 (punctuation), Q10 (names every stage) and Q14 (distinguish acoustic from text-audio delay). Q12 now asks for the §3.1-versus-Table-2 discrepancy explicitly. Q19 is now a yes/no question about a multi-human experiment.
- **Two key fixes after verification (assistant found, author approved, 2026-11-10).** Under strict grading, both would have marked correct answers `wrong`:
  - Q12: "Large H-Net (2-stage) matches XL BPE" moved from required to optional, because the reworded question no longer asks what the results indicate.
  - Q15: the decoder point now also accepts HNET §2.1.2 and §2.3. The sentence it paraphrases is in §2.1.2; only §2.1.1 was listed before.

  Still 109 cites, all resolving. Two small slips in the verification report are left as the author wrote them: eight questions changed, not five; and Q18's 8,192-byte setup is in HNET §3, not §3.2.
- **Grading rule amended (assistant recommended, author approved, 2026-11-10).**
  - **Citations count by support, not by list membership.** A citation counts if the cited passage supports the claim; the key's per-point cite lists are a guide, not an exhaustive set. Why: papers repeat facts in several places, and the list-only rule had already mis-graded Q15. Unsupported citations are still `wrong`. Required points stay all-or-nothing, and refusal rules are unchanged.
  - **Point coverage is recorded as a diagnostic, not a grade.** For each answerable question the grader writes points covered / points required to `trace.jsonl`. It never enters `results.csv` or changes `answer_grade`. Table 3's quality column stays the strict % correct from the CSV. Coverage is used only in the write-up, to explain differences between retrieval settings when 15 strict correct/wrong grades are too coarse, and is labelled as coming from the trace.
- **No trap questions added (author, 2026-11-10).** The examples in "How this library will mislead a naive retriever" will draw on the existing set: the table-number collisions in Q4, Q9 and Q12, and the near-miss unanswerables in Q16–Q20. Findings from ingestion, such as the self-contradictions and unpublished source text, are described rather than measured.
- **Q1–Q3 kept as written (author overruled the assistant, 2026-11-10).** The assistant suggested replacing two of them because a model can answer them from memory, which could inflate correctness when retrieval fails. The author kept them. Consequence to report: Part A's scores measure grounding and citation more than retrieval.
- **A second self-contradiction found (H-Net):** §3.1 says H-Net (2-stage) beats the BPE Transformer by "+2.2% and +2.6%" at Large and XL, but Table 2 gives 58.2 − 55.5 = 2.7 at XL. The H-Net (2-stage) Large score (55.5) equals the XL Transformer's.
- **Parser defects found and fixed (2026-11-10):**
  - siunitx units were dropped: `\SI{160}{\ms}` now gives "160 ms";
  - `\looseness=-1` debris removed;
  - `\ref`s to labels placed mid-section now resolve, with 0 unresolved;
  - nested `\frac` now gives "softmax((QK^T)/(sqrt(d_k)))V";
  - `\multirow` nested in `\multicolumn` now gives Moshi Table 1's "same" cells;
  - `\midrule[0.3pt]` debris removed;
  - Moshi's `\ensuremathmode` (`\ifmmode X \else $X$ \fi`) no longer duplicates symbols or splits paragraphs.

  A scan finds no leftover LaTeX commands. The answer key's 108 citations still resolve after re-chunking.

### Phase 0 findings (Groq docs, read 2026-11-09)

- **Free-tier numbers aren't public.** `console.groq.com/docs/rate-limits` shows only the Developer-plan table. Free-tier values are on the account's limits page in the console. The author copied that page on 2026-11-09: see `results/groq_console_limits_2026-11-09.md`. gpt-oss-20b, gpt-oss-120b and qwen3.8-27b each have RPM 30, RPD 1K, TPM 8K and **TPD 200K**.
- **The headers cover only two limits.** `x-ratelimit-limit-requests` is RPD and `x-ratelimit-limit-tokens` is TPM. The `remaining-*` and `reset-*` headers follow the same split. `retry-after` (seconds) is sent only on a 429. **RPM and TPD aren't in the headers.** They come from the console or from hitting them.
- **Scope:** limits apply at the organisation level, and the table varies by model.
- **"Cached tokens do not count towards your rate limits."** Every probe prompt starts with a unique nonce to defeat caching.
- **Models:** the production text models are `llama-3.1-8b-instant`, `llama-3.3-70b-versatile`, `openai/gpt-oss-20b` and `openai/gpt-oss-120b`. Only the gpt-oss models list public prices ($0.075/$0.30 and $0.15/$0.60 per 1M tokens in/out). Those prices are needed for cost per 1,000 answers.
- **gpt-oss reasoning tokens:** these models reason before answering. The probe records the first reasoning token and the first answer token separately; the answer token is what users experience as time to first token.
- **Concurrency:** the docs don't mention a limit, so it is measured.

### Probe findings so far (Day 1; recompute with `python -m probe.analyze_probe`)

| Finding | Evidence (raw file: `results/probe_raw.jsonl`) | Confidence |
|---|---|---|
| `gpt-oss-20b`, `gpt-oss-120b` and `qwen3.8-27b` each allow RPD 1,000 and TPM 8,000. `allam-2-7b` allows RPD 7,000 and TPM 6,000. | `x-ratelimit-limit-*` headers | High |
| `llama-3.1-8b-instant` and `llama-3.3-70b-versatile` aren't available to this account. The Llama 8B call returned 404 `model_not_found`. | Probe record; `results/models_raw.json` | High |
| **The daily budget refills continuously; it doesn't reset at a fixed time.** `reset-requests` = requests used × 86,400/RPD exactly, at 1, 2, 17 and 32 used, and on 3 models. | Analyzer's "daily refill check" | High |
| **TPM refills continuously too.** 117 tokens used gave `reset-tokens` = 877 ms, versus 117/8,000 × 60 s = 878 ms. | Headers phase | High |
| `max_tokens` includes reasoning tokens. With `max_tokens=16`, `gpt-oss-120b` returned an **empty** answer. | Headers phase, `finish_reason=length` | High |
| With `reasoning_effort=low`, the reasoning cost is small: ~8 tokens (max 22). | Latency phase usage | High (n=45) |
| **Prompt size barely changes latency, but sets the budget.** Going from 264 to 2,813 prompt tokens added only ~200 ms to first-token p50 (492 → 696 ms), while using 10× the TPM. | Latency phase, `gpt-oss-120b` | Medium (n=15 per size) |
| **TPM is the binding limit.** At ~3K tokens per drafting call and TPM 8,000, `gpt-oss-120b` can serve about 2.7 drafts per minute, far below the RPM of 30. | Arithmetic on the measured limits | High |
| A large share of time to first token is spent waiting in Groq's queue: `queue_time` p50 300–370 ms against server compute of 290–430 ms. | Latency phase `usage.queue_time` | Medium |
| **Slow successes happen after the first token, as a trickle, not a stall.** On `gpt-oss-120b`, 3 of 45 requests in the first run and 2 of 30 in the instrumented rerun (6.7% in both) took 1.4–3.3 s *after* the first answer token. Groq's own `completion_time` for them was ≤ 0.28 s. In the instrumented rerun the longest silence was ≤ 347 ms: tokens arrived steadily at ~70 tokens/s against the ~476 Groq reports. Not seen on `gpt-oss-20b` (0 of 30). Groq-side or network-side is still unknown. | First run seq=5, 11, 15; rerun seq=2, 11 | Medium |

| **RPM = 30.** The first 429 came at request #31, 13.0 s into a back-to-back burst. | `rpm_burst` | High |
| **A 429 identifies which limit was hit.** The body has `type: "requests"` or `"tokens"`, `code: "rate_limit_exceeded"`, and a message like "…on requests per minute (RPM): Limit 30, Used 30, Requested 1. Please try again in 2s." The `retry-after` header gives whole seconds, rounded up; the body gives the exact figure (e.g. "3.4125s"). | `rpm_burst`, `tpm_burst` 429 bodies | High |
| **A 429 is fast and free.** It returns in 51–59 ms, and `remaining-requests` didn't drop, so rejected calls don't count toward the daily limit. | Same records | High |
| **Recovery is quick: one bucket's worth of refill.** RPM recovered 2.0 s after the 429 (one request refills every 60/30 = 2 s). TPM recovered 4.0 s after (its deficit ÷ 133 tokens/s). There's no penalty period. | `*_recovery` records | High |
| **Groq reserves `max_tokens` against TPM before running a request.** The TPM 429 said "Requested 2876" = 2,812 prompt tokens + `max_tokens` 64. So an unused `max_tokens` allowance still uses budget. | `tpm_burst` 429 body | Medium–High (one 429; recheck on the daily run) |
| **TPD = 200,000, confirmed by hitting it.** On qwen3.8-27b, the first daily 429 came after 72 successful requests in 30.9 min. Message: "…on tokens per day (TPD): Limit 200000, Used 199910, Requested 2821. Please try again in 19m39.792s." `type: "tokens"`, `retry-after: 1180`. | `daily_tokens` run 20261109T180359Z | High |
| **The daily token budget refills continuously, at 2.31 tokens/s.** On refused polls, "Used" fell by ~139 every 60 s. The fitted rate is 2.31 tokens/s × 86,400 = 199,937/day, i.e. TPD/86,400. There's no fixed reset. Stated waits fell by exactly 60 s per 60 s. | `daily_watch` 429 readings | High |
| **The daily budget charges tokens actually used, not `max_tokens`.** "Used" at the 429 (199,910) matches tokens actually used minus refill (≈197,007), not reserved minus refill (≈218,535). 36 one-word requests with `max_tokens=600` made the two hypotheses ~21K apart. So `max_tokens` costs per-minute throughput (TPM reserves it) but not the daily budget. The remaining ~2.9K gap is about one large prompt, consistent with the one 503 request having been charged (unconfirmed). | Same run | High (actual vs reserved); Low (503 charged) |
| **Recovery from an empty daily budget depends on request size.** After the large request was refused, only ~90 tokens were left: 2 tiny requests got through before a tiny one was refused too. A tiny request then succeeded 68 s later. A large one (~2,830 tokens) succeeded **1,277 s (21.3 min)** after the first daily 429. | `daily_drain`, `daily_watch` | High |
| **The daily 429's stated wait is exact.** The large request succeeded on the first probe sent at the stated wait + 1 s. Across 20 refused polls, "Used" fell at a fitted 2.31 tokens/s (= 199,994/day). The ledger can trust the stated wait, and the 3pm message can quote it. | `daily_watch`, final poll | High |
| **Daily budgets are per model.** gpt-oss-20b served normally right after qwen's daily 429. Its `remaining-requests` went 984 → 999, i.e. refill only, so qwen's 74 requests didn't count against it. | `daily_check_before/after` | High |
| **New failure mode: 503 "over capacity" after ~30 s.** 2 of ~80 qwen calls returned `503 internal_server_error`, "qwen/qwen3.8-27b is currently over capacity. Please try again and back off exponentially", after 30.0–30.1 s. Headers arrived only then: it fails slowly, not fast. One arrived while the daily budget was empty, so the capacity check can come before the limit check. It counts against RPD. | `daily_tokens` seq ~5, `daily_drain` poll 0 | Medium (seen on qwen only, n=2) |
| **Second time window (evening, 23:34 IST): medians stable, tails change cause.** 120b first-token p50 afternoon → evening: small 465 → 420, medium 514 → 513, large 696 → 619 ms. In the evening the slow tail came from Groq queue episodes *before* the first token: `queue_time` p95 ~2.0–2.3 s on small and medium. The afternoon's tail came from trickles *after* the first token. Trickles across both windows: 5 of 120 on 120b (4.2%), all in the afternoon. The earlier "~7%" came from one window. The stand-in models trickles as episodic, not as a constant rate. | `latency` phase, runs before and after 17:00 UTC | Medium (n=15 per size in the evening) |
| **The console's limits match every measured value.** RPM 30, RPD 1K and TPM 8K agree with the headers and the 429 bodies. **TPD is 200K** on gpt-oss-20b, gpt-oss-120b and qwen3.8-27b. That value comes from the console; tonight's daily test checks it. | `results/groq_console_limits_2026-11-09.md` | High for RPM, RPD and TPM (two sources each). Medium for TPD until measured. |
| **Tokens per day, not requests per day, is the binding daily limit.** A full answer reserves about 3.6K tokens on the drafter (~3K prompt + ~600 `max_tokens`), so 200K tokens buys about **55 full answers per day**, against an RPD of 1,000. If the token budget refills like the request budget, it comes back at 200K/86,400 ≈ 2.3 tokens/s, or about one full draft every 26 min. | Arithmetic on console TPD and the measured prompt sizes | Medium (refill rate assumed until tonight) |
| **No concurrency limit seen up to 16 requests at once.** 16 of 16 succeeded. In practice RPM and TPM cap concurrency long before any in-flight limit would. | `concurrency` | Medium (only tested to 16) |
| **Slowness is Groq-side queueing, and it comes in episodes.** In the 8-request batch, 5 of 8 waited 2.07–2.24 s in Groq's `queue_time` while server compute stayed at 125–142 ms. The 16-request batch, sent 30 s later, saw no such wait (max 642 ms). Requests sent together were delayed together, so the slowness is episodic and not caused by our concurrency. | `concurrency` level 8 vs 16 | Medium |

**Design consequences:**
- **Set `max_tokens` tightly per step.** Router ~150, drafter ~600, with reasoning headroom. `max_tokens` is charged against TPM up front, so a draft with a 3K-token prompt and `max_tokens=1024` reserves ~4K tokens. That allows only ~2 drafts per minute.
- **Retry a per-minute 429 at the time the body states, not with blind backoff.** The precise figure is in the body; the header rounds it up. Never retry a 429 whose message says "per day".
- **Model slow episodes as correlated across requests.** In the stand-in, queue delay spikes should hit requests that arrive close together, not each request independently. Groq's `usage.queue_time` gives the real service's waiting-versus-working split for heading 4.
- **The load-slowdown curve can't be calibrated from the free tier.** No concurrency effect is visible up to 16, and RPM caps us at 30/min. The stand-in's slowdown under load is therefore assumed, not measured. This is "where it is least like the real thing" (heading 2).
- Retrieval size is limited by the token budget, not by latency. When the limit is close, reduce chunks fetched (k) first.
- The answer timeout splits into a first-token timeout and a total-time timeout. A silence timeout between stream events would **not** catch the trickle, because the stream never goes quiet. A slow trickle is mostly harmless to someone reading as it streams, so the total-time timeout should be generous.
- The stand-in models delivery as a mix: most requests stream at the server-reported rate, and ~7% trickle at ~70 tokens/s after the first token.
- Exhausting the drafting model's daily budget at end of day costs the next morning's budget: ~580 requests are back after 14 h.
- **A 503 can hang for ~30 s before any header arrives.** The first-token timeout must be well below that. The measured first-token p99 is ~2.4 s, so a timeout of ~4 s lets one retry still fit inside the deadline. The stand-in gets a "503 after 30 s" fault at ~2.5%.
- **`max_tokens` affects throughput, not the daily budget.** A generous `max_tokens` lowers drafts per minute (TPM reserves it) but doesn't spend the day. Keep it tight for burst throughput; the daily ledger charges actual `usage`.
- **The 3pm reply can be honest and specific.** With the daily budget empty, a small request gets through in about a minute, and a full draft in ~(cost − level) ÷ 2.31 s ≈ 20 min. The stated wait in the 429 is exact, so the ledger can use it directly.
- **The real-call budget for the whole project is about 55 full answers per day on the drafter.** Day 1 probing already used ~82K of gpt-oss-120b's 200K and ~70K of gpt-oss-20b's. The gold-set evaluation (≥15 questions × ≥3 retrieval settings ≈ 45+ drafts ≈ 160K tokens) nearly fills one day on its own. The real load rows need their own day. Action item B8 (daily budget plan) is now a blocker, not a nice-to-have.

### B. Probe the real service (Day 1; log everything to `results/probe_raw.jsonl` via `python -m probe.groq_probe`)
1. [ ] Phase 0: read Groq's rate-limit docs and the console limits page for 2–3 candidate models.
2. [ ] Phase 1: one `max_tokens=1` call per model, capturing all `x-ratelimit-*` headers.
3. [ ] Phase 2: 60–100 streamed requests below RPM, at 3 input sizes (~200, ~1.5K, ~4K tokens), across 2 time windows. Gives first-token and total-time distributions.
4. [ ] Phase 3: RPM test. Send RPM+5 tiny requests within 10 s, then poll every 2 s. Records the 429 body, `retry-after`, and time to recover.
5. [ ] Phase 4: TPM test. Send ~4K-token prompts back to back until 429; confirm it is the token limit.
6. [ ] Phase 5: concurrency at 1, 2, 4, 8, 16 parallel streams.
7. [ ] Phase 6: exhaust the daily limit on **one dedicated model not otherwise used**, near the end of a day, to see the real daily 429 and its reset.
8. [ ] Write a daily budget plan: how many real calls are spent on probes, retrieval evaluation and real load runs on each day.

### C. Build
1. [x] Ingestion: fetch the LaTeX sources at the exact versions, parse them into sections, chunk them into children with section paths, build the BM25 and embedding indexes.
2. [x] Pipeline stages 0–6 with deadlines and cancellation (2026-11-10, on the stand-in; first real call still to do).
3. [x] Stand-in service, calibrated from §B data.
4. [ ] Load generator with scenarios S1 steady, S2 burst, S3 sustained overload, S4 per-minute limit halved, S5 daily budget exhausted. Questions mix simple, multi-step and unanswerable, so step counts vary. Patience is sampled per simulated user.
5. [ ] Telemetry: `results.csv`, `trace.jsonl`, and `tables.py`.

### D. Evaluate
1. [ ] Gold set of ≥20 questions, written by you and not answerable from memory, including ≥5 that the papers don't answer.
2. [ ] Retrieval settings, at least:
   - RS1: BM25 only, 256-token pieces, k=8. This is the naive baseline that feeds the "mislead a naive retriever" section.
   - RS2: hybrid, 512-token pieces, k=4, no re-rank.
   - RS3: hybrid, 512-token pieces, k=4, re-rank.
   - RS4: RS3 plus a capped parent expansion.
3. [ ] Record three ways a naive retriever goes wrong, each with an example question, observed on RS1.
4. [ ] Run real S1/S2 at small scale; run S1–S5 on the stand-in. Then run the retry experiment.
5. [ ] Pick one slow request from `trace.jsonl` and split it into waiting, working and retrying (heading 4).

### E. Write-up (≤2 pages, headings in the brief's order)
1. [ ] Headings 1–5; then "Retrieval: quality versus cost"; then "How this library will mislead a naive retriever"; then headings 6–8; then "Your feedback on Vaani"; then "How I worked with the assistant", with the raw history.
2. [ ] Fill in Tables 1, 2a (≥5 rows), 2b and 3 completely. Use "not measured" where something wasn't measured. Times in ms; percentages to one decimal place.
3. [ ] Include at least one number that couldn't have been guessed without running it, e.g. the gate-fail rate or the recovery time after a 429.
4. [ ] Cost per 1,000 answers at Groq's **paid** prices, stated as such.
