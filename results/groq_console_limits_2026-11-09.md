# Groq console limits page (free plan), captured 2026-11-09

Source: console.groq.com → Settings → Limits → "Organization Limits", copied by the author on 2026-11-09.
The page says these are base limits for the organisation and "can be customized per project". The response headers
(`x-ratelimit-limit-requests`, `x-ratelimit-limit-tokens`) match the RPD and TPM values below for every model probed,
so the project limits equal the organisation limits.

Pasted as shown on the page (K = thousand):

## Chat Completions

| Model | Requests per Minute | Requests per Day | Tokens per Minute | Tokens per Day |
|---|---|---|---|---|
| allam-2-7b | 30 | 7K | 6K | 500K |
| meta-llama/llama-prompt-guard-2-22m | 30 | 14.4K | 15K | 500K |
| meta-llama/llama-prompt-guard-2-86m | 30 | 14.4K | 15K | 500K |
| openai/gpt-oss-120b | 30 | 1K | 8K | 200K |
| openai/gpt-oss-20b | 30 | 1K | 8K | 200K |
| openai/gpt-oss-safeguard-20b | 3 | 1K | 2K | 200K |
| qwen/qwen3.8-27b | 30 | 1K | 8K | 200K |

## Speech to Text

| Model | Requests per Minute | Requests per Day | Audio Seconds per Hour | Audio Seconds per Day |
|---|---|---|---|---|
| whisper-large-v3 | 20 | 2K | 7.2K | 28.8K |
| whisper-large-v3-turbo | 20 | 2K | 7.2K | 28.8K |

## Text to Speech

| Model | Requests per Minute | Requests per Day | Tokens per Minute | Tokens per Day |
|---|---|---|---|---|
| canopylabs/orpheus-arabic-saudi | 10 | 100 | 1.2K | 3.6K |
| canopylabs/orpheus-v1-english | 10 | 100 | 1.2K | 3.6K |

## Cross-check against measurements (`results/probe_raw.jsonl`)

| Model | Limit | Console | Measured | How measured | Agree? |
|---|---|---|---|---|---|
| gpt-oss-20b | RPM | 30 | 30 | first 429 at request #31 in a back-to-back burst | yes |
| gpt-oss-20b, gpt-oss-120b, qwen3.8-27b | RPD | 1K | 1,000 | `x-ratelimit-limit-requests` | yes |
| gpt-oss-20b, gpt-oss-120b, qwen3.8-27b | TPM | 8K | 8,000 | `x-ratelimit-limit-tokens`; TPM 429 body "Limit 8000" | yes |
| allam-2-7b | RPD / TPM | 7K / 6K | 7,000 / 6,000 | headers | yes |
| all | TPD | 200K (500K allam) | not measured yet | daily test on qwen, night of 2026-11-09 | — |
