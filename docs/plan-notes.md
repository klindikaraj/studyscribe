# Step 3 plan: study notes

Agreed plan for step 3 of the roadmap (approved 2026-11-03; local provider added 2026-11-09;
piece 2 revised 2026-11-16 after the correction-pass probe). Update it when a decision changes.

## Goal

Turn a session's transcript into study notes with an LLM (local by default, cloud optional):
validated JSON, rendered to
`exports/notes.md` and a self-contained `exports/notes.html` with an audio player and clickable
timestamps. Roadmap: "every timestamp in the notes points to the right audio position; no
invented references". The acceptance test uses two copies of real sessions:

- `%LOCALAPPDATA%\StudyScribe-tests\3h-copy`, session `20261030-134407-4ba5ed` (2:57:34,
  English, freeCodeCamp "Learn JavaScript - Full Course for Beginners");
- `%LOCALAPPDATA%\StudyScribe-tests\it-copy`, session `20261102-173400-e04c56` (0:19:59,
  Italian, a PyTorch tutorial with "TEC" for `detach`).

The originals are never touched.

## Decisions

Agreed with the user on 2026-11-03; 1, 5, 6 and 9 changed on 2026-11-09 (local provider).

1. Providers behind one interface, three backends:
   - **Ollama (default)**: a local model on the RTX 3060, through Ollama's native API. Free, no
     quota, the transcript never leaves the PC. Details in "Local provider (Ollama)".
   - Google Gemini (optional). The free tier failed every generation in piece 1 (503/504), so it
     is no longer the default; requests are still paced and retried with backoff.
   - Anthropic Claude (optional): Haiku 5.5 and Sonnet 5.5, once the account has credit.
   Default: ollama / `qwen3.5:9b`, thinking off (confirmed by the user on 2026-11-09).
   `gemma4:12b-it-qat` is pulled only later, for the comparison in the acceptance test. An
   OpenAI-compatible backend (OpenRouter, FreeLLMAPI) is possible later; it is not part of step 3.
2. Notes are written in the transcript's `language`; `--notes-language` overrides it. Section
   headings are localised in the template (en, it).
3. Default template "Studio" with seven sections: Summary (5-8 sentences), Outline (topics with
   their start time), Key concepts, Definitions (faithful to the lecturer's wording), Examples and
   code, Review questions (with answers), Unclear points. Templates are data (JSON), not code.
4. Output in `<session>/exports/`: `notes.md` and `notes.html`. The HTML page works offline when
   opened from disk, plays `audio.flac` (generated as `export --audio` does when missing), every
   timestamp seeks the player and starts it, and answers to review questions stay hidden until
   opened.
5. Commands: `studyscribe notes SESSION_ID [--provider] [--model] [--notes-language]
   [--template]` and `transcribe --notes` for the whole chain. Before sending, the tokens are
   estimated. Cloud providers: cost estimate, and above a configurable threshold (default $0.50)
   the CLI asks for confirmation; on the free tier the cost is $0 but the token estimate is still
   shown. Ollama: cost 0, tokens and estimated time instead, never a confirmation. At the end,
   tokens, time and cost are printed and stored.
6. Privacy: only text goes to the provider: the transcript, the session title, the course name
   (they help with technical terms) and the instructions. Never the audio. With Ollama on
   localhost nothing leaves the PC; if `OLLAMA_HOST` points to another machine the CLI says so.
   When the Gemini free tier is used, the CLI prints: "Gemini free tier: Google may use this
   transcript to improve its products." (The user accepted the free tier for the test lectures,
   which are public videos.)
7. Regeneration keeps versions: every generation is kept (each one is paid for); the exports
   show the latest completed one, and `export --notes-version N` writes `notes-vN.md/.html`.
8. API keys in the Windows Credential Manager through `keyring`, entered with
   `studyscribe config set-key gemini|anthropic` (no echo). Environment variables as fallback.
   Never in repo files, never in logs or error messages.
9. Effort: `medium` by default for the cloud providers (thinking level for Gemini, `effort` for
   Claude). Ollama: new value `none` (thinking off) by default; `low|medium|high` turn thinking
   on where the model supports it (checked from the model's capabilities). The acceptance test
   also runs `qwen3.5:9b` with thinking on, since it may help on the long summaries (reduce).

## Findings before the plan (verified 2026-11-03)

Checked against the official Anthropic and Google docs and PyPI. Prices are written in the code
with this date and must be checked again before they are trusted.

**Anthropic** (platform.claude.com: pricing, rate limits, structured outputs)

| Model | ID | Input / output per MTok |
|---|---|---|
| Haiku 5.5 | `claude-haiku-5-5` | $0.10 / $0.50 for prompts up to 100,000 tokens; $0.50 / $2.50 above (all input tokens count, cache reads and writes included) |
| Sonnet 5.5 | `claude-sonnet-5-5` | $2 / $10 at any length |

- No date suffix in the IDs; 1M context, 128K output.
- Haiku 5.5: adaptive thinking on by default (effort `medium`); `budget_tokens`, sampling
  parameters (`temperature`, `top_p`, `top_k`) and assistant prefill are rejected with a 400. Its
  tokenizer produces about 30% more tokens than older Claude models. Refusals have no server-side
  fallback.
- Sonnet 5.5: effort `high` by default; `thinking: {type: "disabled"}` and forced `tool_choice`
  are rejected with a 400.
- Thinking tokens are billed as output.
- Structured outputs are generally available (`output_config.format`; in Python
  `client.messages.parse(output_format=Model)` → `response.parsed_output`). Not supported:
  `minLength`, `maxLength`, `minimum`, `maximum`, `maxItems`, `minItems` above 1, recursive
  schemas; `additionalProperties` must be false. The SDK moves unsupported constraints into
  descriptions and checks them client-side.
- Rate limits, Start tier: 4,000 RPM, 2,000,000 input tokens/min for both models. A 429 carries
  `retry-after`; an exhausted spend cap returns 429 with `enforced_spend_limit_reached` and no
  `retry-after`.
- Token counting: `client.messages.count_tokens(...)` (needs a key; never estimate Claude tokens
  with another tokenizer).
- SDK `anthropic`: 2.0.0 (2026-10-21) moved to API version `2026-10-21`; latest 2.4.0
  (2026-11-02). Built on `httpx2` (2.16.0 already in the venv). The SDK retries 408/409/429/5xx
  by itself (`max_retries=2`); the app turns that off and retries itself.

**Gemini** (aistudio.google.com/docs, where ai.google.dev now redirects)

| Model | Free tier | Paid input / output per MTok |
|---|---|---|
| `gemini-3.8-flash` (stable, 1M in / 65k out, thinking low/medium/high, structured output) | yes | $0.75 / $3.75; doubles to $1.50 / $7.50 from 2027-01-01 |
| `gemini-3.5-flash-lite` (stable) | yes | $0.30 / $2.50 |

- Free tier: content is used to improve Google products (not on the paid tier).
- Free-tier limits are not published in the docs, only in AI Studio for the project. The user's
  project shows for `gemini-3.8-flash`: 5 RPM, 250,000 TPM, 20 requests per day. Limits apply
  per project, not per key; the daily count resets at midnight Pacific time (09:00 in Italy).
- SDK `google-genai` 2.34.0 (2026-10-29), Python >=3.11. Structured output through
  `response_mime_type="application/json"` plus a JSON schema; supports `minItems`/`maxItems`,
  strips `minLength`/`maxLength`. The docs' examples now use a new `client.interactions` API;
  `client.models.generate_content` is assumed to still work in 2.x (checked in piece 1).

**The two test transcripts** (kept segments of the done run)

| Session | Segments | Speech | Words | Characters |
|---|---|---|---|---|
| 3h-copy (en) | 1415 | 2:45:58 | 24,673 | 126,356 |
| it-copy (it) | 241 | 0:17:28 | 2,922 | 18,381 |

## Local provider (Ollama)

Added 2026-11-09. Checked on docs.ollama.com (structured outputs, OpenAI compatibility, FAQ,
`/api/chat`, `/api/ps`, Windows), on ollama.com/library (tags and sizes) and on the Hugging Face
model cards. Ollama is not installed yet: everything marked *(probe)* is checked by the probe
once it is.

### Native API, not the OpenAI-compatible one

`providers/ollama.py` talks to Ollama's native API (`POST /api/chat`), for these reasons:

| | Native `/api/chat` | OpenAI-compatible `/v1/chat/completions` |
|---|---|---|
| JSON schema output | `format: <JSON schema>`, documented with a pydantic example | `response_format`: the page confirms only JSON mode, not `json_schema` |
| Context window | `options.num_ctx` per request | "The OpenAI API does not have a way of setting the context size": only a custom Modelfile |
| Unloading the model | `keep_alive` per request (`0` = unload right after) | not documented |
| Usage and timing | `prompt_eval_count`, `prompt_eval_cached_count`, `eval_count`, `load_duration`, `prompt_eval_duration`, `eval_duration`, `done_reason` | usage only, with `stream_options` |
| Thinking | `think: false/true/"low"…` | `reasoning_effort` |

The context size alone decides it: through the OpenAI-compatible endpoint the app could not set
it, and Ollama's default is 4096 tokens. An `openai_compatible` backend stays useful for
OpenRouter or FreeLLMAPI later, but it would not be the right way to talk to Ollama.

No new package: plain HTTP with the standard library (`http.client`), streaming NDJSON. Streaming
gives a live progress line (tokens so far, tokens/s) and keeps Ctrl+C responsive during long
generations. The docs list timing and token counts only for the non-streamed response; the
final streamed chunk carries them too (verified with 0.40.15).

The model's output is still validated with pydantic as for the cloud providers: the docs do not
say whether the schema is enforced by constrained decoding, and they recommend repeating the
schema in the prompt, which the map and reduce prompts will do.

### Candidate models (checked on ollama.com/library, 2026-11-09)

The VRAM budget: 12,288 MiB, of which the Windows desktop already uses about 1.3 GB (measured
with `nvidia-smi`), so about 10.5 GB for weights, KV cache and buffers.

| | `qwen3.5:9b` (recommended) | `gemma4:12b-it-qat` | `qwen3.5:4b` |
|---|---|---|---|
| Download | 6.6 GB (Q4_K_M) | 7.2 GB (QAT 4-bit) | 3.3 GB (Q4_K_M) |
| Architecture | dense 9B, hybrid: 24 Gated DeltaNet layers + 8 attention layers (4 KV heads, dim 256) | dense 12B, 48 layers, sliding window 1024 interleaved with global layers | same family as 9B *(layout not checked)* |
| KV cache, f16 | 32 KiB/token (only 8 layers keep a KV cache): 1 GiB at 32k, 2 GiB at 64k | not computable from the card (KV heads and global-layer ratio not published) | smaller than 9B |
| Estimated VRAM at 32k | ~8.5 GB | ~10-11 GB (tight) | ~5 GB |
| Instructions | IFEval 91.5, IFBench 64.5 (Qwen card) | IFBench 74 (Gemma card); IFEval not published | lower *(not checked)* |
| Multilingual | 201 languages; MMMLU 81.2, MMLU-ProX 76.3 | 35+ out of the box, 140+ pre-trained; MMMLU 83.4 | 201 languages |
| Native context | 262,144 | 256K | 256K |
| Thinking | on by default, `think: false` turns it off *(probe)* | off unless enabled; with thinking off the 12B still emits an empty thought block | as 9B |

- **Recommended: `qwen3.5:9b`.** Best published instruction-following score of the three, and
  its hybrid attention keeps the KV cache small, so a 32k context fits with room to spare.
- `gemma4:12b-it-qat` as the second model for the A/B test: possibly better Italian, but the
  VRAM margin is thin and its KV size could not be computed. If it does not fit entirely on
  the GPU at 32k, the test uses 16k.
- `qwen3.5:4b` as the fallback when VRAM is short (another app using the GPU) and for quick
  tests; lower quality expected.
- Not chosen: `granite4.2:8b` (Italian supported and JSON output advertised, but dense with
  128K context: KV cache unknown, probably several GB at 32k), `ministral-3:14b` (9.1 GB, no
  room for the context), the 26-35B MoE models (16-22 GB).
- Unverified: VRAM figures are estimates (weights + KV + about 1 GB of buffers); the vision
  weights may be loaded as well. The probe measures them with `/api/ps` and `nvidia-smi`.

### Context window

- Every request sets `options.num_ctx` explicitly. Default 32768, setting `ollama.num_ctx`. It
  stays the same for the whole run: a different `num_ctx` makes Ollama reload the model.
- Budget per request: `num_ctx − num_predict − 512` tokens of margin. `num_predict` (output
  cap): 6144 for map, 8192 for reduce with thinking off; 16384 for both with thinking on (a
  15-minute block went past 6144 while still thinking).
- Map blocks (10-15 min) are about 2.5-3.6k tokens of transcript plus about 2k of prompt: far
  inside the budget.
- **Hierarchical reduce.** The single reduce on the 3 h lecture would take about 35-38k tokens
  of map outputs, more than the 32k budget. `plan_reduce` (pure) groups consecutive map outputs
  so that every group fits the budget and the groups are balanced; each group is reduced into
  one partial note with the map schema (refs kept, duplicates merged), and the final reduce
  works on the partials. More levels if needed. Since small models also lose quality on long
  inputs, the reduce input is capped at `ollama.reduce_max_input` tokens (default 16k) even when
  the context is larger. Requests:
  - 3 h lecture: 14 map requests, then 3 partial reduces (groups of 5, 5 and 4 map outputs,
    about 12k tokens each), then 1 final reduce on the 3 partials: 18 requests in all.
  - 20 min lecture: 2 map requests and 1 final reduce (no partials): 3 requests.
  - The cloud providers keep a single reduce (their limit is the 90k guard above).
- **No silent truncation.** Ollama's behaviour on a prompt longer than `num_ctx` is not
  documented; several reports say it drops the start of the prompt without an error. Three
  guards:
  1. Before sending: estimated input tokens = characters / `chars_per_token_min` for the model
     (default 2.5, conservative; then the lowest ratio measured on this model's past calls
     × 0.9). Over the budget → `ContextTooLong`, nothing sent. The pipeline sizes blocks and
     reduce groups from the same budget, so this only fires on a bug.
  2. After the response: `prompt_eval_count + eval_count ≥ num_ctx` (`prompt_eval_count`
     already includes the cached tokens), or input tokens below characters / 8 (implausibly
     few) → the response is discarded as possibly truncated, with an error.
  3. Every request carries `"truncate": false` at the top level: Ollama then refuses a prompt
     longer than the context with a 400 instead of cutting it (verified, see findings).

### GPU memory

- Whisper (~2.3 GB) and the LLM are never in VRAM together. In `transcribe --notes` the
  transcriber gets a `close()` that drops the CTranslate2 model and runs `gc.collect()` before
  the notes start *(probe: VRAM back to the baseline in `nvidia-smi`)*. The notes then run in
  the same process, which no longer holds any GPU memory.
- Ollama does not fail when a model does not fit: it splits the layers between GPU and CPU and
  becomes many times slower. After loading (a request with no messages loads the model), the
  provider reads `/api/ps`: if `size_vram < size`, it stops with: "qwen3.5:9b fits only partly in
  GPU memory (6.1 of 8.4 GB). Close apps that use the GPU (games, video editors, browsers with
  hardware acceleration), or use a smaller model (`--model qwen3.5:4b`) or a smaller context
  (`config set ollama.num_ctx 16384`)." `--allow-cpu` proceeds anyway. `/api/ps` also reports the
  `context_length` actually used, which must equal `num_ctx`.
- Before loading, free VRAM from `nvidia-smi` (installed with the driver; skipped if missing) is
  compared with the model's size from `/api/ps` of the last run or its download size + KV
  estimate: if clearly too small, the same message before waiting for the load.
- `keep_alive`: Ollama keeps a model loaded 5 minutes after the last request by default. During
  a run every request sends `keep_alive: "10m"` (no reload between blocks); at the end, on error
  and on Ctrl+C, a `finally` sends `keep_alive: 0` with no messages, which unloads it at once,
  so the VRAM is free again for Whisper or other apps *(probe: `/api/ps` empty afterwards)*.
- Ollama runs out of process, so a crash of StudyScribe leaves the model loaded for at most the
  10 minutes of `keep_alive`.

### Sampling and determinism

- Thinking off (`think: false`), fixed `seed` (stored in `options_json`), sampling from a
  small per-model table with the values recommended by the model card for non-thinking use.
  Qwen3.5: temperature 0.7, top_p 0.8, top_k 20, presence_penalty 1.5. Gemma 4: temperature 1.0,
  top_p 0.95, top_k 64.
- Not temperature 0: Qwen's cards warn that greedy decoding causes endless repetitions (stated
  for Qwen3; assumed for 3.5). With a fixed seed and the same input, Ollama is expected to give
  the same output on the same machine *(probe: two identical requests)*. If the probe shows
  repetition loops anyway, the table changes, not the code.
- `presence_penalty` is not documented for these models, but Ollama accepts it without an
  "invalid option" warning (verified); whether it has an effect is not measured.

### Estimates and speed (to be measured)

- Cost 0; the CLI shows requests, estimated input/output tokens and estimated time.
- Time = prefill tokens / prefill speed + output tokens / generation speed. Speeds come from
  the model's past calls in `llm_calls` (median); before any run, a default per model.
- Expectation for `qwen3.5:9b` on the 3060 (360 GB/s of memory bandwidth, ~6 GB read per
  token): about 35-50 tokens/s generated, 1,000-3,000 tokens/s of prefill. 3 h lecture: about
  70k tokens of prefill (under a minute) and 40-50k generated (map ~2.5k per block, partial
  and final reduces ~3-8k each): **about 15-25 minutes**. 20 min lecture: about 3 minutes.
  These are guesses until the probe measures them.

### Errors (each with a remedy, no traceback)

| Situation | Detected by | Message / remedy |
|---|---|---|
| Ollama not running | connection refused on `OLLAMA_HOST` (default `127.0.0.1:11434`) | "Ollama is not running. Start it from the Start menu, or run `ollama serve`." |
| Ollama not installed | same, and no `ollama.exe` on PATH or in `%LOCALAPPDATA%\Programs\Ollama` | "Ollama is not installed: https://ollama.com/download/windows" |
| Model not downloaded | `POST /api/show` → 404 | "Model qwen3.5:9b is not downloaded. Run: `ollama pull qwen3.5:9b` (6.6 GB)." |
| Ollama too old for the model | load error mentioning the architecture | "Update Ollama (installed: 0.x; the model needs a newer one)." |
| Partly on CPU | `/api/ps`: `size_vram < size` | see "GPU memory" |
| Out of GPU memory | 500 from the load with a CUDA/memory error | same remedies as above |
| Input too long | guard 1 or 2 | bug report text: the request is not sent, or its response is discarded |
| Output cut off | `done_reason: "length"` | `Truncated`: one retry with a larger `num_predict` if the context allows, else stop |
| Invalid JSON / schema | pydantic | the pipeline's retry with the validation error (2 at most) |
| Stalled stream | no chunk for 120 s | `NetworkError` (retried) |

### Probe (`scripts/llm_probe.py --provider ollama`)

Version, model check (`/api/show`: capabilities, quantisation, context length), the three-field
JSON request, then on `--session`: the whole map prompt of one 15-minute block of the 3 h
lecture. Prints: valid JSON or not, input/output tokens, chars/token, load time, prefill and
generation tokens/s, total time, `size`/`size_vram`/`context_length` from `/api/ps`, VRAM from
`nvidia-smi` before, during and after, whether the final `keep_alive: 0` unloaded the model,
whether the streamed final chunk has the counts, whether `truncate: false` is accepted, and
whether two identical seeded requests give the same text.

## Pieces (one at a time, stop after each with a test summary)

1. Provider layer: interface, Gemini and Claude backends, retry and pacing, keys,
   `config set-key/show/delete-key`, `scripts/llm_probe.py`. Probe: one tiny request with a
   three-field schema per provider (model, latency, tokens, cost) and an exact token count of
   the two transcripts. Unit tests: backoff with a fake clock, error mapping, key lookup order,
   key redaction. Anthropic only once the user has a key, and only after asking. **Done.**
1b. Ollama provider (`providers/ollama.py`, `gpu.py`), `config show` with Ollama's state, the
   probe above. Unit tests with a fake HTTP server (local socket, no Ollama): error mapping,
   context guards, `/api/ps` offload check, unload in `finally`. Only after the user has
   installed Ollama and pulled the model.
2. Pipeline (revised 2026-11-16, see "Piece 2 in detail"): transcript lines, uncertain words,
   blocks, templates, schema from the template, prompts, reference checks, code-name check,
   reduce plan, estimate, map / reduce with validation and retries, resume from stored map
   outputs. No database, no network: all tests offline with a fake LLM (`FakeLlm`).
3. Database v3 and CLI: `notes`, `transcribe --notes`, estimate and confirmation, recovery,
   `template list/import`, NOTES column in `sessions`.
4. Export: `notes.md`, `notes.html` with player and clickable timestamps, `audio.flac` when
   missing.
5. Acceptance: notes on the 3 h lecture and on the Italian lecture with `qwen3.5:9b` (thinking
   off and on) and `gemma4:12b-it-qat` (quality, time, VRAM from `llm_calls` and the probe);
   cloud models only
   if the user wants them later (Gemini when it answers, Claude once the account has credit,
   after asking). A checklist in
   `docs/manual-test-notes.md`: timestamps to click, definitions to compare with the audio,
   "TEC" (S186, 0:14:08) listed in "Words to check by listening", and no invented method name
   for it in the notes; code items with a warning checked against the audio.

### Piece 2 in detail (agreed 2026-11-16)

Changes from the first plan, after the correction-pass probe (see "Findings so far"):

- **No automatic corrections of recognition errors in step 3** (known limit). No correction
  pass, no `corrections` field in the schema, no glossary in the map prompt. The map prompt
  says: keep the transcript's words; when a passage looks misrecognised, add an open point.
- **Code form stays** (map prompt v4 without the example): code read aloud is written as code,
  multi-word names joined, spoken punctuation as symbols.
- **Code items are never dropped for their names.** A name written in code form often does not
  appear literally in the transcript ("ipsilon pred" → `y_pred`), so the check only warns, in
  two grades, and counts. Dropped is only what is certainly invalid: references that do not
  exist or fall outside the block (and items left without one), and output that is still not
  valid JSON for the schema after the retries.
- **Uncertain words**: a pure module lists the words Whisper was unsure about, with no LLM and
  no network, for an appendix "Words to check by listening" (rendered in piece 4).

Modules, one commit each, in this order:

1. `transcript.py`: `Line` (S-number, segment id, start/end ms, text, Whisper's words) and the
   prompt's line format (`S<n> text`, an `[H:MM:SS]` marker every minute), moved out of the
   probe. Lines are numbered from 1 over the kept segments, in time order.
2. `uncertain.py`: words with probability < 0.4, at least 3 letters, said at most twice in the
   lecture (counted on Whisper's words, punctuation removed, case ignored): word, S-number,
   timestamp of the word, probability. Same rule as the probe; 3 words in the Italian lecture,
   11 in the English one.
3. `blocks.py`: blocks of 10-15 minutes cut at the longest gap between two lines whose cut
   falls in that range; a last block under 5 minutes joins the previous one; a lecture of at
   most 15 minutes is one block.
4. `templates.py` + `resources/templates/studio.json`: template model (pydantic), loading of the
   built-ins and of a user JSON file, targets per section from the speech duration (items per
   hour, clamped to a minimum and a maximum). The database table and `template import` stay in
   piece 3.
5. `schema.py`: pydantic models per section kind, the output model built from a template. Map,
   partial reduce and final reduce use the same model; only the targets in the prompt differ.
6. `prompts.py` + `resources/prompts/map.md`, `code-rule.md`, `reduce.md`: `PROMPT_VERSION`, a hash of the prompt
   files, rendering of the map, partial and final prompts.
7. `refs.py`: reference checks (inside the block for map, among the input's references for the
   reduces), drops and counts, the 25% rule, S-numbers inside text, resolution `S<n>` →
   segment id and `start_ms`.
8. `code_check.py`: graded check of code names against the cited lines ±1 (see "Code form").
9. `reduce_plan.py`: groups of consecutive outputs for the hierarchical reduce.
10. `estimate.py`: requests, input and output tokens, time (Ollama) or cost (cloud).
11. `pipeline.py` + `FakeLlm` in `tests/fakes.py`: map, partial and final reduces, validation
   retries, `Truncated` handling, a record of every call for the runner, resume from stored
   map outputs.

## Dependencies

Installed only after the user confirms. All four in the main dependencies, no extra.

| Package | Constraint | Resolves today to | Why |
|---|---|---|---|
| anthropic | `>=2.3,<3` | 2.4.0 (2026-11-02) | Claude backend; 2.x speaks API version 2026-10-21 |
| google-genai | `>=2.33,<3` | 2.34.0 (2026-10-29) | Gemini backend |
| keyring | `>=25.7,<26` | 25.7.0 | API keys in the Windows Credential Manager |
| pydantic | `>=2.14,<3` | 2.14.2 | output schemas and validation (both SDKs depend on it anyway) |

Ollama needs no Python package (standard-library HTTP). Ollama itself is a separate program the
user installs (`OllamaSetup.exe` from ollama.com, no administrator rights, NVIDIA driver 551.61 or
newer: 617.42 installed); models go to `%USERPROFILE%\.ollama\models`.

## Modules

```
src/studyscribe/notes/
  providers/base.py     LlmProvider protocol, LlmRequest / LlmResponse / Usage, LlmError
                        subclasses with a remedy, key redaction
  providers/gemini.py   GeminiProvider (only module importing google.genai)
  providers/claude.py   ClaudeProvider (only module importing anthropic; streaming +
                        get_final_message)
  providers/ollama.py   OllamaProvider: native /api/chat (stdlib HTTP, NDJSON stream), model
                        check, context guards, /api/ps offload check, load/unload, sampling table
  providers/retry.py    backoff and pacing (pure, injectable clock)
  gpu.py                free VRAM from nvidia-smi (optional; None when unavailable)
  pricing.py            price table with its verification date, cost from usage (Haiku's
                        100k threshold)
  keys.py               keyring + environment variables (only module importing keyring)
  transcript.py         Line, the prompt's line format (pure)
  uncertain.py          words Whisper was unsure about, for the appendix (pure)
  templates.py          template model, loading, targets (built-in sync to the db: piece 3)
  resources/templates/studio.json   (not `data/`: .gitignore ignores every `data/`)
  schema.py             pydantic models per section kind, output model built from a template
  prompts.py            PROMPT_VERSION, prompt rendering, hash of the prompt files
  resources/prompts/map.md, code-rule.md, reduce.md
  blocks.py             block cuts in pauses (pure)
  reduce_plan.py        grouping of map outputs for the hierarchical reduce (pure)
  refs.py               reference validation and resolution (pure)
  code_check.py         graded check of code names against the cited lines (pure)
  estimate.py           token and cost estimate (pure)
  pipeline.py           map / reduce, validation, retries (no database)
  runner.py             generate_notes: checks, lock, lifecycle, resume, usage rows (database)
  export.py             notes.md, notes.html
scripts/llm_probe.py    piece 1 probe
docs/manual-test-notes.md
```

Changed: `pyproject.toml`, `db/migrations.py` (V3), `db/repo.py`, `session.py` (lock byte 3),
`recovery.py`, `cli.py`, `tests/fakes.py` (FakeLlm), `transcribe/whisper.py` (`close()`),
`README.md` and `CLAUDE.md` at the end.

## Flow

1. Checks: the session exists and has a done transcript; at least 5 minutes of speech (sum of
   `end_ms - start_ms` of kept segments); the transcript has a language or `--notes-language` is
   given; a key exists for the provider (cloud), or Ollama runs and has the model (local).
2. Blocks: 10 to 15 minutes each, cut at the longest gap between two segments in that range; a
   last block under 5 minutes joins the previous one. 15 blocks for the 3 h lecture (one of
   5.6 minutes, cut at the 5-minute pause), 2 for the 20-minute one (measured 2026-11-16).
3. Estimate: requests, input and output tokens; cloud: cost (charged and list price) and
   requests against the per-minute and per-day limits, confirmation above the threshold (with no
   terminal and no `--yes`, nothing is sent); Ollama: estimated time, no confirmation, then the
   model is loaded and the GPU check runs before the first block.
4. Map, sequential: one request per block with the title, course and language. References must
   fall inside the block. (The glossary of corrections from earlier blocks was dropped on
   2026-11-16 with the corrections.)
5. Reduce: on the validated map outputs only (not on the transcript). It merges duplicates,
   joins their references and sizes each section to the lecture's length. One request when the
   map outputs fit the budget, otherwise partial reduces first (see "Context window").
6. Resolution: `S<n>` → segment id → `start_ms`; content, Markdown and stats stored; exports
   written.

### References

The prompt lists segments as `S<n> text`, where `n` is the segment's position in the transcript
(1…1415), with an `[H:MM:SS]` marker every minute. Database IDs would cost more tokens and change
on every re-transcription. The code maps `S<n>` to the segment id and its `start_ms`, and stores
`start_ms` in the notes, so they stay readable after a re-transcription. Links seek to
`floor(start_ms / 1000)` seconds: up to 1 s early, never cutting the start of the sentence.

### Validation

- Output parsed by pydantic; on failure, a new attempt with the validation error (2 at most).
- References that do not exist or fall outside the block are dropped; items left without a valid
  reference are dropped.
- If more than 25% of a response's references are invalid, the response is discarded and asked
  again.
- Every drop is counted, stored in `stats_json` and printed at the end.
- Nothing else is dropped: code items with names not in the transcript only get a warning (see
  "Code form and uncertain words").
- `Truncated` (output cap reached): one more attempt with twice the cap if the context allows
  it; otherwise, or if it happens again, the generation stops with the error.

### Code form and uncertain words (revised 2026-11-16)

The first plan had the map fix recognition errors and list them in a `corrections` field. The
probes showed that `qwen3.5:9b` cannot do it reliably, in the map or in a separate pass, with
thinking off or on (see "Findings so far"), so **automatic corrections are a known limit of
step 3**. The transcript stays faithful to the audio and so do the notes' words.

- Code form stays: the map writes code read aloud as code ("PyTorch.neuralNetwork" →
  `torch.nn`, "ipsilon pred" → `y_pred`). These acceptance cases never appear in a prompt, and
  the prompts have no example at all (the made-up example was copied into the notes).
- Code items whose text contains an S-number fail validation and are asked again.
- **Code-name check** (`code_check.py`), a warning only. Every name in a code item (string
  literals and common keywords left out) is split into its words (snake_case, camelCase, dots);
  a word of 3+ letters is *found* when a word of the cited lines ±1 contains it, is at least
  80% similar to it, or when it is the initials of consecutive words there ("mean squared
  error" → `MSE`); shorter words (`nn`, `y`, `lr`) are not checked. A name is found when all its
  words are; partly found when some are, or when its words are found only elsewhere in the
  lecture (added in the implementation: `y_pred` cited on "y previsti" while `YPRED` is said
  four lines later); missing when none is found anywhere. The item gets:
  - `ok`: every name found (`y_pred` from "ipsilon pred", `torch.nn.Linear` from "PyTorch
    neural network Linear");
  - `check` ("to verify"): a name only partly found;
  - `unsupported` ("name not in the transcript, to verify"): a name with no word found
    (`misure_notte` for "the two tensors").
  Both warnings keep the item, are counted in `stats_json` and rendered next to it (piece 4).
- **Words to check by listening** (`uncertain.py`): the words Whisper was unsure about
  (probability < 0.4, 3+ letters, at most twice in the lecture), with S-number, timestamp and
  probability. No LLM, no claim that they are wrong: an appendix with links to the audio.

### Retries and pacing

- Client-side pacing, configurable per provider (default for the Gemini free tier: 5 RPM).
- Exponential backoff on 429, 5xx, 529 and network errors, honouring `retry-after` / Gemini's
  `retryDelay`.
- Daily quota exhausted (recognised from the 429 body, not from the configured limit): stop with
  "resume after 09:00 (midnight Pacific)"; the work done stays saved.
- SDK retries are off, so all retries go through one place.

### Resume

Every LLM call is stored, failed ones included (they are paid for too). The same command with the
same options reuses the map blocks already done. Lock: byte 3 of `session.lock`. Recovery marks
generations of dead processes as `interrupted`, as it does for transcription runs.

### Keys

- Lookup order: Windows Credential Manager (`keyring`, service `studyscribe`), then
  `STUDYSCRIBE_GEMINI_API_KEY` / `STUDYSCRIBE_ANTHROPIC_API_KEY`, then `GEMINI_API_KEY` /
  `ANTHROPIC_API_KEY`.
- The key is always passed explicitly to the SDK client, so the SDK never picks up other
  credentials on its own.
- Every error message goes through a function that removes the key from the text.

## Database (migration v3)

```sql
CREATE TABLE settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
  -- default provider/model, confirmation threshold, gemini tier, rpm/rpd,
  -- ollama.num_ctx, ollama.reduce_max_input, ollama.keep_alive

CREATE TABLE templates (
  id TEXT PRIMARY KEY, title_json TEXT NOT NULL, description TEXT,
  sections_json TEXT NOT NULL, builtin INTEGER NOT NULL DEFAULT 0, updated_at TEXT NOT NULL
);  -- built-ins synced from the package JSON files; user templates via `template import`

CREATE TABLE generated_notes (
  id                INTEGER PRIMARY KEY,
  session_id        TEXT NOT NULL REFERENCES sessions(id),
  transcript_run_id INTEGER REFERENCES transcription_runs(id) ON DELETE SET NULL,
  status            TEXT NOT NULL CHECK (status IN ('running','interrupted','done','failed')),
  provider          TEXT NOT NULL,
  model             TEXT NOT NULL,
  template_id       TEXT NOT NULL,
  template_json     TEXT NOT NULL,   -- copy of the template used
  language          TEXT NOT NULL,
  language_source   TEXT NOT NULL CHECK (language_source IN ('transcript','forced')),
  prompt_version    TEXT NOT NULL,
  options_json      TEXT NOT NULL,   -- effort, blocks, targets: resume only when equal
  blocks_json       TEXT NOT NULL,
  content_json      TEXT,            -- validated notes, references resolved (segment id + start_ms)
  content_md        TEXT,
  stats_json        TEXT,            -- dropped references/items, code warnings, uncertain words
  started_at TEXT NOT NULL, finished_at TEXT, error TEXT
);
CREATE UNIQUE INDEX idx_notes_open ON generated_notes(session_id)
  WHERE status IN ('running','interrupted');

CREATE TABLE llm_calls (
  id INTEGER PRIMARY KEY,
  notes_id INTEGER NOT NULL REFERENCES generated_notes(id) ON DELETE CASCADE,
  stage TEXT NOT NULL CHECK (stage IN ('map','partial','reduce')),  -- partial: hierarchical reduce
  block INTEGER NOT NULL, attempt INTEGER NOT NULL,                  -- block = group for 'partial'
  status TEXT NOT NULL CHECK (status IN ('ok','invalid','error')),
  model_version TEXT, request_id TEXT,
  input_tokens INTEGER, output_tokens INTEGER, thinking_tokens INTEGER, cached_tokens INTEGER,
  cost_usd REAL,        -- charged (0 on the free tier)
  list_cost_usd REAL,   -- at list price
  duration_s REAL,
  load_s REAL, prefill_s REAL, generate_s REAL,  -- Ollama timings (speed estimates); NULL for cloud
  output_json TEXT, error TEXT, created_at TEXT NOT NULL
);
```

`transcript_run_id` uses `ON DELETE SET NULL`: a re-transcription deletes the previous run, the
notes stay readable and are flagged as based on an earlier transcript. The template is copied into
`template_json` rather than referenced, because a template can change after a generation.

## CLI

```
studyscribe config set-key gemini|anthropic   (getpass, no echo) | delete-key | show | set NAME VALUE
studyscribe notes SESSION_ID [--provider] [--model] [--notes-language] [--template studio]
                  [--effort none|low|medium|high] [--yes] [--restart] [--list] [--allow-cpu]
studyscribe transcribe SESSION_ID --notes [same options]
studyscribe export SESSION_ID [--audio] [--notes-version N]
studyscribe template list | import FILE.json
```

- Before sending: blocks, requests, estimated tokens (in/out), estimated cost (on the free tier
  "$0, $0.44 at list price"), requests against the limits; the free-tier privacy notice. Ollama:
  blocks, requests, tokens, estimated time, model and context, "local: nothing leaves this PC".
- `config show` also prints Ollama's state: reachable or not, version, models downloaded, the
  default model and whether it is present.
- At the end: tokens, cost, dropped references. `sessions` gets a NOTES column.
- Expected errors, each with a message and a remedy and no traceback: missing or invalid key,
  per-minute quota, daily quota, spend limit reached (`enforced_spend_limit_reached`, "usage
  limits"), network, overload (529), refusal (`stop_reason: refusal`) or Gemini block (SAFETY,
  RECITATION), truncated output (`max_tokens`). Exit codes as in the other commands: 0 ok, 1
  failure, 2 usage or unknown session.

## Prompts (draft, versioned)

`PROMPT_VERSION` plus a hash of the prompt files is stored with every generation.

Map (system):

```
You turn one part of a recorded lecture into study notes. The input is an automatic
transcript: lines "S<n> text" (speech recognition, it can contain errors), with [H:MM:SS]
markers. Lecture: "{title}", course: "{course}". Part {i} of {n} ({start}-{end}).

Rules
1. Use only what the lecturer says in this part. Never add facts, definitions, examples or
   code that are not in the transcript, even if you know them. If something is unclear,
   put it in open_points instead of guessing.
2. "ref" and "refs" hold the S-numbers of the lines an item comes from, the line where it
   starts first. Cite only S-numbers of this part. S-numbers go only in "ref" and "refs",
   never inside names, explanations or code.
3. Write in {language_name}. Keep technical terms as the lecturer uses them.
4. Code: every code element the lecturer reads aloud or names goes into "code", even a single
   method, written as in a source file (spoken punctuation as symbols, multi-word names joined
   the way the taught language does, quotes, parentheses with the arguments given). Only names
   in the transcript or their evident code form; never rebuild code from a description.
5. Recognition errors: keep the transcript's words. When a passage looks misrecognised, add an
   open point instead of guessing.
6. Definitions: keep the lecturer's wording; remove only spoken fillers.
Sections for this part:
{for each section: key - instructions - about {target} items}
```

The wording comes from the probe's prompt v4 (`BLOCK_SYSTEM_V4` in `scripts/llm_probe.py`,
which has no example), with rule 5 replaced as above (2026-11-16).

The user message holds the block's lines.

Reduce (system):

```
You merge the partial notes of consecutive parts of one lecture into its final study notes.
Rules
1. Use only the content of the partial notes; add nothing new.
2. Merge duplicates: keep the most complete wording, join their refs (earliest first).
3. Copy refs exactly; never write an S-number that is not in the input.
4. Write in {language_name}; keep code exactly as written in the partial notes.
Sections:
{summary: 5-8 sentences on the thread of the lecture, no refs}
{outline: about {n} topics in order, each with the ref where it starts}
{... other sections with whole-lecture targets}
```

### Templates

A template is `{id, title, description, sections}`; each section has `key`, `kind`, `heading`
per language, `instructions` and a number of items per hour of speech (with a minimum and a
maximum). Kinds: `summary`, `outline`, `concepts`, `definitions`, `examples`, `questions`,
`open_points`, `bullets`. The output schema is built from the template, so a new template needs
no code; a new kind needs its rendering. (Corrections are no longer part of the schema.) "Unclear
points" are points the lecturer leaves unexplained or defers, and passages where the transcript
looks wrong. Question difficulty is `easy|medium|hard`, translated when rendered.

## Exports

- `notes.md`: timestamps are links `notes.html#t=SEC`.
- `notes.html`: self-contained (inline CSS and JS, no external resources); `<audio>` on
  `audio.flac` in the same folder; clicking a timestamp sets `currentTime` and plays; `#t=SEC`
  in the URL seeks on load; answers in `<details>`. All LLM text is escaped; only backticks
  become `<code>` and examples `<pre>`.
- Written through a temporary file and a rename, like the transcript exports.

## Token and cost estimate

Assumptions: about 3.2 characters per token with the Claude tokenizer, about 4 with Gemini; output
= JSON plus thinking; 15% margin for retries. Piece 1 measures the exact input tokens with
`count_tokens`.

| | Requests | Input | Output | Haiku 5.5 | Sonnet 5.5 | 3.8 Flash paid (from 2027) | Free tier |
|---|---|---|---|---|---|---|---|
| 3 h | 14 map + 1 reduce | ~130k (map ~5.5k each, reduce ~38k) | ~90k | ~$0.06 | ~$1.15 (0.8-1.5) | ~$0.44 ($0.88) | $0 |
| it 20 min | 2 + 1 | ~19k | ~14k | ~$0.01 | ~$0.18 | ~$0.07 | $0 |

Measured in piece 1: the Gemini tokenizer gives 3.38 (en) and 3.51 (it) characters per token,
about 15% more tokens than assumed. Ollama (`qwen3.5:9b`, thinking off, hierarchical reduce):
3 h about 18 requests, ~70k tokens in, ~40-50k out, $0, about 15-25 minutes (unmeasured).

- Every request stays under 40k tokens, far from Haiku's 100k threshold. The code still refuses a
  request estimated above 90k (only lectures over about 9 hours would get there).
- Sonnet on the 3 h lecture is above the $0.50 threshold and asks for confirmation.
- The full acceptance test (Haiku and Sonnet on both lectures) costs about $1.40; it runs only
  after the user confirms.
- On the free tier the 3 h lecture takes about 15 requests out of 20 per day: the A/B test with
  Gemini may span two days.

## Tests

- Pure modules (blocks, refs, estimate, pricing, retry, schema, templates) with unit tests.
- Pipeline and runner with `FakeLlm` (scripted responses, invalid JSON, invented references,
  429s, daily quota, interruption after N calls): no network in `pytest`.
- Provider modules: error mapping tested with the SDKs' exception classes, no request sent.
  Ollama: a small fake HTTP server on a local socket (scripted NDJSON, 404, 500, refused
  connection, `/api/ps` with partial offload), no Ollama needed.
- Real requests only in `scripts/llm_probe.py` and in the acceptance test.

## Risks and unverified assumptions

- Gemini free tier: 20 requests per day; content used by Google (accepted for public test videos;
  a notice is printed).
- `generate_content` versus the new `interactions` API in google-genai 2.x: checked in piece 1.
- RECITATION blocks on Gemini: the test lecture is public, and definitions quoted almost verbatim
  may be blocked. If it happens: one retry asking for a paraphrase, and a warning.
- References that exist but point to the wrong place: validation cannot see them. The
  "inside the block" rule limits them; clicking them in the acceptance test checks the rest.
- Recognition errors stay in the notes (known limit since 2026-11-16): "TEC" is not turned into
  `detach`. The uncertain-words appendix points the reader to them; it misses confident errors
  ("PyTorch.neuralNetwork" has probability 0.82-0.87).
- Code-name check: tuned on a handful of probe outputs. An invented name close to a spoken word
  (`inputi` for "input") passes as `ok`; a right code form whose words were not said (an alias
  shown only on screen) gets a warning. It never drops anything, so a wrong grade costs a
  warning, not content.
- "Faithful to the lecturer": an LLM tends to paraphrase. Not checkable automatically; checked by
  listening.
- FLAC seeking in a page opened from disk (Edge, Chrome): tested in piece 4 (seek table, accuracy).
- Estimates: about ±30%; the real cost comes from the usage rows. Prices change (Gemini 3.8 Flash
  doubles in January 2027).
- Haiku 5.5 refusals: no server-side fallback; treated as an error with a message.
- `anthropic` 2.x is recent (2.0.0 on 2026-10-21, 2.4.0 on 2026-11-02).
- Local model quality: a 9B model at 4 bits writes weaker notes than the cloud models, and may
  follow "use only the transcript" less strictly. The validation catches invented references,
  not invented content; the acceptance test checks it by reading.
- Ollama on a prompt longer than `num_ctx`: undocumented, reported as silent truncation.
  Covered by the guards in "Context window".
- Ollama's schema enforcement is not documented as constrained decoding; small models can also
  loop until `num_predict`. Both end in validation errors or `Truncated`, never in silent bad
  notes.
- VRAM figures and speeds of the candidate models are estimates until the probe.
- Ollama moves fast (0.40.12 to 0.40.15 in a week): the probe prints the version, and a version
  that breaks the API shows up as a clear error, not as wrong notes.

## Findings so far

### Piece 1: provider layer (tested by the user, 2026-11-04 to 2026-11-09)

Installed: anthropic 2.4.0, google-genai 2.34.0, keyring 25.7.0, pydantic 2.14.2.

- `config set-key gemini` with a made-up key: "Not stored: Gemini rejected the API key." With the
  real key: stored, "checked: Gemini 3.8 Flash is available". `config show` prints the key
  masked, from the Windows Credential Manager.
- Exact token counts of the transcripts in the pipeline's line format (`llm_probe.py
  --count-only`, with the `gemini-3.8-flash` counter):

  | Session | Segments | Characters | Input tokens | Chars/token |
  |---|---|---|---|---|
  | 3h-copy (en) | 1415 | 135,479 | 40,141 | 3.38 |
  | it-copy (it) | 241 | 19,668 | 5,601 | 3.51 |

  The estimate above assumed about 4 characters per token for Gemini: the real figure is about
  15% more tokens than that.
- Generation probe with `gemini-3.8-flash` on the free tier: it has failed for days, every time
  after 5 retries, with a clean exit and a readable message: 2026-11-04 `503 UNAVAILABLE` ("high
  demand"), 2026-11-06 `504 DEADLINE_EXCEEDED`. `generate_content` reaches the model (the errors
  are server-side, not "unknown method"), but no successful structured response has been seen
  yet. **The free tier is not reliable enough to be the default.**
- Anthropic: key stored, but the account has no credit, so no request was sent.

Decision (2026-11-09): the main provider becomes a local LLM served by Ollama on the RTX 3060.
Free, no quota, and the transcript never leaves the PC (fine for paid courses too). Gemini and
Anthropic stay as optional providers. See "Local provider (Ollama)" above.

### Piece 1b: Ollama installed (by the user, 2026-11-09)

- `ollama --version` → 0.40.15.
- `ollama pull qwen3.5:9b` → success: two layers of 921 MB and 5.6 GB (assumed: the vision
  weights and the language model), sha256 verified.

Checked by hand with curl and a few small requests (2026-11-09):

- **Silent truncation confirmed.** `num_ctx` 2048 and a 4104-token prompt: HTTP 200, the log
  says `truncating input prompt limit=1026 prompt=4104`, `prompt_eval_count` 1026, and the
  answer is nonsense. With `"truncate": false` at the top level of the request: HTTP 400
  `request (4104 tokens) exceeds the available context size (2048 tokens), try increasing
  it`. Inside `options`, `truncate` is rejected as "invalid option". The 400 also gives an
  exact token count, which `count_tokens` uses for texts longer than the context.
- `num_ctx` below 2048 is raised to 2048 (`/api/ps` `context_length`).
- `prompt_eval_count` includes the cached tokens (2004 with 2000 cached).
- The last streamed chunk carries `done_reason` and every count and duration.
- A chat request with no messages and `keep_alive: 0` answers `done_reason: "unload"`; the
  model leaves `/api/ps` and the VRAM goes back to the baseline (1071 MiB).
- `/api/show` for qwen3.5:9b: family `qwen35`, Q4_K_M, capabilities completion, vision, tools,
  thinking; `thinking: {values: [false, true], default: true}` (on/off only, no levels);
  default parameters temperature 1, top_p 0.95, top_k 20, presence_penalty 1.5 (the
  thinking-mode values); `full_attention_interval` 4, 4 KV heads of 256: the KV estimate of 32
  KiB/token is confirmed.
- First load from disk 43 s; later loads 7 s.

First run of the probe (tiny request only, by Claude, 2026-11-09):

- At `num_ctx` 32768 Ollama allocates 6.58 GB, all of it in GPU memory; `nvidia-smi` goes from
  1,071 to 8,344 MiB in use (about 3.9 GB left).
- Tiny request: 89 input tokens, 27 output, generation 27.9 tokens/s (too short to measure the
  real speed); valid JSON; the same request with the same seed gave the identical answer.
- No "invalid option" in Ollama's log, so `presence_penalty`, `top_k` and the others are
  accepted.
- Quality signal: with thinking off the model listed `S1`, `S2` as "identifiers" instead of
  `loss.backward()` and the like. Watch for it in the block test.

Block probes by the user (2026-11-09; draft map prompt v1). The figures marked *(reported)*
come from the user's summary: the console output itself was not kept.

| | 3 h, 0:30-0:45, effort none | it, 0:05-0:20, effort none | it, 0:05-0:20, effort high |
|---|---|---|---|
| Result | valid JSON | valid JSON | `Truncated`: past 6144 output tokens while thinking, after 123 s *(reported)* |
| Time | 22-34 s per block *(reported)* | | |
| Generation | about 52 tokens/s *(reported)* | | |
| GPU | 8.3 GB in use, all of the model on the GPU, unloaded at the end *(reported)* | | |
| Refs outside the block | 0 of 42 (S243-S327, checked) | 0 of 48 (S62-S241, checked) | |
| Code examples | 8 | **0** | |
| Corrections | 0 | **0** | |

Checked by Claude in the saved JSON files (2026-11-10):

- Italian block: `detach` appears nowhere (not in concepts, definitions or open points);
  `code` and `corrections` are empty although the block names `PyTorch.neuralNetwork` →
  `Linear` (S88), `unsqueeze` with index 1 (S113, S115) and "il metodo di TEC" (S186).
  `unsqueeze` appears only as a plain word inside a concept. 6 topics, 10 concepts and 8
  definitions are otherwise faithful to the transcript.
- English block: the 8 code items are spoken words, not code: `my string is my name is + name`,
  `first name.length`, `last name[last name.length - 1]`, `word blanks(noun, adjective, verb,
  adverb)`: names not joined, strings without quotes.
- So with thinking off and prompt v1, the 9B model summarises well but does not write code in
  code form and makes no corrections.
- "thinking None" in the probe with effort high: the provider did not count thinking tokens
  (Ollama's `eval_count` covers thinking and answer together). Fixed: the thinking chunks are
  counted (about one token each; 29 content chunks for an `eval_count` of 33 in an earlier
  check, so the count is approximate).

Prompt iterations on the same two blocks (2026-11-10; v2 run by the user, v3 and v4 by Claude;
qwen3.5:9b, all on the GPU, 44-52 tokens/s, 0 refs outside the block in every run). Prompts in
`scripts/llm_probe.py`: v1 draft; v2 code and correction rules plus a made-up example; v3 = v2
plus targets per section (about 4 topics, 6 concepts, 4 definitions per 15 minutes) and "never
invent a name, code said in words is not code"; v4 = v3 without the example.

| Italian block 0:05-0:20 | v1 none | v2 none | v2 high | v3 none | v4 none |
|---|---|---|---|---|---|
| Time | | 36 s | 201 s (7,183 thinking tokens) | 31 s | 37 s |
| Concepts / definitions | 10 / 8 | 5 / 7 | 7 / 2 | 4 / 4 | 6 / 2 |
| Code items | 0 | 12 | 4 | 7 | 8 |
| Code items with names not in the cited lines | | 11 of 12 (`y_prediatti` for "y previsti", `inputi`, `output_preveduti`, `misure_notte`, `plt`) | 1 (`plt`) | 6 of 7 (`nn.MSELoss()`, `torch.optim.Adam(…, lr=0.01)` rebuilt from a description) | 0 of 8, but "loss function", `Adam`, `relu` listed as code |
| Corrections | 0 | 0 | "di TEC" → `.detach()` ✓ | 2, both invented: "bush" → "push" copied from the example, "pesciamo" → "peschiamo" (the transcript already says "peschiamo") | "TEC" → "tensor" ✗, "pescchiamo" → "peschiamo" (invented) |

| English block 0:30-0:45 | v1 none | v2 none | v3 none | v4 none |
|---|---|---|---|---|
| Time | | 26 s | 23 s | 28 s |
| Concepts / definitions | 6 / 0 | 0 / 5 | 4 / 3 | 4 / 3 |
| Code items | 8, spoken words (`first name.length`) | 18, code form (`firstName[firstName.length - 1]`), 1 prose | 7, all code form | 8, all code form |
| Corrections | 0 | 0 | 0 | 0 |

Reading:
- The code rules work in English (v2-v4). In Italian the 9B model with thinking off either
  invents names (v2), rebuilds code from descriptions (v3), or lists terms as code (v4).
- The made-up example is harmful on this model: it was copied into the notes (v3).
- Corrections with thinking off are unreliable: none, invented, or wrong ("TEC" → "tensor").
  Only thinking on found "TEC" → `detach`, at 5.5 times the time (3 h lecture: about 35-45
  minutes instead of 7-8 for the map).
- Two deterministic checks catch most of the bad output and go into the pipeline (piece 2):
  a correction is dropped when its heard text is not in the cited lines (±1), or when the
  written form is already there (both invented v3/v4 corrections fail this); code items whose
  names do not appear in the cited lines (±1), allowing for words joined in camelCase or
  snake_case and for listed corrections, are counted and flagged (11 of 12 in v2). Wrong but
  plausible corrections ("tensor") cannot be detected: the corrections appendix with its
  links lets the reader check them.
- The `identifiers: ["S1", "S2"]` answer of the tiny request did not change with a clearer
  field description; with thinking on, the same request ignored "at most eight words".

`gemma4:12b-it-qat` with prompt v4, effort none (Claude, 2026-11-10; pulled after the user
confirmed). `/api/show`: family `gemma4` (the sampling table's key is right), 11.9B Q4_0,
thinking on/off only, default sampling temperature 1, top_k 64, top_p 0.95 (= the table).
48 layers: 40 sliding-window layers (8 KV heads of 256, window 1024) and 8 global ones (1 KV
head of 512). The first KV estimate took the largest head count for every layer (about 24 GiB
at 32k, so the pre-load check would have refused the model): fixed, now 16 KiB/token plus
320 MiB of windows = 0.81 GiB at 32k.

| | qwen3.5:9b v4 | gemma4:12b-it-qat v4 |
|---|---|---|
| Allocated / GPU in use after loading | 6.58 GB / 8.3-8.7 GB | 8.02 GB, all on the GPU / 10.4-10.8 GB (1.5 GB left) |
| Load | 7-8 s | 87 s cold, 17 s warm; once the load crashed: `CUDA error: shared object initialization failed`, exit 0xc0000409 in Ollama's `cuda_v13` runner (not out of memory); the next load worked |
| Generation | 44-45 tokens/s | 20-21 tokens/s (the first request after the cold load: 0.7 tokens/s) |
| Italian block | 37 s; 6 concepts, 2 definitions, 8 code items (0 flagged; "loss function", `Adam`, `relu` as code); "TEC" → "tensor" ✗ | 90 s; 14 concepts, 4 definitions, 10 code items, 7 flagged (`torch.manual_seed(42)`, `criterion(y_pred, y_real)`, `YPRED[..., 1]` rebuilt or invented); "TEC" → "slicing" ✗ |
| English block | 28 s; 4 concepts, 3 definitions, 8 code items, all right | 53 s; 4 concepts, 2 definitions, 6 code items, right but `lastName.length - 1` lost its brackets; "plus equals" → `+=` listed as a correction |

Reading: Gemma writes richer Italian concepts but invents more code, is 2.2 times slower,
leaves 1.5 GB of VRAM and crashed once on load. Neither model finds "TEC" → `detach` with
thinking off. qwen3.5:9b stays the default.

Correction pass, analysis only (2026-11-10). `words_json` holds Whisper's probability of
every word:

- "TEC" (S186) has probability 0.35, among the lowest 0.2% of the Italian lecture's words; its
  neighbours too ("di" 0.41, "per" 0.31).
- Lines with a word below 0.4: Italian 11 of 241 (5%), English 189 of 1415 (13%), mostly
  function words ("the", "at", "Okay"). Keeping only words of 3+ letters that occur at most
  twice in the lecture: Italian 3 lines (Adam, significa, TEC), English 11 (known, jack, team,
  Holland, left, stems, left, frozen, Adam, keyboard, indent). With 2 lines of context on each
  side: about 1.4k and 5k characters.
- Confident errors are not caught: "PyTorch.neuralNetwork" has 0.82-0.87 and "YPRED" 0.67.
  They are code-form problems, which the map's code rules handle (qwen v4 wrote
  `torch.nn.Linear`).
- The threshold is tuned on the only known error; how many real errors it misses is unknown.

Correction pass, probe (Claude, 2026-11-16; Ollama 0.40.16, qwen3.5:9b, all on the GPU). One
request per lecture, before the map: the candidate words (probability < 0.4, 3+ letters, at most
twice in the lecture; same lists as the analysis above) marked `[[word]]` in excerpts of ±2
lines, and for every one a `keep`/`fix` verdict with a reason (`llm_probe.py --corrections`).
Automatic checks: the verdict's line has a marked word, `heard` is that word, a fix differs from
it and is at most three words. The session titles in the test data say nothing about the topic
("Lezione italiano", "3 h test").

Expected verdicts, from the transcript's context (Claude's reading; not checked against the
audio):

| Lecture | Line | Word | Expected |
|---|---|---|---|
| it | S152 | Adam | right (the optimiser) |
| it | S173 | significa | right |
| it | S186 | TEC | error: `detach` ("stacchiamoli col metodo di TEC") |
| en | S69 | keyboard | error, certain: "keyword" (S70: "keyword var stands for variable") |
| en | S75 | left | error, certain: `let` ("So I can say let our name") |
| en | S103 | known | probably an error ("it's known as seven"), intended word uncertain: keep |
| en | S406 | stems | probably an error (the exercise array, likely "Stimpson"), not decidable: keep |
| en | S719 | Holland | probably an error (likely "Hole-in-one"), not decidable: keep |
| en | S759 | indent | right |
| en | S849 | team | error, almost certain: "statement" ("use the switch team") |
| en | S863 | jack | right (10, jack, queen, king) |
| en | S941 | Adam | right (the value returned for "alpha") |
| en | S1323 | left | error, certain: `let` (S1321: "use let instead of var") |
| en | S1365 | frozen | right ("Now I frozen math constants", after `Object.freeze`) |

So the English list is not "almost all right": 4 certain or almost certain errors, 3 probable
errors that the context cannot settle, 4 right words. A false positive is a fix of a right word
or a fix with a wrong written form.

| | it, effort none | it, effort high | en, effort none | en, effort high |
|---|---|---|---|---|
| Time | 3.9 s | 267 s | 18.2 s | about 5.5 min (from the output file's times), then `Truncated` |
| Input / output tokens | 932 / 169 | 930 / 13,499 | 2,066 / 881 | not printed / 16,384 (the cap) |
| Thinking tokens | 0 | 12,658 | 0 | 15,513, still thinking at the cap |
| Fixes proposed / passing the checks | 1 / 1 | 0 / 0 | 8 / 8 | no answer |
| Right fixes | 0 | 0 | 3: keyboard → keyword, left → let (twice) | |
| Wrong fixes (false positives) | 1: TEC → `drop` | 0 | 5: Adam → alpha, frozen → freeze (right words changed), stems → "j cat", Holland → "one", team → "teams" | |
| Kept | Adam, significa | Adam, significa, **TEC** ("likely a specific method acronym") | indent, jack, known (cited as S102: rejected by the check, so kept) | |

- Effort none is deterministic (the second run gave the same answers) but unreliable: 6 false
  positives out of 9 fixes in the two lectures, and "TEC" became `drop`.
- Effort high does not find `detach` either: on the Italian lecture it keeps everything (no
  false positive, no fix) after 12.7k thinking tokens for three words; on the English one it
  had not finished after 15.5k thinking tokens. The map with prompt v2 and thinking on had
  found "di TEC" → `.detach()` once (2026-11-10), with the whole 15-minute block as context: one
  run, not repeated, so it is unknown whether the wider context or chance made the difference.
- The automatic checks caught one malformed answer (a wrong line number) and none of the six
  wrong fixes: those are plausible words in the right place. Telling them apart needs the
  audio or a stronger model.
- The candidate selection alone is a useful signal: in English 7 of the 11 words are probably
  recognition errors; in Italian it contains "TEC".

Decision (per the rule agreed with the user on 2026-11-16): the correction pass fails with
effort none and with effort high, so automatic corrections of recognition errors stay out of
step 3 and are a known limit. Writing code read aloud in code form (map prompt v4) stays.

Changes after the first block results (2026-11-10): prompt v2 in the probe (explicit code and correction
rules, a made-up example that avoids the acceptance cases, S-numbers only in refs), an output
cap of 16384 with thinking on, `--expect WORD` to find a word in the notes, and a check for
S-numbers inside code and for text copied from the example. Defaults unchanged: ollama /
`qwen3.5:9b`, effort none.

### Piece 2: pipeline (Claude, 2026-11-16; offline, no model)

Modules `transcript`, `uncertain`, `blocks`, `templates`, `schema`, `prompts`, `refs`,
`code_check`, `reduce_plan`, `estimate`, `pipeline`, each with tests (`FakeLlm`, no network);
457 tests pass. Checked on the two test copies, read-only:

- Uncertain words: the same 3 (it) and 11 (en) words as the probe, with word timestamps.
- Blocks: 2 (it) and 15 (en). The largest map prompt is about 21k characters (about 8.4k tokens
  at the guard's 2.5 characters per token), far inside the 26,112-token budget at `num_ctx`
  32768.
- Estimate, thinking off: it 2 maps + 1 reduce, about 5 minutes; en 15 maps, 3 partial
  reduces and 1 final reduce, about 150k tokens in and 64k out, about 25 minutes. Not measured.
- Code-name check on the saved probe answers of 2026-11-10 (the `v4` files had been
  overwritten by the Gemma run): English, 28 of 31 items `ok` (3 `check`, a name cited on the
  wrong line); Italian, mostly warnings: `y_prediatti`, `optimizer`, `criterion` and the alias
  `plt` are `unsupported`, `y_pred` and `torch.manual_seed` are `check`, `nn.Linear` and
  `unsqueeze` are `ok`. In an Italian lecture an English name for a word said in Italian
  ("ottimizzatore") cannot be told from an invented one.
- Prompt v5 (map and reduce) has not been run on a model yet: v4's results do not carry over
  automatically (rules as bullets, section instructions from the template, the code section
  named `examples` instead of `code`).

### Map prompt v4 vs v5 (Claude, 2026-11-16)

qwen3.5:9b, effort none, Ollama 0.40.16, all on the GPU; the same lines as the probes of
2026-11-10 (en 0:30-0:45, S243-S327; it 0:05-0:20, S62-S241). v4 run again because its files
had been overwritten by the Gemma run; answers now in `%TEMP%\studyscribe-probe\`, one file per
session, prompt, model and effort. v4 uses the probe's schema (topics, concepts, definitions,
code, open points, corrections), v5 the pipeline's Studio schema (summary, outline, concepts,
definitions, examples, questions, open points). Code levels from `code_check.py` for both.

| | en v4 | en v5 | it v4 | it v5 |
|---|---|---|---|---|
| Time | 24.4 s | 18.6 s | 32.2 s | 30.3 s |
| Input / output tokens | 3,533 / 1,180 | 4,019 / 860 | 5,328 / 1,518 | 5,813 / 1,382 |
| Generation / prefill tokens/s | 52.3 / 1,899 | 52.1 / 1,887 | 51.7 / 1,903 | 50.9 / 1,882 |
| Valid JSON at the first attempt | yes | yes | yes | yes |
| Summary sentences | — | 2 (target 3) | — | 3 |
| Topics / outline | 4 | 3 (target 3) | 4 | 3 |
| Concepts | 4 | 4 (target 5) | 6 | 5 |
| Definitions | 3 | **0** (target "up to about 3") | 2 | 3 |
| Code items | 8 | 5 | 8 | 5 |
| Questions | — | 1 (target 2) | — | 2 |
| Open points | 0 | 0 | 0 | 0 |
| Code ok / check / unsupported | 8 / 0 / 0 | 5 / 0 / 0 | 8 / 0 / 0 | 1 / 2 / 2 |
| References / outside the block | 40 / 0 | 20 / 0 | 47 / 0 | 46 / 0 |
| Corrections (v4 schema only) | 0 | — | 2, both wrong | — |

English block:
- v5 stops at S291 (0:40:30): nothing from the last third of the block. The Mad Libs part
  (S301-S327: the game, `wordBlanks(noun, adjective, verb, adverb)`, `result += ...`) is in v4
  and missing from v5's outline, concepts and code.
- v5 has no definitions; v4 has 3, close to the wording ("strings are immutable, meaning they
  cannot be altered once created", S275; the Mad Libs game, S301-S302).
- Code: all 13 items of both versions are right and in code form (`firstName[firstName.length
  - 1]`, `myString += adjective`); v5 just has fewer. No prose written as code.

Italian block:
- v4 lists terms as code: `loss function` (S135), `Adam` (S151), `relu` (S223), `torch.Tensor`;
  the code check calls them `ok` because the words were said (it cannot tell a term from code).
  It still makes corrections, both wrong: "TEC" → "tensor" (S186) and "pescchiamo" →
  "peschiamo" (S88 already says "peschiamo"). The v5 schema has no corrections field.
- v4 writes `torch.nn.Linear` for "Da PyTorch.neuralNetwork peschiamo la classe Linear" (S88),
  an acceptance case, and `pyplot.scatter` (S65). v5 loses both.
- v5 rebuilds code from descriptions, the failure of prompt v3: `Linear(in_features=1,
  out_features=1)` (S98-S101 say only "quanti input... Solo 1"), `loss_function =
  MeanSquaredError()` (S139: "l'errore quadratico medio"; the class name is invented),
  `optimizer = Adam(params, lr=0.01)` (S158-S159 describe it in words), `for epoch in
  range(1000):` (S163: "un ciclo di mille ripetizioni"). The code check flags 4 of these 5
  (`unsupported` 2, `check` 2); only `tensor.unsqueeze(1)` (S113) is said.
- v5's concepts and definitions are good and richer (the seed, S123-S129; the mean squared
  error, S139; the learning rate, S156-S157); the definition term "Error quadratico medio" has
  a typo. Questions: 2, answerable from the cited lines.
- Neither finds `detach`, as expected.

Likely causes (not tested one by one):
1. Section lines: definitions (min 0 in Studio) are rendered "Up to about 3 items; empty if
   there are none", which the model took as optional: 0 in English. v4 asked "about 4
   definitions" in a rule.
2. Studio's examples instruction says "and worked examples": an invitation to build examples,
   which in Italian came out as code rebuilt from descriptions.
3. The code rule was compressed into one paragraph and lost v4's "If a name is not clear,
   leave the item out or add an open point".
4. Targets come from the speech (12-13 minutes) rather than the block's 15 minutes: one item
   fewer per section than v4.
5. Coverage: no rule asks to cover the whole part; v5 (with summary and questions to write)
   stopped early in English.

Estimates (point 4 of the request):
- The 22-34 s per block of 2026-11-09 are confirmed (18.6-32.2 s). "About 8-10 minutes for the
  3 h lecture" counted the map only: 15 blocks take about 6.7 minutes.
- `estimate.py` said about 25 minutes because of an assumption of mine with no measurement
  behind it (2,500 output tokens per block, commented as if measured; the real answers are
  860-1,518), a slower generation (45 instead of 51-52 tokens/s) and fixed reduce sizes.
  Corrected (commit 162c60b): every answer is sized from the template's targets at 65 tokens
  per item (measured 55-69), 50 tokens/s, 1,850 tokens/s of prefill, 3.7 characters per token.
  New estimate: it 2 maps + 1 reduce, about 2 minutes; en 15 maps (6.7 min), 2 partial reduces
  and the final one, about 19 minutes. Reduce requests are still not measured.
- **Risk found by the corrected estimate:** with Studio's whole-lecture maxima (30 topics, 40
  concepts, 30 definitions, 50 code items, 20 questions) the 3 h final reduce would write about
  12,000 tokens and each partial reduce about 9,300, above the 8,192 output cap. `Truncated`
  would ask again with 16,384, which does not fit the 32k context with a 19k-token input, so the
  generation would stop. To be decided before piece 3 (see the recommendation).
