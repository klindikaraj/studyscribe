# Step 3 plan: study notes

Agreed plan for step 3 of the roadmap (approved 2026-11-03). Update it when a decision changes.

## Goal

Turn a session's transcript into study notes with a cloud LLM: validated JSON, rendered to
`exports/notes.md` and a self-contained `exports/notes.html` with an audio player and clickable
timestamps. Roadmap: "every timestamp in the notes points to the right audio position; no
invented references". The acceptance test uses two copies of real sessions:

- `%LOCALAPPDATA%\StudyScribe-tests\3h-copy`, session `20261030-134407-4ba5ed` (2:57:34,
  English, freeCodeCamp "Learn JavaScript - Full Course for Beginners");
- `%LOCALAPPDATA%\StudyScribe-tests\it-copy`, session `20261102-173400-e04c56` (0:19:59,
  Italian, a PyTorch tutorial with "TEC" for `detach`).

The originals are never touched.

## Decisions

Agreed with the user on 2026-11-03.

1. Providers behind one interface, two backends:
   - Google Gemini, free tier for development and tests. The free tier is rate limited, so
     requests are paced and retried with backoff.
   - Anthropic Claude: Haiku 5.5 and Sonnet 5.5.
   The default model is chosen after an A/B test on the 3 h lecture (gemini-3.8-flash, Haiku 5.5,
   Sonnet 5.5; gemini-3.5-flash-lite only if Flash turns out poor). Until then the default is
   gemini / `gemini-3.8-flash`, effort `medium`.
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
   [--template]` and `transcribe --notes` for the whole chain. Before sending, the cost is
   estimated; above a configurable threshold (default $0.50) the CLI asks for confirmation. On
   the free tier the cost is $0 but the token estimate is still shown. At the end, tokens and
   cost are printed and stored.
6. Privacy: only text goes to the provider: the transcript, the session title, the course name
   (they help with technical terms) and the instructions. Never the audio. When the Gemini free
   tier is used, the CLI prints: "Gemini free tier: Google may use this transcript to improve its
   products." (The user accepted the free tier for the test lectures, which are public videos.)
7. Regeneration keeps versions: every generation is kept (each one is paid for); the exports
   show the latest completed one, and `export --notes-version N` writes `notes-vN.md/.html`.
8. API keys in the Windows Credential Manager through `keyring`, entered with
   `studyscribe config set-key gemini|anthropic` (no echo). Environment variables as fallback.
   Never in repo files, never in logs or error messages.
9. Effort `medium` by default (thinking level for Gemini, `effort` for Claude).

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

## Pieces (one at a time, stop after each with a test summary)

1. Provider layer: interface, Gemini and Claude backends, retry and pacing, keys,
   `config set-key/show/delete-key`, `scripts/llm_probe.py`. Probe: one tiny request with a
   three-field schema per provider (model, latency, tokens, cost) and an exact token count of
   the two transcripts. Unit tests: backoff with a fake clock, error mapping, key lookup order,
   key redaction. Anthropic only once the user has a key, and only after asking.
2. Pipeline: blocks, map, reduce, schema from the template, validation, reference checks,
   identifier corrections, resume. All tests offline with a fake LLM.
3. Database v3 and CLI: `notes`, `transcribe --notes`, estimate and confirmation, recovery,
   `template list/import`, NOTES column in `sessions`.
4. Export: `notes.md`, `notes.html` with player and clickable timestamps, `audio.flac` when
   missing.
5. Acceptance: notes on the 3 h lecture with gemini-3.8-flash, Haiku 5.5 and Sonnet 5.5 (quality
   and real cost from `llm_calls`), notes on the Italian lecture, and a checklist in
   `docs/manual-test-notes.md`: timestamps to click, definitions to compare with the audio, the
   "TEC" → `detach` correction (0:14:01 to 0:14:08).

## Dependencies

Installed only after the user confirms. All four in the main dependencies, no extra.

| Package | Constraint | Resolves today to | Why |
|---|---|---|---|
| anthropic | `>=2.3,<3` | 2.4.0 (2026-11-02) | Claude backend; 2.x speaks API version 2026-10-21 |
| google-genai | `>=2.33,<3` | 2.34.0 (2026-10-29) | Gemini backend |
| keyring | `>=25.7,<26` | 25.7.0 | API keys in the Windows Credential Manager |
| pydantic | `>=2.14,<3` | 2.14.2 | output schemas and validation (both SDKs depend on it anyway) |

## Modules

```
src/studyscribe/notes/
  providers/base.py     LlmProvider protocol, LlmRequest / LlmResponse / Usage, LlmError
                        subclasses with a remedy, key redaction
  providers/gemini.py   GeminiProvider (only module importing google.genai)
  providers/claude.py   ClaudeProvider (only module importing anthropic; streaming +
                        get_final_message)
  providers/retry.py    backoff and pacing (pure, injectable clock)
  pricing.py            price table with its verification date, cost from usage (Haiku's
                        100k threshold)
  keys.py               keyring + environment variables (only module importing keyring)
  templates.py          template model, loading, built-in sync
  templates/studio.json
  schema.py             pydantic models per section kind, output model built from a template
  prompts.py            PROMPT_VERSION, prompt rendering, hash of the prompt files
  prompts/map.md, prompts/reduce.md
  blocks.py             block cuts in pauses (pure)
  refs.py               reference validation and resolution (pure)
  estimate.py           token and cost estimate (pure)
  pipeline.py           map / reduce, validation, retries (no database)
  runner.py             generate_notes: checks, lock, lifecycle, resume, usage rows (database)
  export.py             notes.md, notes.html
scripts/llm_probe.py    piece 1 probe
docs/manual-test-notes.md
```

Changed: `pyproject.toml`, `db/migrations.py` (V3), `db/repo.py`, `session.py` (lock byte 3),
`recovery.py`, `cli.py`, `tests/fakes.py` (FakeLlm), `README.md` and `CLAUDE.md` at the end.

## Flow

1. Checks: the session exists and has a done transcript; at least 5 minutes of speech (sum of
   `end_ms - start_ms` of kept segments); the transcript has a language or `--notes-language` is
   given; a key exists for the provider.
2. Blocks: 10 to 15 minutes each, cut at the longest gap between two segments in that range; a
   last block under 5 minutes joins the previous one. About 14 blocks for 3 h, 2 for 20 min.
3. Estimate: requests, input and output tokens, cost (charged and list price), requests against
   the per-minute and per-day limits. Confirmation above the threshold; with no terminal and no
   `--yes`, nothing is sent.
4. Map, sequential: one request per block with the title, course, language and the glossary of
   corrections from earlier blocks (so identifiers stay consistent). References must fall inside
   the block.
5. Reduce: one request on the validated map outputs only (not on the transcript). It merges
   duplicates, joins their references and sizes each section to the lecture's length.
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

### Identifier corrections

In the notes only; the transcript stays faithful to the audio. The prompt asks to write code read
aloud in code form ("torch dot nn" → `torch.nn`, "ipsilon pred" → `y_pred`) and to fix a
recognition error only when the context makes the word certain ("TEC" → `detach`). Every fix is
listed in a `corrections` field (heard, written, refs) and rendered as an appendix.

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
  -- default provider/model, confirmation threshold, gemini tier, rpm/rpd

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
  stats_json        TEXT,            -- dropped references/items, corrections
  started_at TEXT NOT NULL, finished_at TEXT, error TEXT
);
CREATE UNIQUE INDEX idx_notes_open ON generated_notes(session_id)
  WHERE status IN ('running','interrupted');

CREATE TABLE llm_calls (
  id INTEGER PRIMARY KEY,
  notes_id INTEGER NOT NULL REFERENCES generated_notes(id) ON DELETE CASCADE,
  stage TEXT NOT NULL CHECK (stage IN ('map','reduce')),
  block INTEGER NOT NULL, attempt INTEGER NOT NULL,
  status TEXT NOT NULL CHECK (status IN ('ok','invalid','error')),
  model_version TEXT, request_id TEXT,
  input_tokens INTEGER, output_tokens INTEGER, thinking_tokens INTEGER, cached_tokens INTEGER,
  cost_usd REAL,        -- charged (0 on the free tier)
  list_cost_usd REAL,   -- at list price
  duration_s REAL, output_json TEXT, error TEXT, created_at TEXT NOT NULL
);
```

`transcript_run_id` uses `ON DELETE SET NULL`: a re-transcription deletes the previous run, the
notes stay readable and are flagged as based on an earlier transcript. The template is copied into
`template_json` rather than referenced, because a template can change after a generation.

## CLI

```
studyscribe config set-key gemini|anthropic   (getpass, no echo) | delete-key | show | set NAME VALUE
studyscribe notes SESSION_ID [--provider] [--model] [--notes-language] [--template studio]
                  [--effort low|medium|high] [--yes] [--restart] [--list]
studyscribe transcribe SESSION_ID --notes [same options]
studyscribe export SESSION_ID [--audio] [--notes-version N]
studyscribe template list | import FILE.json
```

- Before sending: blocks, requests, estimated tokens (in/out), estimated cost (on the free tier
  "$0, $0.44 at list price"), requests against the limits; the free-tier privacy notice.
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
2. Every item cites in "refs" the lines it comes from, the line where it starts first.
   Cite only S-numbers that appear in this part.
3. Write in {language_name}. Keep technical terms as the lecturer uses them.
4. Write code the lecturer reads aloud in code form ("torch dot nn" -> `torch.nn`,
   "ipsilon pred" -> `y_pred`). Fix a speech-recognition error only when the context makes
   the intended word certain, and list every fix in "corrections" (heard, written, refs).
   Corrections from earlier parts: {glossary}.
5. Definitions: keep the lecturer's wording; remove only spoken fillers.
Sections for this part:
{for each section: key - instructions - about {target} items}
```

The user message holds the block's lines.

Reduce (system):

```
You merge the partial notes of consecutive parts of one lecture into its final study notes.
Rules
1. Use only the content of the partial notes; add nothing new.
2. Merge duplicates: keep the most complete wording, join their refs (earliest first).
3. Copy refs exactly; never write an S-number that is not in the input.
4. Write in {language_name}; keep code in code form and the corrections consistent.
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
no code; a new kind needs its rendering. Corrections are a fixed part of the schema. "Unclear
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
- Real requests only in `scripts/llm_probe.py` and in the acceptance test.

## Risks and unverified assumptions

- Gemini free tier: 20 requests per day; content used by Google (accepted for public test videos;
  a notice is printed).
- `generate_content` versus the new `interactions` API in google-genai 2.x: checked in piece 1.
- RECITATION blocks on Gemini: the test lecture is public, and definitions quoted almost verbatim
  may be blocked. If it happens: one retry asking for a paraphrase, and a warning.
- References that exist but point to the wrong place: validation cannot see them. The
  "inside the block" rule limits them; clicking them in the acceptance test checks the rest.
- Over-eager identifier corrections: mitigated by the "only when certain" rule and by the
  corrections appendix, each with its reference.
- "Faithful to the lecturer": an LLM tends to paraphrase. Not checkable automatically; checked by
  listening.
- FLAC seeking in a page opened from disk (Edge, Chrome): tested in piece 4 (seek table, accuracy).
- Estimates: about ±30%; the real cost comes from the usage rows. Prices change (Gemini 3.8 Flash
  doubles in January 2027).
- Haiku 5.5 refusals: no server-side fallback; treated as an error with a message.
- `anthropic` 2.x is recent (2.0.0 on 2026-10-21, 2.4.0 on 2026-11-02).

## Findings so far

(Filled in as the pieces are done.)
