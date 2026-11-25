# StudyScribe

Personal app for taking notes from online courses: it records the system audio
of a lecture, transcribes it locally and generates study notes with clickable
timestamps. The repo is public; the app runs on one Windows PC.

## Decisions already made

- Target machine: Windows 11 desktop with an NVIDIA RTX 3060 (12 GB).
- Python 3.12 venv in `.venv` (`requires-python >=3.11`), `src/` layout, package `studyscribe`.
- Storage: SQLite (WAL; FTS5 trigram search later) plus one folder per session.
- Transcription: local faster-whisper (step 2). Notes: local LLM through
  Ollama by default, cloud optional, configurable model (step 3).
- Lectures are mostly English, some Italian: automatic language detection with a manual override.
- MVP is CLI only (steps 1-3). Live UI (FastAPI + WebSocket), slide capture and automatic
  session detection come later: keep modules decoupled so they can be added without rewrites.
- Written from scratch. The product idea is inspired by Granola; never use or reproduce its code.

## Roadmap

| Step | What | Done when |
|---|---|---|
| 1 | Recording: loopback → 60 s WAV segments, manifest, recovery, anti-standby | 3 h recording with headphones unplugged/replugged and a process kill loses at most a few seconds and the timeline stays aligned |
| 2 | Transcription worker on closed segments → SQLite, plus a final pass | 1 h lecture transcribed with correct timestamps and no hallucinated repeats on silence |
| 3 | Notes: LLM map-reduce, validated JSON, MD/HTML export with audio player | every timestamp in the notes points to the right audio position; no invented references |
| 4 | Live UI (FastAPI + WebSocket) | a full session can be run from the UI |
| 5 | Slides: screenshots, perceptual hash, OCR | slides appear next to the right note section |
| 6 | Extras: cloud ASR, automatic detection, desktop packaging | — |

Current state: steps 1, 2 and 3 done: the command-line MVP is complete.
Step 1's acceptance test (`docs/manual-test-recording.md`,
section C: 3 h recording) passed on 2026-10-30. Step 2 (plan and findings:
`docs/plan-transcription.md`) passed its acceptance test on the same 3 h recording on
2026-11-02 (`docs/manual-test-transcription.md`, section C), and the Italian check (section D)
passed too. Step 3 (plan and findings: `docs/plan-notes.md`) done on 2026-11-25, with the known
limits below. Map prompt v5.1 is the default and stays frozen until the user decides to work on
fix 10 (after the MVP): do not change it before. Its pieces: piece 1 (cloud
providers) and 1b (Ollama, default `qwen3.5:9b`, thinking off) done; piece 2 (pipeline, no
database) done offline; piece 3 (database
v3, CLI) done; piece 4 (exports `notes.md` / `notes.html`, 2026-11-22) done, the user's browser
check passed on 2026-11-25 (`docs/manual-test-notes.md`, A1-A6); piece 5
(acceptance with real notes) failed its first measurement (the 3 h lecture stopped in the final
reduce, `Truncated`: "Piece 5" in `docs/plan-notes.md`) and **passed with known limits on
2026-11-25**, after the fixes 1-5, 8 and 9 of piece 5b ("Piece 5b") and A6 of the manual test.
Real runs on 3h-copy, default settings: A (generation #1 resumed) done in 8 min 04 s, its 15 stored
maps reused, 3 partial reduces 211 s, final reduce 205 s, token counts 68 s; B (from scratch,
uninterrupted) done in 15 min 01 s, 15 maps 420 s, 3 partials 211 s, final 203 s, counts 59 s.
No request failed or was repeated in either; B's 15 maps (and its other answers) are the same
text as #1's. Known limits: the final reduce overshoots its targets (156 items for 91: fix 10,
postponed), code extraction in Italian is unstable, fixes 6 (gaps inside a block) and 7 (a resume
redoes the partial reduces) postponed. Code read aloud
is written in code form ("ipsilon pred" -> `y_pred`); library names stay as heard
("PyTorch.neuralNetwork" is not turned into `torch.nn`); code names not found in the transcript
get a warning, never a drop. Known limit: recognition errors are not corrected automatically
("TEC" stays "TEC", not `detach`): a correction pass failed with thinking off and on
(2026-11-16). Instead, the words Whisper was unsure about are listed in a "Words to check by
listening" appendix.

## Architecture principles

- The chain *capture → segments on disk* depends on nothing else. Transcription, LLM and UI
  read from disk and may crash or restart without losing audio.
- Capture runs in its own process (`multiprocessing`, spawn) and writes the WAV segments and
  their manifest rows itself. The parent only starts/stops it and shows levels/events.
- Internal audio format: PCM int16, mono, 16 kHz.
- Windows APIs (WASAPI, COM, SetThreadExecutionState, file locks) live behind small modules;
  everything else is pure and testable without audio hardware (inject clocks, fake sources).

## Step 1 design (recording)

Full agreed plan, including test hardware and open assumptions: `docs/plan-recording.md`.

- WASAPI loopback of the default output via PyAudioWPatch in callback mode; the callback only
  queues `(perf_counter(), bytes)`. Downmix to mono, resample with `soxr.ResampleStream`, int16.
- Segments hold exactly 60 s of samples, so segment N starts at (N-1)·60000 ms:
  `<data>/sessions/<session_id>/audio/000001.wav …`. Unbuffered writes, header patched every
  second, fsync every 10 s. Manifest row in `audio_segments` when a segment closes.
- Timeline aligned to `time.perf_counter` (`audio/timeline.py`): while audio flows, silence is
  inserted only when the deficit exceeds 100 ms; after 0.5 s without data the timeline is filled
  up to now − 100 ms; the first packet after silence (start of the recording, pause, device
  switch) is placed exactly, so a recording that starts in silence is filled from t0. Real
  audio is never dropped.
- Device changes: reopen when `is_active()` turns False (checked every tick) or the default
  output's endpoint ID changes (COM via ctypes, `audio/devices.py`, polled every 0.5 s). Reopen
  = new `LoopbackSource` (fresh PortAudio, whose device list is cached), retried every 0.5 s.
  The hole, from `max(last packet, detection − 0.5 s)` to the reopen, is filled with silence
  and counted in `gap_ms` (milliseconds of the segment that are synthetic because of a device
  switch). The first stream must open; after that a lost device never stops the recording.
- Auto-stop after 30 min without sound (packets above -80 dBFS; digital silence does not count),
  configurable: session `recorded` with a warning. The timeline is never filled past last
  sound + limit, so a clock jump after a system sleep cannot write hours of silence.
- Anti-standby: `SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)` in the capture
  thread, released in `finally`.
- Recovery (`recovery.py`; at start of `record`/`sessions`, or `recover`): a session in
  `recording` is touched only if both byte locks of `<session>/session.lock` are free (byte 0:
  owner/CLI, taken before the row exists; byte 1: capture process). Headers of unregistered
  segments are repaired from the file size, the segments added to the manifest, status
  `interrupted` (`failed` without audio), `ended_at` = mtime of the last segment. If only the
  capture process dies, the owner recovers its own session (`owner_lock=`).
- Deviations from the original schema: `sessions.status` also allows `recorded` (recorded, not
  yet transcribed); `courses` exists from migration v1 (unique `title_key`: NFC + casefold of
  the title, computed in Python);
  session IDs look like `20261028-142301-3f9a2c` (sortable, readable).
- Migrations: Python list applied in order, tracked with `PRAGMA user_version`, one transaction each.

## Step 2 design (transcription)

Full agreed plan, findings and test results: `docs/plan-transcription.md`.

- `studyscribe transcribe <id>` on a recorded session, local faster-whisper (`large-v3-turbo`,
  float16, pinned to `dropbox-dash/faster-whisper-large-v3-turbo@0a363e9`). Only cuBLAS 12 is
  needed (the `cuda` extra); `transcribe/cuda.py` preloads its DLLs because ctranslate2's
  `LoadLibraryA` would not find them. No cuDNN.
- The session is read as one signal (`audio/reader.py`) and processed in windows of up to 5 min
  cut in pauses (VAD), never at file boundaries; times are mapped to the session timeline
  (`gap_ms` is not used).
- Level normalisation only on the float copy fed to VAD and Whisper (towards -20 dBFS, 0..+30 dB,
  never pushes a sample above -1 dBFS); WAV files are opened read-only.
- Language: VAD first, then the average of up to six 30 s chunks of speech; no speech = language
  NULL, never a guess from silence. `--language` forces it.
- Decoding: sequential (not batched), `vad_filter`, `word_timestamps`, no conditioning on
  previous text, OpenAI's long-form thresholds. Segments over 12 s are split at sentence ends.
  Filters (`transcribe/filters.py`) mark segments as dropped instead of deleting them.
- Hints are off by default: the A/B test showed them inserting repeated sentences.
- Schema v2: `transcription_runs` + `transcript_segments`. Committed window by window; a run
  becomes the session's transcript only when done, replacing the old one in one transaction.
  `sessions.status` is not touched by transcription. Lock: byte 2 of `session.lock`; recovery
  marks runs whose process died as `interrupted`, and the same command resumes them.
- Exports in `<session>/exports/`: `transcript.srt`, `transcript.md`, `transcript-dropped.txt`,
  and `audio.flac` (`export --audio`, lossless, same timeline).

## Layout

```
pyproject.toml        package, dependencies, the `cuda` and `dev` extras, pytest settings
README.md             what the app is, setup, usage
src/studyscribe/      the package
  cli.py              `studyscribe record|transcribe|notes|export|sessions|recover|template|
                      config` (also `python -m studyscribe`, __main__.py): live status/progress
                      lines, recovery first
  cli_notes.py        CLI of the notes: `notes`, the options of `transcribe --notes`, `template`,
                      `config set`
  config.py           data directory resolution and shared settings
  recording.py        RecorderHandle: parent side of the capture process (start/events/stop/kill)
  session.py          new_session (parent, owner lock) and SessionSink (capture process: capture
                      lock, segments + manifest)
  recovery.py         recover_all / recover_session: repair abandoned sessions, transcriptions
                      and notes generations
  verify.py           checks on a recorded session: manifest vs files, where the sound is
  audio/              step 1: from WASAPI to 60 s WAV segments
    capture.py        AudioSource/Sink protocols, LoopbackSource, run_capture loop (device
                      reopen), capture_main
    devices.py        DefaultOutputMonitor: default output endpoint ID through COM (ctypes)
    dsp.py            downmix, streaming resampler, int16 conversion, level meter
    timeline.py       TimelineAligner: packets -> gap-free timeline aligned to the clock (pure)
    segments.py       SegmentWriter: exact 60 s segments, gap frames per segment
    wav.py            WavWriter (unbuffered, header patched every 1 s, fsync every 10 s),
                      repair_wav_header
    reader.py         SessionAudio: a session's segments as one read-only int16 signal
  db/                 SQLite
    connection.py     connect (no migration), open_database (migrates), transaction
    migrations.py     MIGRATIONS list, PRAGMA user_version
    repo.py           queries on courses, sessions, audio segments, transcripts and notes
  system/             thin wrappers around Windows APIs
    power.py          keep_system_awake (SetThreadExecutionState)
    locks.py          ByteLock: one-byte file lock released by the OS when the process dies
                      (asynchronously on Windows: acquire(wait_s=...) when the holder is dead)
  transcribe/         step 2: local transcription
    engine.py         Transcriber protocol, AsrSegment/AsrWord, TranscriberError
    whisper.py        FasterWhisperTranscriber (only module importing faster_whisper),
                      GPU error messages
    cuda.py           preload of cuBLAS from the nvidia-cublas-cu12 wheel (Windows)
    vad.py            Silero VAD wrapper (speech spans, pause spans)
    levels.py         normalisation of the ASR copy (pure)
    windows.py        window cuts in pauses (pure)
    language.py       language from speech chunks (pure)
    segments.py       TranscriptSegment, session-time mapping, split of long segments
    filters.py        anti-hallucination rules (pure)
    pipeline.py       detect_language, transcribe_windows (no database)
    runner.py         transcribe_session: checks, lock, run lifecycle, resume (database)
    export.py         srt / md / dropped list / audio.flac
  notes/              step 3: study notes
    transcript.py     the transcript as numbered lines (S-numbers), the prompt's line format
    blocks.py         cuts the transcript into blocks of 10-15 min, in pauses (pure)
    templates.py      note templates: model, loading, item targets
    schema.py         pydantic model per section kind, output model built from a template
    prompts.py        PROMPT_VERSION, rendering of the prompt files, their hash
    pipeline.py       map / partial reduces / final reduce, validation, retries (no database)
    reduce_plan.py    grouping of map outputs for the hierarchical reduce (pure)
    refs.py           reference validation and resolution to timestamps (pure)
    code_check.py     graded check of code names against the cited lines (pure)
    uncertain.py      words Whisper was unsure about, for the appendix (pure)
    estimate.py       requests, tokens, time and cost before a run (pure)
    runner.py         prepare and generate a session's notes: checks, lock, lifecycle, resume,
                      call records (database)
    pricing.py        list prices with their verification date, cost from usage
    keys.py           API keys: keyring, environment variables (only module importing keyring)
    gpu.py            VRAM from nvidia-smi (optional)
    doc.py            a generation read for rendering: versions, same-transcript check, audio
    labels.py         fixed texts around the notes, en/it
    render_common.py  warnings, appendix rows, footer shared by both renderers
    render_md.py      notes.md (escaped text, fenced code, <details> answers)
    render_html.py    notes.html (inline style/script from resources/export, CSP by hash)
    export.py         writes notes.md / notes.html (or notes-vN.*) atomically; never audio.flac
    providers/        one interface, three backends
      base.py         LlmProvider protocol, request/response/usage, LlmError subclasses with a
                      remedy, key redaction
      ollama.py       default: native /api/chat (stdlib HTTP), context and GPU checks, load/unload
      claude.py       Anthropic (only module importing anthropic)
      gemini.py       Google Gemini (only module importing google.genai)
      retry.py        backoff and pacing (pure, injectable clock)
    resources/        prompts/ (map.md, code-rule.md, reduce.md: frozen, v5.1), templates/
                      (studio.json), export/ (notes.css, notes.js)
scripts/              tools run by hand, on real hardware or on test copies
  capture_probe.py    real-hardware probe: one WAV + stream stats, --session "Title",
                      --recover, --silence-limit SECONDS
  manual_recording_test.py   guided manual test: media key for play/pause/resume (or
                      --manual), unplug/replug, kill, checks; --check SESSION_ID
  transcribe_probe.py GPU probe on one segment (model hash, timings, VRAM), or the
                      whole pipeline on a stretch: --from/--to [--no-normalize] [--hints]
  llm_probe.py        LLM providers: one tiny structured request, exact token counts; with the
                      local one also a map-like request on a block of a lecture
  check_notes_export.py      notes.html timestamps vs transcript.srt and the FLAC length
tests/                pytest, no audio hardware required
  conftest.py         fixtures: data_dir, db, start_session
  fakes.py            fakes and helpers (clock, audio sources, FakeTranscriber, FakeLlm)
  notes_fixtures.py   hand-made NotesDocs for the renderer tests
  test_*.py           about one per module (test_notes_*, test_llm_*: step 3)
docs/
  plan-recording.md, plan-transcription.md, plan-notes.md   plan and findings of steps 1, 2, 3
  manual-test-recording.md      manual checklist, including the 3 h acceptance test
  manual-test-transcription.md  transcription checklist, acceptance test and Italian check
  manual-test-notes.md          notes.html checked in the browser (A1-A7)
```

Fakes passed to the capture process (via `CaptureConfig` factories) must be picklable:
top-level classes/functions or `functools.partial` of them, never lambdas. Tests always pass a
fake `default_output_factory` (e.g. `ScriptedDefaultOutput`), otherwise the real COM monitor
runs. In-process loop tests use `FakeClock` + `ClockedSource` (packets follow the fake clock).
`tests/conftest.py` has the `data_dir`, `db` and `start_session` fixtures (`start_session`
releases the owner lock after the test). Tests that start a capture process must kill it in a
`finally`, or a failed assertion leaves pytest waiting for the process forever.
Transcription tests use `FakeTranscriber` and the energy VAD `loud_spans` (tests/fakes.py) with
small windows; tests marked `gpu` load the real model and are skipped unless `pytest -m gpu`.

## Data directory

Default `%LOCALAPPDATA%\StudyScribe`, never relative to the working directory. Override with
`--data-dir` or `STUDYSCRIBE_DATA_DIR`. The CLI always prints the path in use.

```
<data>/app.db
<data>/sessions/<session_id>/audio/000001.wav …
<data>/sessions/<session_id>/exports/transcript.srt, transcript.md, transcript-dropped.txt, audio.flac,
                                              notes.md, notes.html, notes-vN.md, notes-vN.html
```

The speech model lives in the Hugging Face cache (`%USERPROFILE%\.cache\huggingface\hub`),
shared by all data directories.

## Commands

```powershell
py -3.12 -m venv .venv                          # create the venv
.venv\Scripts\python -m pip install -e ".[dev,cuda]" # install (ask first)
.venv\Scripts\python -m pytest                  # run tests (-m gpu: tests with the real model)
.venv\Scripts\studyscribe --help                # CLI
.venv\Scripts\studyscribe record --title "Lecture 3" --course "Calculus"   # Ctrl+C to stop
.venv\Scripts\studyscribe transcribe SESSION_ID  # transcribe (--language it, --restart)
.venv\Scripts\studyscribe export SESSION_ID --audio   # rewrite exports (and the latest notes), join audio.flac
.venv\Scripts\studyscribe export SESSION_ID --notes-version 1   # notes-v1.md/.html (the 1st finished generation)
.venv\Scripts\python scripts\check_notes_export.py <data>\sessions\SESSION_ID\exports   # timestamps vs SRT
.venv\Scripts\studyscribe notes SESSION_ID         # study notes with Ollama (--restart, --effort)
.venv\Scripts\studyscribe transcribe SESSION_ID --notes   # transcription, then the notes
.venv\Scripts\studyscribe template list            # note templates (import FILE.json)
.venv\Scripts\studyscribe config set ollama.num_ctx 32768   # settings (config show lists them)
.venv\Scripts\studyscribe sessions               # list (recovers abandoned sessions first)
.venv\Scripts\studyscribe recover                # recovery on demand
.venv\Scripts\python scripts\capture_probe.py --seconds 60   # real-hardware capture probe
.venv\Scripts\python scripts\capture_probe.py --recover      # same recovery, from the probe
.venv\Scripts\python scripts\manual_recording_test.py         # guided manual test (~4 min)
```

## Working rules

- Show the plan first and wait for confirmation; then work one piece at a time and stop after
  each piece with a summary of what to test.
- Ask before installing packages, deleting files or doing any git operation other than a local
  commit. Never create the remote repository and never push.
- Flag errors, unverified assumptions and places where the design notes may be wrong.
- Code, comments and repo docs in English; replies to the user in Italian.
