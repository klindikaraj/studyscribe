# StudyScribe

Personal app for taking notes from online courses: it records the system audio
of a lecture, transcribes it locally and generates study notes with clickable
timestamps. The repo is public; the app runs on one Windows PC.

## Decisions already made

- Target machine: Windows 11 desktop with an NVIDIA RTX 3060 (12 GB).
- Python 3.12 venv in `.venv` (`requires-python >=3.11`), `src/` layout, package `studyscribe`.
- Storage: SQLite (WAL; FTS5 trigram search later) plus one folder per session.
- Transcription: local faster-whisper (step 2). Notes: cloud LLM with a configurable model (step 3).
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

Current state: step 1, silence filling and device changes done (next: recovery).

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
- Recovery (at start of `record`/`sessions`, or `recover`): sessions in `recording` whose lock
  is free become `interrupted`; WAV headers are repaired from the file size; unregistered
  segments are added to the manifest.
- Deviations from the original schema: `sessions.status` also allows `recorded` (recorded, not
  yet transcribed); `courses` exists from migration v1 (unique title, case-insensitive);
  session IDs look like `20261028-142301-3f9a2c` (sortable, readable).
- Migrations: Python list applied in order, tracked with `PRAGMA user_version`, one transaction each.

## Layout

```
src/studyscribe/
  cli.py              command-line entry point (`studyscribe`, `python -m studyscribe`)
  config.py           data directory resolution and shared settings
  recording.py        RecorderHandle: parent side of the capture process (start/events/stop)
  session.py          new_session (parent) and SessionSink (capture process: segments + manifest)
  audio/dsp.py        downmix, streaming resampler, int16 conversion, level meter
  audio/timeline.py   TimelineAligner: packets -> gap-free timeline aligned to the clock (pure)
  audio/devices.py    DefaultOutputMonitor: default output endpoint ID through COM (ctypes)
  audio/capture.py    AudioSource/Sink protocols, LoopbackSource, run_capture loop (device
                      reopen), capture_main
  audio/wav.py        WavWriter: unbuffered, header patched every 1 s, fsync every 10 s
  audio/segments.py   SegmentWriter: exact 60 s segments, gap frames per segment
  db/connection.py    connect (no migration), open_database (migrates), transaction
  db/migrations.py    MIGRATIONS list, PRAGMA user_version
  db/repo.py          queries on courses, sessions, audio_segments
  system/power.py     keep_system_awake (SetThreadExecutionState)
tests/                pytest, no audio hardware required; tests/fakes.py holds fakes and helpers
scripts/capture_probe.py   real-hardware probe: one WAV + stream stats, or --session "Title"
```

Still planned for step 1: `recovery.py`, `system/locks.py`, CLI subcommands,
`scripts/manual_recording_test.py`, `docs/manual-test-recording.md`.

Fakes passed to the capture process (via `CaptureConfig` factories) must be picklable:
top-level classes/functions or `functools.partial` of them, never lambdas. Tests always pass a
fake `default_output_factory` (e.g. `ScriptedDefaultOutput`), otherwise the real COM monitor
runs. In-process loop tests use `FakeClock` + `ClockedSource` (packets follow the fake clock).

## Data directory

Default `%LOCALAPPDATA%\StudyScribe`, never relative to the working directory. Override with
`--data-dir` or `STUDYSCRIBE_DATA_DIR`. The CLI always prints the path in use.

```
<data>/app.db
<data>/sessions/<session_id>/audio/000001.wav …
```

## Commands

```powershell
py -3.12 -m venv .venv                          # create the venv
.venv\Scripts\python -m pip install -e ".[dev]" # install (ask first)
.venv\Scripts\python -m pytest                  # run tests
.venv\Scripts\studyscribe --help                # CLI
.venv\Scripts\python scripts\capture_probe.py --seconds 60   # real-hardware capture probe
```

## Working rules

- Show the plan first and wait for confirmation; then work one piece at a time and stop after
  each piece with a summary of what to test.
- Ask before installing packages, deleting files or doing any git operation other than a local
  commit. Never create the remote repository and never push.
- Flag errors, unverified assumptions and places where the design notes may be wrong.
- Code, comments and repo docs in English; replies to the user in Italian.
