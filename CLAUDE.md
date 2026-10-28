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

Current state: step 1, setup done.

## Architecture principles

- The chain *capture → segments on disk* depends on nothing else. Transcription, LLM and UI
  read from disk and may crash or restart without losing audio.
- Capture runs in its own process (`multiprocessing`, spawn) and writes the WAV segments and
  their manifest rows itself. The parent only starts/stops it and shows levels/events.
- Internal audio format: PCM int16, mono, 16 kHz.
- Windows APIs (WASAPI, COM, SetThreadExecutionState, file locks) live behind small modules;
  everything else is pure and testable without audio hardware (inject clocks, fake sources).

## Step 1 design (recording)

- WASAPI loopback of the default output via PyAudioWPatch in callback mode; the callback only
  queues `(perf_counter(), bytes)`. Downmix to mono, resample with `soxr.ResampleStream`, int16.
- Segments hold exactly 60 s of samples, so segment N starts at (N-1)·60000 ms:
  `<data>/sessions/<session_id>/audio/000001.wav …`. Unbuffered writes, header patched every
  second, fsync every 10 s. Manifest row in `audio_segments` when a segment closes.
- Timeline aligned to `time.perf_counter`: silence is inserted only when the deficit exceeds a
  threshold (100 ms), and after 0.5 s without data the timeline is filled up to now − 100 ms.
  Real audio is never dropped.
- Default output polled via COM (ctypes) every ~1.5 s. On change, or if the stream dies: close,
  re-init PortAudio (its device list is cached), reopen, fill the hole and add it to `gap_ms`
  (`gap_ms` = milliseconds of the segment that are synthetic because of a device switch).
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
  cli.py        command-line entry point (`studyscribe`, `python -m studyscribe`)
  config.py     data directory resolution and shared settings
tests/          pytest, no audio hardware required
```

Planned for step 1: `recording.py`, `recovery.py`, `db/` (connection, migrations, repo),
`audio/` (dsp, timeline, wav, segments, devices, capture), `system/` (power, locks),
`scripts/manual_recording_test.py`, `docs/manual-test-recording.md`.

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
```

## Working rules

- Show the plan first and wait for confirmation; then work one piece at a time and stop after
  each piece with a summary of what to test.
- Ask before installing packages, deleting files or doing any git operation other than a local
  commit. Never create the remote repository and never push.
- Flag errors, unverified assumptions and places where the design notes may be wrong.
- Code, comments and repo docs in English; replies to the user in Italian.
