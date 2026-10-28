# Step 1 plan: reliable recording

Agreed plan for step 1 of the roadmap. Update it when a decision changes.

## Goal

Record the system audio of a lecture for 1-3 hours without losing data, even if the
process crashes or the output device changes, with a timeline that stays aligned to real
time. "Done" means: a 3 h recording with headphones unplugged and replugged and a process
kill loses at most a few seconds and the timeline stays aligned.

## Pieces (one at a time, stop after each with a test summary)

1. Setup (done)
2. Capture: loopback in callback mode, downmix + resample, separate process, anti-standby
3. Segments and manifest: 60 s WAV segments, SQLite schema v1 and migrations
4. Silence filling and device changes
5. Recovery
6. CLI: `record`, `sessions`, `recover`
7. Tests: remaining unit tests, manual checklist and verification script

## Dependencies

- Runtime: `PyAudioWPatch`, `numpy`, `soxr`. Dev: `pytest`.
- Python 3.12 venv in `.venv`.
- Default output detection uses COM through `ctypes` (no extra dependency). If that code
  becomes fragile or hard to test, stop and propose `pycaw` instead.

## Modules

```
src/studyscribe/
  cli.py            argparse: record / sessions / recover
  config.py         constants (16 kHz, 60 s, thresholds) and data directory
  recording.py      parent side: RecorderHandle (start / stop / events), reusable by a future API
  recovery.py
  db/   connection.py  migrations.py  repo.py
  audio/
    dsp.py          downmix, streaming soxr resampler, float -> int16, dBFS levels
    timeline.py     TimelineAligner: silence filling with an injectable clock (pure)
    wav.py          WAV writer with periodic header patching, repair_wav_header
    segments.py     rotation every 60 s, gap_ms per segment, callback on close
    devices.py      default render endpoint ID (COM via ctypes) + loopback lookup
    capture.py      capture process entry point
  system/  power.py (SetThreadExecutionState)  locks.py (session lock)
tests/              no audio hardware required
scripts/manual_recording_test.py
docs/manual-test-recording.md
```

Steps 2-6 add `transcribe/`, `notes/`, `api/` next to these without changing them.

## Processes

- The parent (CLI) creates the session row and starts the capture process with
  `multiprocessing` (spawn context).
- The child owns everything else: session lock, anti-standby, stream, WAV writing,
  `audio_segments` rows, final session status.
- Parent → child: a `multiprocessing.Event` to stop. Child → parent: a bounded
  `multiprocessing.Queue` of events (levels, stats, device changes, errors), written with
  `put_nowait`; if the parent does not read, events are dropped and capture never blocks.
  The child calls `cancel_join_thread()` so it can always exit.
- Ctrl+C reaches every process attached to the console on Windows, so the child ignores
  SIGINT and stops only through the Event.
- If the parent dies, the child notices via `multiprocessing.parent_process().is_alive()`,
  closes the segment cleanly and marks the session `interrupted`.
- If the child dies, the parent runs recovery for that session.
- The CLI prints the PIDs of both processes: on Windows the venv and console-script
  launchers add extra `python.exe` processes, so it is easy to kill the wrong one.

## Capture

- PyAudioWPatch exposes the WASAPI loopback of each output as an input device. Open the
  loopback of the default output in callback mode, `paFloat32`, the device's channel count
  and rate (typically 48 kHz stereo).
- The callback only queues `(time.perf_counter(), bytes, status)`; the main loop runs every
  ~50 ms, drains the queue, downmixes (mean of channels), resamples to 16 kHz with
  `soxr.ResampleStream` and converts to int16.
- Callback mode rather than blocking `read()`: when nothing plays, a blocking read would
  stall the loop, so no silence filling and no device checks.

## Timeline alignment (`TimelineAligner`, pure, injectable clock)

- `expected = (clock() - t0) * 16000`; `written` = samples emitted so far.
- On a block of n samples: if `expected_at_block_start - written` exceeds 100 ms, emit that
  much silence first, then the block. Below the threshold do nothing, so normal jitter never
  moves anything.
- No data for 0.5 s (paused video): emit silence up to `now - 100 ms`. Files keep growing and
  segments keep rotating during pauses; the margin avoids writing "into the future".
- Error stays within about one threshold (100 ms) and does not accumulate, because every fill
  re-anchors to the clock.
- Real audio is never dropped. If the device clock runs faster than the system clock
  (ppm-level drift, about 0.3 s over 3 h at 30 ppm), the surplus is absorbed at the next pause.

## Segments

- Rotation by sample count: exactly 960,000 samples per segment, so segment N always starts
  at (N-1)·60000 ms. Files: `<data>/sessions/<session_id>/audio/000001.wav …`.
- Unbuffered writes (every tick reaches the OS), header patched every second, fsync every
  10 s. Maximum loss on a process kill: about one tick (< 100 ms).
- A row is inserted in `audio_segments` when a segment closes.

## Device changes

- Reopen the stream when (a) `is_active()` turns False (checked every tick: the recorded device
  vanished) or (b) the default render endpoint ID changes (read through COM,
  `IMMDeviceEnumerator`, every 0.5 s: a manual switch of the default output leaves the old
  stream active but silent).
- Reopen = close the stream, re-initialize PortAudio (it caches the device list at init), open
  the loopback of the current default. Measured cost: 25-65 ms.
- If the reopen fails (e.g. the endpoint is still appearing after replugging), retry every
  0.5 s. If no output exists at all, the timeline keeps advancing with silence meanwhile.
- The hole is always filled by the aligner and counted in `gap_ms`, from
  `max(last packet, detection - 0.5 s)` to the successful reopen, so real silence before an
  unplug (e.g. during a pause) is not counted as a gap.
- `gap_ms` = milliseconds of a segment that are synthetic because of a device switch. The
  timeline itself never has holes.
- Closing a stream whose device has vanished must not crash or hang the capture loop.

### Test hardware (verified by the user)

- Wired headphones in the Realtek jack. Plugged in: default output is
  "Speakers (Realtek(R) Audio)"; "KG251Q (NVIDIA High Definition Audio)" also exists.
- Unplugged: the Realtek endpoint disappears entirely (its microphone too) and Windows
  switches the default to "KG251Q (NVIDIA High Definition Audio)".
- Replugged: Realtek becomes the default again.
- So unplugging makes the device being recorded vanish while the stream is open (a stream
  error, not a clean close). Replugging switches back. The manual test covers both
  directions, each with the hole filled with silence and counted in `gap_ms`.
- While the headphones are unplugged the lecture keeps playing on the monitor, so after the
  reopen it is recorded from the NVIDIA endpoint: each switch costs a gap of about 0.5 s,
  not the whole unplugged period.

## Anti-standby

- `SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)` in the capture thread,
  released in `finally`. If the process is killed, Windows drops the request with the thread.
- Does not prevent the display from turning off or a manual sleep / lid close.
- Check with `powercfg /requests` during a recording.

## Database (schema v1)

- WAL, `foreign_keys=ON`, `busy_timeout`. Migrations are a Python list applied in order,
  tracked with `PRAGMA user_version`, one transaction each.
- v1 creates `courses`, `sessions`, `audio_segments`. Other tables come in later migrations.

```sql
CREATE TABLE courses (
  id          INTEGER PRIMARY KEY,
  title       TEXT NOT NULL,
  platform    TEXT,
  url         TEXT,
  created_at  TEXT NOT NULL
);
CREATE UNIQUE INDEX idx_courses_title ON courses(title COLLATE NOCASE);

CREATE TABLE sessions (
  id          TEXT PRIMARY KEY,     -- e.g. 20261028-142301-3f9a2c
  course_id   INTEGER REFERENCES courses(id),
  title       TEXT,
  source_app  TEXT,
  source_url  TEXT,
  status      TEXT NOT NULL CHECK (status IN
                ('recording','recorded','interrupted','transcribing','ready','failed')),
  started_at  TEXT NOT NULL,        -- set by the capture process at timeline t0
  ended_at    TEXT,
  sample_rate INTEGER NOT NULL DEFAULT 16000,
  asr_engine  TEXT,
  asr_model   TEXT,
  language    TEXT
);

CREATE TABLE audio_segments (
  session_id  TEXT NOT NULL REFERENCES sessions(id),
  seq         INTEGER NOT NULL,
  path        TEXT NOT NULL,
  start_ms    INTEGER NOT NULL,     -- relative to session start
  end_ms      INTEGER,
  gap_ms      INTEGER NOT NULL DEFAULT 0,
  transcribed INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY (session_id, seq)
);
```

Agreed deviations from the original design notes:

- `status` also allows `recorded` (recorded, not yet transcribed).
- `courses` already exists in v1: `--course` needs somewhere to store the course, and
  `course_id` references it. With foreign keys on, a non-null `course_id` pointing to a missing
  table makes the insert fail. The unique case-insensitive index on the title lets
  `--course "Calculus 1"` reuse the existing course instead of creating a duplicate.
- Session IDs are readable and sortable: `YYYYMMDD-HHMMSS-xxxxxx` (6 random hex digits).

## Recovery

- Runs automatically at the start of `record` and `sessions`, and on demand with `recover`.
- A session in `recording` whose lock is free (no live process) becomes `interrupted`.
  The lock (`msvcrt.locking`, released by the OS when the process dies) prevents "repairing"
  a recording that is still running in another terminal.
- For each WAV without a manifest row: repair the header from the file size and register it.
- `ended_at` comes from the mtime of the last segment.
- Known loss: the `gap_ms` of the segment open at crash time is lost (recorded as 0), since
  rows are written when a segment closes.

## CLI

- `record --title "..." [--course "..."]`: start recording, show level and elapsed time on one
  line, Ctrl+C stops cleanly.
- `sessions`: list sessions with status and duration.
- `recover`: run recovery manually.
- Standard library only (argparse, `\r` status line).

## Data directory

Default `%LOCALAPPDATA%\StudyScribe`, overridable with `--data-dir` or `STUDYSCRIBE_DATA_DIR`.
The CLI always prints the path in use.

## Known limits

- Downmix by mean: on 5.1 output the centre channel (voice) drops by about 15 dB. Not an issue
  with stereo headphones or speakers.
- No surplus correction during continuous playback (see timeline alignment).
- Anti-standby does not cover the display or a manual sleep.

## Unverified assumptions (check during capture work)

- (Verified, see findings: no packets during silence; `is_active()` turns False when the
  recorded device vanishes.)
- That a manual switch of the default output leaves the old stream active (handled anyway by
  the default-ID poll).
- Whether lowering an app's volume lowers the recording, and what a muted master volume does
  (probably driver dependent).
- DRM-protected content might be captured as silence (unlikely for course platforms).
- That NTFS mtime after a kill matches the last write (used for `ended_at` and the ±1 s check).

## Findings so far

- Capture piece (2026-10-28), Realtek loopback at 48 kHz stereo while audio was playing: about
  100 callbacks per second (10 ms packets), longest gap between packets 12-13 ms, no status
  flags. Audio written matched wall time (5.70 s for 5.70 s).
- Killing the parent process (TerminateProcess): the capture child noticed within 0.2 s,
  stopped cleanly and left a WAV with a consistent header.
- 180 s probe by the user (video paused, headphones unplugged and replugged, Ctrl+C):
  - Silence: after a pause the player kept sending zero-filled packets for about 10 s, then
    packets stopped entirely while `is_active()` stayed True; 19.8 s were missing without
    filling. The ~10 s tail is the player's behaviour, not WASAPI's: other players may stop at
    once or never.
  - Unplug: `is_active()` turned False within about 0.5 s, no error, no hang. The dead stream did
    not recover when the endpoint came back: reopening must be explicit. Closing the dead stream
    at the end raised nothing.
  - Clock drift: `missing` stayed within ±0.02 s over 56 s of continuous playback. This only
    rules out drift above a few hundred ppm; measure it in the long test.
  - Ctrl+C: clean stop, exit code 0, no traceback from the child.
- PortAudio re-init + open + close of the loopback: 25-65 ms.

## Tests

- Unit (pytest, no audio hardware): downmix/resample (length, 1 kHz FFT peak preserved,
  int16 clipping); aligner with a fake clock (sub-threshold jitter, pauses, resume, device-switch
  gaps, 10 simulated minutes with random pauses staying within 150 ms of real time);
  segmentation (boundaries, contiguity, gap_ms split across two segments); WAV header repair
  (odd trailing byte, foreign header rejected); migrations (idempotence, rollback, status
  CHECK); recovery on a fake session.
- Manual, interactive script: walks through the checklist (video paused mid-way, headphones
  unplugged and replugged in both directions, `powercfg /requests`, process kill), timestamps
  each step, then after `recover` checks total segment duration vs real time (±1 s),
  contiguity, header consistency, and that the silence→speech transition falls near the moment
  the video was resumed (alignment proof).
