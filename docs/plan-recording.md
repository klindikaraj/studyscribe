# Step 1 plan: reliable recording

Agreed plan for step 1 of the roadmap. Update it when a decision changes.

## Goal

Record the system audio of a lecture for 1-3 hours without losing data, even if the
process crashes or the output device changes, with a timeline that stays aligned to real
time. "Done" means: a 3 h recording with headphones unplugged and replugged and a process
kill loses at most a few seconds and the timeline stays aligned.

## Pieces (one at a time, stop after each with a test summary)

1. Setup (done)
2. Capture: loopback in callback mode, downmix + resample, separate process, anti-standby (done)
3. Segments and manifest: 60 s WAV segments, SQLite schema v1 and migrations (done)
4. Silence filling and device changes (done)
5. Recovery (done)
6. CLI: `record`, `sessions`, `recover` (done)
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
- The child owns everything else: its session lock, anti-standby, stream, WAV writing,
  `audio_segments` rows, final session status. The parent holds its own session lock (see
  Recovery) from before the session row exists until the child has exited.
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

## Timeline alignment (`audio/timeline.py`, `TimelineAligner`, pure)

- `expected(t) = (t - t0) * 16000`; `frames` = samples on the timeline so far. All times are
  passed in, so tests drive it with made-up packets and clocks.
- Start of a packet's audio: `arrival - duration`, tightened by the following packets of the
  same drain (a packet ends no later than the next one starts). After a hiccup, packets held
  in the driver arrive together; this keeps such a burst from being taken for a hole.
- While audio flows: if `expected(packet start) - frames` exceeds 100 ms, write that much
  silence first, then the packet. Below the threshold do nothing, so jitter never moves audio.
- No data for 0.5 s (paused video, nothing playing yet, device being reopened): write silence
  up to `now - 100 ms`. Files keep growing and segments keep rotating during pauses; the
  margin avoids writing "into the future".
- The first packet after silence (start of the recording, a pause, a reopened stream) is
  placed exactly at its start time, threshold or not: nothing continuous is broken there.
  So a recording started before pressing play is filled from t0 and segment 1 starts at the
  start of the recording (found in the user's test of piece 3: without this, the whole
  timeline was shifted by the 1.1 s before the first packet).
- Before inserting silence the resampler is flushed and restarted, so the last ~20 ms of
  audio it holds land before the silence, not after it.
- At stop: flush, then fill up to the stop time, so the timeline ends at wall time.
- Error stays within about one threshold (100 ms) and does not accumulate, because every fill
  re-anchors to the clock.
- Real audio is never dropped. If the device clock runs faster than the system clock
  (ppm-level drift, about 0.3 s over 3 h at 30 ppm), the surplus is absorbed at the next pause.
  If it runs slower, about 100 ms of silence is inserted each time the deficit crosses the
  threshold (once every ~55 min at 30 ppm).

## Silence limit (auto-stop)

- After 30 minutes in a row without sound (`SILENCE_LIMIT_S`, configurable through
  `CaptureConfig.silence_limit_s`; `None` disables it) the recording stops by itself: reason
  `silence_limit`, session `recorded`, plus a warning saying when the last sound was.
- "Sound" = a packet peaking above -80 dBFS (`SOUND_FLOOR_DB`). Packets of digital silence do
  not count: players keep sending zeros after a pause (10 s seen in the tests) and some apps
  never stop, which would keep a forgotten recording alive. The counter starts at t0, so a
  recording where nothing ever plays stops too.
- The timeline is never filled past `last sound + limit`. A single clock jump bigger than the
  time left (e.g. waking from sleep, if `perf_counter` counted the sleep) is filled only up to
  the limit, and the recording stops; packets arriving after that are not part of the session.
  The warning mentions the jump when the clock moved more than a minute between two ticks.
- Agreed with the user (2026-10-29), added to piece 4 rather than to recovery.

## Segments

- Rotation by sample count: exactly 960,000 samples per segment, so segment N always starts
  at (N-1)·60000 ms. Files: `<data>/sessions/<session_id>/audio/000001.wav …`.
- Unbuffered writes (every tick reaches the OS), header patched every second, fsync every
  10 s. Maximum loss on a process kill: about one tick (< 100 ms).
- A row is inserted in `audio_segments` when a segment closes, with the path relative to the
  session folder (`audio/000001.wav`) so the data directory can be moved.
- Files are created on their first sample (no empty segments) and never overwritten.
- A database error never stops the recording: the capture process reports a warning and keeps
  writing audio; recovery registers the missing rows later. This includes failing to open the
  database at all.
- The parent creates the session row (`recording`, provisional `started_at`) and runs the
  migrations; the capture process only connects, sets `started_at` at timeline t0 and sets the
  final status: `recorded` (stop requested), `interrupted` (parent died or error after some
  audio), `failed` (error before any audio).

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
- Implementation (`audio/capture.py`, `audio/devices.py`):
  - One `LoopbackSource` per stream; after a loss the old one is closed, its queued packets
    are still written, then the gap opens.
  - The default output is read with `IMMDeviceEnumerator::GetDefaultAudioEndpoint(eRender,
    eMultimedia)`, the role PortAudio uses for its default output; `None` when there is no
    output at all. COM is initialised (apartment-threaded) on the capture thread.
  - If COM cannot be read: one warning, and the recording goes on without the default-ID poll
    (an unplugged device is still followed because its stream stops).
  - A failed reopen emits one warning per distinct error, not one per attempt.
  - Events for the parent: `device_lost` (reason `stream stopped` or `default output changed`)
    and `device_opened` (device, gap in ms, attempts).
  - The first stream must open (otherwise the session is `failed`); after that a lost device
    never stops the recording.

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
  title_key   TEXT NOT NULL,        -- NFC + casefold + NFC of the title, computed in Python
  platform    TEXT,
  url         TEXT,
  created_at  TEXT NOT NULL
);
CREATE UNIQUE INDEX idx_courses_title_key ON courses(title_key);

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
  table makes the insert fail. The unique index on `title_key` lets `--course "Calculus 1"`
  reuse the existing course instead of creating a duplicate. The key is computed in Python
  (Unicode NFC, case folding, NFC again) because SQLite's `COLLATE NOCASE` folds ASCII letters
  only: "Èlite" and "èlite" would have become two courses. Changed directly in v1, which had
  not been used yet.
- Session IDs are readable and sortable: `YYYYMMDD-HHMMSS-xxxxxx` (6 random hex digits).

## Recovery (`recovery.py`, `system/locks.py`)

- Runs automatically at the start of `record` and `sessions`, and on demand with `recover`
  (CLI piece). Until then: `scripts/capture_probe.py --recover`.
- Locks: `<session>/session.lock`, two bytes locked with `msvcrt.locking` (released by the OS
  when the holding process dies, also on TerminateProcess). Byte 0 is held by the owner (the
  process that started the recording), taken in `new_session` before the row is inserted;
  byte 1 by the capture process (`SessionSink`). Recovery touches a session only if it can
  take both, so it never "repairs" a recording still running in another terminal, nor one
  whose row was just created and whose capture process is still starting.
  Deviation from the first plan (one lock, held by the child): with a single child lock, a
  recovery started in another terminal during the ~1 s spawn of the child would have seen a
  `recording` row with a free lock.
- If only the capture process dies, its owner recovers the session itself, passing its own
  lock (`recover_session(..., owner_lock=...)`); recovery in other processes keeps skipping it.
- Sessions in `recording` with both locks free: read the mtime of the last segment first (the
  repair would change it), repair the header of every segment without a manifest row from the
  file size (half a trailing sample is cut; a header cut short is completed with 0 frames),
  add those rows, set `interrupted` (`failed` if there is no audio at all) and `ended_at` =
  that mtime. One transaction per session.
- Sessions already `recorded` / `interrupted`: only missing manifest rows are added (rows lost
  to a database error during recording).
- Segment N is placed at (N-1) full segments by construction. Reported as problems, without
  stopping the rest: files that are not our segments (left untouched), numbering holes, rows
  whose file is missing, a non-final segment shorter than 60 s, a missing session folder
  (`failed`). Problems alone do not make a session show up again on every run.
- Known loss: the `gap_ms` of a segment added by recovery is 0, since rows are written when a
  segment closes and only the capture process knew its gap.

## CLI (`cli.py`)

- Every command prints the data directory first.
- `record --title "..." [--course "..."] [--silence-limit MINUTES]`: recovers abandoned
  sessions, creates the session (holding the owner lock), prints session ID, folder, the
  silence limit and both PIDs, then starts the capture process. On a terminal one live line is
  redrawn with `\r`: elapsed time, level bar (-60 to 0 dBFS), current segment, gap; events
  (device lost/reopened, warnings, errors, stop) are printed on their own timestamped lines
  above it. When output is not a terminal only the events are printed.
- Ctrl+C asks the capture process to stop and waits for it (a second Ctrl+C keeps waiting).
  If the capture process exits abnormally, the CLI recovers the session itself as its owner.
  Exit code 0 only if the session ends `recorded` (Ctrl+C or silence limit).
- `--silence-limit` is in minutes (default 30); 0 disables the auto-stop.
- `sessions`: recovers first, then one row per session: ID, status, local start time, duration
  and gap (from the manifest), course, title. A session still recording shows the duration of
  its closed segments.
- `recover`: recovery on demand, one line per repaired session plus its problems.
- Standard library only (argparse, `\r` status line). Output that cannot be encoded (redirected
  to a file with the ANSI code page) is escaped instead of crashing.

## Data directory

Default `%LOCALAPPDATA%\StudyScribe`, overridable with `--data-dir` or `STUDYSCRIBE_DATA_DIR`.
The CLI always prints the path in use.

## Known limits

- Downmix by mean: on 5.1 output the centre channel (voice) drops by about 15 dB. Not an issue
  with stereo headphones or speakers.
- No surplus correction during continuous playback (see timeline alignment).
- Anti-standby does not cover the display or a manual sleep.
- A hiccup whose burst of delayed packets is split across two drains can still insert up to
  the hiccup's length of silence, shifting the audio after it by that much until the next
  pause. Not seen so far (longest gap between packets during playback: 12-19 ms).
- `gap_ms` can undercount by up to ~0.5 s per switch if a loss is noticed more than 0.5 s after
  the last packet: silence already written as ordinary silence cannot be relabelled.
- On a switch of the default output (e.g. replugging), audio that Windows already routes to the
  new device before the poll notices (up to 0.5 s) is not captured. If the old device keeps
  delivering packets meanwhile, that time is recorded as the old device's (silent) audio, not
  as gap. In the user's test the old device still had sound until the switch was noticed.
- The capture level follows the device volume (loopback is taken after it): the same lecture
  measured about -20 dBFS on the Realtek output and about -40 dBFS on the monitor.

## Unverified assumptions (check during capture work)

- (Verified, see findings: no packets during silence; `is_active()` turns False when the
  recorded device vanishes.)
- That a manual switch of the default output leaves the old stream active (handled anyway by
  the default-ID poll).
- Whether lowering an app's volume lowers the recording, and what a muted master volume does
  (probably driver dependent).
- DRM-protected content might be captured as silence (unlikely for course platforms).
- That NTFS mtime after a kill matches the last write (used for `ended_at` and the ±1 s check).
- That the eMultimedia default (what PortAudio opens) always equals the default shown in the
  sound settings. Windows sets the console and multimedia roles together, so it should.
- What `time.perf_counter` does across a system sleep. If it keeps counting, the jump on wake
  is filled with silence, but only up to the silence limit (see above), so never hours of it.

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
- Segments piece, tests by the user (2026-10-28):
  - 150 s with a video playing continuously: segments of 60 + 60 + 29.76 s, file durations
    equal to the manifest, `missing` between 0.01 and 0.04 s throughout, 0.00 at the end.
  - Ctrl+C at ~90 s: clean stop, exit code 0, segments consistent with the manifest. The
    video started ~1.1 s after the recording, so without silence filling the whole timeline
    was shifted by 1.1 s; this led to the "fill from t0" rule above.
- Silence filling piece: `DefaultOutputMonitor` returns the Realtek endpoint ID on the test PC.
  A 12 s session with nothing playing produced one 11.77 s segment of silence for 11.77 s of
  wall time (before silence filling the same test produced no segment at all).
- Silence filling piece, 150 s test by the user (2026-10-29):
  - Play pressed ~5 s after the start: `missing` stayed at 0.10 s while waiting, then 0.02 s.
  - Pause at ~41 s, resume at ~60 s: the player sent ~10 s of zeros, then the timeline was
    filled (silence +9.7 s) with `missing` at 0.10 s.
  - Headphones unplugged at ~91 s: `stream stopped`, reopened on the monitor, gap 174 ms,
    first attempt. The player did not pause.
  - Replugged at ~111 s: `default output changed`, reopened on Realtek, gap 250 ms, first
    attempt. Reopening the just-returned Realtek endpoint took longer than the 25-65 ms seen
    with stable devices.
  - Ctrl+C: exit 0, `recorded`, `missing` 0.00, manifest total 150.232 s = wall time, manifest
    gap 414 ms (both switches fell in segment 2).
  - The stats line of the replug tick showed `missing` -0.02 s: it used a clock reading taken
    before the old stream was closed, while the packets drained after the close and the
    resampler's tail were already written. Cosmetic; the loop now reads the clock again after
    device handling.
- Silence limit: with `--silence-limit 3` and nothing playing, the probe stopped by itself
  after 3.0 s of timeline, status `recorded`.
- Recovery piece, tests by the user (2026-10-29), video playing:
  - `taskkill /F` on the capture process at ~90 s: the probe recovered the session itself:
    `interrupted`, 90.251 s, segment 2 repaired and added, child exit code 1.
  - `taskkill /F /T` on the probe at ~65 s (both processes; the probe's own parent was the venv
    launcher), then `--recover`: `interrupted`, 65.387 s; a second `--recover`: nothing to do.
  - `--silence-limit 20` with nothing playing: stopped by itself at 20.0 s, `recorded`, warning.
- Recovery piece, on the test PC (nothing playing, so the timeline was silence):
  - `taskkill /F` on the capture process after ~4 s: the probe (owner) recovered the session:
    `interrupted`, one segment of 3.972 s, header repaired; `ended_at - started_at` = 4.07 s
    (the 0.1 s difference is the idle-fill margin).
  - `taskkill /F /T` on the probe (both processes) after ~4 s, then `--recover`: `interrupted`,
    3.955 s, header repaired; `ended_at - started_at` = 4.06 s.

## Carried over to step 2 (transcription)

- Normalise the level before transcription (no code in step 1). The loopback records after the
  device volume, so the same lecture can be ~20 dB quieter on one output than on another (seen
  between the Realtek output and the monitor), and the user's volume setting changes it too.
  Store the audio as captured; normalise only the copy fed to the ASR (e.g. gain to a target
  loudness per segment or per window, never clipping).

## Tests

- Unit (pytest, no audio hardware): downmix/resample (length, 1 kHz FFT peak preserved,
  int16 clipping); aligner with a fake clock (sub-threshold jitter, initial silence before the
  first packet, pauses, resume, bursts after a hiccup, device-switch gaps, 10 simulated minutes
  with random pauses staying within 150 ms of real time); capture loop on a fake clock (initial
  silence, pause, device vanishing, default output changing, failed reopens, stop during an
  outage, COM failures);
  segmentation (boundaries, contiguity, gap_ms split across two segments); WAV header repair
  (odd trailing byte, foreign header rejected); migrations (idempotence, rollback, status
  CHECK); recovery on a fake session.
- Manual, interactive script: walks through the checklist (recording started a few seconds
  before pressing play, so the first segment must start at the start of the recording; video
  paused mid-way; headphones unplugged and replugged in both directions; `powercfg /requests`;
  process kill), timestamps
  each step, then after `recover` checks total segment duration vs real time (±1 s),
  contiguity, header consistency, and that the silence→speech transition falls near the moment
  the video was resumed (alignment proof).
