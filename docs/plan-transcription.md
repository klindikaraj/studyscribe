# Step 2 plan: transcription

Agreed plan for step 2 of the roadmap (approved 2026-11-01). Update it when a decision changes.

## Goal

Transcribe a recorded session locally on the GPU, with timestamps on the session timeline and no
hallucinated text on silence, into SQLite plus SRT and Markdown exports. Roadmap: "a 1 h lecture
transcribed with correct timestamps and no hallucinated repeats on silence". The acceptance test
uses the 3 h recording of step 1 (session `20261030-134407-4ba5ed`, 2:57:34):

a. speed: total time to transcribe the 2:57:34, reported;
b. spot check of 5 points on different topics, listening against the text;
c. zero hallucinations in the 5 minute pause (about 0:29:33 to 0:34:44): no invented sentence in
   that interval.

## Decisions

1. Transcription at the end of a session: `studyscribe transcribe <session_id>` on a recorded
   session. The session is read as one continuous signal, so no word is cut where one WAV file
   ends and the next begins. Live transcription comes in step 4: the modules are designed so it
   can be added without rewrites.
2. Local only: faster-whisper on the GPU (RTX 3060, 12 GB), `large-v3-turbo` in float16 by
   default, configurable. A `Transcriber` interface leaves room for a cloud backend (not
   implemented).
3. Language: detected once per session on the first minutes of speech (after the VAD), not on
   the first 30 s of audio, which can be music or silence. `--language en|it` overrides it.
   Mixed Italian/English sessions are rare: their language is forced by hand.
4. Output: rows in SQLite (`transcript_segments`) plus `exports/transcript.srt` and
   `exports/transcript.md` in the session folder.
5. The WAV files are never modified. Level normalisation applies only to the copy fed to the ASR.
6. The 3 h test recording is never modified: development and tests use a copy of its data
   directory, `3h-copy`, next to the original.
7. Agreed with the user on 2026-11-01: windows of about 5 minutes cut in pauses (see Windows);
   transcription state in its own table, `sessions.status` unchanged (see Database); cuBLAS as
   the `cuda` extra, documented in the README; an `export` command, with `--audio` to join the
   segments into one audio file; hints off by default.

## Findings before the plan (verified 2026-11-01)

Checked against PyPI, the Windows build script of CTranslate2 4.8.4 and the source of
faster-whisper 1.2.1. Some of them change details of the original brief.

- cuDNN is not needed. The Windows wheel of ctranslate2 4.8.4 is built with CUDA 12.8,
  `WITH_CUDNN=OFF` and `CUDA_DYNAMIC_LOADING=ON`: Conv1D has a pure CUDA implementation since
  4.6.3. It needs only cuBLAS 12 (`cublas64_12.dll`, `cublasLt64_12.dll`), which the
  `nvidia-cublas-cu12` wheel provides. The installed driver (617.42, CUDA 13.4) runs programs
  built with CUDA 12.8.
- ctranslate2 loads cuBLAS with `LoadLibraryA("cublas64_12.dll")`: the plain search order,
  which ignores `os.add_dll_directory` (if `CUDA_PATH` is set, `%CUDA_PATH%\bin` is searched
  first). The DLLs of the pip wheel are not on that path, so `transcribe/cuda.py` preloads them
  by full path with `ctypes` before faster-whisper is imported; `LoadLibraryA` by name then
  returns the module already loaded. No PATH change and no system install.
- The newest `nvidia-cublas-cu12` (12.9.3.13, 2026-10-30) has no Windows wheel. pip skips it and
  takes 12.9.2.10 (553 MB).
- `no_speech_prob`, `avg_logprob` and `compression_ratio` of a faster-whisper segment belong to
  its decoding window (up to 30 s): all the segments of a window share them. In sequential mode
  faster-whisper already skips windows with `no_speech_prob > 0.6` and `avg_logprob < -1.0`, so
  the same rule applied afterwards never fires there. A segment's own confidence comes from its
  word probabilities (`word_timestamps=True`).
- Built-in language detection: with `vad_filter=True` it runs on speech, but only on the first
  30 s. With `language_detection_segments > 1` it stops at the first 30 s whose top language is
  above 0.5 and returns only that segment's probabilities, so a song in an intro decides the
  language. Hence the averaging described under Language.
- `BatchedInferencePipeline` forces `condition_on_previous_text=False` and supports `vad_filter`
  and `word_timestamps`, but drops the temperature fallback (the main defence against
  repetition loops), `hallucination_silence_threshold` and the no-speech skip. It is not used;
  it would be considered only if the sequential speed turned out not to be enough.
- The model alias `mobiuslabsgmbh/faster-whisper-large-v3-turbo` now redirects to
  `dropbox-dash/faster-whisper-large-v3-turbo` (MIT licence, not gated, `model.bin`
  1,617.9 MB). The app pins that repository at revision
  `0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf` instead of relying on the alias. Piece 1 checks
  that `model.bin` matches the original mobiuslabsgmbh upload (hash and size).
- huggingface-hub 2.4.0 still accepts the arguments faster-whisper passes to
  `snapshot_download` through `WhisperModel(download_root=...)` (`cache_dir`, `tqdm_class`).
  `local_dir_use_symlinks`, removed in 2.x, is passed only by `download_model(output_dir=...)`,
  which the app never calls.
- The test PC has 16 GB of RAM (Ryzen 5 5600G). One call on 3 h of audio would hold about
  680 MB of float32 samples plus 545 MB of mel features, and copies of both.

## Pieces (one at a time, stop after each with a test summary)

0. Fix `--check` (from step 1) on sessions recorded without a guided run
1. GPU check: dependencies, model, one minute of the copy: time, VRAM, language
2. Transcription core: continuous reading, normalisation, language detection on speech,
   anti-hallucination filters, timestamp mapping; unit tests without a GPU where possible
3. Database and CLI: migration v2, `transcribe` with `--language` and `--model`, progress line,
   recovery and resume after an interruption
4. Exports: `.srt`, `.md`, list of dropped segments, `export` command, `export --audio`
5. Acceptance test on the copy of the 3 h recording (speed, 5 points, the pause); A/B test of
   hints
6. Italian: a test on an Italian recording, which the user records when asked

What each piece contains and what the user tests: see "Pieces in detail".

## Dependencies

| Package | Constraint | Resolves to | Notes |
|---|---|---|---|
| faster-whisper | `>=1.2.1,<2` | 1.2.1 | brings onnxruntime 1.31.1 (Silero VAD on the CPU), av 19.0.2, tokenizers 0.23.3, huggingface-hub 2.4.0 |
| ctranslate2 | `>=4.6.3,<5` | 4.8.4 | 4.6.3 is the first without cuDNN; 5.x may move to CUDA 13 |
| av | `>=11` | 19.0.2 | already required by faster-whisper; declared because `export --audio` writes FLAC with it |
| nvidia-cublas-cu12 | `>=12.8,<13`, `cuda` extra, Windows only | 12.9.2.10 | brings `nvidia-cuda-nvrtc-cu12` |

Install with `.venv\Scripts\python -m pip install -e ".[dev,cuda]"` (documented in the README).
The model goes to the standard Hugging Face cache (`%USERPROFILE%\.cache\huggingface\hub`),
shared by every data directory. Without Windows developer mode Hugging Face warns that it cannot
create symlinks and copies the files instead, which is harmless. About 2.3 GB of downloads in
all.

## Modules

```
src/studyscribe/
  audio/reader.py       SessionAudio: the session as one int16 signal across segment files
  transcribe/
    engine.py           Transcriber protocol, AsrSegment / AsrWord, TranscriberError
    cuda.py             Windows: preload cuBLAS from the nvidia-cublas-cu12 wheel
    whisper.py          FasterWhisperTranscriber (the only module importing faster_whisper),
                        model loading, GPU error messages
    vad.py              thin wrapper over faster_whisper.vad (Silero)
    levels.py           level normalisation of the ASR copy (pure)
    windows.py          window cuts in pauses (pure)
    language.py         speech sample for detection, averaging of probabilities (pure)
    filters.py          anti-hallucination rules (pure)
    pipeline.py         run_transcription: windows -> engine -> filters -> commit per window
    export.py           transcript.srt, transcript.md, transcript-dropped.txt, audio.flac
scripts/transcribe_probe.py        piece 1: one WAV, timings, VRAM (nvidia-smi), language
docs/manual-test-transcription.md  acceptance checklist (piece 5)
```

Changed: `pyproject.toml` (dependencies, `cuda` extra, `gpu` pytest marker), `db/migrations.py`
(V2), `db/repo.py`, `session.py` (transcription lock byte), `recovery.py`, `cli.py`,
`README.md`, `CLAUDE.md` (at the end).

## Flow of `studyscribe transcribe <session_id>`

1. Open the database (migration v2), run recovery (sessions and orphaned transcription runs),
   print the data directory.
2. Check the session: refuse `recording` (live transcription is step 4) and `failed`. Run
   `verify.manifest_problems` and stop, listing them, if there are any.
3. Take the transcription lock (byte 2 of `session.lock`). If it is held, another `transcribe`
   is running on that session.
4. Load the model. If that fails nothing is written to the database and a clear message is
   printed (see GPU errors).
5. If an interrupted run with the same options exists, resume it from `done_ms`; otherwise
   delete it and create a new run. `--restart` always starts from scratch.
6. Language: forced, saved in the run being resumed, or detected.
7. Window loop: read, normalise the copy, transcribe, map the times to the session timeline,
   filter. One transaction writes the window's segments and advances `done_ms`. The progress
   line is updated on every segment.
8. Completion, in one transaction: the new run becomes `done`, the previous done run and its
   segments are deleted, `sessions.asr_engine`, `asr_model` and `language` are updated. Then the
   exports are written.
9. Ctrl+C or an error halfway (e.g. out of GPU memory): the run stays `interrupted` with the
   windows already committed, and the same command resumes it. Exit code 1.

It runs in the CLI process for now. `run_transcription` takes a progress callback and no CLI
objects, so step 4 can run it in a worker process.

## Reading the session (`audio/reader.py`)

- Segment N starts at sample (N-1)·960,000 by construction. `SessionAudio.read(start, n)`
  returns int16 samples across file boundaries; `total_frames` is the sum of the files.
- Files are opened read-only. A test checks that the size, mtime and content of the WAV files
  are unchanged after a transcription.
- `gap_ms` is not used. The timeline has no holes (device-switch gaps are silence on it), and
  `gap_ms` is not an exact measure of missing audio anyway (see "Known limits" in
  `docs/plan-recording.md`).

## Level normalisation (`transcribe/levels.py`)

Carried over from step 1: the loopback records after the device volume, so the same lecture was
about -20 dBFS on the Realtek output and -40 dBFS on the monitor. Normalisation applies to the
float32 copy fed to the VAD and to Whisper; the WAV files stay as captured.

- Frame level: RMS of 50 ms frames. Block level: 90th percentile of the frames above -70 dBFS
  within ±10 s of each 10 s block (±30 s, as first planned, left the first 30 s after a 20 dB
  drop unboosted: the percentile stays on the louder side).
- Gain towards -20 dBFS, limited to 0..+30 dB. A block whose level is below -60 dBFS gets 0 dB,
  so faint noise is not boosted into something the VAD or Whisper may take for speech.
- No sample above -1 dBFS: each block's gain is capped by its own peak and by its neighbours'
  (the gain is interpolated between block centres, so a sample near a block boundary sees both).
- Gains in dB are interpolated linearly between block centres: no steps, and a level change
  such as a device switch is followed within about 10 s.
- Starting values, to be checked on the 3 h recording around the unplug and replug (1:26:07 to
  1:29:28), where the level drops by about 20 dB.

## Windows (`transcribe/windows.py`)

- The session is processed in windows of at most 300 s. A window ends in the longest pause the
  VAD finds in its last 30 s (cut in the middle of that pause). If there is none, it ends in the
  middle of the quietest 0.5 s of that stretch, with a warning. The last window ends with the
  audio.
- Windows do not overlap and are cut only in pauses, never at file boundaries, so no word is
  split between two windows or between two WAV files.
- No context is lost: with `condition_on_previous_text=False` Whisper carries no text from one
  30 s decoding window to the next anyway.
- The cursor (the start of the next window) is the resume point. In step 4 the same cursor waits
  for new audio before cutting the next window.
- Why not one call on the whole session: about 2 GB of memory on 3 h, no progress, no resume,
  nothing reusable for live transcription.

## Language (`transcribe/language.py`)

- Silero VAD, with the same options as the transcription, on the normalised copy from the start
  of the session, until 180 s of speech are collected or the audio ends.
- That speech is cut into chunks of 30 s (at most 6). Whisper gives the language probabilities
  of each chunk, and the language with the highest average wins.
- Warning when the winner is below 0.6 or is neither English nor Italian, suggesting
  `--language`. No speech at all: the run ends with no segments and a warning.
- `--language` accepts any Whisper language code (the help lists en and it); `auto` is the
  default.
- The run stores the language, its source (`forced` or `detected`) and the top 3 probabilities.

## Decoding (`transcribe/whisper.py`)

`WhisperModel.transcribe`, sequential, on each window:

- `vad_filter=True` with the default `VadOptions` (threshold 0.5, minimum silence 2 s, speech
  padding 400 ms), `word_timestamps=True`, `condition_on_previous_text=False`, and `language`
  always set (never detected per window);
- `beam_size=5`, temperature fallback 0.0, 0.2, ... 1.0 (the default);
- `no_speech_threshold=0.6`, `log_prob_threshold=-1.0`, `compression_ratio_threshold=2.4`;
- `hallucination_silence_threshold=None`: the VAD removes silence before decoding. Tried only
  if piece 5 shows hallucinations;
- `initial_prompt=None`, and `hotwords=None` unless `--hints` is given (see Hints).

Models: `large-v3-turbo` maps to `dropbox-dash/faster-whisper-large-v3-turbo` at the pinned
revision; other names go to faster-whisper's own list. `--compute-type` (default float16) and
`--device` (default cuda) are passed through.

## Anti-hallucination filters (`transcribe/filters.py`)

Pure functions on the segments of a window plus the last kept segments before it. A dropped
segment stays in the database with the rule that dropped it (`dropped`) and is listed in
`exports/transcript-dropped.txt`, so a rule that is too strict shows up, and thresholds can be
tuned later from the stored metrics without transcribing again.

| Rule | Drops | Why this threshold |
|---|---|---|
| silence | `no_speech_prob > 0.6` and `avg_logprob < -1.0` | the values of OpenAI's reference implementation for long-form audio, tuned by the model's authors, and faster-whisper's defaults. Redundant in sequential mode; kept for the batched pipeline and other backends |
| repetition | `compression_ratio > 2.4` after every fallback temperature | same source: text that compresses this well is a loop |
| duplicates | the same normalised text in 3 or more consecutive segments: all but the first | the shape of a loop across decoding windows |
| known phrases | a whole segment equal to a typical subtitle hallucination ("Thanks for watching", "Subtitles by the Amara.org community", "Sottotitoli creati dalla comunità Amara.org", ...), with at least 2 s without words before and after it | the most frequent Whisper hallucinations, learned from subtitle files; a heuristic |
| low confidence | mean word probability below 0.35, and either shorter than 1.5 s or with at least 2 s without words on both sides | a heuristic |

The last two rules are assumptions to validate on the 3 h recording.

## Timestamps

- Every window knows its first sample on the session timeline. faster-whisper returns times
  relative to the audio it was given, already mapped back from the VAD-filtered audio to that
  audio (`restore_speech_timestamps`). So `start_ms = round(window_start_frame / 16 +
  start_s * 1000)`, and the same for the end of the segment and for every word.
- The session timeline is the one the segments define: 0 is the first sample of `000001.wav`,
  and segment N covers `(N-1)·60000` to `N·60000` ms.
- To listen to the point at `t` ms: file `audio/NNNNNN.wav` with `NNNNNN = floor(t / 60000) + 1`,
  at second `(t mod 60000) / 1000` of that file; or `exports/audio.flac` (see Exports) at `t`.
- `gap_ms` plays no part (see "Reading the session").
- Precision: word times from faster-whisper's alignment are usually within a few tenths of a
  second; faster-whisper rounds them to 10 ms.

## Hints (title, course, keywords)

- Off by default: prompts can make Whisper write their words where nobody says them, especially
  next to silence or music.
- `--hints TEXT` passes the text as `hotwords`, which apply to every 30 s decoding window
  (`initial_prompt` would affect only the first 30 s of each window).
- A/B test in piece 5 on three 10 minute excerpts of the 3 h recording, one of them across the
  edges of the pause: no hints against
  `--hints "Learn JavaScript - Full Course for Beginners. JavaScript, const, let, typeof, NaN, array, object"`.
  Compare the texts (technical terms above all), look for hint words where they are not spoken
  and for anything inside the pause. Then decide whether the session title and course become
  automatic hints.

## GPU errors

A pure function (tested) turns ctranslate2 exceptions into a short message without a traceback:

- out of memory: suggest `--compute-type int8_float16` or `--model medium` / `small`, and
  closing applications that use the GPU. Halfway through a run, the run stays interrupted and
  resumable;
- no CUDA device: a hint about the driver, plus `--device cpu --model small --compute-type int8`
  (slow);
- cuBLAS not found: `pip install -e ".[cuda]"`.

Checked before loading the model: `ctranslate2.get_cuda_device_count()` and
`ctranslate2.get_supported_compute_types("cuda")`.

## Transcriber interface (`transcribe/engine.py`)

```python
class Transcriber(Protocol):
    engine: str  # e.g. "faster-whisper"
    model: str   # e.g. "dropbox-dash/faster-whisper-large-v3-turbo@0a363e9"

    def language_probabilities(self, speech: np.ndarray) -> dict[str, float]:
        """Language probabilities of one chunk of up to 30 s of speech (float32, 16 kHz)."""

    def transcribe(
        self, audio: np.ndarray, language: str, hints: str | None
    ) -> Iterator[AsrSegment]:
        """Segments with times relative to `audio`; the engine applies its own VAD."""
```

`AsrSegment`: start and end (s), text, words (start, end, text, probability), `avg_logprob`,
`no_speech_prob`, `compression_ratio` (None when an engine does not provide them). One
implementation, `FasterWhisperTranscriber`; tests use a `FakeTranscriber`.

## Database (schema v2)

```sql
CREATE TABLE transcription_runs (
  id              INTEGER PRIMARY KEY,
  session_id      TEXT NOT NULL REFERENCES sessions(id),
  pass            TEXT NOT NULL DEFAULT 'final' CHECK (pass IN ('live', 'final')),
  status          TEXT NOT NULL CHECK (status IN ('running', 'interrupted', 'done')),
  engine          TEXT NOT NULL,           -- faster-whisper
  model           TEXT NOT NULL,           -- dropbox-dash/faster-whisper-large-v3-turbo@0a363e9
  compute_type    TEXT NOT NULL,           -- float16
  language        TEXT,                    -- NULL in a done run: no speech found
  language_source TEXT CHECK (language_source IN ('forced', 'detected')),
  language_probs  TEXT,                    -- JSON: top 3 of the detection
  options_json    TEXT NOT NULL,           -- VAD, decoding, filters, hints: all that shapes the output
  audio_ms        INTEGER NOT NULL,
  done_ms         INTEGER NOT NULL DEFAULT 0,  -- audio before this is transcribed and committed
  started_at      TEXT NOT NULL,
  finished_at     TEXT,
  processing_s    REAL NOT NULL DEFAULT 0, -- summed over resumed attempts
  error           TEXT                     -- what interrupted it last
);
CREATE UNIQUE INDEX idx_runs_done ON transcription_runs(session_id, pass) WHERE status = 'done';
CREATE UNIQUE INDEX idx_runs_open ON transcription_runs(session_id, pass) WHERE status <> 'done';

CREATE TABLE transcript_segments (
  id                INTEGER PRIMARY KEY,   -- the "S0123" the notes cite in step 3
  run_id            INTEGER NOT NULL REFERENCES transcription_runs(id) ON DELETE CASCADE,
  start_ms          INTEGER NOT NULL,      -- session timeline
  end_ms            INTEGER NOT NULL,
  text              TEXT NOT NULL,
  words_json        TEXT,                  -- [[start_ms, end_ms, "word", probability], ...]
  avg_logprob       REAL,                  -- per decoding window, not per segment
  no_speech_prob    REAL,                  -- per decoding window
  compression_ratio REAL,                  -- per decoding window
  dropped           TEXT                   -- NULL = kept, else the rule that dropped it
);
CREATE INDEX idx_transcript_run_time ON transcript_segments(run_id, start_ms);
```

- Deviations from the original design notes: no `audio_seq` (it follows from `start_ms`, and
  with windows a transcript segment has no single source segment); `pass` and `engine` belong to
  the run, not to each segment; a segment reaches its session through its run.
- Atomic replacement: readers take only the segments of the session's single `done` run (for a
  pass). A new run is invisible until it becomes `done`, in the same transaction that deletes
  the previous one, so a re-transcription replaces the old text at once and never duplicates it.
  The partial unique indexes forbid two done runs, or two unfinished ones, per session and pass.
- `sessions.status` is not changed by transcription. Live transcription (step 4) runs while a
  session is `recording`; a single status would lose `recorded` versus `interrupted`; and
  `transcribed` is not in the v1 CHECK, which SQLite can change only by rebuilding `sessions`,
  referenced by `audio_segments`. `transcribing` and `ready` stay unused.
- `audio_segments.transcribed` stays unused until step 4.
- Transcript segment IDs change when a session is transcribed again: the notes of step 3 must
  record the run they cite.

## Recovery and locks

- Byte 2 of `session.lock` is held by the process transcribing the session.
- A run in `running` whose lock byte is free becomes `interrupted`, at the start of
  `transcribe`, `sessions` and `recover`. The previous done run is left alone.
- A run is resumed only with the same engine, model, compute type, language option, options
  (`options_json`) and audio length. Otherwise the unfinished run is deleted and a new one
  starts, with a message.

## CLI

```
studyscribe transcribe SESSION_ID [--language auto|en|it|...] [--model large-v3-turbo]
        [--compute-type float16] [--device cuda|cpu] [--restart] [--hints TEXT]
studyscribe export SESSION_ID [--audio]
```

Example output (numbers made up):

```
Model large-v3-turbo (float16, cuda) loaded in 4.1 s
Language: en (detected on 180 s of speech: en 0.97, it 0.02, de 0.00)
Transcribing [########------------] 41%  1:12:40 / 2:57:34  elapsed 1:52  left ~2:40  39x
Done: 2:57:34 in 4:31 (39x real time), 2140 segments, 12 dropped (see exports/transcript-dropped.txt)
```

- On a terminal one progress line is redrawn: position, percentage, elapsed time, estimate,
  speed. Otherwise one line per window.
- `sessions` gets a TRANSCRIPT column: `-`, the language of the done run (e.g. `en`), `running`,
  `interrupted 47%`.
- Exit code 0 only when the run is done.

## Exports (`<session>/exports/`, each written to a temporary file, then renamed)

- `transcript.srt`: the kept segments. A segment longer than 84 characters or 7 s is split at
  word boundaries. UTF-8.
- `transcript.md`: a header (title, course, date, duration, language and how it was chosen,
  model), then paragraphs starting with `[H:MM:SS]`; a new paragraph after a pause of 3 s or
  more, or after about 90 s. One line explains how to find a point in the audio.
- `transcript-dropped.txt`: every dropped segment with its time, rule, metrics and text.
- `audio.flac` (`export --audio`): the session's segments joined in order, i.e. the same samples
  and the same timeline as the transcript. The `.srt` plays against it in VLC, and any time `t`
  of the transcript is at `t` in the file; step 3 needs it for the player anyway. FLAC because
  it is lossless and sample-exact (no encoder delay or padding as with MP3, AAC or Opus, so no
  shift), about half the size of WAV for speech (digital silence almost vanishes), plays and
  seeks in VLC and in browsers, and is written with PyAV, already installed with faster-whisper,
  so no new package. A test decodes it back and compares it with the segments.
- `transcribe` writes the three text exports; `export` rewrites them from the database without
  transcribing; `export --audio` also writes `audio.flac`.

## Pieces in detail

| # | What | What the user tests |
|---|---|---|
| 0 | `--check` on a session without a guided run: the header says there are no guided-run notes, so the step and switch checks are skipped; the gap and warning lines say that no capture events were recorded, instead of "reported 0 ms" and "none"; checklist and tests updated. Copy of the 3 h test data directory to `3h-copy` | `--check` on the copy |
| 1 | Dependencies and model download (after confirmation), `transcribe/cuda.py`, `scripts/transcribe_probe.py` on one 60 s WAV of the copy; `model.bin` hash and size against the original mobiuslabsgmbh upload | the probe: load time, transcription time, VRAM, language, text |
| 2 | reader, levels, windows, language, filters, pipeline (no database yet), unit tests without a GPU (`FakeTranscriber`), one optional `gpu` test | `pytest`; a 10 minute excerpt through the probe |
| 3 | migration v2, repo, lock, recovery, `transcribe` command, progress line, resume | Ctrl+C halfway, then resume; killing the process, then `sessions` |
| 4 | srt / md / dropped exports, `export`, `export --audio` | the files; the `.srt` in VLC against `audio.flac` |
| 5 | 3 h acceptance test on the copy; A/B test of hints | the total time; listening to the 5 points; the text in the pause |
| 6 | Italian: the user records an Italian lecture as asked | the detected language, a spot check of the text |

Piece 5:

- The 3 h recording is "Learn JavaScript - Full Course for Beginners" by freeCodeCamp.org (one
  English voice, many JavaScript terms; the video lasts 3:26:43, the recording covers its first
  2:57:34).
- The 5 points, on different topics, come from the chapters in the video description. The
  recording timeline differs from the video's: by an offset at the start (when play was pressed)
  and, after 0:29:33, by the 5 min 11 s of the pause. Both are measured from the transcript,
  not assumed.
- For each point: time on the timeline, WAV file and second, time in `audio.flac`, chapter,
  transcript text. The user listens.
- The pause: every kept and dropped segment between 0:29:33 and 0:34:44.

## Tests

- Unit (pytest, no GPU, no model): reader across file boundaries (offsets, partial last segment,
  missing file, files unchanged); normalisation (target level, no clipping, silence untouched, a
  20 dB step followed); window cuts (in pauses, the fallback, the last window, resume from a
  cursor); language averaging; every filter rule; time mapping; SRT and Markdown formatting and
  cue splitting; FLAC round trip; migration v2 (constraints, idempotence); atomic replacement
  (old transcript visible until the commit, intact after a failure); recovery of an orphaned
  run; resume; CLI with a `FakeTranscriber`; GPU error messages.
- `gpu` marker, skipped unless asked for: load the real model and transcribe a short sample.
- Manual: `docs/manual-test-transcription.md` and the acceptance test (piece 5).

## Risks and unverified assumptions

- That the ctranslate2 wheel links the CUDA runtime statically (no `cudart64_12.dll` needed) and
  that its "Common" architecture list includes sm_86 (RTX 3060): checked in piece 1.
- Speed and VRAM are estimates (3-8 minutes for 3 h, 2.5-3.5 GB of VRAM): measured in piece 1.
- A percentile level can be wrong with background music. The limits (0 dB below -60 dBFS, at
  most +30 dB) bound the damage.
- Language detection can be fooled by accented English or a sung intro: hence the warning and
  `--language`.
- After about 30 s of speech without a pause of 0.5 s, a window cut may split a word (reported).
- The known-phrase and low-confidence rules may drop real speech: dropped segments are kept and
  listed for review.
- Out of GPU memory because another application uses the GPU: the run stays interrupted and
  resumable.
- Lectures played at 1.5x-2x transcribe worse; out of scope.

## Findings so far

- Piece 0 (2026-11-01): `--check` on the copy of the 3 h session now says that there are no
  guided-run notes and that the step, switch and capture event checks are skipped; the gap line
  shows only the manifest total (248 ms), the warnings line says they were not recorded. The
  copy `3h-copy` matches the original file by file (SHA-256 of 183 files).
- Piece 0: the test suite showed two step 1 timing assumptions that fail on a busy PC: Windows
  releases the locks of a killed process asynchronously, and the capture process can take more
  than 1.5 s to start. Both fixed; details in `docs/plan-recording.md`, Findings.
- Piece 1 (2026-11-02), `scripts/transcribe_probe.py` on segment 10 (0:09:00-0:10:00) of the
  copy, RTX 3060, float16:
  - Installed: faster-whisper 1.2.1, ctranslate2 4.8.4, nvidia-cublas-cu12 12.9.2.10 (+ nvrtc
    12.9.86), onnxruntime 1.31.1, av 19.0.2, huggingface-hub 2.4.0. With only the preloaded
    cuBLAS DLLs the GPU works: the CUDA runtime is linked statically and sm_86 is supported
    (both assumptions confirmed). ctranslate2 offers float16, bfloat16, int8_float16 and more.
  - `model.bin`: 1,617,884,929 bytes, SHA-256 `e76620f8…c018da`, equal to the hash published at
    the pinned revision. The repository history shows mobiuslabsgmbh and dropbox-dash are the
    same repository (renamed): `model.bin` was uploaded on 2024-10-08 (commit 0c94664) and is
    unchanged since; the only later commit (0a363e9, 2025-11-05) edits the README. Download
    took 171 s.
  - Model load 1.9-2.2 s; VRAM +2.1 GB after loading, peak +2.35 GB while transcribing (3.4 of
    12 GB in use, 1.06 GB before).
  - 60 s transcribed in 2.36 s cold, 2.06 s warm (25-29x real time). On 3 h that would be about
    6 minutes; longer windows should do better (less per-call overhead): measured in piece 5.
  - Language: en with probability 1.00 (VAD kept 56.1 of 60 s).
  - Text accurate; "Beau" came out as "bow". The last segment ends with "here we are a sign":
    "assigning" cut by the end of the file, which is the case continuous reading avoids. Its
    avg_logprob (-0.52) and lowest word probability (0.28) are the worst of the minute.
  - Hugging Face warns that symlinks are unavailable (no developer mode) and copies files:
    harmless, as expected.
- Piece 2 (2026-11-02), pipeline mode of the probe on the copy (real model and VAD):
  - Language is detected only on speech: inside the pause (0:30:00-0:34:00) the result is "no
    speech" and the language stays unknown, never a guess from silence (a plain call on the
    silent segment 31 had answered "en 0.51" at random). On 0:09:00-0:11:00: en 1.00 on 116 s
    of speech, in 1.7 s.
  - Continuous reading completes the sentences cut at file ends: "here we are a sign" (end of
    segment 10) becomes "And then here we are assigning a variable" across 0:10:00, and "If
    that." (end of segment 88) becomes "If value is more than 10" across 1:28:00.
  - Normalisation on 1:27:00-1:29:00 (monitor output, -39 dBFS): without it the first four
    segments (1:27:00-1:27:25) have no punctuation or capitals and one segment runs 33 s
    (1:28:21-1:28:54); with it punctuation and capitals are there from the first line and no
    segment is longer than 9 s. The words are almost identical.
  - Across the pause (0:29:00-0:35:30): nothing between 0:29:32.4 and 0:34:46.3, the window cut
    fell inside the pause (0:33:45).
  - While testing, the level neighbourhood went from ±30 s to ±10 s (see Level normalisation).
