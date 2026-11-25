# StudyScribe

A personal, local-first note-taking app for online courses. It records what your
PC is playing (the lecture video), transcribes it locally on the GPU and turns
the transcript into study notes with clickable timestamps.

Inspired by [Granola](https://www.granola.ai/), but written from scratch: it
contains no code from Granola and is not affiliated with it.

## Status

The command-line MVP is complete (steps 1-3): it records the system audio, transcribes it
locally, writes study notes with a local model (Ollama) and exports them as `notes.html`, a page
with an audio player and clickable timestamps. Windows only, developed and tested on one PC.

| Step | What | State |
|---|---|---|
| 1 | Reliable recording: WASAPI loopback, 60 s WAV segments, crash recovery | done |
| 2 | Local transcription with faster-whisper: SRT/Markdown export, resume | done |
| 3 | Study notes with Ollama (cloud models optional), Markdown/HTML export | done |
| 4+ | Live web UI, slide screenshots, automatic session detection | later |

## Requirements

- Windows 10/11
- Python 3.11+ (developed on 3.12)
- An NVIDIA GPU with a recent driver, for the transcription and the notes (developed on an RTX
  3060 with 12 GB; about 8 GB in use with the notes model loaded, desktop included)
- [Ollama](https://ollama.com/), running, with the model `qwen3.5:9b` (6.6 GB), for the notes

## Development setup

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev,cuda]"
.venv\Scripts\python -m pytest
```

The `cuda` extra installs cuBLAS 12 (`nvidia-cublas-cu12`, about 550 MB), the only CUDA
library the transcription needs: no system-wide CUDA toolkit or cuDNN is required, only a
recent NVIDIA driver. Leave it out (`".[dev]"`) on a machine without an NVIDIA GPU. The
speech model (`large-v3-turbo`, about 1.6 GB) is downloaded on first use into the Hugging Face
cache (`%USERPROFILE%\.cache\huggingface\hub`).

For the notes, install Ollama and download the model once:

```powershell
ollama pull qwen3.5:9b
.venv\Scripts\studyscribe config show   # is Ollama reachable, is the model there
```

Data is stored in `%LOCALAPPDATA%\StudyScribe` by default; override it with
`--data-dir` or the `STUDYSCRIBE_DATA_DIR` environment variable.

## Usage

From a lecture to its notes:

```powershell
.venv\Scripts\studyscribe record --title "Lecture 3" --course "Calculus"   # Ctrl+C to stop
.venv\Scripts\studyscribe sessions                        # find the SESSION_ID
.venv\Scripts\studyscribe transcribe SESSION_ID --notes   # transcription, then the notes
.venv\Scripts\studyscribe export SESSION_ID --audio       # audio.flac, for the player in notes.html
```

Then open `<data>\sessions\SESSION_ID\exports\notes.html` in Chrome or Edge and click a timestamp:
the audio starts there. The other commands:

```powershell
.venv\Scripts\studyscribe transcribe SESSION_ID   # transcription only
.venv\Scripts\studyscribe notes SESSION_ID        # notes of a session already transcribed
.venv\Scripts\studyscribe export SESSION_ID --notes-version 1   # an earlier version of the notes
.venv\Scripts\studyscribe recover                 # repair sessions left behind by a crash
```

`record` captures whatever the default output plays, follows headphones being
unplugged or replugged, and stops by itself after 30 minutes without sound
(`--silence-limit MINUTES`, 0 = never). Audio is written as 60-second WAV
segments, so a crash loses at most a fraction of a second; abandoned sessions
are repaired automatically the next time `record` or `sessions` runs.

`transcribe` detects the language on the first minutes of speech (`--language en` forces it)
and stores the transcript in the database. If it is interrupted (Ctrl+C, a crash, the GPU
running out of memory), running it again resumes where it stopped; transcribing a session
again replaces its transcript.

`notes` needs a transcript with at least 5 minutes of speech and writes `notes.md` and
`notes.html`: summary, outline, key concepts, definitions, examples and code, review questions,
unclear points. Every item except the summary cites transcript lines, shown as timestamps that
open the audio there. If it is interrupted, running it again reuses the finished blocks; after a
finished run it writes a new version. With the default provider nothing leaves the PC (unless
`OLLAMA_HOST` points to another one). On the RTX 3060 the notes of a 3-hour lecture took 15
minutes (transcription about 5 more), those of a 20-minute one 2 minutes. `--provider gemini` and
`--provider anthropic` send the transcript to a cloud model instead (API key with
`config set-key`): experimental, not tested end to end.

`export` writes the files in `exports` again: the transcript (`transcript.srt`, `transcript.md`),
the latest notes and, with `--audio`, `audio.flac`, which `notes.html` plays: without it the
timestamps are plain text.

## Known limits

- The notes are written by a small local model: they can be wrong or miss something. Check what
  matters by clicking its timestamp.
- Recognition errors are not corrected. The words the transcription was unsure about are listed at
  the end of the notes, to check by listening.
- Code read aloud can be rebuilt instead of quoted, and its extraction is unstable in Italian.
  Names not found in the cited lines get a warning.
- The notes of a long lecture can have many more items than the template asks for (156 instead of
  91 on the 3-hour one).
- Coverage warnings only see what is missing at the end of a 10-15 minute block.
- An interrupted run repeats the partial merges of a long lecture.
- Tested on one English and one Italian lecture, with `qwen3.5:9b` and thinking off.

## A note on use

Record only content you are allowed to record, and keep recordings of paid
courses for personal study.

## License

MIT, see [LICENSE](LICENSE).
