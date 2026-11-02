# StudyScribe

A personal, local-first note-taking app for online courses. It records what your
PC is playing (the lecture video), transcribes it locally on the GPU and turns
the transcript into study notes with clickable timestamps.

Inspired by [Granola](https://www.granola.ai/), but written from scratch: it
contains no code from Granola and is not affiliated with it.

## Status

Early work in progress. Windows only.

| Step | What | State |
|---|---|---|
| 1 | Reliable recording: WASAPI loopback, 60 s WAV segments, crash recovery | done |
| 2 | Local transcription with faster-whisper | in progress |
| 3 | Study notes with an LLM (configurable model), Markdown/HTML export | planned |
| 4+ | Live web UI, slide screenshots, automatic session detection | later |

## Requirements

- Windows 10/11
- Python 3.11+ (developed on 3.12)
- An NVIDIA GPU for local transcription (from step 2)

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

Data is stored in `%LOCALAPPDATA%\StudyScribe` by default; override it with
`--data-dir` or the `STUDYSCRIBE_DATA_DIR` environment variable.

## Usage

```powershell
.venv\Scripts\studyscribe record --title "Lecture 3" --course "Calculus"   # Ctrl+C to stop
.venv\Scripts\studyscribe transcribe SESSION_ID   # local transcription on the GPU
.venv\Scripts\studyscribe sessions    # list sessions with status, duration, transcript
.venv\Scripts\studyscribe recover     # repair sessions left behind by a crash
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

## A note on use

Record only content you are allowed to record, and keep recordings of paid
courses for personal study.

## License

MIT, see [LICENSE](LICENSE).
