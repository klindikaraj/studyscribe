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
| 1 | Reliable recording: WASAPI loopback, 60 s WAV segments, crash recovery | in progress |
| 2 | Local transcription with faster-whisper | planned |
| 3 | Study notes with an LLM (configurable model), Markdown/HTML export | planned |
| 4+ | Live web UI, slide screenshots, automatic session detection | later |

## Requirements

- Windows 10/11
- Python 3.11+ (developed on 3.12)
- An NVIDIA GPU for local transcription (from step 2)

## Development setup

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
.venv\Scripts\python -m pytest
```

Data is stored in `%LOCALAPPDATA%\StudyScribe` by default; override it with
`--data-dir` or the `STUDYSCRIBE_DATA_DIR` environment variable.

## A note on use

Record only content you are allowed to record, and keep recordings of paid
courses for personal study.

## License

MIT, see [LICENSE](LICENSE).
