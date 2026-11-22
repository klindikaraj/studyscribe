# Manual test: study notes

Checklist for the parts of step 3 that need a human: a browser, headphones, and listening.
Section A is the export of piece 4 (`notes.md`, `notes.html`). Piece 5 (acceptance on the 3 h
and the Italian lecture) adds its own sections here.

Use only the copies in `%LOCALAPPDATA%\StudyScribe-tests` (`it-copy`, `3h-copy`), never the
originals (`it`, `3h`). In the commands below, run PowerShell from the repository folder.

## A. Export: notes.html in the browser

Prepared by Claude on 2026-11-22: `it-copy` has two finished generations of notes (qwen3.5:9b,
prompt v5.1) and `audio.flac`. Claude could check the page only in headless Edge over the
DevTools protocol (no console or CSP errors, clicks seek, `#t=` seeks); what only a person can
check is marked **listen**.

### A1. Write the files

```powershell
$data = "$env:LOCALAPPDATA\StudyScribe-tests\it-copy"
$sid = "20261102-173400-e04c56"
.venv\Scripts\studyscribe --data-dir $data export $sid
```

Expected: a line `Notes version 2 of 2 (qwen3.5:9b, ...) in ...\exports: notes.md, notes.html`
and no warning (audio.flac is there). Then the timestamps against the transcript:

```powershell
.venv\Scripts\python scripts\check_notes_export.py "$data\sessions\$sid\exports"
```

Expected: `OK` for every page, exit code 0 (`echo $LASTEXITCODE`).

### A2. The page

Open `$data\sessions\$sid\exports\notes.html` in Chrome or Edge (double click).

1. The audio player is at the top and stays there while you scroll.
2. Under the title: course/date/duration/language and "Note: versione 2 di 2".
3. A yellow box "Parti della lezione che le note potrebbero non coprire" with a link
   "da 0:17:12 a 0:18:07 (righe da S226 a S241)".
4. Sections in this order: Riepilogo, Scaletta, Concetti chiave, Definizioni, Domande di ripasso,
   then the appendix "Parole da verificare ascoltando". (Esempi e codice and Punti poco chiari
   are empty in these notes, so they are left out.)
5. **listen** Press play, then click the first Scaletta timestamp (0:01:06): playback jumps there
   and starts by itself; what you hear is the introduction ("modello AI ... scatola nera").
6. **listen** Click a few other timestamps (0:04:28 on "tensor", 0:12:41 on the conclusion of the
   training): each one lands at the start of the sentence the item talks about, at most a
   second before it.
7. The address bar now ends with `#t=` and the second you clicked. Copy that address into a new
   tab: the player is at that second, **paused**.
8. Review questions: the answer is hidden; "Mostra la risposta" opens it, and clicking again
   closes it.
9. Appendix: three words (Adam, significa, TEC) with line, time and probability, the word
   highlighted in its line. **listen** Click the time of TEC (0:14:11): you hear "...stacchiamoli
   col metodo di TEC"; the notes keep the word as heard, they do not say `detach`.
10. Footer: model, effort, prompt version, tokens and time of the generation.

### A3. Without audio.flac

1. In `exports`, rename `audio.flac` to `audio.flac.off`, reload the page: the player is
   empty and a yellow banner says the audio could not be loaded ("controlla che audio.flac sia
   nella stessa cartella"). Timestamps are still readable.
2. Rename it back and reload: the banner is gone and the player works again.
3. With `audio.flac` still renamed, run the export again (A1): the CLI prints
   `warning: audio.flac is not in ...`, with the command to make it. The new `notes.html` shows
   a banner from the start ("audio.flac non è stato trovato accanto a questa pagina"), and
   `audio.flac` is **not** created by itself.
4. Rename it back and export again: no warning.

### A4. Versions

```powershell
.venv\Scripts\studyscribe --data-dir $data export $sid --notes-version 1
.venv\Scripts\studyscribe --data-dir $data export $sid --notes-version 9
```

The first writes `notes-v1.md` and `notes-v1.html` ("version 1 of 2" in the page header). The
second fails with exit code 1 and lists the versions that exist (number, date, model, effort).
`notes.html` is still version 2.

### A5. Markdown

Open `notes.md` in the VS Code preview (or on GitHub): the same sections; the timestamps are
links to `notes.html#t=...` (they open the page at that second when the file is opened in a
browser); the answers of the review questions are collapsed under "Mostra la risposta"; the
appendix is a table. If your viewer shows the raw `<details>` instead, note which viewer.

### A6. The 3 h lecture (after piece 5)

`3h-copy` has no notes yet. When it has (piece 5), repeat A1 on session
`20261030-134407-4ba5ed`, then **listen** at a timestamp near the end (e.g. 2:50:00): the sound
must be the sentence of `transcript.md` at that time. Claude checked in headless Edge that
seeking in the 161 MB FLAC (it has no seek table) takes 11-70 ms and lands on the second, but
could not hear the result.

### A7. Safety (optional)

"View source" on `notes.html`: one `<script>` and one `<style>`, a Content-Security-Policy
meta tag, no `http` addresses, no `on...=` attributes. The automated tests build a page from
hostile strings (`<script>`, `onerror=`, `javascript:`) in every field and audit it.
