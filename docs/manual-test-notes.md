# Manual test: study notes

Checklist for the parts of step 3 that need a human: a browser, headphones, and listening.
Section A is the export of piece 4 (`notes.md`, `notes.html`). Piece 5 (acceptance on the 3 h
and the Italian lecture) adds its own sections here.

Use only the copies in `%LOCALAPPDATA%\StudyScribe-tests` (`it-copy`, `3h-copy`), never the
originals (`it`, `3h`). In the commands below, run PowerShell from the repository folder.

## A. Export: notes.html in the browser

Result: A1-A5 passed on `it-copy` by the user on 2026-11-25 (exact seek, `#t=`, questions,
appendix, banner without audio). A6: all 3 points passed on the same day (see A6). A7 optional.

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

Result (user, 2026-11-25): passed, all 7 seeks (points 1 and 2) land on the right sentence. At
0:08:14 the voice says "let": "left" is a recognition error of the transcript, not of the seek.
`3h-copy-p` was deleted afterwards. Point 3 (user, 2026-11-25): passed on the page generated again
after the fixes of piece 5b, the three links of the coverage warning start on the right sentence.

Prepared on 2026-11-25 in **`3h-copy-p`** (a separate copy of `3h-copy`, with `audio.flac`): with
the default settings the 3 h generation stops in the final reduce (piece 5), so these notes come
from a run with `reduce_max_input` 10,000. The export and `check_notes_export.py` are OK (147 exact
links). Open the page:
`3h-copy-p/sessions/20261030-134407-4ba5ed/exports/notes.html`.

1. Click each of these timestamps in the notes and **listen**: what you hear must be the sentence,
   at most a second early (recognition errors kept as in `transcript.md`):

   | Click | Where | You hear |
   |---|---|---|
   | 0:08:14 | Examples and code, `let ourName = 'free code camp'` | "there's a way called left [let]. So I can say let our name..." |
   | 0:28:05 | Key concepts, String Concatenation (before the 5 min pause) | "example, we have the r string equals I come first..." |
   | 0:36:15 | Examples and code, `myString += adjective` (after the pause at 0:29:33) | "equals, and then I can put some adjective..." |
   | 1:26:29 | Examples and code, `if val < 10 \|\| val > 20` (22 s after the device switch at 1:26:07) | "which is just two pipes. So I'm gonna put Val..." |
   | 1:57:19 | Examples and code, `const entreeValue = ...` | "use bracket notation. So the entree value, we're going to test object..." |
   | 2:55:34 | Examples and code, `squaredIntegers` (last cited line) | "I want you to look at is the arrow functions..." |

2. The notes have no link after 2:55:34: add `#t=10623` to the address and reload. The player
   stands at 2:57:03, **paused**. Press play: "that are more than zero, and also integers. So
   there'll be 442 and six". The transcript ends at 2:57:29 ("...So we're going to").
3. The links of the yellow warning, on the page generated again by piece 5b (version 2 in
   **`3h-copy`**: `3h-copy/sessions/20261030-134407-4ba5ed/exports/notes.html`). Click each one and
   **listen**: you hear the first uncovered line, at most a second early.

   | Link | First line | You hear |
   |---|---|---|
   | 0:42:35 to 0:45:01 (lines S308 to S327) | S308 | "types of words. You pass in a noun, an adjective, a verb and an adverb..." |
   | 1:45:27 to 1:47:38 (lines S811 to S825) | S811 | "we're going to set the answer to the answer." then "For case one..." |
   | 2:10:56 to 2:15:32 (lines S1016 to S1051) | S1016 | "to delete collection. And then we have to use bracket notation ID and then prop..." |

   The header says version 2 of 2 and there are no other links in the box.

### A7. Safety (optional)

"View source" on `notes.html`: one `<script>` and one `<style>`, a Content-Security-Policy
meta tag, no `http` addresses, no `on...=` attributes. The automated tests build a page from
hostile strings (`<script>`, `onerror=`, `javascript:`) in every field and audit it.
