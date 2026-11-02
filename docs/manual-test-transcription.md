# Manual tests: transcription (step 2)

Plan: `docs/plan-transcription.md`. Run everything on a copy of the data directory: recovery,
`--check` and `transcribe` write into the session folder and the database.

```powershell
$d = "$env:LOCALAPPDATA\StudyScribe-tests\3h-copy"   # a copy, never the original
$s = "<session id>"
```

## A. Quick checks

- `pytest`, and `pytest -m gpu` (loads the real model).
- `scripts\transcribe_probe.py --data-dir $d --session $s --segment 10`: model hash MATCH,
  load time, speed, VRAM, language, text of one 60 s segment.
- `scripts\transcribe_probe.py --data-dir $d --session $s --from H:MM:SS --to H:MM:SS`: the
  whole pipeline on a stretch (`--no-normalize`, `--language`, `--hints` to compare).

## B. Interruptions

1. `studyscribe --data-dir $d transcribe $s --restart`, Ctrl+C halfway: "Interrupted at
   H:MM:SS. Run the same command to resume.", exit code 1.
2. `sessions`: TRANSCRIPT `interrupted NN%`; the previous transcript, if any, is still the one
   exported.
3. `transcribe $s` again: "Resuming an interrupted transcription at H:MM:SS", then done.
4. The same with the process ended from Task Manager (End process tree): `sessions` first
   prints "Transcription of ... interrupted at NN%", then the resume works.
5. `transcribe $s --language xx`: "Unknown language code", exit code 2.

## C. Acceptance test (3 h recording)

1. `Measure-Command { studyscribe --data-dir $d transcribe $s --restart }`: report the total
   time (model load and exports included).
2. `studyscribe --data-dir $d export $s --audio`, then open `exports\audio.flac` in VLC with
   `exports\transcript.srt` as subtitles.
3. Listen to the 5 points (below) against the text.
4. The pause: no kept or dropped segment between its start and end.

Passes when the time is reported, the 5 points match what is said (small word errors are fine;
no invented or missing sentences), and nothing is transcribed inside the pause.

### Result (2026-11-02), session `20261030-134407-4ba5ed` (2:57:34)

Lecture: freeCodeCamp.org, "Learn JavaScript - Full Course for Beginners". The recording is
about 2 s ahead of the video before the pause (0:29:33 to 0:34:44) and about 5 min 14 s after
it (the video's chapter times are approximate: "Comment Your Code" is listed at 4:23, the topic
starts at 4:31 in the video).

- Time: 4 min 51 s from launching the command to the last export (model load 3.7 s), 37x real
  time; 1415 segments, 0 dropped. An earlier run measured by the user: 40x.
- Points to listen to (time in `audio.flac` = time on the timeline; file and second for the
  WAV segments):

| # | Chapter (video time) | Timeline | WAV | Transcript |
|---|---|---|---|---|
| 1 | Comment Your Code (4:23) | 0:04:33 | `000005.wav` at 33 s | "The first thing we'll talk about in JavaScript is comments. Comments are lines of code that JavaScript will intentionally ignore. They don't do anything. They're just used to create notes for yourself and others about what the code does. So if you do a slash slash, you can make an inline comment." |
| 2 | Escape Sequences (25:18) | 0:25:30 | `000026.wav` at 30 s | "...double quote character by using the backslash before the double quote. There's actually quite a few other things you can escape out. You can escape out a single quote character. You can escape out a backslash. In fact, anytime you're going to use a backslash, you're going to have to put two backslashes." |
| 3 | Global Scope (55:43) | 1:00:58 | `000061.wav` at 58 s (into `000062.wav`) | "...output at 10. For this one, scope refers to the visibility of variables, variables which are defined outside of a function block have global scope, global scope means they can be seen everywhere in your JavaScript code." |
| 4 | Switch Statements (1:32:15) | 1:37:31 | `000098.wav` at 31 s | "...completed this challenge. Instead of using chained else if statements, you can use a switch statement, a switch statement, test a value and can have many case statements, which define various possible values." |
| 5 | Arrow Functions (2:47:17) | 2:52:31 | `000173.wav` at 31 s | "...change, use object dot freeze. This function here is called an anonymous function. It doesn't have a name. It is assigned to this variable magic." |
| 6 | (highest compression ratio, 2.25) | 2:27:03 | `000148.wav` at 3 s | "...then we're linking looking at the length of this array, then we're looking at the length of this array, then we're looking at the length of this array," (spoken, or the start of a loop?) |

- The pause: the last segment before it ends at about 0:29:32 ("So I'm going to do my name
  equals"), the next starts at 0:34:46 ("And then my string is going to equal..."); nothing
  kept or dropped in between.
- Listening by the user (2026-11-02, VLC with `audio.flac` and `transcript.srt`): points 1-5
  correct; point 6 is spoken for real (the lecturer looks at three arrays), so it is right to
  keep it. Accepted.

## D. Italian

A lecture in Italian recorded with `studyscribe record` (here: `it-copy`, session
`20261102-173400-e04c56`, 0:19:59, a PyTorch tutorial in Italian). Transcribe with the
language detected, then with `--language it`; listen to a few points.

Result (2026-11-02): detected `it` 1.00 on 180 s of speech; `--language it` gives the same
`.srt`; 241 segments, 0 dropped; 49 s in all. Points to listen to:

| # | Timeline | WAV | Transcript | Why |
|---|---|---|---|---|
| 1 | 0:01:55 | `000002.wav` at 55 s | "forward pass, consiste nel fornire un input al modello per ottenere un output. ... nel secondo step, detto calcolo dell'errore o loss, viene confrontato l'output previsto con..." | English terms in Italian |
| 2 | 0:05:07 | `000006.wav` at 7 s | "Dal popolarissimo package per creare grafici matplotlib importiamo il modulo pyplot che offre la funzione scatter utile a creare un grafico a dispersione." | library names |
| 3 | 0:06:30 | `000007.wav` at 30 s | "Da PyTorch.neuralNetwork peschiamo la classe Linear." | suspect: probably `torch.nn` |
| 4 | 0:11:05 | `000012.wav` at 5 s | "Uno dei più usati ed apprezzati si chiama Adam ... eta è la velocità di apprendimento o learning rate." | terms |
| 5 | 0:14:01 | `000015.wav` at 1 s | "Occhio che su YPRED sono presenti sì gli output, ma anche dei dati relativi ai gradienti ... stacchiamoli col metodo di TEC per plottare solo gli output." | suspect: `y_pred`, `detach` |

Listening by the user (2026-11-02, VLC): points 1, 2 and 4 correct. Point 3 is faithful: the
lecturer does say "Da PyTorch.neuralNetwork" (the code on screen is `model =
torch.nn.Linear()`). Point 5: he says "ipsilon pred" (`y_pred` in the code), so "YPRED" is
faithful too; then he says "detach", and "TEC" is the only real error. Accepted.

For step 3 (notes): correct errors like "TEC" -> `detach` from the context, and write code
identifiers read aloud in their code form (`PyTorch.neuralNetwork` -> `torch.nn`, "ipsilon
pred" -> `y_pred`). Both need the context of the lecture, so they belong to the LLM, not to
the ASR (hints did not help, see the plan).
