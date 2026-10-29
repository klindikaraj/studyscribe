# Manual test: recording (step 1)

What the unit tests cannot cover: the real loopback, real device changes, a process killed by
Windows, the anti-standby request. Run these on the target PC. The guided script keeps its
sessions in `%TEMP%\studyscribe-manual-test` unless you pass `--data-dir`.

## Before you start

- `.venv\Scripts\python -m pytest` passes.
- Headphones plugged into the jack whose endpoint disappears when they are unplugged (on the
  test PC: Realtek), with a second output available (the monitor).
- A lecture video ready and **paused** at a point where someone is talking.
- Task Manager open on the Details tab, and an **administrator** PowerShell for
  `powercfg /requests`.

## A. Guided test (about 4 minutes)

```powershell
.venv\Scripts\python scripts\manual_recording_test.py
```

The script prompts each step at the suggested time; do it and press Enter at the same moment.

| When | You do | Expected on screen |
|---|---|---|
| start | nothing (video paused) | `Recording from Speakers (Realtek(R) Audio) [Loopback] …` |
| 0:05 | press play | — |
| 1:00 | pause | — |
| 1:20 | resume | — |
| 1:40 | unplug the headphones | `Device lost (stream stopped): Speakers …`, then `Switched to <monitor> …, gap N ms` |
| 2:05 | plug them back in | `Device lost (default output changed): <monitor> …`, then `Switched to Speakers …` |
| ~2:10 | `powercfg /requests` in the admin shell; answer whether python.exe is listed under SYSTEM | — |
| 3:10 | end the **capture** process (PID shown) in Task Manager, not the script's PID | `Capture process exit code: 1. Recovering the session.`, `Recovered … recording -> interrupted` |

Then the script prints the checks and saves them as `manual-test-report.txt` in the session
folder:

| Check | Passes when |
|---|---|
| Session status | `interrupted` (recovered after the kill) |
| Segments and files agree | numbering 1..N, segment N at (N-1)·60 s, all but the last exactly 60 s, header sizes match the files, no file missing from the manifest |
| Duration matches real time | within ±1 s of the time from the start to the kill |
| ended_at matches the kill | within ±1 s (last write time after a kill, an assumption of the design) |
| First sound = play pressed | within ±1 s: the initial silence is on the timeline, segment 1 starts at the start |
| Last sound before the pause / first sound after it | within ±1 s each: the timeline stays aligned across a pause |
| Unplug followed / Replug followed | a switch reported within 5 s, to the other output / back to the first one |
| Device gaps in the manifest | manifest gap = sum of the reported gaps, ±100 ms per switch |
| Anti-standby | you answered `y` |
| INFO lines | short digital silences inside sound, warnings: for the record |

The marks are pressed by hand: if a timing check misses by a little more than 1 s, repeat the
test before suspecting the code. If you kill the wrong process (the script), check the session
afterwards with `--check` (section B, step 4): the marks are saved as you go.

## B. Recovery by another process (both processes killed)

1. `studyscribe --data-dir <dir> record --title "Kill both"` with a video playing.
2. After a minute or more: Task Manager > the parent python.exe (the "Parent pid") > End process
   tree, or `taskkill /F /T /PID <parent pid>`.
3. `studyscribe --data-dir <dir> sessions`: `Recovered <id>: recording -> interrupted` is printed
   before the table.
4. `.venv\Scripts\python scripts\manual_recording_test.py --data-dir <dir> --check <id>`: status,
   segments and duration (against `ended_at`) pass.

## C. Acceptance test: 3 hours (roadmap step 1)

1. `studyscribe record --title "3 h test"` with a lecture or playlist of at least 3 hours.
2. Meanwhile: pause once for about 5 minutes; unplug and replug the headphones at least once.
3. End with Ctrl+C (or kill the capture process in the last minutes: the CLI recovers it).
4. `manual_recording_test.py --check <id>`.

Done when all checks pass and a few seconds listened around each device switch, around the
pause and at the end sound right. After a clean stop, the sign in "Duration matches real time"
measures the device clock drift over the whole recording: + means the device clock ran fast
(the surplus is absorbed only at pauses); expect well under a second. For a continuous drift
reading, run `scripts\capture_probe.py --session "Drift" --seconds 10800` instead and watch
`missing` over time.

## D. One-off checks

- Silence limit: `studyscribe record --title "Limit" --silence-limit 1` with nothing playing:
  after a minute `Warning: no sound since the start (60 s limit): recording stopped
  automatically`, status `recorded`, exit code 0.
- Second Ctrl+C while stopping: `Still stopping: closing the last segment, please wait.`, the
  session still ends `recorded`.
- Optional, no output device at the start (all outputs disabled in the sound settings):
  `Error: …`, session `failed`, exit code 1.
