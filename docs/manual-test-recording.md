# Manual test: recording (step 1)

What the unit tests cannot cover: the real loopback, real device changes, a process killed by
Windows, the anti-standby request. Run these on the target PC. The guided script keeps its
sessions in `%TEMP%\studyscribe-manual-test` unless you pass `--data-dir`.

## Before you start

- `.venv\Scripts\python -m pytest` passes.
- Headphones plugged into the jack whose endpoint disappears when they are unplugged (on the
  test PC: Realtek), with a second output available (the monitor).
- The lecture video open in the browser: **play it, then pause it** at a point where someone is
  talking. Playing it once gives the browser the media session that receives the media
  Play/Pause key; the browser does not need to be in the foreground.
- Close other media apps (music players, other tabs playing audio): Windows sends the media key
  to the most recent media session, which must be the video.
- Task Manager open on the Details tab, and an **administrator** PowerShell for
  `powercfg /requests`. Keep administrator windows out of the foreground at 0:05, 1:00 and
  1:20: Windows silently drops keys that a normal program sends to an elevated window.

## A. Guided test (about 4 minutes)

```powershell
.venv\Scripts\python scripts\manual_recording_test.py
```

The script plays, pauses and resumes the video itself by sending the media Play/Pause key, so
those moments are known exactly. For the other steps it shows a prompt: **first do the action,
then press Enter**; the checks look for its effect anywhere between the prompt and a few seconds
after your Enter.

| When | What happens | Expected on screen |
|---|---|---|
| start | nothing (video paused) | `Recording from Speakers (Realtek(R) Audio) [Loopback] …` |
| 0:05 | the script sends Play/Pause: the video plays | `PLAY the video …: media Play/Pause key sent.` |
| 1:00 | the script sends Play/Pause: the video pauses | `PAUSE the video: media Play/Pause key sent.` |
| 1:20 | the script sends Play/Pause: the video plays again | `RESUME the video: media Play/Pause key sent.` |
| 1:40 | you unplug the headphones, then press Enter | `Device lost (stream stopped): Speakers …` or `(default output changed)`, whichever Windows signals first; then `Switched to <monitor> …, gap N ms` |
| 2:05 | you plug them back in, then press Enter | `Device lost (default output changed): <monitor> …`, then `Switched to Speakers …` |
| after | you run `powercfg /requests` in the admin shell and answer whether python.exe is listed under SYSTEM | — |
| 3:10 | you end the **capture** process (PID shown) in Task Manager, not the script's PID | `Capture process exit code: 1. Recovering the session.`, `Recovered … recording -> interrupted` |

If a media key has no audible effect, the script says so at once (`No sound after the key …`):
stop with Ctrl+C, fix the browser (see "Before you start") and start again. Without media keys,
`--manual` makes play, pause and resume prompts too: do the action, then press Enter.

Then the script prints the checks and saves them as `manual-test-report.txt` in the session
folder:

| Check | Passes when |
|---|---|
| Session status | `interrupted` (recovered after the kill) |
| Segments and files agree | numbering 1..N, segment N at (N-1)·60 s, all but the last exactly 60 s, header sizes match the files, no file missing from the manifest |
| Duration matches real time | within ±1 s of the time from the start to the kill |
| ended_at matches the kill | within ±1 s (last write time after a kill, an assumption of the design) |
| Sound starts at play | media key: from 0.25 s before to 2 s after the key (the first play can be slow: the browser may have to reopen its audio output; seen +1.08 s); `--manual`: between the prompt and your Enter, ±0.5 s. Proves the initial silence is on the timeline: if it were lost, the sound would come before the key |
| Sound stops at pause | media key: from 1 s before (paused between two words) to 0.5 s after the key; `--manual`: as above. The audio from play to pause is continuous, so this also shows that the first run of audio sits at the right place |
| Sound starts again at resume | media key: from 0.25 s before to 1 s after the key (seen +0.41 s); `--manual`: as above |
| Unplug followed / Replug followed | a switch reported between the prompt and 5 s after Enter, to the other output / back to the first one (gaps seen: 53-624 ms; reopening the Realtek endpoint just after replugging is the slow part) |
| Device gaps in the manifest | manifest gap = sum of the reported gaps, ±100 ms per switch |
| Anti-standby | you answered `y` |
| INFO lines | short digital silences inside sound, warnings: for the record |

A timeline shifted by a second or more fails the timing checks. If you kill the wrong process
(the script), check the session afterwards with `--check` (section B, step 4): the steps are
saved as they happen.

## B. Recovery by another process (both processes killed)

1. `studyscribe --data-dir <dir> record --title "Kill both"` with a video playing.
2. After a minute or more: Task Manager > the parent python.exe (the "Parent pid") > End process
   tree, or `taskkill /F /T /PID <parent pid>`.
3. `studyscribe --data-dir <dir> sessions`: `Recovered <id>: recording -> interrupted` is printed
   before the table.
4. `.venv\Scripts\python scripts\manual_recording_test.py --data-dir <dir> --check <id>`: status,
   segments and duration (against `ended_at`) pass.

## C. Acceptance test: 3 hours (roadmap step 1)

Roadmap: a 3 h recording with the headphones unplugged and replugged and a process kill loses
at most a few seconds, and the timeline stays aligned.

Prepare:

- At least 3 h 10 min of lecture in one browser tab (a long video, or a playlist with autoplay
  on). If playback stops for more than 30 minutes the recording ends by itself.
- About 120 MB of disk per hour (350 MB in all).
- Nothing that sleeps or restarts the PC: the recording keeps Windows awake, but a manual sleep
  or an update restart would stop it.
- A data directory of its own: `$d = "$env:TEMP\studyscribe-3h"`.

Run:

1. `.venv\Scripts\studyscribe --data-dir $d record --title "3 h test"`, then start the video.
   Note the capture PID it prints.
2. Around 0:30, pause the video for about 5 minutes, then resume it.
3. Around 1:30, unplug the headphones; a few minutes later plug them back in.
4. After 3:00:00, end the capture process in Task Manager (Details > the capture PID > End
   task). The CLI recovers the session: `interrupted`, exit code 1.
5. `.venv\Scripts\python scripts\manual_recording_test.py --data-dir $d --check <id>`.

Passes when:

- every check passes (without a guided run, the timing and switch checks are SKIP);
- "Duration matches real time" is within ±1 s. Its sign is the device clock drift since the
  last pause (+ means the device clock ran fast: the surplus is absorbed only at pauses);
- the CLI output shows both device switches with gaps of at most about a second;
- the minutes around each switch, the pause and the end sound right. To find them: the
  session ID holds the start time (`20261030-131710` started at 13:17:10), so an event
  printed at 14:47:40 is at 1:30:30 on the timeline, i.e. 30 s into segment 91 (`000091.wav`
  covers 1:30:00-1:31:00).

For a continuous drift reading, run `scripts\capture_probe.py --session "Drift" --seconds 10800`
instead and watch `missing` over time.

## D. One-off checks

- Silence limit: `studyscribe record --title "Limit" --silence-limit 1` with nothing playing:
  after a minute `Warning: no sound since the start (60 s limit): recording stopped
  automatically`, status `recorded`, exit code 0.
- Second Ctrl+C while stopping: `Still stopping: closing the last segment, please wait.`, the
  session still ends `recorded`.
- Optional, no output device at the start (all outputs disabled in the sound settings):
  `Error: …`, session `failed`, exit code 1.
