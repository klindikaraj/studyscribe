"""Check the timestamps of a session's notes exports against its transcript and audio.

Usage:
    .venv\\Scripts\\python scripts\\check_notes_export.py EXPORTS_DIR [FILE.html ...]

EXPORTS_DIR is a session's `exports` folder (the one holding transcript.srt). With no file given
every `notes*.html` in it is checked. It reads files only (no database, no network) and checks,
for every timestamp link of the page that carries its exact time (`data-ms`, the links of the
notes' items):

1. the time is the start of a cue of transcript.srt, to the millisecond: the notes and the
   transcript export are on the same timeline;
2. the link seeks `data-ms // 1000` seconds and its label is that time as H:MM:SS;
3. the time is inside the audio (when audio.flac is there: its length is read from the FLAC
   header; no decoding).

Every other timestamp link (the coverage warning and the appendix, which point to a block or a
word rather than to a line) is only checked for seeking inside the audio. Exit code 0 when
everything matches, 1 when something does not, 2 on a usage error.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path

SRT_START = re.compile(r"^(\d+):(\d\d):(\d\d),(\d\d\d) --> ", re.MULTILINE)


@dataclass(frozen=True)
class Link:
    label: str
    seconds: int
    ms: int | None  # exact time, for the links of the notes' items
    line: int | None  # S-number of the line


class _Links(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[Link] = []
        self._open: dict[str, str | None] | None = None
        self._label = ""

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if tag == "a" and "ts" in (values.get("class") or "").split():
            self._open, self._label = values, ""

    def handle_data(self, data):
        if self._open is not None:
            self._label += data

    def handle_endtag(self, tag):
        if tag == "a" and self._open is not None:
            attrs = self._open
            ms, n = attrs.get("data-ms"), attrs.get("data-n")
            self.links.append(Link(self._label.strip(), int(attrs.get("data-t") or -1),
                                   int(ms) if ms else None, int(n) if n else None))
            self._open = None


def html_links(text: str) -> list[Link]:
    parser = _Links()
    parser.feed(text)
    return parser.links


def srt_cue_starts(text: str) -> set[int]:
    """The start time of every cue, in milliseconds."""
    return {((int(h) * 60 + int(m)) * 60 + int(s)) * 1000 + int(ms)
            for h, m, s, ms in SRT_START.findall(text)}


def flac_duration_ms(path: Path) -> int | None:
    """Length of a FLAC file from its STREAMINFO block; None when it is not a FLAC file."""
    with path.open("rb") as f:
        if f.read(4) != b"fLaC":
            return None
        header = f.read(4)
        if len(header) < 4 or header[0] & 0x7F != 0:  # STREAMINFO is always the first block
            return None
        body = f.read(int.from_bytes(header[1:4], "big"))
    sample_rate = int.from_bytes(body[10:13], "big") >> 4
    total = int.from_bytes(body[13:18], "big") & 0xFFFFFFFFF
    return total * 1000 // sample_rate if sample_rate and total else None


def clock(seconds: int) -> str:
    return f"{seconds // 3600}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"


def check_page(links: list[Link], cue_starts: set[int], duration_ms: int | None
               ) -> tuple[list[str], int]:
    """(problems, number of links with an exact time that were checked against the SRT)."""
    problems = []
    exact = 0
    for link in links:
        where = f"{link.label} (S{link.line})" if link.line else link.label
        if link.ms is not None:
            exact += 1
            if link.ms not in cue_starts:
                problems.append(f"{where}: {link.ms} ms is not the start of any cue of "
                                "transcript.srt")
            if link.seconds != link.ms // 1000:
                problems.append(f"{where}: seeks {link.seconds} s but its time is "
                                f"{link.ms} ms")
        if link.label != clock(link.seconds):
            problems.append(f"{where}: the label does not show the {link.seconds} s it seeks")
        if duration_ms is not None and link.seconds * 1000 >= duration_ms:
            problems.append(f"{where}: {link.seconds} s is past the end of the audio "
                            f"({duration_ms / 1000:.1f} s)")
    return problems, exact


def main(argv: list[str]) -> int:
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 2
    folder = Path(argv[0])
    srt = folder / "transcript.srt"
    if not srt.is_file():
        print(f"No transcript.srt in {folder}: give a session's exports folder.")
        return 2
    pages = [folder / name for name in argv[1:]] or sorted(folder.glob("notes*.html"))
    if not pages:
        print(f"No notes*.html in {folder}: run `studyscribe export SESSION_ID` first.")
        return 2
    cues = srt_cue_starts(srt.read_text(encoding="utf-8"))
    audio = folder / "audio.flac"
    duration = flac_duration_ms(audio) if audio.is_file() else None
    print(f"{srt.name}: {len(cues)} cue starts; "
          + (f"audio.flac: {duration / 1000:.1f} s" if duration else "no audio.flac (length "
                                                                      "not checked)"))
    failed = False
    for page in pages:
        links = html_links(page.read_text(encoding="utf-8"))
        problems, exact = check_page(links, cues, duration)
        status = "OK" if not problems else f"{len(problems)} PROBLEMS"
        print(f"{page.name}: {len(links)} timestamp links, {exact} with an exact time "
              f"checked against the SRT: {status}")
        for problem in problems[:20]:
            print(f"  {problem}")
        if len(problems) > 20:
            print(f"  ... and {len(problems) - 20} more")
        failed = failed or bool(problems)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
