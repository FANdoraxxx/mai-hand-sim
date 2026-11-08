"""Simai (maidata.txt and raw inote string) chart parser aligned with MajdataPlay."""
from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from sim_core.config import FRAME_LENGTH_SEC
from sim_core.geometry import SlideAreaStep, SlideRouteSpec, build_slide_route


class NoteType(str, Enum):
    TAP = "TAP"
    HOLD = "HOLD"
    TOUCH = "TOUCH"
    TOUCH_HOLD = "TOUCH_HOLD"
    SLIDE = "SLIDE"


@dataclass
class SlideSegmentData:
    shape: str               # '-', '^', '<', '>', 'v', 'p', 'q', 'pp', 'qq', 's', 'z', 'w', 'V'
    start_btn: int           # 1..8
    end_btn: int             # 1..8
    via_btn: Optional[int]   # 1..8 for 'V', else None
    route: SlideRouteSpec


@dataclass
class ParsedNote:
    note_id: int
    note_type: NoteType
    time: float              # Head / strike timestamp (sec)
    bar: int                 # 0-based measure index
    beat_in_bar: float       # Beat offset within measure (in 4/4 whole-measure units)
    bpm: float               # Active BPM at note time
    key: str                 # '1'..'8' for outer/A ring notes, or 'A1'..'E8', 'C', 'C1', 'C2'
    button: Optional[int]    # 1..8 for button-indexed notes, None for inner-only Touch
    sensor_zone: str         # Primary sensor zone name ('A1'..'E8', 'C1', 'C2', 'C')
    is_break: bool = False
    is_ex: bool = False
    is_firework: bool = False
    is_each: bool = False
    is_pseudo_each: bool = False
    is_star_head: bool = False
    is_headless_slide: bool = False
    duration: float = 0.0                # Hold duration or Slide travel duration (sec)
    wait_time: float = 0.0               # Slide pre-launch wait time (sec, default 1 beat at BPM)
    shoot_time: float = 0.0              # Slide star launch time = time + wait_time
    end_time: float = 0.0                # Note completion time
    judge_time: float = 0.0              # Answer frame timing (MajdataPlay _judgeTiming)
    last_wait_time: float = 0.0          # Slide tail residence window (sec)
    slide_segments: Tuple[SlideSegmentData, ...] = ()
    slide_queue: Tuple[SlideAreaStep, ...] = ()
    wifi_queues: Optional[Tuple[Tuple[SlideAreaStep, ...], Tuple[SlideAreaStep, ...], Tuple[SlideAreaStep, ...]]] = None
    slide_waypoints_mm: Tuple[Tuple[float, float], ...] = ()
    raw_token: str = ""


@dataclass
class ChartEvent:
    """A chronological event group of simultaneous (or near-simultaneous) notes."""
    event_id: int
    time: float
    bar: int
    beat_in_bar: float
    bpm: float
    notes: List[ParsedNote] = field(default_factory=list)

    @property
    def strike_notes(self) -> List[ParsedNote]:
        """Notes requiring an instantaneous strike/press at `self.time` (excludes SLIDE body, which launches later)."""
        return [n for n in self.notes if n.note_type in (NoteType.TAP, NoteType.HOLD, NoteType.TOUCH, NoteType.TOUCH_HOLD)]

    @property
    def slide_notes(self) -> List[ParsedNote]:
        return [n for n in self.notes if n.note_type == NoteType.SLIDE]


@dataclass
class ParsedChart:
    title: str
    artist: str
    difficulty_label: str
    level_str: str
    first_offset_sec: float
    initial_bpm: float
    notes: List[ParsedNote]
    events: List[ChartEvent]
    total_duration_sec: float
    bar_timestamps: List[Tuple[int, float, float]]  # (bar_idx, start_sec, end_sec)
    source_path: Optional[str] = None


_DIFFICULTY_MAP: Dict[str, str] = {
    "1": "1", "easy": "1",
    "2": "2", "basic": "2", "bas": "2",
    "3": "3", "advanced": "3", "adv": "3",
    "4": "4", "expert": "4", "exp": "4",
    "5": "5", "master": "5", "mas": "5",
    "6": "6", "remaster": "6", "remas": "6", "re:master": "6",
    "7": "7", "utage": "7",
}

# Slide shape regex ordered longest-first ('pp', 'qq' before 'p', 'q')
_SLIDE_OP_PATTERN = re.compile(r"(pp|qq|[-^<>vpqszwV])")


def _strip_comments(text: str) -> str:
    out_lines: List[str] = []
    for line in text.splitlines():
        for marker in ("||", "//"):
            idx = line.find(marker)
            if idx != -1:
                line = line[:idx]
        out_lines.append(line)
    return "\n".join(out_lines)


def parse_maidata_fields(raw_text: str) -> Dict[str, str]:
    """Extract `&key=value` blocks from a maidata.txt file."""
    cleaned = _strip_comments(raw_text)
    fields: Dict[str, str] = {}
    # Split on '&key=' at start of text or after newline
    matches = list(re.finditer(r"(?:^|\n)&([A-Za-z0-9_]+)=", cleaned))
    for i, m in enumerate(matches):
        key = m.group(1).strip().lower()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(cleaned)
        fields[key] = cleaned[start:end].strip()
    return fields


def _parse_Bracket_Timing(
    bracket_body: str,
    current_bpm: float,
    is_slide: bool,
) -> Tuple[float, float]:
    """Parse Simai `[...]` timing specification into `(wait_sec, duration_sec)`.

    Supports:
      - `[div:count]` -> beat ratio at current_bpm
      - `[bpm#div:count]` -> wait at 60/bpm, duration from beat ratio at `bpm`
      - `[#sec]` or `[sec]` -> direct seconds (wait = 60/current_bpm for slide)
      - `[wait##dur]` -> explicit wait_sec and duration_sec
      - `[wait##div:count]` -> explicit wait_sec, duration from beat ratio at current_bpm
      - `[wait##bpm#div:count]` -> explicit wait_sec, duration from beat ratio at `bpm`
    """
    body = bracket_body.strip()
    default_wait = (60.0 / current_bpm) if is_slide else 0.0
    if not body:
        return (default_wait, 0.0)

    if "##" in body:
        wait_str, rhs = body.split("##", 1)
        wait_sec = float(wait_str) if wait_str.strip() else default_wait
        if "#" in rhs:
            bpm_str, ratio_str = rhs.split("#", 1)
            ref_bpm = float(bpm_str) if bpm_str.strip() else current_bpm
            div_str, cnt_str = ratio_str.split(":", 1)
            dur_sec = (240.0 / ref_bpm) * (float(cnt_str) / float(div_str))
        elif ":" in rhs:
            div_str, cnt_str = rhs.split(":", 1)
            dur_sec = (240.0 / current_bpm) * (float(cnt_str) / float(div_str))
        else:
            dur_sec = float(rhs)
        return (wait_sec, max(0.0, dur_sec))

    if "#" in body:
        lhs, rhs = body.split("#", 1)
        if ":" in rhs:
            ref_bpm = float(lhs) if lhs.strip() else current_bpm
            wait_sec = (60.0 / ref_bpm) if is_slide else 0.0
            div_str, cnt_str = rhs.split(":", 1)
            dur_sec = (240.0 / ref_bpm) * (float(cnt_str) / float(div_str))
            return (wait_sec, max(0.0, dur_sec))
        # `[#sec]` explicit duration
        dur_sec = float(rhs) if rhs.strip() else float(lhs)
        return (default_wait, max(0.0, dur_sec))

    if ":" in body:
        div_str, cnt_str = body.split(":", 1)
        div_val = float(div_str)
        cnt_val = float(cnt_str)
        dur_sec = 0.0 if div_val == 0 else (240.0 / current_bpm) * (cnt_val / div_val)
        return (default_wait, max(0.0, dur_sec))

    return (default_wait, max(0.0, float(body)))


def _parse_slide_chain(
    start_btn: int,
    chain_str: str,
    current_bpm: float,
) -> Tuple[List[SlideSegmentData], float, float]:
    """Parse a single slide arm (possibly chained like `-4-7[4:1]` or `V73[4:1]` or `-4[8:1]-7[8:1]`)
    into `(segments, wait_sec, total_duration_sec)`.
    """
    # Extract all `[...]` timing blocks
    brackets = list(re.finditer(r"\[([^\]]*)\]", chain_str))
    wait_sec = 60.0 / current_bpm
    total_dur = 0.0
    if brackets:
        w0, _ = _parse_Bracket_Timing(brackets[0].group(1), current_bpm, is_slide=True)
        wait_sec = w0
        for b in brackets:
            _, d = _parse_Bracket_Timing(b.group(1), current_bpm, is_slide=True)
            total_dur += d

    # Remove `[...]` and modifier letters (`b`, `x`) to parse segment geometry
    geom_str = re.sub(r"\[[^\]]*\]", "", chain_str)
    geom_str = re.sub(r"[bxBX\s]", "", geom_str)

    tokens = _SLIDE_OP_PATTERN.split(geom_str)
    # tokens looks like ['', op1, digits1, op2, digits2, ...]
    cur_start = start_btn
    segments: List[SlideSegmentData] = []
    idx = 1
    while idx < len(tokens):
        op = tokens[idx]
        digits = tokens[idx + 1] if idx + 1 < len(tokens) else ""
        idx += 2
        if not digits:
            continue
        if op == "V":
            via_btn = int(digits[0])
            end_btn = int(digits[1]) if len(digits) >= 2 else via_btn
            route = build_slide_route("V", cur_start, end_btn, via_btn)
            segments.append(SlideSegmentData("V", cur_start, end_btn, via_btn, route))
            cur_start = end_btn
        else:
            end_btn = int(digits[0])
            route = build_slide_route(op, cur_start, end_btn, None)
            segments.append(SlideSegmentData(op, cur_start, end_btn, None, route))
            cur_start = end_btn

    if total_dur <= 0.0:
        total_dur = 120.0 / current_bpm
    return segments, wait_sec, total_dur


def _combine_chained_slide_routes(
    segments: Sequence[SlideSegmentData],
) -> Tuple[Tuple[SlideAreaStep, ...], Optional[Tuple[Tuple[SlideAreaStep, ...], ...]], Tuple[Tuple[float, float], ...], float]:
    """Combine sensor queues and waypoints across a chained slide (`ConnSlide` in MajdataPlay)."""
    if not segments:
        return ((), None, (), 0.15)
    if len(segments) == 1:
        r = segments[0].route
        return (r.queue, r.wifi_queues, r.waypoints_mm, r.const_proportion)

    combined_queue: List[SlideAreaStep] = []
    combined_waypoints: List[Tuple[float, float]] = []
    n_segs = len(segments)
    for s_idx, seg in enumerate(segments):
        q = seg.route.queue
        is_last_seg = (s_idx == n_segs - 1)
        for a_idx, step in enumerate(q):
            # Avoid duplicating shared boundary sensor between segment i end and segment i+1 start
            if s_idx > 0 and a_idx == 0 and combined_queue and combined_queue[-1].areas == step.areas:
                continue
            combined_queue.append(
                SlideAreaStep(
                    areas=step.areas,
                    is_skippable=step.is_skippable,
                    is_last=(is_last_seg and a_idx == len(q) - 1),
                )
            )
        for w_idx, wp in enumerate(seg.route.waypoints_mm):
            if s_idx > 0 and w_idx == 0 and combined_waypoints:
                continue
            combined_waypoints.append(wp)

    last_const = segments[-1].route.const_proportion / max(1, n_segs)
    return (tuple(combined_queue), None, tuple(combined_waypoints), last_const)


def _split_chord_subtokens(step_body: str) -> List[Tuple[str, bool]]:
    """Split a chord body (between commas) into individual note tokens and pseudo-each flags.

    Returns list of `(token_str, is_pseudo_after_prev)`.
    Handles:
      - `/` simultaneous separator
      - `` ` `` pseudo-simultaneous separator
      - Compact multi-digit tap/hold/slide strings like `45`, `15b`, `4h[4:1]5h[4:1]`, `12`
    """
    s = step_body.strip()
    if not s:
        return []

    # First split by `/` and `` ` `` outside `[...]`
    parts: List[Tuple[str, bool]] = []
    bracket_depth = 0
    cur: List[str] = []
    next_is_pseudo = False
    for ch in s:
        if ch == "[":
            bracket_depth += 1
            cur.append(ch)
        elif ch == "]":
            bracket_depth = max(0, bracket_depth - 1)
            cur.append(ch)
        elif bracket_depth == 0 and ch in ("/", "`"):
            tok = "".join(cur).strip()
            if tok:
                parts.append((tok, next_is_pseudo))
            cur = []
            next_is_pseudo = (ch == "`")
        else:
            cur.append(ch)
    tail = "".join(cur).strip()
    if tail:
        parts.append((tail, next_is_pseudo))

    # Expand compact multi-note expressions within each part (e.g. "45", "45b", "1h[4:1]5h[4:1]")
    expanded: List[Tuple[str, bool]] = []
    single_note_re = re.compile(
        r"""
        (?:
            # Touch or TouchHold: A1..E8, C1, C2, C followed by optional modifiers & [dur]
            (?:[ABDE][1-8]|C[12]?)[bfxBFX]*(?:h[bfxBFX]*(?:\[[^\]]*\])?)?[bfxBFX]*
            |
            # Button note (1..8): Tap, Hold, or Slide (including *-chained arms)
            [1-8][bfxBFX?!$]*(?:h[bfxBFX]*(?:\[[^\]]*\])?)?[bfxBFX?!$]*
            (?:
                (?:\*?[-^<>vpqszwV]+[1-8]{1,2}[bxBX]*(?:\[[^\]]*\])?[bxBX]*)+
            )?
        )
        """,
        re.VERBOSE,
    )

    for raw_part, is_pseudo in parts:
        # Special case: pure digits with trailing b/x like "45b" -> apply b/x to both if compact chord
        m_compact_mod = re.fullmatch(r"([1-8]{2,})([bxBX]+)", raw_part)
        if m_compact_mod:
            digits, mod = m_compact_mod.group(1), m_compact_mod.group(2)
            for idx, d in enumerate(digits):
                expanded.append((f"{d}{mod}", is_pseudo if idx == 0 else False))
            continue

        sub_matches = [m.group(0) for m in single_note_re.finditer(raw_part) if m.group(0).strip()]
        if len(sub_matches) <= 1:
            expanded.append((raw_part, is_pseudo))
        else:
            for idx, sm in enumerate(sub_matches):
                expanded.append((sm, is_pseudo if idx == 0 else False))

    return expanded


def _parse_single_note_token(
    tok: str,
    note_time: float,
    bar_idx: int,
    beat_in_bar: float,
    current_bpm: float,
    next_id: int,
) -> List[ParsedNote]:
    """Parse a single note token into 1 or more `ParsedNote` objects (e.g. Slide produces Star Tap + Slide)."""
    t_str = tok.strip()
    if not t_str:
        return []

    # 1. Check Touch / TouchHold (`A1`..`E8`, `C`, `C1`, `C2`)
    m_touch = re.match(r"^(C[12]?|[ABDE][1-8])(.*)$", t_str, re.IGNORECASE)
    if m_touch:
        zone = m_touch.group(1).upper()
        rest = m_touch.group(2)
        rest_no_bracket = re.sub(r"\[[^\]]*\]", "", rest).lower()
        is_hold = "h" in rest_no_bracket
        is_firework = "f" in rest_no_bracket
        is_break = "b" in rest_no_bracket
        is_ex = "x" in rest_no_bracket
        btn = int(zone[1]) if (len(zone) == 2 and zone[0] == "A") else None
        dur = 0.0
        if is_hold:
            m_br = re.search(r"\[([^\]]*)\]", rest)
            if m_br:
                _, dur = _parse_Bracket_Timing(m_br.group(1), current_bpm, is_slide=False)
            else:
                dur = FRAME_LENGTH_SEC
        return [
            ParsedNote(
                note_id=next_id,
                note_type=NoteType.TOUCH_HOLD if is_hold else NoteType.TOUCH,
                time=note_time,
                bar=bar_idx,
                beat_in_bar=beat_in_bar,
                bpm=current_bpm,
                key=zone,
                button=btn,
                sensor_zone=zone,
                is_break=is_break,
                is_ex=is_ex,
                is_firework=is_firework,
                duration=dur,
                end_time=note_time + dur,
                judge_time=note_time,
                raw_token=t_str,
            )
        ]

    # 2. Button note starting with 1..8
    if not t_str[0].isdigit():
        return []
    btn = int(t_str[0])
    if not (1 <= btn <= 8):
        return []

    rest = t_str[1:]
    # Check if this token contains a slide operator outside brackets
    rest_outside = re.sub(r"\[[^\]]*\]", "[]", rest)
    slide_match = _SLIDE_OP_PATTERN.search(rest_outside)

    if slide_match is not None:
        # Find position of first slide operator in original `rest`
        first_op_match = _SLIDE_OP_PATTERN.search(rest)
        assert first_op_match is not None
        head_mods = rest[: first_op_match.start()]
        slide_body = rest[first_op_match.start() :]

        is_headless = ("?" in head_mods) or ("!" in head_mods)
        head_is_break = ("b" in head_mods.lower())
        head_is_ex = ("x" in head_mods.lower())

        created: List[ParsedNote] = []
        cur_id = next_id
        if not is_headless:
            created.append(
                ParsedNote(
                    note_id=cur_id,
                    note_type=NoteType.TAP,
                    time=note_time,
                    bar=bar_idx,
                    beat_in_bar=beat_in_bar,
                    bpm=current_bpm,
                    key=str(btn),
                    button=btn,
                    sensor_zone=f"A{btn}",
                    is_break=head_is_break,
                    is_ex=head_is_ex,
                    is_star_head=True,
                    duration=0.0,
                    end_time=note_time,
                    judge_time=note_time,
                    raw_token=t_str,
                )
            )
            cur_id += 1

        # Split multi-arm slides separated by `*` (e.g. `1-4[8:1]*-6[8:1]`)
        arms = [a.strip() for a in slide_body.split("*") if a.strip()]
        for arm_str in arms:
            arm_no_br = re.sub(r"\[[^\]]*\]", "", arm_str)
            slide_is_break = "b" in arm_no_br.lower()
            segments, wait_sec, dur_sec = _parse_slide_chain(btn, arm_str, current_bpm)
            if not segments:
                continue
            q, wifi_q, waypoints, const_prop = _combine_chained_slide_routes(segments)
            shoot_t = note_time + wait_sec
            end_t = shoot_t + dur_sec
            last_wait = dur_sec * const_prop
            judge_t = shoot_t + dur_sec * (1.0 - const_prop)
            created.append(
                ParsedNote(
                    note_id=cur_id,
                    note_type=NoteType.SLIDE,
                    time=note_time,
                    bar=bar_idx,
                    beat_in_bar=beat_in_bar,
                    bpm=current_bpm,
                    key=str(btn),
                    button=btn,
                    sensor_zone=f"A{btn}",
                    is_break=slide_is_break,
                    is_headless_slide=is_headless,
                    duration=dur_sec,
                    wait_time=wait_sec,
                    shoot_time=shoot_t,
                    end_time=end_t,
                    judge_time=judge_t,
                    last_wait_time=last_wait,
                    slide_segments=tuple(segments),
                    slide_queue=q,
                    wifi_queues=wifi_q,
                    slide_waypoints_mm=waypoints,
                    raw_token=t_str,
                )
            )
            cur_id += 1
        return created

    # Hold or Tap
    rest_no_br = re.sub(r"\[[^\]]*\]", "", rest).lower()
    is_hold = "h" in rest_no_br
    is_break = "b" in rest_no_br
    is_ex = "x" in rest_no_br
    is_star = "$" in rest_no_br

    if is_hold:
        m_br = re.search(r"\[([^\]]*)\]", rest)
        if m_br:
            _, dur = _parse_Bracket_Timing(m_br.group(1), current_bpm, is_slide=False)
        else:
            dur = FRAME_LENGTH_SEC
        return [
            ParsedNote(
                note_id=next_id,
                note_type=NoteType.HOLD,
                time=note_time,
                bar=bar_idx,
                beat_in_bar=beat_in_bar,
                bpm=current_bpm,
                key=str(btn),
                button=btn,
                sensor_zone=f"A{btn}",
                is_break=is_break,
                is_ex=is_ex,
                duration=dur,
                end_time=note_time + dur,
                judge_time=note_time,
                raw_token=t_str,
            )
        ]

    return [
        ParsedNote(
            note_id=next_id,
            note_type=NoteType.TAP,
            time=note_time,
            bar=bar_idx,
            beat_in_bar=beat_in_bar,
            bpm=current_bpm,
            key=str(btn),
            button=btn,
            sensor_zone=f"A{btn}",
            is_break=is_break,
            is_ex=is_ex,
            is_star_head=is_star,
            duration=0.0,
            end_time=note_time,
            judge_time=note_time,
            raw_token=t_str,
        )
    ]


def parse_simai_chart(
    chart_source: str | os.PathLike,
    difficulty: str | int = "auto",
    first_offset_sec: Optional[float] = None,
) -> ParsedChart:
    """Parse either a `maidata.txt` file (or directory containing it) or an inline Simai string."""
    title = "InlineSimaiChart"
    artist = "Unknown"
    diff_label = str(difficulty)
    level_str = "14"
    offset_sec = float(first_offset_sec or 0.0)
    inote_text = ""

    # Check if chart_source is a path on disk
    source_str = str(chart_source)
    resolved_source_path: Optional[str] = None
    is_path_candidate = (
        isinstance(chart_source, os.PathLike)
        or ("\n" not in source_str and len(source_str) < 260 and ("/" in source_str or "\\" in source_str or source_str.endswith(".txt") or source_str.endswith(".simai")))
    )
    if is_path_candidate:
        p = Path(source_str)
        if p.is_dir():
            p = p / "maidata.txt"
        if p.exists():
            resolved_source_path = str(p.resolve())
            raw_file = p.read_text(encoding="utf-8", errors="replace")
            if "&inote" in raw_file or "&title" in raw_file:
                fields = parse_maidata_fields(raw_file)
                title = fields.get("title", p.parent.name or "MaimaiChart")
                artist = fields.get("artist", "Unknown")
                if first_offset_sec is None and "first" in fields:
                    try:
                        offset_sec = float(fields["first"])
                    except ValueError:
                        offset_sec = 0.0
                diff_key = _DIFFICULTY_MAP.get(str(difficulty).strip().lower(), "auto")
                if diff_key == "auto":
                    for candidate in ("5", "6", "4", "3", "2", "1", "7"):
                        if fields.get(f"inote_{candidate}", "").strip():
                            diff_key = candidate
                            break
                inote_text = fields.get(f"inote_{diff_key}", "") or fields.get("inote", "")
                level_str = fields.get(f"lv_{diff_key}", "14")
                diff_label = f"inote_{diff_key}"
            else:
                inote_text = raw_file
                title = p.stem

    if not inote_text:
        if "&inote" in source_str:
            fields = parse_maidata_fields(source_str)
            title = fields.get("title", title)
            artist = fields.get("artist", artist)
            if first_offset_sec is None and "first" in fields:
                try:
                    offset_sec = float(fields["first"])
                except ValueError:
                    offset_sec = 0.0
            diff_key = _DIFFICULTY_MAP.get(str(difficulty).strip().lower(), "auto")
            if diff_key == "auto":
                for candidate in ("5", "6", "4", "3", "2", "1", "7"):
                    if fields.get(f"inote_{candidate}", "").strip():
                        diff_key = candidate
                        break
            inote_text = fields.get(f"inote_{diff_key}", "") or fields.get("inote", "")
            level_str = fields.get(f"lv_{diff_key}", level_str)
            diff_label = f"inote_{diff_key}"
        else:
            inote_text = source_str

    cleaned = _strip_comments(inote_text)
    current_bpm = 120.0
    initial_bpm: Optional[float] = None
    current_div = 4.0
    fixed_step_sec: Optional[float] = None
    current_time = offset_sec
    measure_acc = 0.0  # In whole-measure units (1.0 = one 4/4 bar)
    bar_starts_exact: Dict[int, float] = {0: offset_sec}

    notes: List[ParsedNote] = []
    next_note_id = 1

    # Tokenize stream into control blocks `(bpm)`, `{div}`, `,`, `E`, and note bodies
    token_pattern = re.compile(r"(\([0-9.+#\s]+\)|\{[0-9.+#\s]+\}|,|E(?![1-8]))")
    raw_pieces = token_pattern.split(cleaned)
    pending_note_chunks: List[str] = []

    def _flush_step(chunks: List[str]) -> None:
        nonlocal next_note_id
        step_str = "".join(chunks).strip()
        chunks.clear()
        if not step_str:
            return
        bar_idx = int(math.floor(measure_acc + 1e-9))
        beat_in_bar = (measure_acc - bar_idx) * 4.0
        subtokens = _split_chord_subtokens(step_str)
        pseudo_offset = 0.0
        step_notes: List[ParsedNote] = []
        for sub_str, is_pseudo in subtokens:
            if is_pseudo:
                pseudo_offset += FRAME_LENGTH_SEC
            created = _parse_single_note_token(
                sub_str,
                note_time=current_time + pseudo_offset,
                bar_idx=bar_idx,
                beat_in_bar=beat_in_bar,
                current_bpm=current_bpm,
                next_id=next_note_id,
            )
            if created:
                next_note_id += len(created)
                for n in created:
                    if is_pseudo:
                        n.is_pseudo_each = True
                step_notes.extend(created)

        strike_cnt = sum(1 for n in step_notes if n.note_type != NoteType.SLIDE)
        if strike_cnt >= 2:
            for n in step_notes:
                n.is_each = True
        notes.extend(step_notes)

    for piece in raw_pieces:
        if not piece:
            continue
        p_strip = piece.strip()
        if not p_strip:
            continue
        if p_strip.startswith("(") and p_strip.endswith(")"):
            val_str = p_strip[1:-1].strip()
            if val_str:
                current_bpm = float(val_str)
                if initial_bpm is None:
                    initial_bpm = current_bpm
        elif p_strip.startswith("{") and p_strip.endswith("}"):
            val_str = p_strip[1:-1].strip()
            if val_str.startswith("#"):
                fixed_step_sec = float(val_str[1:])
            else:
                fixed_step_sec = None
                current_div = float(val_str)
        elif p_strip == ",":
            _flush_step(pending_note_chunks)
            prev_t_step = current_time
            prev_m_step = measure_acc
            if fixed_step_sec is not None:
                current_time += fixed_step_sec
                measure_acc += fixed_step_sec / (240.0 / current_bpm)
            else:
                step_measures = 1.0 / current_div if current_div > 0 else 0.25
                current_time += step_measures * (240.0 / current_bpm)
                measure_acc += step_measures
            b_prev = int(math.floor(prev_m_step + 1e-9))
            b_curr = int(math.floor(measure_acc + 1e-9))
            if b_curr > b_prev and measure_acc > prev_m_step + 1e-12:
                for b_k in range(b_prev + 1, b_curr + 1):
                    if b_k not in bar_starts_exact:
                        frac = (float(b_k) - prev_m_step) / (measure_acc - prev_m_step)
                        bar_starts_exact[b_k] = prev_t_step + frac * (current_time - prev_t_step)
        elif p_strip == "E":
            _flush_step(pending_note_chunks)
            break
        else:
            pending_note_chunks.append(p_strip)

    if pending_note_chunks:
        _flush_step(pending_note_chunks)

    if initial_bpm is None:
        initial_bpm = current_bpm

    # Sort notes chronologically by time, then note_id
    notes.sort(key=lambda n: (round(n.time, 6), n.note_id))

    # Group into ChartEvents (notes within 1.5 ms are treated as simultaneous chord event)
    events: List[ChartEvent] = []
    for n in notes:
        if events and abs(n.time - events[-1].time) <= 0.0015:
            events[-1].notes.append(n)
        else:
            events.append(
                ChartEvent(
                    event_id=len(events),
                    time=n.time,
                    bar=n.bar,
                    beat_in_bar=n.beat_in_bar,
                    bpm=n.bpm,
                    notes=[n],
                )
            )

    # Compute bar timestamps
    max_bar = max((n.bar for n in notes), default=0)
    total_dur = max((n.end_time for n in notes), default=current_time)
    bar_dur_default = 240.0 / initial_bpm
    bar_timestamps: List[Tuple[int, float, float]] = []
    for b in range(max_bar + 1):
        if b in bar_starts_exact:
            b_start = bar_starts_exact[b]
            b_end = bar_starts_exact.get(b + 1, b_start + (240.0 / current_bpm))
            bar_timestamps.append((b, max(0.0, b_start), max(b_start + 1e-3, b_end)))
        else:
            b_start = offset_sec + b * bar_dur_default
            bar_timestamps.append((b, b_start, b_start + bar_dur_default))

    return ParsedChart(
        title=title,
        artist=artist,
        difficulty_label=diff_label,
        level_str=level_str,
        first_offset_sec=offset_sec,
        initial_bpm=initial_bpm,
        notes=notes,
        events=events,
        total_duration_sec=total_dur,
        bar_timestamps=bar_timestamps,
        source_path=resolved_source_path,
    )


def get_available_difficulties(source: str | Path) -> List[Tuple[str, str, str]]:
    """Return list of `(diff_key, short_name, level_str)` available in a `maidata.txt` file or folder."""
    p = Path(source)
    if p.is_dir():
        p = p / "maidata.txt"
    if not p.exists():
        return [("5", "MAS", "14")]
    try:
        raw_text = p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return [("5", "MAS", "14")]
    fields = parse_maidata_fields(raw_text)
    short_names = {
        "2": "BAS",
        "3": "ADV",
        "4": "EXP",
        "5": "MAS",
        "6": "ReM",
        "7": "UTA",
    }
    out: List[Tuple[str, str, str]] = []
    for k in ("2", "3", "4", "5", "6", "7"):
        inote = fields.get(f"inote_{k}", "").strip()
        if inote:
            lv = fields.get(f"lv_{k}", "?").strip() or "?"
            out.append((k, short_names[k], lv))
    return out if out else [("5", "MAS", "14")]
