"""Direct zero-regex converter from `mai.simai` tick/beat `Note`s + `TempoMap` into `ParsedChart`.

Eliminates string serialization + regex parsing overhead in `evaluate_notes` and `window_score`,
cutting 4-bar window parse time from ~10 ms down to < 0.15 ms while preserving exact timing
and route geometry.
"""
from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

from sim_core.config import FRAME_LENGTH_SEC
from sim_core.geometry import build_slide_route
from sim_core.maidata_parser import (
    ChartEvent,
    NoteType,
    ParsedChart,
    ParsedNote,
    SlideSegmentData,
    _combine_chained_slide_routes,
)

TPB: int = 48

_SENSORS: Tuple[str, ...] = (
    *(f"A{i}" for i in range(1, 9)),
    *(f"B{i}" for i in range(1, 9)),
    "C",
    *(f"D{i}" for i in range(1, 9)),
    *(f"E{i}" for i in range(1, 9)),
)

_SHAPE_TO_OP = {
    "STRAIGHT": "-",
    "VEE": "v",
    "P": "p",
    "Q": "q",
    "PP": "pp",
    "QQ": "qq",
    "S": "s",
    "Z": "z",
    "FAN": "w",
    "GRANDV": "V",
}


def _canon_shape_to_simai_op(start_btn: int, shape: str) -> str:
    if shape in _SHAPE_TO_OP:
        return _SHAPE_TO_OP[shape]
    is_up = start_btn in (1, 2, 7, 8)
    if shape == "CW":
        return ">" if is_up else "<"
    if shape == "CCW":
        return "<" if is_up else ">"
    return "-"


def build_parsed_chart_from_notes(
    notes: Sequence,
    tm,
    slot: int = 5,
    title: str = "TickNotesChart",
    level_str: str = "14",
    unit_ticks: bool = True,
    source_path: Optional[str] = None,
    include_first_offset: bool = False,
) -> ParsedChart:
    """Build a `ParsedChart` directly from `mai.simai` `Note` objects and `TempoMap`."""
    scale = float(TPB) if unit_ticks else 1.0
    offset_sec = float(getattr(tm, "first", 0.0)) if include_first_offset else 0.0
    changes = list(tm.changes())
    initial_bpm = float(changes[0][1]) if changes else 120.0

    parsed_notes: List[ParsedNote] = []
    next_id = 1

    for n in notes:
        beat = float(n.beat) / scale
        bar_idx = max(0, int(math.floor(beat / 4.0 + 1e-9)))
        beat_in_bar = beat - bar_idx * 4.0
        t_sec = offset_sec + float(tm.to_sec(beat))
        cur_bpm = float(tm.bpm_at(beat))
        kind = n.kind

        if kind in ("touch", "thold"):
            zone = _SENSORS[int(n.pos)]
            btn = int(zone[1]) if (len(zone) == 2 and zone[0] == "A") else None
            is_hold = (kind == "thold")
            if is_hold:
                end_beat = float(n.beat + n.dur) / scale
                end_t = offset_sec + float(tm.to_sec(end_beat))
                dur_sec = max(FRAME_LENGTH_SEC, end_t - t_sec)
                end_t = t_sec + dur_sec
            else:
                dur_sec = 0.0
                end_t = t_sec
            parsed_notes.append(
                ParsedNote(
                    note_id=next_id,
                    note_type=NoteType.TOUCH_HOLD if is_hold else NoteType.TOUCH,
                    time=t_sec,
                    bar=bar_idx,
                    beat_in_bar=beat_in_bar,
                    bpm=cur_bpm,
                    key=zone,
                    button=btn,
                    sensor_zone=zone,
                    is_break=bool(getattr(n, "brk", False)),
                    is_ex=bool(getattr(n, "ex", False)),
                    is_firework=bool(getattr(n, "fw", False)),
                    duration=dur_sec,
                    end_time=end_t,
                    judge_time=t_sec,
                    raw_token=f"{zone}{'h' if is_hold else ''}",
                )
            )
            next_id += 1
        elif kind == "tap":
            btn_i = int(n.pos)
            parsed_notes.append(
                ParsedNote(
                    note_id=next_id,
                    note_type=NoteType.TAP,
                    time=t_sec,
                    bar=bar_idx,
                    beat_in_bar=beat_in_bar,
                    bpm=cur_bpm,
                    key=str(btn_i),
                    button=btn_i,
                    sensor_zone=f"A{btn_i}",
                    is_break=bool(getattr(n, "brk", False)),
                    is_ex=bool(getattr(n, "ex", False)),
                    duration=0.0,
                    end_time=t_sec,
                    judge_time=t_sec,
                    raw_token=f"{btn_i}{'b' if getattr(n, 'brk', False) else ''}{'x' if getattr(n, 'ex', False) else ''}",
                )
            )
            next_id += 1
        elif kind == "hold":
            btn_i = int(n.pos)
            end_beat = float(n.beat + n.dur) / scale
            end_t = offset_sec + float(tm.to_sec(end_beat))
            dur_sec = max(FRAME_LENGTH_SEC, end_t - t_sec)
            end_t = t_sec + dur_sec
            parsed_notes.append(
                ParsedNote(
                    note_id=next_id,
                    note_type=NoteType.HOLD,
                    time=t_sec,
                    bar=bar_idx,
                    beat_in_bar=beat_in_bar,
                    bpm=cur_bpm,
                    key=str(btn_i),
                    button=btn_i,
                    sensor_zone=f"A{btn_i}",
                    is_break=bool(getattr(n, "brk", False)),
                    is_ex=bool(getattr(n, "ex", False)),
                    duration=dur_sec,
                    end_time=end_t,
                    judge_time=t_sec,
                    raw_token=f"{btn_i}h",
                )
            )
            next_id += 1
        elif kind == "slide":
            btn_i = int(n.pos)
            is_headless = bool(getattr(n, "headless", False))
            if not is_headless:
                parsed_notes.append(
                    ParsedNote(
                        note_id=next_id,
                        note_type=NoteType.TAP,
                        time=t_sec,
                        bar=bar_idx,
                        beat_in_bar=beat_in_bar,
                        bpm=cur_bpm,
                        key=str(btn_i),
                        button=btn_i,
                        sensor_zone=f"A{btn_i}",
                        is_break=bool(getattr(n, "brk", False)),
                        is_ex=bool(getattr(n, "ex", False)),
                        is_star_head=True,
                        duration=0.0,
                        end_time=t_sec,
                        judge_time=t_sec,
                        raw_token=f"{btn_i}",
                    )
                )
                next_id += 1
            for arm in getattr(n, "arms", ()):
                shoot_beat = float(n.beat + arm.wait) / scale
                end_beat = float(n.beat + arm.wait + arm.dur) / scale
                shoot_t = offset_sec + float(tm.to_sec(shoot_beat))
                end_t = offset_sec + float(tm.to_sec(end_beat))
                wait_sec = max(0.0, shoot_t - t_sec)
                dur_sec = max(0.01, end_t - shoot_t)
                cur_start = btn_i
                seg_list: List[SlideSegmentData] = []
                tok_parts = [str(btn_i)]
                for seg in arm.segs:
                    op = _canon_shape_to_simai_op(cur_start, seg.shape)
                    via_b = int(seg.via) if (op == "V" and getattr(seg, "via", 0)) else None
                    end_b = int(seg.end)
                    route = build_slide_route(op, cur_start, end_b, via_b)
                    seg_list.append(SlideSegmentData(op, cur_start, end_b, via_b, route))
                    tok_parts.append(f"{op}{via_b or ''}{end_b}")
                    cur_start = end_b
                if not seg_list:
                    continue
                q, wifi_q, waypoints, const_prop = _combine_chained_slide_routes(seg_list)
                last_wait = dur_sec * const_prop
                judge_t = shoot_t + dur_sec * (1.0 - const_prop)
                parsed_notes.append(
                    ParsedNote(
                        note_id=next_id,
                        note_type=NoteType.SLIDE,
                        time=t_sec,
                        bar=bar_idx,
                        beat_in_bar=beat_in_bar,
                        bpm=cur_bpm,
                        key=str(btn_i),
                        button=btn_i,
                        sensor_zone=f"A{btn_i}",
                        is_break=bool(getattr(arm, "brk", False)),
                        is_headless_slide=is_headless,
                        duration=dur_sec,
                        wait_time=wait_sec,
                        shoot_time=shoot_t,
                        end_time=end_t,
                        judge_time=judge_t,
                        last_wait_time=last_wait,
                        slide_segments=tuple(seg_list),
                        slide_queue=q,
                        wifi_queues=wifi_q,
                        slide_waypoints_mm=waypoints,
                        raw_token="".join(tok_parts),
                    )
                )
                next_id += 1

    parsed_notes.sort(key=lambda x: (round(x.time, 6), x.note_id))

    events: List[ChartEvent] = []
    for pn in parsed_notes:
        if events and abs(pn.time - events[-1].time) <= 0.0015:
            events[-1].notes.append(pn)
        else:
            events.append(
                ChartEvent(
                    event_id=len(events),
                    time=pn.time,
                    bar=pn.bar,
                    beat_in_bar=pn.beat_in_bar,
                    bpm=pn.bpm,
                    notes=[pn],
                )
            )
    for ev in events:
        if sum(1 for x in ev.notes if x.note_type != NoteType.SLIDE) >= 2:
            for x in ev.notes:
                x.is_each = True

    max_bar = max((pn.bar for pn in parsed_notes), default=0)
    total_dur = max((pn.end_time for pn in parsed_notes), default=offset_sec)
    bar_timestamps: List[Tuple[int, float, float]] = []
    for b in range(max_bar + 1):
        t0 = offset_sec + float(tm.to_sec(4.0 * b))
        t1 = offset_sec + float(tm.to_sec(4.0 * (b + 1)))
        bar_timestamps.append((b, max(0.0, t0), max(t0 + 1e-3, t1)))

    return ParsedChart(
        title=title,
        artist="Unknown",
        difficulty_label=f"inote_{slot}",
        level_str=str(level_str),
        first_offset_sec=offset_sec,
        initial_bpm=initial_bpm,
        notes=parsed_notes,
        events=events,
        total_duration_sec=total_dur,
        bar_timestamps=bar_timestamps,
        source_path=source_path,
    )
