"""Inner-screen Slide Jump-Zone detector (`SLIDE_JUMP_ZONE`) backed by MaiMuriDX's
600-route slide judgement area table (`sim_core/slide_areas.json`).

Physical & MajdataPlay mechanism:
  While a Slide is active on `[shoot_time, end_time]`, if the OTHER hand is already
  holding a `HOLD` (`A_b`) or `TOUCH_HOLD` (`sensor`) that remains depressed through
  the slide's completion (`h.end_time >= sl.end_time`) and lies in the slide's final
  judgement area (`last=1`), advancing the slide to the penultimate sensor causes the
  already-pressed final area to immediately complete the slide prematurely ("内屏跳区无理").

  - Final area held to slide end (`last=1`, `need_end=True`): emitted with `severity=2.5`
    (HARD `SLIDE_JUMP_ZONE`, 100% superset of `mai.innermuri.check(..., mode='btn_last_end')`
    with <=0.2% official MiLK+ chart rate).
  - Penultimate area held to slide end (`last=2`, `need_end=True`, not in `last=1`):
    emitted with `severity=1.1` (SOFT `SLIDE_JUMP_ZONE` warning).
"""
from __future__ import annotations

from functools import lru_cache
import json
import os
from typing import Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

from sim_core.config import UnreasonableType
from sim_core.maidata_parser import NoteType, ParsedNote

_SLIDE_AREAS_TABLE: Optional[Dict[str, Dict]] = None


def get_slide_areas_table() -> Dict[str, Dict]:
    global _SLIDE_AREAS_TABLE
    if _SLIDE_AREAS_TABLE is not None:
        return _SLIDE_AREAS_TABLE
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    candidates = (
        os.path.join(here, "slide_areas.json"),
        os.path.join(root, "work", "slide_areas.json"),
        os.path.join(root, "release", "v6d2_echo", "work", "slide_areas.json"),
    )
    for p in candidates:
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                _SLIDE_AREAS_TABLE = json.load(f)
            return _SLIDE_AREAS_TABLE
    _SLIDE_AREAS_TABLE = {}
    return _SLIDE_AREAS_TABLE


@lru_cache(maxsize=4096)
def _cached_chain_areas(
    seg_keys: Tuple[str, ...],
) -> Optional[Tuple[FrozenSet[str], ...]]:
    tab = get_slide_areas_table()
    if not tab or not seg_keys:
        return None
    seq: List[FrozenSet[str]] = []
    for si, k in enumerate(seg_keys):
        entry = tab.get(k)
        if entry is None:
            return None
        for ai, area_list in enumerate(entry["areas"]):
            if si == 0 and ai == 0:
                continue
            seq.append(frozenset(area_list))
    return tuple(seq)


def slide_seg_key(start_btn: int, shape: str, end_btn: int, via_btn: Optional[int] = None) -> Optional[str]:
    if shape == "w":
        return None
    if shape == "V":
        if via_btn is None:
            return None
        return f"{start_btn}V{via_btn}{end_btn}"
    return f"{start_btn}{shape}{end_btn}"


def get_slide_later_area_sequence(slide: ParsedNote, include_grandv: bool = True) -> Optional[Tuple[FrozenSet[str], ...]]:
    if not slide.slide_segments:
        return None
    keys: List[str] = []
    for seg in slide.slide_segments:
        if seg.shape == "V" and not include_grandv:
            return None
        k = slide_seg_key(seg.start_btn, seg.shape, seg.end_btn, seg.via_btn)
        if k is None:
            return None
        keys.append(k)
    return _cached_chain_areas(tuple(keys))


def get_slide_all_traversed_zones(slide: ParsedNote) -> FrozenSet[str]:
    """Return all sensor zones traversed by `slide` (including start and C1/C2 aliases)."""
    zones: Set[str] = {slide.sensor_zone}
    seq = get_slide_later_area_sequence(slide, include_grandv=True)
    if seq is not None:
        for fs in seq:
            zones.update(fs)
    else:
        for step in slide.slide_queue:
            zones.update(step.areas)
        if slide.wifi_queues:
            for wq in slide.wifi_queues:
                for step in wq:
                    zones.update(step.areas)
    if "C" in zones or "C1" in zones or "C2" in zones:
        zones.update(("C", "C1", "C2"))
    return frozenset(zones)


def hold_sensor_pad(hold_note: ParsedNote) -> str:
    if hold_note.note_type == NoteType.TOUCH_HOLD:
        z = hold_note.sensor_zone
        return "C" if z in ("C", "C1", "C2") else z
    return f"A{hold_note.button or 1}"


def detect_slide_jump_zone_events(
    notes: Sequence[ParsedNote],
    include_soft_last2: bool = True,
) -> List[Tuple[float, int, List[str], str, float]]:
    """Detect inner-screen jump-zone conflicts across `notes`.

    Returns list of `(time, bar, keys, reason, severity)` tuples:
      - `severity = 2.5` for HARD final-area jump-zone (`last=1` + held to slide end)
      - `severity = 1.1` for SOFT penultimate-area overlap (`last=2` + held to slide end)
    """
    if not get_slide_areas_table():
        return []
    holds = [n for n in notes if n.note_type in (NoteType.HOLD, NoteType.TOUCH_HOLD)]
    if not holds:
        return []

    results: List[Tuple[float, int, List[str], str, float]] = []
    for sl in notes:
        if sl.note_type != NoteType.SLIDE:
            continue
        t0 = sl.shoot_time
        t1 = sl.end_time
        seq: Optional[Tuple[FrozenSet[str], ...]] = None
        checked_seq = False
        for h in holds:
            if h.note_id == sl.note_id:
                continue
            # Overlap on [t0, t1] and held until slide completion (within 2ms float tolerance)
            if h.time >= t1 - 1e-4 or h.end_time <= t0 + 1e-4:
                continue
            if h.end_time < t1 - 2e-3:
                continue
            if not checked_seq:
                seq = get_slide_later_area_sequence(sl, include_grandv=True)
                checked_seq = True
            if not seq:
                break
            hp = hold_sensor_pad(h)
            final_pads = seq[-1]
            ev_time = max(t0, h.time)
            ev_bar = sl.bar if t0 >= h.time else h.bar
            if hp in final_pads:
                results.append(
                    (
                        ev_time,
                        ev_bar,
                        [sl.raw_token or sl.key, h.raw_token or hp],
                        (
                            f"Slide jump-zone (final area {hp}): {h.note_type.value} on {hp} "
                            f"[{h.time:.2f}..{h.end_time:.2f}s] is held across Slide {sl.raw_token or sl.key} "
                            f"[{t0:.2f}..{t1:.2f}s] final area"
                        ),
                        2.5,
                    )
                )
            elif include_soft_last2 and len(seq) >= 2 and h.note_type == NoteType.HOLD and hp in seq[-2]:
                results.append(
                    (
                        ev_time,
                        ev_bar,
                        [sl.raw_token or sl.key, h.raw_token or hp],
                        (
                            f"Slide penultimate jump-zone risk ({hp}): HOLD on {hp} "
                            f"[{h.time:.2f}..{h.end_time:.2f}s] held across Slide {sl.raw_token or sl.key} "
                            f"penultimate area"
                        ),
                        1.1,
                    )
                )
    return results
