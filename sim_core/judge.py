"""MajdataPlay-compatible Judgement Engine (Tap, Break, Hold, Touch, TouchHold, Slide, Wifi)
and 101.0000% DX Achievement Rate calculator.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from sim_core.config import (
    DELUXE_HOLD_RELEASE_IGNORE_SEC,
    HOLD_HEAD_IGNORE_SEC,
    HOLD_TAIL_IGNORE_SEC,
    HitMode,
    SCORE_WEIGHT_BREAK_BASE,
    SCORE_WEIGHT_BREAK_EXTRA,
    SCORE_WEIGHT_HOLD,
    SCORE_WEIGHT_SLIDE,
    SCORE_WEIGHT_TAP,
    SCORE_WEIGHT_TOUCH,
    SLIDE_CHECKABLE_LEAD_SEC,
    SLIDE_JUDGE_BASE_PERFECT_MSEC,
    SLIDE_JUDGE_GOOD_MSEC,
    SLIDE_JUDGE_GREAT_1ST_MSEC,
    SLIDE_JUDGE_GREAT_2ND_MSEC,
    SLIDE_JUDGE_GREAT_3RD_MSEC,
    SLIDE_JUDGE_MAX_EXT_MSEC,
    TAP_JUDGE_CRITICAL_MSEC,
    TAP_JUDGE_GOOD_MSEC,
    TAP_JUDGE_GREAT_1ST_MSEC,
    TAP_JUDGE_GREAT_2ND_MSEC,
    TAP_JUDGE_GREAT_3RD_MSEC,
    TAP_JUDGE_PERFECT_2ND_MSEC,
    TAP_JUDGE_PERFECT_3RD_MSEC,
    TOUCH_HOLD_HEAD_IGNORE_SEC,
    TOUCH_HOLD_TAIL_IGNORE_SEC,
    TOUCH_JUDGE_CRITICAL_MSEC,
    TOUCH_JUDGE_GOOD_MSEC,
    TOUCH_JUDGE_GREAT_1ST_MSEC,
    TOUCH_JUDGE_GREAT_2ND_MSEC,
    TOUCH_JUDGE_GREAT_3RD_MSEC,
    TOUCH_JUDGE_PERFECT_2ND_MSEC,
    TOUCH_JUDGE_PERFECT_3RD_MSEC,
)
from sim_core.geometry import SlideAreaStep
from sim_core.maidata_parser import NoteType, ParsedNote


class JudgeGrade(str, Enum):
    CRITICAL_PERFECT = "CRITICAL_PERFECT"
    FAST_PERFECT_2ND = "FAST_PERFECT_2ND"
    LATE_PERFECT_2ND = "LATE_PERFECT_2ND"
    FAST_PERFECT_3RD = "FAST_PERFECT_3RD"
    LATE_PERFECT_3RD = "LATE_PERFECT_3RD"
    FAST_GREAT_1ST = "FAST_GREAT_1ST"
    LATE_GREAT_1ST = "LATE_GREAT_1ST"
    FAST_GREAT_2ND = "FAST_GREAT_2ND"
    LATE_GREAT_2ND = "LATE_GREAT_2ND"
    FAST_GREAT_3RD = "FAST_GREAT_3RD"
    LATE_GREAT_3RD = "LATE_GREAT_3RD"
    FAST_GOOD = "FAST_GOOD"
    LATE_GOOD = "LATE_GOOD"
    MISS = "MISS"
    TOO_FAST = "TOO_FAST"

    @property
    def is_miss(self) -> bool:
        return self in (JudgeGrade.MISS, JudgeGrade.TOO_FAST)

    @property
    def is_perfect_or_better(self) -> bool:
        return self in (
            JudgeGrade.CRITICAL_PERFECT,
            JudgeGrade.FAST_PERFECT_2ND,
            JudgeGrade.LATE_PERFECT_2ND,
            JudgeGrade.FAST_PERFECT_3RD,
            JudgeGrade.LATE_PERFECT_3RD,
        )

    @property
    def is_great(self) -> bool:
        return self in (
            JudgeGrade.FAST_GREAT_1ST,
            JudgeGrade.LATE_GREAT_1ST,
            JudgeGrade.FAST_GREAT_2ND,
            JudgeGrade.LATE_GREAT_2ND,
            JudgeGrade.FAST_GREAT_3RD,
            JudgeGrade.LATE_GREAT_3RD,
        )

    @property
    def is_good(self) -> bool:
        return self in (JudgeGrade.FAST_GOOD, JudgeGrade.LATE_GOOD)


@dataclass
class NoteJudgementResult:
    note_id: int
    note_type: NoteType
    key: str
    is_break: bool
    is_ex: bool
    grade: JudgeGrade
    diff_ms: float
    judged_time: float
    triggered_by: str = ""          # 'L', 'R', 'LR', or 'NONE'
    hold_release_sec: float = 0.0


# ============================================================================
# 1. MajdataPlay Timing Window Evaluators
# ============================================================================
def judge_tap_timing(diff_sec: float, is_ex: bool = False) -> Optional[JudgeGrade]:
    """Evaluate Tap / Hold-head timing difference `diff_sec = current_sec - target_sec`.
    Returns None if outside the +/- 150 ms judgeable range.
    """
    diff_ms = abs(diff_sec) * 1000.0
    if diff_ms > TAP_JUDGE_GOOD_MSEC:
        return None
    if is_ex:
        return JudgeGrade.CRITICAL_PERFECT
    is_fast = diff_sec < 0.0
    if diff_ms <= TAP_JUDGE_CRITICAL_MSEC:
        return JudgeGrade.CRITICAL_PERFECT
    if diff_ms <= TAP_JUDGE_PERFECT_2ND_MSEC:
        return JudgeGrade.FAST_PERFECT_2ND if is_fast else JudgeGrade.LATE_PERFECT_2ND
    if diff_ms <= TAP_JUDGE_PERFECT_3RD_MSEC:
        return JudgeGrade.FAST_PERFECT_3RD if is_fast else JudgeGrade.LATE_PERFECT_3RD
    if diff_ms <= TAP_JUDGE_GREAT_1ST_MSEC:
        return JudgeGrade.FAST_GREAT_1ST if is_fast else JudgeGrade.LATE_GREAT_1ST
    if diff_ms <= TAP_JUDGE_GREAT_2ND_MSEC:
        return JudgeGrade.FAST_GREAT_2ND if is_fast else JudgeGrade.LATE_GREAT_2ND
    if diff_ms <= TAP_JUDGE_GREAT_3RD_MSEC:
        return JudgeGrade.FAST_GREAT_3RD if is_fast else JudgeGrade.LATE_GREAT_3RD
    return JudgeGrade.FAST_GOOD if is_fast else JudgeGrade.LATE_GOOD


def judge_touch_timing(diff_sec: float, is_ex: bool = False) -> Optional[JudgeGrade]:
    """Evaluate Touch / TouchHold-head timing difference `diff_sec = current_sec - target_sec`."""
    diff_ms = abs(diff_sec) * 1000.0
    is_fast = diff_sec < 0.0
    if is_fast and diff_ms > TOUCH_JUDGE_CRITICAL_MSEC:
        return None
    if not is_fast and diff_ms > TOUCH_JUDGE_GOOD_MSEC:
        return None
    if is_ex:
        return JudgeGrade.CRITICAL_PERFECT
    if diff_ms <= TOUCH_JUDGE_CRITICAL_MSEC:
        return JudgeGrade.CRITICAL_PERFECT
    if diff_ms <= TOUCH_JUDGE_PERFECT_2ND_MSEC:
        return JudgeGrade.LATE_PERFECT_2ND
    if diff_ms <= TOUCH_JUDGE_PERFECT_3RD_MSEC:
        return JudgeGrade.LATE_PERFECT_3RD
    if diff_ms <= TOUCH_JUDGE_GREAT_1ST_MSEC:
        return JudgeGrade.LATE_GREAT_1ST
    if diff_ms <= TOUCH_JUDGE_GREAT_2ND_MSEC:
        return JudgeGrade.LATE_GREAT_2ND
    if diff_ms <= TOUCH_JUDGE_GREAT_3RD_MSEC:
        return JudgeGrade.LATE_GREAT_3RD
    return JudgeGrade.LATE_GOOD


def judge_slide_tail_timing(
    current_sec: float,
    judge_timing_sec: float,
    last_wait_time_sec: float,
    is_ex: bool = False,
) -> JudgeGrade:
    """Evaluate Slide / Wifi completion timing (`SlideBase.cs::Judge`)."""
    if is_ex:
        return JudgeGrade.CRITICAL_PERFECT
    stay_ms = max(0.0, last_wait_time_sec * 1000.0)
    diff_sec = current_sec - judge_timing_sec
    is_fast = diff_sec < 0.0
    diff_ms = abs(diff_sec) * 1000.0

    ext_ms = min(stay_ms / 4.0, SLIDE_JUDGE_MAX_EXT_MSEC)
    seg_3rd_perfect = SLIDE_JUDGE_BASE_PERFECT_MSEC + ext_ms
    seg_1st_perfect = seg_3rd_perfect / 3.0
    seg_2nd_perfect = seg_3rd_perfect * (2.0 / 3.0)

    if diff_ms <= seg_1st_perfect:
        return JudgeGrade.CRITICAL_PERFECT
    if diff_ms <= seg_2nd_perfect:
        return JudgeGrade.FAST_PERFECT_2ND if is_fast else JudgeGrade.LATE_PERFECT_2ND
    if diff_ms <= seg_3rd_perfect:
        return JudgeGrade.FAST_PERFECT_3RD if is_fast else JudgeGrade.LATE_PERFECT_3RD
    if diff_ms <= SLIDE_JUDGE_GREAT_1ST_MSEC:
        return JudgeGrade.FAST_GREAT_1ST if is_fast else JudgeGrade.LATE_GREAT_1ST
    if diff_ms <= SLIDE_JUDGE_GREAT_2ND_MSEC:
        return JudgeGrade.FAST_GREAT_2ND if is_fast else JudgeGrade.LATE_GREAT_2ND
    if diff_ms <= SLIDE_JUDGE_GREAT_3RD_MSEC:
        return JudgeGrade.FAST_GREAT_3RD if is_fast else JudgeGrade.LATE_GREAT_3RD
    return JudgeGrade.FAST_GOOD if is_fast else JudgeGrade.LATE_GOOD


def hold_end_judge(
    head_grade: JudgeGrade,
    head_diff_ms: float,
    hold_length_sec: float,
    player_release_sec: float,
    is_touch_hold: bool = False,
) -> JudgeGrade:
    """MajdataPlay `NoteLongDrop.cs::HoldEndJudge` downgrade logic based on hold retention ratio."""
    ignore_sec = (
        (TOUCH_HOLD_HEAD_IGNORE_SEC + TOUCH_HOLD_TAIL_IGNORE_SEC)
        if is_touch_hold
        else (HOLD_HEAD_IGNORE_SEC + HOLD_TAIL_IGNORE_SEC)
    )
    if hold_length_sec <= ignore_sec:
        return head_grade

    offset_sec = 0.0 if head_diff_ms < 0.0 else (head_diff_ms / 1000.0)
    reality_ht = max(0.0, min(hold_length_sec - 0.30, hold_length_sec - ignore_sec - offset_sec))
    if reality_ht <= 1e-6:
        return head_grade

    percent = max(0.0, min(1.0, (reality_ht - player_release_sec) / reality_ht))
    is_late = head_diff_ms >= 0.0

    if percent >= 0.999:
        if head_grade.is_perfect_or_better:
            return head_grade
        if head_grade.is_great or head_grade.is_good:
            return JudgeGrade.LATE_GREAT_1ST if is_late else JudgeGrade.FAST_GREAT_1ST
        return JudgeGrade.LATE_GOOD
    if percent >= 0.67:
        if head_grade == JudgeGrade.CRITICAL_PERFECT:
            return JudgeGrade.LATE_PERFECT_2ND if is_late else JudgeGrade.FAST_PERFECT_2ND
        if head_grade.is_perfect_or_better:
            return head_grade
        if head_grade.is_great or head_grade.is_good:
            return JudgeGrade.LATE_GREAT_1ST if is_late else JudgeGrade.FAST_GREAT_1ST
        return JudgeGrade.LATE_GOOD
    if percent >= 0.33:
        if head_grade == JudgeGrade.CRITICAL_PERFECT:
            return JudgeGrade.LATE_GREAT_2ND if is_late else JudgeGrade.FAST_GREAT_2ND
        return JudgeGrade.LATE_GREAT_1ST if is_late else JudgeGrade.FAST_GREAT_1ST
    if percent >= 0.05:
        return JudgeGrade.LATE_GOOD if is_late else JudgeGrade.FAST_GOOD
    if head_grade.is_miss:
        return head_grade
    return JudgeGrade.LATE_GOOD if is_late else JudgeGrade.FAST_GOOD


# ============================================================================
# 2. Slide Sensor Queue State Machine (MajdataPlay SlideArea + Skip-1 Rule)
# ============================================================================
@dataclass
class RuntimeSlideArea:
    """Mutable state of one SlideArea step in a slide sensor queue."""
    areas: Tuple[str, ...]
    is_skippable: bool
    is_last: bool
    area_was_on: Dict[str, bool] = field(default_factory=dict)
    area_was_off: Dict[str, bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for a in self.areas:
            self.area_was_on[a] = False
            self.area_was_off[a] = False

    @property
    def on(self) -> bool:
        return any(self.area_was_on.values())

    @property
    def is_finished(self) -> bool:
        # Matches MajdataPlay SlideArea.cs:
        # If IsLast -> _wasOn; else -> _wasOn && _wasOff (OR policy across IncludedAreas)
        for a in self.areas:
            if self.is_last:
                if self.area_was_on[a]:
                    return True
            else:
                if self.area_was_on[a] and self.area_was_off[a]:
                    return True
        return False

    def check_sensors(self, active_sensors: Set[str]) -> None:
        for a in self.areas:
            is_currently_on = (a in active_sensors)
            if is_currently_on:
                self.area_was_on[a] = True
            else:
                if self.area_was_on[a]:
                    self.area_was_off[a] = True


class SlideSensorQueueTracker:
    """Implements MajdataPlay's `SensorCheckInternal` with Skip-1 Rule and
    tail-judgement anti-early-trigger residence semantics.
    """
    def __init__(
        self,
        queue_steps: Sequence[SlideAreaStep],
        wifi_queues: Optional[Tuple[Sequence[SlideAreaStep], Sequence[SlideAreaStep], Sequence[SlideAreaStep]]] = None,
    ) -> None:
        if wifi_queues is not None:
            self.queues: List[List[RuntimeSlideArea]] = [
                [RuntimeSlideArea(s.areas, s.is_skippable, s.is_last) for s in q]
                for q in wifi_queues
            ]
        else:
            self.queues = [
                [RuntimeSlideArea(s.areas, s.is_skippable, s.is_last) for s in queue_steps]
            ]

    @property
    def remaining_count(self) -> int:
        return max((len(q) for q in self.queues), default=0)

    @property
    def is_finished(self) -> bool:
        return self.remaining_count == 0

    @property
    def is_pending_finish(self) -> bool:
        """True when only the final sensor area remains (hand can leave to hit Taps and return for tail)."""
        return self.remaining_count == 1

    def current_target_areas(self) -> Tuple[str, ...]:
        targets: List[str] = []
        for q in self.queues:
            if q:
                for a in q[0].areas:
                    if a not in targets:
                        targets.append(a)
        return tuple(targets)

    def update(self, active_sensors: Set[str]) -> bool:
        """Run one tick of `SensorCheckInternal` across all active lanes. Returns True if finished."""
        for q in self.queues:
            self._sensor_check_internal(q, active_sensors)
        return self.is_finished

    @staticmethod
    def _sensor_check_internal(queue: List[RuntimeSlideArea], active_sensors: Set[str]) -> None:
        while queue:
            first = queue[0]
            first.check_sensors(active_sensors)

            if len(queue) >= 2 and (first.is_skippable or first.on):
                second = queue[1]
                second.check_sensors(active_sensors)
                if second.is_finished:
                    del queue[:2]
                    continue
                if second.on:
                    del queue[:1]
                    continue

            if first.is_finished:
                del queue[:1]
            return


# ============================================================================
# 3. Trigger Source Filter (OuterOnly / InnerOnly / Hybrid) & DX Score Summary
# ============================================================================
def is_button_note_triggered(
    button_idx: int,
    active_zones: Set[str],
    hit_mode: HitMode,
) -> bool:
    """Check if button 1..8 is triggered under `OuterOnly`, `InnerOnly`, or `Hybrid`."""
    outer_key = f"OB{button_idx}"
    inner_key = f"A{button_idx}"
    if hit_mode == HitMode.OUTER:
        return outer_key in active_zones
    if hit_mode == HitMode.INNER:
        return inner_key in active_zones
    return (outer_key in active_zones) or (inner_key in active_zones)


def compute_dx_achievement_rate(
    notes: Sequence[ParsedNote],
    results: Dict[int, NoteJudgementResult],
) -> float:
    """Compute MajdataPlay DX Achievement Rate (0.0000% ~ 101.0000%) from per-note judgement results.

    Follows `MajdataPlay/Assets/Scripts/Scenes/Game/ObjectCounter.cs::UpdateNoteScoreCount`
    and `UpdateAccRate` (`_accRate[4]`).
    """
    if not notes:
        return 101.0000

    total_base_score = 0
    total_extra_score = 0
    current_base_score = 0
    current_extra_score = 0

    for n in notes:
        if n.is_break:
            base_w = SCORE_WEIGHT_BREAK_BASE
            extra_w = SCORE_WEIGHT_BREAK_EXTRA
        elif n.note_type in (NoteType.TAP, NoteType.TOUCH):
            base_w = SCORE_WEIGHT_TAP
            extra_w = 0
        elif n.note_type in (NoteType.HOLD, NoteType.TOUCH_HOLD):
            base_w = SCORE_WEIGHT_HOLD
            extra_w = 0
        else:  # SLIDE
            base_w = SCORE_WEIGHT_SLIDE
            extra_w = 0

        total_base_score += base_w
        total_extra_score += extra_w

        res = results.get(n.note_id)
        grade = res.grade if res is not None else JudgeGrade.MISS

        if not n.is_break:
            if grade.is_miss:
                pass
            elif grade.is_good:
                current_base_score += int(base_w * 0.5)
            elif grade.is_great:
                current_base_score += int(base_w * 0.8)
            else:
                current_base_score += base_w
        else:
            # Break note scoring from ObjectCounter.cs
            if grade.is_miss:
                pass
            elif grade.is_good:
                current_base_score += 1000
                current_extra_score += 30
            elif grade in (JudgeGrade.FAST_GREAT_3RD, JudgeGrade.LATE_GREAT_3RD):
                current_base_score += 1250
                current_extra_score += 40
            elif grade in (JudgeGrade.FAST_GREAT_2ND, JudgeGrade.LATE_GREAT_2ND):
                current_base_score += 1500
                current_extra_score += 40
            elif grade in (JudgeGrade.FAST_GREAT_1ST, JudgeGrade.LATE_GREAT_1ST):
                current_base_score += 2000
                current_extra_score += 40
            elif grade in (JudgeGrade.FAST_PERFECT_3RD, JudgeGrade.LATE_PERFECT_3RD):
                current_base_score += 2500
                current_extra_score += 50
            elif grade in (JudgeGrade.FAST_PERFECT_2ND, JudgeGrade.LATE_PERFECT_2ND):
                current_base_score += 2500
                current_extra_score += 75
            else:  # CRITICAL_PERFECT
                current_base_score += 2500
                current_extra_score += 100

    if total_base_score == 0:
        return 101.0000

    base_pct = (current_base_score / float(total_base_score)) * 100.0
    if total_extra_score > 0:
        extra_pct = (current_extra_score / float(total_extra_score)) * 1.0
    else:
        # If a chart has 0 break notes, full combo critical perfect is 100.0000% (or +1% bonus if all perfect)
        all_cp = all(
            (results.get(n.note_id) is not None and results[n.note_id].grade == JudgeGrade.CRITICAL_PERFECT)
            for n in notes
        )
        extra_pct = 1.0 if all_cp else (base_pct * 0.01)

    return round(min(101.0000, max(0.0, base_pct + extra_pct)), 4)
