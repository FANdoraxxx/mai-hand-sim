"""Alias module for `sim_core.judge` matching Section 2 specification naming."""
from sim_core.judge import (
    JudgeGrade,
    NoteJudgementResult,
    RuntimeSlideArea,
    SlideSensorQueueTracker,
    compute_dx_achievement_rate,
    hold_end_judge,
    is_button_note_triggered,
    judge_slide_tail_timing,
    judge_tap_timing,
    judge_touch_timing,
)

__all__ = [
    "JudgeGrade",
    "NoteJudgementResult",
    "RuntimeSlideArea",
    "SlideSensorQueueTracker",
    "compute_dx_achievement_rate",
    "hold_end_judge",
    "is_button_note_triggered",
    "judge_slide_tail_timing",
    "judge_tap_timing",
    "judge_touch_timing",
]
