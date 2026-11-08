"""Chart Evaluator & Scorer for Maimai DX Transformer Downstream Critic / RL Reward."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from sim_core.config import (
    AVOID_FLYING_HAND_DEFAULT,
    DEFAULT_HIT_MODE,
    DEFAULT_SKILL_PROFILE,
    HitMode,
    SkillProfileConfig,
    UnreasonableType,
)
from sim_core.geometry import get_active_sensors_for_palm
from sim_core.judge import (
    JudgeGrade,
    NoteJudgementResult,
    SlideSensorQueueTracker,
    compute_dx_achievement_rate,
    hold_end_judge,
    is_button_note_triggered,
    judge_slide_tail_timing,
    judge_tap_timing,
    judge_touch_timing,
)
from sim_core.maidata_parser import NoteType, ParsedChart, ParsedNote, parse_simai_chart
from sim_core.motion import compute_motion_features, export_trajectory
from sim_core.planner import (
    PlannedEventStep,
    PlannerState,
    PlanResult,
    UnreasonableEventRecord,
    plan_chart_movements,
)
from sim_core.player_profile import get_skill_profile


@dataclass
class ChartEvaluationResult:
    """Structured evaluation result returned by the scorer."""
    is_playable: bool
    total_ergonomic_score: float
    simulated_achievement_rate: float
    simulated_achievement_rate_str: str
    fatigue_curve: List[float]
    fatigue_curve_detailed: List[Dict[str, float]]
    unreasonable_events: List[Dict[str, Any]]
    decomposition_summary: Dict[str, int]
    judge_summary: Dict[str, int]
    fly_hand_rate: float = 0.0
    fly_hand_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    plan_result: Optional[PlanResult] = field(default=None, repr=False)
    soft_events: List[Dict[str, Any]] = field(default_factory=list)
    motion: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_playable": self.is_playable,
            "total_ergonomic_score": round(self.total_ergonomic_score, 2),
            "simulated_achievement_rate": round(self.simulated_achievement_rate, 4),
            "simulated_achievement_rate_str": self.simulated_achievement_rate_str,
            "fly_hand_rate": round(self.fly_hand_rate, 4),
            "fly_hand_count": self.fly_hand_count,
            "fatigue_curve": [round(x, 3) for x in self.fatigue_curve],
            "fatigue_curve_detailed": self.fatigue_curve_detailed,
            "unreasonable_events": self.unreasonable_events,
            "soft_events": self.soft_events,
            "decomposition_summary": self.decomposition_summary,
            "judge_summary": self.judge_summary,
            "metadata": self.metadata,
            "motion": self.motion,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    def export_trajectory(self, out_path: Optional[str | Path] = None, compressed: bool = True):
        return export_trajectory(self, out_path=out_path, compressed=compressed)

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]


def _simulate_chart_judgements(
    plan: PlanResult,
) -> Tuple[Dict[int, NoteJudgementResult], Dict[str, int]]:
    """Simulate MajdataPlay judgements across all notes in `plan.chart`."""
    chart = plan.chart
    profile = plan.profile
    hit_mode = plan.hit_mode
    steps_by_time: Dict[int, PlannedEventStep] = {
        int(round(st.time * 1000)): st for st in plan.planned_steps
    }

    # Index unreasonable events by timestamp window
    unr_list = plan.unreasonable_events

    results: Dict[int, NoteJudgementResult] = {}
    grade_counts: Dict[str, int] = {g.value: 0 for g in JudgeGrade}

    for n in chart.notes:
        hand_side, target_zone = plan.note_hand_map.get(n.note_id, ("R", n.sensor_zone))
        t_ms_key = int(round(n.time * 1000))
        step = steps_by_time.get(t_ms_key)

        # Check if note overlaps an unreasonable event
        matching_unr = [
            u for u in unr_list
            if abs(u.time - n.time) <= 0.30 and (not u.keys or n.key in u.keys or n.sensor_zone in u.keys)
        ]
        if not matching_unr:
            matching_unr = [u for u in unr_list if abs(u.time - n.time) <= 0.08]

        # Compute biomechanical timing error (ms)
        strain_ms = profile.timing_jitter_ms * 0.55
        if step is not None:
            h_ass = step.left if hand_side == "L" else step.right
            if h_ass.instant_freq_hz > 0:
                freq_ratio = h_ass.instant_freq_hz / profile.max_single_hand_tap_hz
                if freq_ratio > 1.0:
                    strain_ms += (freq_ratio - 1.0) * 95.0
                elif freq_ratio > 0.8:
                    strain_ms += (freq_ratio - 0.8) * 20.0
            strain_ms += min(35.0, step.body_state.posture_cost * 0.12)

        if matching_unr:
            u0 = matching_unr[0]
            if u0.type in (UnreasonableType.THREE_HAND_CONFLICT.value, UnreasonableType.DUAL_JACK_IMPOSSIBLE.value):
                strain_ms += 110.0 * u0.severity
            elif u0.type in (UnreasonableType.FORCED_SINGLE_HAND_JACK.value, UnreasonableType.FORCED_SINGLE_HAND_JUMP.value):
                strain_ms += 95.0 * u0.severity
            else:
                strain_ms += 70.0 * u0.severity

        diff_sec = strain_ms / 1000.0

        if n.note_type == NoteType.TAP:
            g = judge_tap_timing(diff_sec, is_ex=n.is_ex) or JudgeGrade.MISS
            results[n.note_id] = NoteJudgementResult(
                note_id=n.note_id,
                note_type=n.note_type,
                key=n.key,
                is_break=n.is_break,
                is_ex=n.is_ex,
                grade=g,
                diff_ms=strain_ms,
                judged_time=n.time + diff_sec,
                triggered_by=hand_side,
            )
        elif n.note_type == NoteType.TOUCH:
            g = judge_touch_timing(diff_sec, is_ex=n.is_ex) or JudgeGrade.MISS
            results[n.note_id] = NoteJudgementResult(
                note_id=n.note_id,
                note_type=n.note_type,
                key=n.key,
                is_break=n.is_break,
                is_ex=n.is_ex,
                grade=g,
                diff_ms=strain_ms,
                judged_time=n.time + diff_sec,
                triggered_by=hand_side,
            )
        elif n.note_type in (NoteType.HOLD, NoteType.TOUCH_HOLD):
            is_th = (n.note_type == NoteType.TOUCH_HOLD)
            head_g = (
                (judge_touch_timing(diff_sec, is_ex=n.is_ex) if is_th else judge_tap_timing(diff_sec, is_ex=n.is_ex))
                or JudgeGrade.MISS
            )
            release_sec = 0.0
            if matching_unr:
                release_sec = n.duration * 0.45
            final_g = hold_end_judge(head_g, strain_ms, n.duration, release_sec, is_touch_hold=is_th)
            results[n.note_id] = NoteJudgementResult(
                note_id=n.note_id,
                note_type=n.note_type,
                key=n.key,
                is_break=n.is_break,
                is_ex=n.is_ex,
                grade=final_g,
                diff_ms=strain_ms,
                judged_time=n.end_time,
                triggered_by=hand_side,
                hold_release_sec=release_sec,
            )
        elif n.note_type == NoteType.SLIDE:
            # Slice trajectory frames in O(1) index range during [n.time - 0.05, n.end_time + 0.35]
            completed_time: Optional[float] = None
            frames = plan.trajectory_frames
            f_dt = max(1e-4, frames[1].time - frames[0].time) if len(frames) >= 2 else 1.0
            if len(frames) >= 2 and f_dt <= 0.025:
                tracker = SlideSensorQueueTracker(n.slide_queue, n.wifi_queues)
                f_t0 = frames[0].time
                idx_start = max(0, int((n.shoot_time - 0.05 - f_t0) / f_dt))
                idx_end = min(len(frames), int((n.end_time + 0.35 - f_t0) / f_dt) + 1)
                relevant_sensors = {a for step in n.slide_queue for a in step.areas}
                if n.wifi_queues:
                    for wq in n.wifi_queues:
                        for step in wq:
                            relevant_sensors.update(step.areas)
                from sim_core.geometry import is_zone_triggered_by_palm
                for frame in frames[idx_start:idx_end]:
                    l_palm = frame.left_palm_xyz
                    r_palm = frame.right_palm_xyz
                    active_sensors = {
                        z_name
                        for z_name in relevant_sensors
                        if is_zone_triggered_by_palm((l_palm[0], l_palm[1]), l_palm[2], z_name)
                        or is_zone_triggered_by_palm((r_palm[0], r_palm[1]), r_palm[2], z_name)
                    }
                    if tracker.update(active_sensors):
                        completed_time = frame.time
                        break

            if completed_time is not None and not matching_unr:
                g = judge_slide_tail_timing(completed_time, n.judge_time, n.last_wait_time, is_ex=n.is_ex)
                slide_diff_ms = (completed_time - n.judge_time) * 1000.0
            elif not matching_unr:
                # Fallback analytical slide evaluation when frame sampling is coarse
                g = judge_slide_tail_timing(n.judge_time + diff_sec, n.judge_time, n.last_wait_time, is_ex=n.is_ex)
                slide_diff_ms = strain_ms
            else:
                g = JudgeGrade.FAST_GOOD if strain_ms < 180.0 else JudgeGrade.MISS
                slide_diff_ms = strain_ms

            results[n.note_id] = NoteJudgementResult(
                note_id=n.note_id,
                note_type=n.note_type,
                key=n.key,
                is_break=n.is_break,
                is_ex=n.is_ex,
                grade=g,
                diff_ms=slide_diff_ms,
                judged_time=completed_time or n.end_time,
                triggered_by=hand_side,
            )

    for res in results.values():
        grade_counts[res.grade.value] = grade_counts.get(res.grade.value, 0) + 1

    return results, grade_counts


def evaluate_chart(
    chart_source: str | Path | ParsedChart,
    difficulty: str | int = "auto",
    hit_mode: str | HitMode = DEFAULT_HIT_MODE,
    skill_profile: str | SkillProfileConfig = DEFAULT_SKILL_PROFILE,
    frame_fps: int = 60,
    avoid_flying_hand: bool = AVOID_FLYING_HAND_DEFAULT,
    include_motion: bool = True,
    mel: Any = None,
    audio_path: Optional[str] = None,
    initial_state: Optional[PlannerState] = None,
    t_range: Optional[Tuple[float, float]] = None,
) -> ChartEvaluationResult:
    """Evaluate a Maimai DX chart and return ergonomic score, playability, achievement rate,
    fatigue curve, unreasonable event diagnostics, and choreography `motion` features.
    """
    chart = (
        chart_source
        if isinstance(chart_source, ParsedChart)
        else parse_simai_chart(chart_source, difficulty=difficulty)
    )
    mode = HitMode.from_str(hit_mode)
    profile = get_skill_profile(skill_profile)
    eff_fps = max(20, int(frame_fps))

    plan = plan_chart_movements(
        chart=chart,
        hit_mode=mode,
        skill_profile=profile,
        frame_fps=eff_fps,
        avoid_flying_hand=avoid_flying_hand,
        initial_state=initial_state,
        t_range=t_range,
    )

    from sim_core.limits import split_events
    hard_events, soft_events = split_events([u.to_dict() for u in plan.unreasonable_events])
    is_playable = len(hard_events) == 0

    # Compute per-bar fatigue curve
    max_bar = max((b[0] for b in chart.bar_timestamps), default=0)
    fatigue_curve: List[float] = []
    fatigue_detailed: List[Dict[str, float]] = []
    for b_idx in range(max_bar + 1):
        b_steps = [s for s in plan.planned_steps if s.bar == b_idx]
        if not b_steps:
            fatigue_curve.append(0.0)
            fatigue_detailed.append({"bar": b_idx, "left": 0.0, "right": 0.0, "torso": 0.0, "total": 0.0})
            continue
        f_l = sum(s.step_fatigue_left for s in b_steps) / len(b_steps)
        f_r = sum(s.step_fatigue_right for s in b_steps) / len(b_steps)
        f_t = sum(s.step_fatigue_torso for s in b_steps) / len(b_steps)
        f_tot = round(f_l + f_r + f_t, 3)
        fatigue_curve.append(f_tot)
        fatigue_detailed.append(
            {
                "bar": b_idx,
                "left": round(f_l, 3),
                "right": round(f_r, 3),
                "torso": round(f_t, 3),
                "total": f_tot,
            }
        )

    # Compute smooth non-saturating ergonomic score (0.0 ~ 100.0)
    if not plan.planned_steps:
        ergonomic_score = 100.0
    else:
        n_st_erg = max(1, len(plan.planned_steps))
        mean_posture_cost = sum(s.body_state.posture_cost for s in plan.planned_steps) / len(plan.planned_steps)
        mean_fatigue = sum(fatigue_curve) / max(1, len(fatigue_curve))
        soft_sev_per_100 = sum(float(u.get("severity", 1.0)) for u in soft_events) / n_st_erg * 100.0
        hard_pen = sum(28.0 * float(u.get("severity", 1.0)) for u in hard_events)
        strain_term = (
            18.0 * (1.0 - math.exp(-mean_posture_cost / 160.0))
            + 16.0 * (1.0 - math.exp(-mean_fatigue / 85.0))
            + min(28.0, 2.2 * soft_sev_per_100)
            + hard_pen
        )
        raw_score = 100.0 - strain_term
        if not is_playable:
            raw_score = min(raw_score, 35.0)
        ergonomic_score = round(max(0.0, min(100.0, raw_score)), 2)

    # Simulate MajdataPlay judgements & 101.0000% DX achievement rate
    note_results, judge_summary = _simulate_chart_judgements(plan)
    ach_rate = compute_dx_achievement_rate(chart.notes, note_results)

    # Decomposition usage counts
    decomp_summary: Dict[str, int] = {
        "OUTER_INNER_SPLIT": sum(1 for s in plan.planned_steps if s.decomposition_used == "OUTER_INNER_SPLIT"),
        "ALTERNATING_TAP": sum(1 for s in plan.planned_steps if s.decomposition_used == "ALTERNATING_TAP"),
        "SINGLE_PALM_RUB": sum(1 for s in plan.planned_steps if s.decomposition_used == "SINGLE_PALM_RUB"),
        "RUB_SWEEP": sum(1 for s in plan.planned_steps if s.decomposition_used == "RUB_SWEEP" or s.left.mode == "Sweep" or s.right.mode == "Sweep"),
        "BUTTON_SWEEP": sum(1 for s in plan.planned_steps if s.decomposition_used == "RUB_SWEEP" or s.left.mode == "Sweep" or s.right.mode == "Sweep"),
    }

    # Compute Flying-Hand Rate (飞手率: fraction of steps where an active hand jumps >220mm above max_arm_velocity)
    fly_hand_count = 0
    last_l_hit: Optional[Tuple[float, Tuple[float, float]]] = None
    last_r_hit: Optional[Tuple[float, Tuple[float, float]]] = None
    for st in plan.planned_steps:
        is_fly = False
        if st.left.note_ids:
            if last_l_hit is not None:
                dt_l = max(0.015, st.time - last_l_hit[0])
                d_l = math.hypot(st.left.target_xy[0] - last_l_hit[1][0], st.left.target_xy[1] - last_l_hit[1][1])
                if d_l > 220.0 and (d_l / dt_l) > profile.max_arm_velocity:
                    is_fly = True
            last_l_hit = (st.time, st.left.target_xy)
        if st.right.note_ids:
            if last_r_hit is not None:
                dt_r = max(0.015, st.time - last_r_hit[0])
                d_r = math.hypot(st.right.target_xy[0] - last_r_hit[1][0], st.right.target_xy[1] - last_r_hit[1][1])
                if d_r > 220.0 and (d_r / dt_r) > profile.max_arm_velocity:
                    is_fly = True
            last_r_hit = (st.time, st.right.target_xy)
        if is_fly:
            fly_hand_count += 1
    fly_hand_rate = round((fly_hand_count / max(1, len(plan.planned_steps))) * 100.0, 2)
    ergonomic_score = round(max(0.0, ergonomic_score - fly_hand_rate * 2.5), 2)
    decomp_summary["FLY_HAND_COUNT"] = fly_hand_count

    slot_num = 5
    if chart.difficulty_label.startswith("inote_") and chart.difficulty_label[-1].isdigit():
        slot_num = int(chart.difficulty_label[-1])
    elif str(difficulty).isdigit():
        slot_num = int(difficulty)

    if include_motion:
        motion_dict, updated_sigs, final_cum_fat = compute_motion_features(
            plan=plan,
            slot=slot_num,
            mel_override=mel,
            audio_path_override=audio_path,
            prior_bar_sigs=initial_state.prior_bar_sigs if initial_state is not None else None,
            initial_cum_fatigue=initial_state.cum_fatigue if initial_state is not None else 0.0,
        )
        if plan.final_state is not None:
            plan.final_state.prior_bar_sigs = updated_sigs
            plan.final_state.cum_fatigue = final_cum_fat
    else:
        motion_dict = {}

    return ChartEvaluationResult(
        is_playable=is_playable,
        total_ergonomic_score=ergonomic_score,
        simulated_achievement_rate=ach_rate,
        simulated_achievement_rate_str=f"{ach_rate:.4f}%",
        fatigue_curve=fatigue_curve,
        fatigue_curve_detailed=fatigue_detailed,
        unreasonable_events=hard_events,
        soft_events=soft_events,
        decomposition_summary=decomp_summary,
        judge_summary=judge_summary,
        fly_hand_rate=fly_hand_rate,
        fly_hand_count=fly_hand_count,
        metadata={
            "title": chart.title,
            "artist": chart.artist,
            "difficulty": chart.difficulty_label,
            "level": chart.level_str,
            "hit_mode": mode.value,
            "skill_profile": profile.name,
            "note_count": len(chart.notes),
            "event_count": len(chart.events),
            "duration_sec": round(chart.total_duration_sec, 3),
        },
        plan_result=plan,
        motion=motion_dict,
    )


def compute_rl_reward(result: ChartEvaluationResult) -> float:
    """Convert `ChartEvaluationResult` into a normalized scalar reward in [-1.0, +1.0]
    for Transformer RL (GRPO / PPO / DPO) and Rejection Sampling ranking.
    """
    erg_norm = result.total_ergonomic_score / 100.0
    ach_norm = min(1.0, max(0.0, (result.simulated_achievement_rate - 90.0) / 11.0))
    fly_pen = min(0.5, result.fly_hand_rate * 0.05)
    unr_pen = sum(0.25 * float(e.get("severity", 1.0)) for e in result.unreasonable_events)
    raw = 0.65 * erg_norm + 0.35 * ach_norm - fly_pen - unr_pen
    if not result.is_playable:
        raw = min(raw, -0.15 - min(0.85, unr_pen * 0.5))
    return round(max(-1.0, min(1.0, raw)), 4)


def _batch_eval_worker(args: Tuple[str, str | int, str, str, int, bool]) -> Dict[str, Any]:
    src, diff, mode_s, prof_s, fps_val, avoid_fly = args
    res = evaluate_chart(
        src,
        difficulty=diff,
        hit_mode=mode_s,
        skill_profile=prof_s,
        frame_fps=fps_val,
        avoid_flying_hand=avoid_fly,
    )
    d = res.to_dict()
    d["rl_reward"] = compute_rl_reward(res)
    return d


def evaluate_charts_batch(
    chart_sources: Sequence[str | Path],
    difficulty: str | int = "auto",
    hit_mode: str | HitMode = DEFAULT_HIT_MODE,
    skill_profile: str | SkillProfileConfig = DEFAULT_SKILL_PROFILE,
    frame_fps: int = 20,
    avoid_flying_hand: bool = AVOID_FLYING_HAND_DEFAULT,
    max_workers: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Evaluate multiple charts in parallel across CPU processes (headless, zero GPU usage)."""
    mode_s = HitMode.from_str(hit_mode).value
    prof_s = get_skill_profile(skill_profile).name
    tasks = [
        (str(src), difficulty, mode_s, prof_s, frame_fps, avoid_flying_hand)
        for src in chart_sources
    ]
    if len(tasks) <= 1 or max_workers == 1:
        return [_batch_eval_worker(t) for t in tasks]
    workers = min(3, max_workers or min(len(tasks), 3))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_batch_eval_worker, tasks))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Human Maimai DX Play Simulator & Chart Quality Scorer"
    )
    parser.add_argument("chart", help="Path to maidata.txt or inline Simai chart string")
    parser.add_argument("--difficulty", "-d", default="auto", help="Difficulty index (1..7) or name (master, expert)")
    parser.add_argument("--mode", "-m", default="hybrid", choices=["outer", "inner", "hybrid"], help="Hit mode")
    parser.add_argument(
        "--profile",
        "-p",
        default=DEFAULT_SKILL_PROFILE,
        help="Skill profile (Lv12_Advanced, Lv13+_Expert, Lv14+_Master, Lv15_Apex)",
    )
    parser.add_argument("--output", "-o", default="", help="Optional path to save JSON output")
    parser.add_argument("--gui", action="store_true", help="Launch interactive Pygame GUI visualizer")
    args = parser.parse_args(argv)

    res = evaluate_chart(
        chart_source=args.chart,
        difficulty=args.difficulty,
        hit_mode=args.mode,
        skill_profile=args.profile,
    )
    js = res.to_json(indent=2)
    if args.output:
        Path(args.output).write_text(js, encoding="utf-8")
    print(js)
    if args.gui and res.plan_result is not None:
        from sim_core.visualizer import KinematicVisualizer
        vis = KinematicVisualizer()
        vis.run_interactive(
            res.plan_result,
            fps=60,
            eval_result=res,
            chart_source_path=args.chart,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
