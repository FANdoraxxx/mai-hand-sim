"""Lookahead Beam Search / Viterbi DP Bimanual Action Planner,
Physiological Unreasonable Pattern Detector, and Continuous Trajectory Generator.
"""
from __future__ import annotations

import bisect
import math
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

from sim_core.config import (
    AVOID_FLYING_HAND_DEFAULT,
    DEFAULT_HIT_MODE,
    DEFAULT_SKILL_PROFILE,
    HAND_COLLISION_MIN_DIST_MM,
    HAND_COLLISION_Z_DIFF_MM,
    HOLD_EDGE_SHIFT_MAX_MM,
    HOLD_HEAD_IGNORE_SEC,
    HOLD_TAIL_EARLY_RELEASE_MAX_SEC,
    HOLD_TAIL_IGNORE_SEC,
    HitMode,
    PLANNER_BEAM_WIDTH,
    PLANNER_LOOKAHEAD_WINDOW,
    R_A_CENTER_MM,
    R_BUTTON_MM,
    R_HAND_MM,
    SAME_HAND_HOLD_SPAN_MM,
    TOUCH_HOLD_HEAD_IGNORE_SEC,
    TOUCH_HOLD_TAIL_IGNORE_SEC,
    UnreasonableType,
    Z_CONTACT_MM,
    Z_HOVER_DEFAULT_MM,
    Z_RELEASE_MM,
    Z_TAP_REBOUND_MM,
    SkillProfileConfig,
)
from sim_core.geometry import (
    OUTER_BUTTONS,
    TOUCH_ZONES,
    Point2D,
    Point3D,
    find_single_palm_cover_center,
    get_button_target_xy,
    get_sensor_zone,
    interpolate_waypoints,
    polar_cw_to_xy,
    solve_two_palm_coverage,
)
from sim_core.kinematics import (
    BodyKinematicsState,
    TorsoState,
    check_hand_collision,
    solve_beam_ik_step_njit,
    solve_coupled_body_ik,
)
from sim_core.maidata_parser import ChartEvent, NoteType, ParsedChart, ParsedNote
from sim_core.player_profile import get_skill_profile
from sim_core.slide_jump import detect_slide_jump_zone_events, get_slide_all_traversed_zones


@dataclass(frozen=True)
class HandAssignment:
    """Planned action for one hand at a discrete chart event."""
    hand: str                        # 'L' or 'R'
    mode: str                        # 'Hover', 'Outer', 'Inner', 'OuterSplit', 'InnerSplit', 'AltTap', 'Hold', 'Slide', 'Rub'
    target_xy: Point2D
    target_z_mm: float
    zone_names: Tuple[str, ...]      # Zones/buttons targeted by this hand
    note_ids: Tuple[int, ...]        # Notes struck or held by this hand at this event
    instant_freq_hz: float           # Current single-hand instantaneous tap frequency (Hz)
    streak_count: int                # Consecutive rapid taps by this hand


@dataclass
class UnreasonableEventRecord:
    """Structured record for any unreasonable / physically impossible pattern."""
    time: float
    bar: int
    type: str                        # Value from UnreasonableType
    keys: List[str]
    reason: str
    severity: float = 1.0
    event_id: int = -1
    raw_m: Optional[float] = None

    def to_dict(self) -> dict:
        d = {
            "time": round(self.time, 4),
            "bar": int(self.bar),
            "type": str(self.type),
            "keys": list(self.keys),
            "reason": self.reason,
            "severity": round(self.severity, 3),
        }
        if self.raw_m is not None:
            d["raw_m"] = round(float(self.raw_m), 4)
        return d


@dataclass
class PlannedEventStep:
    """Optimal planned bilateral state at a single ChartEvent."""
    event_id: int
    time: float
    bar: int
    bpm: float
    left: HandAssignment
    right: HandAssignment
    body_state: BodyKinematicsState
    decomposition_used: str = "NONE"   # 'NONE', 'OUTER_INNER_SPLIT', 'ALTERNATING_TAP', 'SINGLE_PALM_RUB'
    step_cost: float = 0.0
    step_fatigue_left: float = 0.0
    step_fatigue_right: float = 0.0
    step_fatigue_torso: float = 0.0


@dataclass(slots=True)
class TrajectoryFrame:
    """Continuous 2.5D kinematic & telemetry frame (for Judge Engine & Visualizer)."""
    time: float
    bar: int
    bpm: float
    left_mode: str
    right_mode: str
    left_freq_hz: float
    right_freq_hz: float
    left_zones: Tuple[str, ...]
    right_zones: Tuple[str, ...]
    active_note_zones: Tuple[str, ...]
    active_unreasonable: Tuple[UnreasonableEventRecord, ...]
    left_palm_xyz: Point3D = (-110.0, -45.0, 148.0)
    right_palm_xyz: Point3D = (110.0, -45.0, 148.0)
    _torso: Optional[TorsoState] = None
    _body: Optional[BodyKinematicsState] = None

    @property
    def body(self) -> BodyKinematicsState:
        if self._body is None:
            from sim_core.kinematics import (
                check_hand_collision,
                compute_shoulder_positions,
                resolve_arm_capsule_collision_and_layers,
                solve_arm_ik,
            )
            t_st = self._torso if self._torso is not None else TorsoState()
            s_l, s_r = compute_shoulder_positions(t_st)
            ik_l = solve_arm_ik("L", s_l, self.left_palm_xyz, t_st)
            ik_r = solve_arm_ik("R", s_r, self.right_palm_xyz, t_st)
            dist_xy, z_diff, is_coll = check_hand_collision(self.left_palm_xyz, self.right_palm_xyz)
            ik_l, ik_r, layer_order, forearm_dist_3d, cross_xy, is_arm_coll = (
                resolve_arm_capsule_collision_and_layers(ik_l, ik_r)
            )
            self._body = BodyKinematicsState(
                time=self.time,
                torso=t_st,
                left_arm=ik_l,
                right_arm=ik_r,
                hand_distance_xy_mm=dist_xy,
                hand_z_diff_mm=z_diff,
                is_hand_collision=is_coll or is_arm_coll,
                is_extreme_cross_arm=False,
                is_torso_whiplash=False,
                posture_cost=ik_l.joint_cost + ik_r.joint_cost,
                arm_layer_order=layer_order,
                forearm_min_dist_3d_mm=forearm_dist_3d,
                crossing_point_xy=cross_xy,
            )
        return self._body


@dataclass
class PlanResult:
    chart: ParsedChart
    hit_mode: HitMode
    profile: SkillProfileConfig
    planned_steps: List[PlannedEventStep]
    unreasonable_events: List[UnreasonableEventRecord]
    note_hand_map: Dict[int, Tuple[str, str]]  # note_id -> (hand 'L'/'R', target_zone)
    trajectory_frames: List[TrajectoryFrame]
    final_state: Optional["PlannerState"] = None


@dataclass
class PlannerState:
    """Serializable bimanual planner state at a bar/window boundary for incremental `window_score`."""
    end_time: float = 0.0
    end_bar: int = 0
    hit_mode: str = "hybrid"
    skill_profile: str = DEFAULT_SKILL_PROFILE
    nodes: Tuple[Dict[str, Any], ...] = ()
    carry_notes: Tuple[Dict[str, Any], ...] = ()
    prior_bar_sigs: Tuple[Tuple[float, ...], ...] = ()
    cum_fatigue: float = 0.0
    tm_changes: Tuple[Tuple[float, float], ...] = ((0.0, 150.0),)
    first_offset_sec: float = 0.0
    slot: int = 5
    source_path: Optional[str] = None
    title: str = ""
    last_flag_times: Tuple[float, float, float, float, float] = (-999.0, -999.0, -999.0, -999.0, -999.0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "end_time": round(float(self.end_time), 6),
            "end_bar": int(self.end_bar),
            "hit_mode": str(self.hit_mode),
            "skill_profile": str(self.skill_profile),
            "nodes": [dict(nd) for nd in self.nodes],
            "carry_notes": [dict(cn) for cn in self.carry_notes],
            "prior_bar_sigs": [list(s) for s in self.prior_bar_sigs],
            "cum_fatigue": round(float(self.cum_fatigue), 4),
            "tm_changes": [[float(b), float(p)] for b, p in self.tm_changes],
            "first_offset_sec": round(float(self.first_offset_sec), 6),
            "slot": int(self.slot),
            "source_path": self.source_path,
            "title": str(self.title),
            "last_flag_times": [round(float(x), 6) for x in self.last_flag_times],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PlannerState":
        tm_c = d.get("tm_changes", ((0.0, 150.0),))
        lft = d.get("last_flag_times", (-999.0, -999.0, -999.0, -999.0, -999.0))
        return cls(
            end_time=float(d.get("end_time", 0.0)),
            end_bar=int(d.get("end_bar", 0)),
            hit_mode=str(d.get("hit_mode", "hybrid")),
            skill_profile=str(d.get("skill_profile", DEFAULT_SKILL_PROFILE)),
            nodes=tuple(dict(nd) for nd in d.get("nodes", ())),
            carry_notes=tuple(dict(cn) for cn in d.get("carry_notes", ())),
            prior_bar_sigs=tuple(tuple(float(x) for x in s) for s in d.get("prior_bar_sigs", ())),
            cum_fatigue=float(d.get("cum_fatigue", 0.0)),
            tm_changes=tuple((float(b), float(p)) for b, p in tm_c),
            first_offset_sec=float(d.get("first_offset_sec", 0.0)),
            slot=int(d.get("slot", 5)),
            source_path=d.get("source_path"),
            title=str(d.get("title", "")),
            last_flag_times=tuple(float(x) for x in lft[:5]) if len(lft) >= 5 else (-999.0, -999.0, -999.0, -999.0, -999.0),
        )


# ============================================================================
# Internal Beam Search / Viterbi State
# ============================================================================
@dataclass
class _BeamState:
    left_xy: Point2D
    right_xy: Point2D
    left_z: float
    right_z: float
    left_tap_times: Tuple[float, ...]
    right_tap_times: Tuple[float, ...]
    left_last_key: str
    right_last_key: str
    # Active holds: tuple of (note_id, hand, zone_name, target_xy, start_time, end_time, is_touch_hold)
    active_holds: Tuple[Tuple[int, str, str, Point2D, float, float, bool], ...]
    # Active slides: tuple of (note_id, hand, shoot_time, end_time, judge_time, start_xy, end_xy, waypoints, end_btn)
    active_slides: Tuple[Tuple[int, str, float, float, float, Point2D, Point2D, Tuple[Point2D, ...], int], ...]
    torso: TorsoState
    cum_cost: float
    left_hold_release: Tuple[float, Point2D] = (-999.0, (-110.0, -45.0))
    right_hold_release: Tuple[float, Point2D] = (-999.0, (110.0, -45.0))
    last_slide_end_l: Tuple[int, float, Point2D] = (-1, -999.0, (0.0, 0.0))
    last_slide_end_r: Tuple[int, float, Point2D] = (-1, -999.0, (0.0, 0.0))
    left_last_contact: Tuple[float, Point2D] = (-999.0, (-110.0, -45.0))
    right_last_contact: Tuple[float, Point2D] = (-999.0, (110.0, -45.0))
    left_same_key_streak: int = 1
    right_same_key_streak: int = 1
    history_node: Optional[Tuple[Any, Tuple[HandAssignment, HandAssignment, Any, str, float, Dict[int, Tuple[str, str]]]]] = None
    forced_single_jack_flags: List[Tuple[float, int, List[str], str, float]] = field(default_factory=list)
    forced_single_jump_flags: List[Tuple[float, int, List[str], str, float]] = field(default_factory=list)
    collision_flags: List[Tuple[float, int, List[str], str, float]] = field(default_factory=list)
    cross_flags: List[Tuple[float, int, List[str], str, float]] = field(default_factory=list)
    flying_hand_flags: List[Tuple[float, int, List[str], str, float]] = field(default_factory=list)

    def unpack_history(self) -> List[Tuple[HandAssignment, HandAssignment, Any, str, float, Dict[int, Tuple[str, str]]]]:
        out = []
        cur = self.history_node
        while cur is not None:
            out.append(cur[1])
            cur = cur[0]
        out.reverse()
        return out

    def to_snapshot_dict(self, base_cost: float = 0.0) -> Dict[str, Any]:
        return {
            "left_xy": [float(self.left_xy[0]), float(self.left_xy[1])],
            "right_xy": [float(self.right_xy[0]), float(self.right_xy[1])],
            "left_z": float(self.left_z),
            "right_z": float(self.right_z),
            "left_tap_times": [float(x) for x in self.left_tap_times[-8:]],
            "right_tap_times": [float(x) for x in self.right_tap_times[-8:]],
            "left_last_key": str(self.left_last_key),
            "right_last_key": str(self.right_last_key),
            "active_holds": [
                [int(h[0]), str(h[1]), str(h[2]), [float(h[3][0]), float(h[3][1])], float(h[4]), float(h[5]), bool(h[6])]
                for h in self.active_holds
            ],
            "active_slides": [
                [
                    int(s[0]), str(s[1]), float(s[2]), float(s[3]), float(s[4]),
                    [float(s[5][0]), float(s[5][1])],
                    [float(s[6][0]), float(s[6][1])],
                    [[float(wp[0]), float(wp[1])] for wp in s[7]],
                    int(s[8]),
                ]
                for s in self.active_slides
            ],
            "torso": [
                float(self.torso.dx_mm), float(self.torso.dy_mm), float(self.torso.yaw_deg),
                float(self.torso.vx_mm_s), float(self.torso.vy_mm_s),
                float(self.torso.yaw_vel_deg_s), float(self.torso.yaw_accel_deg_s2),
            ],
            "cum_cost": float(self.cum_cost - base_cost),
            "left_hold_release": [float(self.left_hold_release[0]), [float(self.left_hold_release[1][0]), float(self.left_hold_release[1][1])]],
            "right_hold_release": [float(self.right_hold_release[0]), [float(self.right_hold_release[1][0]), float(self.right_hold_release[1][1])]],
            "last_slide_end_l": [int(self.last_slide_end_l[0]), float(self.last_slide_end_l[1]), [float(self.last_slide_end_l[2][0]), float(self.last_slide_end_l[2][1])]],
            "last_slide_end_r": [int(self.last_slide_end_r[0]), float(self.last_slide_end_r[1]), [float(self.last_slide_end_r[2][0]), float(self.last_slide_end_r[2][1])]],
            "left_last_contact": [float(self.left_last_contact[0]), [float(self.left_last_contact[1][0]), float(self.left_last_contact[1][1])]],
            "right_last_contact": [float(self.right_last_contact[0]), [float(self.right_last_contact[1][0]), float(self.right_last_contact[1][1])]],
            "left_same_key_streak": int(self.left_same_key_streak),
            "right_same_key_streak": int(self.right_same_key_streak),
        }

    @classmethod
    def from_snapshot_dict(cls, d: Dict[str, Any]) -> "_BeamState":
        tr = d.get("torso", [0.0] * 7)
        lhr = d.get("left_hold_release", [-999.0, [-110.0, -45.0]])
        rhr = d.get("right_hold_release", [-999.0, [110.0, -45.0]])
        lse = d.get("last_slide_end_l", [-1, -999.0, [0.0, 0.0]])
        rse = d.get("last_slide_end_r", [-1, -999.0, [0.0, 0.0]])
        llc = d.get("left_last_contact", [-999.0, [-110.0, -45.0]])
        rlc = d.get("right_last_contact", [-999.0, [110.0, -45.0]])
        return cls(
            left_xy=(float(d["left_xy"][0]), float(d["left_xy"][1])),
            right_xy=(float(d["right_xy"][0]), float(d["right_xy"][1])),
            left_z=float(d["left_z"]),
            right_z=float(d["right_z"]),
            left_tap_times=tuple(float(x) for x in d.get("left_tap_times", ())),
            right_tap_times=tuple(float(x) for x in d.get("right_tap_times", ())),
            left_last_key=str(d.get("left_last_key", "")),
            right_last_key=str(d.get("right_last_key", "")),
            active_holds=tuple(
                (int(h[0]), str(h[1]), str(h[2]), (float(h[3][0]), float(h[3][1])), float(h[4]), float(h[5]), bool(h[6]))
                for h in d.get("active_holds", ())
            ),
            active_slides=tuple(
                (
                    int(s[0]), str(s[1]), float(s[2]), float(s[3]), float(s[4]),
                    (float(s[5][0]), float(s[5][1])),
                    (float(s[6][0]), float(s[6][1])),
                    tuple((float(wp[0]), float(wp[1])) for wp in s[7]),
                    int(s[8]),
                )
                for s in d.get("active_slides", ())
            ),
            torso=TorsoState(
                dx_mm=float(tr[0]), dy_mm=float(tr[1]), yaw_deg=float(tr[2]),
                vx_mm_s=float(tr[3]), vy_mm_s=float(tr[4]),
                yaw_vel_deg_s=float(tr[5]), yaw_accel_deg_s2=float(tr[6]),
            ),
            cum_cost=float(d.get("cum_cost", 0.0)),
            left_hold_release=(float(lhr[0]), (float(lhr[1][0]), float(lhr[1][1]))),
            right_hold_release=(float(rhr[0]), (float(rhr[1][0]), float(rhr[1][1]))),
            last_slide_end_l=(int(lse[0]), float(lse[1]), (float(lse[2][0]), float(lse[2][1]))),
            last_slide_end_r=(int(rse[0]), float(rse[1]), (float(rse[2][0]), float(rse[2][1]))),
            left_last_contact=(float(llc[0]), (float(llc[1][0]), float(llc[1][1]))),
            right_last_contact=(float(rlc[0]), (float(rlc[1][0]), float(rlc[1][1]))),
            left_same_key_streak=int(d.get("left_same_key_streak", 1)),
            right_same_key_streak=int(d.get("right_same_key_streak", 1)),
        )


def _resolve_note_default_target(note: ParsedNote, hit_mode: HitMode) -> Tuple[str, Point2D, str]:
    """Return (zone_name, target_xy, mode_label) for a note under `hit_mode`."""
    if note.note_type in (NoteType.TOUCH, NoteType.TOUCH_HOLD):
        z = get_sensor_zone(note.sensor_zone)
        return (z.name, z.center, "Inner")
    btn = note.button or 1
    if hit_mode == HitMode.OUTER:
        return (f"OB{btn}", get_button_target_xy(btn, use_outer=True), "Outer")
    if hit_mode == HitMode.INNER:
        return (f"A{btn}", get_button_target_xy(btn, use_outer=False), "Inner")
    # Hybrid defaults: Outer for Break/Hold or upper arc, Inner A for fast standard taps
    if note.is_break:
        return (f"OB{btn}", get_button_target_xy(btn, use_outer=True), "Outer")
    return (f"A{btn}", get_button_target_xy(btn, use_outer=False), "Inner")


def _compute_hand_tap_metrics(
    prev_tap_times: Tuple[float, ...],
    current_time: float,
) -> Tuple[float, int, Tuple[float, ...]]:
    """Compute (instant_freq_hz, streak_count, updated_tap_times) when a hand strikes at `current_time`."""
    if not prev_tap_times:
        return (0.0, 1, (current_time,))
    dt = current_time - prev_tap_times[-1]
    if dt <= 1e-4:
        # Simultaneous multi-note rub by the same hand counts as 1 strike impulse
        return (
            (1.0 / (prev_tap_times[-1] - prev_tap_times[-2])) if len(prev_tap_times) >= 2 else 0.0,
            len(prev_tap_times),
            prev_tap_times,
        )
    inst_hz = 1.0 / dt
    # Reset streak if pause > 0.35 sec
    if dt > 0.35:
        return (inst_hz, 1, (current_time,))
    new_times = (*prev_tap_times[-7:], current_time)
    # Count how many recent taps had intervals < 0.22 sec
    streak = 1
    for i in range(len(new_times) - 1, 0, -1):
        if new_times[i] - new_times[i - 1] <= 0.22:
            streak += 1
        else:
            break
    return (inst_hz, streak, new_times)


def _is_hold_locking_hand(
    hold_tuple: Tuple[int, str, str, Point2D, float, float, bool],
    t: float,
) -> bool:
    """Check if an active Hold/TouchHold is locking its assigned hand at time `t`.
    Accounts for MajdataPlay's 200ms tail-ignore window on sustained holds (`dur >= 0.25s`),
    allowing safe early release up to `min(0.145, 0.28 * dur)` before `t_end`.
    """
    _nid, _hand, _zname, _xy, t_start, t_end, _is_th = hold_tuple
    dur = max(0.0, t_end - t_start)
    tail_slack = min(0.145, 0.28 * dur) if dur >= 0.25 else 0.012
    effective_lock_end = max(t_start + 0.02, t_end - tail_slack)
    return t_start - 1e-4 <= t <= effective_lock_end


def _is_slide_locking_hand(
    slide_tuple: Tuple[int, str, float, float, float, Point2D, Point2D, Tuple[Point2D, ...], int],
    t: float,
    max_arm_vel: float = 4500.0,
) -> bool:
    """Section 6: Slide Trajectory Continuity Constraint.
    - Fast slides (`dur <= 0.26s`): active hand follows `shoot_t .. shoot_t + 0.80 * dur`.
    - Medium/slow slides (`dur > 0.26s`): players can early-park to the penultimate sensor within
      `min(0.20, 0.50 * dur)` after launch, freeing the hand to alternate taps before brushing the tail!
    """
    _nid, _hand, shoot_t, end_t, judge_t, _s_xy, _e_xy, _wps, _ebtn = slide_tuple
    dur = max(0.01, end_t - shoot_t)
    if _hand == "LR":
        return shoot_t <= t <= (shoot_t + min(0.32, 0.82 * dur))
    active_stroke_end = shoot_t + (0.80 * dur if dur <= 0.26 else min(0.20, 0.52 * dur))
    return shoot_t <= t <= active_stroke_end


def _detect_dual_jackhammer_streaks(
    events: Sequence[ChartEvent],
    profile: SkillProfileConfig,
) -> List[UnreasonableEventRecord]:
    """Law 1 (Section 4.1.1): High-BPM Dual-Tap Jackhammer Ban (`DUAL_JACK_IMPOSSIBLE`).

    Detects consecutive N >= 3 dual-tap events where the bilateral strike frequency `1 / dt`
    exceeds `profile.max_dual_jack_hz(N)`.
    """
    records: List[UnreasonableEventRecord] = []
    streak_events: List[ChartEvent] = []

    def _evaluate_streak(seq: List[ChartEvent]) -> None:
        if len(seq) < 3:
            return
        dts = [seq[i].time - seq[i - 1].time for i in range(1, len(seq))]
        valid_dts = [d for d in dts if d > 1e-4]
        if not valid_dts:
            return
        mean_dt = sum(valid_dts) / len(valid_dts)
        freq_hz = 1.0 / mean_dt
        n_pairs = len(seq)
        # Check how many consecutive steps repeat at least one key (true jackhammer vs rotating spin)
        sweep_transitions = 0
        same_key_transitions = 0
        for i in range(1, len(seq)):
            k_prev = {n.key for n in seq[i - 1].strike_notes}
            k_curr = {n.key for n in seq[i].strike_notes}
            if k_prev & k_curr:
                same_key_transitions += 1
            b_prev = sorted(n.button for n in seq[i - 1].strike_notes if n.button is not None)
            b_curr = sorted(n.button for n in seq[i].strike_notes if n.button is not None)
            if len(b_prev) == 2 and len(b_curr) == 2:
                d0 = min((b_curr[0] - b_prev[0]) % 8, (b_prev[0] - b_curr[0]) % 8)
                d1 = min((b_curr[1] - b_prev[1]) % 8, (b_prev[1] - b_curr[1]) % 8)
                d0_flip = min((b_curr[1] - b_prev[0]) % 8, (b_prev[0] - b_curr[1]) % 8)
                d1_flip = min((b_curr[0] - b_prev[1]) % 8, (b_prev[1] - b_curr[0]) % 8)
                if (d0 == 1 and d1 == 1) or (d0_flip == 1 and d1_flip == 1):
                    sweep_transitions += 1

        if sweep_transitions >= len(seq) - 1 and same_key_transitions == 0:
            return
        if sweep_transitions >= len(seq) - 2 and same_key_transitions <= 1:
            return

        # Moving/stepping chord streams where <40% of transitions repeat a button are not stationary dual-jackhammers
        is_rotating_spin = (same_key_transitions == 0 and n_pairs <= 5) or (same_key_transitions < max(2, int(0.45 * (n_pairs - 1))))
        all_ex = all(
            n.is_ex for ev in seq for n in ev.strike_notes if n.note_type in (NoteType.TAP, NoteType.HOLD)
        )
        limit_hz = (
            profile.dual_jack_short_burst_hz * (1.12 if all_ex else 1.04)
            if is_rotating_spin
            else profile.max_dual_jack_hz(n_pairs, all_ex=all_ex)
        )
        if freq_hz > limit_hz:
            ev0 = seq[0]
            keys_involved = sorted({n.key for ev in seq for n in ev.strike_notes})
            records.append(
                UnreasonableEventRecord(
                    time=ev0.time,
                    bar=ev0.bar,
                    type=UnreasonableType.DUAL_JACK_IMPOSSIBLE.value,
                    keys=keys_involved,
                    reason=(
                        f"High-BPM dual-tap jackhammer: {n_pairs} consecutive dual taps at "
                        f"{freq_hz:.2f} Hz (BPM {ev0.bpm:.0f}) > physiological limit {limit_hz:.2f} Hz"
                    ),
                    severity=min(3.0, (freq_hz - limit_hz) / 2.5 + 1.0),
                    event_id=ev0.event_id,
                    raw_m=freq_hz / max(0.1, limit_hz),
                )
            )

    for ev in events:
        # Only button TAP/HOLD notes require discrete lift-and-strike impulses;
        # Inner-screen TOUCH cascades/piles are wiped continuously ("大手一推") without re-strike lock!
        strikes = [n for n in ev.strike_notes if n.note_type in (NoteType.TAP, NoteType.HOLD)]
        if len(strikes) >= 2:
            if not streak_events:
                streak_events = [ev]
            else:
                dt = ev.time - streak_events[-1].time
                prev_dt = (
                    (streak_events[-1].time - streak_events[-2].time)
                    if len(streak_events) >= 2
                    else dt
                )
                # Check if part of a continuous high-speed dual-tap stream (dt <= 0.14s, i.e. >= 7.1 Hz)
                if 1e-4 < dt <= 0.14 and (ev.bar // 4) == (streak_events[-1].bar // 4) and (prev_dt <= 1e-4 or (0.55 <= dt / prev_dt <= 1.80)):
                    streak_events.append(ev)
                else:
                    _evaluate_streak(streak_events)
                    streak_events = [ev]
        else:
            # If a single tap interrupts, flush any accumulated dual-tap streak
            _evaluate_streak(streak_events)
            streak_events = []

    _evaluate_streak(streak_events)
    return records


def plan_chart_movements(
    chart: ParsedChart,
    hit_mode: str | HitMode = DEFAULT_HIT_MODE,
    skill_profile: str | SkillProfileConfig = DEFAULT_SKILL_PROFILE,
    beam_width: int = PLANNER_BEAM_WIDTH,
    lookahead_window: int = PLANNER_LOOKAHEAD_WINDOW,
    frame_fps: int = 60,
    avoid_flying_hand: bool = AVOID_FLYING_HAND_DEFAULT,
    time_budget_sec: float = 2.5,
    initial_state: Optional[PlannerState] = None,
    t_range: Optional[Tuple[float, float]] = None,
    skip_trajectory: bool = False,
    calibrate_handflow: bool = True,
) -> PlanResult:
    """Run Lookahead Beam Search / Viterbi DP over `chart.events` to solve the optimal
    bimanual movement sequence and detect all unreasonable patterns.
    """
    mode = HitMode.from_str(hit_mode)
    profile = get_skill_profile(skill_profile)
    events = chart.events

    unreasonable_records: List[UnreasonableEventRecord] = []

    # 1. Pre-check Law 1: High-BPM Dual-Tap Jackhammer Ban
    dual_jack_records = _detect_dual_jackhammer_streaks(events, profile)
    unreasonable_records.extend(dual_jack_records)

    # 1b. Inner-screen Slide Jump-Zone detection (including cross-window carried holds/slides)
    jump_check_notes: List[ParsedNote] = list(chart.notes)
    if initial_state is not None and initial_state.carry_notes:
        from sim_core.geometry import build_slide_route
        from sim_core.maidata_parser import SlideSegmentData
        for cn in initial_state.carry_notes:
            if cn["kind"] in ("HOLD", "TOUCH_HOLD"):
                jump_check_notes.append(
                    ParsedNote(
                        note_id=int(cn["note_id"]),
                        note_type=NoteType.TOUCH_HOLD if cn["kind"] == "TOUCH_HOLD" else NoteType.HOLD,
                        time=float(cn["time"]),
                        bar=int(cn["bar"]),
                        beat_in_bar=0.0,
                        bpm=chart.initial_bpm,
                        key=str(cn["key"]),
                        button=cn.get("button"),
                        sensor_zone=str(cn["sensor_zone"]),
                        duration=float(cn["end_time"]) - float(cn["time"]),
                        end_time=float(cn["end_time"]),
                        judge_time=float(cn["time"]),
                        raw_token=str(cn.get("raw_token", cn["key"])),
                    )
                )
            elif cn["kind"] == "SLIDE":
                segs_rebuilt = tuple(
                    SlideSegmentData(
                        s["shape"], int(s["start_btn"]), int(s["end_btn"]), s.get("via_btn"),
                        build_slide_route(s["shape"], int(s["start_btn"]), int(s["end_btn"]), s.get("via_btn")),
                    )
                    for s in cn.get("segments", ())
                )
                jump_check_notes.append(
                    ParsedNote(
                        note_id=int(cn["note_id"]),
                        note_type=NoteType.SLIDE,
                        time=float(cn["time"]),
                        bar=int(cn["bar"]),
                        beat_in_bar=0.0,
                        bpm=chart.initial_bpm,
                        key=str(cn["key"]),
                        button=cn.get("button"),
                        sensor_zone=str(cn["sensor_zone"]),
                        duration=float(cn["end_time"]) - float(cn["shoot_time"]),
                        wait_time=float(cn["shoot_time"]) - float(cn["time"]),
                        shoot_time=float(cn["shoot_time"]),
                        end_time=float(cn["end_time"]),
                        judge_time=float(cn["judge_time"]),
                        slide_segments=segs_rebuilt,
                        raw_token=str(cn.get("raw_token", cn["key"])),
                    )
                )
    t_win_lo = t_range[0] if t_range is not None else -1e9
    for jz_t, jz_bar, jz_keys, jz_reason, jz_sev in detect_slide_jump_zone_events(jump_check_notes):
        if jz_t >= t_win_lo - 1e-4:
            unreasonable_records.append(
                UnreasonableEventRecord(
                    time=jz_t,
                    bar=jz_bar,
                    type=UnreasonableType.SLIDE_JUMP_ZONE.value,
                    keys=jz_keys,
                    reason=jz_reason,
                    severity=jz_sev,
                )
            )

    all_slides_with_zones: List[Tuple[float, float, FrozenSet[str]]] = [
        (n.shoot_time - 0.14, n.end_time + 0.14, get_slide_all_traversed_zones(n))
        for n in jump_check_notes
        if n.note_type == NoteType.SLIDE
    ]

    # Initial resting posture: Left hand near A6/B6 (-110, -45), Right hand near A3/B3 (+110, -45), Hover z=148mm
    init_torso = TorsoState(dx_mm=0.0, dy_mm=0.0, yaw_deg=0.0)
    init_state = _BeamState(
        left_xy=(-110.0, -45.0),
        right_xy=(110.0, -45.0),
        left_z=Z_HOVER_DEFAULT_MM,
        right_z=Z_HOVER_DEFAULT_MM,
        left_tap_times=(),
        right_tap_times=(),
        left_last_key="",
        right_last_key="",
        active_holds=(),
        active_slides=(),
        torso=init_torso,
        cum_cost=0.0,
    )

    if initial_state is not None and initial_state.nodes:
        beam: List[_BeamState] = [_BeamState.from_snapshot_dict(nd) for nd in initial_state.nodes[:beam_width]]
    else:
        beam = [init_state]
    eff_beam_width = max(1, int(beam_width))
    three_hand_events_flagged: Set[int] = set()

    # Pre-compute lookahead single-key jackhammer density for each event i within `lookahead_window`
    n_events = len(events)
    is_fast_single_jack_event: List[bool] = [False] * n_events
    jack_streak_length: List[int] = [1] * n_events
    same_hemi_fast_len: List[int] = [1] * n_events
    is_unidirectional_sweep_event: List[bool] = [False] * n_events
    sweep_run_len: List[int] = [0] * n_events
    sweep_run_dir: List[int] = [0] * n_events
    sweep_target_btn: List[int] = [0] * n_events

    # Section 3: Strict Physical Conditions for `RUB_SWEEP` / `BUTTON_SWEEP` (Unidirectional Monotonic Adjacent Sweep):
    # 1) Fast continuous sweeps: dt <= 0.072 s (covering 225 BPM {24} = 66.7ms and {32}/{48} = 31..33ms).
    # 2) Chord-initiated short sweeps (e.g. `2/4, 3, 4`, `3/5, 4, 5`, `5/7, 6, 5` at 87s of Regulus [DX]):
    #    starts with a 2-note chord and steps monotonically across >= 3 distinct adjacent buttons with dt <= 0.100 s.
    # 3) Strictly monotonic direction (+1 mod 8 CW or -1 mod 8 CCW) visiting >= 3 distinct buttons with NO reversal.
    run_start = 0
    while run_start < n_events:
        st0 = events[run_start].strike_notes
        st0_btns = [n.button for n in st0 if n.button is not None and n.note_type in (NoteType.TAP, NoteType.HOLD)]
        if not st0_btns or len(st0) > 2:
            run_start += 1
            continue
        max_dt_allowed = 0.100 if len(st0_btns) == 2 else 0.072
        best_end = run_start
        best_dir = 0
        best_chain: List[int] = []
        for b0 in st0_btns:
            for cand_dir in (1, -1):
                cur_b = b0
                r_end = run_start
                chain = [b0]
                while r_end + 1 < n_events and (events[r_end + 1].bar // 4) == (events[run_start].bar // 4):
                    st_next = events[r_end + 1].strike_notes
                    if not (1 <= len(st_next) <= 2):
                        break
                    dt_step = events[r_end + 1].time - events[r_end].time
                    if dt_step > max_dt_allowed or dt_step <= 1e-4:
                        break
                    # If this step is a 2-note chord after single notes (like `2/4 -> 3 -> 4 -> 3/5`),
                    # stop the short sweep before the new chord if the new chord starts its own short sweep!
                    if len(st_next) == 2 and dt_step > 0.065 and len(chain) >= 3 and len(events[r_end].strike_notes) == 1:
                        break
                    expected_b = ((cur_b - 1 + cand_dir) % 8) + 1
                    next_btns = [n.button for n in st_next if n.button is not None and n.note_type in (NoteType.TAP, NoteType.HOLD)]
                    if expected_b not in next_btns:
                        break
                    cur_b = expected_b
                    r_end += 1
                    chain.append(cur_b)
                if len(chain) > len(best_chain):
                    best_chain = chain
                    best_end = r_end
                    best_dir = cand_dir

        run_len = len(best_chain)
        if run_len >= 3 and len(set(best_chain)) >= 3:
            for idx_k, k in enumerate(range(run_start, best_end + 1)):
                is_unidirectional_sweep_event[k] = True
                sweep_run_len[k] = run_len
                sweep_run_dir[k] = best_dir
                sweep_target_btn[k] = best_chain[idx_k]
            run_start = best_end + 1
        else:
            run_start += 1

    for i in range(n_events):
        ev = events[i]
        strikes = ev.strike_notes
        if not strikes:
            continue
        # Check each struck key in this event for a high-speed same-key longitudinal stream
        best_cnt = 1
        best_min_dt = float("inf")
        for sn_item in strikes:
            k = sn_item.key
            cnt = 1
            min_dt = float("inf")
            # Backward
            j = i - 1
            while j >= max(0, i - lookahead_window) and (events[j].bar // 4) == (ev.bar // 4):
                st_j = events[j].strike_notes
                dt_j = events[j + 1].time - events[j].time
                if any(x.key == k for x in st_j) and dt_j <= 0.16:
                    cnt += 1
                    min_dt = min(min_dt, dt_j)
                    j -= 1
                else:
                    break
            # Forward
            j = i + 1
            while j < min(n_events, i + lookahead_window) and (events[j].bar // 4) == (ev.bar // 4):
                st_j = events[j].strike_notes
                dt_j = events[j].time - events[j - 1].time
                if any(x.key == k for x in st_j) and dt_j <= 0.16:
                    cnt += 1
                    min_dt = min(min_dt, dt_j)
                    j += 1
                else:
                    break
            if cnt > best_cnt:
                best_cnt = cnt
                best_min_dt = min_dt
        jack_streak_length[i] = best_cnt
        is_midline_btn = (len(strikes) == 1 and strikes[0].button in (1, 4, 5, 8))
        if len(strikes) == 1 and (best_cnt >= 3 or (best_cnt >= 2 and is_midline_btn)) and best_min_dt < (1.0 / profile.max_single_hand_tap_hz):
            is_fast_single_jack_event[i] = True
        if len(strikes) == 1:
            xi = _resolve_note_default_target(strikes[0], mode)[1][0]
            if abs(xi) >= 15.0:
                sh_cnt = 1
                j = i - 1
                while j >= 0 and (events[j + 1].time - events[j].time) <= 0.175 and len(events[j].strike_notes) == 1:
                    xj = _resolve_note_default_target(events[j].strike_notes[0], mode)[1][0]
                    if xi * xj > 0:
                        sh_cnt += 1
                        j -= 1
                    else:
                        break
                j = i + 1
                while j < n_events and (events[j].time - events[j - 1].time) <= 0.175 and len(events[j].strike_notes) == 1:
                    xj = _resolve_note_default_target(events[j].strike_notes[0], mode)[1][0]
                    if xi * xj > 0:
                        sh_cnt += 1
                        j += 1
                    else:
                        break
                same_hemi_fast_len[i] = sh_cnt

    # 2. Step through events with Beam Search / Viterbi DP
    commit_bars = max(4, int(lookahead_window) // 3) + (2 if eff_beam_width > 6 else 0)
    sig_grid_mm = 12.0 if eff_beam_width > 6 else 25.0
    for ev_idx, ev in enumerate(events):
        t = ev.time
        # Causal 4-bar phrase horizon commitment: at 4-bar boundaries (`bar // 4` change),
        # commit to the top hypothesis so future bars 40 measures later cannot retroactively flip past hand choices!
        if ev_idx > 0 and (ev.bar // commit_bars) > (events[ev_idx - 1].bar // commit_bars) and len(beam) > 1:
            beam = [beam[0]]
        if ev_idx > 0:
            prev_t = events[ev_idx - 1].time
        elif initial_state is not None and initial_state.end_time > -100.0 and initial_state.nodes:
            prev_t = min(t - 0.005, initial_state.end_time)
        else:
            prev_t = max(0.0, t - 0.25)
        dt = max(0.005, t - prev_t)
        next_dt = (events[ev_idx + 1].time - t) if (ev_idx + 1 < n_events) else 1.0
        raw_strikes = ev.strike_notes
        if raw_strikes and all_slides_with_zones:
            filtered_strikes: List[ParsedNote] = []
            for sn_chk in raw_strikes:
                if sn_chk.note_type == NoteType.TOUCH and any(
                    t0_s <= t <= t1_s and sn_chk.sensor_zone in zset
                    for t0_s, t1_s, zset in all_slides_with_zones
                ):
                    continue
                filtered_strikes.append(sn_chk)
            strikes = filtered_strikes
        else:
            strikes = raw_strikes
        slides_starting = ev.slide_notes

        if len(beam) > eff_beam_width:
            beam = beam[:eff_beam_width]
        next_beam: List[_BeamState] = []

        for b_state in beam:
            # Update hand position and one-stroke (`一笔画`) slide history for any slide that finished before `t`
            cur_left_xy = b_state.left_xy
            cur_right_xy = b_state.right_xy
            cur_left_last_key = b_state.left_last_key
            cur_right_last_key = b_state.right_last_key
            last_sl_l = b_state.last_slide_end_l
            last_sl_r = b_state.last_slide_end_r
            for s_old in b_state.active_slides:
                if s_old[4] <= t + 0.01:  # judge_time passed -> hand arrived at slide end_xy
                    # Use `min(t, s_old[4])` (when the hand actually reached slide completion), NEVER future `s_old[3] > t`!
                    sl_done_t = min(t, s_old[4])
                    if s_old[1] in ("L", "LR") and b_state.left_last_contact[0] <= s_old[2] + 0.22:
                        cur_left_xy = s_old[6]
                        cur_left_last_key = f"A{s_old[8]}"
                        last_sl_l = (s_old[8], sl_done_t, s_old[6])
                    if s_old[1] in ("R", "LR") and b_state.right_last_contact[0] <= s_old[2] + 0.22:
                        cur_right_xy = s_old[6]
                        cur_right_last_key = f"A{s_old[8]}"
                        last_sl_r = (s_old[8], sl_done_t, s_old[6])

            # Track recently released holds for Section 5.2 (120ms Hold Release Recovery)
            cur_l_rel = b_state.left_hold_release
            cur_r_rel = b_state.right_hold_release
            for h_old in b_state.active_holds:
                if not _is_hold_locking_hand(h_old, t):
                    if h_old[1] == "L" and h_old[5] > cur_l_rel[0]:
                        cur_l_rel = (h_old[5], h_old[3])
                    elif h_old[1] == "R" and h_old[5] > cur_r_rel[0]:
                        cur_r_rel = (h_old[5], h_old[3])

            # Prune expired holds & slides at timestamp `t`
            surviving_holds = tuple(
                h for h in b_state.active_holds if _is_hold_locking_hand(h, t)
            )
            surviving_slides = tuple(
                s for s in b_state.active_slides if s[4] > t + 0.01
            )

            locked_l_holds = [h for h in surviving_holds if h[1] == "L"]
            locked_r_holds = [h for h in surviving_holds if h[1] == "R"]
            if profile.allow_slide_hand_switch and surviving_slides:
                reassigned_slides = []
                for s_item in surviving_slides:
                    if s_item[1] in ("L", "R") and abs(t - s_item[2]) <= 0.06:
                        dl_s = math.hypot(s_item[5][0] - cur_left_xy[0], s_item[5][1] - cur_left_xy[1])
                        dr_s = math.hypot(s_item[5][0] - cur_right_xy[0], s_item[5][1] - cur_right_xy[1])
                        if s_item[1] == "R" and dl_s + 65.0 < dr_s and not locked_l_holds:
                            s_item = (s_item[0], "L", *s_item[2:])
                        elif s_item[1] == "L" and dr_s + 65.0 < dl_s and not locked_r_holds:
                            s_item = (s_item[0], "R", *s_item[2:])
                    elif len(surviving_slides) == 1 and len(strikes) == 1 and s_item[1] in ("L", "R") and s_item[2] <= t <= s_item[3]:
                        s_prog = max(0.0, min(1.0, (t - s_item[2]) / max(0.01, s_item[3] - s_item[2])))
                        s_cur_xy = interpolate_waypoints(s_item[7], s_prog) if s_item[7] else s_item[5]
                        tap_xy = _resolve_note_default_target(strikes[0], mode)[1]
                        if s_cur_xy[0] < -35.0 and tap_xy[0] > 35.0 and s_item[1] == "R" and not locked_l_holds:
                            s_item = (s_item[0], "L", *s_item[2:])
                        elif s_cur_xy[0] > 35.0 and tap_xy[0] < -35.0 and s_item[1] == "L" and not locked_r_holds:
                            s_item = (s_item[0], "R", *s_item[2:])
                    reassigned_slides.append(s_item)
                surviving_slides = tuple(reassigned_slides)

            locked_l_slides = [
                s for s in surviving_slides
                if s[1] in ("L", "LR") and _is_slide_locking_hand(s, t, profile.max_arm_velocity)
            ]
            locked_r_slides = [
                s for s in surviving_slides
                if s[1] in ("R", "LR") and _is_slide_locking_hand(s, t, profile.max_arm_velocity)
            ]
            # Also check pending slides in their wait period (`t < shoot_t`):
            # If a hand has a pending slide launching at `shoot_t` at `s_xy`, check if detouring to a tap
            # and returning to `s_xy` by `shoot_t` would cause a flying-hand speed violation!
            pending_l_slides = [s for s in surviving_slides if s[1] in ("L", "LR") and s[2] > t]
            pending_r_slides = [s for s in surviving_slides if s[1] in ("R", "LR") and s[2] > t]

            l_locked = bool(locked_l_holds or locked_l_slides)
            r_locked = bool(locked_r_holds or locked_r_slides)

            # Determine locked hand positions if locked
            l_locked_xy: Optional[Point2D] = None
            l_locked_zones: Tuple[str, ...] = ()
            l_locked_mode = "Hold"
            if locked_l_holds:
                l_locked_xy = locked_l_holds[0][3]
                l_locked_zones = tuple(h[2] for h in locked_l_holds)
                l_locked_mode = "Hold"
            elif locked_l_slides:
                s0 = locked_l_slides[0]
                prog = max(0.0, min(1.0, (t - s0[2]) / max(0.01, s0[3] - s0[2])))
                l_locked_xy = interpolate_waypoints(s0[7], prog) if s0[7] else s0[5]
                l_locked_zones = ()
                l_locked_mode = "Slide"

            r_locked_xy: Optional[Point2D] = None
            r_locked_zones: Tuple[str, ...] = ()
            r_locked_mode = "Hold"
            if locked_r_holds:
                r_locked_xy = locked_r_holds[0][3]
                r_locked_zones = tuple(h[2] for h in locked_r_holds)
                r_locked_mode = "Hold"
            elif locked_r_slides:
                s0 = locked_r_slides[0]
                prog = max(0.0, min(1.0, (t - s0[2]) / max(0.01, s0[3] - s0[2])))
                r_locked_xy = interpolate_waypoints(s0[7], prog) if s0[7] else s0[5]
                r_locked_zones = ()
                r_locked_mode = "Slide"

            # Collect all simultaneous target zones required at timestamp `t` (strikes + locked holds)
            all_required_zones: List[str] = list(l_locked_zones) + list(r_locked_zones)
            for sn in strikes:
                all_required_zones.append(sn.sensor_zone)

            unique_req_zones = list(dict.fromkeys(all_required_zones))

            # ---- Law 3: Multi-Touch & Rub Feasibility (3+ simultaneous targets) ----
            if len(unique_req_zones) >= 3:
                strict_anchor_zones = list(
                    dict.fromkeys(
                        list(l_locked_zones)
                        + list(r_locked_zones)
                        + [sn.sensor_zone for sn in strikes if sn.note_type != NoteType.TOUCH]
                    )
                )
                has_touch_wave = any(sn.note_type == NoteType.TOUCH for sn in strikes)
                all_bce_touch = has_touch_wave and len(strict_anchor_zones) == 0 and all(z[0] in ("B", "C", "E") for z in unique_req_zones)
                if len(strict_anchor_zones) <= 2 and has_touch_wave and (len(unique_req_zones) >= 6 or all_bce_touch):
                    l_part = [z for z in unique_req_zones if get_sensor_zone(z).center[0] <= 0.0] or [unique_req_zones[0]]
                    r_part = [z for z in unique_req_zones if get_sensor_zone(z).center[0] > 0.0] or [unique_req_zones[-1]]
                    l_wx = sum(get_sensor_zone(z).center[0] for z in l_part) / len(l_part)
                    l_wy = sum(get_sensor_zone(z).center[1] for z in l_part) / len(l_part)
                    r_wx = sum(get_sensor_zone(z).center[0] for z in r_part) / len(r_part)
                    r_wy = sum(get_sensor_zone(z).center[1] for z in r_part) / len(r_part)
                    if math.hypot(l_wx - r_wx, l_wy - r_wy) < 90.0:
                        l_wx, r_wx = min(l_wx, -55.0), max(r_wx, 55.0)
                    cover_sol = ((l_wx, l_wy), (r_wx, r_wy), tuple(l_part), tuple(r_part))
                    all_inner_touch = True
                else:
                    all_inner_touch = all(sn.note_type in (NoteType.TOUCH, NoteType.TOUCH_HOLD) for sn in strikes) and not l_locked_zones and not r_locked_zones
                    check_zones = unique_req_zones if all_inner_touch else (strict_anchor_zones if len(strict_anchor_zones) >= 3 else unique_req_zones)
                    eff_palm_r = 74.0 if (all_inner_touch or (len(strict_anchor_zones) <= 2 and has_touch_wave)) else (35.0 if mode == HitMode.OUTER else R_HAND_MM)
                    cover_sol = solve_two_palm_coverage(check_zones, eff_palm_r)
                if cover_sol is None:
                    if ev.event_id not in three_hand_events_flagged:
                        three_hand_events_flagged.add(ev.event_id)
                        unreasonable_records.append(
                            UnreasonableEventRecord(
                                time=t,
                                bar=ev.bar,
                                type=UnreasonableType.THREE_HAND_CONFLICT.value,
                                keys=[sn.key for sn in strikes] + list(l_locked_zones) + list(r_locked_zones),
                                reason=(
                                    f"Multi-touch conflict: {len(unique_req_zones)} simultaneous targets "
                                    f"{unique_req_zones} cannot be covered by two {R_HAND_MM:.0f}mm palms"
                                ),
                                severity=2.5,
                                event_id=ev.event_id,
                                raw_m=float(len(unique_req_zones)),
                            )
                        )
                else:
                    # Feasible 2-palm rub! Generate candidate branch using the solved rub centers
                    l_rub_xy, r_rub_xy, l_cov, r_cov = cover_sol
                    l_ids = tuple(sn.note_id for sn in strikes if sn.sensor_zone in l_cov or (sn.sensor_zone not in r_cov and get_sensor_zone(sn.sensor_zone).center[0] <= 0.0))
                    r_ids = tuple(sn.note_id for sn in strikes if sn.note_id not in l_ids)
                    l_hz, l_strk, l_times = (
                            (0.0, 1, b_state.left_tap_times) if all_inner_touch else _compute_hand_tap_metrics(b_state.left_tap_times, t)
                        if l_ids else (0.0, 0, b_state.left_tap_times)
                    )
                    r_hz, r_strk, r_times = (
                            (0.0, 1, b_state.right_tap_times) if all_inner_touch else _compute_hand_tap_metrics(b_state.right_tap_times, t)
                        if r_ids else (0.0, 0, b_state.right_tap_times)
                    )
                    l_assign = HandAssignment("L", "Rub", l_rub_xy, Z_CONTACT_MM, l_cov, l_ids, l_hz, l_strk)
                    r_assign = HandAssignment("R", "Rub", r_rub_xy, Z_CONTACT_MM, r_cov, r_ids, r_hz, r_strk)
                    body = solve_coupled_body_ik(
                        t, (l_rub_xy[0], l_rub_xy[1], Z_CONTACT_MM), (r_rub_xy[0], r_rub_xy[1], Z_CONTACT_MM),
                        b_state.torso, dt, profile,
                        left_is_holding=bool(locked_l_holds),
                        right_is_holding=bool(locked_r_holds),
                    )
                    nh_map = {sn.note_id: ("L" if sn.note_id in l_ids else "R", sn.sensor_zone) for sn in strikes}
                    new_holds = list(surviving_holds)
                    for sn in strikes:
                        if sn.note_type in (NoteType.HOLD, NoteType.TOUCH_HOLD):
                            h_side = "L" if sn.note_id in l_ids else "R"
                            h_xy = l_rub_xy if h_side == "L" else r_rub_xy
                            new_holds.append((sn.note_id, h_side, sn.sensor_zone, h_xy, sn.time, sn.end_time, sn.note_type == NoteType.TOUCH_HOLD))
                    new_slides = list(surviving_slides)
                    for sl in slides_starting:
                        sl_side = "L" if sl.button in (5, 6, 7, 8) else "R"
                        s_xy = sl.slide_waypoints_mm[0] if sl.slide_waypoints_mm else (0.0, 0.0)
                        e_xy = sl.slide_waypoints_mm[-1] if sl.slide_waypoints_mm else (0.0, 0.0)
                        ebtn = sl.slide_segments[-1].end_btn if sl.slide_segments else (sl.button or 1)
                        new_slides.append((sl.note_id, sl_side, sl.shoot_time, sl.end_time, sl.judge_time, s_xy, e_xy, sl.slide_waypoints_mm, ebtn))
                        nh_map[sl.note_id] = (sl_side, sl.sensor_zone)

                    next_beam.append(
                        _BeamState(
                            left_xy=l_rub_xy,
                            right_xy=r_rub_xy,
                            left_z=Z_CONTACT_MM,
                            right_z=Z_CONTACT_MM,
                            left_tap_times=l_times,
                            right_tap_times=r_times,
                            left_last_key=l_cov[0] if l_cov else b_state.left_last_key,
                            right_last_key=r_cov[0] if r_cov else b_state.right_last_key,
                            active_holds=tuple(new_holds),
                            active_slides=tuple(new_slides),
                            torso=body.torso,
                            cum_cost=b_state.cum_cost + body.posture_cost + 15.0,
                            left_hold_release=cur_l_rel,
                            right_hold_release=cur_r_rel,
                            last_slide_end_l=last_sl_l,
                            last_slide_end_r=last_sl_r,
                            left_last_contact=(t, l_rub_xy) if l_ids else b_state.left_last_contact,
                            right_last_contact=(t, r_rub_xy) if r_ids else b_state.right_last_contact,
                            history_node=(b_state.history_node, (l_assign, r_assign, body, "SINGLE_PALM_RUB", body.posture_cost, nh_map)),
                            forced_single_jack_flags=b_state.forced_single_jack_flags,
                            forced_single_jump_flags=b_state.forced_single_jump_flags,
                            collision_flags=b_state.collision_flags,
                            cross_flags=b_state.cross_flags,
                            flying_hand_flags=b_state.flying_hand_flags,
                        )
                    )
                    continue

            # ---- Standard 0, 1, or 2 Strike Notes at Event `ev` ----
            candidate_options: List[Tuple[HandAssignment, HandAssignment, str, Dict[int, Tuple[str, str]]]] = []

            if len(strikes) == 0:
                # Headless slide only or empty event
                l_xy = l_locked_xy if l_locked_xy is not None else b_state.left_xy
                r_xy = r_locked_xy if r_locked_xy is not None else b_state.right_xy
                l_z = Z_CONTACT_MM if l_locked else Z_HOVER_DEFAULT_MM
                r_z = Z_CONTACT_MM if r_locked else Z_HOVER_DEFAULT_MM
                l_ass = HandAssignment("L", l_locked_mode if l_locked else "Hover", l_xy, l_z, l_locked_zones, (), 0.0, 0)
                r_ass = HandAssignment("R", r_locked_mode if r_locked else "Hover", r_xy, r_z, r_locked_zones, (), 0.0, 0)
                candidate_options.append((l_ass, r_ass, "NONE", {}))

            elif len(strikes) == 1:
                sn = strikes[0]
                def_zone, def_xy, def_mode = _resolve_note_default_target(sn, mode)
                is_fast_jack = is_fast_single_jack_event[ev_idx]

                # Candidate hand choices: 'L' or 'R' (unless locked on a different zone)
                allowed_hands: List[str] = []
                l_same_hand_hold_span = bool(
                    locked_l_holds
                    and l_locked_xy is not None
                    and sn.note_type == NoteType.TOUCH
                    and math.hypot(def_xy[0] - l_locked_xy[0], def_xy[1] - l_locked_xy[1]) <= SAME_HAND_HOLD_SPAN_MM
                )
                r_same_hand_hold_span = bool(
                    locked_r_holds
                    and r_locked_xy is not None
                    and sn.note_type == NoteType.TOUCH
                    and math.hypot(def_xy[0] - r_locked_xy[0], def_xy[1] - r_locked_xy[1]) <= SAME_HAND_HOLD_SPAN_MM
                )

                if (
                    not l_locked
                    or (l_locked_zones and sn.sensor_zone in l_locked_zones)
                    or l_same_hand_hold_span
                    or (locked_l_slides and not locked_l_holds and l_locked_xy is not None and math.hypot(def_xy[0] - l_locked_xy[0], def_xy[1] - l_locked_xy[1]) <= (108.0 if abs(t - locked_l_slides[0][2]) <= 0.04 else 90.0))
                ):
                    allowed_hands.append("L")
                if (
                    not r_locked
                    or (r_locked_zones and sn.sensor_zone in r_locked_zones)
                    or r_same_hand_hold_span
                    or (locked_r_slides and not locked_r_holds and r_locked_xy is not None and math.hypot(def_xy[0] - r_locked_xy[0], def_xy[1] - r_locked_xy[1]) <= (108.0 if abs(t - locked_r_slides[0][2]) <= 0.04 else 90.0))
                ):
                    allowed_hands.append("R")
                if not allowed_hands:
                    dl = math.hypot(def_xy[0] - b_state.left_xy[0], def_xy[1] - b_state.left_xy[1])
                    dr = math.hypot(def_xy[0] - b_state.right_xy[0], def_xy[1] - b_state.right_xy[1])
                    allowed_hands.append("L" if dl <= dr else "R")

                for h_choice in allowed_hands:
                    # Check if we should use Law 2: Single Jackhammer Decomposition
                    # (1) Outer-Inner Split in hybrid mode when both hands are free
                    # (2) Alternating Tap with micro-offset + z-rebound separation
                    both_hands_free = (not l_locked) and (not r_locked)
                    decomp_tag = "NONE"
                    target_zone = def_zone
                    target_xy = def_xy
                    mode_label = def_mode
                    other_xy = (
                        (r_locked_xy if r_locked_xy is not None else cur_right_xy)
                        if h_choice == "L"
                        else (l_locked_xy if l_locked_xy is not None else cur_left_xy)
                    )
                    other_z = Z_CONTACT_MM if (r_locked if h_choice == "L" else l_locked) else Z_HOVER_DEFAULT_MM
                    other_mode = (r_locked_mode if r_locked else "Hover") if h_choice == "L" else (l_locked_mode if l_locked else "Hover")
                    using_hold_span = (h_choice == "L" and l_same_hand_hold_span) or (h_choice == "R" and r_same_hand_hold_span)
                    if using_hold_span:
                        h_anchor_xy = l_locked_xy if h_choice == "L" else r_locked_xy
                        assert h_anchor_xy is not None
                        target_xy = (0.55 * h_anchor_xy[0] + 0.45 * def_xy[0], 0.55 * h_anchor_xy[1] + 0.45 * def_xy[1])
                        mode_label = "Hold"
                        decomp_tag = "SINGLE_PALM_RUB"
                    # If the other hand is locked on a Slide/Hold at `A{btn}` (`other_z == 0`) and this hand strikes the same button,
                    # use Outer-Inner Split (`OB{btn}`) in hybrid mode so the striking palm never collides with the sliding palm!
                    if not using_hold_span and other_z <= Z_RELEASE_MM and sn.button is not None and mode == HitMode.HYBRID:
                        if math.hypot(target_xy[0] - other_xy[0], target_xy[1] - other_xy[1]) < HAND_COLLISION_MIN_DIST_MM:
                            target_zone = f"OB{sn.button}"
                            target_xy = get_button_target_xy(sn.button, use_outer=True)
                            mode_label = "OuterSplit"
                            decomp_tag = "OUTER_INNER_SPLIT"
                    elif not using_hold_span and other_z <= Z_RELEASE_MM and (locked_r_holds if h_choice == "L" else locked_l_holds):
                        dx_sep = other_xy[0] - target_xy[0]
                        dy_sep = other_xy[1] - target_xy[1]
                        d_sep = math.hypot(dx_sep, dy_sep)
                        if 1.0 < d_sep < 90.0:
                            other_xy = (other_xy[0] + 22.0 * dx_sep / d_sep, other_xy[1] + 22.0 * dy_sep / d_sep)
                    # If the non-striking hand is hovering (not locked) and too close to target_xy,
                    # retract it smoothly toward its natural home side so it never blocks the striking hand
                    if other_z > Z_RELEASE_MM and math.hypot(target_xy[0] - other_xy[0], target_xy[1] - other_xy[1]) < HAND_COLLISION_MIN_DIST_MM + 15.0:
                        home_xy = (115.0, -45.0) if h_choice == "L" else (-115.0, -45.0)
                        other_xy = (
                            0.45 * other_xy[0] + 0.55 * home_xy[0],
                            0.45 * other_xy[1] + 0.55 * home_xy[1],
                        )

                    last_l_t_for_jack = b_state.left_tap_times[-1] if b_state.left_tap_times else -999.0
                    last_r_t_for_jack = b_state.right_tap_times[-1] if b_state.right_tap_times else -999.0
                    prev_jack_hand = "L" if last_l_t_for_jack > last_r_t_for_jack + 1e-4 else ("R" if last_r_t_for_jack > last_l_t_for_jack + 1e-4 else "")
                    prev_jack_key = b_state.left_last_key if prev_jack_hand == "L" else (b_state.right_last_key if prev_jack_hand == "R" else "")
                    same_as_prev_jack = (
                        sn.button is not None
                        and prev_jack_key != ""
                        and prev_jack_key[-1:].isdigit()
                        and int(prev_jack_key[-1:]) == sn.button
                    )
                    if is_fast_jack and both_hands_free and sn.button is not None and (not same_as_prev_jack or h_choice != prev_jack_hand):
                        btn = sn.button
                        if mode == HitMode.HYBRID and profile.allow_inner_outer_split:
                            # Outer-Inner Split:
                            # To keep left and right forearms parallel without crossing:
                            # - On right side (buttons 1..4): Left hand takes Outer OB{btn} (310mm), Right hand takes Inner A{btn} (215mm)
                            # - On left side (buttons 5..8): Right hand takes Outer OB{btn} (310mm), Left hand takes Inner A{btn} (215mm)
                            inner_xy = get_button_target_xy(btn, use_outer=False)
                            outer_xy = get_button_target_xy(btn, use_outer=True)
                            l_uses_outer = btn in (1, 2, 3, 4)
                            if h_choice == "L":
                                target_zone = f"OB{btn}" if l_uses_outer else f"A{btn}"
                                target_xy = outer_xy if l_uses_outer else inner_xy
                                mode_label = "OuterSplit" if l_uses_outer else "InnerSplit"
                                other_xy = inner_xy if l_uses_outer else outer_xy
                                other_z = Z_TAP_REBOUND_MM
                                other_mode = "InnerSplit" if l_uses_outer else "OuterSplit"
                            else:
                                target_zone = f"A{btn}" if l_uses_outer else f"OB{btn}"
                                target_xy = inner_xy if l_uses_outer else outer_xy
                                mode_label = "InnerSplit" if l_uses_outer else "OuterSplit"
                                other_xy = outer_xy if l_uses_outer else inner_xy
                                other_z = Z_TAP_REBOUND_MM
                                other_mode = "OuterSplit" if l_uses_outer else "InnerSplit"
                            decomp_tag = "OUTER_INNER_SPLIT"
                        elif mode != HitMode.INNER:
                            # Alternating Tap with tangential edge offset + Z-phase separation
                            ang_rad = math.radians(TOUCH_ZONES[f"A{btn}"].angle_cw_deg)
                            tx, ty = math.cos(ang_rad), -math.sin(ang_rad)
                            offset_mm = -22.0 if h_choice == "L" else 22.0
                            target_xy = (def_xy[0] + offset_mm * tx, def_xy[1] + offset_mm * ty)
                            other_xy = (def_xy[0] - offset_mm * tx, def_xy[1] - offset_mm * ty)
                            other_z = Z_TAP_REBOUND_MM + 12.0
                            mode_label = "AltTap"
                            other_mode = "AltTap"
                            decomp_tag = "ALTERNATING_TAP"

                    prev_key = cur_left_last_key if h_choice == "L" else cur_right_last_key
                    is_touch_wipe = (
                        sn.note_type == NoteType.TOUCH
                        and prev_key != ""
                        and prev_key != target_zone
                        and (not prev_key.startswith("OB") or dt <= 0.085)
                    )
                    if is_touch_wipe:
                        decomp_tag = "SINGLE_PALM_RUB"
                        if both_hands_free and other_mode == "Hover" and dt <= 0.085:
                            # Bimanual parallel push ("大手一推"): keep the companion palm gliding smoothly alongside!
                            other_xy = (
                                0.65 * other_xy[0] + 0.35 * (85.0 if h_choice == "L" else -85.0),
                                0.45 * other_xy[1] + 0.55 * target_xy[1],
                            )
                            other_z = Z_CONTACT_MM
                            other_mode = "Rub"
                    # Check if this is a button Sweep (扫键: consecutive adjacent buttons along the ring)
                    is_button_sweep = False
                    prev_taps = b_state.left_tap_times if h_choice == "L" else b_state.right_tap_times
                    use_outer_sweep = (mode == HitMode.OUTER)
                    next_is_15_chord_here = (
                        ev_idx + 1 < n_events
                        and {n.button for n in events[ev_idx + 1].strike_notes} == {1, 5}
                    )
                    allowed_hemi_sweep = True
                    other_cur_xy = cur_right_xy if h_choice == "L" else cur_left_xy
                    this_cur_xy = cur_left_xy if h_choice == "L" else cur_right_xy
                    other_would_fly = (
                        dt <= 0.080
                        and math.hypot(def_xy[0] - other_cur_xy[0], def_xy[1] - other_cur_xy[1]) > 245.0
                        and math.hypot(def_xy[0] - this_cur_xy[0], def_xy[1] - this_cur_xy[1]) <= 195.0
                    )
                    if both_hands_free and sweep_run_len[ev_idx] > 8 and sn.button is not None and not other_would_fly:
                        if h_choice == "L":
                            allowed_hemi_sweep = (sn.button in (5, 6, 7, 8)) or (sn.button == 4 and next_is_15_chord_here)
                        else:
                            allowed_hemi_sweep = (sn.button in (1, 2, 3)) or (sn.button == 4 and not next_is_15_chord_here)
                    if (
                        is_unidirectional_sweep_event[ev_idx]
                        and sn.note_type in (NoteType.TAP, NoteType.HOLD)
                        and sn.button is not None
                        and allowed_hemi_sweep
                    ):
                        prev_btn = int(prev_key[-1]) if (prev_key != "" and prev_key[-1].isdigit()) else -99
                        if prev_btn in range(1, 9) and prev_taps and (t - prev_taps[-1]) <= 0.102:
                            btn_diff = min((sn.button - prev_btn) % 8, (prev_btn - sn.button) % 8)
                            if btn_diff == 1:
                                is_button_sweep = True
                        elif ev_idx + 1 < n_events and is_unidirectional_sweep_event[ev_idx + 1]:
                            # Initiating or taking over a hemisphere sweep at the entry button
                            if (h_choice == "R" and sn.button in (1, 2, 3, 4)) or (h_choice == "L" and sn.button in (4, 5, 6, 7, 8)):
                                is_button_sweep = True
                        if is_button_sweep:
                            decomp_tag = "RUB_SWEEP"
                            target_zone = f"OB{sn.button}" if use_outer_sweep else f"A{sn.button}"
                            target_xy = get_button_target_xy(sn.button, use_outer=use_outer_sweep)
                            # Pre-position the idle hand smoothly toward the upcoming hemisphere relay / chord anchor button
                            if both_hands_free and other_mode == "Hover":
                                s_dir = sweep_run_dir[ev_idx]
                                prep_btn = (5 if s_dir >= 0 else 8) if h_choice == "R" else (1 if s_dir >= 0 else 4)
                                # Check if a 2-note chord appears within the next 4 events
                                for look_k in range(ev_idx + 1, min(n_events, ev_idx + 5)):
                                    st_lk = events[look_k].strike_notes
                                    if len(st_lk) == 2:
                                        sw_b = sweep_target_btn[look_k]
                                        chord_set = {n.button for n in st_lk if n.button is not None}
                                        if h_choice == "R" and chord_set == {1, 5} and sw_b == 5:
                                            # Left hand must take over at button 4 one step before `1/5` so Right hand can move 3 -> 1!
                                            prep_btn = 4
                                        elif h_choice == "R":
                                            l_cands = [b for b in chord_set if b in (5, 6, 7, 8)]
                                            if l_cands:
                                                prep_btn = l_cands[0]
                                        else:
                                            r_cands = [b for b in chord_set if b in (1, 2, 3, 4)]
                                            if r_cands:
                                                prep_btn = r_cands[0]
                                        break
                                prep_xy = get_button_target_xy(prep_btn, use_outer=use_outer_sweep)
                                other_xy = (
                                    0.25 * other_xy[0] + 0.75 * prep_xy[0],
                                    0.25 * other_xy[1] + 0.75 * prep_xy[1],
                                )

                    is_touch_to_btn_wipe = (
                        not is_touch_wipe
                        and not is_button_sweep
                        and prev_key != ""
                        and prev_key[0] in ("B", "C", "D", "E")
                        and sn.button is not None
                        and mode != HitMode.OUTER
                        and math.hypot(target_xy[0] - this_cur_xy[0], target_xy[1] - this_cur_xy[1]) <= 135.0
                        and dt <= 0.095
                    )
                    is_continuous_stroke = is_touch_wipe or is_button_sweep or using_hold_span or is_touch_to_btn_wipe
                    stroke_label = (
                        "Sweep"
                        if is_button_sweep
                        else ("Hold" if (sn.note_type in (NoteType.HOLD, NoteType.TOUCH_HOLD) or using_hold_span) else ("Rub" if (is_touch_wipe or is_touch_to_btn_wipe) else mode_label))
                    )
                    if h_choice == "L":
                        inst_hz, strk, _ = (0.0, 1, b_state.left_tap_times) if is_continuous_stroke else _compute_hand_tap_metrics(b_state.left_tap_times, t)
                        l_ass = HandAssignment(
                            "L",
                            stroke_label,
                            target_xy,
                            Z_CONTACT_MM,
                            (target_zone,),
                            (sn.note_id,),
                            inst_hz,
                            strk,
                        )
                        r_ass = HandAssignment(
                            "R",
                            other_mode,
                            other_xy,
                            other_z,
                            r_locked_zones,
                            (),
                            0.0,
                            0,
                        )
                    else:
                        inst_hz, strk, _ = (0.0, 1, b_state.right_tap_times) if is_continuous_stroke else _compute_hand_tap_metrics(b_state.right_tap_times, t)
                        r_ass = HandAssignment(
                            "R",
                            stroke_label,
                            target_xy,
                            Z_CONTACT_MM,
                            (target_zone,),
                            (sn.note_id,),
                            inst_hz,
                            strk,
                        )
                        l_ass = HandAssignment(
                            "L",
                            other_mode,
                            other_xy,
                            other_z,
                            l_locked_zones,
                            (),
                            0.0,
                            0,
                        )
                    candidate_options.append(
                        (l_ass, r_ass, decomp_tag, {sn.note_id: (h_choice, target_zone)})
                    )

            else:
                # 2 simultaneous strike notes (or 2 groups)
                n0, n1 = strikes[0], strikes[1]
                z0, xy0, m0 = _resolve_note_default_target(n0, mode)
                z1, xy1, m1 = _resolve_note_default_target(n1, mode)

                # Section 2: Single-Hand Single-Button Mutual Exclusion!
                # NEVER allow a single palm to rub two button-indexed Tap/Hold notes (1~8).
                # For 2-note Touch pairs, only collapse onto 1 palm if the other hand is locked or both sensors are on the same side;
                # when both hands are free and wiping a symmetric Touch pile (e.g. B1+B8), keep L on B8 and R on B1 ("双手平推")!
                same_side_touch_pair = (xy0[0] * xy1[0] > -100.0) or l_locked or r_locked
                if n0.note_type == NoteType.TOUCH and n1.note_type == NoteType.TOUCH and same_side_touch_pair:
                    if math.hypot(xy0[0] - xy1[0], xy0[1] - xy1[1]) < HAND_COLLISION_MIN_DIST_MM + 15.0:
                        rub_center = find_single_palm_cover_center([n0.sensor_zone, n1.sensor_zone], 74.0)
                        if rub_center is not None:
                            for h_rub in ("L", "R"):
                                if (h_rub == "L" and l_locked) or (h_rub == "R" and r_locked):
                                    continue
                                nh_rub = {n0.note_id: (h_rub, n0.sensor_zone), n1.note_id: (h_rub, n1.sensor_zone)}
                                if h_rub == "L":
                                    l_hz, l_strk, _ = _compute_hand_tap_metrics(b_state.left_tap_times, t)
                                    r_idle_xy = r_locked_xy if r_locked_xy is not None else (115.0, -45.0)
                                    r_idle_z = Z_CONTACT_MM if r_locked else Z_HOVER_DEFAULT_MM
                                    l_ass = HandAssignment("L", "Rub", rub_center, Z_CONTACT_MM, (n0.sensor_zone, n1.sensor_zone), (n0.note_id, n1.note_id), l_hz, l_strk)
                                    r_ass = HandAssignment("R", r_locked_mode if r_locked else "Hover", r_idle_xy, r_idle_z, r_locked_zones, (), 0.0, 0)
                                else:
                                    r_hz, r_strk, _ = _compute_hand_tap_metrics(b_state.right_tap_times, t)
                                    l_idle_xy = l_locked_xy if l_locked_xy is not None else (-115.0, -45.0)
                                    l_idle_z = Z_CONTACT_MM if l_locked else Z_HOVER_DEFAULT_MM
                                    r_ass = HandAssignment("R", "Rub", rub_center, Z_CONTACT_MM, (n0.sensor_zone, n1.sensor_zone), (n0.note_id, n1.note_id), r_hz, r_strk)
                                    l_ass = HandAssignment("L", l_locked_mode if l_locked else "Hover", l_idle_xy, l_idle_z, l_locked_zones, (), 0.0, 0)
                                candidate_options.append((l_ass, r_ass, "SINGLE_PALM_RUB", nh_rub))

                pairings = [
                    (n0, z0, xy0, m0, n1, z1, xy1, m1),
                    (n1, z1, xy1, m1, n0, z0, xy0, m0),
                ]
                for l_note, lz_name, l_xy, l_m, r_note, rz_name, r_xy, r_m in pairings:
                    decomp_tag = "NONE"
                    if l_note.button is not None and l_note.button == r_note.button and mode == HitMode.HYBRID:
                        btn = l_note.button
                        lz_name, l_xy, l_m = f"A{btn}", get_button_target_xy(btn, use_outer=False), "InnerSplit"
                        rz_name, r_xy, r_m = f"OB{btn}", get_button_target_xy(btn, use_outer=True), "OuterSplit"
                        decomp_tag = "OUTER_INNER_SPLIT"

                    l_is_sweep = False
                    l_is_touch_rub = (l_note.note_type == NoteType.TOUCH and b_state.left_tap_times and (t - b_state.left_tap_times[-1]) <= 0.120)
                    if l_note.button is not None and cur_left_last_key and b_state.left_tap_times and (t - b_state.left_tap_times[-1]) <= 0.102:
                        pb_l = int(cur_left_last_key[-1]) if cur_left_last_key[-1].isdigit() else -99
                        if pb_l in range(1, 9) and min((l_note.button - pb_l) % 8, (pb_l - l_note.button) % 8) == 1 and (
                            l_note.button in (5, 6, 7, 8) or l_note.button in (1, 4) or sweep_run_len[ev_idx] <= 4
                        ):
                            l_is_sweep = True
                            decomp_tag = "RUB_SWEEP"
                    elif is_unidirectional_sweep_event[ev_idx] and sweep_run_len[ev_idx] <= 4 and l_note.button == sweep_target_btn[ev_idx] and l_note.button in (5, 6, 7, 8):
                        l_is_sweep = True
                        decomp_tag = "RUB_SWEEP"
                    r_is_sweep = False
                    r_is_touch_rub = (r_note.note_type == NoteType.TOUCH and b_state.right_tap_times and (t - b_state.right_tap_times[-1]) <= 0.120)
                    if r_note.button is not None and cur_right_last_key and b_state.right_tap_times and (t - b_state.right_tap_times[-1]) <= 0.102:
                        pb_r = int(cur_right_last_key[-1]) if cur_right_last_key[-1].isdigit() else -99
                        if pb_r in range(1, 9) and min((r_note.button - pb_r) % 8, (pb_r - r_note.button) % 8) == 1 and (
                            r_note.button in (1, 2, 3, 4) or r_note.button in (5, 8) or sweep_run_len[ev_idx] <= 4
                        ):
                            r_is_sweep = True
                            decomp_tag = "RUB_SWEEP"
                    elif is_unidirectional_sweep_event[ev_idx] and sweep_run_len[ev_idx] <= 4 and r_note.button == sweep_target_btn[ev_idx] and r_note.button in (1, 2, 3, 4):
                        r_is_sweep = True
                        decomp_tag = "RUB_SWEEP"
                    if l_is_touch_rub or r_is_touch_rub:
                        decomp_tag = "SINGLE_PALM_RUB"

                    l_hz, l_strk, _ = (0.0, 1, b_state.left_tap_times) if (l_is_sweep or l_is_touch_rub) else _compute_hand_tap_metrics(b_state.left_tap_times, t)
                    r_hz, r_strk, _ = (0.0, 1, b_state.right_tap_times) if (r_is_sweep or r_is_touch_rub) else _compute_hand_tap_metrics(b_state.right_tap_times, t)
                    l_ass = HandAssignment(
                        "L",
                        "Sweep" if l_is_sweep else ("Rub" if l_is_touch_rub else ("Hold" if l_note.note_type in (NoteType.HOLD, NoteType.TOUCH_HOLD) else l_m)),
                        l_xy,
                        Z_CONTACT_MM,
                        (lz_name,),
                        (l_note.note_id,),
                        l_hz,
                        l_strk,
                    )
                    r_ass = HandAssignment(
                        "R",
                        "Sweep" if r_is_sweep else ("Rub" if r_is_touch_rub else ("Hold" if r_note.note_type in (NoteType.HOLD, NoteType.TOUCH_HOLD) else r_m)),
                        r_xy,
                        Z_CONTACT_MM,
                        (rz_name,),
                        (r_note.note_id,),
                        r_hz,
                        r_strk,
                    )
                    nh_map = {
                        l_note.note_id: ("L", lz_name),
                        r_note.note_id: ("R", rz_name),
                    }
                    candidate_options.append((l_ass, r_ass, decomp_tag, nh_map))

            # Evaluate all candidate transitions from `b_state`
            for l_ass, r_ass, decomp_tag, nh_map in candidate_options:
                l_xyz: Point3D = (l_ass.target_xy[0], l_ass.target_xy[1], l_ass.target_z_mm)
                r_xyz: Point3D = (r_ass.target_xy[0], r_ass.target_xy[1], r_ass.target_z_mm)
                l_is_holding_now = bool(locked_l_holds) or l_ass.mode == "Hold"
                r_is_holding_now = bool(locked_r_holds) or r_ass.mode == "Hold"

                ik_s = solve_beam_ik_step_njit(
                    float(l_xyz[0]), float(l_xyz[1]), float(l_xyz[2]),
                    float(r_xyz[0]), float(r_xyz[1]), float(r_xyz[2]),
                    True,
                    b_state.torso.dx_mm, b_state.torso.dy_mm, b_state.torso.yaw_deg, b_state.torso.yaw_vel_deg_s,
                    float(dt), float(profile.torso_response_rate), float(profile.max_torso_angular_accel),
                    l_is_holding_now, r_is_holding_now,
                )
                cand_torso = TorsoState(
                    dx_mm=ik_s[0], dy_mm=ik_s[1], yaw_deg=ik_s[2],
                    vx_mm_s=ik_s[3], vy_mm_s=ik_s[4], yaw_vel_deg_s=ik_s[5], yaw_accel_deg_s2=ik_s[6],
                )
                b_posture_cost = ik_s[7]
                b_dist_xy = ik_s[8]
                b_z_diff = ik_s[9]
                b_forearm_3d = ik_s[10]
                b_flags = ik_s[11]
                b_is_coll = bool(b_flags & 1)
                _palm_coll = bool(b_flags & 2)
                b_is_ext_cross = bool(b_flags & 4)

                dist_l = math.hypot(l_ass.target_xy[0] - cur_left_xy[0], l_ass.target_xy[1] - cur_left_xy[1])
                dist_r = math.hypot(r_ass.target_xy[0] - cur_right_xy[0], r_ass.target_xy[1] - cur_right_xy[1])
                vel_l = dist_l / dt
                vel_r = dist_r / dt
                l_ref_t, l_ref_xy = b_state.left_last_contact
                l_from_hold_tail = False
                if cur_l_rel[0] > l_ref_t:
                    l_ref_t, l_ref_xy = cur_l_rel
                    l_from_hold_tail = True
                if last_sl_l[1] > l_ref_t:
                    l_ref_t, l_ref_xy = last_sl_l[1], last_sl_l[2]
                    l_from_hold_tail = False
                r_ref_t, r_ref_xy = b_state.right_last_contact
                r_from_hold_tail = False
                if cur_r_rel[0] > r_ref_t:
                    r_ref_t, r_ref_xy = cur_r_rel
                    r_from_hold_tail = True
                if last_sl_r[1] > r_ref_t:
                    r_ref_t, r_ref_xy = last_sl_r[1], last_sl_r[2]
                    r_from_hold_tail = False

                if l_ref_t > -100.0:
                    dt_hand_l = max(0.145 if l_from_hold_tail else 0.020, t - l_ref_t + (0.145 if l_from_hold_tail else 0.0))
                    dist_hand_l = math.hypot(l_ass.target_xy[0] - l_ref_xy[0], l_ass.target_xy[1] - l_ref_xy[1])
                else:
                    dt_hand_l = max(0.080, dt)
                    dist_hand_l = dist_l
                if r_ref_t > -100.0:
                    dt_hand_r = max(0.145 if r_from_hold_tail else 0.020, t - r_ref_t + (0.145 if r_from_hold_tail else 0.0))
                    dist_hand_r = math.hypot(r_ass.target_xy[0] - r_ref_xy[0], r_ass.target_xy[1] - r_ref_xy[1])
                else:
                    dt_hand_r = max(0.080, dt)
                    dist_hand_r = dist_r

                l_is_pure_touch = bool(l_ass.note_ids) and all(sn.note_type == NoteType.TOUCH for sn in strikes if sn.note_id in l_ass.note_ids)
                r_is_pure_touch = bool(r_ass.note_ids) and all(sn.note_type == NoteType.TOUCH for sn in strikes if sn.note_id in r_ass.note_ids)
                eff_dt_l = dt_hand_l + (0.045 if l_is_pure_touch else 0.0)
                eff_dt_r = dt_hand_r + (0.045 if r_is_pure_touch else 0.0)
                eff_vel_l = dist_hand_l / eff_dt_l
                eff_vel_r = dist_hand_r / eff_dt_r

                move_cost = 0.08 * (dist_l + dist_r)
                for ps in pending_l_slides:
                    if ps[1] == "L" and l_ass.note_ids:
                        d_back = math.hypot(ps[5][0] - l_ass.target_xy[0], ps[5][1] - l_ass.target_xy[1])
                        v_back = d_back / max(0.035, ps[2] - t)
                        if d_back > 110.0 and v_back > profile.max_arm_velocity * 0.80:
                            move_cost += 650.0 + (v_back - profile.max_arm_velocity * 0.80) * 0.45
                for ps in pending_r_slides:
                    if ps[1] == "R" and r_ass.note_ids:
                        d_back = math.hypot(ps[5][0] - r_ass.target_xy[0], ps[5][1] - r_ass.target_xy[1])
                        v_back = d_back / max(0.035, ps[2] - t)
                        if d_back > 110.0 and v_back > profile.max_arm_velocity * 0.80:
                            move_cost += 650.0 + (v_back - profile.max_arm_velocity * 0.80) * 0.45
                # Anti-Flying-Hand Penalty ("防飞手硬约束"):
                # Whenever a striking hand jumps >200mm at a speed exceeding 80% of max_arm_velocity,
                # apply a heavy flying-hand penalty so the planner strongly prefers torso rotation / cross-hand
                # ("反手") or one-stroke slide continuity over teleporting hands across the screen!
                fly_thresh = profile.max_arm_velocity * (0.76 if avoid_flying_hand else 0.95)
                new_fly_flags = list(b_state.flying_hand_flags)
                if l_ass.note_ids and l_ass.mode not in ("Rub", "Sweep") and dist_hand_l > 210.0 and eff_vel_l > fly_thresh:
                    move_cost += (3600.0 if avoid_flying_hand else 1200.0) + (eff_vel_l - fly_thresh) * 2.4
                    if dist_hand_l > 260.0 and eff_vel_l > profile.max_arm_velocity * 1.18:
                        new_fly_flags.append(
                            (
                                t,
                                ev.bar,
                                list(l_ass.zone_names),
                                f"Forced flying hand (L): jump {dist_hand_l:.0f}mm in {eff_dt_l*1000:.0f}ms ({eff_vel_l:.0f} mm/s > limit {profile.max_arm_velocity:.0f} mm/s)",
                                eff_vel_l / profile.max_arm_velocity,
                            )
                        )
                elif vel_l > profile.max_arm_velocity:
                    move_cost += (vel_l - profile.max_arm_velocity) * 0.45
                if r_ass.note_ids and r_ass.mode not in ("Rub", "Sweep") and dist_hand_r > 210.0 and eff_vel_r > fly_thresh:
                    move_cost += (3600.0 if avoid_flying_hand else 1200.0) + (eff_vel_r - fly_thresh) * 2.4
                    if dist_hand_r > 260.0 and eff_vel_r > profile.max_arm_velocity * 1.18:
                        new_fly_flags.append(
                            (
                                t,
                                ev.bar,
                                list(r_ass.zone_names),
                                f"Forced flying hand (R): jump {dist_hand_r:.0f}mm in {eff_dt_r*1000:.0f}ms ({eff_vel_r:.0f} mm/s > limit {profile.max_arm_velocity:.0f} mm/s)",
                                eff_vel_r / profile.max_arm_velocity,
                            )
                        )
                elif vel_r > profile.max_arm_velocity:
                    move_cost += (vel_r - profile.max_arm_velocity) * 0.45

                # ---- Hemisphere Adherence Rule (Left hand -> Left side 5..8; Right hand -> Right side 1..4) ----
                prev_l_key = cur_left_last_key
                prev_r_key = cur_right_last_key
                hemisphere_penalty = 0.0
                if len(strikes) == 1 and is_unidirectional_sweep_event[ev_idx]:
                    last_l_t = b_state.left_tap_times[-1] if b_state.left_tap_times else -999.0
                    last_r_t = b_state.right_tap_times[-1] if b_state.right_tap_times else -999.0
                    prev_active_hand = "L" if last_l_t > last_r_t + 1e-4 else ("R" if last_r_t > last_l_t + 1e-4 else "")
                    cur_active_hand = "L" if l_ass.note_ids else "R"
                    cur_btn_s = strikes[0].button or 0
                    # Reward continuing a smooth sweep on the same hand within its hemisphere
                    if l_ass.mode == "Sweep" or r_ass.mode == "Sweep":
                        if cur_active_hand == prev_active_hand:
                            move_cost -= 95.0
                    # On long sweeps (`sweep_run_len >= 6`), enforce Two-Handed Hemisphere Relay (`R` on 1..4, `L` on 5..8);
                    # On short 3~4 button chord-anchored sweeps (e.g. `2/4, 3, 4` or `3/5, 4, 5` at 87s), allow the sweeping
                    # hand to finish its 3-button arc without a hemisphere penalty!
                    other_dist_here = dist_r if cur_active_hand == "L" else dist_l
                    this_dist_here = dist_l if cur_active_hand == "L" else dist_r
                    preventing_fly = (dt <= 0.080 and this_dist_here <= 195.0 and other_dist_here > 245.0)
                    if not l_locked and not r_locked and sweep_run_len[ev_idx] >= 6 and not preventing_fly:
                        next_is_15_chord = (
                            ev_idx + 1 < n_events
                            and {n.button for n in events[ev_idx + 1].strike_notes} == {1, 5}
                        )
                        if cur_active_hand == "L" and cur_btn_s in (1, 2, 3, 4):
                            if cur_btn_s == 4 and next_is_15_chord:
                                hemisphere_penalty -= 120.0
                            else:
                                hemisphere_penalty += 850.0 + l_ass.target_xy[0] * 1.8
                        elif cur_active_hand == "R" and cur_btn_s == 4 and next_is_15_chord:
                            # Right hand must release after 3 so it can reach 1 of the upcoming `1/5` chord
                            hemisphere_penalty += 680.0
                        elif cur_active_hand == "R" and cur_btn_s in (5, 6, 7, 8):
                            hemisphere_penalty += 850.0 + abs(r_ass.target_xy[0]) * 1.8
                elif len(strikes) == 1 and decomp_tag == "NONE":
                    # Only treat as a free two-hand interactive trill if the PREVIOUS step was ALREADY fast (dt <= 0.165)
                    # AND neither hand is preparing for an imminent slide launch on its own side!
                    is_first_of_song = (ev_idx == 0 and (initial_state is None or not initial_state.nodes))
                    last_l_t = b_state.left_tap_times[-1] if b_state.left_tap_times else -999.0
                    last_r_t = b_state.right_tap_times[-1] if b_state.right_tap_times else -999.0
                    prev_active_hand = "L" if last_l_t > last_r_t + 1e-4 else ("R" if last_r_t > last_l_t + 1e-4 else "")
                    cur_active_hand = "L" if l_ass.note_ids else "R"
                    strike_x = l_ass.target_xy[0] if l_ass.note_ids else r_ass.target_xy[0]
                    prev_ev_x = (
                        _resolve_note_default_target(events[ev_idx - 1].strike_notes[0], mode)[1][0]
                        if (ev_idx > 0 and len(events[ev_idx - 1].strike_notes) == 1)
                        else 0.0
                    )
                    is_sustained_same_hemi_trill = (
                        same_hemi_fast_len[ev_idx] >= 3
                        and dt <= 0.175
                    )
                    is_cross_hemi_fast = (dt <= 0.165 and strike_x * prev_ev_x < -400.0)
                    is_interactive_trill = (
                        (
                            (is_sustained_same_hemi_trill or is_cross_hemi_fast or (is_first_of_song and next_dt <= 0.165))
                            if calibrate_handflow
                            else (dt <= 0.165 or (is_first_of_song and next_dt <= 0.165))
                        )
                        and (not is_unidirectional_sweep_event[ev_idx])
                        and (not l_locked)
                        and (not r_locked)
                        and (not pending_l_slides)
                        and (not pending_r_slides)
                    )

                    if is_interactive_trill:
                        if prev_active_hand != "":
                            if cur_active_hand == prev_active_hand and l_ass.mode != "Rub" and r_ass.mode != "Rub":
                                hemisphere_penalty += 1600.0
                            else:
                                hemisphere_penalty -= 65.0
                                if l_ass.target_xy[0] > r_ass.target_xy[0] + 20.0:
                                    hemisphere_penalty += (2100.0 if calibrate_handflow else 280.0) + (l_ass.target_xy[0] - r_ass.target_xy[0]) * 2.0
                        if calibrate_handflow and is_cross_hemi_fast:
                            if l_ass.note_ids and l_ass.target_xy[0] > 20.0:
                                hemisphere_penalty += 950.0 + l_ass.target_xy[0] * 1.8
                            elif r_ass.note_ids and r_ass.target_xy[0] < -20.0:
                                hemisphere_penalty += 950.0 + abs(r_ass.target_xy[0]) * 1.8
                    elif (
                        calibrate_handflow
                        and same_hemi_fast_len[ev_idx] >= 3
                        and next_dt <= 0.175
                        and (not is_unidirectional_sweep_event[ev_idx])
                        and (not l_locked)
                        and (not r_locked)
                        and ev_idx + 1 < n_events
                        and len(events[ev_idx + 1].strike_notes) == 1
                    ):
                        # Entry note of a >=3-note same-hemisphere trill: choose starting hand so L stays left of R (x_L <= x_R)
                        nx_trill = _resolve_note_default_target(events[ev_idx + 1].strike_notes[0], mode)[1][0]
                        if strike_x < nx_trill - 10.0:
                            want_entry_h = "L"
                        elif strike_x > nx_trill + 10.0:
                            want_entry_h = "R"
                        else:
                            want_entry_h = "L" if strike_x < 0.0 else "R"
                        if cur_active_hand != want_entry_h:
                            hemisphere_penalty += 750.0
                    else:
                        is_hold_note = strikes[0].note_type in (NoteType.HOLD, NoteType.TOUCH_HOLD)
                        hemi_base = 95.0 if is_hold_note else (540.0 if calibrate_handflow else 360.0)
                        midline_tol = 18.0
                        if l_ass.note_ids and l_ass.target_xy[0] > midline_tol and not r_locked and l_ass.mode != "Sweep":
                            hemisphere_penalty += hemi_base + l_ass.target_xy[0] * (0.4 if is_hold_note else 1.2)
                        if r_ass.note_ids and r_ass.target_xy[0] < -midline_tol and not l_locked and r_ass.mode != "Sweep":
                            hemisphere_penalty += hemi_base + abs(r_ass.target_xy[0]) * (0.4 if is_hold_note else 1.2)
                elif len(strikes) >= 2:
                    if l_ass.target_xy[0] > r_ass.target_xy[0] + 15.0:
                        # If crossing the midline continues a smooth bilateral sweep (`1/5 -> 4/8 -> 3-6/7-2` at 52.8s
                        # or `4/8 -> 1qq2/5qq6` at 120.0s), allow torso-rotated cross-hand (`反手`) with minimal penalty!
                        allow_chord_cross = (
                            (l_ass.mode == "Sweep" and r_ass.mode == "Sweep")
                            or (len(slides_starting) >= 1 and dist_l <= 195.0 and dist_r <= 195.0 and dt <= 0.095)
                            or ((not calibrate_handflow) and dist_l <= 195.0 and dist_r <= 195.0 and dt <= 0.095)
                        )
                        if allow_chord_cross:
                            hemisphere_penalty += 60.0
                        else:
                            hemisphere_penalty += (1250.0 if calibrate_handflow else 650.0) + (l_ass.target_xy[0] - r_ass.target_xy[0]) * 2.0

                # ---- Section 5.2: Hold Release Recovery Penalty (120 ms) ----
                hold_release_penalty = 0.0
                l_hold_recovering = (-0.150 <= t - cur_l_rel[0] <= 0.120)
                r_hold_recovering = (-0.150 <= t - cur_r_rel[0] <= 0.120)
                l_slide_recovering = (0.0 <= t - last_sl_l[1] <= 0.120)
                r_slide_recovering = (0.0 <= t - last_sl_r[1] <= 0.120)
                if l_ass.note_ids and l_hold_recovering:
                    d_rel_l = math.hypot(l_ass.target_xy[0] - cur_l_rel[1][0], l_ass.target_xy[1] - cur_l_rel[1][1])
                    if d_rel_l > 25.0:
                        # MajdataPlay Hold tail leniency allows releasing ~85ms early for same-hemisphere transitions;
                        # Crossing to the opposite hemisphere at t_release (e.g. Case 8: 7h -> 3) is heavily penalized.
                        crosses_hemi_l = (l_ass.target_xy[0] > 35.0)
                        if crosses_hemi_l or d_rel_l > 330.0:
                            dt_rel_l = max(0.095, t - cur_l_rel[0] + 0.120)
                            v_rel_l = d_rel_l / dt_rel_l
                            hold_release_penalty += 450.0 + d_rel_l * 1.8
                            if v_rel_l > profile.max_arm_velocity:
                                hold_release_penalty += 3600.0 + (v_rel_l - profile.max_arm_velocity) * 0.6
                        else:
                            hold_release_penalty += 80.0 + d_rel_l * 0.25
                if l_ass.note_ids and l_slide_recovering:
                    d_sl_l = math.hypot(l_ass.target_xy[0] - last_sl_l[2][0], l_ass.target_xy[1] - last_sl_l[2][1])
                    if d_sl_l > 45.0:
                        dt_sl_l = max(0.025, t - last_sl_l[1])
                        v_sl_l = d_sl_l / dt_sl_l
                        hold_release_penalty += 450.0 + d_sl_l * 1.8
                        if v_sl_l > profile.max_arm_velocity:
                            hold_release_penalty += 3800.0 + (v_sl_l - profile.max_arm_velocity) * 0.6
                if r_ass.note_ids and r_hold_recovering:
                    d_rel_r = math.hypot(r_ass.target_xy[0] - cur_r_rel[1][0], r_ass.target_xy[1] - cur_r_rel[1][1])
                    if d_rel_r > 25.0:
                        crosses_hemi_r = (r_ass.target_xy[0] < -35.0)
                        if crosses_hemi_r or d_rel_r > 330.0:
                            dt_rel_r = max(0.095, t - cur_r_rel[0] + 0.120)
                            v_rel_r = d_rel_r / dt_rel_r
                            hold_release_penalty += 450.0 + d_rel_r * 1.8
                            if v_rel_r > profile.max_arm_velocity:
                                hold_release_penalty += 3600.0 + (v_rel_r - profile.max_arm_velocity) * 0.6
                        else:
                            hold_release_penalty += 80.0 + d_rel_r * 0.25
                if r_ass.note_ids and r_slide_recovering:
                    d_sl_r = math.hypot(r_ass.target_xy[0] - last_sl_r[2][0], r_ass.target_xy[1] - last_sl_r[2][1])
                    if d_sl_r > 45.0:
                        dt_sl_r = max(0.025, t - last_sl_r[1])
                        v_sl_r = d_sl_r / dt_sl_r
                        hold_release_penalty += 450.0 + d_sl_r * 1.8
                        if v_sl_r > profile.max_arm_velocity:
                            hold_release_penalty += 3800.0 + (v_sl_r - profile.max_arm_velocity) * 0.6

                # ---- Law 2: Single-Hand Over-Speed Check & Decomposition Reward ----
                freq_penalty = 0.0
                new_forced_flags = list(b_state.forced_single_jack_flags)
                new_jump_flags = list(b_state.forced_single_jump_flags)
                new_l_sk_streak = b_state.left_same_key_streak
                new_r_sk_streak = b_state.right_same_key_streak

                for h_ass, prev_h_key, prev_sk_streak, is_other_locked, other_holds, other_slides, is_other_recovering in (
                    (l_ass, prev_l_key, b_state.left_same_key_streak, r_locked, locked_r_holds, locked_r_slides, r_hold_recovering or r_slide_recovering),
                    (r_ass, prev_r_key, b_state.right_same_key_streak, l_locked, locked_l_holds, locked_l_slides, l_hold_recovering or l_slide_recovering),
                ):
                    if not h_ass.note_ids or h_ass.mode in ("Rub", "Sweep"):
                        continue
                    cur_z_key = h_ass.zone_names[0] if h_ass.zone_names else ""
                    same_btn = (
                        prev_h_key != ""
                        and cur_z_key != ""
                        and (prev_h_key == cur_z_key or (prev_h_key[-1:].isdigit() and prev_h_key[-1:] == cur_z_key[-1:]))
                    )
                    if same_btn:
                        cur_sk_streak = (prev_sk_streak + 1) if h_ass.instant_freq_hz >= 4.5 else 2
                        if h_ass.hand == "L":
                            new_l_sk_streak = cur_sk_streak
                        else:
                            new_r_sk_streak = cur_sk_streak
                        eff_jack_streak = max(jack_streak_length[ev_idx], cur_sk_streak)
                        single_limit_hz = profile.max_single_jack_hz(eff_jack_streak)
                        if h_ass.instant_freq_hz > single_limit_hz:
                            freq_penalty += 2500.0 + (h_ass.instant_freq_hz - single_limit_hz) * 400.0
                            if eff_jack_streak >= 3 and is_other_locked:
                                lock_desc = "other hand busy"
                                if other_holds:
                                    lock_desc = f"other hand locked on Hold ({other_holds[0][2]})"
                                elif other_slides:
                                    lock_desc = "other hand locked on active Slide"
                                new_forced_flags.append(
                                    (
                                        t,
                                        ev.bar,
                                        [sn.key for sn in strikes],
                                        (
                                            f"Forced single-hand jackhammer on key {[sn.key for sn in strikes]}: "
                                            f"{h_ass.hand} hand at {h_ass.instant_freq_hz:.2f} Hz > limit "
                                            f"{single_limit_hz:.2f} Hz ({lock_desc})"
                                        ),
                                        h_ass.instant_freq_hz / single_limit_hz,
                                    )
                                )
                    elif prev_h_key != "" and cur_z_key != "":
                        if h_ass.hand == "L":
                            new_l_sk_streak = 1
                        else:
                            new_r_sk_streak = 1
                        pb = int(prev_h_key[-1]) if prev_h_key[-1:].isdigit() else -1
                        cb = int(cur_z_key[-1]) if cur_z_key[-1:].isdigit() else -1
                        if pb in range(1, 9) and cb in range(1, 9):
                            btn_step = min((cb - pb) % 8, (pb - cb) % 8)
                        else:
                            btn_step = 1
                        hop_limit_hz = (
                            profile.max_single_hand_adjacent_hop_hz
                            if btn_step <= 1
                            else profile.max_single_hand_far_jump_hz
                        )
                        # When the other hand is free, any single-hand hop > hop_limit_hz is penalized so the planner uses L-R alternation;
                        # When the other hand is locked on a Slide (e.g. Bar 30 `3, 2, 1-4` = 3 notes while L slides `8-5`),
                        # a short <=3-note adjacent roll is allowed, whereas a >=4-note forced oscillation (Case 6 `6,7,6,7,6,7,6,7`)
                        # triggers FORCED_SINGLE_HAND_JUMP!
                        # Note: only count fast hops (`dt <= 0.145s`) toward forced-oscillation streak, and allow <=3-step adjacent rolls!
                        fast_streak = 1
                        h_taps_chk = b_state.left_tap_times if h_ass.hand == "L" else b_state.right_tap_times
                        if h_taps_chk and (t - h_taps_chk[-1]) <= 0.145:
                            fast_streak = 2
                            for k_st in range(len(h_taps_chk) - 1, 0, -1):
                                if h_taps_chk[k_st] - h_taps_chk[k_st - 1] <= 0.145:
                                    fast_streak += 1
                                else:
                                    break
                        eff_hop_limit = hop_limit_hz * (2.60 if (btn_step <= 1 and fast_streak <= 3) else (1.40 if fast_streak <= 3 else 1.0))
                        if h_ass.instant_freq_hz > hop_limit_hz:
                            is_short_native_2roll = (
                                calibrate_handflow
                                and len(strikes) == 1
                                and same_hemi_fast_len[ev_idx] <= 2
                                and btn_step <= 1
                                and fast_streak <= 2
                                and ((h_ass.hand == "L" and h_ass.target_xy[0] < -15.0) or (h_ass.hand == "R" and h_ass.target_xy[0] > 15.0))
                            )
                            if is_short_native_2roll:
                                freq_penalty += 45.0
                            elif (not is_other_locked and not is_other_recovering) or (is_other_locked and fast_streak >= 4 and h_ass.instant_freq_hz > eff_hop_limit) or (is_other_locked and btn_step >= 2 and h_ass.instant_freq_hz > hop_limit_hz * 1.55):
                                freq_penalty += 3200.0 + (h_ass.instant_freq_hz - hop_limit_hz) * 500.0
                            if is_other_locked and ((fast_streak >= 4 and h_ass.instant_freq_hz > eff_hop_limit) or (btn_step >= 3 and fast_streak >= 3 and h_ass.instant_freq_hz > hop_limit_hz * 1.85)):
                                lock_desc = "other hand busy"
                                if other_holds:
                                    lock_desc = f"other hand locked on Hold ({other_holds[0][2]})"
                                elif other_slides:
                                    lock_desc = "other hand locked on active Slide"
                                new_jump_flags.append(
                                    (
                                        t,
                                        ev.bar,
                                        [prev_h_key, cur_z_key],
                                        (
                                            f"Forced single-hand cross-key jump ({prev_h_key} -> {cur_z_key}): "
                                            f"{h_ass.hand} hand at {h_ass.instant_freq_hz:.2f} Hz > hop limit "
                                            f"{hop_limit_hz:.2f} Hz ({lock_desc})"
                                        ),
                                        (h_ass.instant_freq_hz / hop_limit_hz) * (1.0 + 0.14 * max(0, fast_streak - 4)),
                                    )
                                )

                if is_fast_single_jack_event[ev_idx] and decomp_tag in ("OUTER_INNER_SPLIT", "ALTERNATING_TAP"):
                    move_cost -= 25.0

                new_l_taps = (
                    _compute_hand_tap_metrics(b_state.left_tap_times, t)[2]
                    if (l_ass.note_ids and l_ass.mode not in ("Rub", "Sweep"))
                    else b_state.left_tap_times
                )
                new_r_taps = (
                    _compute_hand_tap_metrics(b_state.right_tap_times, t)[2]
                    if (r_ass.note_ids and r_ass.mode not in ("Rub", "Sweep"))
                    else b_state.right_tap_times
                )

                new_coll_flags = list(b_state.collision_flags)
                # simval round2: two simultaneous targets <= 2*R_HAND apart (touch stack B5/E5, C+B...) are covered by ONE palm
                _stack_ok = bool(
                    (l_ass.note_ids or l_is_holding_now)
                    and (r_ass.note_ids or r_is_holding_now)
                    and (l_ass.note_ids or r_ass.note_ids)
                    and (
                        (l_ass.zone_names == r_ass.zone_names and any(z.startswith(("B", "C", "D", "E")) for z in l_ass.zone_names))
                        or (l_ass.note_ids and r_ass.note_ids and math.hypot(l_ass.target_xy[0] - r_ass.target_xy[0], l_ass.target_xy[1] - r_ass.target_xy[1]) <= 2.0 * 48.0 - 1.0 and mode != HitMode.INNER)
                    )
                )
                eff_hand_coll = b_is_coll and not _stack_ok
                if eff_hand_coll:
                    coll_raw_m = (85.0 - b_dist_xy) if _palm_coll else (120.0 - b_forearm_3d)
                    new_coll_flags.append(
                        (
                            t,
                            ev.bar,
                            list(l_ass.zone_names) + list(r_ass.zone_names),
                            (
                                ("Palm collision" if _palm_coll else "Forearm collision")
                                + f" (palm dist {b_dist_xy:.1f} mm, |dz| {b_z_diff:.1f} mm, forearm 3D {b_forearm_3d:.1f} mm)"
                            ),
                            coll_raw_m,
                        )
                    )

                # Only flag `EXTREME_CROSS_ARM` as an unreasonable error when a hand is forced into a high-speed
                # cross-body oscillation while the other hand is locked (e.g. Case 8: 7h + 2,3,2,3), NOT on intentional
                # static cross-hand Holds (92.8s `2h + 6h`) or smooth bilateral windmill sweeps (52.8s / 120.0s)!
                is_intentional_cross_tech = (
                    (len(strikes) >= 2 and dist_l <= 210.0 and dist_r <= 210.0)
                    or (len(strikes) == 1 and strikes[0].note_type in (NoteType.HOLD, NoteType.TOUCH_HOLD) and dt >= 0.20)
                )
                eff_extreme_cross = b_is_ext_cross and not is_intentional_cross_tech
                new_cross_flags = list(b_state.cross_flags)
                if eff_extreme_cross:
                    new_cross_flags.append(
                        (
                            t,
                            ev.bar,
                            list(l_ass.zone_names) + list(r_ass.zone_names),
                            (
                                f"Extreme cross-body / torso whiplash: L_x={l_xyz[0]:.0f}mm, R_x={r_xyz[0]:.0f}mm, "
                                f"yaw_accel={cand_torso.yaw_accel_deg_s2:.0f} deg/s^2"
                            ),
                            abs(float(cand_torso.yaw_accel_deg_s2)),
                        )
                    )

                new_holds = list(surviving_holds)
                for sn in strikes:
                    if sn.note_type in (NoteType.HOLD, NoteType.TOUCH_HOLD):
                        assigned_h, assigned_z = nh_map.get(sn.note_id, ("L", sn.sensor_zone))
                        h_xy = l_ass.target_xy if assigned_h == "L" else r_ass.target_xy
                        new_holds.append(
                            (sn.note_id, assigned_h, assigned_z, h_xy, sn.time, sn.end_time, sn.note_type == NoteType.TOUCH_HOLD)
                        )

                new_slides = list(surviving_slides)
                step_nh_map = dict(nh_map)
                one_stroke_bonus = 0.0
                for sl in slides_starting:
                    s_xy = sl.slide_waypoints_mm[0] if sl.slide_waypoints_mm else (0.0, 0.0)
                    e_xy = sl.slide_waypoints_mm[-1] if sl.slide_waypoints_mm else (0.0, 0.0)
                    ebtn = sl.slide_segments[-1].end_btn if sl.slide_segments else (sl.button or 1)
                    is_wifi = any(seg.shape == "w" for seg in sl.slide_segments)
                    # Check active surviving slides for an ongoing One-Stroke (`一笔画`) chain:
                    # e.g. `1V35[4:1]` (slid by R, ending at btn 5 at t=59.72s) connecting into `5-2[8:1]` (launching from btn 5 at shoot_time=59.72s)!
                    active_chain_hand: Optional[str] = None
                    for s_act in surviving_slides:
                        if s_act[1] in ("L", "R") and s_act[8] == sl.button and abs(s_act[3] - sl.shoot_time) <= 0.16:
                            active_chain_hand = s_act[1]
                            break
                    if is_wifi:
                        # WifiSlide ('w') ALWAYS occupies BOTH hands!
                        sl_hand = "LR"
                    elif active_chain_hand is not None and not (locked_l_holds if active_chain_hand == "L" else locked_r_holds):
                        sl_hand = active_chain_hand
                        one_stroke_bonus -= 140.0
                    elif sl.button == last_sl_l[0] and abs(sl.shoot_time - last_sl_l[1]) <= 0.30 and not locked_l_holds:
                        sl_hand = "L"
                        one_stroke_bonus -= 85.0
                    elif sl.button == last_sl_r[0] and abs(sl.shoot_time - last_sl_r[1]) <= 0.30 and not locked_r_holds:
                        sl_hand = "R"
                        one_stroke_bonus -= 85.0
                    elif len(slides_starting) == 2:
                        # When 2 slides start simultaneously (like `3-6 / 7-2` at 52.92s or `1qq2 / 5qq6` at 120.00s),
                        # each slide is slid by the hand that just struck its star head at `t`!
                        head_strike = next((sn_h for sn_h in strikes if sn_h.button == sl.button), None)
                        if head_strike is not None and head_strike.note_id in nh_map:
                            sl_hand = nh_map[head_strike.note_id][0]
                        else:
                            sl_hand = "L" if s_xy[0] <= 0 else "R"
                    elif locked_l_slides and not locked_r_slides and not locked_r_holds:
                        sl_hand = "R"
                    elif locked_r_slides and not locked_l_slides and not locked_l_holds:
                        sl_hand = "L"
                    else:
                        # Check if this slide crosses hemispheres (`s_xy[0] * e_xy[0] < -1000`) and simultaneous taps
                        # occur while it is moving (`shoot_time .. end_time`):
                        sim_tap_xs: List[float] = []
                        slide_crosses_hemi = (s_xy[0] * e_xy[0] < -1000.0)
                        if slide_crosses_hemi:
                            for look_ev in events[ev_idx + 1 : min(n_events, ev_idx + 12)]:
                                if (look_ev.bar // 4) != (ev.bar // 4):
                                    break
                                if look_ev.time > sl.end_time - 0.04:
                                    break
                                if look_ev.time >= sl.shoot_time - 0.04:
                                    for sn_lk in look_ev.strike_notes:
                                        if sn_lk.note_type in (NoteType.TAP, NoteType.HOLD):
                                            sim_tap_xs.append(_resolve_note_default_target(sn_lk, mode)[1][0])
                        if slide_crosses_hemi and sim_tap_xs and all(tx < -120.0 for tx in sim_tap_xs):
                            sl_hand = "R"
                        elif slide_crosses_hemi and sim_tap_xs and all(tx > 120.0 for tx in sim_tap_xs):
                            sl_hand = "L"
                        else:
                            head_strike_1 = next((sn_h for sn_h in strikes if sn_h.button == sl.button), None)
                            # Otherwise start the slide with the hand matching its start hemisphere `s_xy[0]`
                            if calibrate_handflow and head_strike_1 is not None and head_strike_1.note_id in nh_map:
                                sl_hand = nh_map[head_strike_1.note_id][0]
                            elif abs(s_xy[0]) > 25.0:
                                sl_hand = "L" if s_xy[0] < 0 else "R"
                            else:
                                sl_hand = "L" if e_xy[0] <= 0 else "R"
                    new_slides.append((sl.note_id, sl_hand, sl.shoot_time, sl.end_time, sl.judge_time, s_xy, e_xy, sl.slide_waypoints_mm, ebtn))
                    step_nh_map[sl.note_id] = ("R" if sl_hand == "R" else "L", sl.sensor_zone)

                step_cost = (
                    (b_posture_cost - (1100.0 if is_intentional_cross_tech and b_is_ext_cross else 0.0))
                    + move_cost
                    + hemisphere_penalty
                    + hold_release_penalty
                    + freq_penalty
                    + one_stroke_bonus
                    + (4000.0 if eff_hand_coll else 0.0)
                    + (3500.0 if eff_extreme_cross else 0.0)
                )
                next_beam.append(
                    _BeamState(
                        left_xy=l_ass.target_xy,
                        right_xy=r_ass.target_xy,
                        left_z=l_ass.target_z_mm,
                        right_z=r_ass.target_z_mm,
                        left_tap_times=new_l_taps,
                        right_tap_times=new_r_taps,
                        left_last_key=l_ass.zone_names[0] if l_ass.zone_names else cur_left_last_key,
                        right_last_key=r_ass.zone_names[0] if r_ass.zone_names else cur_right_last_key,
                        active_holds=tuple(new_holds),
                        active_slides=tuple(new_slides),
                        torso=cand_torso,
                        cum_cost=b_state.cum_cost + step_cost,
                        left_hold_release=cur_l_rel,
                        right_hold_release=cur_r_rel,
                        last_slide_end_l=last_sl_l,
                        last_slide_end_r=last_sl_r,
                        left_last_contact=(t, l_ass.target_xy) if l_ass.note_ids else (cur_l_rel if l_is_holding_now else b_state.left_last_contact),
                        right_last_contact=(t, r_ass.target_xy) if r_ass.note_ids else (cur_r_rel if r_is_holding_now else b_state.right_last_contact),
                        left_same_key_streak=new_l_sk_streak,
                        right_same_key_streak=new_r_sk_streak,
                        history_node=(b_state.history_node, (l_ass, r_ass, (l_is_holding_now, r_is_holding_now), decomp_tag, step_cost, step_nh_map)),
                        forced_single_jack_flags=new_forced_flags,
                        forced_single_jump_flags=new_jump_flags,
                        collision_flags=new_coll_flags,
                        cross_flags=new_cross_flags,
                        flying_hand_flags=new_fly_flags,
                    )
                )

        next_beam.sort(key=lambda s: s.cum_cost)
        dedup_beam: List[_BeamState] = []
        seen_sigs: Set[Tuple[Any, ...]] = set()
        for st in next_beam:
            last_item = st.history_node[1] if st.history_node is not None else None
            prev_node = st.history_node[0] if st.history_node is not None else None
            prev_item = prev_node[1] if prev_node is not None else None
            prev2_node = prev_node[0] if prev_node is not None else None
            prev2_item = prev2_node[1] if prev2_node is not None else None
            sig = (
                int(round(st.left_xy[0] / sig_grid_mm)),
                int(round(st.left_xy[1] / sig_grid_mm)),
                int(round(st.right_xy[0] / sig_grid_mm)),
                int(round(st.right_xy[1] / sig_grid_mm)),
                st.left_last_key,
                st.right_last_key,
                int(round(st.torso.yaw_deg / 4.0)),
                last_item[0].mode if last_item is not None else "",
                last_item[1].mode if last_item is not None else "",
                (bool(prev_item[0].note_ids), bool(prev_item[1].note_ids)) if prev_item is not None else (False, False),
                ((bool(prev2_item[0].note_ids), bool(prev2_item[1].note_ids)) if (eff_beam_width > 6 and prev2_item is not None) else ()),
            )
            if sig not in seen_sigs:
                seen_sigs.add(sig)
                dedup_beam.append(st)
            if len(dedup_beam) >= eff_beam_width:
                break
        beam = dedup_beam if dedup_beam else next_beam[:eff_beam_width]

    best_state = beam[0] if beam else init_state
    init_lft = initial_state.last_flag_times if initial_state is not None else (-999.0, -999.0, -999.0, -999.0, -999.0)
    last_flag_t, last_jump_t, last_coll_t, last_cross_t, last_fly_t = init_lft

    # Consolidate unreasonable flags from best path
    if best_state.forced_single_jack_flags:
        for f_t, f_bar, f_keys, f_reason, f_sev in best_state.forced_single_jack_flags:
            if f_t - last_flag_t > 0.25:
                unreasonable_records.append(
                    UnreasonableEventRecord(
                        time=f_t,
                        bar=f_bar,
                        type=UnreasonableType.FORCED_SINGLE_HAND_JACK.value,
                        keys=f_keys,
                        reason=f_reason,
                        severity=min(3.0, f_sev),
                        raw_m=f_sev,
                    )
                )
                last_flag_t = f_t

    if best_state.forced_single_jump_flags:
        for j_t, j_bar, j_keys, j_reason, j_sev in best_state.forced_single_jump_flags:
            if j_t - last_jump_t > 0.20:
                unreasonable_records.append(
                    UnreasonableEventRecord(
                        time=j_t,
                        bar=j_bar,
                        type=UnreasonableType.FORCED_SINGLE_HAND_JUMP.value,
                        keys=j_keys,
                        reason=j_reason,
                        severity=min(3.0, j_sev),
                        raw_m=j_sev,
                    )
                )
                last_jump_t = j_t

    if best_state.collision_flags:
        for c_t, c_bar, c_keys, c_reason, c_raw_m in best_state.collision_flags:
            if c_t - last_coll_t > 0.15:
                unreasonable_records.append(
                    UnreasonableEventRecord(
                        time=c_t,
                        bar=c_bar,
                        type=(UnreasonableType.HAND_COLLISION.value if c_reason.startswith("Palm") else UnreasonableType.ARM_COLLISION.value),
                        keys=c_keys,
                        reason=c_reason,
                        severity=2.0,
                        raw_m=c_raw_m,
                    )
                )
                last_coll_t = c_t

    if best_state.cross_flags:
        for x_t, x_bar, x_keys, x_reason, x_raw_m in best_state.cross_flags:
            if x_t - last_cross_t > 0.20:
                unreasonable_records.append(
                    UnreasonableEventRecord(
                        time=x_t,
                        bar=x_bar,
                        type=UnreasonableType.EXTREME_CROSS_ARM.value,
                        keys=x_keys,
                        reason=x_reason,
                        severity=2.2,
                        raw_m=x_raw_m,
                    )
                )
                last_cross_t = x_t

    if best_state.flying_hand_flags:
        for fl_t, fl_bar, fl_keys, fl_reason, fl_sev in best_state.flying_hand_flags:
            if fl_t - last_fly_t > 0.25:
                unreasonable_records.append(
                    UnreasonableEventRecord(
                        time=fl_t,
                        bar=fl_bar,
                        type=UnreasonableType.FLYING_HAND.value,
                        keys=fl_keys,
                        reason=fl_reason,
                        severity=min(3.0, fl_sev),
                        raw_m=fl_sev,
                    )
                )
                last_fly_t = fl_t

    unreasonable_records.sort(key=lambda r: (r.time, r.type))

    planned_steps: List[PlannedEventStep] = []
    note_hand_map: Dict[int, Tuple[str, str]] = {}
    prev_l_xy = (-110.0, -45.0)
    prev_r_xy = (110.0, -45.0)
    prev_torso = init_torso

    for ev_idx, (l_ass, r_ass, body, decomp_tag, step_cost, step_map) in enumerate(best_state.unpack_history()):
        ev = events[ev_idx]
        note_hand_map.update(step_map)
        dt_step = max(0.04, ev.time - (events[ev_idx - 1].time if ev_idx > 0 else 0.0))
        if not isinstance(body, BodyKinematicsState):
            lh_flag, rh_flag = body
            dt_ik = max(0.005, ev.time - (events[ev_idx - 1].time if ev_idx > 0 else max(0.0, ev.time - 0.25)))
            body = solve_coupled_body_ik(
                ev.time,
                (l_ass.target_xy[0], l_ass.target_xy[1], l_ass.target_z_mm),
                (r_ass.target_xy[0], r_ass.target_xy[1], r_ass.target_z_mm),
                prev_torso,
                dt_ik,
                profile,
                left_is_holding=lh_flag,
                right_is_holding=rh_flag,
            )

        dl = math.hypot(l_ass.target_xy[0] - prev_l_xy[0], l_ass.target_xy[1] - prev_l_xy[1])
        dr = math.hypot(r_ass.target_xy[0] - prev_r_xy[0], r_ass.target_xy[1] - prev_r_xy[1])

        fat_l = (0.016 * (dl if l_ass.note_ids else 0.0) + (1.4 * (l_ass.instant_freq_hz / profile.max_single_hand_tap_hz) ** 2 if l_ass.note_ids else 0.05)) / max(0.25, dt_step)
        fat_r = (0.016 * (dr if r_ass.note_ids else 0.0) + (1.4 * (r_ass.instant_freq_hz / profile.max_single_hand_tap_hz) ** 2 if r_ass.note_ids else 0.05)) / max(0.25, dt_step)
        fat_t = (
            0.03 * math.hypot(body.torso.dx_mm - prev_torso.dx_mm, body.torso.dy_mm - prev_torso.dy_mm)
            + 0.18 * abs(body.torso.yaw_deg - prev_torso.yaw_deg)
            + 0.015 * (body.left_arm.cross_cost + body.right_arm.cross_cost)
        ) / max(0.2, dt_step)

        planned_steps.append(
            PlannedEventStep(
                event_id=ev.event_id,
                time=ev.time,
                bar=ev.bar,
                bpm=ev.bpm,
                left=l_ass,
                right=r_ass,
                body_state=body,
                decomposition_used=decomp_tag,
                step_cost=step_cost,
                step_fatigue_left=round(fat_l, 3),
                step_fatigue_right=round(fat_r, 3),
                step_fatigue_torso=round(fat_t, 3),
            )
        )
        prev_l_xy = l_ass.target_xy
        prev_r_xy = r_ass.target_xy
        prev_torso = body.torso

    if skip_trajectory:
        trajectory_frames = []
    else:
        trajectory_frames = _generate_continuous_trajectory(
            chart=chart,
            planned_steps=planned_steps,
            note_hand_map=note_hand_map,
            unreasonable_events=unreasonable_records,
            profile=profile,
            fps=frame_fps,
            t_range=t_range,
            initial_state=initial_state,
        )

    # Always record the exact timestamp of the last event (`events[-1].time`) so the next window's
    # `dt = t_first - prev_t` matches whole-chart `events[i].time - events[i-1].time` bit-for-bit!
    end_t_val = events[-1].time if events else (initial_state.end_time if initial_state is not None else -999.0)
    end_bar_val = (
        events[-1].bar
        if events
        else (initial_state.end_bar if initial_state else 0)
    )
    base_cost = beam[0].cum_cost if beam else 0.0
    carry_list: List[Dict[str, Any]] = []
    seen_carry_ids: Set[int] = set()
    for n_item in jump_check_notes:
        if n_item.end_time > end_t_val + 0.005 and n_item.note_id not in seen_carry_ids:
            seen_carry_ids.add(n_item.note_id)
            if n_item.note_type in (NoteType.HOLD, NoteType.TOUCH_HOLD):
                carry_list.append(
                    {
                        "note_id": n_item.note_id,
                        "kind": n_item.note_type.value,
                        "time": n_item.time,
                        "end_time": n_item.end_time,
                        "bar": n_item.bar,
                        "key": n_item.key,
                        "button": n_item.button,
                        "sensor_zone": n_item.sensor_zone,
                        "raw_token": n_item.raw_token,
                    }
                )
            elif n_item.note_type == NoteType.SLIDE:
                carry_list.append(
                    {
                        "note_id": n_item.note_id,
                        "kind": "SLIDE",
                        "time": n_item.time,
                        "shoot_time": n_item.shoot_time,
                        "end_time": n_item.end_time,
                        "judge_time": n_item.judge_time,
                        "bar": n_item.bar,
                        "key": n_item.key,
                        "button": n_item.button,
                        "sensor_zone": n_item.sensor_zone,
                        "raw_token": n_item.raw_token,
                        "segments": [
                            {"shape": sg.shape, "start_btn": sg.start_btn, "end_btn": sg.end_btn, "via_btn": sg.via_btn}
                            for sg in n_item.slide_segments
                        ],
                    }
                )
    final_planner_state = PlannerState(
        end_time=end_t_val,
        end_bar=end_bar_val,
        hit_mode=mode.value,
        skill_profile=profile.name,
        nodes=tuple(st.to_snapshot_dict(base_cost=base_cost) for st in beam[:1]),
        carry_notes=tuple(carry_list),
        prior_bar_sigs=initial_state.prior_bar_sigs if initial_state is not None else (),
        cum_fatigue=initial_state.cum_fatigue if initial_state is not None else 0.0,
        tm_changes=initial_state.tm_changes if initial_state is not None else ((0.0, chart.initial_bpm),),
        first_offset_sec=initial_state.first_offset_sec if initial_state is not None else chart.first_offset_sec,
        slot=initial_state.slot if initial_state is not None else 5,
        source_path=chart.source_path or (initial_state.source_path if initial_state is not None else None),
        title=chart.title if chart.title not in ("InlineSimaiChart", "TickNotesChart") else (initial_state.title if initial_state is not None else chart.title),
        last_flag_times=(last_flag_t, last_jump_t, last_coll_t, last_cross_t, last_fly_t),
    )

    return PlanResult(
        chart=chart,
        hit_mode=mode,
        profile=profile,
        planned_steps=planned_steps,
        unreasonable_events=unreasonable_records,
        note_hand_map=note_hand_map,
        trajectory_frames=trajectory_frames,
        final_state=final_planner_state,
    )


def _smoothstep(u: float) -> float:
    u = max(0.0, min(1.0, u))
    return u * u * (3.0 - 2.0 * u)


def _generate_continuous_trajectory(
    chart: ParsedChart,
    planned_steps: Sequence[PlannedEventStep],
    note_hand_map: Dict[int, Tuple[str, str]],
    unreasonable_events: Sequence[UnreasonableEventRecord],
    profile: SkillProfileConfig,
    fps: int = 60,
    t_range: Optional[Tuple[float, float]] = None,
    initial_state: Optional[PlannerState] = None,
) -> List[TrajectoryFrame]:
    """Synthesize smooth 60+ FPS continuous bilateral palm & torso trajectories."""
    init_l_xy: Point2D = (-110.0, -45.0)
    init_r_xy: Point2D = (110.0, -45.0)
    init_torso = TorsoState()
    if initial_state is not None and initial_state.nodes:
        nd0 = initial_state.nodes[0]
        init_l_xy = (float(nd0["left_xy"][0]), float(nd0["left_xy"][1]))
        init_r_xy = (float(nd0["right_xy"][0]), float(nd0["right_xy"][1]))
        tr0 = nd0.get("torso", [0.0] * 7)
        init_torso = TorsoState(dx_mm=float(tr0[0]), dy_mm=float(tr0[1]), yaw_deg=float(tr0[2]))

    if not planned_steps:
        t0_empty = t_range[0] if t_range is not None else 0.0
        body = solve_coupled_body_ik(t0_empty, (init_l_xy[0], init_l_xy[1], Z_HOVER_DEFAULT_MM), (init_r_xy[0], init_r_xy[1], Z_HOVER_DEFAULT_MM), init_torso, 0.016, profile)
        return [
            TrajectoryFrame(
                time=t0_empty,
                bar=initial_state.end_bar if initial_state is not None else 0,
                bpm=chart.initial_bpm,
                left_mode="Hover",
                right_mode="Hover",
                left_freq_hz=0.0,
                right_freq_hz=0.0,
                left_zones=(),
                right_zones=(),
                active_note_zones=(),
                active_unreasonable=(),
                left_palm_xyz=(init_l_xy[0], init_l_xy[1], Z_HOVER_DEFAULT_MM),
                right_palm_xyz=(init_r_xy[0], init_r_xy[1], Z_HOVER_DEFAULT_MM),
                _torso=init_torso,
                _body=body,
            )
        ]

    dt = 1.0 / float(max(20, fps))
    t_start = t_range[0] if t_range is not None else 0.0
    t_end = t_range[1] if t_range is not None else max(chart.total_duration_sec + 0.25, planned_steps[-1].time + 0.25)
    if t_end <= t_start:
        t_end = t_start + dt
    n_frames = max(1, int(math.ceil((t_end - t_start) / dt)) + 1)

    l_keyframes: List[Tuple[float, Point2D, str, Tuple[str, ...], float]] = [
        (t_start, init_l_xy, "Hover", (), 0.0)
    ]
    r_keyframes: List[Tuple[float, Point2D, str, Tuple[str, ...], float]] = [
        (t_start, init_r_xy, "Hover", (), 0.0)
    ]
    if planned_steps[0].time > t_start + 0.35:
        l_keyframes.append((planned_steps[0].time - 0.25, init_l_xy, "Hover", (), 0.0))
        r_keyframes.append((planned_steps[0].time - 0.25, init_r_xy, "Hover", (), 0.0))
    for st in planned_steps:
        if st.left.note_ids or len(l_keyframes) == 1:
            l_keyframes.append((st.time, st.left.target_xy, st.left.mode, st.left.zone_names, st.left.instant_freq_hz))
        if st.right.note_ids or len(r_keyframes) == 1:
            r_keyframes.append((st.time, st.right.target_xy, st.right.mode, st.right.zone_names, st.right.instant_freq_hz))

    holds_by_hand: Dict[str, List[Tuple[ParsedNote, Point2D, str]]] = {"L": [], "R": []}
    slides_by_hand: Dict[str, List[ParsedNote]] = {"L": [], "R": []}
    for n in chart.notes:
        h_side, z_name = note_hand_map.get(n.note_id, ("L", n.sensor_zone))
        if n.note_type in (NoteType.HOLD, NoteType.TOUCH_HOLD):
            z_xy = get_sensor_zone(z_name).center
            holds_by_hand[h_side].append((n, z_xy, z_name))
        elif n.note_type == NoteType.SLIDE:
            if any(seg.shape == "w" for seg in n.slide_segments):
                slides_by_hand["L"].append(n)
                slides_by_hand["R"].append(n)
            else:
                slides_by_hand[h_side].append(n)

    l_kf_times = [kf[0] for kf in l_keyframes]
    r_kf_times = [kf[0] for kf in r_keyframes]

    notes_by_sec: Dict[int, List[ParsedNote]] = {}
    for n in chart.notes:
        s0 = int(math.floor(n.time - 0.1))
        s1 = int(math.ceil(n.end_time + 0.1))
        for sec_k in range(s0, s1 + 1):
            notes_by_sec.setdefault(sec_k, []).append(n)

    bar_starts = [b[1] for b in chart.bar_timestamps]

    def _eval_hand_at_time(
        hand: str,
        t: float,
        keyframes: Sequence[Tuple[float, Point2D, str, Tuple[str, ...], float]],
        kf_times: Sequence[float],
    ) -> Tuple[Point3D, str, float, Tuple[str, ...]]:
        # Check if this hand has a discrete Tap/Hold strike right near `t` (detouring from a long slide if needed)
        idx = max(0, bisect.bisect_right(kf_times, t) - 1)
        near_tap = False
        if idx < len(keyframes) and abs(t - keyframes[idx][0]) <= 0.045 and keyframes[idx][2] != "Hover":
            near_tap = True
        elif idx + 1 < len(keyframes) and abs(keyframes[idx + 1][0] - t) <= 0.045 and keyframes[idx + 1][2] != "Hover":
            near_tap = True

        for sl in slides_by_hand[hand]:
            if not near_tap and (sl.shoot_time <= t <= sl.end_time + 0.015):
                dur = max(0.01, sl.end_time - sl.shoot_time)
                prog = max(0.0, min(1.0, (t - sl.shoot_time) / dur))
                if any(seg.shape == "w" for seg in sl.slide_segments) and sl.wifi_queues is not None:
                    # Wifi slide: Left hand sweeps left fan wing, Right hand sweeps right fan wing
                    lane_q = sl.wifi_queues[0] if hand == "L" else sl.wifi_queues[2]
                    lane_wps = [get_sensor_zone(step.areas[0]).center for step in lane_q]
                    xy = interpolate_waypoints(lane_wps, prog)
                else:
                    xy = interpolate_waypoints(sl.slide_waypoints_mm, prog)
                return ((xy[0], xy[1], Z_CONTACT_MM), "Slide", 0.0, (sl.sensor_zone,))

        for h_note, h_xy, h_zone in holds_by_hand[hand]:
            if h_note.time - 0.010 <= t <= h_note.end_time + 0.010:
                return ((h_xy[0], h_xy[1], Z_CONTACT_MM), "Hold", 0.0, (h_zone,))

        t0, xy0, m0, z0, hz0 = keyframes[idx]
        if idx + 1 < len(keyframes):
            t1, xy1, m1, z1, hz1 = keyframes[idx + 1]
            span = max(1e-4, t1 - t0)
            u = _smoothstep((t - t0) / span)
            x = xy0[0] + u * (xy1[0] - xy0[0])
            y = xy0[1] + u * (xy1[1] - xy0[1])

            # If sweeping adjacent buttons (`Sweep`) or rubbing adjacent touches (`Rub`), stay on screen (z=0)
            # ONLY when moving between adjacent sensors (`span <= 0.085s` and `step_dist <= 265mm`);
            # When resetting across the screen for the next hemisphere relay (`span > 0.085s`), lift into Hover!
            step_dist_xy = math.hypot(xy1[0] - xy0[0], xy1[1] - xy0[1])
            if (m1 == "Sweep" and span <= 0.085 and step_dist_xy <= 265.0) or (m1 == "Rub" and span <= 0.20 and step_dist_xy <= 220.0):
                return ((x, y, Z_CONTACT_MM), m1, hz1, z1 if u > 0.5 else z0)

            dist_nearest = min(abs(t - t0), abs(t1 - t))
            if dist_nearest <= 0.018:
                is_closer_to_1 = abs(t1 - t) < abs(t - t0)
                return (
                    (xy1[0] if is_closer_to_1 else xy0[0], xy1[1] if is_closer_to_1 else xy0[1], Z_CONTACT_MM),
                    m1 if is_closer_to_1 else m0,
                    hz1 if is_closer_to_1 else hz0,
                    z1 if is_closer_to_1 else z0,
                )
            peak_z = Z_TAP_REBOUND_MM if span <= 0.25 else Z_HOVER_DEFAULT_MM
            phase = max(0.0, min(1.0, (t - t0) / span))
            z = peak_z * math.sin(math.pi * phase)
            mode_now = m1 if phase > 0.5 else ("Hover" if z > Z_RELEASE_MM else m0)
            return ((x, y, z), mode_now, hz1 if phase > 0.5 else hz0, ())

        dt_after = t - t0
        if dt_after <= 0.018:
            return ((xy0[0], xy0[1], Z_CONTACT_MM), m0, hz0, z0)
        lift_u = _smoothstep(min(1.0, (dt_after - 0.018) / 0.18))
        z = Z_CONTACT_MM + lift_u * Z_HOVER_DEFAULT_MM
        return ((xy0[0], xy0[1], z), "Hover" if z > Z_RELEASE_MM else m0, 0.0, ())

    frames: List[TrajectoryFrame] = []
    prev_torso = init_torso
    prev_l_xy_frame: Point2D = init_l_xy
    prev_r_xy_frame: Point2D = init_r_xy
    from sim_core.config import TORSO_DX_LIMIT_MM, TORSO_DY_LIMIT_MM, TORSO_YAW_LIMIT_DEG

    for f_idx in range(n_frames):
        t = min(t_end, t_start + f_idx * dt)
        l_xyz, l_mode, l_hz, l_zones = _eval_hand_at_time("L", t, l_keyframes, l_kf_times)
        r_xyz, r_mode, r_hz, r_zones = _eval_hand_at_time("R", t, r_keyframes, r_kf_times)
        # Dual-Slide Continuous Branch Tracking & Meeting-Point Uncrossing ("双滑条交会顺手分流 + 防飞手"):
        # 1) When two active slides are far apart (`dist > 95mm`, e.g. `1qq2 / 5qq6` at 120.0s), each hand
        #    strictly tracks the branch closest to its previous frame position (never teleporting mid-slide!).
        # 2) When two active slides meet at a shared crossing point (`dist <= 95mm`, e.g. `4b>5` + `5b<4` at 48.4s),
        #    uncross them at the meeting point so Left hand takes the `min_x` branch and Right hand takes `max_x`!
        if l_mode == "Slide" and r_mode == "Slide":
            slide_sep = math.hypot(l_xyz[0] - r_xyz[0], l_xyz[1] - r_xyz[1])
            d_keep = (
                math.hypot(l_xyz[0] - prev_l_xy_frame[0], l_xyz[1] - prev_l_xy_frame[1])
                + math.hypot(r_xyz[0] - prev_r_xy_frame[0], r_xyz[1] - prev_r_xy_frame[1])
            )
            d_swap = (
                math.hypot(r_xyz[0] - prev_l_xy_frame[0], r_xyz[1] - prev_l_xy_frame[1])
                + math.hypot(l_xyz[0] - prev_r_xy_frame[0], l_xyz[1] - prev_r_xy_frame[1])
            )
            if slide_sep <= 95.0 and abs(d_keep - d_swap) <= 45.0:
                if l_xyz[0] > r_xyz[0] + 1.0:
                    l_xyz, r_xyz = r_xyz, l_xyz
                    l_zones, r_zones = r_zones, l_zones
            elif d_swap + 15.0 < d_keep:
                l_xyz, r_xyz = r_xyz, l_xyz
                l_zones, r_zones = r_zones, l_zones
        prev_l_xy_frame = (l_xyz[0], l_xyz[1])
        prev_r_xy_frame = (r_xyz[0], r_xyz[1])

        # Fast analytical coupled torso + 2-link arm IK for high-FPS continuous trajectory
        lx, ly, lz = l_xyz
        rx, ry, rz = r_xyz
        lw = 1.35 if lz <= Z_RELEASE_MM else 0.65
        rw = 1.35 if rz <= Z_RELEASE_MM else 0.65
        mx = (lw * lx + rw * rx) / (lw + rw)
        my = (lw * ly + rw * ry) / (lw + rw)
        opt_dx = max(TORSO_DX_LIMIT_MM[0], min(TORSO_DX_LIMIT_MM[1], mx * 0.42))
        opt_dy = max(TORSO_DY_LIMIT_MM[0], min(TORSO_DY_LIMIT_MM[1], my * 0.26 - 10.0))
        opt_yaw = max(
            TORSO_YAW_LIMIT_DEG[0],
            min(TORSO_YAW_LIMIT_DEG[1], (max(0.0, lx + 25.0) - max(0.0, 25.0 - rx)) * 0.14 + (ly - ry) * 0.045),
        )
        alpha_f = min(1.0, max(0.12, dt * profile.torso_response_rate * 2.2))
        cur_dx = prev_torso.dx_mm + alpha_f * (opt_dx - prev_torso.dx_mm)
        cur_dy = prev_torso.dy_mm + alpha_f * (opt_dy - prev_torso.dy_mm)
        cur_yaw = prev_torso.yaw_deg + alpha_f * (opt_yaw - prev_torso.yaw_deg)
        torso = TorsoState(
            dx_mm=cur_dx,
            dy_mm=cur_dy,
            yaw_deg=cur_yaw,
            vx_mm_s=(cur_dx - prev_torso.dx_mm) / dt,
            vy_mm_s=(cur_dy - prev_torso.dy_mm) / dt,
            yaw_vel_deg_s=(cur_yaw - prev_torso.yaw_deg) / dt,
        )
        prev_torso = torso

        b_pos = max(0, bisect.bisect_right(bar_starts, t) - 1)
        cur_bar = chart.bar_timestamps[b_pos][0] if chart.bar_timestamps else 0
        cur_bpm = chart.initial_bpm

        if fps >= 45:
            cand_notes = notes_by_sec.get(int(math.floor(t)), ())
            active_zones = tuple(
                n.sensor_zone
                for n in cand_notes
                if (n.note_type in (NoteType.TAP, NoteType.TOUCH) and abs(n.time - t) <= 0.035)
                or (n.note_type in (NoteType.HOLD, NoteType.TOUCH_HOLD) and n.time <= t <= n.end_time)
                or (n.note_type == NoteType.SLIDE and n.shoot_time <= t <= n.end_time)
            )
            active_unr = tuple(
                u for u in unreasonable_events if abs(u.time - t) <= 0.18
            )
        else:
            active_zones = ()
            active_unr = ()

        frames.append(
            TrajectoryFrame(
                time=round(t, 5),
                bar=cur_bar,
                bpm=cur_bpm,
                left_mode=l_mode,
                right_mode=r_mode,
                left_freq_hz=round(l_hz, 2),
                right_freq_hz=round(r_hz, 2),
                left_zones=l_zones,
                right_zones=r_zones,
                active_note_zones=active_zones,
                active_unreasonable=active_unr,
                left_palm_xyz=l_xyz,
                right_palm_xyz=r_xyz,
                _torso=torso,
                _body=None,
            )
        )

    return frames
