"""Choreography (Bimanual + Upper-Body Motion) Feature Extractor & Trajectory Exporter.

Quantifies "how good the chart feels to dance":
  1. Frame-level (>=20 fps) 3D trajectory export (`export_trajectory` -> dict or `.npz`).
  2. Per-bar choreography features: speed/acceleration/jerk smoothness, travel distance,
     path curvature & circular sweep ratio, L-R hand alternation & rhythm regularity,
     cross-hand & forearm layering, sweep/split/rub/one-stroke technique shares,
     torso sway & waist yaw amplitude, bilateral mirror & rotational symmetry,
     motion novelty vs exact-bar loop repetition, audio-recurrence motion echo
     (invariant to L-R mirror and 8-way rotation!), and cumulative stamina fatigue.
  3. Music-coupling metrics: loudness/onset correlation, sub-bar strike-onset sync,
     chorus-vs-verse movement amplitude & circularity contrast, and calm-section convergence.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from sim_core.audio_coupling import MEL_FPS, AudioCouplingData, extract_audio_coupling
from sim_core.config import Z_RELEASE_MM
from sim_core.kinematics import compute_arm_layers_batch_njit
from sim_core.maidata_parser import NoteType, ParsedChart
from sim_core.planner import PlanResult, TrajectoryFrame


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or len(b) != len(a):
        return 0.0
    sa = float(np.std(a))
    sb = float(np.std(b))
    if sa < 1e-6 or sb < 1e-6:
        return 0.0
    c = float(np.mean((a - np.mean(a)) * (b - np.mean(b))) / (sa * sb))
    if math.isnan(c) or math.isinf(c):
        return 0.0
    return max(-1.0, min(1.0, c))


def _unit_vec(u: np.ndarray) -> np.ndarray:
    n = math.sqrt(float(np.dot(u, u)))
    return (u / n) if n > 1e-7 else np.zeros_like(u)


def _fast_p90(arr: np.ndarray) -> float:
    k = len(arr)
    if k == 0:
        return 0.0
    if k == 1:
        return float(arr[0])
    idx = int(0.9 * (k - 1))
    return float(np.partition(arr, idx)[idx])


def extract_trajectory_arrays(plan: PlanResult) -> Dict[str, np.ndarray]:
    """Convert `plan.trajectory_frames` (>=20 fps) into contiguous NumPy arrays."""
    frames: Sequence[TrajectoryFrame] = plan.trajectory_frames
    n = len(frames)
    if n == 0:
        return {
            "time": np.zeros(0, dtype=np.float32),
            "bar": np.zeros(0, dtype=np.int32),
            "bpm": np.zeros(0, dtype=np.float32),
            "left_xyz": np.zeros((0, 3), dtype=np.float32),
            "right_xyz": np.zeros((0, 3), dtype=np.float32),
            "left_vel_xyz": np.zeros((0, 3), dtype=np.float32),
            "right_vel_xyz": np.zeros((0, 3), dtype=np.float32),
            "left_speed": np.zeros(0, dtype=np.float32),
            "right_speed": np.zeros(0, dtype=np.float32),
            "left_accel": np.zeros(0, dtype=np.float32),
            "right_accel": np.zeros(0, dtype=np.float32),
            "left_zone": np.zeros(0, dtype="<U8"),
            "right_zone": np.zeros(0, dtype="<U8"),
            "left_contact": np.zeros(0, dtype=bool),
            "right_contact": np.zeros(0, dtype=bool),
            "torso_dx_dy_yaw": np.zeros((0, 3), dtype=np.float32),
            "arm_layer": np.zeros(0, dtype=np.int8),
            "forearm_dist_3d": np.zeros(0, dtype=np.float32),
            "left_tech": np.zeros(0, dtype="<U16"),
            "right_tech": np.zeros(0, dtype="<U16"),
        }

    t_arr = np.empty(n, dtype=np.float32)
    bar_arr = np.empty(n, dtype=np.int32)
    bpm_arr = np.empty(n, dtype=np.float32)
    l_xyz = np.empty((n, 3), dtype=np.float32)
    r_xyz = np.empty((n, 3), dtype=np.float32)
    torso_arr = np.empty((n, 3), dtype=np.float32)
    l_zone = np.empty(n, dtype="<U8")
    r_zone = np.empty(n, dtype="<U8")
    l_tech = np.empty(n, dtype="<U16")
    r_tech = np.empty(n, dtype="<U16")

    for i, fr in enumerate(frames):
        t_arr[i] = fr.time
        bar_arr[i] = fr.bar
        bpm_arr[i] = fr.bpm
        lp = fr.left_palm_xyz
        rp = fr.right_palm_xyz
        l_xyz[i, 0], l_xyz[i, 1], l_xyz[i, 2] = lp[0], lp[1], lp[2]
        r_xyz[i, 0], r_xyz[i, 1], r_xyz[i, 2] = rp[0], rp[1], rp[2]
        ts = fr._torso
        if ts is not None:
            torso_arr[i, 0], torso_arr[i, 1], torso_arr[i, 2] = ts.dx_mm, ts.dy_mm, ts.yaw_deg
        else:
            torso_arr[i] = 0.0
        l_zone[i] = fr.left_zones[0] if fr.left_zones else ""
        r_zone[i] = fr.right_zones[0] if fr.right_zones else ""
        l_tech[i] = fr.left_mode
        r_tech[i] = fr.right_mode

    # Velocities & accelerations via central/forward differences
    l_vel = np.zeros_like(l_xyz)
    r_vel = np.zeros_like(r_xyz)
    if n >= 2:
        dts = np.maximum(1e-3, np.diff(t_arr))[:, None]
        l_vel[1:] = np.diff(l_xyz, axis=0) / dts
        r_vel[1:] = np.diff(r_xyz, axis=0) / dts
        l_vel[0] = l_vel[1]
        r_vel[0] = r_vel[1]
    l_speed = np.linalg.norm(l_vel[:, :2], axis=1).astype(np.float32)
    r_speed = np.linalg.norm(r_vel[:, :2], axis=1).astype(np.float32)

    l_accel = np.zeros(n, dtype=np.float32)
    r_accel = np.zeros(n, dtype=np.float32)
    if n >= 3:
        dts_1d = np.maximum(1e-3, np.diff(t_arr))
        l_accel[1:] = np.abs(np.diff(l_speed)) / dts_1d
        r_accel[1:] = np.abs(np.diff(r_speed)) / dts_1d

    l_contact = l_xyz[:, 2] <= (Z_RELEASE_MM + 1.0)
    r_contact = r_xyz[:, 2] <= (Z_RELEASE_MM + 1.0)
    arm_layer, f_dist = compute_arm_layers_batch_njit(l_xyz, r_xyz, torso_arr)

    return {
        "time": t_arr,
        "bar": bar_arr,
        "bpm": bpm_arr,
        "left_xyz": l_xyz,
        "right_xyz": r_xyz,
        "left_vel_xyz": l_vel,
        "right_vel_xyz": r_vel,
        "left_speed": l_speed,
        "right_speed": r_speed,
        "left_accel": l_accel,
        "right_accel": r_accel,
        "left_zone": l_zone,
        "right_zone": r_zone,
        "left_contact": l_contact,
        "right_contact": r_contact,
        "torso_dx_dy_yaw": torso_arr,
        "arm_layer": arm_layer,
        "forearm_dist_3d": f_dist,
        "left_tech": l_tech,
        "right_tech": r_tech,
    }


def export_trajectory(
    result_or_plan: Any,
    out_path: Optional[str | Path] = None,
    compressed: bool = True,
) -> Dict[str, np.ndarray]:
    """Export continuous (>=20 fps) 3D hand/body trajectory arrays, optionally saving to `.npz`."""
    plan: Optional[PlanResult] = (
        result_or_plan
        if isinstance(result_or_plan, PlanResult)
        else getattr(result_or_plan, "plan_result", None)
    )
    if plan is None:
        raise ValueError("`export_trajectory` requires a `ChartEvaluationResult` (with `plan_result`) or `PlanResult`.")
    arrays = extract_trajectory_arrays(plan)
    if out_path is not None:
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        if compressed:
            np.savez_compressed(p, **arrays)
        else:
            np.savez(p, **arrays)
    return arrays


def _build_bar_motion_signatures(
    l_xy: np.ndarray,
    r_xy: np.ndarray,
    l_c: np.ndarray,
    r_c: np.ndarray,
    circ_ratio: float,
    alt_rate: float,
    sym_mir: float,
    sweep_r: float,
    slide_r: float,
    torso_yaw_amp: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct:
      - `sig_exact` [22]: exact 8-sector L + 8-sector R angular contact/presence histogram + 6 kinematics
      - `sig_mirror` [22]: horizontally mirrored (`x -> -x`, L <-> R swapped) signature
      - `sig_rot_inv` [14]: 8-way rotation & mirror invariant Fourier-magnitude + kinematic signature
    """
    # 8 angular sectors clockwise from +Y (matching buttons 1..8)
    def _sector_hist(xy: np.ndarray, contact: np.ndarray) -> np.ndarray:
        if len(xy) == 0:
            return np.zeros(8, dtype=np.float32)
        w = np.where(contact, 1.0, 0.2).astype(np.float32)
        ang = (np.degrees(np.arctan2(xy[:, 0], xy[:, 1])) % 360.0)
        sec = np.clip((ang / 45.0).astype(np.int32), 0, 7)
        h = np.bincount(sec, weights=w, minlength=8).astype(np.float32)
        s = float(h.sum())
        return h / s if s > 1e-6 else h

    hl = _sector_hist(l_xy, l_c)
    hr = _sector_hist(r_xy, r_c)
    # Horizontal mirror maps sector k (0..7 = btns 1..8) -> 7 - k (1<->8, 2<->7, 3<->6, 4<->5) and swaps L<->R
    hl_mir = hr[::-1].copy()
    hr_mir = hl[::-1].copy()

    kin = np.array(
        [
            circ_ratio,
            alt_rate,
            max(0.0, sym_mir),
            sweep_r,
            slide_r,
            min(1.0, torso_yaw_amp / 30.0),
        ],
        dtype=np.float32,
    )
    sig_exact = np.concatenate([hl, hr, kin])
    sig_mirror = np.concatenate([hl_mir, hr_mir, kin])

    h_tot = 0.5 * (hl + hr)
    fft_mag = np.abs(np.fft.rfft(h_tot)).astype(np.float32)  # [5] rotation-invariant circular spectrum!
    rad_l = float(np.mean(np.hypot(l_xy[:, 0], l_xy[:, 1])) / 310.0) if len(l_xy) else 0.5
    rad_r = float(np.mean(np.hypot(r_xy[:, 0], r_xy[:, 1])) / 310.0) if len(r_xy) else 0.5
    rad_sym = np.array([min(rad_l, rad_r), max(rad_l, rad_r), abs(rad_l - rad_r)], dtype=np.float32)
    sig_inv = np.concatenate([fft_mag, rad_sym, kin])
    return sig_exact, sig_mirror, sig_inv


def compute_motion_features(
    plan: PlanResult,
    audio_data: Optional[AudioCouplingData] = None,
    slot: int = 5,
    mel_override: Optional[np.ndarray] = None,
    audio_path_override: Optional[str] = None,
    prior_bar_sigs: Optional[Sequence[Sequence[float]]] = None,
    initial_cum_fatigue: float = 0.0,
) -> Tuple[Dict[str, Any], Tuple[Tuple[float, ...], ...], float]:
    """Compute per-bar choreography features, whole-chart summary, and music-coupling metrics.

    Returns `(motion_dict, updated_prior_bar_sigs, final_cum_fatigue)`.
    """
    chart: ParsedChart = plan.chart
    profile = plan.profile
    traj = extract_trajectory_arrays(plan)

    bar_ts = chart.bar_timestamps
    if not bar_ts:
        max_b = max((s.bar for s in plan.planned_steps), default=0)
        b_len = 240.0 / max(1.0, chart.initial_bpm)
        bar_ts = [(b, b * b_len, (b + 1) * b_len) for b in range(max_b + 1)]

    # When evaluating a slice/window (e.g. bars 16..19), restrict `active_bar_ts` to bars covered by `t_arr`
    t_arr = traj["time"]
    if len(t_arr) >= 2 and plan.planned_steps:
        min_b_in_plan = min(s.bar for s in plan.planned_steps)
        max_b_in_plan = max(s.bar for s in plan.planned_steps)
        if min_b_in_plan > 0 and t_arr[0] > bar_ts[0][2] + 0.05:
            bar_ts_eval = [bt for bt in bar_ts if min_b_in_plan <= bt[0] <= max_b_in_plan]
        else:
            bar_ts_eval = list(bar_ts)
    else:
        bar_ts_eval = list(bar_ts)

    if audio_data is None:
        audio_data = extract_audio_coupling(
            bar_timestamps=bar_ts_eval,
            source_path=chart.source_path,
            title=chart.title,
            slot=slot,
            mel_override=mel_override,
            audio_path_override=audio_path_override,
        )

    # Group notes & planned steps by bar index
    notes_by_bar: Dict[int, List] = {}
    for n in chart.notes:
        notes_by_bar.setdefault(n.bar, []).append(n)
    steps_by_bar: Dict[int, List] = {}
    for st in plan.planned_steps:
        steps_by_bar.setdefault(st.bar, []).append(st)

    # Detect One-Stroke slide chains (`一笔画`: slide ending at btn K right as next slide starts/shoots from K)
    slides_sorted = sorted([n for n in chart.notes if n.note_type == NoteType.SLIDE], key=lambda x: x.shoot_time)
    one_stroke_bars: Dict[int, int] = {}
    for idx_s in range(1, len(slides_sorted)):
        s_prev = slides_sorted[idx_s - 1]
        s_curr = slides_sorted[idx_s]
        end_b_prev = s_prev.slide_segments[-1].end_btn if s_prev.slide_segments else -1
        if end_b_prev == s_curr.button and (
            abs(s_curr.shoot_time - s_prev.end_time) <= 0.22 or abs(s_curr.time - s_prev.end_time) <= 0.22
        ):
            one_stroke_bars[s_curr.bar] = one_stroke_bars.get(s_curr.bar, 0) + 1

    l_xyz = traj["left_xyz"]
    r_xyz = traj["right_xyz"]
    l_vel = traj["left_vel_xyz"]
    r_vel = traj["right_vel_xyz"]
    l_speed = traj["left_speed"]
    r_speed = traj["right_speed"]
    l_accel = traj["left_accel"]
    r_accel = traj["right_accel"]
    l_contact = traj["left_contact"]
    r_contact = traj["right_contact"]
    torso_arr = traj["torso_dx_dy_yaw"]
    arm_layer = traj["arm_layer"]
    l_tech = traj["left_tech"]
    r_tech = traj["right_tech"]

    history_exact_sigs: List[np.ndarray] = [
        _unit_vec(np.asarray(s, dtype=np.float32)) for s in (prior_bar_sigs or ())
    ]
    history_mirror_sigs: List[np.ndarray] = []
    history_inv_sigs: List[np.ndarray] = []

    per_bar_rows: List[Dict[str, Any]] = []
    cum_fatigue = float(initial_cum_fatigue)
    bar_amps_history: List[float] = []

    for local_i, (b_idx, b_t0, b_t1) in enumerate(bar_ts_eval):
        b_dur = max(0.25, b_t1 - b_t0)
        f0 = int(np.searchsorted(t_arr, b_t0, side="left"))
        f1 = int(np.searchsorted(t_arr, b_t1, side="right"))
        b_notes = notes_by_bar.get(b_idx, [])
        b_steps = steps_by_bar.get(b_idx, [])
        n_notes_b = sum(1 for x in b_notes if not x.is_star_head or not any(y.note_type == NoteType.SLIDE and abs(y.time - x.time) < 1e-3 and y.button == x.button for y in b_notes))
        n_notes_raw = len(b_notes)

        if f1 - f0 >= 2:
            sl_l = l_xyz[f0:f1, :2]
            sl_r = r_xyz[f0:f1, :2]
            vl = l_vel[f0:f1, :2]
            vr = r_vel[f0:f1, :2]
            spl = l_speed[f0:f1]
            spr = r_speed[f0:f1]
            acl = l_accel[f0:f1]
            acr = r_accel[f0:f1]
            cl = l_contact[f0:f1]
            cr = r_contact[f0:f1]
            tr_b = torso_arr[f0:f1]
            lay_b = arm_layer[f0:f1]
            lt_b = l_tech[f0:f1]
            rt_b = r_tech[f0:f1]
            t_sub = t_arr[f0:f1]

            dl_xy = np.diff(sl_l, axis=0)
            dr_xy = np.diff(sl_r, axis=0)
            step_dl = np.hypot(dl_xy[:, 0], dl_xy[:, 1])
            step_dr = np.hypot(dr_xy[:, 0], dr_xy[:, 1])
            dist_l = float(step_dl.sum())
            dist_r = float(step_dr.sum())
            dist_tot = dist_l + dist_r
            sp_both = 0.5 * (spl + spr)
            sp_max_pair = np.maximum(spl, spr)
            ac_both = 0.5 * (acl + acr)
            speed_mean = float(sp_both.mean())
            speed_p90 = _fast_p90(sp_max_pair)
            speed_max = float(sp_max_pair.max())
            accel_mean = float(ac_both.mean())
            accel_p90 = _fast_p90(np.maximum(acl, acr))
            dt_f = max(1e-3, float((t_sub[-1] - t_sub[0]) / max(1, len(t_sub) - 1)))
            jerk_val = float(np.mean(np.abs(np.diff(ac_both))) / dt_f) if len(ac_both) >= 3 else 0.0
            jerk_norm = jerk_val / max(200.0, speed_mean * 25.0)

            # Curvature / Tortuosity over ~0.25s chunks & Tangential Circular Motion Ratio
            chunk = max(2, int(round(0.25 / dt_f)))
            curv_vals: List[float] = []
            for xy_h, st_d in ((sl_l, step_dl), (sl_r, step_dr)):
                for k0 in range(0, len(xy_h) - chunk, max(1, chunk // 2)):
                    k1_c = min(len(xy_h) - 1, k0 + chunk)
                    arc_len = float(st_d[k0:k1_c].sum())
                    chord_len = math.hypot(float(xy_h[k1_c, 0] - xy_h[k0, 0]), float(xy_h[k1_c, 1] - xy_h[k0, 1]))
                    if arc_len > 35.0:
                        curv_vals.append(min(3.0, arc_len / max(15.0, chord_len) - 1.0))
            path_curvature = (sum(curv_vals) / len(curv_vals)) if curv_vals else 0.0

            # Tangential vs radial velocity around screen center (0, 0) for r > 80mm
            tang_e = 0.0
            tot_e = 1e-6
            ang_mom_sum = 0.0
            ang_mom_abs = 1e-6
            for xy_h, v_h in ((sl_l, vl), (sl_r, vr)):
                radii = np.hypot(xy_h[:, 0], xy_h[:, 1])
                mask_r = radii > 80.0
                if np.any(mask_r):
                    rx_u = xy_h[mask_r, 0] / radii[mask_r]
                    ry_u = xy_h[mask_r, 1] / radii[mask_r]
                    vx_m = v_h[mask_r, 0]
                    vy_m = v_h[mask_r, 1]
                    v_rad = vx_m * rx_u + vy_m * ry_u
                    v_tan = -vx_m * ry_u + vy_m * rx_u
                    tang_e += float(np.sum(v_tan * v_tan))
                    tot_e += float(np.sum(v_tan * v_tan + v_rad * v_rad))
                    ang_mom_sum += float(np.sum(v_tan))
                    ang_mom_abs += float(np.sum(np.abs(v_tan)))
            circular_ratio = tang_e / tot_e
            angular_coh = abs(ang_mom_sum) / ang_mom_abs

            # Cross-hand & layering
            cross_layer_ratio = float(np.mean(lay_b != 0))
            slide_active_ratio = float(np.mean((lt_b == "Slide") | (rt_b == "Slide")))
            hold_active_ratio = float(np.mean((lt_b == "Hold") | (rt_b == "Hold")))

            # Torso amplitude
            torso_dx_amp = float(tr_b[:, 0].max() - tr_b[:, 0].min())
            torso_dy_amp = float(tr_b[:, 1].max() - tr_b[:, 1].min())
            torso_yaw_amp = float(tr_b[:, 2].max() - tr_b[:, 2].min())
            torso_sway_energy = float(np.mean(np.abs(np.diff(tr_b[:, 0]))) / dt_f * 0.01 + np.mean(np.abs(np.diff(tr_b[:, 2]))) / dt_f * 0.1)

            # Choreographic Symmetries:
            # 1) Simultaneous bilateral horizontal mirror: L(t) vs (-Rx(t), Ry(t))
            c_x_mir = _safe_corr(sl_l[:, 0], -sl_r[:, 0])
            c_y_mir = _safe_corr(sl_l[:, 1], sl_r[:, 1])
            pos_mir = float(np.exp(-np.mean(np.hypot(sl_l[:, 0] + sl_r[:, 0], sl_l[:, 1] - sl_r[:, 1])) / 180.0))
            sym_mirror_simul = max(0.0, 0.45 * max(0.0, 0.5 * (c_x_mir + c_y_mir)) + 0.55 * pos_mir)

            # 2) Half-bar call-and-response mirror echo: first half [0, M) L/R vs second half [M, 2M) mirrored R/L
            mid = (f1 - f0) // 2
            if mid >= 3:
                h1_l, h1_r = sl_l[:mid], sl_r[:mid]
                h2_l, h2_r = sl_l[mid : 2 * mid], sl_r[mid : 2 * mid]
                d_echo = 0.5 * (
                    float(np.mean(np.hypot(h1_l[:, 0] + h2_r[:, 0], h1_l[:, 1] - h2_r[:, 1])))
                    + float(np.mean(np.hypot(h1_r[:, 0] + h2_l[:, 0], h1_r[:, 1] - h2_l[:, 1])))
                )
                sym_mirror_echo = float(np.exp(-d_echo / 175.0))
                # 3) Point / 180-deg rotational symmetry: L(t) vs -R(t) or half-bar 180-deg rotation
                d_rot_sim = float(np.mean(np.hypot(sl_l[:, 0] + sl_r[:, 0], sl_l[:, 1] + sl_r[:, 1])))
                d_rot_echo = 0.5 * (
                    float(np.mean(np.hypot(h1_l[:, 0] + h2_r[:, 0], h1_l[:, 1] + h2_r[:, 1])))
                    + float(np.mean(np.hypot(h1_r[:, 0] + h2_l[:, 0], h1_r[:, 1] + h2_l[:, 1])))
                )
                sym_rotational = float(max(np.exp(-d_rot_sim / 180.0), np.exp(-d_rot_echo / 180.0)))
            else:
                sym_mirror_echo = 0.0
                sym_rotational = 0.0

            bar_amp = float(np.mean(np.hypot(sl_l[:, 0], sl_l[:, 1]) + np.hypot(sl_r[:, 0], sl_r[:, 1])))
        else:
            sl_l = np.zeros((0, 2), dtype=np.float32)
            sl_r = np.zeros((0, 2), dtype=np.float32)
            cl = np.zeros(0, dtype=bool)
            cr = np.zeros(0, dtype=bool)
            dist_l = dist_r = dist_tot = speed_mean = speed_p90 = speed_max = 0.0
            accel_mean = accel_p90 = jerk_norm = path_curvature = circular_ratio = angular_coh = 0.0
            cross_layer_ratio = slide_active_ratio = hold_active_ratio = 0.0
            torso_dx_amp = torso_dy_amp = torso_yaw_amp = torso_sway_energy = 0.0
            sym_mirror_simul = sym_mirror_echo = sym_rotational = 0.0
            bar_amp = 0.0

        # Event-step level features in bar `b_idx`
        n_st = len(b_steps)
        single_hands = [
            ("L" if s.left.note_ids else "R")
            for s in b_steps
            if bool(s.left.note_ids) ^ bool(s.right.note_ids)
        ]
        if len(single_hands) >= 2:
            alternation_rate = sum(
                1 for k in range(1, len(single_hands)) if single_hands[k] != single_hands[k - 1]
            ) / float(len(single_hands) - 1)
        else:
            alternation_rate = 0.0

        if n_st >= 3:
            iois = np.diff([s.time for s in b_steps])
            iois = iois[iois > 1e-3]
            rhythm_reg = float(1.0 / (1.0 + (np.std(iois) / max(1e-3, np.mean(iois))))) if len(iois) >= 2 else 1.0
        else:
            rhythm_reg = 1.0 if n_st > 0 else 0.0

        cross_hand_count = sum(
            1 for s in b_steps
            if (s.left.note_ids and s.left.target_xy[0] > 35.0)
            or (s.right.note_ids and s.right.target_xy[0] < -35.0)
            or s.body_state.arm_layer_order != "NORMAL"
        )
        sweep_cnt = sum(
            1 for s in b_steps
            if s.decomposition_used == "RUB_SWEEP" or s.left.mode == "Sweep" or s.right.mode == "Sweep"
        )
        split_cnt = sum(
            1 for s in b_steps
            if s.decomposition_used in ("OUTER_INNER_SPLIT", "ALTERNATING_TAP")
        )
        rub_cnt = sum(
            1 for s in b_steps
            if s.decomposition_used == "SINGLE_PALM_RUB" or s.left.mode == "Rub" or s.right.mode == "Rub"
        )
        denom_st = float(max(1, n_st))
        sweep_ratio = sweep_cnt / denom_st
        split_ratio = split_cnt / denom_st
        rub_ratio = rub_cnt / denom_st
        one_stroke_cnt = one_stroke_bars.get(b_idx, 0)
        hand_balance = 1.0 - abs(dist_l - dist_r) / max(50.0, dist_tot) if dist_tot > 20.0 else 1.0

        # Bar motion signatures -> Novelty, Exact-Repeat Loop Detection, & Audio-Matched Motion Echo
        sig_ex, sig_mir, sig_inv = _build_bar_motion_signatures(
            sl_l, sl_r, cl, cr,
            circular_ratio, alternation_rate, sym_mirror_simul,
            sweep_ratio, slide_active_ratio, torso_yaw_amp,
        )
        sig_ex_u = _unit_vec(sig_ex)
        sig_mir_u = _unit_vec(sig_mir)
        sig_inv_u = _unit_vec(sig_inv)
        if history_exact_sigs and n_notes_raw > 0:
            h_mat = np.stack(history_exact_sigs, axis=0)
            sims_vec = h_mat @ sig_ex_u
            motion_novelty = max(0.0, 1.0 - float(sims_vec.max()))
            motion_exact_repeat = float(sims_vec[-min(4, len(sims_vec)):].max())
        else:
            motion_novelty = 1.0 if n_notes_raw > 0 else 0.0
            motion_exact_repeat = 0.0

        # Audio-coupled repetition echo (including L-R mirror & rotational equivalence!)
        audio_loud = float(audio_data.bar_loudness_db[local_i]) if audio_data.has_audio and local_i < len(audio_data.bar_loudness_db) else 0.0
        audio_flux = float(audio_data.bar_onset_flux[local_i]) if audio_data.has_audio and local_i < len(audio_data.bar_onset_flux) else 0.0
        is_chorus = bool(audio_data.is_chorus_bar[local_i]) if audio_data.has_audio and local_i < len(audio_data.is_chorus_bar) else False

        audio_repeat_sim = 0.0
        motion_audio_echo: Optional[float] = None
        motion_echo_transformed: Optional[float] = None
        if audio_data.has_audio and local_i >= 4 and len(history_inv_sigs) >= 4 and n_notes_raw > 0:
            cur_ae = audio_data.bar_audio_embed[local_i]
            # Find the best matching prior musical bar at least 4 bars earlier (`0 .. local_i - 4`)
            prev_aes = audio_data.bar_audio_embed[: local_i - 3]
            a_sims = prev_aes @ cur_ae
            best_p = int(np.argmax(a_sims))
            audio_repeat_sim = float(a_sims[best_p])
            if audio_repeat_sim >= 0.80 and best_p < len(history_inv_sigs):
                ref_ex = history_exact_sigs[-len(history_inv_sigs) + best_p]
                sim_id = float(np.dot(sig_ex_u, ref_ex))
                sim_mr = float(np.dot(sig_mir_u, ref_ex))
                sim_iv = float(np.dot(sig_inv_u, history_inv_sigs[best_p]))
                motion_audio_echo = max(sim_id, sim_mr, sim_iv)
                motion_echo_transformed = max(sim_mr, sim_iv) - 0.35 * max(0.0, sim_id - 0.96)

        history_exact_sigs.append(sig_ex_u)
        history_mirror_sigs.append(sig_mir_u)
        history_inv_sigs.append(sig_inv_u)

        bar_amps_history.append(bar_amp)
        if len(bar_amps_history) >= 4:
            w4 = bar_amps_history[-4:]
            amp_expansion = (w4[-1] - w4[0]) / max(100.0, float(np.mean(w4)))
        else:
            amp_expansion = 0.0

        # Cumulative fatigue with exponential recovery
        fat_inst = (
            sum(s.step_fatigue_left + s.step_fatigue_right + s.step_fatigue_torso for s in b_steps) / denom_st
            if b_steps else 0.0
        )
        decay = math.exp(-b_dur * (profile.stamina_recovery_per_sec / 65.0))
        cum_fatigue = max(0.0, cum_fatigue * decay + fat_inst * (b_dur / 2.0))

        per_bar_rows.append(
            {
                "bar": int(b_idx),
                "n_notes": int(n_notes_b),
                "n_events": int(n_st),
                "dist_l_mm": round(dist_l, 1),
                "dist_r_mm": round(dist_r, 1),
                "dist_total_mm": round(dist_tot, 1),
                "dist_per_note_mm": round(dist_tot / max(1, n_notes_b), 2) if n_notes_b > 0 else 0.0,
                "hand_balance": round(hand_balance, 4),
                "speed_mean": round(speed_mean, 1),
                "speed_p90": round(speed_p90, 1),
                "speed_max": round(speed_max, 1),
                "accel_mean": round(accel_mean, 1),
                "accel_p90": round(accel_p90, 1),
                "jerk_norm": round(jerk_norm, 4),
                "path_curvature": round(path_curvature, 4),
                "circular_motion_ratio": round(circular_ratio, 4),
                "angular_coherence": round(angular_coh, 4),
                "alternation_rate": round(alternation_rate, 4),
                "rhythm_ioi_regularity": round(rhythm_reg, 4),
                "cross_hand_count": int(cross_hand_count),
                "cross_layer_ratio": round(cross_layer_ratio, 4),
                "sweep_ratio": round(sweep_ratio, 4),
                "split_ratio": round(split_ratio, 4),
                "rub_ratio": round(rub_ratio, 4),
                "one_stroke_count": int(one_stroke_cnt),
                "slide_active_ratio": round(slide_active_ratio, 4),
                "hold_active_ratio": round(hold_active_ratio, 4),
                "torso_dx_amp": round(torso_dx_amp, 2),
                "torso_dy_amp": round(torso_dy_amp, 2),
                "torso_yaw_amp": round(torso_yaw_amp, 2),
                "torso_sway_energy": round(torso_sway_energy, 3),
                "sym_mirror_simul": round(sym_mirror_simul, 4),
                "sym_mirror_echo": round(sym_mirror_echo, 4),
                "sym_rotational": round(sym_rotational, 4),
                "motion_novelty": round(motion_novelty, 4),
                "motion_exact_repeat": round(motion_exact_repeat, 4),
                "amplitude_expansion": round(amp_expansion, 4),
                "audio_loudness_db": round(audio_loud, 3),
                "audio_onset_flux": round(audio_flux, 4),
                "is_chorus": bool(is_chorus),
                "audio_repeat_sim": round(audio_repeat_sim, 4),
                "motion_audio_echo": round(motion_audio_echo, 4) if motion_audio_echo is not None else None,
                "motion_echo_transformed": round(motion_echo_transformed, 4) if motion_echo_transformed is not None else None,
                "fatigue_bar": round(fat_inst, 3),
                "fatigue_cum": round(cum_fatigue, 3),
            }
        )

    # Whole-chart choreography summary (computed over active bars `n_notes > 0`)
    active_rows = [r for r in per_bar_rows if r["n_notes"] > 0]
    src_rows = active_rows if active_rows else per_bar_rows

    def _m(key: str) -> float:
        vals = [float(r[key]) for r in src_rows if r.get(key) is not None]
        return round(float(np.mean(vals)), 4) if vals else 0.0

    def _p90(key: str) -> float:
        vals = [float(r[key]) for r in src_rows if r.get(key) is not None]
        return round(float(np.percentile(vals, 90)), 4) if vals else 0.0

    # Technique diversity (Shannon entropy over active step modes: Inner/Outer, Sweep, Rub, Split, Hold, Slide)
    tech_counts = np.zeros(6, dtype=np.float64)
    for st in plan.planned_steps:
        for ha in (st.left, st.right):
            if ha.mode == "Hover":
                continue
            if ha.mode == "Sweep":
                tech_counts[1] += 1
            elif ha.mode == "Rub":
                tech_counts[2] += 1
            elif ha.mode in ("OuterSplit", "InnerSplit", "AltTap"):
                tech_counts[3] += 1
            elif ha.mode == "Hold":
                tech_counts[4] += 1
            elif ha.mode == "Slide":
                tech_counts[5] += 1
            else:
                tech_counts[0] += 1
    if tech_counts.sum() > 0:
        tp = tech_counts / tech_counts.sum()
        tp = tp[tp > 0]
        tech_entropy = float(-np.sum(tp * np.log(tp)) / math.log(6.0))
    else:
        tech_entropy = 0.0

    exact_loop_bars = sum(1 for r in active_rows if r["motion_exact_repeat"] >= 0.965 and r["n_notes"] >= 4)
    exact_repeat_rate = exact_loop_bars / float(max(1, len(active_rows)))
    fat_vals = [r["fatigue_cum"] for r in per_bar_rows] if per_bar_rows else [0.0]
    fat_peak = float(max(fat_vals))
    fat_p25 = float(np.percentile(fat_vals, 25))
    fat_contrast = (fat_peak - fat_p25) / max(5.0, fat_peak)

    # Music-Coupling whole-chart metrics
    if audio_data.has_audio and len(per_bar_rows) >= 4:
        loud_arr = np.array([r["audio_loudness_db"] for r in per_bar_rows], dtype=np.float32)
        flux_arr = np.array([r["audio_onset_flux"] for r in per_bar_rows], dtype=np.float32)
        spd_arr = np.array([r["speed_mean"] for r in per_bar_rows], dtype=np.float32)
        dst_arr = np.array([r["dist_total_mm"] for r in per_bar_rows], dtype=np.float32)
        kin_arr = np.array([r["accel_mean"] for r in per_bar_rows], dtype=np.float32)

        corr_spd_loud = _safe_corr(spd_arr, loud_arr)
        corr_dst_loud = _safe_corr(dst_arr, loud_arr)
        corr_kin_flux = _safe_corr(kin_arr, flux_arr)

        # Sub-bar strike-to-onset alignment (at 200 fps!)
        onset_env = audio_data.onset_env_200fps
        if onset_env is not None and len(onset_env) > 20 and plan.planned_steps:
            strike_times = [s.time for s in plan.planned_steps if s.left.note_ids or s.right.note_ids]
            f_idxs = np.clip(np.round(np.asarray(strike_times) * MEL_FPS).astype(np.int32), 0, len(onset_env) - 1)
            # Max onset within +-1 frame (+-5ms) around each strike vs overall mean onset
            win_vals = np.maximum(
                onset_env[f_idxs],
                np.maximum(
                    onset_env[np.maximum(0, f_idxs - 1)],
                    onset_env[np.minimum(len(onset_env) - 1, f_idxs + 1)],
                ),
            )
            base_env = float(np.mean(onset_env))
            onset_strike_sync = float(np.mean(win_vals) / max(1e-4, base_env)) if len(win_vals) else 1.0
        else:
            onset_strike_sync = 1.0

        ch_rows = [r for r in per_bar_rows if r["is_chorus"] and r["n_notes"] > 0]
        vs_rows = [r for r in per_bar_rows if (not r["is_chorus"]) and r["n_notes"] > 0]
        if ch_rows and vs_rows:
            ch_dist = float(np.mean([r["dist_total_mm"] for r in ch_rows]))
            vs_dist = float(np.mean([r["dist_total_mm"] for r in vs_rows]))
            ch_spd = float(np.mean([r["speed_mean"] for r in ch_rows]))
            vs_spd = float(np.mean([r["speed_mean"] for r in vs_rows]))
            ch_tor = float(np.mean([r["torso_yaw_amp"] + 0.1 * r["torso_dx_amp"] for r in ch_rows]))
            vs_tor = float(np.mean([r["torso_yaw_amp"] + 0.1 * r["torso_dx_amp"] for r in vs_rows]))
            ch_circ = float(np.mean([r["circular_motion_ratio"] for r in ch_rows]))
            vs_circ = float(np.mean([r["circular_motion_ratio"] for r in vs_rows]))
            ch_swp = float(np.mean([r["sweep_ratio"] + r["slide_active_ratio"] for r in ch_rows]))
            vs_swp = float(np.mean([r["sweep_ratio"] + r["slide_active_ratio"] for r in vs_rows]))
            cv_dist_ratio = ch_dist / max(80.0, vs_dist)
            cv_spd_ratio = ch_spd / max(40.0, vs_spd)
            cv_tor_ratio = (ch_tor + 1.5) / (vs_tor + 1.5)
            cv_circ_diff = ch_circ - vs_circ
            cv_swp_diff = ch_swp - vs_swp
        else:
            cv_dist_ratio = cv_spd_ratio = cv_tor_ratio = 1.0
            cv_circ_diff = cv_swp_diff = 0.0

        q25_loud = float(np.percentile(loud_arr, 25))
        calm_rows = [r for r in per_bar_rows if r["audio_loudness_db"] <= q25_loud]
        calm_spd = float(np.mean([r["speed_p90"] for r in calm_rows])) if calm_rows else 0.0
        overall_spd = max(100.0, float(np.mean([r["speed_p90"] for r in src_rows])))
        calm_convergence = max(0.0, min(1.5, 1.0 - (calm_spd / overall_spd) + 0.35))
    else:
        corr_spd_loud = corr_dst_loud = corr_kin_flux = 0.0
        onset_strike_sync = 1.0
        cv_dist_ratio = cv_spd_ratio = cv_tor_ratio = 1.0
        cv_circ_diff = cv_swp_diff = 0.0
        calm_convergence = 0.5

    music_coupling = {
        "has_audio": bool(audio_data.has_audio),
        "section_source": audio_data.section_source,
        "corr_speed_loudness": round(corr_spd_loud, 4),
        "corr_dist_loudness": round(corr_dst_loud, 4),
        "corr_kinetic_onset": round(corr_kin_flux, 4),
        "onset_strike_sync": round(onset_strike_sync, 4),
        "chorus_verse_dist_ratio": round(cv_dist_ratio, 4),
        "chorus_verse_speed_ratio": round(cv_spd_ratio, 4),
        "chorus_verse_torso_ratio": round(cv_tor_ratio, 4),
        "chorus_verse_circular_diff": round(cv_circ_diff, 4),
        "chorus_verse_sweep_diff": round(cv_swp_diff, 4),
        "calm_convergence": round(calm_convergence, 4),
    }

    echo_vals = [float(r["motion_audio_echo"]) for r in active_rows if r["motion_audio_echo"] is not None]
    echo_tr_vals = [float(r["motion_echo_transformed"]) for r in active_rows if r["motion_echo_transformed"] is not None]
    motion_echo_mean = float(np.mean(echo_vals)) if echo_vals else 0.0
    motion_motif_var = float(np.mean(echo_tr_vals)) if echo_tr_vals else 0.0

    summary = {
        "n_bars": len(per_bar_rows),
        "n_active_bars": len(active_rows),
        "dist_per_bar_mm": _m("dist_total_mm"),
        "dist_per_note_mm": _m("dist_per_note_mm"),
        "hand_balance_mean": _m("hand_balance"),
        "speed_mean": _m("speed_mean"),
        "speed_p90": _p90("speed_p90"),
        "accel_mean": _m("accel_mean"),
        "jerk_norm_mean": _m("jerk_norm"),
        "path_curvature_mean": _m("path_curvature"),
        "circular_motion_ratio": _m("circular_motion_ratio"),
        "angular_coherence_mean": _m("angular_coherence"),
        "alternation_rate_mean": _m("alternation_rate"),
        "rhythm_ioi_regularity": _m("rhythm_ioi_regularity"),
        "cross_hand_per_bar": _m("cross_hand_count"),
        "cross_layer_ratio": _m("cross_layer_ratio"),
        "sweep_ratio": _m("sweep_ratio"),
        "split_ratio": _m("split_ratio"),
        "rub_ratio": _m("rub_ratio"),
        "one_stroke_total": int(sum(r["one_stroke_count"] for r in per_bar_rows)),
        "slide_active_ratio": _m("slide_active_ratio"),
        "hold_active_ratio": _m("hold_active_ratio"),
        "technique_entropy": round(tech_entropy, 4),
        "torso_dx_amp_mean": _m("torso_dx_amp"),
        "torso_yaw_amp_mean": _m("torso_yaw_amp"),
        "torso_sway_energy_mean": _m("torso_sway_energy"),
        "sym_mirror_simul_mean": _m("sym_mirror_simul"),
        "sym_mirror_echo_mean": _m("sym_mirror_echo"),
        "sym_rotational_mean": _m("sym_rotational"),
        "motion_novelty_mean": _m("motion_novelty"),
        "motion_exact_repeat_mean": _m("motion_exact_repeat"),
        "motion_exact_repeat_rate": round(exact_repeat_rate, 4),
        "amplitude_expansion_p90": _p90("amplitude_expansion"),
        "motion_audio_echo_mean": round(motion_echo_mean, 4),
        "motion_motif_variation": round(motion_motif_var, 4),
        "fatigue_cum_peak": round(fat_peak, 3),
        "fatigue_contrast": round(fat_contrast, 4),
    }

    updated_sigs = tuple(
        tuple(round(float(x), 5) for x in s) for s in history_exact_sigs[-64:]
    )
    motion_dict = {
        "fps": round(1.0 / max(1e-3, float(t_arr[1] - t_arr[0])), 1) if len(t_arr) >= 2 else 20.0,
        "n_frames": int(len(t_arr)),
        "summary": summary,
        "music_coupling": music_coupling,
        "per_bar": per_bar_rows,
    }
    return motion_dict, updated_sigs, cum_fatigue
