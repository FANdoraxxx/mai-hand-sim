"""Full-Body Biomechanical State, Joint Kinematics, Visual Burden Model, and `body_timeline` API (`simv3` Section 3 & 6.1).

Biomechanical & Ergonomic Literature References:
  1. ISO 11226:2000, "Ergonomics — Evaluation of static working postures":
     - Trunk anterior flexion (pitch): comfort [0, 20] deg, acceptable [-5, 40] deg, limit [-15, 60] deg.
     - Trunk lateral bending (roll) & axial rotation (yaw): comfort [-10, 10] / [-15, 15] deg, limit [-30, 30] / [-45, 45] deg.
     - Upper arm flexion: comfort [0, 60] deg, acceptable [-20, 110] deg; abduction: comfort [0, 45] deg, acceptable [-15, 90] deg.
  2. McAtamney, L., & Corlett, E. N. (1993), "RULA: a survey method for the investigation of work-related upper limb disorders",
     Applied Ergonomics, 24(2), 91-99:
     - Elbow flexion/interior angle: comfort [65, 125] deg (60..100 deg flexion), penalty outside [35, 160] deg.
     - Forearm pronation/supination: neutral thumb-up 0 deg, comfort [-35, +65] deg, extreme >75 deg.
     - Wrist flexion/extension: comfort [-15, +15] deg, penalty outside [-40, +45] deg.
     - Wrist radial/ulnar deviation: comfort [-10, +15] deg, extreme outside [-20, +28] deg.
  3. EN 1005-4:2005 & NASA-STD-3000 Vol. I Sec. 3.3 (Anthropometry and Biomechanics):
     - Dynamic joint angular velocity comfort and arm swivel-angle redundancy resolution.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from sim_core.config import (
    ARM_L1_MM,
    ARM_L2_MM,
    ARM_MIN_REACH_MM,
    DEFAULT_HIT_MODE,
    DEFAULT_SKILL_PROFILE,
    HEAD_NEUTRAL_Y_MM,
    HitMode,
    R_HAND_MM,
    SHOULDER_NEUTRAL_Y_MM,
    SHOULDER_NEUTRAL_Z_MM,
    W_SHOULDER_MM,
    Z_RELEASE_MM,
    SkillProfileConfig,
)
from sim_core.geometry import get_button_target_xy, get_sensor_zone
from sim_core.kinematics import _njit
from sim_core.maidata_parser import NoteType, ParsedChart, ParsedNote, parse_simai_chart
from sim_core.planner import PlanResult, plan_chart_movements
from sim_core.player_profile import get_skill_profile

# ============================================================================
# Literature-Grounded Joint Range of Motion (ROM) & Comfort Zones (Degrees)
# ============================================================================
ERGONOMIC_JOINT_SPECS: Dict[str, Dict[str, Any]] = {
    "torso_pitch_deg": {"comfort": (0.0, 20.0), "acceptable": (-5.0, 40.0), "rom": (-15.0, 60.0), "source": "ISO 11226:2000 Sec 3.2"},
    "torso_roll_deg": {"comfort": (-10.0, 10.0), "acceptable": (-20.0, 20.0), "rom": (-30.0, 30.0), "source": "ISO 11226:2000 Sec 3.2"},
    "torso_yaw_deg": {"comfort": (-15.0, 15.0), "acceptable": (-30.0, 30.0), "rom": (-45.0, 45.0), "source": "EN 1005-4:2005"},
    "shoulder_flex_deg": {"comfort": (10.0, 75.0), "acceptable": (-20.0, 115.0), "rom": (-45.0, 170.0), "source": "ISO 11226:2000 / RULA"},
    "shoulder_abd_deg": {"comfort": (0.0, 45.0), "acceptable": (-15.0, 90.0), "rom": (-35.0, 135.0), "source": "ISO 11226:2000 / RULA"},
    "shoulder_rot_deg": {"comfort": (-30.0, 45.0), "acceptable": (-60.0, 75.0), "rom": (-85.0, 90.0), "source": "NASA-STD-3000 Sec 3.3"},
    "elbow_angle_deg": {"comfort": (65.0, 125.0), "acceptable": (35.0, 160.0), "rom": (25.0, 175.0), "source": "RULA (McAtamney & Corlett 1993)"},
    "forearm_pro_sup_deg": {"comfort": (-35.0, 65.0), "acceptable": (-70.0, 80.0), "rom": (-85.0, 90.0), "source": "RULA / AAOS"},
    "wrist_flex_deg": {"comfort": (-15.0, 15.0), "acceptable": (-40.0, 45.0), "rom": (-65.0, 70.0), "source": "RULA (McAtamney & Corlett 1993)"},
    "wrist_dev_deg": {"comfort": (-10.0, 15.0), "acceptable": (-20.0, 25.0), "rom": (-25.0, 35.0), "source": "RULA (McAtamney & Corlett 1993)"},
}


@_njit(cache=True)
def _zone_strain(val: float, c_lo: float, c_hi: float, a_lo: float, a_hi: float) -> float:
    """Continuous 0..2 RULA/ISO strain: 0 inside comfort [c_lo, c_hi], 0..0.5 in acceptable, >0.5 outside acceptable."""
    if c_lo <= val <= c_hi:
        return 0.0
    if val < c_lo:
        span = max(1.0, c_lo - a_lo)
        d = (c_lo - val) / span
        return 0.35 * d if d <= 1.0 else 0.35 + 0.85 * (d - 1.0)
    span = max(1.0, a_hi - c_hi)
    d = (val - c_hi) / span
    return 0.35 * d if d <= 1.0 else 0.35 + 0.85 * (d - 1.0)


@_njit(cache=True)
def _solve_arm_7dof_with_swivel_njit(
    is_left: bool,
    sx: float, sy: float, sz: float,
    px: float, py: float, pz: float,
    vx: float, vy: float,
    other_ex: float, other_ey: float, other_ez: float,
    other_px: float, other_py: float, other_pz: float,
    has_other: bool,
    cx_torso: float, yaw_deg: float, pitch_deg: float,
):
    """Solve 7-DoF arm IK with optimal swivel angle phi in [-55..+65 deg] minimizing
    gravity + RULA joint strain + chest clearance + 3D forearm collision.
    """
    l1 = ARM_L1_MM
    l2 = ARM_L2_MM
    dx = px - sx
    dy = py - sy
    dz = pz - sz
    dist_3d = math.sqrt(dx * dx + dy * dy + dz * dz)
    max_r = l1 + l2 - 1.0
    min_r = ARM_MIN_REACH_MM
    dc = min(max_r, max(min_r, dist_3d))

    a_proj = (l1 * l1 - l2 * l2 + dc * dc) / (2.0 * dc)
    rem = l1 * l1 - a_proj * a_proj
    h_el = math.sqrt(rem) if rem > 0.0 else 0.0
    cos_b = max(-1.0, min(1.0, (l1 * l1 + l2 * l2 - dc * dc) / (2.0 * l1 * l2)))
    elbow_deg = max(25.0, min(175.0, math.degrees(math.acos(cos_b))))

    if dist_3d > 1e-6:
        ux, uy, uz = dx / dist_3d, dy / dist_3d, dz / dist_3d
    else:
        ux, uy, uz = 0.0, 0.0, -1.0

    # Orthonormal basis (n0, b0) in plane perpendicular to u=(ux,uy,uz), with n0 pointing downward (-Y)
    gx, gy, gz = (-0.25 if is_left else 0.25), -0.95, -0.12
    dot_gu = gx * ux + gy * uy + gz * uz
    n0x = gx - dot_gu * ux
    n0y = gy - dot_gu * uy
    n0z = gz - dot_gu * uz
    nn = math.sqrt(n0x * n0x + n0y * n0y + n0z * n0z)
    if nn > 1e-6:
        n0x, n0y, n0z = n0x / nn, n0y / nn, n0z / nn
    else:
        n0x, n0y, n0z = (-1.0 if is_left else 1.0), 0.0, 0.0
    # b0 = u x n0 (outward lateral-upward vector)
    b0x = uy * n0z - uz * n0y
    b0y = uz * n0x - ux * n0z
    b0z = ux * n0y - uy * n0x
    if (is_left and b0x > 0.0) or ((not is_left) and b0x < 0.0):
        b0x, b0y, b0z = -b0x, -b0y, -b0z

    mx = sx + ux * a_proj
    my = sy + uy * a_proj
    mz = sz + uz * a_proj

    # Search optimal swivel angle phi in {-30, -15, 0, +15, +30, +45, +60} deg
    best_phi = 12.0
    best_cost = 1e18
    best_ex, best_ey, best_ez = mx + n0x * h_el, my + n0y * h_el, mz + n0z * h_el
    best_s_flex = 45.0
    best_s_abd = 20.0
    best_s_rot = 0.0

    for k_p in range(-2, 5):
        phi_deg = 15.0 * k_p
        phi_rad = math.radians(phi_deg)
        cp = math.cos(phi_rad)
        sp = math.sin(phi_rad)
        nx = cp * n0x + sp * b0x
        ny = cp * n0y + sp * b0y
        nz = cp * n0z + sp * b0z
        ex = mx + nx * h_el
        ey = my + ny * h_el
        ez = mz + nz * h_el

        # Upper arm vector U = E - S
        uax, uay, uaz = ex - sx, ey - sy, ez - sz
        # Shoulder flexion (forward -Z / upward +Y angle from vertical -Y hang)
        s_flex = math.degrees(math.atan2(max(1.0, -uaz), max(-150.0, -uay))) + pitch_deg * 0.5
        # Shoulder abduction (+ outward away from torso midline, - cross-chest adduction)
        lat_out = -uax if is_left else uax
        s_abd = math.degrees(math.atan2(lat_out, max(40.0, math.hypot(uay, uaz))))
        # Forearm vector F = P - E -> Shoulder internal/external rotation
        fax, fay, faz = px - ex, py - ey, pz - ez
        s_rot = math.degrees(math.atan2((-fax if is_left else fax) * 0.6 + fay * 0.4, max(40.0, -faz))) - phi_deg * 0.55

        c_swiv = 0.0012 * (phi_deg - 12.0) * (phi_deg - 12.0)
        c_rom = (
            _zone_strain(s_flex, 10.0, 75.0, -20.0, 115.0)
            + _zone_strain(s_abd, 0.0, 45.0, -15.0, 90.0)
            + _zone_strain(s_rot, -30.0, 45.0, -60.0, 75.0)
        )
        # Chest non-penetration
        c_chest = 0.0
        if is_left and ex > cx_torso + 10.0 and ez > sz - 100.0:
            c_chest = 2.5
        elif (not is_left) and ex < cx_torso - 10.0 and ez > sz - 100.0:
            c_chest = 2.5
        # Forearm 3D clearance if other arm is already placed
        c_coll = 0.0
        if has_other:
            mid_lx, mid_ly, mid_lz = 0.5 * (ex + px), 0.5 * (ey + py), 0.5 * (ez + pz)
            mid_rx, mid_ry, mid_rz = 0.5 * (other_ex + other_px), 0.5 * (other_ey + other_py), 0.5 * (other_ez + other_pz)
            d_m = math.sqrt((mid_lx - mid_rx) ** 2 + (mid_ly - mid_ry) ** 2 + (mid_lz - mid_rz) ** 2)
            if d_m < 85.0:
                c_coll = (85.0 - d_m) * 0.04

        tot_c = c_swiv + c_rom + c_chest + c_coll
        if tot_c < best_cost:
            best_cost = tot_c
            best_phi = phi_deg
            best_ex, best_ey, best_ez = ex, ey, ez
            best_s_flex = s_flex
            best_s_abd = s_abd
            best_s_rot = s_rot

    # Forearm pronation/supination & wrist angles from forearm vector F and palm target/velocity
    fax, fay, faz = px - best_ex, py - best_ey, pz - best_ez
    fa_ang_xy = math.degrees(math.atan2(fay, fax))
    # Neutral thumb-up when palm faces inward; pressing flat on vertical screen requires ~35..55 deg pronation
    cross_x = (px - cx_torso) if is_left else (cx_torso - px)
    pro_sup = 38.0 + 0.11 * cross_x - 0.06 * py
    pro_sup = max(-85.0, min(90.0, pro_sup))

    # Palm heading: aligned with movement velocity when sweeping/sliding fast, else natural forearm heading
    spd_xy = math.hypot(vx, vy)
    if spd_xy > 350.0:
        vel_ang = math.degrees(math.atan2(vy, vx))
        diff_a = ((vel_ang - fa_ang_xy + 180.0) % 360.0) - 180.0
        palm_heading = fa_ang_xy + max(-28.0, min(28.0, diff_a * 0.45))
    else:
        palm_heading = fa_ang_xy

    # Wrist flexion/extension (extension + when palm is pressed flat against z=0 screen with forearm angled from z=sz)
    fa_len_xy = math.hypot(fax, fay)
    approach_deg = math.degrees(math.atan2(max(0.0, -faz), max(25.0, fa_len_xy)))
    wrist_flex = (approach_deg - 24.0) * 0.55 if pz <= Z_RELEASE_MM + 5.0 else -5.0
    wrist_flex = max(-60.0, min(65.0, wrist_flex))

    # Wrist radial (-)/ulnar (+) deviation from angular difference between palm target and forearm axis
    rad_ang = math.degrees(math.atan2(py, px))
    dev_raw = ((rad_ang - fa_ang_xy + 180.0) % 360.0) - 180.0
    wrist_dev = max(-25.0, min(32.0, dev_raw * 0.22))

    awk = (
        _zone_strain(best_s_flex, 10.0, 75.0, -20.0, 115.0)
        + _zone_strain(best_s_abd, 0.0, 45.0, -15.0, 90.0)
        + _zone_strain(best_s_rot, -30.0, 45.0, -60.0, 75.0)
        + _zone_strain(elbow_deg, 65.0, 125.0, 35.0, 160.0)
        + _zone_strain(pro_sup, -35.0, 65.0, -70.0, 80.0)
        + _zone_strain(wrist_flex, -15.0, 15.0, -40.0, 45.0)
        + _zone_strain(wrist_dev, -10.0, 15.0, -20.0, 25.0)
    ) / 3.5
    awk = min(1.0, awk)

    return (
        best_ex, best_ey, best_ez,
        best_s_flex, best_s_abd, best_s_rot,
        elbow_deg, best_phi, pro_sup,
        wrist_flex, wrist_dev, palm_heading, awk,
    )


@_njit(cache=True)
def _compute_body_frames_batch_njit(
    times: np.ndarray,
    l_xyz: np.ndarray,
    r_xyz: np.ndarray,
    l_vel: np.ndarray,
    r_vel: np.ndarray,
    torso_dx_dy_yaw: np.ndarray,
    arm_layers: np.ndarray,
    forearm_dists: np.ndarray,
    note_times: np.ndarray,
    note_xs: np.ndarray,
    note_ys: np.ndarray,
):
    """Vectorized Numba batch computation of all full-body 7-DoF arm + 6-DoF torso + visual
    occlusion parameters across T frames in <2 ms.
    """
    n = len(times)
    # Torso: [dx, dy, dz_lean, yaw, pitch, roll]
    torso_6 = np.zeros((n, 6), dtype=np.float32)
    # Arm: [ex, ey, ez, s_flex, s_abd, s_rot, elbow_deg, swivel_deg, pro_sup, w_flex, w_dev, heading, awk, j_vel]
    l_arm_14 = np.zeros((n, 14), dtype=np.float32)
    r_arm_14 = np.zeros((n, 14), dtype=np.float32)
    # Bimanual: [is_crossed, layer_code, palm_dist, forearm_3d, l_cross_mm, r_cross_mm, body_awk]
    biman_7 = np.zeros((n, 7), dtype=np.float32)
    # Visual: [occluded_notes_cnt, attention_zones_cnt, gaze_x, gaze_y, gaze_angle_deg, gaze_speed_deg_s]
    vis_6 = np.zeros((n, 6), dtype=np.float32)

    half_w = 0.5 * W_SHOULDER_MM
    n_notes = len(note_times)

    for i in range(n):
        t = times[i]
        dx_m = torso_dx_dy_yaw[i, 0]
        dy_m = torso_dx_dy_yaw[i, 1]
        yaw_d = torso_dx_dy_yaw[i, 2]
        lx, ly, lz = l_xyz[i, 0], l_xyz[i, 1], l_xyz[i, 2]
        rx, ry, rz = r_xyz[i, 0], r_xyz[i, 1], r_xyz[i, 2]

        # Forward trunk lean (pitch_deg, dz_lean) when reaching upper buttons or stretching far radially
        mean_r = 0.5 * (math.hypot(lx, ly) + math.hypot(rx, ry))
        mean_y = 0.5 * (ly + ry)
        pitch_d = max(-5.0, min(28.0, 4.0 + max(0.0, mean_y) * 0.045 + max(0.0, mean_r - 180.0) * 0.055))
        dz_lean = pitch_d * 2.4
        roll_d = max(-22.0, min(22.0, dx_m * 0.095 + (ry - ly) * 0.018))

        torso_6[i, 0] = dx_m
        torso_6[i, 1] = dy_m
        torso_6[i, 2] = dz_lean
        torso_6[i, 3] = yaw_d
        torso_6[i, 4] = pitch_d
        torso_6[i, 5] = roll_d

        rad = math.radians(yaw_d)
        cos_y = math.cos(rad)
        sin_y = math.sin(rad)
        cy_sh = SHOULDER_NEUTRAL_Y_MM + dy_m
        cz_sh = SHOULDER_NEUTRAL_Z_MM - dz_lean
        sl_x = dx_m - half_w * cos_y
        sl_y = cy_sh + half_w * sin_y * 0.65
        sl_z = max(55.0, cz_sh - half_w * sin_y * 0.45)
        sr_x = dx_m + half_w * cos_y
        sr_y = cy_sh - half_w * sin_y * 0.65
        sr_z = max(55.0, cz_sh + half_w * sin_y * 0.45)

        l_res = _solve_arm_7dof_with_swivel_njit(
            True, sl_x, sl_y, sl_z, lx, ly, lz, l_vel[i, 0], l_vel[i, 1],
            0.0, 0.0, 0.0, rx, ry, rz, False, dx_m, yaw_d, pitch_d,
        )
        r_res = _solve_arm_7dof_with_swivel_njit(
            False, sr_x, sr_y, sr_z, rx, ry, rz, r_vel[i, 0], r_vel[i, 1],
            l_res[0], l_res[1], l_res[2], lx, ly, lz, True, dx_m, yaw_d, pitch_d,
        )
        for k in range(13):
            l_arm_14[i, k] = l_res[k]
            r_arm_14[i, k] = r_res[k]

        if i > 0:
            dt_i = max(1e-3, times[i] - times[i - 1])
            l_arm_14[i, 13] = math.sqrt(
                (l_arm_14[i, 3] - l_arm_14[i - 1, 3]) ** 2
                + (l_arm_14[i, 4] - l_arm_14[i - 1, 4]) ** 2
                + (l_arm_14[i, 6] - l_arm_14[i - 1, 6]) ** 2
                + (l_arm_14[i, 8] - l_arm_14[i - 1, 8]) ** 2
            ) / dt_i
            r_arm_14[i, 13] = math.sqrt(
                (r_arm_14[i, 3] - r_arm_14[i - 1, 3]) ** 2
                + (r_arm_14[i, 4] - r_arm_14[i - 1, 4]) ** 2
                + (r_arm_14[i, 6] - r_arm_14[i - 1, 6]) ** 2
                + (r_arm_14[i, 8] - r_arm_14[i - 1, 8]) ** 2
            ) / dt_i

        mid_x = dx_m + yaw_d * 1.8
        l_cross = max(0.0, lx - mid_x)
        r_cross = max(0.0, mid_x - rx)
        is_cr = 1.0 if (lx > rx + 10.0 or arm_layers[i] != 0) else 0.0
        p_dist = math.hypot(lx - rx, ly - ry)
        t_awk = (
            _zone_strain(pitch_d, 0.0, 20.0, -5.0, 40.0)
            + _zone_strain(roll_d, -10.0, 10.0, -20.0, 20.0)
            + _zone_strain(yaw_d, -15.0, 15.0, -30.0, 30.0)
        ) / 3.0
        body_awk = min(1.0, 0.42 * l_res[12] + 0.42 * r_res[12] + 0.16 * t_awk + (0.18 if is_cr > 0.5 else 0.0))

        biman_7[i, 0] = is_cr
        biman_7[i, 1] = float(arm_layers[i])
        biman_7[i, 2] = p_dist
        biman_7[i, 3] = forearm_dists[i]
        biman_7[i, 4] = l_cross
        biman_7[i, 5] = r_cross
        biman_7[i, 6] = body_awk

        # Visual Burden: check upcoming notes in [t + 0.06, t + 0.52]s
        # Projected forearm shadow on z=0 from eye E_eye = (dx_m, HEAD_NEUTRAL_Y_MM + dy_m, 340.0 - dz_lean)
        eye_x = dx_m
        eye_y = HEAD_NEUTRAL_Y_MM + dy_m
        eye_z = max(180.0, 340.0 - dz_lean)

        def _proj_shadow(ax: float, ay: float, az: float) -> Tuple[float, float]:
            u_z = eye_z / max(40.0, eye_z - max(0.0, min(eye_z - 45.0, az)))
            return (eye_x + (ax - eye_x) * u_z, eye_y + (ay - eye_y) * u_z)

        el_sx, el_sy = _proj_shadow(l_res[0], l_res[1], l_res[2])
        pl_sx, pl_sy = _proj_shadow(lx, ly, lz)
        er_sx, er_sy = _proj_shadow(r_res[0], r_res[1], r_res[2])
        pr_sx, pr_sy = _proj_shadow(rx, ry, rz)

        occ_cnt = 0
        sec_mask = 0
        gx_sum = 0.0
        gy_sum = 0.0
        g_cnt = 0
        # Binary search start in note_times
        lo_n = 0
        hi_n = n_notes
        while lo_n < hi_n:
            mid_n = (lo_n + hi_n) // 2
            if note_times[mid_n] < t + 0.05:
                lo_n = mid_n + 1
            else:
                hi_n = mid_n
        for jn in range(lo_n, n_notes):
            nt = note_times[jn]
            if nt > t + 0.52:
                break
            # Note approaches radially from r=55mm at (nt - 0.50) to target (nx, ny) at nt
            prog_n = max(0.20, min(1.0, 1.0 - (nt - t) / 0.52))
            nx_cur = note_xs[jn] * prog_n
            ny_cur = note_ys[jn] * prog_n
            gx_sum += nx_cur
            gy_sum += ny_cur
            g_cnt += 1
            ang_s = int((math.degrees(math.atan2(note_xs[jn], note_ys[jn])) % 360.0) / 90.0)
            sec_mask |= (1 << ang_s)

            # Check distance from (nx_cur, ny_cur) to L/R projected forearm segments & palms
            for s1x, s1y, s2x, s2y in ((el_sx, el_sy, pl_sx, pl_sy), (er_sx, er_sy, pr_sx, pr_sy)):
                ddx, ddy = s2x - s1x, s2y - s1y
                l2_s = ddx * ddx + ddy * ddy
                u_p = 0.0 if l2_s < 1e-6 else max(0.0, min(1.0, ((nx_cur - s1x) * ddx + (ny_cur - s1y) * ddy) / l2_s))
                qx = s1x + u_p * ddx
                qy = s1y + u_p * ddy
                # Effective shadow half-width: 44mm forearm, 52mm palm
                r_sh = 44.0 + 10.0 * u_p
                if math.hypot(nx_cur - qx, ny_cur - qy) <= r_sh:
                    occ_cnt += 1
                    break

        att_zones = 0
        for bit_k in range(4):
            if sec_mask & (1 << bit_k):
                att_zones += 1
        if g_cnt > 0:
            gx = gx_sum / g_cnt
            gy = gy_sum / g_cnt
        else:
            gx = 0.5 * (lx + rx)
            gy = 0.5 * (ly + ry)
        g_ang = math.degrees(math.atan2(math.hypot(gx - eye_x, gy - (eye_y - 120.0)), eye_z))
        vis_6[i, 0] = float(occ_cnt)
        vis_6[i, 1] = float(max(1, att_zones))
        vis_6[i, 2] = gx
        vis_6[i, 3] = gy
        vis_6[i, 4] = g_ang
        if i > 0:
            dt_i = max(1e-3, times[i] - times[i - 1])
            dg = math.hypot(gx - vis_6[i - 1, 2], gy - vis_6[i - 1, 3])
            vis_6[i, 5] = math.degrees(math.atan2(dg, eye_z)) / dt_i

    return torso_6, l_arm_14, r_arm_14, biman_7, vis_6


def _classify_fine_technique(
    mode_str: str,
    decomp_str: str,
    note: Optional[ParsedNote],
    speed_mm_s: float,
    is_one_stroke: bool,
) -> str:
    """Map planner mode & note properties to readable human technique label:
    'PointTap' (点), 'Slap' (拍), 'Sweep' (扫), 'Thumb' (拇指), 'PalmHeel' (掌根),
    'Forearm' (小臂), 'OuterInnerSplit' (内外拆), 'AltTap' (交替), 'Hold' (按住),
    'Slide' (滑条), 'OneStroke' (一笔画).
    """
    if is_one_stroke and (note is None or note.note_type == NoteType.SLIDE):
        return "OneStroke"
    if note is not None and note.note_type == NoteType.SLIDE:
        return "Slide"
    if mode_str == "Slide":
        return "Slide"
    if mode_str == "Sweep" or decomp_str == "RUB_SWEEP":
        return "Sweep"
    if mode_str in ("OuterSplit", "InnerSplit") or decomp_str == "OUTER_INNER_SPLIT":
        return "OuterInnerSplit"
    if mode_str == "AltTap" or decomp_str == "ALTERNATING_TAP":
        return "AltTap"
    if mode_str == "Hold" and decomp_str == "SINGLE_PALM_RUB":
        # Holding one sensor with fingertips while tapping an adjacent sensor with thumb / palm heel
        return "Thumb" if (note is not None and note.note_type == NoteType.TOUCH) else "PalmHeel"
    if mode_str == "Hold" or (note is not None and note.note_type in (NoteType.HOLD, NoteType.TOUCH_HOLD)):
        return "Hold"
    if mode_str == "Rub" or decomp_str == "SINGLE_PALM_RUB":
        if speed_mm_s > 900.0:
            return "Forearm"
        return "PalmHeel"
    if note is not None and (note.is_break or note.is_ex or mode_str == "Outer" or speed_mm_s > 1350.0):
        return "Slap"
    return "PointTap"


@dataclass
class BodyState:
    """Complete human body state at a single frame (>=30 fps)."""
    time: float
    bar: int
    bpm: float
    torso: Dict[str, float]
    left_arm: Dict[str, Any]
    right_arm: Dict[str, Any]
    bimanual: Dict[str, Any]
    visual: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "time": self.time,
            "bar": self.bar,
            "bpm": self.bpm,
            "torso": self.torso,
            "left_arm": self.left_arm,
            "right_arm": self.right_arm,
            "bimanual": self.bimanual,
            "visual": self.visual,
        }


@dataclass
class BodyTimeline:
    """Full body-movement decomposition of a chart:
      - `frames`: list of `BodyState` at >=30 fps
      - `note_table`: per-note hand assignment, fine technique, body state, move distance & available time
      - `per_bar`: per-bar body, visual burden, and awkwardness summaries
      - `summary`: whole-chart body & visual burden summary
      - `arrays`: contiguous NumPy arrays for fast downstream ML/clustering
    """
    fps: float
    frames: List[BodyState]
    note_table: List[Dict[str, Any]]
    per_bar: List[Dict[str, Any]]
    summary: Dict[str, Any]
    arrays: Dict[str, np.ndarray] = field(default_factory=dict, repr=False)
    plan_result: Optional[PlanResult] = field(default=None, repr=False)

    def to_dict(self, include_frames: bool = False) -> Dict[str, Any]:
        d = {
            "fps": self.fps,
            "n_frames": len(self.frames),
            "summary": self.summary,
            "per_bar": self.per_bar,
            "note_table": self.note_table,
        }
        if include_frames:
            d["frames"] = [f.to_dict() for f in self.frames]
        return d


def build_body_timeline_from_plan(
    plan: PlanResult,
    materialize_frame_dicts: bool = True,
) -> BodyTimeline:
    """Construct `BodyTimeline` (frames, note_table, per_bar, summary) from `plan`."""
    from sim_core.motion import extract_trajectory_arrays

    chart = plan.chart
    traj = extract_trajectory_arrays(plan)
    t_arr = traj["time"]
    bar_arr = traj["bar"]
    bpm_arr = traj["bpm"]
    l_xyz = traj["left_xyz"]
    r_xyz = traj["right_xyz"]
    l_vel = traj["left_vel_xyz"]
    r_vel = traj["right_vel_xyz"]
    l_spd = traj["left_speed"]
    r_spd = traj["right_speed"]
    l_acc = traj["left_accel"]
    r_acc = traj["right_accel"]
    l_con = traj["left_contact"]
    r_con = traj["right_contact"]
    torso_3 = traj["torso_dx_dy_yaw"]
    arm_lay = traj["arm_layer"]
    f_dist = traj["forearm_dist_3d"]
    l_tech = traj["left_tech"]
    r_tech = traj["right_tech"]

    n_f = len(t_arr)
    strike_notes_all = [n for n in chart.notes if n.note_type != NoteType.SLIDE]
    nt_arr = np.array([n.time for n in strike_notes_all], dtype=np.float32) if strike_notes_all else np.zeros(0, dtype=np.float32)
    nx_arr = np.array(
        [
            (get_sensor_zone(n.sensor_zone).center[0] if n.note_type in (NoteType.TOUCH, NoteType.TOUCH_HOLD) else get_button_target_xy(n.button or 1, use_outer=False)[0])
            for n in strike_notes_all
        ],
        dtype=np.float32,
    ) if strike_notes_all else np.zeros(0, dtype=np.float32)
    ny_arr = np.array(
        [
            (get_sensor_zone(n.sensor_zone).center[1] if n.note_type in (NoteType.TOUCH, NoteType.TOUCH_HOLD) else get_button_target_xy(n.button or 1, use_outer=False)[1])
            for n in strike_notes_all
        ],
        dtype=np.float32,
    ) if strike_notes_all else np.zeros(0, dtype=np.float32)

    if n_f > 0:
        torso_6, l_arm_14, r_arm_14, biman_7, vis_6 = _compute_body_frames_batch_njit(
            t_arr, l_xyz, r_xyz, l_vel, r_vel, torso_3, arm_lay, f_dist, nt_arr, nx_arr, ny_arr
        )
        dts_1d = np.maximum(1e-3, np.diff(t_arr, prepend=t_arr[0] - 0.033)).astype(np.float32)
        l_jerk = np.abs(np.diff(l_acc, prepend=l_acc[0])) / dts_1d
        r_jerk = np.abs(np.diff(r_acc, prepend=r_acc[0])) / dts_1d
        torso_ang_vel = np.linalg.norm(np.diff(torso_6[:, 3:6], axis=0, prepend=torso_6[:1, 3:6]), axis=1) / dts_1d
    else:
        torso_6 = np.zeros((0, 6), dtype=np.float32)
        l_arm_14 = np.zeros((0, 14), dtype=np.float32)
        r_arm_14 = np.zeros((0, 14), dtype=np.float32)
        biman_7 = np.zeros((0, 7), dtype=np.float32)
        vis_6 = np.zeros((0, 6), dtype=np.float32)
        l_jerk = np.zeros(0, dtype=np.float32)
        r_jerk = np.zeros(0, dtype=np.float32)
        torso_ang_vel = np.zeros(0, dtype=np.float32)

    fps_actual = round(1.0 / max(1e-3, float(t_arr[1] - t_arr[0])), 1) if n_f >= 2 else 30.0
    dt_frame = 1.0 / max(10.0, fps_actual)

    # Build per-note table (`note_table`) with exact hand, technique, body state, move_dist_mm, avail_time_sec, occlusion
    steps_by_time: Dict[int, Any] = {int(round(s.time * 1000)): s for s in plan.planned_steps}
    slides_sorted = sorted([n for n in chart.notes if n.note_type == NoteType.SLIDE], key=lambda x: x.shoot_time)
    one_stroke_ids = set()
    for idx_s in range(1, len(slides_sorted)):
        sp = slides_sorted[idx_s - 1]
        sc = slides_sorted[idx_s]
        eb = sp.slide_segments[-1].end_btn if sp.slide_segments else -1
        if eb == sc.button and (abs(sc.shoot_time - sp.end_time) <= 0.22 or abs(sc.time - sp.end_time) <= 0.22):
            one_stroke_ids.add(sc.note_id)

    note_table: List[Dict[str, Any]] = []
    last_hand_t = {"L": -999.0, "R": -999.0}
    last_hand_xy = {"L": (-110.0, -45.0), "R": (110.0, -45.0)}

    for n in chart.notes:
        h_side, z_target = plan.note_hand_map.get(n.note_id, ("R", n.sensor_zone))
        if n.note_type == NoteType.SLIDE and any(sg.shape == "w" for sg in n.slide_segments):
            h_side = "LR"
        f_idx = int(np.clip(np.searchsorted(t_arr, n.time), 0, max(0, n_f - 1))) if n_f > 0 else 0
        step_obj = steps_by_time.get(int(round(n.time * 1000)))
        decomp_used = step_obj.decomposition_used if step_obj is not None else "NONE"

        if h_side in ("L", "LR"):
            m_str = str(l_tech[f_idx]) if n_f > 0 else "Inner"
            spd_n = float(l_spd[f_idx]) if n_f > 0 else 0.0
            arm_row = l_arm_14[f_idx] if n_f > 0 else np.zeros(14)
            cur_xy = (float(l_xyz[f_idx, 0]), float(l_xyz[f_idx, 1])) if n_f > 0 else (-110.0, -45.0)
            prev_t_h = last_hand_t["L"]
            prev_xy_h = last_hand_xy["L"]
        else:
            m_str = str(r_tech[f_idx]) if n_f > 0 else "Inner"
            spd_n = float(r_spd[f_idx]) if n_f > 0 else 0.0
            arm_row = r_arm_14[f_idx] if n_f > 0 else np.zeros(14)
            cur_xy = (float(r_xyz[f_idx, 0]), float(r_xyz[f_idx, 1])) if n_f > 0 else (110.0, -45.0)
            prev_t_h = last_hand_t["R"]
            prev_xy_h = last_hand_xy["R"]

        move_dist = math.hypot(cur_xy[0] - prev_xy_h[0], cur_xy[1] - prev_xy_h[1]) if prev_t_h > -100.0 else 0.0
        avail_dt = max(0.0, n.time - prev_t_h) if prev_t_h > -100.0 else 0.5
        if n.note_type == NoteType.SLIDE:
            end_xy = n.slide_waypoints_mm[-1] if n.slide_waypoints_mm else cur_xy
            if h_side in ("L", "LR"):
                last_hand_t["L"] = max(last_hand_t["L"], n.judge_time)
                last_hand_xy["L"] = end_xy
            if h_side in ("R", "LR"):
                last_hand_t["R"] = max(last_hand_t["R"], n.judge_time)
                last_hand_xy["R"] = end_xy
        else:
            if h_side in ("L", "LR"):
                last_hand_t["L"] = max(last_hand_t["L"], n.time)
                last_hand_xy["L"] = cur_xy
            if h_side in ("R", "LR"):
                last_hand_t["R"] = max(last_hand_t["R"], n.time)
                last_hand_xy["R"] = cur_xy

        fine_tech = _classify_fine_technique(m_str, decomp_used, n, spd_n, n.note_id in one_stroke_ids)

        # Check how many frames during [n.time - 0.45, n.time - 0.05] this note's target was occluded
        f_pre0 = max(0, int(np.searchsorted(t_arr, n.time - 0.45))) if n_f > 0 else 0
        f_pre1 = max(f_pre0, int(np.searchsorted(t_arr, n.time - 0.04))) if n_f > 0 else 0
        occ_frames = int(np.sum(vis_6[f_pre0:f_pre1, 0] > 0)) if f_pre1 > f_pre0 else 0
        occ_dur = round(occ_frames * dt_frame, 3)

        t_row = torso_6[f_idx] if n_f > 0 else np.zeros(6)
        b_row = biman_7[f_idx] if n_f > 0 else np.zeros(7)
        note_table.append(
            {
                "note_id": int(n.note_id),
                "time": round(float(n.time), 4),
                "bar": int(n.bar),
                "beat_in_bar": round(float(n.beat_in_bar), 3),
                "key": str(n.key),
                "note_type": n.note_type.value,
                "hand": h_side,
                "target_zone": z_target,
                "technique": fine_tech,
                "move_dist_mm": round(move_dist, 1),
                "avail_time_sec": round(avail_dt, 4),
                "req_speed_mm_s": round(move_dist / max(0.025, avail_dt), 1) if avail_dt > 0 else 0.0,
                "occlusion_duration_sec": occ_dur,
                "is_occluded": occ_dur >= 0.08,
                "body_state": {
                    "torso_dx_mm": round(float(t_row[0]), 1),
                    "torso_yaw_deg": round(float(t_row[3]), 1),
                    "torso_pitch_deg": round(float(t_row[4]), 1),
                    "torso_roll_deg": round(float(t_row[5]), 1),
                    "shoulder_flex_deg": round(float(arm_row[3]), 1),
                    "shoulder_abd_deg": round(float(arm_row[4]), 1),
                    "shoulder_rot_deg": round(float(arm_row[5]), 1),
                    "elbow_angle_deg": round(float(arm_row[6]), 1),
                    "forearm_pro_sup_deg": round(float(arm_row[8]), 1),
                    "wrist_flex_deg": round(float(arm_row[9]), 1),
                    "wrist_dev_deg": round(float(arm_row[10]), 1),
                    "cross_midline_mm": round(float(b_row[4] if h_side == "L" else b_row[5]), 1),
                    "awkwardness": round(float(b_row[6]), 3),
                },
            }
        )

    # Materialize frame objects if requested
    frames_list: List[BodyState] = []
    if materialize_frame_dicts and n_f > 0:
        for i in range(n_f):
            lay_i = int(biman_7[i, 1])
            lay_s = "L_OVER_R" if lay_i == 1 else ("R_OVER_L" if lay_i == -1 else "NORMAL")
            frames_list.append(
                BodyState(
                    time=round(float(t_arr[i]), 4),
                    bar=int(bar_arr[i]),
                    bpm=round(float(bpm_arr[i]), 2),
                    torso={
                        "dx_mm": round(float(torso_6[i, 0]), 1),
                        "dy_mm": round(float(torso_6[i, 1]), 1),
                        "dz_mm": round(float(torso_6[i, 2]), 1),
                        "yaw_deg": round(float(torso_6[i, 3]), 2),
                        "pitch_deg": round(float(torso_6[i, 4]), 2),
                        "roll_deg": round(float(torso_6[i, 5]), 2),
                        "angular_vel_deg_s": round(float(torso_ang_vel[i]), 1),
                    },
                    left_arm={
                        "palm_xyz": (round(float(l_xyz[i, 0]), 1), round(float(l_xyz[i, 1]), 1), round(float(l_xyz[i, 2]), 1)),
                        "elbow_xyz": (round(float(l_arm_14[i, 0]), 1), round(float(l_arm_14[i, 1]), 1), round(float(l_arm_14[i, 2]), 1)),
                        "shoulder_flex_deg": round(float(l_arm_14[i, 3]), 1),
                        "shoulder_abd_deg": round(float(l_arm_14[i, 4]), 1),
                        "shoulder_rot_deg": round(float(l_arm_14[i, 5]), 1),
                        "elbow_angle_deg": round(float(l_arm_14[i, 6]), 1),
                        "swivel_angle_deg": round(float(l_arm_14[i, 7]), 1),
                        "forearm_pro_sup_deg": round(float(l_arm_14[i, 8]), 1),
                        "wrist_flex_deg": round(float(l_arm_14[i, 9]), 1),
                        "wrist_dev_deg": round(float(l_arm_14[i, 10]), 1),
                        "palm_heading_deg": round(float(l_arm_14[i, 11]), 1),
                        "contact": bool(l_con[i]),
                        "technique": str(l_tech[i]),
                        "speed_mm_s": round(float(l_spd[i]), 1),
                        "accel_mm_s2": round(float(l_acc[i]), 1),
                        "jerk_mm_s3": round(float(l_jerk[i]), 1),
                        "joint_vel_deg_s": round(float(l_arm_14[i, 13]), 1),
                        "awkwardness": round(float(l_arm_14[i, 12]), 3),
                    },
                    right_arm={
                        "palm_xyz": (round(float(r_xyz[i, 0]), 1), round(float(r_xyz[i, 1]), 1), round(float(r_xyz[i, 2]), 1)),
                        "elbow_xyz": (round(float(r_arm_14[i, 0]), 1), round(float(r_arm_14[i, 1]), 1), round(float(r_arm_14[i, 2]), 1)),
                        "shoulder_flex_deg": round(float(r_arm_14[i, 3]), 1),
                        "shoulder_abd_deg": round(float(r_arm_14[i, 4]), 1),
                        "shoulder_rot_deg": round(float(r_arm_14[i, 5]), 1),
                        "elbow_angle_deg": round(float(r_arm_14[i, 6]), 1),
                        "swivel_angle_deg": round(float(r_arm_14[i, 7]), 1),
                        "forearm_pro_sup_deg": round(float(r_arm_14[i, 8]), 1),
                        "wrist_flex_deg": round(float(r_arm_14[i, 9]), 1),
                        "wrist_dev_deg": round(float(r_arm_14[i, 10]), 1),
                        "palm_heading_deg": round(float(r_arm_14[i, 11]), 1),
                        "contact": bool(r_con[i]),
                        "technique": str(r_tech[i]),
                        "speed_mm_s": round(float(r_spd[i]), 1),
                        "accel_mm_s2": round(float(r_acc[i]), 1),
                        "jerk_mm_s3": round(float(r_jerk[i]), 1),
                        "joint_vel_deg_s": round(float(r_arm_14[i, 13]), 1),
                        "awkwardness": round(float(r_arm_14[i, 12]), 3),
                    },
                    bimanual={
                        "is_crossed": bool(biman_7[i, 0] > 0.5),
                        "layer": lay_s,
                        "palm_dist_mm": round(float(biman_7[i, 2]), 1),
                        "forearm_dist_3d_mm": round(float(biman_7[i, 3]), 1),
                        "left_cross_midline_mm": round(float(biman_7[i, 4]), 1),
                        "right_cross_midline_mm": round(float(biman_7[i, 5]), 1),
                        "posture_awkwardness": round(float(biman_7[i, 6]), 3),
                    },
                    visual={
                        "occluded_upcoming_notes": int(vis_6[i, 0]),
                        "concurrent_attention_zones": int(vis_6[i, 1]),
                        "gaze_angle_deg": round(float(vis_6[i, 4]), 2),
                        "gaze_shift_speed_deg_s": round(float(vis_6[i, 5]), 1),
                    },
                )
            )

    # Per-bar body & visual summary
    bar_ts = chart.bar_timestamps or [(0, 0.0, max(1.0, chart.total_duration_sec))]
    notes_by_bar: Dict[int, List[Dict[str, Any]]] = {}
    for nr in note_table:
        notes_by_bar.setdefault(nr["bar"], []).append(nr)

    per_bar_body: List[Dict[str, Any]] = []
    for b_idx, b_t0, b_t1 in bar_ts:
        f0 = int(np.searchsorted(t_arr, b_t0, side="left")) if n_f > 0 else 0
        f1 = int(np.searchsorted(t_arr, b_t1, side="right")) if n_f > 0 else 0
        b_n_rows = notes_by_bar.get(b_idx, [])
        occ_notes_b = sum(1 for nr in b_n_rows if nr["is_occluded"])
        occ_dur_b = sum(float(nr["occlusion_duration_sec"]) for nr in b_n_rows)

        if f1 > f0:
            awk_mean = float(np.mean(biman_7[f0:f1, 6]))
            awk_p90 = float(np.percentile(biman_7[f0:f1, 6], 90))
            cross_frac = float(np.mean(biman_7[f0:f1, 0]))
            pro_sup_range = float(
                0.5 * ((l_arm_14[f0:f1, 8].max() - l_arm_14[f0:f1, 8].min()) + (r_arm_14[f0:f1, 8].max() - r_arm_14[f0:f1, 8].min()))
            )
            w_dev_max = float(max(np.max(np.abs(l_arm_14[f0:f1, 10])), np.max(np.abs(r_arm_14[f0:f1, 10]))))
            el_mean = float(0.5 * (np.mean(l_arm_14[f0:f1, 6]) + np.mean(r_arm_14[f0:f1, 6])))
            j_vel_mean = float(0.5 * (np.mean(l_arm_14[f0:f1, 13]) + np.mean(r_arm_14[f0:f1, 13])))
            pitch_mean = float(np.mean(torso_6[f0:f1, 4]))
            yaw_range = float(torso_6[f0:f1, 3].max() - torso_6[f0:f1, 3].min())
            att_mean = float(np.mean(vis_6[f0:f1, 1]))
            gaze_spd = float(np.mean(vis_6[f0:f1, 5]))
            occ_frame_ratio = float(np.mean(vis_6[f0:f1, 0] > 0))
        else:
            awk_mean = awk_p90 = cross_frac = pro_sup_range = w_dev_max = 0.0
            el_mean = 95.0
            j_vel_mean = pitch_mean = yaw_range = gaze_spd = occ_frame_ratio = 0.0
            att_mean = 1.0

        # Combined "playable-but-awkward" score [0..1] for bar `b_idx`:
        # combines RULA joint awkwardness, sustained cross-hand, sustained waist twist, and visual occlusion
        awk_bar_score = min(
            1.0,
            0.45 * awk_p90
            + 0.20 * cross_frac
            + 0.15 * min(1.0, yaw_range / 38.0)
            + 0.20 * min(1.0, occ_frame_ratio * 1.5),
        )
        per_bar_body.append(
            {
                "bar": int(b_idx),
                "n_notes": len(b_n_rows),
                "awkward_score": round(awk_bar_score, 4),
                "joint_awkward_mean": round(awk_mean, 4),
                "joint_awkward_p90": round(awk_p90, 4),
                "cross_time_fraction": round(cross_frac, 4),
                "forearm_rot_range_deg": round(pro_sup_range, 2),
                "wrist_dev_max_deg": round(w_dev_max, 2),
                "elbow_angle_mean_deg": round(el_mean, 2),
                "joint_vel_mean_deg_s": round(j_vel_mean, 1),
                "torso_pitch_mean_deg": round(pitch_mean, 2),
                "torso_yaw_range_deg": round(yaw_range, 2),
                "occluded_notes": int(occ_notes_b),
                "occlusion_duration_sec": round(occ_dur_b, 3),
                "occlusion_frame_ratio": round(occ_frame_ratio, 4),
                "attention_zones_mean": round(att_mean, 2),
                "gaze_shift_speed_deg_s": round(gaze_spd, 1),
            }
        )

    active_pb = [r for r in per_bar_body if r["n_notes"] > 0] or per_bar_body
    tech_dist: Dict[str, int] = {}
    for nr in note_table:
        t_k = nr["technique"]
        tech_dist[t_k] = tech_dist.get(t_k, 0) + 1

    summary = {
        "awkward_score_mean": round(float(np.mean([r["awkward_score"] for r in active_pb])), 4) if active_pb else 0.0,
        "awkward_score_p90": round(float(np.percentile([r["awkward_score"] for r in active_pb], 90)), 4) if active_pb else 0.0,
        "joint_awkward_mean": round(float(np.mean([r["joint_awkward_mean"] for r in active_pb])), 4) if active_pb else 0.0,
        "cross_time_fraction": round(float(np.mean([r["cross_time_fraction"] for r in active_pb])), 4) if active_pb else 0.0,
        "forearm_rot_range_mean_deg": round(float(np.mean([r["forearm_rot_range_deg"] for r in active_pb])), 2) if active_pb else 0.0,
        "wrist_dev_max_p90_deg": round(float(np.percentile([r["wrist_dev_max_deg"] for r in active_pb], 90)), 2) if active_pb else 0.0,
        "elbow_angle_mean_deg": round(float(np.mean([r["elbow_angle_mean_deg"] for r in active_pb])), 2) if active_pb else 95.0,
        "joint_vel_mean_deg_s": round(float(np.mean([r["joint_vel_mean_deg_s"] for r in active_pb])), 1) if active_pb else 0.0,
        "torso_pitch_mean_deg": round(float(np.mean([r["torso_pitch_mean_deg"] for r in active_pb])), 2) if active_pb else 0.0,
        "occluded_notes_total": int(sum(r["occluded_notes"] for r in per_bar_body)),
        "occluded_note_ratio": round(sum(r["occluded_notes"] for r in per_bar_body) / max(1, len(note_table)), 4),
        "occlusion_frame_ratio_mean": round(float(np.mean([r["occlusion_frame_ratio"] for r in active_pb])), 4) if active_pb else 0.0,
        "attention_zones_mean": round(float(np.mean([r["attention_zones_mean"] for r in active_pb])), 2) if active_pb else 1.0,
        "gaze_shift_speed_mean_deg_s": round(float(np.mean([r["gaze_shift_speed_deg_s"] for r in active_pb])), 1) if active_pb else 0.0,
        "technique_counts": tech_dist,
    }

    arrays = {
        "time": t_arr,
        "bar": bar_arr,
        "bpm": bpm_arr,
        "torso_6dof": torso_6,
        "left_arm_14": l_arm_14,
        "right_arm_14": r_arm_14,
        "bimanual_7": biman_7,
        "visual_6": vis_6,
        "left_xyz": l_xyz,
        "right_xyz": r_xyz,
        "left_speed": l_spd,
        "right_speed": r_spd,
    }
    return BodyTimeline(
        fps=fps_actual,
        frames=frames_list,
        note_table=note_table,
        per_bar=per_bar_body,
        summary=summary,
        arrays=arrays,
        plan_result=plan,
    )


def body_timeline(
    chart_or_notes: Any,
    tm: Any = None,
    difficulty: str | int = "auto",
    hit_mode: str | HitMode = DEFAULT_HIT_MODE,
    skill_profile: str | SkillProfileConfig = DEFAULT_SKILL_PROFILE,
    frame_fps: int = 30,
    materialize_frame_dicts: bool = True,
) -> BodyTimeline:
    """Public API (`simv3` Section 6.1): parse a chart (or tick notes + `TempoMap` or `PlanResult`)
    into a full `BodyTimeline` at `>= 30 fps` (frame-level `BodyState`, `note_table`, and `per_bar` summary).
    """
    eff_fps = max(30, int(frame_fps))
    if isinstance(chart_or_notes, PlanResult):
        return build_body_timeline_from_plan(chart_or_notes, materialize_frame_dicts=materialize_frame_dicts)
    if hasattr(chart_or_notes, "plan_result") and chart_or_notes.plan_result is not None:
        return build_body_timeline_from_plan(chart_or_notes.plan_result, materialize_frame_dicts=materialize_frame_dicts)

    if isinstance(chart_or_notes, ParsedChart):
        chart = chart_or_notes
    elif tm is not None and isinstance(chart_or_notes, (list, tuple)):
        from sim_core.fast_simai import build_parsed_chart_from_notes
        slot_i = int(difficulty) if str(difficulty).isdigit() else 5
        chart = build_parsed_chart_from_notes(chart_or_notes, tm, slot=slot_i, unit_ticks=True)
    else:
        chart = parse_simai_chart(chart_or_notes, difficulty=difficulty)

    plan = plan_chart_movements(
        chart=chart,
        hit_mode=HitMode.from_str(hit_mode),
        skill_profile=get_skill_profile(skill_profile),
        frame_fps=eff_fps,
    )
    return build_body_timeline_from_plan(plan, materialize_frame_dicts=materialize_frame_dicts)
