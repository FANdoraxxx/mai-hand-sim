"""Coupled Upper-Body Torso (3-DoF) + 2-Link Planar/Z Arm Inverse Kinematics
and Doraemon Round Palm Biomechanics.
"""
from __future__ import annotations

from functools import lru_cache
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from sim_core.config import (
    ARM_L1_MM,
    ARM_L2_MM,
    ARM_MAX_REACH_MM,
    ARM_MIN_REACH_MM,
    CROSS_BODY_EXP_SCALE_MM,
    ELBOW_MAX_ANGLE_DEG,
    ELBOW_MIN_ANGLE_DEG,
    HAND_COLLISION_MIN_DIST_MM,
    HAND_COLLISION_Z_DIFF_MM,
    HOLD_EDGE_SHIFT_MAX_MM,
    PELVIS_ANCHOR_MM,
    R_HAND_MM,
    R_HOLD_FINGERTIP_MM,
    R_STRIKE_FINGER_CLEARANCE_MM,
    SHOULDER_NEUTRAL_Y_MM,
    SHOULDER_NEUTRAL_Z_MM,
    TORSO_DX_LIMIT_MM,
    TORSO_DY_LIMIT_MM,
    TORSO_YAW_LIMIT_DEG,
    W_SHOULDER_MM,
    Z_HOLD_ARCHED_PALM_MM,
    Z_RELEASE_MM,
    SkillProfileConfig,
)

try:
    from numba import njit as _njit
    _HAS_NUMBA = True
except Exception:  # pragma: no cover
    _HAS_NUMBA = False
    def _njit(*args, **kwargs):  # type: ignore
        def _wrap(fn):
            return fn
        return _wrap

Point2D = Tuple[float, float]
Point3D = Tuple[float, float, float]


@dataclass(slots=True)
class TorsoState:
    """3-DoF Upper Body / Torso state and kinematics."""
    dx_mm: float = 0.0              # Lateral shift [-120, +120] mm
    dy_mm: float = 0.0              # Vertical dip/rise [-100, +60] mm
    yaw_deg: float = 0.0            # Waist rotation [-35, +35] deg (+CW = left shoulder forward)
    vx_mm_s: float = 0.0
    vy_mm_s: float = 0.0
    yaw_vel_deg_s: float = 0.0
    yaw_accel_deg_s2: float = 0.0

    @property
    def pelvis_xy(self) -> Point2D:
        return (PELVIS_ANCHOR_MM[0] + self.dx_mm, PELVIS_ANCHOR_MM[1] + self.dy_mm * 0.5)

    @property
    def shoulder_center_xyz(self) -> Point3D:
        return (self.dx_mm, SHOULDER_NEUTRAL_Y_MM + self.dy_mm, SHOULDER_NEUTRAL_Z_MM)


@dataclass(slots=True)
class ArmIKSolution:
    """2-Link Arm + Doraemon Round Palm inverse kinematics solution."""
    side: str                       # 'L' or 'R'
    shoulder_xyz: Point3D
    elbow_xyz: Point3D
    palm_xyz: Point3D
    shoulder_angle_deg: float       # Upper arm angle alpha in plane (deg)
    elbow_angle_deg: float          # Interior elbow angle beta (deg, ~45..170)
    reach_distance_mm: float        # Planar shoulder-to-palm distance (mm)
    reach_margin_mm: float          # Remaining reach before full extension (mm)
    cross_depth_mm: float           # Distance palm crosses past torso midline (mm)
    cross_cost: float               # Exponential cross-body biomechanical cost
    joint_cost: float               # Total postural & joint comfort cost
    is_reachable: bool              # True if within physical arm length without hyperextension
    is_chest_penetrating: bool      # True if upper arm violates chest boundary
    is_joint_limit_violated: bool = False


@dataclass(slots=True)
class BodyKinematicsState:
    """Complete full-body posture at a single timestamp or event."""
    time: float
    torso: TorsoState
    left_arm: ArmIKSolution
    right_arm: ArmIKSolution
    hand_distance_xy_mm: float
    hand_z_diff_mm: float
    is_hand_collision: bool
    is_extreme_cross_arm: bool
    is_torso_whiplash: bool
    posture_cost: float
    arm_layer_order: str = "NORMAL"         # 'NORMAL', 'L_OVER_R', or 'R_OVER_L'
    forearm_min_dist_3d_mm: float = 200.0   # 3D center-to-center distance between L/R forearm capsules
    crossing_point_xy: Optional[Point2D] = None
    is_forearm_intersection: bool = False


def compute_shoulder_positions(torso: TorsoState) -> Tuple[Point3D, Point3D]:
    """Compute 3D positions of Left and Right shoulder joints from TorsoState.

    When torso rotates by `yaw_deg` (positive = turning right / clockwise in top view):
      - Left shoulder moves forward (+Y) and closer to the screen (-Z), and inward in X
      - Right shoulder moves backward (-Y) and further from the screen (+Z)
    """
    cx, cy, cz = torso.shoulder_center_xyz
    half_w = 0.5 * W_SHOULDER_MM
    rad = math.radians(torso.yaw_deg)
    cos_y = math.cos(rad)
    sin_y = math.sin(rad)

    # Foreshortened lateral width + anteroposterior / depth shift
    lx = cx - half_w * cos_y
    ly = cy + half_w * sin_y * 0.65
    lz = max(60.0, cz - half_w * sin_y * 0.45)

    rx = cx + half_w * cos_y
    ry = cy - half_w * sin_y * 0.65
    rz = max(60.0, cz + half_w * sin_y * 0.45)

    return ((lx, ly, lz), (rx, ry, rz))


def _arm_joint_cost_scalar(
    is_left: bool,
    sx: float,
    sy: float,
    sz: float,
    px: float,
    py: float,
    pz: float,
    cx: float,
    yaw_deg: float,
    l1_mm: float = ARM_L1_MM,
    l2_mm: float = ARM_L2_MM,
) -> float:
    """Allocation-free scalar equivalent of `solve_arm_ik(...).joint_cost`."""
    dx = px - sx
    dy = py - sy
    dz = pz - sz
    dist_3d = math.sqrt(dx * dx + dy * dy + dz * dz)
    max_reach = l1_mm + l2_mm
    min_reach = max(ARM_MIN_REACH_MM, abs(l1_mm - l2_mm) + 15.0)
    d_clamped = max(min_reach, min(max_reach - 1.0, dist_3d))
    a_proj = (l1_mm * l1_mm - l2_mm * l2_mm + d_clamped * d_clamped) / (2.0 * d_clamped)
    h_elbow = math.sqrt(max(0.0, l1_mm * l1_mm - a_proj * a_proj))
    cos_beta = (l1_mm * l1_mm + l2_mm * l2_mm - d_clamped * d_clamped) / (2.0 * l1_mm * l2_mm)
    cos_beta = max(-1.0, min(1.0, cos_beta))
    beta_deg = max(28.0, min(172.0, math.degrees(math.acos(cos_beta))))
    if dist_3d > 1e-6:
        ux, uy, uz = dx / dist_3d, dy / dist_3d, dz / dist_3d
    else:
        ux, uy, uz = 0.0, 0.0, -1.0
    out_x = -0.65 if is_left else 0.65
    rx_ref, ry_ref, rz_ref = out_x, -0.88, -0.15
    dot_ru = rx_ref * ux + ry_ref * uy + rz_ref * uz
    nx = rx_ref - dot_ru * ux
    ny = ry_ref - dot_ru * uy
    nz = rz_ref - dot_ru * uz
    n_norm = math.sqrt(nx * nx + ny * ny + nz * nz)
    if n_norm > 1e-6:
        nx, ny, nz = nx / n_norm, ny / n_norm, nz / n_norm
    else:
        nx, ny, nz = (-1.0 if is_left else 1.0), 0.0, 0.0
    ex = sx + ux * a_proj + nx * h_elbow
    ey = sy + uy * a_proj + ny * h_elbow
    ez = sz + uz * a_proj + nz * h_elbow
    joint_violated = (beta_deg <= 30.0) or (beta_deg >= 170.0) or (ey > sy + 110.0)
    effective_midline_x = cx + yaw_deg * 1.8
    if is_left:
        is_chest_penetrating = (ex > cx + 15.0 and ez > sz - 95.0)
        cross_depth = max(0.0, px - effective_midline_x)
    else:
        is_chest_penetrating = (ex < cx - 15.0 and ez > sz - 95.0)
        cross_depth = max(0.0, effective_midline_x - px)
    cross_cost = 6.0 * (math.exp(cross_depth / CROSS_BODY_EXP_SCALE_MM) - 1.0) if cross_depth > 0.0 else 0.0
    elbow_dev = max(0.0, abs(beta_deg - 95.0) - 25.0)
    elbow_limit_penalty = 80.0 if (beta_deg <= 30.0 or beta_deg >= 170.0 or joint_violated) else 0.0
    high_elbow_penalty = max(0.0, (ey - (sy + 35.0)) * 0.65)
    overreach_penalty = (120.0 + (dist_3d - max_reach) * 2.5) if dist_3d > max_reach else 0.0
    chest_penalty = 90.0 if is_chest_penetrating else 0.0
    return (
        0.04 * (elbow_dev ** 1.6)
        + elbow_limit_penalty
        + high_elbow_penalty
        + overreach_penalty
        + chest_penalty
        + cross_cost
    )


def solve_arm_ik(
    side: str,
    shoulder_xyz: Point3D,
    palm_xyz: Point3D,
    torso: TorsoState,
    l1_mm: float = ARM_L1_MM,
    l2_mm: float = ARM_L2_MM,
) -> ArmIKSolution:
    """Solve 3D 2-Link Inverse Kinematics for Left ('L') or Right ('R') arm with a
    standing player posture (shoulders near y = -35mm, z = 315mm; elbows hanging naturally
    downward/outward under gravity, reaching forward to the cabinet screen z = 0mm).

    Enforces:
      1. True 3D triangle law of cosines with natural gravity-directed elbow-down plane
         (Left elbow hangs downward & outward-left; Right elbow hangs downward & outward-right).
      2. Chest non-penetration constraint.
      3. Exponential cross-body cost as the palm crosses the torso midline.
    """
    sx, sy, sz = shoulder_xyz
    px, py, pz = palm_xyz

    dx = px - sx
    dy = py - sy
    dz = pz - sz

    planar_dist = math.hypot(dx, dy)
    dist_3d = math.sqrt(dx * dx + dy * dy + dz * dz)
    max_reach = l1_mm + l2_mm
    min_reach = max(ARM_MIN_REACH_MM, abs(l1_mm - l2_mm) + 15.0)

    is_reachable = (min_reach <= dist_3d <= max_reach)
    reach_margin = max(0.0, max_reach - dist_3d)

    # Clamp 3D distance for exact 3D law-of-cosines triangle solution
    d_clamped = max(min_reach, min(max_reach - 1.0, dist_3d))

    # Distance along S->P line to the elbow circle center, and circle radius h_elbow
    a_proj = (l1_mm * l1_mm - l2_mm * l2_mm + d_clamped * d_clamped) / (2.0 * d_clamped)
    h_elbow = math.sqrt(max(0.0, l1_mm * l1_mm - a_proj * a_proj))

    # Interior 3D elbow angle beta (deg)
    cos_beta = (l1_mm * l1_mm + l2_mm * l2_mm - d_clamped * d_clamped) / (2.0 * l1_mm * l2_mm)
    cos_beta = max(-1.0, min(1.0, cos_beta))
    beta_deg = math.degrees(math.acos(cos_beta))
    beta_deg = max(28.0, min(172.0, beta_deg))

    # Unit vector along S -> P in 3D
    if dist_3d > 1e-6:
        ux, uy, uz = dx / dist_3d, dy / dist_3d, dz / dist_3d
    else:
        ux, uy, uz = 0.0, 0.0, -1.0

    # Natural gravity-directed elbow reference vector:
    # Upper arm hangs downward (-Y) and outward (-X for L, +X for R) from the standing shoulder
    out_x = -0.65 if side.upper() == "L" else 0.65
    rx_ref, ry_ref, rz_ref = out_x, -0.88, -0.15

    # Gram-Schmidt orthogonal projection of r_ref perpendicular to u = (ux, uy, uz)
    dot_ru = rx_ref * ux + ry_ref * uy + rz_ref * uz
    nx = rx_ref - dot_ru * ux
    ny = ry_ref - dot_ru * uy
    nz = rz_ref - dot_ru * uz
    n_norm = math.sqrt(nx * nx + ny * ny + nz * nz)
    if n_norm > 1e-6:
        nx, ny, nz = nx / n_norm, ny / n_norm, nz / n_norm
    else:
        nx, ny, nz = (-1.0 if side.upper() == "L" else 1.0), 0.0, 0.0

    # 3D Elbow coordinates: circle center along S->P plus h_elbow along natural downward-outward normal
    mx = sx + ux * a_proj
    my = sy + uy * a_proj
    mz = sz + uz * a_proj
    ex = mx + nx * h_elbow
    ey = my + ny * h_elbow
    ez = mz + nz * h_elbow

    alpha = math.atan2(ey - sy, ex - sx)
    joint_violated = (beta_deg <= 30.0) or (beta_deg >= 170.0) or (ey > sy + 110.0)

    # Check chest penetration: upper arm / elbow must not cross inside the chest line
    # Torso midline X is torso.dx_mm; chest Y is torso.shoulder_center_xyz[1]
    cx = torso.dx_mm
    cy = torso.shoulder_center_xyz[1]
    if side.upper() == "L":
        is_chest_penetrating = (ex > cx + 15.0 and ez > sz - 95.0)
        # Cross-body depth: how far left palm crosses to the right of torso center line
        # Compensated partially by positive torso yaw (left shoulder forward)
        effective_midline_x = cx + torso.yaw_deg * 1.8
        cross_depth = max(0.0, px - effective_midline_x)
    else:
        is_chest_penetrating = (ex < cx - 15.0 and ez > sz - 95.0)
        effective_midline_x = cx + torso.yaw_deg * 1.8
        cross_depth = max(0.0, effective_midline_x - px)

    # Exponential cross-body cost as required by Section 3.1.2
    if cross_depth > 0.0:
        cross_cost = 6.0 * (math.exp(cross_depth / CROSS_BODY_EXP_SCALE_MM) - 1.0)
    else:
        cross_cost = 0.0

    # Joint comfort cost (optimal interior elbow angle around 75..125 deg)
    elbow_dev = max(0.0, abs(beta_deg - 95.0) - 25.0)
    elbow_limit_penalty = 0.0
    if beta_deg <= 30.0 or beta_deg >= 170.0 or joint_violated:
        elbow_limit_penalty = 80.0
    # High-elbow internal shoulder rotation penalty (when elbow rises above shoulder height)
    high_elbow_penalty = max(0.0, (ey - (sy + 35.0)) * 0.65)
    overreach_penalty = 0.0
    if dist_3d > max_reach:
        overreach_penalty = 120.0 + (dist_3d - max_reach) * 2.5
    chest_penalty = 90.0 if is_chest_penetrating else 0.0

    joint_cost = (
        0.04 * (elbow_dev ** 1.6)
        + elbow_limit_penalty
        + high_elbow_penalty
        + overreach_penalty
        + chest_penalty
        + cross_cost
    )

    return ArmIKSolution(
        side=side.upper(),
        shoulder_xyz=shoulder_xyz,
        elbow_xyz=(ex, ey, ez),
        palm_xyz=palm_xyz,
        shoulder_angle_deg=math.degrees(alpha),
        elbow_angle_deg=beta_deg,
        reach_distance_mm=planar_dist,
        reach_margin_mm=reach_margin,
        cross_depth_mm=cross_depth,
        cross_cost=cross_cost,
        joint_cost=joint_cost,
        is_reachable=is_reachable,
        is_chest_penetrating=is_chest_penetrating,
        is_joint_limit_violated=joint_violated,
    )


def segments_intersect_2d(
    a1: Point2D,
    a2: Point2D,
    b1: Point2D,
    b2: Point2D,
) -> bool:
    """Return True if 2D forearm line segment `Seg_L = (a1, a2)` and `Seg_R = (b1, b2)`
    strictly intersect in 2D (X-type forearm cross-lock).
    """
    ux, uy = a2[0] - a1[0], a2[1] - a1[1]
    vx, vy = b2[0] - b1[0], b2[1] - b1[1]
    wx, wy = a1[0] - b1[0], a1[1] - b1[1]
    denom = ux * vy - uy * vx
    if abs(denom) <= 1e-6:
        return False
    s = (vx * wy - vy * wx) / denom
    t = (ux * wy - uy * wx) / denom
    return (0.02 < s < 0.98) and (0.02 < t < 0.98)


def _segment_closest_points_2d(
    a1: Point2D,
    a2: Point2D,
    b1: Point2D,
    b2: Point2D,
) -> Tuple[float, float, float, Point2D]:
    """Compute 2D minimum distance between segment A (a1->a2) and segment B (b1->b2).
    Returns `(dist_2d_mm, u_a, u_b, midpoint_xy)` where `u_a, u_b in [0, 1]`.
    """
    ux, uy = a2[0] - a1[0], a2[1] - a1[1]
    vx, vy = b2[0] - b1[0], b2[1] - b1[1]
    wx, wy = a1[0] - b1[0], a1[1] - b1[1]

    # Check exact 2D segment intersection first
    denom = ux * vy - uy * vx
    if abs(denom) > 1e-6:
        s = (vx * wy - vy * wx) / denom
        t = (ux * wy - uy * wx) / denom
        if 0.0 <= s <= 1.0 and 0.0 <= t <= 1.0:
            ix = a1[0] + s * ux
            iy = a1[1] + s * uy
            return (0.0, s, t, (ix, iy))

    # Otherwise evaluate endpoint-to-segment projections
    def _pt_to_seg(p: Point2D, s1: Point2D, s2: Point2D) -> Tuple[float, float, Point2D]:
        dx, dy = s2[0] - s1[0], s2[1] - s1[1]
        l2 = dx * dx + dy * dy
        u = 0.0 if l2 < 1e-9 else max(0.0, min(1.0, ((p[0] - s1[0]) * dx + (p[1] - s1[1]) * dy) / l2))
        qx, qy = s1[0] + u * dx, s1[1] + u * dy
        return (math.hypot(p[0] - qx, p[1] - qy), u, (qx, qy))

    d_a1, ub1, q_b1 = _pt_to_seg(a1, b1, b2)
    d_a2, ub2, q_b2 = _pt_to_seg(a2, b1, b2)
    d_b1, ua1, q_a1 = _pt_to_seg(b1, a1, a2)
    d_b2, ua2, q_a2 = _pt_to_seg(b2, a1, a2)

    candidates = [
        (d_a1, 0.0, ub1, (0.5 * (a1[0] + q_b1[0]), 0.5 * (a1[1] + q_b1[1]))),
        (d_a2, 1.0, ub2, (0.5 * (a2[0] + q_b2[0]), 0.5 * (a2[1] + q_b2[1]))),
        (d_b1, ua1, 0.0, (0.5 * (q_a1[0] + b1[0]), 0.5 * (q_a1[1] + b1[1]))),
        (d_b2, ua2, 1.0, (0.5 * (q_a2[0] + b2[0]), 0.5 * (q_a2[1] + b2[1]))),
    ]
    return min(candidates, key=lambda item: item[0])


def resolve_arm_capsule_collision_and_layers(
    left_arm: ArmIKSolution,
    right_arm: ArmIKSolution,
) -> Tuple[ArmIKSolution, ArmIKSolution, str, float, Optional[Point2D], bool]:
    """Evaluate 3D capsule collision volume between Left and Right forearms/palms
    and determine upper/lower Z-layer hierarchy (`NORMAL`, `L_OVER_R`, `R_OVER_L`).

    Forearm physical radius = 34 mm (sum of two forearm radii = 68 mm).
    When forearms cross in 2D (d_2d < 68 mm), one arm must arch OVER the other in Z
    by at least sqrt(68^2 - d_2d^2) mm. If both wrists are pinned at z=0 near the crossing,
    a true 3D arm collision is flagged.
    """
    e_l = (left_arm.elbow_xyz[0], left_arm.elbow_xyz[1])
    p_l = (left_arm.palm_xyz[0], left_arm.palm_xyz[1])
    e_r = (right_arm.elbow_xyz[0], right_arm.elbow_xyz[1])
    p_r = (right_arm.palm_xyz[0], right_arm.palm_xyz[1])

    d_2d, u_l, u_r, cross_xy = _segment_closest_points_2d(e_l, p_l, e_r, p_r)
    forearm_sum_r = 68.0  # 2 * 34mm forearm radius

    zl_at_u = (1.0 - u_l) * left_arm.elbow_xyz[2] + u_l * left_arm.palm_xyz[2]
    zr_at_u = (1.0 - u_r) * right_arm.elbow_xyz[2] + u_r * right_arm.palm_xyz[2]
    dz = zl_at_u - zr_at_u
    dist_3d = math.hypot(d_2d, dz)

    if d_2d >= forearm_sum_r:
        return (left_arm, right_arm, "NORMAL", dist_3d, None, False)

    # Forearms overlap/cross in 2D projection! Determine which arm goes OVER (Upper Layer)
    # Prefer the arm with higher palm Z, or higher y target, or higher natural z at u
    if abs(left_arm.palm_xyz[2] - right_arm.palm_xyz[2]) > 12.0:
        l_over_r = left_arm.palm_xyz[2] > right_arm.palm_xyz[2]
    elif abs(left_arm.palm_xyz[1] - right_arm.palm_xyz[1]) > 25.0:
        # Arm reaching to the upper screen naturally passes over the arm on the lower screen
        l_over_r = left_arm.palm_xyz[1] > right_arm.palm_xyz[1]
    else:
        l_over_r = dz >= 0.0

    layer_order = "L_OVER_R" if l_over_r else "R_OVER_L"
    req_dz = math.sqrt(max(0.0, forearm_sum_r * forearm_sum_r - d_2d * d_2d))

    # Can the upper arm lift its elbow in Z to clear the lower forearm?
    # At parameter u (0=elbow, 1=palm), lifting the elbow by +delta_ez lifts the forearm at u by (1 - u)*delta_ez.
    # If u > 0.85 and both palms are pressed at z=0, the wrists collide and cannot arch!
    both_palms_low = (left_arm.palm_xyz[2] < 20.0) and (right_arm.palm_xyz[2] < 20.0)
    if l_over_r:
        cur_dz = zl_at_u - zr_at_u
        if cur_dz < req_dz:
            needed = req_dz - cur_dz
            lift_ez = min(95.0, needed / max(0.15, 1.0 - u_l))
            el_x, el_y, el_z = left_arm.elbow_xyz
            left_arm.elbow_xyz = (el_x, el_y, el_z + lift_ez)
            cur_dz += (1.0 - u_l) * lift_ez
        dist_3d = math.hypot(d_2d, cur_dz)
    else:
        cur_dz = zr_at_u - zl_at_u
        if cur_dz < req_dz:
            needed = req_dz - cur_dz
            lift_ez = min(95.0, needed / max(0.15, 1.0 - u_r))
            er_x, er_y, er_z = right_arm.elbow_xyz
            right_arm.elbow_xyz = (er_x, er_y, er_z + lift_ez)
            cur_dz += (1.0 - u_r) * lift_ez
        dist_3d = math.hypot(d_2d, cur_dz)

    is_arm_collision = (both_palms_low and dist_3d < forearm_sum_r - 12.0) or (both_palms_low and u_l > 0.82 and u_r > 0.82 and d_2d < 52.0)
    return (left_arm, right_arm, layer_order, dist_3d, cross_xy, is_arm_collision)


def check_hand_collision(
    left_palm_xyz: Point3D,
    right_palm_xyz: Point3D,
    left_is_holding: bool = False,
    right_is_holding: bool = False,
) -> Tuple[float, float, bool]:
    """Check Doraemon round palm / held-fingertip collision constraint:
    - Two flat striking palms: when |z_L - z_R| < 35 mm, ||P_L - P_R|| >= 85 mm.
    - Held hand(s): player anchors with fingertips (r=16mm, arched wrist) and can shift
      within the sensor edge by up to 28mm away from the other hand.
    """
    dx = left_palm_xyz[0] - right_palm_xyz[0]
    dy = left_palm_xyz[1] - right_palm_xyz[1]
    dist_xy = math.hypot(dx, dy)
    z_diff = abs(left_palm_xyz[2] - right_palm_xyz[2])
    if left_is_holding or right_is_holding:
        both_hold = left_is_holding and right_is_holding
        shift_gain = (1.6 * HOLD_EDGE_SHIFT_MAX_MM) if both_hold else HOLD_EDGE_SHIFT_MAX_MM
        min_clear = (2.0 * R_HOLD_FINGERTIP_MM) if both_hold else (R_HOLD_FINGERTIP_MM + R_STRIKE_FINGER_CLEARANCE_MM)
        eff_dist = dist_xy + shift_gain
        is_collision = (z_diff < HAND_COLLISION_Z_DIFF_MM) and (eff_dist < min_clear) and (dist_xy < 24.0)
        return (dist_xy, z_diff, is_collision)
    is_collision = (z_diff < HAND_COLLISION_Z_DIFF_MM) and (dist_xy < HAND_COLLISION_MIN_DIST_MM)
    return (dist_xy, z_diff, is_collision)


@_njit(cache=True)
def _pt_to_seg_2d_njit(px: float, py: float, s1x: float, s1y: float, s2x: float, s2y: float) -> Tuple[float, float, float, float]:
    dx = s2x - s1x
    dy = s2y - s1y
    l2 = dx * dx + dy * dy
    if l2 < 1e-9:
        u = 0.0
    else:
        u = ((px - s1x) * dx + (py - s1y) * dy) / l2
        if u < 0.0:
            u = 0.0
        elif u > 1.0:
            u = 1.0
    qx = s1x + u * dx
    qy = s1y + u * dy
    return (math.hypot(px - qx, py - qy), u, qx, qy)


@_njit(cache=True)
def _seg_closest_2d_njit(
    a1x: float, a1y: float, a2x: float, a2y: float,
    b1x: float, b1y: float, b2x: float, b2y: float,
) -> Tuple[float, float, float, float, float, bool]:
    ux = a2x - a1x
    uy = a2y - a1y
    vx = b2x - b1x
    vy = b2y - b1y
    wx = a1x - b1x
    wy = a1y - b1y
    denom = ux * vy - uy * vx
    strict_inter = False
    if abs(denom) > 1e-6:
        s = (vx * wy - vy * wx) / denom
        t = (ux * wy - uy * wx) / denom
        if 0.02 < s < 0.98 and 0.02 < t < 0.98:
            strict_inter = True
        if 0.0 <= s <= 1.0 and 0.0 <= t <= 1.0:
            return (0.0, s, t, a1x + s * ux, a1y + s * uy, strict_inter)

    d1, ub1, qb1x, qb1y = _pt_to_seg_2d_njit(a1x, a1y, b1x, b1y, b2x, b2y)
    best_d = d1
    best_ua = 0.0
    best_ub = ub1
    best_mx = 0.5 * (a1x + qb1x)
    best_my = 0.5 * (a1y + qb1y)

    d2, ub2, qb2x, qb2y = _pt_to_seg_2d_njit(a2x, a2y, b1x, b1y, b2x, b2y)
    if d2 < best_d:
        best_d, best_ua, best_ub, best_mx, best_my = d2, 1.0, ub2, 0.5 * (a2x + qb2x), 0.5 * (a2y + qb2y)

    d3, ua1, qa1x, qa1y = _pt_to_seg_2d_njit(b1x, b1y, a1x, a1y, a2x, a2y)
    if d3 < best_d:
        best_d, best_ua, best_ub, best_mx, best_my = d3, ua1, 0.0, 0.5 * (qa1x + b1x), 0.5 * (qa1y + b1y)

    d4, ua2, qa2x, qa2y = _pt_to_seg_2d_njit(b2x, b2y, a1x, a1y, a2x, a2y)
    if d4 < best_d:
        best_d, best_ua, best_ub, best_mx, best_my = d4, ua2, 1.0, 0.5 * (qa2x + b2x), 0.5 * (qa2y + b2y)

    return (best_d, best_ua, best_ub, best_mx, best_my, strict_inter)


@_njit(cache=True)
def _arm_ik_scalar_njit(
    is_left: bool,
    sx: float, sy: float, sz: float,
    px: float, py: float, pz: float,
    cx: float, yaw_deg: float,
):
    l1_mm = ARM_L1_MM
    l2_mm = ARM_L2_MM
    dx = px - sx
    dy = py - sy
    dz = pz - sz
    planar_dist = math.hypot(dx, dy)
    dist_3d = math.sqrt(dx * dx + dy * dy + dz * dz)
    max_reach = l1_mm + l2_mm
    min_reach = ARM_MIN_REACH_MM
    is_reachable = (min_reach <= dist_3d <= max_reach)
    reach_margin = max_reach - dist_3d if max_reach > dist_3d else 0.0
    d_clamped = dist_3d
    if d_clamped < min_reach:
        d_clamped = min_reach
    elif d_clamped > max_reach - 1.0:
        d_clamped = max_reach - 1.0

    a_proj = (l1_mm * l1_mm - l2_mm * l2_mm + d_clamped * d_clamped) / (2.0 * d_clamped)
    rem = l1_mm * l1_mm - a_proj * a_proj
    h_elbow = math.sqrt(rem) if rem > 0.0 else 0.0
    cos_beta = (l1_mm * l1_mm + l2_mm * l2_mm - d_clamped * d_clamped) / (2.0 * l1_mm * l2_mm)
    if cos_beta < -1.0:
        cos_beta = -1.0
    elif cos_beta > 1.0:
        cos_beta = 1.0
    beta_deg = math.degrees(math.acos(cos_beta))
    if beta_deg < 28.0:
        beta_deg = 28.0
    elif beta_deg > 172.0:
        beta_deg = 172.0

    if dist_3d > 1e-6:
        ux, uy, uz = dx / dist_3d, dy / dist_3d, dz / dist_3d
    else:
        ux, uy, uz = 0.0, 0.0, -1.0

    out_x = -0.65 if is_left else 0.65
    rx_ref, ry_ref, rz_ref = out_x, -0.88, -0.15
    dot_ru = rx_ref * ux + ry_ref * uy + rz_ref * uz
    nx = rx_ref - dot_ru * ux
    ny = ry_ref - dot_ru * uy
    nz = rz_ref - dot_ru * uz
    n_norm = math.sqrt(nx * nx + ny * ny + nz * nz)
    if n_norm > 1e-6:
        nx, ny, nz = nx / n_norm, ny / n_norm, nz / n_norm
    else:
        nx, ny, nz = (-1.0 if is_left else 1.0), 0.0, 0.0

    ex = sx + ux * a_proj + nx * h_elbow
    ey = sy + uy * a_proj + ny * h_elbow
    ez = sz + uz * a_proj + nz * h_elbow
    alpha_deg = math.degrees(math.atan2(ey - sy, ex - sx))
    joint_violated = (beta_deg <= 30.0) or (beta_deg >= 170.0) or (ey > sy + 110.0)

    effective_midline_x = cx + yaw_deg * 1.8
    if is_left:
        is_chest = (ex > cx + 15.0) and (ez > sz - 95.0)
        cross_depth = px - effective_midline_x if px > effective_midline_x else 0.0
    else:
        is_chest = (ex < cx - 15.0) and (ez > sz - 95.0)
        cross_depth = effective_midline_x - px if effective_midline_x > px else 0.0

    cross_cost = 6.0 * (math.exp(cross_depth / CROSS_BODY_EXP_SCALE_MM) - 1.0) if cross_depth > 0.0 else 0.0
    elbow_dev = abs(beta_deg - 95.0) - 25.0
    if elbow_dev < 0.0:
        elbow_dev = 0.0
    elbow_pen = 80.0 if (beta_deg <= 30.0 or beta_deg >= 170.0 or joint_violated) else 0.0
    high_el = (ey - (sy + 35.0)) * 0.65 if ey > sy + 35.0 else 0.0
    over_pen = (120.0 + (dist_3d - max_reach) * 2.5) if dist_3d > max_reach else 0.0
    chest_pen = 90.0 if is_chest else 0.0
    joint_cost = 0.04 * (elbow_dev ** 1.6) + elbow_pen + high_el + over_pen + chest_pen + cross_cost
    return (
        ex, ey, ez, alpha_deg, beta_deg, planar_dist, reach_margin,
        cross_depth, cross_cost, is_reachable, is_chest, joint_violated,
    ), joint_cost


@_njit(cache=True)
def _coupled_body_ik_njit(
    lx: float, ly: float, lz: float,
    rx: float, ry: float, rz: float,
    has_prev: bool,
    prev_dx: float, prev_dy: float, prev_yaw: float, prev_yv: float,
    dt: float,
    resp_hz: float,
    max_torso_ang_accel: float,
    l_hold: bool,
    r_hold: bool,
):
    # 1. Ideal torso pose
    lw = 1.35 if lz <= Z_RELEASE_MM else 0.65
    rw = 1.35 if rz <= Z_RELEASE_MM else 0.65
    w_sum = lw + rw
    mean_x = (lw * lx + rw * rx) / w_sum
    mean_y = (lw * ly + rw * ry) / w_sum
    opt_dx = max(TORSO_DX_LIMIT_MM[0], min(TORSO_DX_LIMIT_MM[1], mean_x * 0.42))
    opt_dy = max(TORSO_DY_LIMIT_MM[0], min(TORSO_DY_LIMIT_MM[1], mean_y * 0.16))
    cross_l = lx + 25.0 if lx > -25.0 else 0.0
    cross_r = 25.0 - rx if rx < 25.0 else 0.0
    desired_yaw = max(TORSO_YAW_LIMIT_DEG[0], min(TORSO_YAW_LIMIT_DEG[1], (cross_l - cross_r) * 0.14 + (ly - ry) * 0.045))

    cy_sh = SHOULDER_NEUTRAL_Y_MM + opt_dy
    cz_sh = SHOULDER_NEUTRAL_Z_MM
    half_w = 0.5 * W_SHOULDER_MM
    reg_xy = 0.002 * (opt_dx * opt_dx + opt_dy * opt_dy)
    best_yaw = desired_yaw
    best_eval_cost = 1e18
    for k_yaw in (-1, 0, 1):
        cand_yaw = desired_yaw + 6.0 * k_yaw
        if cand_yaw < TORSO_YAW_LIMIT_DEG[0]:
            cand_yaw = TORSO_YAW_LIMIT_DEG[0]
        elif cand_yaw > TORSO_YAW_LIMIT_DEG[1]:
            cand_yaw = TORSO_YAW_LIMIT_DEG[1]
        rad = math.radians(cand_yaw)
        cos_y = math.cos(rad)
        sin_y = math.sin(rad)
        sl_x = opt_dx - half_w * cos_y
        sl_y = cy_sh + half_w * sin_y * 0.65
        sl_z = max(60.0, cz_sh - half_w * sin_y * 0.45)
        sr_x = opt_dx + half_w * cos_y
        sr_y = cy_sh - half_w * sin_y * 0.65
        sr_z = max(60.0, cz_sh + half_w * sin_y * 0.45)
        _, cl = _arm_ik_scalar_njit(True, sl_x, sl_y, sl_z, lx, ly, lz, opt_dx, cand_yaw)
        _, cr = _arm_ik_scalar_njit(False, sr_x, sr_y, sr_z, rx, ry, rz, opt_dx, cand_yaw)
        ctot = cl + cr + reg_xy + 0.04 * (cand_yaw * cand_yaw)
        if ctot < best_eval_cost:
            best_eval_cost = ctot
            best_yaw = cand_yaw
    opt_yaw = best_yaw

    # 2. Torso temporal response
    vx, vy, yaw_vel, yaw_accel = 0.0, 0.0, 0.0, 0.0
    if has_prev and dt > 1e-4:
        alpha_filt = min(1.0, max(0.15, dt * resp_hz * 2.2))
        dx_mm = prev_dx + alpha_filt * (opt_dx - prev_dx)
        dy_mm = prev_dy + alpha_filt * (opt_dy - prev_dy)
        yaw_deg = prev_yaw + alpha_filt * (opt_yaw - prev_yaw)
        vx = (dx_mm - prev_dx) / dt
        vy = (dy_mm - prev_dy) / dt
        yaw_vel = (yaw_deg - prev_yaw) / dt
        eff_dt = dt if dt > 0.18 else 0.18
        yaw_accel = (yaw_vel - prev_yv) / eff_dt
    else:
        dx_mm, dy_mm, yaw_deg = opt_dx, opt_dy, opt_yaw

    rad_f = math.radians(yaw_deg)
    cos_f = math.cos(rad_f)
    sin_f = math.sin(rad_f)
    cy_f = SHOULDER_NEUTRAL_Y_MM + dy_mm
    sl_x = dx_mm - half_w * cos_f
    sl_y = cy_f + half_w * sin_f * 0.65
    sl_z = max(60.0, cz_sh - half_w * sin_f * 0.45)
    sr_x = dx_mm + half_w * cos_f
    sr_y = cy_f - half_w * sin_f * 0.65
    sr_z = max(60.0, cz_sh + half_w * sin_f * 0.45)

    lz_ik = max(lz, Z_HOLD_ARCHED_PALM_MM) if l_hold else lz
    rz_ik = max(rz, Z_HOLD_ARCHED_PALM_MM) if r_hold else rz

    l_tup, l_jcost = _arm_ik_scalar_njit(True, sl_x, sl_y, sl_z, lx, ly, lz_ik, dx_mm, yaw_deg)
    r_tup, r_jcost = _arm_ik_scalar_njit(False, sr_x, sr_y, sr_z, rx, ry, rz_ik, dx_mm, yaw_deg)
    el_x, el_y, el_z = l_tup[0], l_tup[1], l_tup[2]
    er_x, er_y, er_z = r_tup[0], r_tup[1], r_tup[2]

    # 3. Palm/fingertip collision
    dist_xy = math.hypot(lx - rx, ly - ry)
    z_diff = abs(lz - rz)
    if l_hold or r_hold:
        both_h = l_hold and r_hold
        shift_gain = 1.6 * HOLD_EDGE_SHIFT_MAX_MM if both_h else HOLD_EDGE_SHIFT_MAX_MM
        min_clear = 2.0 * R_HOLD_FINGERTIP_MM if both_h else (R_HOLD_FINGERTIP_MM + R_STRIKE_FINGER_CLEARANCE_MM)
        is_palm_coll = (z_diff < HAND_COLLISION_Z_DIFF_MM) and (dist_xy + shift_gain < min_clear) and (dist_xy < 24.0)
    else:
        is_palm_coll = (z_diff < HAND_COLLISION_Z_DIFF_MM) and (dist_xy < HAND_COLLISION_MIN_DIST_MM)

    # 4. Forearm capsule collision & layer order
    d_2d, u_l, u_r, cross_x, cross_y, inter_2d = _seg_closest_2d_njit(el_x, el_y, lx, ly, er_x, er_y, rx, ry)
    forearm_sum_r = 68.0
    zl_at_u = (1.0 - u_l) * el_z + u_l * lz_ik
    zr_at_u = (1.0 - u_r) * er_z + u_r * rz_ik
    dz_u = zl_at_u - zr_at_u
    dist_3d = math.hypot(d_2d, dz_u)
    layer_code = 0  # 0=NORMAL, 1=L_OVER_R, -1=R_OVER_L
    has_cross_pt = False
    is_arm_coll = False
    if d_2d < forearm_sum_r:
        has_cross_pt = True
        if abs(lz_ik - rz_ik) > 12.0:
            l_over_r = lz_ik > rz_ik
        elif abs(ly - ry) > 25.0:
            l_over_r = ly > ry
        else:
            l_over_r = dz_u >= 0.0
        layer_code = 1 if l_over_r else -1
        req_dz = math.sqrt(max(0.0, forearm_sum_r * forearm_sum_r - d_2d * d_2d))
        both_palms_low = (lz_ik < 20.0) and (rz_ik < 20.0)
        if l_over_r:
            cur_dz = zl_at_u - zr_at_u
            if cur_dz < req_dz:
                needed = req_dz - cur_dz
                lift_ez = min(95.0, needed / max(0.15, 1.0 - u_l))
                el_z += lift_ez
                cur_dz += (1.0 - u_l) * lift_ez
                # Upper hand naturally arches wrist/metacarpals (+26mm) unless hands are deeply crossed
                if lx - rx <= 140.0:
                    cur_dz += 26.0 * u_l
            dist_3d = math.hypot(d_2d, cur_dz)
        else:
            cur_dz = zr_at_u - zl_at_u
            if cur_dz < req_dz:
                needed = req_dz - cur_dz
                lift_ez = min(95.0, needed / max(0.15, 1.0 - u_r))
                er_z += lift_ez
                cur_dz += (1.0 - u_r) * lift_ez
                if lx - rx <= 140.0:
                    cur_dz += 26.0 * u_r
            dist_3d = math.hypot(d_2d, cur_dz)
        hands_crossed_x = (lx - rx > 110.0)
        is_arm_coll = (both_palms_low and hands_crossed_x and dist_3d < forearm_sum_r - 12.0) or (
            both_palms_low and hands_crossed_x and u_l > 0.82 and u_r > 0.82 and d_2d < 52.0
        )

    _, _, _, _, _, inter_xz = _seg_closest_2d_njit(el_x, el_z, lx, lz_ik, er_x, er_z, rx, rz_ik)
    vert_sep = abs(ly - ry)
    both_crossed_midline = (
        lx > 75.0
        and rx < -75.0
        and (vert_sep < 220.0 or (lx > 140.0 and rx < -140.0))
    )
    forearm_intersect = both_crossed_midline and (
        inter_2d or inter_xz or (d_2d < 45.0 and 0.05 < u_l < 0.95 and 0.05 < u_r < 0.95)
    )

    both_engaged = (lz <= Z_RELEASE_MM + 35.0) and (rz <= Z_RELEASE_MM + 35.0)
    severe_horiz_cross = (
        both_engaged and (lx - rx > 250.0) and (lx > 135.0 and rx < -135.0)
    ) or (both_engaged and forearm_intersect)

    max_ang = max_torso_ang_accel * 1.45
    prev_yaw_ref = prev_yaw if has_prev else 0.0
    is_whiplash = (
        abs(yaw_accel) > max_ang
        and abs(opt_yaw - prev_yaw_ref) > 28.0
        and (l_tup[7] > 145.0 and r_tup[7] > 145.0)
    )
    is_extreme_cross = severe_horiz_cross or forearm_intersect or is_whiplash or (l_tup[10] and r_tup[10])
    is_collision = is_palm_coll or is_arm_coll
    posture_cost = (
        l_jcost
        + r_jcost
        + (250.0 if is_collision else 0.0)
        + (850.0 if forearm_intersect else 0.0)
        + (500.0 if is_extreme_cross else 0.0)
    )
    return (
        dx_mm, dy_mm, yaw_deg, vx, vy, yaw_vel, yaw_accel,
        sl_x, sl_y, sl_z, el_x, el_y, el_z, l_tup[3], l_tup[4], l_tup[5], l_tup[6], l_tup[7], l_tup[8], l_jcost, l_tup[9], l_tup[10], l_tup[11],
        sr_x, sr_y, sr_z, er_x, er_y, er_z, r_tup[3], r_tup[4], r_tup[5], r_tup[6], r_tup[7], r_tup[8], r_jcost, r_tup[9], r_tup[10], r_tup[11],
        dist_xy, z_diff, is_collision, is_palm_coll, is_extreme_cross, is_whiplash, posture_cost,
        layer_code, dist_3d, has_cross_pt, cross_x, cross_y, forearm_intersect,
    )


@_njit(cache=True)
def solve_beam_ik_step_njit(
    lx: float, ly: float, lz: float,
    rx: float, ry: float, rz: float,
    has_prev: bool,
    prev_dx: float, prev_dy: float, prev_yaw: float, prev_yv: float,
    dt: float,
    resp_hz: float,
    max_torso_ang_accel: float,
    l_hold: bool,
    r_hold: bool,
) -> Tuple[float, float, float, float, float, float, float, float, float, float, float, int]:
    """Compact beam-search step evaluator returning 11 floats + 1 packed int bitmask:
    `(dx, dy, yaw, vx, vy, yv, ya, posture_cost, dist_xy, z_diff, forearm_3d, flags)`
    where `flags`: bit0 = is_collision, bit1 = is_palm_coll, bit2 = is_extreme_cross.
    Avoids boxing 52 tuple fields and 4 Python dataclasses on pruned beam candidates.
    """
    res = _coupled_body_ik_njit(
        lx, ly, lz, rx, ry, rz,
        has_prev, prev_dx, prev_dy, prev_yaw, prev_yv,
        dt, resp_hz, max_torso_ang_accel, l_hold, r_hold,
    )
    flags = (1 if res[41] else 0) | (2 if res[42] else 0) | (4 if res[43] else 0)
    return (
        res[0], res[1], res[2], res[3], res[4], res[5], res[6],
        res[45], res[39], res[40], res[47], flags,
    )


@lru_cache(maxsize=8192)
def _solve_ideal_torso_pose_cached(
    left_palm_xyz: Point3D,
    right_palm_xyz: Point3D,
) -> Tuple[float, float, float]:
    """Exact cached solver for `(opt_dx, opt_dy, opt_yaw)` given bilateral palm targets."""
    lx, ly, lz = left_palm_xyz
    rx, ry, rz = right_palm_xyz
    l_weight = 1.35 if lz <= Z_RELEASE_MM else 0.65
    r_weight = 1.35 if rz <= Z_RELEASE_MM else 0.65
    w_sum = l_weight + r_weight
    mean_x = (l_weight * lx + r_weight * rx) / w_sum
    mean_y = (l_weight * ly + r_weight * ry) / w_sum

    desired_dx = max(TORSO_DX_LIMIT_MM[0], min(TORSO_DX_LIMIT_MM[1], mean_x * 0.42))
    desired_dy = max(TORSO_DY_LIMIT_MM[0], min(TORSO_DY_LIMIT_MM[1], mean_y * 0.16))

    cross_l = max(0.0, lx - (-25.0))
    cross_r = max(0.0, 25.0 - rx)
    reach_asym = (ly - ry) * 0.045
    desired_yaw = (cross_l - cross_r) * 0.14 + reach_asym
    desired_yaw = max(TORSO_YAW_LIMIT_DEG[0], min(TORSO_YAW_LIMIT_DEG[1], desired_yaw))

    best_tuple = (desired_dx, desired_dy, desired_yaw)
    best_eval_cost = float("inf")
    cy_sh = SHOULDER_NEUTRAL_Y_MM + desired_dy
    cz_sh = SHOULDER_NEUTRAL_Z_MM
    half_w = 0.5 * W_SHOULDER_MM
    reg_xy = 0.002 * (desired_dx * desired_dx + desired_dy * desired_dy)
    for d_yaw in (-6.0, 0.0, 6.0):
        cand_yaw = max(TORSO_YAW_LIMIT_DEG[0], min(TORSO_YAW_LIMIT_DEG[1], desired_yaw + d_yaw))
        rad = math.radians(cand_yaw)
        cos_y = math.cos(rad)
        sin_y = math.sin(rad)
        sl_x = desired_dx - half_w * cos_y
        sl_y = cy_sh + half_w * sin_y * 0.65
        sl_z = max(60.0, cz_sh - half_w * sin_y * 0.45)
        sr_x = desired_dx + half_w * cos_y
        sr_y = cy_sh - half_w * sin_y * 0.65
        sr_z = max(60.0, cz_sh + half_w * sin_y * 0.45)
        c_l = _arm_joint_cost_scalar(True, sl_x, sl_y, sl_z, lx, ly, lz, desired_dx, cand_yaw)
        c_r = _arm_joint_cost_scalar(False, sr_x, sr_y, sr_z, rx, ry, rz, desired_dx, cand_yaw)
        c_total = c_l + c_r + reg_xy + 0.04 * (cand_yaw * cand_yaw)
        if c_total < best_eval_cost:
            best_eval_cost = c_total
            best_tuple = (desired_dx, desired_dy, cand_yaw)
    return best_tuple


def solve_coupled_body_ik(
    time: float,
    left_palm_xyz: Point3D,
    right_palm_xyz: Point3D,
    prev_torso: Optional[TorsoState] = None,
    dt: float = 0.1,
    profile: Optional[SkillProfileConfig] = None,
    left_is_holding: bool = False,
    right_is_holding: bool = False,
) -> BodyKinematicsState:
    """Solve coupled 3-DoF Torso + Left/Right 2-Link Arm IK for given bilateral palm targets.

    Computes the biomechanically optimal torso shift `(dx, dy)` and waist rotation `yaw_deg`
    subject to physiological limits and the player profile's `torso_response_rate`.
    """
    lx, ly, lz = left_palm_xyz
    rx, ry, rz = right_palm_xyz
    resp_hz = profile.torso_response_rate if profile is not None else 6.0
    max_ang_accel = profile.max_torso_angular_accel if profile is not None else 680.0
    has_prev = prev_torso is not None
    p_dx = prev_torso.dx_mm if prev_torso is not None else 0.0
    p_dy = prev_torso.dy_mm if prev_torso is not None else 0.0
    p_yaw = prev_torso.yaw_deg if prev_torso is not None else 0.0
    p_yv = prev_torso.yaw_vel_deg_s if prev_torso is not None else 0.0

    res = _coupled_body_ik_njit(
        float(lx), float(ly), float(lz),
        float(rx), float(ry), float(rz),
        has_prev, p_dx, p_dy, p_yaw, p_yv,
        float(dt), float(resp_hz), float(max_ang_accel),
        bool(left_is_holding), bool(right_is_holding),
    )

    torso = TorsoState(
        dx_mm=res[0],
        dy_mm=res[1],
        yaw_deg=res[2],
        vx_mm_s=res[3],
        vy_mm_s=res[4],
        yaw_vel_deg_s=res[5],
        yaw_accel_deg_s2=res[6],
    )
    left_arm = ArmIKSolution(
        side="L",
        shoulder_xyz=(res[7], res[8], res[9]),
        elbow_xyz=(res[10], res[11], res[12]),
        palm_xyz=left_palm_xyz,
        shoulder_angle_deg=res[13],
        elbow_angle_deg=res[14],
        reach_distance_mm=res[15],
        reach_margin_mm=res[16],
        cross_depth_mm=res[17],
        cross_cost=res[18],
        joint_cost=res[19],
        is_reachable=res[20],
        is_chest_penetrating=res[21],
        is_joint_limit_violated=res[22],
    )
    right_arm = ArmIKSolution(
        side="R",
        shoulder_xyz=(res[23], res[24], res[25]),
        elbow_xyz=(res[26], res[27], res[28]),
        palm_xyz=right_palm_xyz,
        shoulder_angle_deg=res[29],
        elbow_angle_deg=res[30],
        reach_distance_mm=res[31],
        reach_margin_mm=res[32],
        cross_depth_mm=res[33],
        cross_cost=res[34],
        joint_cost=res[35],
        is_reachable=res[36],
        is_chest_penetrating=res[37],
        is_joint_limit_violated=res[38],
    )
    layer_code = res[46]
    layer_order = "L_OVER_R" if layer_code == 1 else ("R_OVER_L" if layer_code == -1 else "NORMAL")
    cross_xy = (res[49], res[50]) if res[48] else None

    return BodyKinematicsState(
        time=time,
        torso=torso,
        left_arm=left_arm,
        right_arm=right_arm,
        hand_distance_xy_mm=res[39],
        hand_z_diff_mm=res[40],
        is_hand_collision=res[41],
        is_extreme_cross_arm=res[43],
        is_torso_whiplash=res[44],
        posture_cost=res[45],
        arm_layer_order=layer_order,
        forearm_min_dist_3d_mm=res[47],
        crossing_point_xy=cross_xy,
        is_forearm_intersection=res[51],
    )


@_njit(cache=True)
def compute_arm_layers_batch_njit(
    left_xyz: np.ndarray,
    right_xyz: np.ndarray,
    torso_dx_dy_yaw: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized Numba loop computing `arm_layer` (0=NORMAL, 1=L_OVER_R, -1=R_OVER_L)
    and 3D forearm distance across T frames without Python object allocation.
    """
    n = left_xyz.shape[0]
    layers = np.zeros(n, dtype=np.int8)
    f_dists = np.zeros(n, dtype=np.float32)
    half_w = 0.5 * W_SHOULDER_MM
    cz_sh = SHOULDER_NEUTRAL_Z_MM
    for i in range(n):
        dx_mm = torso_dx_dy_yaw[i, 0]
        dy_mm = torso_dx_dy_yaw[i, 1]
        yaw_deg = torso_dx_dy_yaw[i, 2]
        rad_f = math.radians(yaw_deg)
        cos_f = math.cos(rad_f)
        sin_f = math.sin(rad_f)
        cy_f = SHOULDER_NEUTRAL_Y_MM + dy_mm
        sl_x = dx_mm - half_w * cos_f
        sl_y = cy_f + half_w * sin_f * 0.65
        sl_z = max(60.0, cz_sh - half_w * sin_f * 0.45)
        sr_x = dx_mm + half_w * cos_f
        sr_y = cy_f - half_w * sin_f * 0.65
        sr_z = max(60.0, cz_sh + half_w * sin_f * 0.45)
        lx, ly, lz = left_xyz[i, 0], left_xyz[i, 1], left_xyz[i, 2]
        rx, ry, rz = right_xyz[i, 0], right_xyz[i, 1], right_xyz[i, 2]
        l_tup, _ = _arm_ik_scalar_njit(True, sl_x, sl_y, sl_z, lx, ly, lz, dx_mm, yaw_deg)
        r_tup, _ = _arm_ik_scalar_njit(False, sr_x, sr_y, sr_z, rx, ry, rz, dx_mm, yaw_deg)
        d_2d, u_l, u_r, _, _, _ = _seg_closest_2d_njit(l_tup[0], l_tup[1], lx, ly, r_tup[0], r_tup[1], rx, ry)
        zl_u = (1.0 - u_l) * l_tup[2] + u_l * lz
        zr_u = (1.0 - u_r) * r_tup[2] + u_r * rz
        dz_u = zl_u - zr_u
        f_dists[i] = math.hypot(d_2d, dz_u)
        if d_2d < 68.0:
            if abs(lz - rz) > 12.0:
                layers[i] = 1 if lz > rz else -1
            elif abs(ly - ry) > 25.0:
                layers[i] = 1 if ly > ry else -1
            else:
                layers[i] = 1 if dz_u >= 0.0 else -1
    return layers, f_dists
