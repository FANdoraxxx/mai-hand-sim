"""Arcade cabinet 2.5D millimeter geometry, 33-zone touch sensor polygons,
8 outer buttons, exact circle-polygon overlap area, and slide trajectory geometry.
"""
from __future__ import annotations

import bisect
from functools import lru_cache
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from sim_core.config import (
    BUTTON_ANGLE_STEP_DEG,
    BUTTON_BASE_ANGLE_DEG,
    HAND_COLLISION_MIN_DIST_MM,
    MIN_SENSOR_OVERLAP_AREA_MM2,
    OUTER_BUTTON_RADIUS_MM,
    OUTER_INNER_RADIAL_DIFF_MM,
    R_A_CENTER_MM,
    R_B_CENTER_MM,
    R_BUTTON_MM,
    R_C_CENTROID_MM,
    R_C_MAX_MM,
    R_D_CENTER_MM,
    R_E_CENTER_MM,
    R_HAND_MM,
    R_SCREEN_MM,
)

Point2D = Tuple[float, float]
Point3D = Tuple[float, float, float]
Polygon2D = Tuple[Point2D, ...]


# ============================================================================
# 1. Angular & Coordinate Helpers
# ============================================================================
# Coordinate system:
# - Origin (0, 0) at screen center
# - +X points Right (towards buttons 2, 3)
# - +Y points Up (towards buttons 8, 1; 12 o'clock is +Y)
# - +Z points Outward from screen towards player (z=0 is screen contact)
# - Clockwise angle theta_deg from +Y (12 o'clock):
#     x = r * sin(radians(theta_deg))
#     y = r * cos(radians(theta_deg))

def button_angle_deg(button_idx: int) -> float:
    """Return clockwise angle in degrees from 12 o'clock (+Y) for button 1..8.

    theta_k = 22.5 + (k - 1) * 45.0 deg.
    """
    if not (1 <= button_idx <= 8):
        raise ValueError(f"button_idx must be in 1..8, got {button_idx}")
    return (BUTTON_BASE_ANGLE_DEG + (button_idx - 1) * BUTTON_ANGLE_STEP_DEG) % 360.0


def de_sector_angle_deg(sector_idx: int) -> float:
    """Return clockwise angle in degrees from 12 o'clock (+Y) for D1..D8 / E1..E8.

    phi_k = (k - 1) * 45.0 deg (D1/E1 at 0 deg top, D2/E2 at 45 deg between 1 & 2, ...).
    """
    if not (1 <= sector_idx <= 8):
        raise ValueError(f"sector_idx must be in 1..8, got {sector_idx}")
    return ((sector_idx - 1) * BUTTON_ANGLE_STEP_DEG) % 360.0


def polar_cw_to_xy(radius_mm: float, angle_cw_deg: float) -> Point2D:
    """Convert (radius, clockwise angle from +Y) to (x, y) in mm."""
    rad = math.radians(angle_cw_deg)
    return (radius_mm * math.sin(rad), radius_mm * math.cos(rad))


def rotate_cw_xy(pt: Point2D, angle_cw_deg: float) -> Point2D:
    """Rotate a 2D point clockwise by angle_cw_deg around origin (0, 0)."""
    rad = math.radians(angle_cw_deg)
    c = math.cos(rad)
    s = math.sin(rad)
    x, y = pt
    return (x * c + y * s, -x * s + y * c)


def rotate_polygon_cw(poly: Sequence[Point2D], angle_cw_deg: float) -> Polygon2D:
    return tuple(rotate_cw_xy(p, angle_cw_deg) for p in poly)


# ============================================================================
# 2. 33 Inner Touch Sensor Polygons (MajdataPlay JudgeAreas.fbx Topology)
# ============================================================================
# Base vertices extracted from MajdataPlay/Assets/Models/JudgeAreas.fbx (in Unity units
# where outer radius = 2.50 units -> scaled by 108.0 mm/unit so R_screen = 270.0 mm).
_FBX_SCALE_MM: float = R_SCREEN_MM / 2.50  # 108.0 mm per Unity unit

# In JudgeAreas.fbx:
# - Base 'C' is right half of center octagon; we scale slightly so octagon radius ~ 66 mm
_C_OCT_OUTER_MM: float = 66.0
_C_OCT_CORNER_MM: float = _C_OCT_OUTER_MM * math.tan(math.radians(22.5))  # ~27.34 mm
_C_GAP_HALF_MM: float = 2.5

# C1: right half of center octagon; C2: left half; C: full octagon (for wireframe & 'C' notes)
POLYGON_C1: Polygon2D = (
    (_C_GAP_HALF_MM, -_C_OCT_OUTER_MM),
    (_C_OCT_CORNER_MM, -_C_OCT_OUTER_MM),
    (_C_OCT_OUTER_MM, -_C_OCT_CORNER_MM),
    (_C_OCT_OUTER_MM, _C_OCT_CORNER_MM),
    (_C_OCT_CORNER_MM, _C_OCT_OUTER_MM),
    (_C_GAP_HALF_MM, _C_OCT_OUTER_MM),
)

POLYGON_C2: Polygon2D = tuple((-x, y) for x, y in reversed(POLYGON_C1))

POLYGON_C_FULL: Polygon2D = (
    (_C_OCT_CORNER_MM, _C_OCT_OUTER_MM),
    (_C_OCT_OUTER_MM, _C_OCT_CORNER_MM),
    (_C_OCT_OUTER_MM, -_C_OCT_CORNER_MM),
    (_C_OCT_CORNER_MM, -_C_OCT_OUTER_MM),
    (-_C_OCT_CORNER_MM, -_C_OCT_OUTER_MM),
    (-_C_OCT_OUTER_MM, -_C_OCT_CORNER_MM),
    (-_C_OCT_OUTER_MM, _C_OCT_CORNER_MM),
    (-_C_OCT_CORNER_MM, _C_OCT_OUTER_MM),
)

# Base 'B' in JudgeAreas.fbx is located at 67.5 deg CW from +Y (which is B2).
_BASE_B2_FBX: Tuple[Point2D, ...] = (
    (1.21330 * _FBX_SCALE_MM, 0.49071 * _FBX_SCALE_MM),
    (1.11550 * _FBX_SCALE_MM, 0.72522 * _FBX_SCALE_MM),
    (0.75656 * _FBX_SCALE_MM, 0.72522 * _FBX_SCALE_MM),
    (0.63384 * _FBX_SCALE_MM, 0.60033 * _FBX_SCALE_MM),
    (0.63384 * _FBX_SCALE_MM, 0.26710 * _FBX_SCALE_MM),
    (0.87733 * _FBX_SCALE_MM, 0.02168 * _FBX_SCALE_MM),
    (1.03892 * _FBX_SCALE_MM, 0.02168 * _FBX_SCALE_MM),
    (1.30092 * _FBX_SCALE_MM, 0.27957 * _FBX_SCALE_MM),
)
# Rotate -45 deg CW (i.e. 45 deg CCW) to get B1 at 22.5 deg CW from +Y
_BASE_B1_MM: Polygon2D = rotate_polygon_cw(_BASE_B2_FBX, -45.0)

# Base 'A' in JudgeAreas.fbx is located at 67.5 deg CW from +Y (which is A2).
_BASE_A2_FBX: Tuple[Point2D, ...] = (
    (2.12026 * _FBX_SCALE_MM, 1.29928 * _FBX_SCALE_MM),
    (2.00472 * _FBX_SCALE_MM, 1.47861 * _FBX_SCALE_MM),
    (1.30094 * _FBX_SCALE_MM, 0.95204 * _FBX_SCALE_MM),
    (1.30053 * _FBX_SCALE_MM, 0.72780 * _FBX_SCALE_MM),
    (1.43050 * _FBX_SCALE_MM, 0.41045 * _FBX_SCALE_MM),
    (1.59367 * _FBX_SCALE_MM, 0.24591 * _FBX_SCALE_MM),
    (2.46331 * _FBX_SCALE_MM, 0.36390 * _FBX_SCALE_MM),
    (2.43342 * _FBX_SCALE_MM, 0.54023 * _FBX_SCALE_MM),
    (2.31352 * _FBX_SCALE_MM, 0.92357 * _FBX_SCALE_MM),
)
_BASE_A1_MM: Polygon2D = rotate_polygon_cw(_BASE_A2_FBX, -45.0)

# Base 'E' in JudgeAreas.fbx is located at +X (90 deg CW from +Y, which is E3).
_BASE_E3_FBX: Tuple[Point2D, ...] = (
    (1.09890 * _FBX_SCALE_MM, 0.0),
    (1.43373 * _FBX_SCALE_MM, -0.33483 * _FBX_SCALE_MM),
    (1.76856 * _FBX_SCALE_MM, 0.0),
    (1.43373 * _FBX_SCALE_MM, 0.33483 * _FBX_SCALE_MM),
)
_BASE_E1_MM: Polygon2D = rotate_polygon_cw(_BASE_E3_FBX, -90.0)

# Base 'D' in JudgeAreas.fbx is located at +X (90 deg CW from +Y, which is D3).
_BASE_D3_FBX: Tuple[Point2D, ...] = (
    (2.49191 * _FBX_SCALE_MM, 0.00025 * _FBX_SCALE_MM),
    (2.47188 * _FBX_SCALE_MM, 0.32088 * _FBX_SCALE_MM),
    (2.06722 * _FBX_SCALE_MM, 0.26214 * _FBX_SCALE_MM),
    (1.63515 * _FBX_SCALE_MM, 0.19934 * _FBX_SCALE_MM),
    (1.82997 * _FBX_SCALE_MM, 0.00012 * _FBX_SCALE_MM),
    (1.63572 * _FBX_SCALE_MM, -0.19756 * _FBX_SCALE_MM),
    (2.06722 * _FBX_SCALE_MM, -0.26355 * _FBX_SCALE_MM),
    (2.47541 * _FBX_SCALE_MM, -0.31566 * _FBX_SCALE_MM),
)
_BASE_D1_MM: Polygon2D = rotate_polygon_cw(_BASE_D3_FBX, -90.0)


@dataclass(frozen=True)
class SensorZone:
    """Represents one of the 33 inner touch zones (or C union) or 8 outer buttons."""
    name: str
    group: str              # 'A', 'B', 'C', 'D', 'E', or 'OB' (Outer Button)
    index: int              # 1..8 (or 1..2 for C1/C2, 0 for C)
    center: Point2D         # Physical aiming / nominal center (x, y) in mm
    angle_cw_deg: float     # Clockwise angle from +Y (12 o'clock)
    radius_mm: float        # Radial distance of center from origin (mm)
    polygon: Polygon2D      # Boundary vertices in mm (counter-clockwise or clockwise)
    bounding_radius_mm: float


def _compute_bounding_radius(center: Point2D, poly: Polygon2D) -> float:
    cx, cy = center
    return max(math.hypot(px - cx, py - cy) for px, py in poly)


def _make_circle_polygon(center: Point2D, radius: float, n_verts: int = 16) -> Polygon2D:
    cx, cy = center
    return tuple(
        (
            cx + radius * math.cos(2.0 * math.pi * i / n_verts),
            cy + radius * math.sin(2.0 * math.pi * i / n_verts),
        )
        for i in range(n_verts)
    )


def _build_zones() -> Tuple[Dict[str, SensorZone], Dict[int, SensorZone]]:
    touch_zones: Dict[str, SensorZone] = {}
    outer_buttons: Dict[int, SensorZone] = {}

    # C1 & C2 (2 zones) + C alias
    c1_center = (R_C_CENTROID_MM, 0.0)
    c2_center = (-R_C_CENTROID_MM, 0.0)
    touch_zones["C1"] = SensorZone(
        name="C1",
        group="C",
        index=1,
        center=c1_center,
        angle_cw_deg=90.0,
        radius_mm=R_C_CENTROID_MM,
        polygon=POLYGON_C1,
        bounding_radius_mm=_compute_bounding_radius(c1_center, POLYGON_C1),
    )
    touch_zones["C2"] = SensorZone(
        name="C2",
        group="C",
        index=2,
        center=c2_center,
        angle_cw_deg=270.0,
        radius_mm=R_C_CENTROID_MM,
        polygon=POLYGON_C2,
        bounding_radius_mm=_compute_bounding_radius(c2_center, POLYGON_C2),
    )
    touch_zones["C"] = SensorZone(
        name="C",
        group="C",
        index=0,
        center=(0.0, 0.0),
        angle_cw_deg=0.0,
        radius_mm=0.0,
        polygon=POLYGON_C_FULL,
        bounding_radius_mm=R_C_MAX_MM,
    )

    # A1~A8, B1~B8, and Outer Buttons 1~8
    for k in range(1, 9):
        rot_deg = (k - 1) * 45.0
        ang_deg = button_angle_deg(k)

        # Bk
        b_poly = rotate_polygon_cw(_BASE_B1_MM, rot_deg)
        b_center = polar_cw_to_xy(R_B_CENTER_MM, ang_deg)
        touch_zones[f"B{k}"] = SensorZone(
            name=f"B{k}",
            group="B",
            index=k,
            center=b_center,
            angle_cw_deg=ang_deg,
            radius_mm=R_B_CENTER_MM,
            polygon=b_poly,
            bounding_radius_mm=_compute_bounding_radius(b_center, b_poly),
        )

        # Ak
        a_poly = rotate_polygon_cw(_BASE_A1_MM, rot_deg)
        a_center = polar_cw_to_xy(R_A_CENTER_MM, ang_deg)
        touch_zones[f"A{k}"] = SensorZone(
            name=f"A{k}",
            group="A",
            index=k,
            center=a_center,
            angle_cw_deg=ang_deg,
            radius_mm=R_A_CENTER_MM,
            polygon=a_poly,
            bounding_radius_mm=_compute_bounding_radius(a_center, a_poly),
        )

        # Outer Button k (R_button = 310 mm, aligned with A1~A8)
        ob_center = polar_cw_to_xy(R_BUTTON_MM, ang_deg)
        ob_poly = _make_circle_polygon(ob_center, OUTER_BUTTON_RADIUS_MM, 16)
        ob_zone = SensorZone(
            name=f"OB{k}",
            group="OB",
            index=k,
            center=ob_center,
            angle_cw_deg=ang_deg,
            radius_mm=R_BUTTON_MM,
            polygon=ob_poly,
            bounding_radius_mm=OUTER_BUTTON_RADIUS_MM,
        )
        outer_buttons[k] = ob_zone

    # D1~D8 and E1~E8 (angles 0, 45, ..., 315 deg CW from +Y)
    for k in range(1, 9):
        rot_deg = (k - 1) * 45.0
        ang_deg = de_sector_angle_deg(k)

        # Dk
        d_poly = rotate_polygon_cw(_BASE_D1_MM, rot_deg)
        d_center = polar_cw_to_xy(R_D_CENTER_MM, ang_deg)
        touch_zones[f"D{k}"] = SensorZone(
            name=f"D{k}",
            group="D",
            index=k,
            center=d_center,
            angle_cw_deg=ang_deg,
            radius_mm=R_D_CENTER_MM,
            polygon=d_poly,
            bounding_radius_mm=_compute_bounding_radius(d_center, d_poly),
        )

        # Ek
        e_poly = rotate_polygon_cw(_BASE_E1_MM, rot_deg)
        e_center = polar_cw_to_xy(R_E_CENTER_MM, ang_deg)
        touch_zones[f"E{k}"] = SensorZone(
            name=f"E{k}",
            group="E",
            index=k,
            center=e_center,
            angle_cw_deg=ang_deg,
            radius_mm=R_E_CENTER_MM,
            polygon=e_poly,
            bounding_radius_mm=_compute_bounding_radius(e_center, e_poly),
        )

    return touch_zones, outer_buttons


TOUCH_ZONES, OUTER_BUTTONS = _build_zones()

# Canonical 33 inner touch sensor names (C1, C2, B1~B8, A1~A8, D1~D8, E1~E8)
CANONICAL_33_SENSORS: Tuple[str, ...] = (
    "C1", "C2",
    *(f"B{k}" for k in range(1, 9)),
    *(f"A{k}" for k in range(1, 9)),
    *(f"D{k}" for k in range(1, 9)),
    *(f"E{k}" for k in range(1, 9)),
)


def get_sensor_zone(name: str) -> SensorZone:
    """Lookup a sensor zone by name ('A1'..'E8', 'C1', 'C2', 'C', or 'OB1'..'OB8')."""
    norm = name.strip().upper()
    if norm in TOUCH_ZONES:
        return TOUCH_ZONES[norm]
    if norm.startswith("OB") and norm[2:].isdigit():
        idx = int(norm[2:])
        if idx in OUTER_BUTTONS:
            return OUTER_BUTTONS[idx]
    if norm.isdigit() and 1 <= int(norm) <= 8:
        return TOUCH_ZONES[f"A{int(norm)}"]
    raise KeyError(f"Unknown sensor or button zone: {name!r}")


def get_button_target_xy(button_idx: int, use_outer: bool = True) -> Point2D:
    """Return (x, y) target in mm for button 1..8 in either Outer (310mm) or Inner A (215mm)."""
    if use_outer:
        return OUTER_BUTTONS[button_idx].center
    return TOUCH_ZONES[f"A{button_idx}"].center


# ============================================================================
# 3. Exact Circle-Polygon Intersection Area & Sensor Activation
# ============================================================================
def _triangle_circle_signed_area(ax: float, ay: float, bx: float, by: float, r: float) -> float:
    """Signed area of intersection between circle centered at origin (radius r)
    and directed triangle (O, A, B).
    """
    cross = ax * by - ay * bx
    if abs(cross) < 1e-12:
        return 0.0

    da2 = ax * ax + ay * ay
    db2 = bx * bx + by * by
    r2 = r * r

    # Both vertices inside circle -> pure triangle area
    if da2 <= r2 and db2 <= r2:
        return 0.5 * cross

    dx = bx - ax
    dy = by - ay
    ab2 = dx * dx + dy * dy
    if ab2 < 1e-12:
        return 0.0

    # Solve |A + t*(B - A)|^2 = r^2 for t in [0, 1]
    # ab2 * t^2 + 2*(A . d)*t + (da2 - r2) = 0
    b_half = ax * dx + ay * dy
    disc = b_half * b_half - ab2 * (da2 - r2)

    def _sector_area(px: float, py: float, qx: float, qy: float) -> float:
        theta = math.atan2(px * qy - py * qx, px * qx + py * qy)
        return 0.5 * r2 * theta

    if disc <= 0.0:
        # Segment stays outside circle -> pure circular sector
        return _sector_area(ax, ay, bx, by)

    sqrt_disc = math.sqrt(disc)
    t1 = (-b_half - sqrt_disc) / ab2
    t2 = (-b_half + sqrt_disc) / ab2

    if t2 <= 0.0 or t1 >= 1.0:
        return _sector_area(ax, ay, bx, by)

    t_enter = max(0.0, t1)
    t_exit = min(1.0, t2)
    if t_enter >= t_exit:
        return _sector_area(ax, ay, bx, by)

    p1x = ax + t_enter * dx
    p1y = ay + t_enter * dy
    p2x = ax + t_exit * dx
    p2y = ay + t_exit * dy

    area = 0.0
    if t_enter > 0.0:
        area += _sector_area(ax, ay, p1x, p1y)
    area += 0.5 * (p1x * p2y - p1y * p2x)
    if t_exit < 1.0:
        area += _sector_area(p2x, p2y, bx, by)
    return area


def circle_polygon_intersection_area(
    circle_center: Point2D,
    radius_mm: float,
    polygon: Polygon2D,
) -> float:
    """Compute exact area (mm^2) of overlap between a circle and a 2D polygon."""
    if radius_mm <= 0.0 or len(polygon) < 3:
        return 0.0
    cx, cy = circle_center
    total = 0.0
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        total += _triangle_circle_signed_area(x1 - cx, y1 - cy, x2 - cx, y2 - cy, radius_mm)
    return abs(total)


def point_to_polygon_distance(pt: Point2D, polygon: Polygon2D) -> float:
    """Signed/unsigned minimum distance from point pt to polygon (0.0 if inside)."""
    px, py = pt
    inside = False
    n = len(polygon)
    min_dist_sq = float("inf")
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        if ((y1 > py) != (y2 > py)) and (px < (x2 - x1) * (py - y1) / (y2 - y1 + 1e-15) + x1):
            inside = not inside
        dx = x2 - x1
        dy = y2 - y1
        seg_len_sq = dx * dx + dy * dy
        if seg_len_sq < 1e-12:
            t = 0.0
        else:
            t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / seg_len_sq))
        proj_x = x1 + t * dx
        proj_y = y1 + t * dy
        d2 = (px - proj_x) ** 2 + (py - proj_y) ** 2
        if d2 < min_dist_sq:
            min_dist_sq = d2
    return 0.0 if inside else math.sqrt(min_dist_sq)


def is_zone_triggered_by_palm(
    palm_xy: Point2D,
    palm_z_mm: float,
    zone_name: str,
    palm_radius_mm: float = R_HAND_MM,
    z_threshold_mm: float = 15.0,
    min_overlap_mm2: float = MIN_SENSOR_OVERLAP_AREA_MM2,
) -> bool:
    """Return True if a Doraemon round palm at (x, y, z) triggers the specified zone."""
    if palm_z_mm > z_threshold_mm:
        return False
    zone = get_sensor_zone(zone_name)
    dx = palm_xy[0] - zone.center[0]
    dy = palm_xy[1] - zone.center[1]
    d2 = dx * dx + dy * dy
    if d2 > (palm_radius_mm + zone.bounding_radius_mm) ** 2:
        return False
    # Fast exact inner-core short-circuit: when palm center is within 24mm of sensor centroid,
    # overlap area with any canonical Maimai sensor is >1800 mm^2 >> min_overlap_mm2 (120 mm^2).
    if palm_radius_mm >= 44.0 and min_overlap_mm2 <= 200.0 and d2 <= 576.0:
        return True
    area = circle_polygon_intersection_area(palm_xy, palm_radius_mm, zone.polygon)
    return area >= min_overlap_mm2


def get_active_sensors_for_palm(
    palm_xy: Point2D,
    palm_z_mm: float,
    palm_radius_mm: float = R_HAND_MM,
    z_threshold_mm: float = 15.0,
) -> set[str]:
    """Return set of all touch sensor names ('A1'..'E8', 'C1', 'C2', 'C') and outer buttons ('OB1'..'OB8')
    currently triggered by one palm.
    """
    if palm_z_mm > z_threshold_mm:
        return set()
    active: set[str] = set()
    for name in CANONICAL_33_SENSORS:
        if is_zone_triggered_by_palm(palm_xy, palm_z_mm, name, palm_radius_mm, z_threshold_mm):
            active.add(name)
    if "C1" in active or "C2" in active:
        active.add("C")
    for k, ob in OUTER_BUTTONS.items():
        if math.hypot(palm_xy[0] - ob.center[0], palm_xy[1] - ob.center[1]) <= palm_radius_mm + ob.bounding_radius_mm * 0.65:
            active.add(f"OB{k}")
    return active


# ============================================================================
# 4. Multi-Touch & Rub Feasibility (1-Palm and 2-Palm Polygon Covering)
# ============================================================================
@lru_cache(maxsize=131072)
def _find_single_palm_cover_cached(
    unique_names: Tuple[str, ...],
    palm_radius_mm: float,
) -> Optional[Point2D]:
    if not unique_names:
        return (0.0, 0.0)
    if len(unique_names) == 1:
        return get_sensor_zone(unique_names[0]).center

    zones = [get_sensor_zone(n) for n in unique_names]
    # simval: necessary condition - one palm touching two polygons needs polygon distance <= 2R; bound by centre distance - bounding radii
    _br = [max(math.hypot(px - z.center[0], py - z.center[1]) for px, py in z.polygon) for z in zones]
    for _i in range(len(zones)):
        for _j in range(_i + 1, len(zones)):
            if math.hypot(zones[_i].center[0] - zones[_j].center[0], zones[_i].center[1] - zones[_j].center[1]) - _br[_i] - _br[_j] > 2.0 * palm_radius_mm:
                return None
    avg_x = sum(z.center[0] for z in zones) / len(zones)
    avg_y = sum(z.center[1] for z in zones) / len(zones)

    closest_pts: List[Point2D] = []
    for z in zones:
        best_pt = z.center
        best_d = point_to_polygon_distance((avg_x, avg_y), z.polygon)
        if best_d == 0.0:
            best_pt = (avg_x, avg_y)
        else:
            min_d2 = float("inf")
            n = len(z.polygon)
            for i in range(n):
                x1, y1 = z.polygon[i]
                x2, y2 = z.polygon[(i + 1) % n]
                dx, dy = x2 - x1, y2 - y1
                l2 = dx * dx + dy * dy
                t = 0.0 if l2 < 1e-12 else max(0.0, min(1.0, ((avg_x - x1) * dx + (avg_y - y1) * dy) / l2))
                px, py = x1 + t * dx, y1 + t * dy
                d2 = (avg_x - px) ** 2 + (avg_y - py) ** 2
                if d2 < min_d2:
                    min_d2 = d2
                    best_pt = (px, py)
        closest_pts.append(best_pt)

    cand_x = sum(p[0] for p in closest_pts) / len(closest_pts)
    cand_y = sum(p[1] for p in closest_pts) / len(closest_pts)

    best_candidate = (cand_x, cand_y)
    for step in (12.0, 5.0, 2.0, 0.75):
        improved = True
        while improved:
            improved = False
            cur_max_d = max(point_to_polygon_distance(best_candidate, z.polygon) for z in zones)
            for ang in range(0, 360, 45):
                nx = best_candidate[0] + step * math.cos(math.radians(ang))
                ny = best_candidate[1] + step * math.sin(math.radians(ang))
                d = max(point_to_polygon_distance((nx, ny), z.polygon) for z in zones)
                if d + 1e-4 < cur_max_d:
                    cur_max_d = d
                    best_candidate = (nx, ny)
                    improved = True

    for z in zones:
        if point_to_polygon_distance(best_candidate, z.polygon) <= palm_radius_mm - 2.5:
            continue
        area = circle_polygon_intersection_area(best_candidate, palm_radius_mm, z.polygon)
        if area < MIN_SENSOR_OVERLAP_AREA_MM2:
            return None
    return best_candidate


def find_single_palm_cover_center(
    zone_names: Sequence[str],
    palm_radius_mm: float = R_HAND_MM,
) -> Optional[Point2D]:
    """Check if a single circular palm of radius `palm_radius_mm` can simultaneously
    overlap/trigger all zones in `zone_names` (e.g., rubbing A1+D1 or B1+B2).

    Returns the optimal palm center (x, y) if feasible, or None if impossible.
    """
    if not zone_names:
        return (0.0, 0.0)
    return _find_single_palm_cover_cached(tuple(sorted(dict.fromkeys(zone_names))), float(palm_radius_mm))  # simval: sorted -> cache hits across permutations


@lru_cache(maxsize=4096)
def _solve_two_palm_coverage_cached(
    unique: Tuple[str, ...],
    palm_radius_mm: float,
) -> Optional[Tuple[Point2D, Point2D, Tuple[str, ...], Tuple[str, ...]]]:
    m = len(unique)
    if m == 0:
        return ((-120.0, -50.0), (120.0, -50.0), (), ())
    if m == 1:
        c = get_sensor_zone(unique[0]).center
        if c[0] <= 0:
            return (c, (140.0, -60.0), (unique[0],), ())
        return ((-140.0, -60.0), c, (), (unique[0],))
    if m == 2:
        c0 = get_sensor_zone(unique[0]).center
        c1 = get_sensor_zone(unique[1]).center
        if c0[0] <= c1[0]:
            return (c0, c1, (unique[0],), (unique[1],))
        return (c1, c0, (unique[1],), (unique[0],))

    single_center = _find_single_palm_cover_cached(unique, palm_radius_mm)
    if single_center is not None:
        if single_center[0] <= 0.0:
            return (single_center, (135.0, -50.0), unique, ())
        return ((-135.0, -50.0), single_center, (), unique)

    best_solution = None
    best_cost = float("inf")
    if m > 12:
        return None

    for mask in range(1, (1 << m) - 1):
        if not (mask & 1):
            continue
        group_a = tuple(unique[i] for i in range(m) if (mask >> i) & 1)
        group_b = tuple(unique[i] for i in range(m) if not ((mask >> i) & 1))
        ca = _find_single_palm_cover_cached(group_a, palm_radius_mm)
        if ca is None:
            continue
        cb = _find_single_palm_cover_cached(group_b, palm_radius_mm)
        if cb is None:
            continue
        dist_ab = math.hypot(ca[0] - cb[0], ca[1] - cb[1])
        if dist_ab < HAND_COLLISION_MIN_DIST_MM:
            continue
        if ca[0] <= cb[0]:
            l_xy, r_xy, l_grp, r_grp = ca, cb, group_a, group_b
        else:
            l_xy, r_xy, l_grp, r_grp = cb, ca, group_b, group_a
        cost = max(0.0, l_xy[0] - r_xy[0]) * 5.0 + abs(len(l_grp) - len(r_grp)) * 10.0
        if cost < best_cost:
            best_cost = cost
            best_solution = (l_xy, r_xy, l_grp, r_grp)
    return best_solution


def solve_two_palm_coverage(
    target_zones: Sequence[str],
    palm_radius_mm: float = R_HAND_MM,
) -> Optional[Tuple[Point2D, Point2D, Tuple[str, ...], Tuple[str, ...]]]:
    """Check if `target_zones` (e.g. >= 3 targets) can be simultaneously covered by TWO
    Doraemon palms of radius `palm_radius_mm` without colliding.

    Returns (left_xy, right_xy, left_covered_zones, right_covered_zones) if feasible,
    or None if 3+ hands are required (THREE_HAND_CONFLICT).
    """
    return _solve_two_palm_coverage_cached(tuple(sorted(dict.fromkeys(target_zones))), float(palm_radius_mm))


# ============================================================================
# 5. Slide Sensor Queue & Trajectory Tables (MajdataPlay SlideTables.cs)
# ============================================================================
@dataclass(frozen=True)
class SlideAreaStep:
    """One step in MajdataPlay's SlideArea sensor queue."""
    areas: Tuple[str, ...]      # OR-ed sensor areas (e.g. ('A2', 'B2'))
    is_skippable: bool = True   # Skip-1 rule allowed if True
    is_last: bool = False       # Tail area flag


@dataclass(frozen=True)
class SlideRouteSpec:
    """Complete sensor queue, tail wait proportion (`const`), and 2D control waypoints."""
    shape: str
    start_btn: int
    end_btn: int
    via_btn: Optional[int]
    queue: Tuple[SlideAreaStep, ...]
    wifi_queues: Optional[Tuple[Tuple[SlideAreaStep, ...], Tuple[SlideAreaStep, ...], Tuple[SlideAreaStep, ...]]]
    const_proportion: float     # Fraction of slide duration spent on final sensor
    waypoints_mm: Tuple[Point2D, ...]


def _rot_sensor(sensor: str, diff: int) -> str:
    """Rotate sensor name ('A1'..'E8', 'C') clockwise by `diff` steps (each step = 45 deg)."""
    if sensor in ("C", "C1", "C2"):
        return "C"
    grp = sensor[0]
    idx = int(sensor[1:])
    new_idx = (idx - 1 + diff) % 8 + 1
    return f"{grp}{new_idx}"


def _mirror_sensor_a1(sensor: str) -> str:
    """Mirror sensor across the axis passing through A1-A5 (matches MajdataPlay SensorArea.Mirror(A1))."""
    if sensor in ("C", "C1", "C2"):
        return "C"
    grp = sensor[0]
    idx = int(sensor[1:])
    if grp in ("A", "B"):
        # 1->1, 2->8, 3->7, 4->6, 5->5, 6->4, 7->3, 8->2
        m_idx = (1 - idx) % 8 + 1
        return f"{grp}{m_idx}"
    # D / E: D1<->D2, D3<->D8, D4<->D7, D5<->D6
    de_map = {1: 2, 2: 1, 3: 8, 4: 7, 5: 6, 6: 5, 7: 4, 8: 3}
    return f"{grp}{de_map[idx]}"


def _transform_queue(
    base_steps: Sequence[Tuple[Sequence[str], bool]],
    mirror: bool,
    rot_diff: int,
) -> Tuple[SlideAreaStep, ...]:
    out: List[SlideAreaStep] = []
    n = len(base_steps)
    for i, (areas, skippable) in enumerate(base_steps):
        transformed: List[str] = []
        for a in areas:
            s = _mirror_sensor_a1(a) if mirror else a
            s = _rot_sensor(s, rot_diff)
            transformed.append(s)
        out.append(
            SlideAreaStep(
                areas=tuple(transformed),
                is_skippable=skippable,
                is_last=(i == n - 1),
            )
        )
    return tuple(out)


# Base queues from MajdataPlay/Assets/Scripts/Scenes/Game/Utils/SlideTables.cs (starting at A1)
_BASE_LINE_QUEUES: Dict[int, Tuple[List[Tuple[Tuple[str, ...], bool]], float]] = {
    3: ([ (("A1",), True), (("A2", "B2"), False), (("A3",), True) ], 0.182),
    4: ([ (("A1",), True), (("B2",), True), (("B3",), True), (("A4",), True) ], 0.190),
    5: ([ (("A1",), True), (("B1",), True), (("C",), True), (("B5",), True), (("A5",), True) ], 0.152),
    6: ([ (("A1",), True), (("B8",), True), (("B7",), True), (("A6",), True) ], 0.190),
    7: ([ (("A1",), True), (("A8", "B8"), False), (("A7",), True) ], 0.182),
}

# Circle CW from A1: dist = (end - 1) % 8
_BASE_CIRCLE_CONST: Dict[int, float] = {
    1: 0.465,  # 1->2
    2: 0.233,  # 1->3
    3: 0.155,  # 1->4
    4: 0.116,  # 1->5
    5: 0.093,  # 1->6
    6: 0.078,  # 1->7
    7: 0.066,  # 1->8
    0: 0.058,  # 1->1 full circle
}

_BASE_V_QUEUES: Dict[int, Tuple[List[Tuple[Tuple[str, ...], bool]], float]] = {
    k: (
        [ (("A1",), True), (("B1",), True), (("C",), True), ((f"B{k}",), True), ((f"A{k}",), True) ],
        0.185 if k == 1 else (0.150 if k == 2 else (0.154 if k == 8 else 0.158)),
    )
    for k in (1, 2, 3, 4, 6, 7, 8)
}

_BASE_PQ_QUEUES: Dict[int, Tuple[List[Tuple[Tuple[str, ...], bool]], float]] = {
    1: ([ (("A1",), True), (("B8",), True), (("B7",), True), (("B6",), True), (("B5",), True), (("B4",), True), (("B3",), True), (("B2",), True), (("A1",), True) ], 0.095),
    2: ([ (("A1",), True), (("B8",), True), (("B7",), True), (("B6",), True), (("B5",), True), (("B4",), True), (("B3",), True), (("A2",), True) ], 0.112),
    3: ([ (("A1",), True), (("B8",), True), (("B7",), True), (("B6",), True), (("B5",), True), (("B4",), True), (("A3",), True) ], 0.125),
    4: ([ (("A1",), True), (("B8",), True), (("B7",), True), (("B6",), True), (("B5",), True), (("A4",), True) ], 0.139),
    5: ([ (("A1",), True), (("B8",), True), (("B7",), True), (("B6",), True), (("A5",), True) ], 0.160),
    6: ([ (("A1",), True), (("B8",), True), (("B7",), True), (("B6",), True), (("B5",), True), (("B4",), True), (("B3",), True), (("B2",), True), (("B1",), True), (("B8",), True), (("B7",), True), (("A6",), True) ], 0.080),
    7: ([ (("A1",), True), (("B8",), True), (("B7",), True), (("B6",), True), (("B5",), True), (("B4",), True), (("B3",), True), (("B2",), True), (("B1",), True), (("B8",), True), (("A7",), True) ], 0.084),
    8: ([ (("A1",), True), (("B8",), True), (("B7",), True), (("B6",), True), (("B5",), True), (("B4",), True), (("B3",), True), (("B2",), True), (("B1",), True), (("A8",), True) ], 0.0895),
}

_BASE_PPQQ_QUEUES: Dict[int, Tuple[List[Tuple[Tuple[str, ...], bool]], float]] = {
    1: ([ (("A1",), True), (("B1",), True), (("C",), True), (("B4",), True), (("A3",), True), (("A2",), True), (("A1",), True) ], 0.065),
    2: ([ (("A1",), True), (("B1",), True), (("C",), True), (("B4",), True), (("A3",), True), (("A2",), True) ], 0.086),
    3: ([ (("A1",), True), (("B1",), True), (("C",), True), (("B4",), True), (("A3",), True) ], 0.157),
    4: ([ (("A1",), True), (("B1",), True), (("C",), True), (("B4",), True), (("A3",), True), (("A2",), True), (("B1",), True), (("C",), True), (("B4",), True), (("A4",), True) ], 0.065),
    5: ([ (("A1",), True), (("B1",), True), (("C",), True), (("B4",), True), (("A3",), True), (("A2",), True), (("B1",), True), (("C",), True), (("B5",), True), (("A5",), True) ], 0.065),
    6: ([ (("A1",), True), (("B1",), True), (("C",), True), (("B4",), True), (("A3",), True), (("A2",), True), (("B1",), True), (("C", "B8"), True), (("B7", "B6"), True), (("A6",), True) ], 0.067),
    7: ([ (("A1",), True), (("B1",), True), (("C",), True), (("B4",), True), (("A3",), True), (("A2",), True), (("B1",), True), (("B8",), True), (("A7",), True) ], 0.079),
    8: ([ (("A1",), True), (("B1",), True), (("C",), True), (("B4",), True), (("A3",), True), (("A2",), True), (("B1", "A1"), True), (("A8",), True) ], 0.0626),
}

_BASE_L_QUEUES: Dict[int, Tuple[List[Tuple[Tuple[str, ...], bool]], float]] = {
    2: ([ (("A1",), True), (("B8", "A8"), False), (("A7",), True), (("B8",), True), (("B1",), True), (("A2",), True) ], 0.100),
    3: ([ (("A1",), True), (("B8", "A8"), False), (("A7",), True), (("B7",), True), (("C",), True), (("B3",), True), (("A3",), True) ], 0.104),
    4: ([ (("A1",), True), (("B8", "A8"), False), (("A7",), True), (("B6",), True), (("B5",), True), (("A4",), True) ], 0.098),
    5: ([ (("A1",), True), (("B8", "A8"), False), (("A7",), True), (("B6", "A6"), False), (("A5",), True) ], 0.105),
}

_BASE_S_QUEUE: Tuple[List[Tuple[Tuple[str, ...], bool]], float] = (
    [ (("A1",), True), (("B8",), True), (("B7",), True), (("C",), True), (("B3",), True), (("B4",), True), (("A5",), True) ],
    0.130,
)

_BASE_WIFI_QUEUES: Tuple[
    List[Tuple[Tuple[str, ...], bool]],
    List[Tuple[Tuple[str, ...], bool]],
    List[Tuple[Tuple[str, ...], bool]],
    float,
] = (
    [ (("A1",), True), (("B8",), True), (("B7",), True), (("A6", "D6"), True) ],  # Left
    [ (("A1",), True), (("B1",), True), (("C",), True), (("A5", "B5"), True) ],   # Center
    [ (("A1",), True), (("B2",), True), (("B3",), True), (("A4", "D5"), True) ],  # Right
    0.162870,
)


@lru_cache(maxsize=2048)
def build_slide_route(
    shape: str,
    start_btn: int,
    end_btn: int,
    via_btn: Optional[int] = None,
) -> SlideRouteSpec:
    """Build the exact MajdataPlay SlideRouteSpec (sensor queue + trajectory waypoints)
    for any valid maimai slide segment.
    """
    rot = (start_btn - 1) % 8
    d_cw = (end_btn - start_btn) % 8
    d_ccw = (start_btn - end_btn) % 8

    wifi_queues = None
    if shape == "w":
        l_base, c_base, r_base, c_prop = _BASE_WIFI_QUEUES
        q_l = _transform_queue(l_base, False, rot)
        q_c = _transform_queue(c_base, False, rot)
        q_r = _transform_queue(r_base, False, rot)
        wifi_queues = (q_l, q_c, q_r)
        queue = q_c
        const_prop = c_prop
    elif shape == "-":
        target = d_cw + 1
        base_q, const_prop = _BASE_LINE_QUEUES.get(target, _BASE_LINE_QUEUES[5])
        queue = _transform_queue(base_q, False, rot)
    elif shape in ("^", ">", "<"):
        # Determine CW vs CCW direction per Simai rules:
        # '^' takes the shorter arc; '>' is CW at top (1,2,7,8) and CCW at bottom (3,4,5,6);
        # '<' is CCW at top (1,2,7,8) and CW at bottom (3,4,5,6).
        if shape == "^":
            is_cw = d_cw <= 4
        elif shape == ">":
            is_cw = start_btn in (1, 2, 7, 8)
        else:  # '<'
            is_cw = start_btn in (3, 4, 5, 6)
        steps = (d_cw if is_cw else d_ccw)
        if steps == 0:
            steps = 8
        base_q = []
        for i in range(steps + 1):
            skippable = True
            if steps == 1 and i == 0:
                skippable = False
            elif steps == 2 and i == 1:
                skippable = False
            base_q.append(((f"A{i % 8 + 1}",), skippable))
        const_prop = _BASE_CIRCLE_CONST[steps % 8]
        queue = _transform_queue(base_q, not is_cw, rot)
    elif shape == "v":
        target = d_cw + 1
        base_q, const_prop = _BASE_V_QUEUES.get(target, _BASE_V_QUEUES[3])
        queue = _transform_queue(base_q, False, rot)
    elif shape in ("p", "q"):
        is_mirror = (shape == "q")
        target = (d_ccw if is_mirror else d_cw) + 1
        base_q, const_prop = _BASE_PQ_QUEUES[target]
        queue = _transform_queue(base_q, is_mirror, rot)
    elif shape in ("pp", "qq"):
        is_mirror = (shape == "qq")
        target = (d_ccw if is_mirror else d_cw) + 1
        base_q, const_prop = _BASE_PPQQ_QUEUES[target]
        queue = _transform_queue(base_q, is_mirror, rot)
    elif shape in ("s", "z"):
        is_mirror = (shape == "z")
        base_q, const_prop = _BASE_S_QUEUE
        queue = _transform_queue(base_q, is_mirror, rot)
    elif shape == "V":
        # Grand V through via_btn (either start-2 or start+2)
        via_offset = ((via_btn or ((start_btn + 5) % 8 + 1)) - start_btn) % 8
        is_mirror = (via_offset == 2)
        target = (d_ccw if is_mirror else d_cw) + 1
        target = max(2, min(5, target))
        base_q, const_prop = _BASE_L_QUEUES[target]
        queue = _transform_queue(base_q, is_mirror, rot)
    else:
        # Fallback linear interpolation through start -> end
        base_q, const_prop = _BASE_LINE_QUEUES[5]
        queue = _transform_queue(base_q, False, rot)

    # Build smooth 2D trajectory waypoints from sensor queue centers
    waypoints: List[Point2D] = []
    for step in queue:
        pts = [get_sensor_zone(a).center for a in step.areas]
        wx = sum(p[0] for p in pts) / len(pts)
        wy = sum(p[1] for p in pts) / len(pts)
        waypoints.append((wx, wy))

    return SlideRouteSpec(
        shape=shape,
        start_btn=start_btn,
        end_btn=end_btn,
        via_btn=via_btn,
        queue=queue,
        wifi_queues=wifi_queues,
        const_proportion=const_prop,
        waypoints_mm=tuple(waypoints),
    )


@lru_cache(maxsize=2048)
def _waypoint_cum_lengths(waypoints: Tuple[Point2D, ...]) -> Tuple[Tuple[float, ...], float]:
    cum: List[float] = [0.0]
    total = 0.0
    for i in range(len(waypoints) - 1):
        x1, y1 = waypoints[i]
        x2, y2 = waypoints[i + 1]
        total += math.hypot(x2 - x1, y2 - y1)
        cum.append(total)
    return (tuple(cum), total)


def interpolate_waypoints(waypoints: Sequence[Point2D], progress: float) -> Point2D:
    """Piecewise-linear arc-length-parameterized interpolation along slide waypoints for progress in [0, 1]."""
    if not waypoints:
        return (0.0, 0.0)
    if len(waypoints) == 1 or progress <= 0.0:
        return waypoints[0]
    if progress >= 1.0:
        return waypoints[-1]

    wp_tuple = waypoints if isinstance(waypoints, tuple) else tuple(waypoints)
    cum, total_len = _waypoint_cum_lengths(wp_tuple)
    if total_len < 1e-6:
        return waypoints[-1]

    target_d = progress * total_len
    idx = max(0, min(len(wp_tuple) - 2, bisect.bisect_right(cum, target_d) - 1))
    seg_d = cum[idx + 1] - cum[idx]
    t = 0.0 if seg_d < 1e-9 else (target_d - cum[idx]) / seg_d
    x1, y1 = wp_tuple[idx]
    x2, y2 = wp_tuple[idx + 1]
    return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))
