"""Centralized physical, biomechanical, and MajdataPlay judgement constants."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Tuple

# ============================================================================
# 1. Arcade Cabinet & Screen Physical Geometry (Millimeters & Degrees)
# ============================================================================
SCREEN_DIAMETER_MM: float = 540.0
R_SCREEN_MM: float = 270.0

# 8 primary button angles (clockwise from 12 o'clock / +Y axis):
# theta_k = 22.5 + (k - 1) * 45.0 deg, k in 1..8
BUTTON_BASE_ANGLE_DEG: float = 22.5
BUTTON_ANGLE_STEP_DEG: float = 45.0

# Radial centers of the 33 inner touch zones & 8 outer buttons (mm)
R_C_MAX_MM: float = 70.0          # C1, C2 center octagon radius (~0 to 70 mm)
R_C_CENTROID_MM: float = 28.0     # C1 / C2 half-octagon centroid offset
R_B_CENTER_MM: float = 125.0      # B1 ~ B8 inner ring pentagons
R_E_CENTER_MM: float = 165.0      # E1 ~ E8 diamond transition zones
R_A_CENTER_MM: float = 215.0      # A1 ~ A8 outer ring touch sectors
R_D_CENTER_MM: float = 235.0      # D1 ~ D8 outer boundary wedges
R_BUTTON_MM: float = 310.0        # Outer physical buttons 1~8

# Radial clearance between Outer Button (310mm) and Inner A Zone (215mm)
OUTER_INNER_RADIAL_DIFF_MM: float = R_BUTTON_MM - R_A_CENTER_MM  # 95.0 mm
OUTER_BUTTON_RADIUS_MM: float = 32.0

# ============================================================================
# 2. Upper Body, 2-Link Arm & Doraemon Round Palm Biomechanics
# ============================================================================
# Standing human posture aligned with Maimai DX cabinet:
# - Screen center is (0, 0, 0), screen radius = 270 mm (top = +270 mm, bottom = -270 mm).
# - Player stands upright in front of the cabinet:
#   * Head / eye level at y ≈ +265 mm (aligned with upper-middle to top section of the screen)
#   * Shoulders at y ≈ +65 mm (middle-upper chest level), z ≈ 315 mm in front of screen
#   * Elbows hang naturally downward under gravity (y ≈ -125 .. -210 mm)
#   * Pelvis / waist anchor at y = -380 mm
PELVIS_ANCHOR_MM: Tuple[float, float] = (0.0, -380.0)
SHOULDER_NEUTRAL_Y_MM: float = 65.0
SHOULDER_NEUTRAL_Z_MM: float = 315.0
HEAD_NEUTRAL_Y_MM: float = 265.0
W_SHOULDER_MM: float = 400.0  # Bi-deltoid / bi-humeral shoulder width

# Torso 3-DoF limits
TORSO_DX_LIMIT_MM: Tuple[float, float] = (-120.0, 120.0)   # Lateral shift
TORSO_DY_LIMIT_MM: Tuple[float, float] = (-100.0, 60.0)    # Vertical dip / rise
TORSO_YAW_LIMIT_DEG: Tuple[float, float] = (-35.0, 35.0)   # Waist yaw rotation

# 2-Link arm segment lengths (mm)
ARM_L1_MM: float = 300.0  # Upper arm (shoulder to elbow)
ARM_L2_MM: float = 295.0  # Forearm (elbow to palm center)
ARM_MAX_REACH_MM: float = ARM_L1_MM + ARM_L2_MM  # 540.0 mm planar max
ARM_MIN_REACH_MM: float = 55.0                   # Minimum folded planar distance

# Physiological joint angle limits (degrees)
# Elbow interior angle beta in [18 deg, 176 deg]; directional bend enforced per arm
ELBOW_MIN_ANGLE_DEG: float = 18.0
ELBOW_MAX_ANGLE_DEG: float = 176.0
CHEST_CLEARANCE_Y_OFFSET_MM: float = 45.0
CROSS_BODY_EXP_SCALE_MM: float = 65.0  # Exponential penalty scale when crossing midline

# Doraemon Round Palm (mm)
R_HAND_MM: float = 48.0
Z_CONTACT_MM: float = 0.0
Z_RELEASE_MM: float = 15.0        # z > 15 mm considered released / hovering
Z_TAP_REBOUND_MM: float = 42.0    # Nominal lift height between rapid taps
Z_HOVER_DEFAULT_MM: float = 148.0 # Idle / long-hover height (matches HUD reference)

# Hand collision constraints
HAND_COLLISION_Z_DIFF_MM: float = 35.0
HAND_COLLISION_EPS_MM: float = 11.0
HAND_COLLISION_MIN_DIST_MM: float = 2.0 * R_HAND_MM - HAND_COLLISION_EPS_MM  # 85.0 mm

# Realistic Held-Hand Biomechanics (Fingertip Anchor + Edge Shift + Same-Hand Span)
R_HOLD_FINGERTIP_MM: float = 16.0          # Fingertip contact footprint radius while holding
R_STRIKE_FINGER_CLEARANCE_MM: float = 26.0 # Effective clearance of an incoming striking hand near a fingertip hold
Z_HOLD_ARCHED_PALM_MM: float = 36.0        # Arched wrist/palm Z height when holding with fingertips
HOLD_EDGE_SHIFT_MAX_MM: float = 28.0       # Max offset of holding fingertips within sensor / toward outer edge
SAME_HAND_HOLD_SPAN_MM: float = 108.0      # Thumb-to-finger/heel span for tapping adjacent inner sensor while holding
HOLD_TAIL_EARLY_RELEASE_MAX_SEC: float = 0.165  # Safe early-release allowance within MajdataPlay's 200ms tail-ignore window

# Minimum overlap area (mm^2) or distance margin for circle-polygon touch trigger
MIN_SENSOR_OVERLAP_AREA_MM2: float = 65.0

# ============================================================================
# 3. MajdataPlay Judgement Engine Constants (60 FPS base = 16.6667 ms/frame)
# ============================================================================
FRAME_LENGTH_SEC: float = 1.0 / 60.0
FRAME_LENGTH_MSEC: float = 1000.0 / 60.0
SIM_TICK_HZ: int = 180  # 3 ticks per 60Hz frame (matches MaiMuriDX / MajdataPlay)

# Tap / Break / Hold head windows (ms)
TAP_JUDGE_CRITICAL_MSEC: float = 1.0 * FRAME_LENGTH_MSEC      # 16.67 ms
TAP_JUDGE_PERFECT_2ND_MSEC: float = 2.0 * FRAME_LENGTH_MSEC   # 33.33 ms
TAP_JUDGE_PERFECT_3RD_MSEC: float = 3.0 * FRAME_LENGTH_MSEC   # 50.00 ms
TAP_JUDGE_GREAT_1ST_MSEC: float = 4.0 * FRAME_LENGTH_MSEC     # 66.67 ms
TAP_JUDGE_GREAT_2ND_MSEC: float = 5.0 * FRAME_LENGTH_MSEC     # 83.33 ms
TAP_JUDGE_GREAT_3RD_MSEC: float = 6.0 * FRAME_LENGTH_MSEC     # 100.00 ms
TAP_JUDGE_GOOD_MSEC: float = 9.0 * FRAME_LENGTH_MSEC          # 150.00 ms

# Touch / TouchHold head windows (ms)
TOUCH_JUDGE_CRITICAL_MSEC: float = 9.0 * FRAME_LENGTH_MSEC       # 150.00 ms (Fast & Late)
TOUCH_JUDGE_PERFECT_2ND_MSEC: float = 10.5 * FRAME_LENGTH_MSEC   # 175.00 ms (Late only)
TOUCH_JUDGE_PERFECT_3RD_MSEC: float = 12.0 * FRAME_LENGTH_MSEC   # 200.00 ms (Late only)
TOUCH_JUDGE_GREAT_1ST_MSEC: float = 13.0 * FRAME_LENGTH_MSEC     # 216.67 ms
TOUCH_JUDGE_GREAT_2ND_MSEC: float = 14.0 * FRAME_LENGTH_MSEC     # 233.33 ms
TOUCH_JUDGE_GREAT_3RD_MSEC: float = 15.0 * FRAME_LENGTH_MSEC     # 250.00 ms
TOUCH_JUDGE_GOOD_MSEC: float = 18.0 * FRAME_LENGTH_MSEC          # 300.00 ms

# Hold / TouchHold body check constants (sec)
HOLD_HEAD_IGNORE_SEC: float = 6.0 * FRAME_LENGTH_SEC             # 100.0 ms
HOLD_TAIL_IGNORE_SEC: float = 12.0 * FRAME_LENGTH_SEC            # 200.0 ms
TOUCH_HOLD_HEAD_IGNORE_SEC: float = 15.0 * FRAME_LENGTH_SEC      # 250.0 ms
TOUCH_HOLD_TAIL_IGNORE_SEC: float = 12.0 * FRAME_LENGTH_SEC      # 200.0 ms
DELUXE_HOLD_RELEASE_IGNORE_SEC: float = 2.0 * FRAME_LENGTH_SEC   # 33.33 ms tolerance

# Slide tail judgement windows (ms)
SLIDE_JUDGE_MAX_EXT_MSEC: float = 22.0 * FRAME_LENGTH_MSEC       # 366.67 ms
SLIDE_JUDGE_BASE_PERFECT_MSEC: float = 14.0 * FRAME_LENGTH_MSEC  # 233.33 ms
SLIDE_JUDGE_GREAT_1ST_MSEC: float = 21.0 * FRAME_LENGTH_MSEC     # 350.00 ms
SLIDE_JUDGE_GREAT_2ND_MSEC: float = 25.0 * FRAME_LENGTH_MSEC     # 416.67 ms
SLIDE_JUDGE_GREAT_3RD_MSEC: float = 29.0 * FRAME_LENGTH_MSEC     # 483.33 ms
SLIDE_JUDGE_GOOD_MSEC: float = 36.0 * FRAME_LENGTH_MSEC          # 600.00 ms
SLIDE_CHECKABLE_LEAD_SEC: float = 0.050                          # 50 ms before star launch

# MajdataPlay DX Score weights
SCORE_WEIGHT_TAP: int = 500
SCORE_WEIGHT_TOUCH: int = 500
SCORE_WEIGHT_HOLD: int = 1000
SCORE_WEIGHT_SLIDE: int = 1500
SCORE_WEIGHT_BREAK_BASE: int = 2500
SCORE_WEIGHT_BREAK_EXTRA: int = 100


# ============================================================================
# 4. Hit Modes & Unreasonable Pattern Types
# ============================================================================
class HitMode(str, Enum):
    OUTER = "outer"
    INNER = "inner"
    HYBRID = "hybrid"

    @classmethod
    def from_str(cls, value: str | "HitMode") -> "HitMode":
        if isinstance(value, cls):
            return value
        norm = str(value).strip().lower()
        mapping = {
            "outer": cls.OUTER,
            "outeronly": cls.OUTER,
            "outer_only": cls.OUTER,
            "inner": cls.INNER,
            "inneronly": cls.INNER,
            "inner_only": cls.INNER,
            "hybrid": cls.HYBRID,
        }
        if norm not in mapping:
            raise ValueError(f"Unsupported hit mode: {value!r}")
        return mapping[norm]


class UnreasonableType(str, Enum):
    DUAL_JACK_IMPOSSIBLE = "DUAL_JACK_IMPOSSIBLE"
    FORCED_SINGLE_HAND_JACK = "FORCED_SINGLE_HAND_JACK"
    FORCED_SINGLE_HAND_JUMP = "FORCED_SINGLE_HAND_JUMP"
    HAND_COLLISION = "HAND_COLLISION"
    ARM_COLLISION = "ARM_COLLISION"
    THREE_HAND_CONFLICT = "THREE_HAND_CONFLICT"
    EXTREME_CROSS_ARM = "EXTREME_CROSS_ARM"
    FLYING_HAND = "FLYING_HAND"
    SLIDE_JUMP_ZONE = "SLIDE_JUMP_ZONE"


# ============================================================================
# 5. Human Skill Profiles (Lv12_Advanced .. Lv15_Apex)
# ============================================================================
@dataclass(frozen=True)
class SkillProfileConfig:
    name: str
    max_single_hand_tap_hz: float
    max_single_hand_adjacent_hop_hz: float
    max_single_hand_far_jump_hz: float
    short_burst_multiplier: float
    short_burst_max_notes: int
    max_alternating_tap_hz: float
    base_dual_jack_hz: float
    dual_jack_short_burst_hz: float
    max_arm_velocity: float          # mm/s
    max_arm_accel: float             # mm/s^2
    torso_response_rate: float       # Hz bandwidth
    max_torso_angular_accel: float   # deg/s^2
    allow_inner_outer_split: bool
    allow_slide_hand_switch: bool
    timing_jitter_ms: float
    stamina_recovery_per_sec: float

    def max_single_jack_hz(self, streak_notes: int) -> float:
        """Return maximum single-hand jackhammer frequency (Hz) for a streak of N taps."""
        if streak_notes <= 2:
            return self.max_single_hand_tap_hz * 1.25
        if streak_notes <= self.short_burst_max_notes:
            return self.max_single_hand_tap_hz * self.short_burst_multiplier
        return self.max_single_hand_tap_hz

    def max_dual_jack_hz(self, streak_pairs: int, all_ex: bool = False) -> float:
        """Return maximum simultaneous dual-tap jackhammer frequency (Hz) for N consecutive pairs.

        Dual-tap jackhammers require synchronous bilateral lift-accelerate-brake-strike without
        hand alternation relief. Short 3~4 pair bursts allow wrist bounce, while sustained N>=5
        streaks decay rapidly. EX chords allow ~10% higher bounce tolerance.
        """
        ex_mult = 1.10 if all_ex else 1.0
        if streak_pairs < 3:
            return self.dual_jack_short_burst_hz * 1.12 * ex_mult
        if streak_pairs == 3:
            return self.dual_jack_short_burst_hz * ex_mult
        if streak_pairs == 4:
            return self.base_dual_jack_hz * 1.12 * ex_mult
        decay = max(0.72, 1.0 - 0.055 * (streak_pairs - 4))
        return self.base_dual_jack_hz * decay * ex_mult


SKILL_PROFILES: Dict[str, SkillProfileConfig] = {
    "Lv12_Advanced": SkillProfileConfig(
        name="Lv12_Advanced",
        max_single_hand_tap_hz=6.5,
        max_single_hand_adjacent_hop_hz=4.0,
        max_single_hand_far_jump_hz=2.8,
        short_burst_multiplier=1.15,
        short_burst_max_notes=4,
        max_alternating_tap_hz=14.0,
        base_dual_jack_hz=9.2,
        dual_jack_short_burst_hz=11.6,
        max_arm_velocity=3650.0,
        max_arm_accel=34000.0,
        torso_response_rate=3.8,
        max_torso_angular_accel=500.0,
        allow_inner_outer_split=False,
        allow_slide_hand_switch=True,
        timing_jitter_ms=14.0,
        stamina_recovery_per_sec=8.0,
    ),
    "Lv13+_Expert": SkillProfileConfig(
        name="Lv13+_Expert",
        max_single_hand_tap_hz=7.5,
        max_single_hand_adjacent_hop_hz=4.3,
        max_single_hand_far_jump_hz=3.0,
        short_burst_multiplier=1.15,
        short_burst_max_notes=4,
        max_alternating_tap_hz=16.5,
        base_dual_jack_hz=10.2,
        dual_jack_short_burst_hz=12.9,
        max_arm_velocity=4400.0,
        max_arm_accel=44000.0,
        torso_response_rate=5.0,
        max_torso_angular_accel=620.0,
        allow_inner_outer_split=True,
        allow_slide_hand_switch=True,
        timing_jitter_ms=10.5,
        stamina_recovery_per_sec=10.0,
    ),
    "Lv14+_Master": SkillProfileConfig(
        name="Lv14+_Master",
        max_single_hand_tap_hz=8.5,
        max_single_hand_adjacent_hop_hz=4.5,
        max_single_hand_far_jump_hz=3.2,
        short_burst_multiplier=1.15,
        short_burst_max_notes=4,
        max_alternating_tap_hz=18.5,
        base_dual_jack_hz=10.9,
        dual_jack_short_burst_hz=13.5,
        max_arm_velocity=5100.0,
        max_arm_accel=54000.0,
        torso_response_rate=6.4,
        max_torso_angular_accel=740.0,
        allow_inner_outer_split=True,
        allow_slide_hand_switch=True,
        timing_jitter_ms=7.5,
        stamina_recovery_per_sec=12.5,
    ),
    "Lv15_Apex": SkillProfileConfig(
        name="Lv15_Apex",
        max_single_hand_tap_hz=9.5,
        max_single_hand_adjacent_hop_hz=4.8,
        max_single_hand_far_jump_hz=3.5,
        short_burst_multiplier=1.15,
        short_burst_max_notes=4,
        max_alternating_tap_hz=20.0,
        base_dual_jack_hz=11.6,
        dual_jack_short_burst_hz=14.2,
        max_arm_velocity=5900.0,
        max_arm_accel=65000.0,
        torso_response_rate=8.0,
        max_torso_angular_accel=880.0,
        allow_inner_outer_split=True,
        allow_slide_hand_switch=True,
        timing_jitter_ms=5.0,
        stamina_recovery_per_sec=15.0,
    ),
}

DEFAULT_SKILL_PROFILE: str = "Lv14+_Master"
DEFAULT_HIT_MODE: HitMode = HitMode.HYBRID

# Beam search / Viterbi planner defaults
PLANNER_BEAM_WIDTH: int = 6
PLANNER_LOOKAHEAD_WINDOW: int = 12
AVOID_FLYING_HAND_DEFAULT: bool = True

# Visualizer style constants (matching dark minimalist reference screenshot)
VIS_BG_COLOR_RGB: Tuple[int, int, int] = (36, 34, 34)          # #242222
VIS_GRID_LINE_RGB: Tuple[int, int, int] = (108, 108, 108)      # #6c6c6c
VIS_SCREEN_RING_RGB: Tuple[int, int, int] = (128, 128, 128)    # #808080
VIS_OUTER_BTN_RGB: Tuple[int, int, int] = (78, 78, 78)         # #4e4e4e
VIS_LEFT_ARM_RGB: Tuple[int, int, int] = (43, 149, 195)        # #2b95c3 Cyan/Blue
VIS_RIGHT_ARM_RGB: Tuple[int, int, int] = (201, 136, 66)       # #c98842 Amber/Orange
VIS_ALERT_RED_RGB: Tuple[int, int, int] = (235, 64, 52)        # #eb4034
VIS_ACTIVE_GREEN_RGB: Tuple[int, int, int] = (72, 219, 118)    # #48db76
