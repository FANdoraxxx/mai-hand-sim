"""Exact MiaCode / MajdataPlay / MajdataView 2D Gameplay Renderer & Real-Time SFX Engine.

Directly ports MiaCode's C++ scene pipeline (`PreviewGuideLayerState.cpp`, `PreviewHeadLayerState.cpp`,
`PreviewTrackLayerState.cpp`, `PreviewSlideMotionLayerState.cpp`, `PreviewTouchLayerState.cpp`)
and loads `slide_data.json`, `noteguide/*.png`, `skinDX/*.png`, and `SFX/*.wav`.
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pygame

from sim_core.config import R_A_CENTER_MM, R_SCREEN_MM
from sim_core.geometry import (
    CANONICAL_33_SENSORS,
    OUTER_BUTTONS,
    POLYGON_C_FULL,
    TOUCH_ZONES,
    Point2D,
    button_angle_deg,
    polar_cw_to_xy,
)
from sim_core.maidata_parser import NoteType, ParsedChart, ParsedNote, SlideSegmentData

DEFAULT_MAJPLAY_ROOT = Path(r"E:\GAME\Majplay")
DEFAULT_MIACODE_ROOT = Path(r"G:\GitHub\MiaCode")
FALLBACK_SLIDE_JSON = Path(r"G:\GitHub\maiAIV3\third_party\muri-extracted\miacode\assets\reference\slide_data.json")

# ============================================================================
# MiaCode Logical Canvas Constants (`PreviewSceneConstants.h` & `PreviewGameplayConfig.h`)
# ============================================================================
LOGICAL_CANVAS_SIZE: float = 540.0
LOGICAL_CENTER: float = 270.0
LOGICAL_DIST_TAP: float = 540.0 * 122.5 / 1080.0   # 61.25
LOGICAL_DIST_EDGE: float = 540.0 * 480.0 / 1080.0  # 240.0
MM_TO_LOGICAL: float = LOGICAL_DIST_EDGE / R_A_CENTER_MM  # 240.0 / 215.0 = 1.116279

TAP_HEAD_SCALE: float = 0.50
SLIDE_TRACK_SCALE: float = 0.50
STAR_ASSET_SCALE: float = 90.0 / 126.0
SLIDE_SPAWN_STAR_REL_SCALE: float = 1.08
HOLD_TARGET_WIDTH: float = 122.0 * 0.5 * (60.0 / 61.0)  # 60.0 logical units
HOLD_CAP_RATIO_NUM: int = 71
HOLD_CAP_RATIO_DEN: int = 200

NOTE_GUIDE_SOURCE_RADIUS: float = 240.75
EACH_LINE_SOURCE_RADII: Dict[int, float] = {
    1: 240.02,
    2: 239.89,
    3: 239.65,
    4: 239.92,
}

TOUCH_CLOSED_OFFSET: float = 12.0
TOUCH_START_OFFSET: float = 12.0 * (0.626 / 0.226)  # 33.2389
TOUCH_HOLD_CLOSED_OFFSET: float = 8.0
TOUCH_HOLD_START_OFFSET: float = 8.0 * (0.626 / 0.226)
TOUCH_ASSET_SCALE: float = 0.50

# Cache for `slide_data.json`
_SLIDE_DATA_CACHE: Optional[Dict] = None


def get_slide_reference_data() -> Dict:
    global _SLIDE_DATA_CACHE
    if _SLIDE_DATA_CACHE is not None:
        return _SLIDE_DATA_CACHE
    for p in (
        DEFAULT_MIACODE_ROOT / "assets" / "reference" / "slide_data.json",
        FALLBACK_SLIDE_JSON,
    ):
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                _SLIDE_DATA_CACHE = json.load(f)
                return _SLIDE_DATA_CACHE
    _SLIDE_DATA_CACHE = {"slides": {}, "wifi": {}}
    return _SLIDE_DATA_CACHE


def resolve_slide_data_key(seg: SlideSegmentData) -> str:
    """Resolve a `SlideSegmentData` into its exact lookup key in `slide_data.json`."""
    s, e, op = seg.start_btn, seg.end_btn, seg.shape
    if op == "V":
        via = seg.via_btn or ((s + 5) % 8 + 1)
        return f"{s}V{via}{e}"
    if op == "^":
        d_cw = (e - s) % 8
        is_cw = (d_cw <= 4)
        if is_cw:
            actual_op = ">" if s in (1, 2, 7, 8) else "<"
        else:
            actual_op = "<" if s in (1, 2, 7, 8) else ">"
        return f"{s}{actual_op}{e}"
    return f"{s}{op}{e}"


def lane_unit_vector_logical(lane: int) -> Tuple[float, float]:
    """MiaCode `laneUnitVector(lane)` in logical canvas space (+X right, +Y down)."""
    deg = -67.5 + (lane - 1) * 45.0
    rad = math.radians(deg)
    return (math.cos(rad), math.sin(rad))


def lane_rotation_degrees(lane: float) -> float:
    """MiaCode `laneRotationDegrees(lane)`: 22.5 + (lane - 1) * 45.0 deg."""
    return 22.5 + (lane - 1.0) * 45.0


@dataclass(frozen=True)
class SFXEvent:
    time: float
    sfx_key: str
    secondary_key: str = ""


class MajplaySFXPlayer:
    """Real-time MajdataPlay / MiaCode SFX playback manager."""

    def __init__(self, majplay_root: Path = DEFAULT_MAJPLAY_ROOT) -> None:
        self.sfx_dir = majplay_root / "MajdataPlay_Data" / "StreamingAssets" / "SFX"
        if not self.sfx_dir.exists():
            self.sfx_dir = DEFAULT_MIACODE_ROOT / "assets" / "SFX"
        self.sounds: Dict[str, pygame.mixer.Sound] = {}
        self.events: List[SFXEvent] = []
        self.event_times: List[float] = []
        self.cursor: int = 0
        self.enabled: bool = False
        self._init_sounds()

    def _init_sounds(self) -> None:
        if not self.sfx_dir.exists():
            return
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
            pygame.mixer.set_num_channels(24)
        except Exception:
            return

        sfx_files = {
            "answer": ("answer.wav", 0.50),
            "tap_perfect": ("tap_perfect.wav", 0.55),
            "tap_ex": ("tap_ex.wav", 0.65),
            "break_tap": ("break_tap.wav", 0.70),
            "break": ("break.wav", 0.75),
            "slide": ("slide.wav", 0.55),
            "slide_break_start": ("slide_break_start.wav", 0.65),
            "slide_break_slide": ("slide_break_slide.wav", 0.70),
            "break_slide": ("break_slide.wav", 0.70),
            "touch": ("touch.wav", 0.55),
            "touch_hanabi": ("touch_hanabi.wav", 0.65),
        }
        for key, (fname, vol) in sfx_files.items():
            fpath = self.sfx_dir / fname
            if fpath.exists():
                try:
                    snd = pygame.mixer.Sound(str(fpath))
                    snd.set_volume(vol)
                    self.sounds[key] = snd
                except Exception:
                    pass
        self.enabled = bool(self.sounds)

    def build_schedule(self, chart: ParsedChart) -> None:
        ev_list: List[SFXEvent] = []
        answered_times: List[float] = []
        for n in chart.notes:
            if n.note_type == NoteType.SLIDE:
                ev_list.append(
                    SFXEvent(
                        time=n.shoot_time,
                        sfx_key="slide_break_start" if n.is_break else "slide",
                    )
                )
                if n.is_break:
                    ev_list.append(
                        SFXEvent(
                            time=n.judge_time,
                            sfx_key="slide_break_slide",
                            secondary_key="break_slide",
                        )
                    )
                continue

            if not answered_times or abs(n.time - answered_times[-1]) > 0.004:
                ev_list.append(SFXEvent(time=n.time, sfx_key="answer"))
                answered_times.append(n.time)

            if n.note_type in (NoteType.HOLD, NoteType.TOUCH_HOLD) and n.duration > 0.04:
                ev_list.append(SFXEvent(time=n.end_time, sfx_key="answer"))

            if n.note_type in (NoteType.TAP, NoteType.HOLD):
                if n.is_break:
                    ev_list.append(SFXEvent(time=n.time, sfx_key="break_tap", secondary_key="break"))
                elif n.is_ex:
                    ev_list.append(SFXEvent(time=n.time, sfx_key="tap_ex"))
                else:
                    ev_list.append(SFXEvent(time=n.time, sfx_key="tap_perfect"))
            elif n.note_type in (NoteType.TOUCH, NoteType.TOUCH_HOLD):
                ev_list.append(
                    SFXEvent(
                        time=n.time,
                        sfx_key="touch",
                        secondary_key="touch_hanabi" if n.is_firework else "",
                    )
                )
        ev_list.sort(key=lambda e: e.time)
        self.events = ev_list
        self.event_times = [e.time for e in ev_list]
        self.cursor = 0

    def sync_to_time(self, play_time: float) -> None:
        self.cursor = bisect.bisect_left(self.event_times, play_time)

    def update(self, play_time: float) -> None:
        if not self.enabled:
            return
        played_this_frame: set[str] = set()
        while self.cursor < len(self.events) and self.events[self.cursor].time <= play_time + 0.008:
            ev = self.events[self.cursor]
            self.cursor += 1
            if play_time - ev.time <= 0.08:
                for k in (ev.sfx_key, ev.secondary_key):
                    if k and k not in played_this_frame and k in self.sounds:
                        self.sounds[k].play()
                        played_this_frame.add(k)


class MajplaySkinRenderer:
    """1:1 MiaCode / MajdataView 2D Playfield Renderer (`540x540` logical coordinate space)."""

    def __init__(
        self,
        canvas_size: int = 640,
        majplay_root: Path = DEFAULT_MAJPLAY_ROOT,
        miacode_root: Path = DEFAULT_MIACODE_ROOT,
        note_speed: float = 7.5,
    ) -> None:
        self.canvas_size = canvas_size
        self.center_px = canvas_size // 2
        self.canvas_scale = canvas_size / LOGICAL_CANVAS_SIZE  # px per logical unit (1 logical unit == 1 mm inner screen)
        self.note_speed = note_speed
        self.majplay_root = majplay_root
        self.miacode_root = miacode_root

        self.skin_dir = miacode_root / "assets" / "skin" / "skinDX"
        if not self.skin_dir.exists():
            self.skin_dir = majplay_root / "Skins" / "Deluxe"
        self.guide_dir = miacode_root / "assets" / "noteguide"

        self.sprites: Dict[str, pygame.Surface] = {}
        self.hold_slices: Dict[str, Tuple[pygame.Surface, pygame.Surface, pygame.Surface, int, int]] = {}
        self.rot_cache: Dict[Tuple[str, int, int, int, int], pygame.Surface] = {}
        self.bg_surface: Optional[pygame.Surface] = None
        self._static_stage_surf: Optional[pygame.Surface] = None
        self._indexed_chart_id: int = -1
        self._notes_by_sec: Dict[int, List[ParsedNote]] = {}
        self._note_end_times: List[float] = []
        self.slide_db = get_slide_reference_data()
        self._load_all_assets()

        if not pygame.font.get_init():
            pygame.font.init()
        self.font_combo = pygame.font.SysFont("impact,arialblack,consolas,sans-serif", 32, bold=True)
        self.font_combo_hdr = pygame.font.SysFont("consolas,arial,sans-serif", 13, bold=True)
        self._rebuild_static_stage()

    def _rebuild_static_stage(self) -> None:
        """Pre-bake the static Maimai stage (bg + 33 sensor wireframe + judgement ring + 8 dots) ONCE."""
        surf = pygame.Surface((self.canvas_size, self.canvas_size))
        surf.fill((16, 16, 20))
        cx, cy = self.center_px, self.center_px
        edge_r_px = int(round(LOGICAL_DIST_EDGE * self.canvas_scale))
        if self.bg_surface is not None:
            surf.blit(self.bg_surface, (cx - edge_r_px, cy - edge_r_px))
        else:
            pygame.draw.circle(surf, (22, 22, 26), (cx, cy), edge_r_px)
        for z_name in CANONICAL_33_SENSORS:
            if z_name in ("C1", "C2"):
                continue
            pts = [self.mm_to_canvas(p) for p in TOUCH_ZONES[z_name].polygon]
            pygame.draw.polygon(surf, (78, 80, 88), pts, width=1)
        c_pts = [self.mm_to_canvas(p) for p in POLYGON_C_FULL]
        pygame.draw.polygon(surf, (78, 80, 88), c_pts, width=1)
        pygame.draw.circle(surf, (235, 238, 245), (cx, cy), edge_r_px, width=3)
        for k in range(1, 9):
            ux, uy = lane_unit_vector_logical(k)
            dot_px = self.logical_to_px(ux * LOGICAL_DIST_EDGE, uy * LOGICAL_DIST_EDGE)
            pygame.draw.circle(surf, (245, 245, 250), dot_px, max(4, int(round(5.0 * self.canvas_scale))))
            pygame.draw.circle(surf, (35, 35, 40), dot_px, max(2, int(round(2.2 * self.canvas_scale))))
        self._static_stage_surf = surf

    def _ensure_chart_indexed(self, chart: ParsedChart) -> None:
        if id(chart) == self._indexed_chart_id:
            return
        self._indexed_chart_id = id(chart)
        self._notes_by_sec.clear()
        end_times: List[float] = []
        for n in chart.notes:
            t0 = int(math.floor(n.time - 2.2))
            t1 = int(math.ceil(max(n.time, n.end_time) + 0.5))
            for s in range(t0, t1 + 1):
                self._notes_by_sec.setdefault(s, []).append(n)
            end_times.append(n.end_time if n.note_type in (NoteType.HOLD, NoteType.TOUCH_HOLD, NoteType.SLIDE) else n.time)
        end_times.sort()
        self._note_end_times = end_times

    def logical_to_px(self, lx: float, ly: float) -> Tuple[int, int]:
        """Convert MiaCode centered logical coordinate (lx, ly) [where +Y is down, r_edge=240] to canvas pixel."""
        return (
            int(round(self.center_px + lx * self.canvas_scale)),
            int(round(self.center_px + ly * self.canvas_scale)),
        )

    def mm_to_canvas(self, pt_mm: Point2D) -> Tuple[int, int]:
        """Convert physical (x_mm, y_mm) [where +Y is up, R_A=215mm -> 240 logical] to canvas pixel."""
        lx = pt_mm[0] * MM_TO_LOGICAL
        ly = -pt_mm[1] * MM_TO_LOGICAL
        return self.logical_to_px(lx, ly)

    @staticmethod
    def _load_rgba_file(path: Path) -> Optional[pygame.Surface]:
        if not path.exists():
            return None
        try:
            arr = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            if arr is None:
                return None
            if arr.ndim == 2:
                arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGBA)
            elif arr.shape[2] == 3:
                arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGBA)
            else:
                arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2RGBA)
            h, w = arr.shape[:2]
            return pygame.image.frombuffer(arr.tobytes(), (w, h), "RGBA").copy()
        except Exception:
            return None

    def _find_and_load(self, candidates: Sequence[Path]) -> Optional[pygame.Surface]:
        for p in candidates:
            s = self._load_rgba_file(p)
            if s is not None:
                return s
        return None

    def _load_all_assets(self) -> None:
        dx_dir = self.miacode_root / "assets" / "skin" / "skinDX"
        maj_deluxe = self.majplay_root / "Skins" / "Deluxe"
        maj_stream = self.majplay_root / "MajdataPlay_Data" / "StreamingAssets" / "Skins"

        def _skin(name: str, sub: str = "") -> Optional[pygame.Surface]:
            cands = [dx_dir / name]
            if sub:
                cands.append(maj_deluxe / sub / name)
            cands.extend([maj_deluxe / name, maj_stream / name])
            return self._find_and_load(cands)

        skin_files = {
            "outline": ("outline.png", ""),
            "tap": ("tap.png", "TapSkins"),
            "tap_each": ("tap_each.png", "TapSkins"),
            "tap_break": ("tap_break.png", "TapSkins"),
            "tap_ex": ("tap_ex.png", "TapSkins"),
            "star": ("star.png", "StarSkins"),
            "star_each": ("star_each.png", "StarSkins"),
            "star_break": ("star_break.png", "StarSkins"),
            "star_double": ("star_double.png", "StarSkins"),
            "star_each_double": ("star_each_double.png", "StarSkins"),
            "star_break_double": ("star_break_double.png", "StarSkins"),
            "hold": ("hold.png", "HoldSkins"),
            "hold_on": ("hold_on.png", "HoldSkins"),
            "hold_each": ("hold_each.png", "HoldSkins"),
            "hold_each_on": ("hold_each_on.png", "HoldSkins"),
            "hold_break": ("hold_break.png", "HoldSkins"),
            "hold_break_on": ("hold_break_on.png", "HoldSkins"),
            "slide": ("slide.png", "SlideSkins"),
            "slide_each": ("slide_each.png", "SlideSkins"),
            "slide_break": ("slide_break.png", "SlideSkins"),
            "touch": ("touch.png", "TouchSkins"),
            "touch_each": ("touch_each.png", "TouchSkins"),
            "touch_break": ("touch_break.png", "TouchSkins"),
            "touch_point": ("touch_point.png", "TouchSkins"),
            "touch_point_each": ("touch_point_each.png", "TouchSkins"),
            "touch_point_break": ("touch_break_point.png", "TouchSkins"),
            "touch_border_2": ("touch_border_2.png", "TouchSkins"),
            "touch_border_2_each": ("touch_border_2_each.png", "TouchSkins"),
            "touchhold_0": ("touchhold_0.png", "TouchHoldSkins"),
            "touchhold_1": ("touchhold_1.png", "TouchHoldSkins"),
            "touchhold_2": ("touchhold_2.png", "TouchHoldSkins"),
            "touchhold_3": ("touchhold_3.png", "TouchHoldSkins"),
            "touchhold_border": ("touchhold_border.png", "TouchHoldSkins"),
            "judge_cp": ("judge_text_cPerfect.png", "JudgeTextSkins"),
            "judge_p": ("judge_text_perfect.png", "JudgeTextSkins"),
            "judge_gr": ("judge_text_great.png", "JudgeTextSkins"),
            "judge_gd": ("judge_text_good.png", "JudgeTextSkins"),
            "judge_miss": ("judge_text_miss.png", "JudgeTextSkins"),
            "judge_break": ("judge_text_break_2600.png", "JudgeTextSkins"),
            "just_str_r": ("just_str_r.png", "SlideOKSkins"),
            "just_curv_r": ("just_curv_r.png", "SlideOKSkins"),
            "just_wifi_u": ("just_wifi_u.png", "SlideOKSkins"),
        }
        for key, (fname, sub) in skin_files.items():
            s = _skin(fname, sub)
            if s is not None:
                self.sprites[key] = s

        for wi in range(11):
            for prefix, sub in (("wifi", "wifi"), ("wifi_each", "wifi_each"), ("wifi_break", "wifi_break")):
                s = _skin(f"{sub}_{wi}.png", "WifiSkins")
                if s is not None:
                    self.sprites[f"{prefix}_{wi}"] = s

        # Load MiaCode `noteguide` sprites (`Normal.png`, `Each.png`, `Break.png`, `Slide.png`, `EachLine1..4.png`, `Hold_End.png`)
        guide_files = {
            "guide_normal": "Normal.png",
            "guide_each": "Each.png",
            "guide_break": "Break.png",
            "guide_slide": "Slide.png",
            "each_line_1": "EachLine1.png",
            "each_line_2": "EachLine2.png",
            "each_line_3": "EachLine3.png",
            "each_line_4": "EachLine4.png",
            "hold_end": "Hold_End.png",
            "hold_each_end": "Hold_Each_End.png",
            "hold_break_end": "Hold_Break_End.png",
        }
        for k, fname in guide_files.items():
            s = self._load_rgba_file(self.guide_dir / fname)
            if s is not None:
                self.sprites[k] = s

        # Pre-slice Hold sprites into (head_cap, body_1px, tail_cap, src_w, cap_raw) per MiaCode `appendHoldStripSlices`
        for hk in ("hold", "hold_on", "hold_each", "hold_each_on", "hold_break", "hold_break_on"):
            hs = self.sprites.get(hk)
            if hs is not None:
                sw, sh = hs.get_width(), hs.get_height()
                cap_raw = max(1, min(sh // 2, int(round(sh * HOLD_CAP_RATIO_NUM / HOLD_CAP_RATIO_DEN))))
                mid_y = sh // 2
                head_cap = hs.subsurface(pygame.Rect(0, 0, sw, cap_raw)).copy()
                body_strip = hs.subsurface(pygame.Rect(0, mid_y, sw, 2)).copy()
                tail_cap = hs.subsurface(pygame.Rect(0, sh - cap_raw, sw, cap_raw)).copy()
                self.hold_slices[hk] = (head_cap, body_strip, tail_cap, sw, cap_raw)

    def load_song_bg(self, chart_source_path: Optional[str | Path]) -> None:
        if not chart_source_path:
            return
        p = Path(chart_source_path)
        folder = p if p.is_dir() else p.parent
        for name in ("bg.jpg", "bg.png", "bg.jpeg"):
            bg_file = folder / name
            if bg_file.exists():
                try:
                    arr = cv2.imdecode(np.fromfile(str(bg_file), dtype=np.uint8), cv2.IMREAD_COLOR)
                    if arr is not None:
                        rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
                        r_px = int(round(LOGICAL_DIST_EDGE * self.canvas_scale))
                        diam = r_px * 2
                        resized = cv2.resize(rgb, (diam, diam), interpolation=cv2.INTER_AREA)
                        # Match MiaCode stage dim (0.28 brightness inside the 480-radius judgement circle)
                        dark = (resized.astype(np.float32) * 0.28).astype(np.uint8)
                        rgba = np.zeros((diam, diam, 4), dtype=np.uint8)
                        rgba[:, :, :3] = dark
                        yy, xx = np.ogrid[:diam, :diam]
                        mask = (xx - r_px) ** 2 + (yy - r_px) ** 2 <= r_px ** 2
                        rgba[:, :, 3] = np.where(mask, 255, 0)
                        self.bg_surface = pygame.image.frombuffer(rgba.tobytes(), (diam, diam), "RGBA").copy()
                        self._rebuild_static_stage()
                        return
                except Exception:
                    pass

    def _blit_sprite(
        self,
        target: pygame.Surface,
        key: str,
        center_px: Tuple[int, int],
        width_px: float,
        height_px: float,
        cw_rotation_deg: float,
        opacity: float = 1.0,
    ) -> None:
        """Blit sprite `key` at `center_px` with target size `(width_px, height_px)` and clockwise rotation
        (matching Qt/MiaCode where `rotationDegrees` is clockwise, whereas `pygame.transform.rotate` is CCW!).
        """
        base = self.sprites.get(key)
        if base is None or opacity <= 0.01 or width_px < 1.0 or height_px < 1.0:
            return
        w_i = max(2, int(round(width_px)))
        h_i = max(2, int(round(height_px)))
        # Convert MiaCode CW rotation to Pygame CCW rotation: -cw_rotation_deg
        ccw_q = int(round((-cw_rotation_deg) / 2.0) * 2) % 360
        alpha_q = max(1, min(20, int(round(opacity * 20))))
        ckey = (key, w_i, h_i, ccw_q, alpha_q)
        cached = self.rot_cache.get(ckey)
        if cached is None:
            scaled = pygame.transform.smoothscale(base, (w_i, h_i))
            if ccw_q != 0:
                scaled = pygame.transform.rotate(scaled, ccw_q)
            if alpha_q < 20:
                scaled = scaled.copy()
                scaled.set_alpha(int(alpha_q * 255 / 20))
            if len(self.rot_cache) > 1600:
                self.rot_cache.clear()
            self.rot_cache[ckey] = scaled
            cached = scaled
        target.blit(cached, cached.get_rect(center=center_px))

    def _sample_tap_approach(self, delta_sec: float) -> Tuple[float, float]:
        """Exact MiaCode `sampleTapApproach` (`PreviewGameplayConfig.h` & `PreviewSceneMath.cpp`):
        `T = 1 + 240 / v` frames at 120 FPS; `spawnDuration = T/120`, `flyDuration = T/120`.
        Returns `(logical_distance, scale)` where `logical_distance in [61.25, 240.0]`.
        """
        scale_frames = 1.0 + 240.0 / max(1.0, self.note_speed)
        spawn_sec = scale_frames / 120.0
        fly_sec = scale_frames / 120.0
        lifecycle_sec = spawn_sec + fly_sec
        if delta_sec < -lifecycle_sec:
            return (LOGICAL_DIST_TAP, 0.0)
        if delta_sec < -fly_sec:
            # Spawning at r = 61.25: scale ramps linearly from 0 -> 1
            u = (delta_sec + lifecycle_sec) / max(1e-4, spawn_sec)
            return (LOGICAL_DIST_TAP, max(0.0, min(1.0, u)))
        if delta_sec < 0.0:
            u = (delta_sec + fly_sec) / max(1e-4, fly_sec)
            dist = LOGICAL_DIST_TAP + u * (LOGICAL_DIST_EDGE - LOGICAL_DIST_TAP)
            return (dist, 1.0)
        return (LOGICAL_DIST_EDGE, 1.0)

    def render_game_canvas(
        self,
        chart: ParsedChart,
        current_time: float,
        active_zones: Sequence[str] = (),
        triggered_zones: Sequence[str] = (),
        has_alert: bool = False,
    ) -> pygame.Surface:
        """Render the exact MiaCode / MajdataView 2D gameplay canvas (`canvas_size x canvas_size`)."""
        self._ensure_chart_indexed(chart)
        if self._static_stage_surf is None:
            self._rebuild_static_stage()
        surf = self._static_stage_surf.copy()
        cx, cy = self.center_px, self.center_px
        trig_set = {z.upper() for z in triggered_zones}
        act_set = {z.upper() for z in active_zones}
        for z_name in (trig_set | act_set):
            if z_name.startswith("OB") and z_name[2:].isdigit():
                k = int(z_name[2:])
                ux, uy = lane_unit_vector_logical(k)
                ob_px = self.logical_to_px(ux * 254.0, uy * 254.0)
                pygame.draw.circle(surf, (80, 240, 140), ob_px, int(round(15.0 * self.canvas_scale)), width=3)
            elif z_name in ("C", "C1", "C2"):
                c_pts = [self.mm_to_canvas(p) for p in POLYGON_C_FULL]
                pygame.draw.polygon(surf, (75, 165, 110) if z_name in trig_set else (50, 90, 125), c_pts)
                pygame.draw.polygon(surf, (130, 220, 160), c_pts, width=2)
            elif z_name in TOUCH_ZONES:
                pts = [self.mm_to_canvas(p) for p in TOUCH_ZONES[z_name].polygon]
                col = (165, 52, 48) if (has_alert and z_name in trig_set) else ((58, 145, 95) if z_name in trig_set else (42, 78, 115))
                pygame.draw.polygon(surf, col, pts)
                pygame.draw.polygon(surf, (140, 225, 165) if z_name in trig_set else (95, 150, 200), pts, width=2)

        judged_combo = bisect.bisect_right(self._note_end_times, current_time)
        if judged_combo > 0:
            hdr_s = self.font_combo_hdr.render("CRITICAL PERFECT", True, (235, 200, 95))
            cmb_s = self.font_combo.render(str(judged_combo), True, (245, 215, 110))
            surf.blit(hdr_s, hdr_s.get_rect(center=(cx, cy - 18)))
            surf.blit(cmb_s, cmb_s.get_rect(center=(cx, cy + 12)))

        win_notes = self._notes_by_sec.get(int(math.floor(current_time)), ())
        self._render_guide_layer(surf, win_notes, current_time)
        self._render_slide_tracks(surf, win_notes, current_time)
        self._render_touch_layer(surf, win_notes, current_time)
        self._render_head_layer(surf, win_notes, current_time)
        self._render_slide_moving_stars(surf, win_notes, current_time)
        self._render_judge_effects(surf, win_notes, current_time)

        return surf

    def _render_guide_layer(
        self,
        surf: pygame.Surface,
        notes: Sequence[ParsedNote],
        current_time: float,
    ) -> None:
        cx, cy = self.center_px, self.center_px
        each_groups: Dict[int, List[ParsedNote]] = {}
        seen_times: set[int] = set()

        for n in notes:
            if n.note_type == NoteType.SLIDE:
                continue
            if n.note_type not in (NoteType.TAP, NoteType.HOLD) or n.button is None:
                continue
            t_end = n.end_time if n.note_type == NoteType.HOLD else n.time
            if current_time > t_end:
                continue
            delta = current_time - n.time
            dist, scale = self._sample_tap_approach(delta)
            if scale <= 0.0:
                continue

            if current_time <= n.time:
                guide_r = max(LOGICAL_DIST_TAP, min(LOGICAL_DIST_EDGE, dist))
                t_k = int(round(n.time * 1000))
                if t_k not in seen_times and scale > 0.25:
                    seen_times.add(t_k)
                    r_px = int(round(guide_r * self.canvas_scale))
                    g_col = (195, 125, 55) if n.is_break else ((185, 175, 65) if n.is_each else ((75, 155, 195) if n.is_star_head else (185, 95, 138)))
                    pygame.draw.circle(surf, g_col, (cx, cy), r_px, width=1)
                if n.is_each:
                    each_groups.setdefault(t_k, []).append(n)

        for grp in each_groups.values():
            lanes = sorted({n.button for n in grp if n.button is not None})
            if len(lanes) < 2:
                continue
            dist, sc = self._sample_tap_approach(current_time - grp[0].time)
            if sc <= 0.0:
                continue
            guide_r = max(LOGICAL_DIST_TAP, min(LOGICAL_DIST_EDGE, dist))
            a0 = -67.5 + (lanes[0] - 1) * 45.0
            a1 = -67.5 + (lanes[1] - 1) * 45.0
            diff = (a1 - a0) % 360.0
            if diff > 180.0:
                a0, a1 = a1, a0
                diff = 360.0 - diff
            arc_pts = [
                self.logical_to_px(
                    guide_r * math.cos(math.radians(a0 + diff * (k / 14.0))),
                    guide_r * math.sin(math.radians(a0 + diff * (k / 14.0))),
                )
                for k in range(15)
            ]
            pygame.draw.lines(surf, (255, 232, 75), False, arc_pts, width=3)

    def _render_slide_tracks(
        self,
        surf: pygame.Surface,
        notes: Sequence[ParsedNote],
        current_time: float,
    ) -> None:
        """Exact MiaCode `PreviewTrackLayerState.cpp` using `slide_data.json` `track_arrows`."""
        scale_frames = 1.0 + 240.0 / max(1.0, self.note_speed)
        appear_lead = (scale_frames + 2.0) / 120.0
        full_bright_lead = 6.0 / 120.0

        for n in notes:
            if n.note_type != NoteType.SLIDE or not n.slide_segments:
                continue
            if current_time < n.time - appear_lead or current_time >= n.end_time:
                continue

            # Opacity curve before star launch (`sampleSlideTrackPreTraceOpacity`)
            if current_time < n.time - full_bright_lead:
                u = (current_time - (n.time - appear_lead)) / max(1e-4, appear_lead - full_bright_lead)
                opacity = 0.70 * (1.0 - (1.0 - max(0.0, min(1.0, u))) ** 2.7)
            else:
                opacity = 1.0

            prog = 0.0 if current_time < n.shoot_time else max(0.0, min(1.0, (current_time - n.shoot_time) / max(0.01, n.duration)))
            seg0 = n.slide_segments[0]

            if seg0.shape == "w":
                w_key = f"{seg0.start_btn}w{seg0.end_btn}"
                w_entry = self.slide_db.get("wifi", {}).get(w_key)
                if not w_entry:
                    continue
                w_prefix = "wifi_break" if n.is_break else ("wifi_each" if n.is_each else "wifi")
                flat_arrows = [arr for area in w_entry["track_arrows"] for arr in area]
                cut_idx = int(prog * len(flat_arrows))
                for arr in flat_arrows[cut_idx:]:
                    img_k = f"{w_prefix}_{arr['image_index']}"
                    base_w = self.sprites.get(img_k)
                    if base_w is not None:
                        px = self.logical_to_px(arr["x"], arr["y"])
                        self._blit_sprite(
                            surf,
                            img_k,
                            px,
                            base_w.get_width() * self.canvas_scale * SLIDE_TRACK_SCALE,
                            base_w.get_height() * self.canvas_scale * SLIDE_TRACK_SCALE,
                            arr["rotation"],
                            opacity=opacity,
                        )
                continue

            # Standard or chained slide
            sl_img_key = "slide_break" if n.is_break else ("slide_each" if n.is_each else "slide")
            base_sl = self.sprites.get(sl_img_key)
            if base_sl is None:
                continue
            w_px = base_sl.get_width() * self.canvas_scale * SLIDE_TRACK_SCALE
            h_px = base_sl.get_height() * self.canvas_scale * SLIDE_TRACK_SCALE

            all_arrows: List[Dict] = []
            for seg in n.slide_segments:
                s_key = resolve_slide_data_key(seg)
                s_entry = self.slide_db.get("slides", {}).get(s_key)
                if s_entry:
                    for area in s_entry["track_arrows"]:
                        all_arrows.extend(area)

            if not all_arrows:
                continue
            cut_idx = int(prog * len(all_arrows))
            # Draw in reverse order so earlier arrows overlap later arrows cleanly like MiaCode
            for arr in reversed(all_arrows[cut_idx:]):
                px = self.logical_to_px(arr["x"], arr["y"])
                self._blit_sprite(surf, sl_img_key, px, w_px, h_px, arr["rotation"], opacity=opacity)

    def _render_slide_moving_stars(
        self,
        surf: pygame.Surface,
        notes: Sequence[ParsedNote],
        current_time: float,
    ) -> None:
        """Exact MiaCode `PreviewSlideMotionLayerState.cpp` using `slide_data.json` `samples` / `lane_samples`."""
        for n in notes:
            if n.note_type != NoteType.SLIDE or not n.slide_segments:
                continue
            if current_time < n.time or current_time > n.end_time:
                continue

            star_key = "star_break" if n.is_break else ("star_each" if n.is_each else "star")
            base_st = self.sprites.get(star_key)
            if base_st is None:
                continue

            if current_time < n.shoot_time:
                wait_dur = max(0.001, n.shoot_time - n.time)
                wait_u = max(0.0, min(1.0, (current_time - n.time) / wait_dur))
                opacity = 0.5 + 0.5 * wait_u
                st_scale = STAR_ASSET_SCALE * (0.65 + 0.35 * wait_u)
                prog = 0.0
            else:
                opacity = 1.0
                st_scale = STAR_ASSET_SCALE
                prog = max(0.0, min(1.0, (current_time - n.shoot_time) / max(0.001, n.duration)))

            w_px = base_st.get_width() * self.canvas_scale * st_scale
            h_px = base_st.get_height() * self.canvas_scale * st_scale
            seg0 = n.slide_segments[0]

            if seg0.shape == "w":
                w_key = f"{seg0.start_btn}w{seg0.end_btn}"
                w_entry = self.slide_db.get("wifi", {}).get(w_key)
                if not w_entry:
                    continue
                for lane_samples in w_entry["lane_samples"]:
                    if not lane_samples:
                        continue
                    idx_f = prog * (len(lane_samples) - 1)
                    i0 = int(math.floor(idx_f))
                    i1 = min(len(lane_samples) - 1, i0 + 1)
                    lt = idx_f - i0
                    lx = lane_samples[i0]["x"] * (1.0 - lt) + lane_samples[i1]["x"] * lt
                    ly = lane_samples[i0]["y"] * (1.0 - lt) + lane_samples[i1]["y"] * lt
                    ang = 360.0 - lane_samples[i0]["angle"]
                    self._blit_sprite(surf, star_key, self.logical_to_px(lx, ly), w_px, h_px, ang, opacity=opacity)
                continue

            all_samples: List[Dict] = []
            for seg in n.slide_segments:
                s_key = resolve_slide_data_key(seg)
                s_entry = self.slide_db.get("slides", {}).get(s_key)
                if s_entry and s_entry.get("samples"):
                    all_samples.extend(s_entry["samples"])
            if not all_samples:
                continue
            idx_f = prog * (len(all_samples) - 1)
            i0 = int(math.floor(idx_f))
            i1 = min(len(all_samples) - 1, i0 + 1)
            lt = idx_f - i0
            lx = all_samples[i0]["x"] * (1.0 - lt) + all_samples[i1]["x"] * lt
            ly = all_samples[i0]["y"] * (1.0 - lt) + all_samples[i1]["y"] * lt
            ang = 360.0 - all_samples[i0]["angle"]
            self._blit_sprite(surf, star_key, self.logical_to_px(lx, ly), w_px, h_px, ang, opacity=opacity)

    def _render_touch_layer(
        self,
        surf: pygame.Surface,
        notes: Sequence[ParsedNote],
        current_time: float,
    ) -> None:
        """Exact MiaCode `PreviewTouchLayerState.cpp` & `PreviewTouchHoldLayerState.cpp`."""
        touch_dur = 0.50
        show_dur = touch_dur / 5.0
        close_dur = touch_dur - show_dur

        for n in notes:
            if n.note_type not in (NoteType.TOUCH, NoteType.TOUCH_HOLD):
                continue
            t_end = n.end_time if n.note_type == NoteType.TOUCH_HOLD else n.time
            delta = current_time - n.time
            if delta <= -touch_dur or current_time > t_end:
                continue

            z_center_mm = TOUCH_ZONES.get(n.sensor_zone, TOUCH_ZONES["C"]).center
            c_px = self.mm_to_canvas(z_center_mm)

            if delta < -close_dur:
                alpha = max(0.0, min(1.0, (delta + touch_dur) / max(1e-4, show_dur)))
                close_u = 0.0
            elif delta < 0.0:
                alpha = 1.0
                u_raw = (delta + close_dur) / max(1e-4, close_dur)
                close_u = max(0.0, min(1.0, u_raw ** 2.2))
            else:
                alpha = 1.0
                close_u = 1.0

            pt_key = "touch_point_break" if n.is_break else ("touch_point_each" if n.is_each else "touch_point")
            base_pt = self.sprites.get(pt_key)
            if base_pt is not None:
                self._blit_sprite(
                    surf,
                    pt_key,
                    c_px,
                    base_pt.get_width() * TOUCH_ASSET_SCALE * self.canvas_scale,
                    base_pt.get_height() * TOUCH_ASSET_SCALE * self.canvas_scale,
                    0.0,
                    opacity=alpha,
                )

            if n.note_type == NoteType.TOUCH:
                offset_logical = TOUCH_START_OFFSET + (TOUCH_CLOSED_OFFSET - TOUCH_START_OFFSET) * close_u
                offset_px = offset_logical * self.canvas_scale
                c_key = "touch_break" if n.is_break else ("touch_each" if n.is_each else "touch")
                base_c = self.sprites.get(c_key)
                if base_c is not None:
                    cw = base_c.get_width() * TOUCH_ASSET_SCALE * self.canvas_scale
                    ch = base_c.get_height() * TOUCH_ASSET_SCALE * self.canvas_scale
                    for dx_p, dy_p, ang in (
                        (0.0, -offset_px, 180.0),
                        (offset_px, 0.0, -90.0),
                        (0.0, offset_px, 0.0),
                        (-offset_px, 0.0, 90.0),
                    ):
                        self._blit_sprite(
                            surf,
                            c_key,
                            (int(round(c_px[0] + dx_p)), int(round(c_px[1] + dy_p))),
                            cw,
                            ch,
                            ang,
                            opacity=alpha,
                        )
            else:
                # TouchHold
                offset_logical = TOUCH_HOLD_START_OFFSET + (TOUCH_HOLD_CLOSED_OFFSET - TOUCH_HOLD_START_OFFSET) * close_u
                offset_px = offset_logical * self.canvas_scale * 0.7071
                if delta >= 0.0 and "touchhold_border" in self.sprites:
                    bd = self.sprites["touchhold_border"]
                    self._blit_sprite(
                        surf,
                        "touchhold_border",
                        c_px,
                        bd.get_width() * TOUCH_ASSET_SCALE * self.canvas_scale,
                        bd.get_height() * TOUCH_ASSET_SCALE * self.canvas_scale,
                        0.0,
                    )
                for idx_p, (dx_s, dy_s) in enumerate(((1, -1), (1, 1), (-1, 1), (-1, -1))):
                    tk = f"touchhold_{idx_p}"
                    base_th = self.sprites.get(tk)
                    if base_th is not None:
                        self._blit_sprite(
                            surf,
                            tk,
                            (int(round(c_px[0] + dx_s * offset_px)), int(round(c_px[1] + dy_s * offset_px))),
                            base_th.get_width() * TOUCH_ASSET_SCALE * self.canvas_scale,
                            base_th.get_height() * TOUCH_ASSET_SCALE * self.canvas_scale,
                            0.0,
                            opacity=alpha,
                        )

    def _render_head_layer(
        self,
        surf: pygame.Surface,
        notes: Sequence[ParsedNote],
        current_time: float,
    ) -> None:
        """Exact MiaCode `PreviewHeadLayerState.cpp`: 3-slice Hold rendering + Tap & Star Heads."""
        scale_frames = 1.0 + 240.0 / max(1.0, self.note_speed)
        fly_sec = scale_frames / 120.0

        for n in notes:
            if n.note_type == NoteType.HOLD and n.button is not None:
                if current_time > n.end_time:
                    continue
                delta_head = current_time - n.time
                delta_end = current_time - n.end_time
                dist_head, sc_head = self._sample_tap_approach(delta_head)
                if sc_head <= 0.0:
                    continue

                is_active = delta_head >= 0.0
                if n.is_break:
                    hk = "hold_break_on" if is_active else "hold_break"
                elif n.is_each:
                    hk = "hold_each_on" if is_active else "hold_each"
                else:
                    hk = "hold_on" if is_active else "hold"

                if hk not in self.hold_slices:
                    continue
                head_cap, body_strip, tail_cap, src_w, cap_raw = self.hold_slices[hk]
                ux, uy = lane_unit_vector_logical(n.button)
                ang_cw = lane_rotation_degrees(n.button)

                if delta_head < -fly_sec:
                    # Spawning at r = 61.25: line_len = 0, scale = sc_head
                    body_logical_len = 0.0
                    center_lx = ux * LOGICAL_DIST_TAP
                    center_ly = uy * LOGICAL_DIST_TAP
                    scale = sc_head
                else:
                    dist_tail, _ = self._sample_tap_approach(delta_end)
                    body_logical_len = max(0.0, dist_head - dist_tail)
                    mid_d = 0.5 * (dist_head + dist_tail)
                    center_lx = ux * mid_d
                    center_ly = uy * mid_d
                    scale = 1.0

                target_w_px = max(2.0, HOLD_TARGET_WIDTH * self.canvas_scale * scale)
                target_cap_h_px = max(2.0, (cap_raw * HOLD_TARGET_WIDTH / src_w) * self.canvas_scale * scale)
                target_body_h_px = max(0.0, body_logical_len * self.canvas_scale * scale)
                w_i = max(4, int(round(target_w_px)))
                cap_i = max(2, int(round(target_cap_h_px)))
                body_i = max(0, int(round(target_body_h_px)))
                total_h_i = cap_i * 2 + body_i
                combined = pygame.Surface((w_i, total_h_i), pygame.SRCALPHA)
                combined.blit(pygame.transform.smoothscale(head_cap, (w_i, cap_i)), (0, 0))
                if body_i > 0:
                    combined.blit(pygame.transform.smoothscale(body_strip, (w_i, body_i)), (0, cap_i))
                combined.blit(pygame.transform.smoothscale(tail_cap, (w_i, cap_i)), (0, cap_i + body_i))
                rot_hold = pygame.transform.rotate(combined, -ang_cw)
                surf.blit(rot_hold, rot_hold.get_rect(center=self.logical_to_px(center_lx, center_ly)))
                continue

            if n.note_type == NoteType.TAP and n.button is not None:
                if current_time > n.time:
                    continue
                delta = current_time - n.time
                dist, sc = self._sample_tap_approach(delta)
                if sc <= 0.0:
                    continue
                ux, uy = lane_unit_vector_logical(n.button)
                pt_px = self.logical_to_px(ux * dist, uy * dist)
                if n.is_star_head:
                    sp_key = "star_break" if n.is_break else ("star_each" if n.is_each else "star")
                    rot = 360.0 - (lane_rotation_degrees(n.button) - delta * 180.0)
                    rel_sc = TAP_HEAD_SCALE * SLIDE_SPAWN_STAR_REL_SCALE * sc
                else:
                    sp_key = "tap_break" if n.is_break else ("tap_each" if n.is_each else "tap")
                    rot = lane_rotation_degrees(n.button)
                    rel_sc = TAP_HEAD_SCALE * sc

                base_sp = self.sprites.get(sp_key)
                if base_sp is not None:
                    if n.is_ex and "tap_ex" in self.sprites:
                        ex_sp = self.sprites["tap_ex"]
                        self._blit_sprite(
                            surf,
                            "tap_ex",
                            pt_px,
                            ex_sp.get_width() * self.canvas_scale * rel_sc * 1.35,
                            ex_sp.get_height() * self.canvas_scale * rel_sc * 1.35,
                            rot,
                        )
                    self._blit_sprite(
                        surf,
                        sp_key,
                        pt_px,
                        base_sp.get_width() * self.canvas_scale * rel_sc,
                        base_sp.get_height() * self.canvas_scale * rel_sc,
                        rot,
                    )

    def _render_judge_effects(
        self,
        surf: pygame.Surface,
        notes: Sequence[ParsedNote],
        current_time: float,
    ) -> None:
        for n in notes:
            j_time = n.end_time if n.note_type in (NoteType.HOLD, NoteType.TOUCH_HOLD, NoteType.SLIDE) else n.time
            dt = current_time - j_time
            if not (0.0 <= dt <= 0.26):
                continue

            if n.note_type == NoteType.SLIDE:
                if n.slide_waypoints_mm:
                    end_px = self.mm_to_canvas(n.slide_waypoints_mm[-1])
                    end_btn = n.slide_segments[-1].end_btn if n.slide_segments else 5
                    ok_key = "just_wifi_u" if (n.slide_segments and n.slide_segments[-1].shape == "w") else "just_str_r"
                    base_ok = self.sprites.get(ok_key)
                    if base_ok is not None:
                        self._blit_sprite(
                            surf,
                            ok_key,
                            end_px,
                            base_ok.get_width() * self.canvas_scale * 0.45,
                            base_ok.get_height() * self.canvas_scale * 0.45,
                            lane_rotation_degrees(end_btn),
                        )
                continue

            if n.note_type in (NoteType.TOUCH, NoteType.TOUCH_HOLD):
                pt_mm = TOUCH_ZONES.get(n.sensor_zone, TOUCH_ZONES["C"]).center
                px = self.mm_to_canvas(pt_mm)
                rot_cw = 0.0
                if n.is_firework and dt <= 0.24:
                    u_fw = dt / 0.24
                    r_fw1 = int(round((22.0 + u_fw * 145.0) * self.canvas_scale))
                    r_fw2 = int(round((12.0 + u_fw * 95.0) * self.canvas_scale))
                    pygame.draw.circle(surf, (255, 165, 220), px, r_fw1, width=2)
                    pygame.draw.circle(surf, (135, 235, 255), px, r_fw2, width=2)
            else:
                btn = n.button or 1
                ux, uy = lane_unit_vector_logical(btn)
                px = self.logical_to_px(ux * (LOGICAL_DIST_EDGE - 34.0), uy * (LOGICAL_DIST_EDGE - 34.0))
                rot_cw = lane_rotation_degrees(btn) + (180.0 if btn in (3, 4, 5, 6) else 0.0)
                if dt <= 0.16:
                    rec_px = self.logical_to_px(ux * LOGICAL_DIST_EDGE, uy * LOGICAL_DIST_EDGE)
                    burst_r = int(round((16.0 + dt * 165.0) * self.canvas_scale))
                    alpha = max(10, int(220 * (1.0 - dt / 0.16)))
                    pygame.draw.circle(surf, (255, 225, 90, alpha), rec_px, burst_r, width=3)

            j_key = "judge_break" if n.is_break else "judge_cp"
            base_j = self.sprites.get(j_key)
            if base_j is not None:
                sc = 0.44 * (1.08 if dt < 0.05 else 0.96)
                self._blit_sprite(
                    surf,
                    j_key,
                    px,
                    base_j.get_width() * self.canvas_scale * sc,
                    base_j.get_height() * self.canvas_scale * sc,
                    rot_cw,
                )
