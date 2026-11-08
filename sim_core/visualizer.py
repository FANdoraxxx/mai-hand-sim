"""3D & 2.5D Maimai DX Physical Play Simulator & MajdataPlay Renderer (`visualizer.py`).

Features:
1. Authentic MajdataPlay 2D game screen (`MajplaySkinRenderer` loading Deluxe skins & `bg.jpg` from `E:\\GAME\\Majplay`)
   perspective-warped onto a 3D tilted Maimai DX arcade cabinet via `cv2.warpPerspective`.
2. Authentic MajdataPlay real-time SFX engine (`MajplaySFXPlayer`: answer, tap, break, ex, slide, touch, hanabi).
3. True 3D Human Upper Body (Pelvis, Spine, Torso cage, Rotating Shoulders) + 3D 2-Link Arms + 3D Doraemon Round Palms
   with forward/backward Z-axis strike motion (`z=0mm` contact vs `z=42..148mm` hover), 3D drop shadows on the screen,
   and 3D Z-depth stems.
4. 4 Camera Presets (`[1] 3rd-Person 3D`, `[2] 1st-Person POV`, `[3] 3D Side Depth`, `[4] Front 2D Majplay`)
   plus free Right-Mouse 3D orbit & Mouse-Wheel zoom.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
import time
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import pygame

from sim_core.config import (
    OUTER_BUTTON_RADIUS_MM,
    R_BUTTON_MM,
    R_HAND_MM,
    R_SCREEN_MM,
    SKILL_PROFILES,
    VIS_ACTIVE_GREEN_RGB,
    VIS_ALERT_RED_RGB,
    VIS_BG_COLOR_RGB,
    VIS_GRID_LINE_RGB,
    VIS_LEFT_ARM_RGB,
    VIS_OUTER_BTN_RGB,
    VIS_RIGHT_ARM_RGB,
    VIS_SCREEN_RING_RGB,
    Z_HOVER_DEFAULT_MM,
    HitMode,
)
from sim_core.geometry import (
    CANONICAL_33_SENSORS,
    OUTER_BUTTONS,
    POLYGON_C_FULL,
    TOUCH_ZONES,
    Point2D,
    Point3D,
    button_angle_deg,
    polar_cw_to_xy,
)
from sim_core.kinematics import ArmIKSolution, BodyKinematicsState, TorsoState
from sim_core.maidata_parser import NoteType, ParsedChart, ParsedNote, get_available_difficulties
from sim_core.majplay_assets import MM_TO_LOGICAL, MajplaySFXPlayer, MajplaySkinRenderer
from sim_core.planner import PlanResult, TrajectoryFrame


@dataclass
class Camera3D:
    """3D Perspective Camera for viewing the Maimai cabinet and 3D human player."""
    name: str = "3rd-Person 3D"
    yaw_deg: float = -26.0          # Horizontal orbit angle (deg; negative = viewing from front-right)
    pitch_deg: float = 14.0         # Vertical elevation angle (deg)
    dist_mm: float = 1160.0         # Camera distance from look-at target (mm)
    focal_px: float = 960.0         # Perspective focal length (px)
    target_xyz: Point3D = (0.0, -25.0, 95.0)
    cabinet_tilt_deg: float = 10.0  # Real Maimai DX cabinet screen backward tilt (deg)
    is_ortho_2d: bool = False
    is_first_person: bool = False

    def to_world_xyz(self, pt_xyz: Point3D, is_screen_plane: bool = False) -> Point3D:
        x, y, z = pt_xyz
        if self.is_ortho_2d:
            return (x, y, z)
        tilt_rad = math.radians(self.cabinet_tilt_deg)
        ct, st = math.cos(tilt_rad), math.sin(tilt_rad)
        wy = y * ct + z * st
        wz = -y * st + z * ct
        return (x, wy, wz)

    def project(
        self,
        pt_xyz: Point3D,
        vp_cx: int,
        vp_cy: int,
        is_screen_plane: bool = False,
    ) -> Tuple[int, int, float, float]:
        """Project 3D point `(x, y, z)` to `(px, py, cam_depth_z, scale_factor)`."""
        wx, wy, wz = self.to_world_xyz(pt_xyz, is_screen_plane=is_screen_plane)
        tx, ty, tz = self.target_xyz
        dx = wx - tx
        dy = wy - ty
        dz = wz - tz

        yaw_rad = math.radians(self.yaw_deg)
        pitch_rad = math.radians(self.pitch_deg)
        cy, sy = math.cos(yaw_rad), math.sin(yaw_rad)
        cp, sp = math.cos(pitch_rad), math.sin(pitch_rad)

        rx = dx * cy + dz * sy
        rz1 = -dx * sy + dz * cy
        ry = dy * cp - rz1 * sp
        rz = dy * sp + rz1 * cp

        cam_z = max(140.0, self.dist_mm - rz)
        scale = self.focal_px / cam_z
        px = int(round(vp_cx + rx * scale))
        py = int(round(vp_cy - ry * scale))
        return (px, py, cam_z, scale)


CAMERA_PRESETS: List[Camera3D] = [
    Camera3D(
        name="1: 3rd-Person 3D (第三人称立体手元)",
        yaw_deg=-26.0,
        pitch_deg=14.0,
        dist_mm=1150.0,
        focal_px=960.0,
        target_xyz=(15.0, -35.0, 95.0),
        cabinet_tilt_deg=10.0,
    ),
    Camera3D(
        name="2: 1st-Person POV (第一人称主视角)",
        yaw_deg=0.0,
        pitch_deg=-9.0,
        dist_mm=780.0,
        focal_px=740.0,
        target_xyz=(0.0, -15.0, 35.0),
        cabinet_tilt_deg=10.0,
        is_first_person=True,
    ),
    Camera3D(
        name="3: 3D Side Depth (3D侧视前后冲程)",
        yaw_deg=-52.0,
        pitch_deg=12.0,
        dist_mm=1240.0,
        focal_px=980.0,
        target_xyz=(0.0, -45.0, 130.0),
        cabinet_tilt_deg=10.0,
    ),
    Camera3D(
        name="4: Front Majplay 2D (正视谱面视角)",
        yaw_deg=0.0,
        pitch_deg=0.0,
        dist_mm=1020.0,
        focal_px=960.0,
        target_xyz=(0.0, -25.0, 0.0),
        cabinet_tilt_deg=0.0,
        is_ortho_2d=True,
    ),
]


class KinematicVisualizer:
    """Renders the 3D Maimai DX Cabinet + MajdataPlay Skin Canvas + 3D Human Upper Body & 2-Link Arms."""

    def __init__(
        self,
        width_px: int = 1120,
        height_px: int = 1140,
        mm_to_px: float = 0.96,
        center_offset_y_px: int = -55,
        use_3d: bool = True,
    ) -> None:
        if not pygame.get_init():
            pygame.init()
        if not pygame.font.get_init():
            pygame.font.init()

        self.width_px = width_px
        self.height_px = height_px
        self.mm_to_px = mm_to_px
        self.cx_px = width_px // 2
        self.cy_px = height_px // 2 + center_offset_y_px
        self.use_3d = use_3d
        self.camera = Camera3D(
            name=CAMERA_PRESETS[0].name,
            yaw_deg=CAMERA_PRESETS[0].yaw_deg,
            pitch_deg=CAMERA_PRESETS[0].pitch_deg,
            dist_mm=CAMERA_PRESETS[0].dist_mm,
            focal_px=CAMERA_PRESETS[0].focal_px,
            target_xyz=CAMERA_PRESETS[0].target_xyz,
            cabinet_tilt_deg=CAMERA_PRESETS[0].cabinet_tilt_deg,
        )
        self.skin_renderer = MajplaySkinRenderer(canvas_size=620)
        self._cached_chart: Optional[ParsedChart] = None
        self._warp_cache_key: Optional[Tuple[float, float, float, bool]] = None
        self._warp_mat: Optional[np.ndarray] = None
        self._warp_bbox: Tuple[int, int, int, int] = (0, 0, width_px, height_px)
        self._static_3d_bg_surf: Optional[pygame.Surface] = None

        self.font_hud = pygame.font.SysFont("consolas,dejavusansmono,monospace", 14, bold=False)
        self.font_palm = pygame.font.SysFont("consolas,arial,sans-serif", 16, bold=True)
        self.font_banner = pygame.font.SysFont("consolas,dejavusansmono,monospace", 16, bold=True)
        self.font_title = pygame.font.SysFont("microsoftyahei,simhei,consolas,sans-serif", 18, bold=True)
        self.font_ui = pygame.font.SysFont("microsoftyahei,simhei,consolas,sans-serif", 14, bold=False)

    def mm_to_screen(self, pt_mm: Point2D) -> Tuple[int, int]:
        sx = int(round(self.cx_px + pt_mm[0] * self.mm_to_px))
        sy = int(round(self.cy_px - pt_mm[1] * self.mm_to_px))
        return (sx, sy)

    def mm_len_to_px(self, length_mm: float) -> int:
        return max(1, int(round(length_mm * self.mm_to_px)))

    def _draw_capsule_3d(
        self,
        surface: pygame.Surface,
        p1_px: Tuple[int, int],
        p2_px: Tuple[int, int],
        r1_px: int,
        r2_px: int,
        color_rgb: Tuple[int, int, int],
        outline_only: bool = False,
    ) -> None:
        """Draw a 3D shaded tapered capsule segment between p1_px (radius r1_px) and p2_px (radius r2_px).
        When `outline_only=True`, draws only the silhouette contour for X-ray occlusion rendering.
        """
        x1, y1 = p1_px
        x2, y2 = p2_px
        dx = x2 - x1
        dy = y2 - y1
        length = math.hypot(dx, dy)
        dark_rgb = (max(0, color_rgb[0] - 42), max(0, color_rgb[1] - 42), max(0, color_rgb[2] - 42))
        hi_rgb = (min(255, color_rgb[0] + 45), min(255, color_rgb[1] + 45), min(255, color_rgb[2] + 45))
        if outline_only:
            out_rgb = (min(255, color_rgb[0] + 75), min(255, color_rgb[1] + 75), min(255, color_rgb[2] + 75))
            if length > 1e-3:
                nx = -dy / length
                ny = dx / length
                p1_top = (int(round(x1 + nx * r1_px)), int(round(y1 + ny * r1_px)))
                p2_top = (int(round(x2 + nx * r2_px)), int(round(y2 + ny * r2_px)))
                p2_bot = (int(round(x2 - nx * r2_px)), int(round(y2 - ny * r2_px)))
                p1_bot = (int(round(x1 - nx * r1_px)), int(round(y1 - ny * r1_px)))
                pygame.draw.line(surface, out_rgb, p1_top, p2_top, width=2)
                pygame.draw.line(surface, out_rgb, p1_bot, p2_bot, width=2)
            pygame.draw.circle(surface, out_rgb, p1_px, r1_px, width=2)
            pygame.draw.circle(surface, out_rgb, p2_px, r2_px, width=2)
            return
        if length > 1e-3:
            nx = -dy / length
            ny = dx / length
            quad = [
                (int(round(x1 + nx * r1_px)), int(round(y1 + ny * r1_px))),
                (int(round(x2 + nx * r2_px)), int(round(y2 + ny * r2_px))),
                (int(round(x2 - nx * r2_px)), int(round(y2 - ny * r2_px))),
                (int(round(x1 - nx * r1_px)), int(round(y1 - ny * r1_px))),
            ]
            pygame.draw.polygon(surface, color_rgb, quad)
            pygame.draw.polygon(surface, dark_rgb, quad, width=2)
            h_p1 = (int(round(x1 + nx * r1_px * 0.35)), int(round(y1 + ny * r1_px * 0.35)))
            h_p2 = (int(round(x2 + nx * r2_px * 0.35)), int(round(y2 + ny * r2_px * 0.35)))
            pygame.draw.line(surface, hi_rgb, h_p1, h_p2, width=max(2, min(r1_px, r2_px) // 4))
        pygame.draw.circle(surface, color_rgb, p1_px, r1_px)
        pygame.draw.circle(surface, color_rgb, p2_px, r2_px)

    def _draw_joint_rivet(
        self,
        surface: pygame.Surface,
        center_px: Tuple[int, int],
        outer_radius_px: int,
        arm_color_rgb: Tuple[int, int, int],
        with_center_pin: bool = True,
    ) -> None:
        ring_color = (28, 28, 30)
        pygame.draw.circle(surface, arm_color_rgb, center_px, outer_radius_px)
        pygame.draw.circle(surface, ring_color, center_px, outer_radius_px, width=2)
        if with_center_pin:
            pygame.draw.circle(surface, (24, 26, 28), center_px, max(4, outer_radius_px // 4))
        else:
            inner_r = max(5, int(outer_radius_px * 0.32))
            pygame.draw.circle(surface, (215, 215, 220), center_px, inner_r, width=2)

    def _warp_majplay_screen_3d(
        self,
        maj_surf: pygame.Surface,
        cam: Camera3D,
    ) -> pygame.Surface:
        """Perspective-warp the 2D `MajplaySkinRenderer` canvas onto the 3D tilted Maimai cabinet quad."""
        half_mm = 270.0 / MM_TO_LOGICAL  # 241.88 mm
        corners_3d = [
            (-half_mm, half_mm, 0.0),
            (half_mm, half_mm, 0.0),
            (half_mm, -half_mm, 0.0),
            (-half_mm, -half_mm, 0.0),
        ]
        ckey = (round(cam.yaw_deg, 2), round(cam.pitch_deg, 2), round(cam.dist_mm, 1), cam.is_ortho_2d)
        if self._warp_cache_key != ckey or self._static_3d_bg_surf is None or self._warp_mat is None:
            self._warp_cache_key = ckey
            dst_pts = np.array(
                [cam.project(c, self.cx_px, self.cy_px, is_screen_plane=True)[:2] for c in corners_3d],
                dtype=np.float32,
            )
            x_min = max(0, int(np.floor(dst_pts[:, 0].min())) - 2)
            y_min = max(0, int(np.floor(dst_pts[:, 1].min())) - 2)
            x_max = min(self.width_px, int(np.ceil(dst_pts[:, 0].max())) + 2)
            y_max = min(self.height_px, int(np.ceil(dst_pts[:, 1].max())) + 2)
            self._warp_bbox = (x_min, y_min, max(16, x_max - x_min), max(16, y_max - y_min))
            cs = maj_surf.get_width()
            # Note: pygame.surfarray.array3d is (W, H, 3) where axis 0 is X and axis 1 is Y.
            # In OpenCV warpPerspective on (W, H, 3), OpenCV treats dim 1 (H) as cols (x) and dim 0 (W) as rows (y)!
            # So if we pass transposed `(y, x)` points directly to `getPerspectiveTransform`, we avoid BOTH np.transpose calls!
            src_yx = np.array([[0, 0], [0, cs - 1], [cs - 1, cs - 1], [cs - 1, 0]], dtype=np.float32)
            dst_yx = np.column_stack([dst_pts[:, 1] - y_min, dst_pts[:, 0] - x_min]).astype(np.float32)
            self._warp_mat = cv2.getPerspectiveTransform(src_yx, dst_yx)

            bg_s = pygame.Surface((self.width_px, self.height_px))
            bg_s.fill((24, 23, 26))
            floor_y = -385.0
            for z_mm in (0.0, 100.0, 200.0, 300.0, 400.0):
                p_l = cam.project((-340.0, floor_y, z_mm), self.cx_px, self.cy_px)[:2]
                p_r = cam.project((340.0, floor_y, z_mm), self.cx_px, self.cy_px)[:2]
                col = (95, 100, 115) if z_mm == 0.0 else (50, 52, 60)
                pygame.draw.line(bg_s, col, p_l, p_r, width=2 if z_mm == 0.0 else 1)
                if not cam.is_ortho_2d and z_mm in (0.0, 100.0, 200.0, 300.0):
                    lbl = self.font_hud.render(f"Z={int(z_mm)}mm", True, (110, 115, 128))
                    bg_s.blit(lbl, (p_r[0] + 6, p_r[1] - 8))
            for x_mm in (-300.0, -150.0, 0.0, 150.0, 300.0):
                p_0 = cam.project((x_mm, floor_y, 0.0), self.cx_px, self.cy_px)[:2]
                p_1 = cam.project((x_mm, floor_y, 420.0), self.cx_px, self.cy_px)[:2]
                pygame.draw.line(bg_s, (46, 48, 56), p_0, p_1, width=1)
            back_z = -75.0
            front_pts = [tuple(map(int, p)) for p in dst_pts]
            back_pts = [cam.project((c[0], c[1], back_z), self.cx_px, self.cy_px, is_screen_plane=True)[:2] for c in corners_3d]
            for i in range(4):
                pygame.draw.line(bg_s, (62, 64, 72), front_pts[i], back_pts[i], width=2)
            self._static_3d_bg_surf = bg_s

        out_surf = self._static_3d_bg_surf.copy()
        bx, by, bw, bh = self._warp_bbox
        if cam.is_ortho_2d:
            scaled = pygame.transform.smoothscale(maj_surf, (bw, bh))
            out_surf.blit(scaled, (bx, by))
        else:
            arr_whc = pygame.surfarray.array3d(maj_surf)
            # Warp directly in (W, H, 3) layout without transposing!
            warped_whc = cv2.warpPerspective(
                arr_whc,
                self._warp_mat,
                (bh, bw),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(24, 23, 26),
            )
            box_surf = pygame.surfarray.make_surface(warped_whc)
            out_surf.blit(box_surf, (bx, by))

        return out_surf

    def _draw_3d_upper_body(
        self,
        surface: pygame.Surface,
        torso: TorsoState,
        left_arm: ArmIKSolution,
        right_arm: ArmIKSolution,
        cam: Camera3D,
    ) -> None:
        """Render the standing 3D human upper body (head aligned with upper-middle/top screen y ≈ +265mm,
        shoulders at y ≈ +65mm, pelvis at y = -380mm) in 3D space.
        """
        if cam.is_first_person:
            return
        s_l_3d: Point3D = left_arm.shoulder_xyz
        s_r_3d: Point3D = right_arm.shoulder_xyz
        chest_3d: Point3D = (
            0.5 * (s_l_3d[0] + s_r_3d[0]),
            0.5 * (s_l_3d[1] + s_r_3d[1]),
            0.5 * (s_l_3d[2] + s_r_3d[2]),
        )
        pelvis_3d: Point3D = (torso.dx_mm, -380.0 + torso.dy_mm * 0.5, chest_3d[2] + 35.0)
        neck_3d: Point3D = (chest_3d[0], chest_3d[1] + 75.0, chest_3d[2] - 15.0)
        head_3d: Point3D = (chest_3d[0], chest_3d[1] + 200.0, chest_3d[2] - 30.0)

        yaw_rad = math.radians(torso.yaw_deg)
        hip_l_3d: Point3D = (pelvis_3d[0] - 110.0 * math.cos(yaw_rad), pelvis_3d[1], pelvis_3d[2] - 90.0 * math.sin(yaw_rad))
        hip_r_3d: Point3D = (pelvis_3d[0] + 110.0 * math.cos(yaw_rad), pelvis_3d[1], pelvis_3d[2] + 90.0 * math.sin(yaw_rad))

        p_pelvis = cam.project(pelvis_3d, self.cx_px, self.cy_px)[:2]
        p_neck = cam.project(neck_3d, self.cx_px, self.cy_px)[:2]
        hx, hy, _, sc_head = cam.project(head_3d, self.cx_px, self.cy_px)
        p_sl = cam.project(s_l_3d, self.cx_px, self.cy_px)[:2]
        p_sr = cam.project(s_r_3d, self.cx_px, self.cy_px)[:2]
        p_hl = cam.project(hip_l_3d, self.cx_px, self.cy_px)[:2]
        p_hr = cam.project(hip_r_3d, self.cx_px, self.cy_px)[:2]

        torso_poly = [p_sl, p_sr, p_hr, p_hl]
        pygame.draw.polygon(surface, (95, 106, 125), torso_poly, width=2)
        pygame.draw.line(surface, (110, 122, 145), p_pelvis, p_neck, width=4)
        pygame.draw.line(surface, (130, 145, 170), p_sl, p_sr, width=5)
        head_r = max(14, int(round(58.0 * sc_head)))
        pygame.draw.circle(surface, (115, 128, 150), (hx, hy), head_r, width=2)

    def _arm_screen_bbox(self, arm: ArmIKSolution, cam: Camera3D) -> pygame.Rect:
        """Compute 2D screen-space bounding box of a 3D arm (shoulder, elbow, palm)."""
        sx, sy, _, s_sc = cam.project(arm.shoulder_xyz, self.cx_px, self.cy_px)
        ex, ey, _, e_sc = cam.project(arm.elbow_xyz, self.cx_px, self.cy_px)
        p_3d: Point3D = (arm.palm_xyz[0], arm.palm_xyz[1], max(0.0, arm.palm_xyz[2]))
        px, py, _, p_sc = cam.project(p_3d, self.cx_px, self.cy_px, is_screen_plane=True)
        pad = max(24, int(round(R_HAND_MM * max(s_sc, e_sc, p_sc) * 1.15)))
        x0 = max(0, min(sx, ex, px) - pad)
        y0 = max(0, min(sy, ey, py) - pad)
        x1 = min(self.width_px, max(sx, ex, px) + pad)
        y1 = min(self.height_px, max(sy, ey, py) + pad)
        return pygame.Rect(x0, y0, max(1, x1 - x0), max(1, y1 - y0))

    def _draw_3d_arm_and_palm(
        self,
        surface: pygame.Surface,
        arm: ArmIKSolution,
        torso: TorsoState,
        color_rgb: Tuple[int, int, int],
        mode_label: str,
        freq_hz: float,
        cam: Camera3D,
        outline_only: bool = False,
        mask_only: bool = False,
        offset_xy: Tuple[int, int] = (0, 0),
    ) -> None:
        """Render a 3D 2-link arm, screen shadow, Z-depth stem, and Doraemon round palm.
        When `outline_only=True`, draws only the silhouette contour for X-ray occlusion rendering.
        When `mask_only=True`, draws only the solid opaque body silhouette for occlusion masking.
        """
        s_3d: Point3D = arm.shoulder_xyz
        e_3d: Point3D = arm.elbow_xyz
        p_z = max(0.0, arm.palm_xyz[2])
        p_3d: Point3D = (arm.palm_xyz[0], arm.palm_xyz[1], p_z)
        shadow_3d: Point3D = (arm.palm_xyz[0], arm.palm_xyz[1], 0.0)

        sx, sy, _, s_sc = cam.project(s_3d, self.cx_px, self.cy_px)
        ex, ey, _, e_sc = cam.project(e_3d, self.cx_px, self.cy_px)
        px, py, _, p_sc = cam.project(p_3d, self.cx_px, self.cy_px, is_screen_plane=True)
        shx, shy, _, sh_sc = cam.project(shadow_3d, self.cx_px, self.cy_px, is_screen_plane=True)

        ox, oy = offset_xy
        s_px = (sx + ox, sy + oy)
        e_px = (ex + ox, ey + oy)
        p_px = (px + ox, py + oy)
        sh_px = (shx + ox, shy + oy)

        r_s = max(10, int(round(25.0 * s_sc)))
        r_e = max(10, int(round(24.0 * e_sc)))
        r_p = max(10, int(round(22.0 * p_sc)))
        palm_r_px = max(14, int(round(R_HAND_MM * p_sc)))

        if mask_only:
            self._draw_capsule_3d(surface, s_px, e_px, r_s, r_e, color_rgb)
            self._draw_capsule_3d(surface, e_px, p_px, r_e, r_p, color_rgb)
            pygame.draw.circle(surface, color_rgb, s_px, max(12, int(round(30.0 * s_sc))))
            pygame.draw.circle(surface, color_rgb, e_px, max(12, int(round(30.0 * e_sc))))
            pygame.draw.circle(surface, color_rgb, p_px, int(round(palm_r_px * 0.62)))
            return

        if outline_only:
            out_rgb = (min(255, color_rgb[0] + 85), min(255, color_rgb[1] + 85), min(255, color_rgb[2] + 85))
            self._draw_capsule_3d(surface, s_px, e_px, r_s, r_e, color_rgb, outline_only=True)
            self._draw_capsule_3d(surface, e_px, p_px, r_e, r_p, color_rgb, outline_only=True)
            pygame.draw.circle(surface, out_rgb, e_px, max(12, int(round(30.0 * e_sc))), width=2)
            pygame.draw.circle(surface, out_rgb, p_px, int(round(palm_r_px * 0.62)), width=2)
            pygame.draw.circle(surface, out_rgb, p_px, palm_r_px, width=1)
            return

        # 1. Draw 3D Drop Shadow on the screen surface (Z = 0mm) & Z-Depth Stem directly on `surface` (no full-screen SRCALPHA!)
        sh_r_px = max(10, int(round(R_HAND_MM * sh_sc * 0.62)))
        if p_z > 4.0:
            pygame.draw.circle(surface, color_rgb, sh_px, sh_r_px, width=2)
            pygame.draw.line(surface, color_rgb, sh_px, p_px, width=2)
            pygame.draw.circle(surface, (230, 230, 235), sh_px, 4)
        else:
            pygame.draw.circle(surface, (255, 255, 240), sh_px, int(sh_r_px * 1.35), width=3)

        # 2. Draw 3D Upper Arm (S -> E) & 3D Forearm (E -> P) with perspective thickness
        self._draw_capsule_3d(surface, s_px, e_px, r_s, r_e, color_rgb)
        self._draw_capsule_3d(surface, e_px, p_px, r_e, r_p, color_rgb)

        # 3. Draw Shoulder & Elbow mechanical joint rivets
        self._draw_joint_rivet(surface, s_px, max(12, int(round(30.0 * s_sc))), color_rgb, with_center_pin=False)
        self._draw_joint_rivet(surface, e_px, max(12, int(round(30.0 * e_sc))), color_rgb, with_center_pin=True)

        # 4. Draw 3D Doraemon Round Palm at `(px, py)`
        palm_r_px = max(14, int(round(R_HAND_MM * p_sc)))
        halo_r_px = int(round(palm_r_px * 1.38))
        p_surf = pygame.Surface((halo_r_px * 2 + 8, halo_r_px * 2 + 8), pygame.SRCALPHA)
        c_loc = (halo_r_px + 4, halo_r_px + 4)
        pygame.draw.circle(p_surf, (*color_rgb, 100), c_loc, halo_r_px, width=1)
        is_contact = p_z <= 15.0
        core_r = int(round(palm_r_px * 0.62))
        pygame.draw.circle(p_surf, (*color_rgb, 235 if is_contact else 190), c_loc, core_r)
        hi_col = (min(255, color_rgb[0] + 60), min(255, color_rgb[1] + 60), min(255, color_rgb[2] + 60), 180)
        pygame.draw.circle(p_surf, hi_col, (c_loc[0] - core_r // 4, c_loc[1] - core_r // 4), max(4, core_r // 3))
        pygame.draw.circle(p_surf, (28, 28, 30, 230), c_loc, core_r, width=2)
        surface.blit(p_surf, (px - halo_r_px - 4, py - halo_r_px - 4))

        lbl_s = self.font_palm.render(arm.side, True, (250, 250, 250))
        surface.blit(lbl_s, lbl_s.get_rect(center=p_px))

        # 5. Telemetry HUD text beside palm & elbow
        hover_state = "Hover" if p_z > 15.0 else "HIT"
        reach_mm = int(round(arm.reach_margin_mm))
        z_mm = int(round(p_z))
        hz_suffix = f" | {freq_hz:.1f}Hz" if freq_hz > 0.1 else ""
        hud_line = f"{hover_state} | {mode_label} | reach {reach_mm}mm | z={z_mm}mm{hz_suffix}"
        hud_surf = self.font_hud.render(hud_line, True, color_rgb)
        surface.blit(hud_surf, (px + int(palm_r_px * 0.85), py - 10))

        elbow_deg_str = f"{int(round(arm.elbow_angle_deg))} deg"
        elbow_surf = self.font_hud.render(elbow_deg_str, True, (180, 185, 195))
        surface.blit(elbow_surf, (ex + 28, ey - 8))

    def render_frame_surface(
        self,
        frame: TrajectoryFrame,
        notes: Sequence[ParsedNote] = (),
        chart: Optional[ParsedChart] = None,
    ) -> pygame.Surface:
        """Render a single `TrajectoryFrame` with 3D MajdataPlay cabinet + 3D human upper body & arms."""
        if chart is None and self._cached_chart is not None:
            chart = self._cached_chart
        if chart is None:
            chart = ParsedChart(
                title="MaimaiChart",
                artist="",
                difficulty_label="Master",
                level_str="14",
                first_offset_sec=0.0,
                initial_bpm=frame.bpm,
                notes=list(notes),
                events=[],
                total_duration_sec=max(10.0, frame.time + 1.0),
                bar_timestamps=[(frame.bar, 0.0, frame.time + 4.0)],
            )

        trig_zones = list(frame.left_zones) + list(frame.right_zones)
        has_alert = len(frame.active_unreasonable) > 0

        maj_canvas = self.skin_renderer.render_game_canvas(
            chart=chart,
            current_time=frame.time,
            active_zones=frame.active_note_zones,
            triggered_zones=trig_zones,
            has_alert=has_alert,
        )

        surface = self._warp_majplay_screen_3d(maj_canvas, self.camera)

        self._draw_3d_upper_body(
            surface,
            torso=frame.body.torso,
            left_arm=frame.body.left_arm,
            right_arm=frame.body.right_arm,
            cam=self.camera,
        )

        # Compute true camera-ray depth (cam_z: larger = farther from camera, smaller = closer to camera)
        l_mid_3d = (
            0.5 * (frame.body.left_arm.elbow_xyz[0] + frame.body.left_arm.palm_xyz[0]),
            0.5 * (frame.body.left_arm.elbow_xyz[1] + frame.body.left_arm.palm_xyz[1]),
            0.5 * (frame.body.left_arm.elbow_xyz[2] + frame.body.left_arm.palm_xyz[2]),
        )
        r_mid_3d = (
            0.5 * (frame.body.right_arm.elbow_xyz[0] + frame.body.right_arm.palm_xyz[0]),
            0.5 * (frame.body.right_arm.elbow_xyz[1] + frame.body.right_arm.palm_xyz[1]),
            0.5 * (frame.body.right_arm.elbow_xyz[2] + frame.body.right_arm.palm_xyz[2]),
        )
        _, _, z_cam_l, _ = self.camera.project(l_mid_3d, self.cx_px, self.cy_px)
        _, _, z_cam_r, _ = self.camera.project(r_mid_3d, self.cx_px, self.cy_px)
        l_tuple = (frame.body.left_arm, VIS_LEFT_ARM_RGB, frame.left_mode, frame.left_freq_hz)
        r_tuple = (frame.body.right_arm, VIS_RIGHT_ARM_RGB, frame.right_mode, frame.right_freq_hz)
        if frame.body.arm_layer_order == "L_OVER_R":
            ordered_arms = [r_tuple, l_tuple]
        elif frame.body.arm_layer_order == "R_OVER_L":
            ordered_arms = [l_tuple, r_tuple]
        elif z_cam_l >= z_cam_r:
            # Left arm is farther from camera (`z_cam_l >= z_cam_r`), so draw Left first, then Right on top
            ordered_arms = [l_tuple, r_tuple]
        else:
            ordered_arms = [r_tuple, l_tuple]

        for draw_idx, (arm_sol, col_rgb, m_lbl, f_hz) in enumerate(ordered_arms):
            if draw_idx == 1 and frame.body.crossing_point_xy is not None:
                cx_mm, cy_mm = frame.body.crossing_point_xy
                c_px_x, c_px_y, _, c_sc = self.camera.project((cx_mm, cy_mm, 35.0), self.cx_px, self.cy_px)
                occ_r = max(16, int(round(36.0 * c_sc)))
                sh_surf = pygame.Surface((occ_r * 2 + 4, occ_r * 2 + 4), pygame.SRCALPHA)
                pygame.draw.circle(sh_surf, (8, 8, 12, 165), (occ_r + 2, occ_r + 2), occ_r)
                surface.blit(sh_surf, (c_px_x - occ_r - 2, c_px_y - occ_r - 2))
            self._draw_3d_arm_and_palm(
                surface,
                arm=arm_sol,
                torso=frame.body.torso,
                color_rgb=col_rgb,
                mode_label=m_lbl,
                freq_hz=f_hz,
                cam=self.camera,
            )

        # X-Ray Occlusion Silhouette Pass:
        # Strictly the portion of the farther arm (`ordered_arms[0]`) that is occluded behind the closer arm
        # (`ordered_arms[1]`) is rendered as a hollow contour outline on top!
        far_arm_sol, far_col_rgb, far_lbl, far_hz = ordered_arms[0]
        near_arm_sol, near_col_rgb, near_lbl, near_hz = ordered_arms[1]
        far_bbox = self._arm_screen_bbox(far_arm_sol, self.camera)
        near_bbox = self._arm_screen_bbox(near_arm_sol, self.camera)
        clip_rect = far_bbox.clip(near_bbox)
        if clip_rect.width > 4 and clip_rect.height > 4:
            cw, ch = clip_rect.width, clip_rect.height
            off_xy = (-clip_rect.x, -clip_rect.y)
            near_mask_surf = pygame.Surface((cw, ch), pygame.SRCALPHA)
            far_out_surf = pygame.Surface((cw, ch), pygame.SRCALPHA)
            self._draw_3d_arm_and_palm(
                near_mask_surf,
                arm=near_arm_sol,
                torso=frame.body.torso,
                color_rgb=near_col_rgb,
                mode_label=near_lbl,
                freq_hz=near_hz,
                cam=self.camera,
                mask_only=True,
                offset_xy=off_xy,
            )
            self._draw_3d_arm_and_palm(
                far_out_surf,
                arm=far_arm_sol,
                torso=frame.body.torso,
                color_rgb=far_col_rgb,
                mode_label=far_lbl,
                freq_hz=far_hz,
                cam=self.camera,
                outline_only=True,
                offset_xy=off_xy,
            )
            mask_alpha = pygame.surfarray.pixels_alpha(near_mask_surf)
            out_alpha = pygame.surfarray.pixels_alpha(far_out_surf)
            out_alpha[:] = np.minimum(out_alpha, mask_alpha)
            del mask_alpha, out_alpha
            surface.blit(far_out_surf, clip_rect.topleft)

        if frame.body.crossing_point_xy is not None:
            cx_mm, cy_mm = frame.body.crossing_point_xy
            c_px_x, c_px_y, _, _ = self.camera.project((cx_mm, cy_mm, 65.0), self.cx_px, self.cy_px)
            layer_lbl = "L OVER R" if frame.body.arm_layer_order == "L_OVER_R" else "R OVER L"
            l_surf = self.font_hud.render(f"[{layer_lbl}]", True, (255, 235, 120))
            surface.blit(l_surf, (c_px_x - 32, c_px_y - 28))

        torso = frame.body.torso
        top_hud = (
            f"[{self.camera.name}]  t={frame.time:6.3f}s | Bar {frame.bar:03d} | {frame.bpm:5.1f} BPM | "
            f"Torso dx={torso.dx_mm:+5.1f}mm dy={torso.dy_mm:+5.1f}mm yaw={torso.yaw_deg:+5.1f}deg"
        )
        surface.blit(self.font_hud.render(top_hud, True, (210, 215, 225)), (18, 14))
        cam_hint = "Camera: [1] 3rd-Person 3D  [2] 1st-Person POV  [3] 3D Side Depth  [4] Front 2D  |  Right-Drag: 3D Orbit  Wheel: Zoom"
        surface.blit(self.font_hud.render(cam_hint, True, (145, 152, 168)), (18, 34))

        if frame.active_unreasonable:
            u0 = frame.active_unreasonable[0]
            alert_text = f"[UNREASONABLE: {u0.type}] {u0.reason}"
            surface.blit(self.font_banner.render(alert_text, True, VIS_ALERT_RED_RGB), (18, 56))

        return surface

    def render_frame_rgb(self, frame: TrajectoryFrame) -> np.ndarray:
        surf = self.render_frame_surface(frame)
        arr_whc = pygame.surfarray.array3d(surf)
        return np.transpose(arr_whc, (1, 0, 2)).copy()

    def save_frame_png(self, frame: TrajectoryFrame, output_path: str | Path) -> Path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        rgb = self.render_frame_rgb(frame)
        cv2.imwrite(str(out), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        return out

    def export_mp4(
        self,
        plan: PlanResult,
        output_path: str | Path,
        fps: int = 60,
        max_frames: Optional[int] = None,
    ) -> Path:
        self._cached_chart = plan.chart
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out), fourcc, float(fps), (self.width_px, self.height_px))
        frames = plan.trajectory_frames[:max_frames] if max_frames else plan.trajectory_frames
        try:
            for frame in frames:
                rgb = self.render_frame_rgb(frame)
                writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        finally:
            writer.release()
        return out

    def save_gui_snapshot_png(
        self,
        plan: PlanResult,
        eval_result: object,
        play_time: float,
        output_path: str | Path,
        chart_source_path: Optional[str | Path] = None,
        camera_preset_idx: int = 0,
    ) -> Path:
        """Render a full 1530x1140 3D GUI snapshot to PNG."""
        self._cached_chart = plan.chart
        if chart_source_path:
            self.skin_renderer.load_song_bg(chart_source_path)
        preset = CAMERA_PRESETS[camera_preset_idx % len(CAMERA_PRESETS)]
        self.camera = Camera3D(**preset.__dict__)

        sidebar_w = 410
        win_w = self.width_px + sidebar_w
        win_h = self.height_px
        screen = pygame.Surface((win_w, win_h))

        frames = plan.trajectory_frames
        t_min = frames[0].time if frames else 0.0
        t_max = frames[-1].time if frames else 1.0
        f_dt = max(1e-4, (t_max - t_min) / max(1, len(frames) - 1))
        f_idx = max(0, min(len(frames) - 1, int(round((play_time - t_min) / f_dt))))
        cur_frame = frames[f_idx]

        main_surf = self.render_frame_surface(cur_frame, notes=plan.chart.notes, chart=plan.chart)
        screen.blit(main_surf, (0, 0))
        avail_diffs = get_available_difficulties(chart_source_path) if chart_source_path else [("5", "MAS", plan.chart.level_str)]
        self._draw_sidebar_and_timeline(
            screen=screen,
            cur_plan=plan,
            cur_eval=eval_result,
            cur_frame=cur_frame,
            play_time=play_time,
            t_min=t_min,
            t_max=t_max,
            paused=False,
            speed_val=1.0,
            avail_diffs=avail_diffs,
        )
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        arr_whc = pygame.surfarray.array3d(screen)
        rgb = np.transpose(arr_whc, (1, 0, 2)).copy()
        cv2.imwrite(str(out), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        return out

    def _draw_sidebar_and_timeline(
        self,
        screen: pygame.Surface,
        cur_plan: PlanResult,
        cur_eval: object,
        cur_frame: TrajectoryFrame,
        play_time: float,
        t_min: float,
        t_max: float,
        paused: bool,
        speed_val: float,
        avail_diffs: Sequence[Tuple[str, str, str]] = (),
    ) -> None:
        sidebar_w = 410
        win_h = self.height_px
        timeline_rect = pygame.Rect(24, win_h - 42, self.width_px - 48, 16)
        pygame.draw.rect(screen, (52, 50, 54), timeline_rect, border_radius=6)
        prog_ratio = max(0.0, min(1.0, (play_time - t_min) / max(1e-4, t_max - t_min)))
        fill_w = int(round(timeline_rect.width * prog_ratio))
        if fill_w > 0:
            pygame.draw.rect(
                screen,
                VIS_LEFT_ARM_RGB,
                pygame.Rect(timeline_rect.x, timeline_rect.y, fill_w, timeline_rect.height),
                border_radius=6,
            )
        for u in cur_plan.unreasonable_events:
            ur = max(0.0, min(1.0, (u.time - t_min) / max(1e-4, t_max - t_min)))
            ux = timeline_rect.x + int(round(ur * timeline_rect.width))
            pygame.draw.line(screen, VIS_ALERT_RED_RGB, (ux, timeline_rect.y - 4), (ux, timeline_rect.bottom + 4), width=3)
        pygame.draw.circle(screen, (245, 245, 245), (timeline_rect.x + fill_w, timeline_rect.centery), 9)

        sb_rect = pygame.Rect(self.width_px, 0, sidebar_w, win_h)
        pygame.draw.rect(screen, (26, 25, 29), sb_rect)
        pygame.draw.line(screen, (68, 70, 78), (self.width_px, 0), (self.width_px, win_h), width=2)

        x0 = self.width_px + 20
        touch_cnt = sum(1 for n in cur_plan.chart.notes if n.note_type in (NoteType.TOUCH, NoteType.TOUCH_HOLD))
        screen.blit(self.font_title.render(f"{cur_plan.chart.title}  [Lv.{cur_plan.chart.level_str}]", True, (240, 240, 245)), (x0, 16))
        screen.blit(
            self.font_ui.render(
                f"Artist: {cur_plan.chart.artist[:18]} | Notes: {len(cur_plan.chart.notes)} (Touch: {touch_cnt})",
                True,
                (165, 168, 175),
            ),
            (x0, 42),
        )

        is_play = getattr(cur_eval, "is_playable", len(cur_plan.unreasonable_events) == 0)
        erg_score = getattr(cur_eval, "total_ergonomic_score", 0.0)
        ach_str = getattr(cur_eval, "simulated_achievement_rate_str", "100.0000%")
        badge_col = VIS_ACTIVE_GREEN_RGB if is_play else VIS_ALERT_RED_RGB
        badge_txt = "PLAYABLE (PASS)" if is_play else f"UNREASONABLE ({len(cur_plan.unreasonable_events)} ERR)"
        pygame.draw.rect(screen, badge_col, pygame.Rect(x0, 68, 370, 28), width=1, border_radius=6)
        screen.blit(self.font_banner.render(badge_txt, True, badge_col), (x0 + 12, 73))
        screen.blit(self.font_banner.render(f"Ergonomic: {erg_score:5.1f}/100   |   Sim DX: {ach_str}", True, (230, 232, 240)), (x0, 102))
        screen.blit(self.font_hud.render(f"Time: {play_time:6.2f}s / {t_max:6.2f}s   |   Bar: {cur_frame.bar:03d}", True, (175, 180, 190)), (x0, 126))

        # Song Library & Open File row (y = 152)
        btn_lib_rect = pygame.Rect(self.width_px + 20, 152, 192, 28)
        btn_open_rect = pygame.Rect(self.width_px + 220, 152, 170, 28)
        for rect, lbl in (
            (btn_lib_rect, "Song Library (Touch/DX) [L]"),
            (btn_open_rect, "Open maidata.txt [O]"),
        ):
            pygame.draw.rect(screen, (66, 112, 92), rect, border_radius=5)
            pygame.draw.rect(screen, (125, 195, 155), rect, width=1, border_radius=5)
            ts = self.font_hud.render(lbl, True, (242, 250, 245))
            screen.blit(ts, ts.get_rect(center=rect.center))

        # Difficulty Switcher row (y = 186)
        diff_map = {d[0]: (d[1], d[2]) for d in avail_diffs}
        diff_colors = {
            "2": (52, 125, 68),
            "3": (155, 125, 42),
            "4": (155, 58, 62),
            "5": (118, 56, 155),
            "6": (145, 95, 175),
        }
        for di, d_key in enumerate(("2", "3", "4", "5", "6")):
            d_rect = pygame.Rect(self.width_px + 20 + di * 75, 186, 70, 26)
            has_d = d_key in diff_map
            s_name, lv_s = diff_map.get(d_key, ({"2": "BAS", "3": "ADV", "4": "EXP", "5": "MAS", "6": "ReM"}[d_key], "-"))
            is_cur_d = has_d and (cur_plan.chart.difficulty_label == f"inote_{d_key}" or cur_plan.chart.level_str == lv_s)
            bg_c = diff_colors[d_key] if is_cur_d else ((48, 50, 58) if has_d else (32, 32, 36))
            pygame.draw.rect(screen, bg_c, d_rect, border_radius=4)
            pygame.draw.rect(screen, (220, 225, 240) if is_cur_d else (82, 86, 96), d_rect, width=2 if is_cur_d else 1, border_radius=4)
            d_txt = f"{s_name} {lv_s}" if has_d else s_name
            ts_d = self.font_hud.render(d_txt[:8], True, (245, 245, 250) if has_d else (95, 98, 105))
            screen.blit(ts_d, ts_d.get_rect(center=d_rect.center))

        btn_play_rect = pygame.Rect(self.width_px + 20, 218, 110, 28)
        btn_speed_rect = pygame.Rect(self.width_px + 140, 218, 110, 28)
        btn_restart_rect = pygame.Rect(self.width_px + 260, 218, 110, 28)
        btn_mode_rect = pygame.Rect(self.width_px + 20, 252, 175, 28)
        btn_prof_rect = pygame.Rect(self.width_px + 205, 252, 185, 28)
        btn_cam1_rect = pygame.Rect(self.width_px + 20, 286, 88, 26)
        btn_cam2_rect = pygame.Rect(self.width_px + 114, 286, 88, 26)
        btn_cam3_rect = pygame.Rect(self.width_px + 208, 286, 88, 26)
        btn_cam4_rect = pygame.Rect(self.width_px + 302, 286, 88, 26)

        for rect, lbl, act in (
            (btn_play_rect, "PAUSE [Spc]" if not paused else "PLAY [Spc]", not paused),
            (btn_speed_rect, f"Speed: {speed_val:.2f}x", False),
            (btn_restart_rect, "Restart [R]", False),
            (btn_mode_rect, f"Mode: {cur_plan.hit_mode.value.upper()} [M]", True),
            (btn_prof_rect, f"{cur_plan.profile.name} [P]", True),
            (btn_cam1_rect, "[1] 3D 3rd", "1:" in self.camera.name),
            (btn_cam2_rect, "[2] 3D POV", "2:" in self.camera.name),
            (btn_cam3_rect, "[3] 3D Side", "3:" in self.camera.name),
            (btn_cam4_rect, "[4] 2D Ortho", "4:" in self.camera.name),
        ):
            pygame.draw.rect(screen, (58, 96, 134) if act else (46, 46, 54), rect, border_radius=5)
            pygame.draw.rect(screen, (98, 102, 115), rect, width=1, border_radius=5)
            ts = self.font_hud.render(lbl, True, (238, 240, 245))
            screen.blit(ts, ts.get_rect(center=rect.center))

        y_tel = 322
        screen.blit(self.font_banner.render("3D Biomechanics & Depth Telemetry", True, (200, 205, 215)), (x0, y_tel))
        t_state = cur_frame.body.torso
        l_arm = cur_frame.body.left_arm
        r_arm = cur_frame.body.right_arm
        decomp = getattr(cur_eval, "decomposition_summary", {})
        fly_rate = getattr(cur_eval, "fly_hand_rate", 0.0)
        fly_cnt = getattr(cur_eval, "fly_hand_count", 0)
        tel_lines = [
            f"Torso Shift : dx={t_state.dx_mm:+6.1f}mm  dy={t_state.dy_mm:+6.1f}mm",
            f"Torso Yaw   : {t_state.yaw_deg:+5.1f} deg  (vel={t_state.yaw_vel_deg_s:+6.1f} deg/s)",
            f"L Palm (3D) : {cur_frame.left_mode:10s} | Z={l_arm.palm_xyz[2]:4.0f}mm | {cur_frame.left_freq_hz:4.1f}Hz",
            f"R Palm (3D) : {cur_frame.right_mode:10s} | Z={r_arm.palm_xyz[2]:4.0f}mm | {cur_frame.right_freq_hz:4.1f}Hz",
            f"Arm Layer   : {cur_frame.body.arm_layer_order:8s} | 3D Dist={cur_frame.body.forearm_min_dist_3d_mm:5.1f}mm",
            f"Tech Stats  : Split={decomp.get('OUTER_INNER_SPLIT',0)} "
            f"Sweep={decomp.get('BUTTON_SWEEP',0)} Rub={decomp.get('SINGLE_PALM_RUB',0)} Fly={fly_rate:.1f}%({fly_cnt})",
        ]
        for idx_l, line_str in enumerate(tel_lines):
            col = VIS_LEFT_ARM_RGB if idx_l == 2 else (VIS_RIGHT_ARM_RGB if idx_l == 3 else (175, 180, 190))
            screen.blit(self.font_hud.render(line_str, True, col), (x0, y_tel + 24 + idx_l * 21))

        y_plot = 472
        screen.blit(self.font_banner.render("Per-Bar Stamina / Fatigue Curve", True, (200, 205, 215)), (x0, y_plot))
        plot_rect = pygame.Rect(x0, y_plot + 24, 370, 92)
        pygame.draw.rect(screen, (36, 35, 40), plot_rect, border_radius=4)
        pygame.draw.rect(screen, (72, 74, 82), plot_rect, width=1, border_radius=4)
        f_curve = getattr(cur_eval, "fatigue_curve", [])
        if len(f_curve) >= 2:
            max_f = max(max(f_curve), 25.0)
            pts_f = [
                (
                    plot_rect.x + int(round((bi / max(1, len(f_curve) - 1)) * (plot_rect.width - 8))) + 4,
                    plot_rect.bottom - 4 - int(round(min(1.0, fv / max_f) * (plot_rect.height - 10))),
                )
                for bi, fv in enumerate(f_curve)
            ]
            pygame.draw.lines(screen, VIS_LEFT_ARM_RGB, False, pts_f, width=2)
            cur_bx = plot_rect.x + int(round((cur_frame.bar / max(1, len(f_curve) - 1)) * (plot_rect.width - 8))) + 4
            pygame.draw.line(screen, (245, 245, 245), (cur_bx, plot_rect.y + 2), (cur_bx, plot_rect.bottom - 2), width=1)

        y_unr = 596
        screen.blit(
            self.font_banner.render(
                f"Unreasonable Events ({len(cur_plan.unreasonable_events)}) [Click to Jump]",
                True,
                (200, 205, 215),
            ),
            (x0, y_unr),
        )
        if not cur_plan.unreasonable_events:
            screen.blit(
                self.font_hud.render("No physical unreasonable patterns detected.", True, VIS_ACTIVE_GREEN_RGB),
                (x0, y_unr + 28),
            )
        else:
            for ui, u_rec in enumerate(cur_plan.unreasonable_events[:14]):
                row_r = pygame.Rect(x0, y_unr + 26 + ui * 30, 370, 26)
                pygame.draw.rect(
                    screen,
                    (90, 38, 38) if abs(u_rec.time - play_time) <= 0.35 else (42, 38, 42),
                    row_r,
                    border_radius=4,
                )
                keys_s = ",".join(u_rec.keys[:3])
                row_txt = f"Bar {u_rec.bar:03d} ({u_rec.time:5.2f}s) | {u_rec.type[:18]} [{keys_s}]"
                screen.blit(self.font_hud.render(row_txt, True, (245, 155, 150)), (row_r.x + 8, row_r.y + 5))

    @staticmethod
    def _scan_song_library() -> List[Tuple[str, str, str]]:
        """Fast scan of local song folders (`official charts`, `test_songs`, `out`).
        Returns list of `(display_name, category_tag, maidata_path)`.
        """
        entries: List[Tuple[str, str, str]] = []
        # 1. Custom / AI generated / test songs first
        custom_dirs = [
            Path(r"G:\GitHub\maiAIV3\out\echo\listen\Habit\on"),
            Path(r"G:\GitHub\maiAIV3\test_songs"),
            Path(r"G:\GitHub\maiAIV3\test_songs_v2"),
        ]
        for c_path in custom_dirs:
            if c_path.is_dir():
                if (c_path / "maidata.txt").exists():
                    entries.append((f"[AI/Out] {c_path.parent.name} ({c_path.name})", "CUSTOM", str(c_path / "maidata.txt")))
                else:
                    try:
                        for sub in os.scandir(str(c_path)):
                            if sub.is_dir():
                                mf = os.path.join(sub.path, "maidata.txt")
                                if os.path.exists(mf):
                                    entries.append((f"[Test] {sub.name}", "CUSTOM", mf))
                    except Exception:
                        pass
        # 2. Official charts (`G:\GitHub\maiAIV3\official charts`)
        off_root = r"G:\GitHub\maiAIV3\official charts"
        if os.path.isdir(off_root):
            try:
                off_list = []
                for sub in os.scandir(off_root):
                    if sub.is_dir():
                        tag = "DX" if "[DX]" in sub.name else "STD"
                        off_list.append((sub.name, tag, os.path.join(sub.path, "maidata.txt")))
                off_list.sort(key=lambda x: (0 if x[1] == "DX" else 1, x[0].lower()))
                entries.extend(off_list)
            except Exception:
                pass
        return entries

    @staticmethod
    def _open_maidata_file_dialog(initial_dir: Optional[str] = None) -> Optional[str]:
        """Open native Windows file picker for `maidata.txt`."""
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            init_d = initial_dir if (initial_dir and os.path.isdir(initial_dir)) else r"G:\GitHub\maiAIV3\official charts"
            selected = filedialog.askopenfilename(
                title="Select Maimai Chart (maidata.txt)",
                initialdir=init_d,
                filetypes=[("Simai Chart (maidata.txt)", "maidata.txt"), ("All Text Files", "*.txt"), ("All Files", "*.*")],
            )
            root.destroy()
            return selected if selected else None
        except Exception:
            return None

    def run_interactive(
        self,
        plan: PlanResult,
        fps: int = 60,
        audio_path: Optional[str | Path] = None,
        eval_result: Optional[object] = None,
        chart_source_path: Optional[str | Path] = None,
    ) -> None:
        """Open the interactive 3D MajdataPlay Simulator + Human Biomechanical Player GUI."""
        from sim_core.scorer import evaluate_chart

        self._cached_chart = plan.chart
        if chart_source_path:
            self.skin_renderer.load_song_bg(chart_source_path)

        sidebar_w = 410
        win_w = self.width_px + sidebar_w
        win_h = self.height_px
        screen = pygame.display.set_mode((win_w, win_h))
        pygame.display.set_caption(
            f"3D MajdataPlay & Human Play Simulator - {plan.chart.title} [{plan.chart.level_str}]"
        )
        clock = pygame.time.Clock()

        sfx_player = MajplaySFXPlayer()
        sfx_player.build_schedule(plan.chart)

        has_audio = False
        resolved_audio: Optional[Path] = None
        if audio_path and Path(audio_path).exists():
            resolved_audio = Path(audio_path)
        elif chart_source_path:
            p_src = Path(chart_source_path)
            folder = p_src if p_src.is_dir() else p_src.parent
            for cand in ("track.mp3", "track.ogg", "track.wav"):
                if (folder / cand).exists():
                    resolved_audio = folder / cand
                    break
        if resolved_audio is not None:
            try:
                if not pygame.mixer.get_init():
                    pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
                pygame.mixer.music.load(str(resolved_audio))
                pygame.mixer.music.set_volume(0.55)
                has_audio = True
            except Exception:
                has_audio = False

        modes = [HitMode.HYBRID, HitMode.OUTER, HitMode.INNER]
        profiles = list(SKILL_PROFILES.keys())
        speeds = [1.0, 0.5, 0.25]
        speed_idx = 0

        cur_plan = plan
        cur_eval = eval_result
        if cur_eval is None:
            cur_eval = evaluate_chart(
                cur_plan.chart,
                hit_mode=cur_plan.hit_mode,
                skill_profile=cur_plan.profile,
                frame_fps=fps,
            )
            cur_plan = cur_eval.plan_result or cur_plan

        cur_source_path: Optional[Path] = Path(chart_source_path) if chart_source_path else None
        avail_diffs: List[Tuple[str, str, str]] = (
            get_available_difficulties(cur_source_path) if cur_source_path else [("5", "MAS", cur_plan.chart.level_str)]
        )
        song_library = self._scan_song_library()
        lib_open = False
        lib_tab = "DX"  # 'DX' (Touch/DX charts), 'ALL', or 'CUSTOM'
        lib_query = ""
        lib_scroll = 0

        frames = cur_plan.trajectory_frames
        t_min = frames[0].time if frames else 0.0
        t_max = frames[-1].time if frames else 1.0
        play_time = 0.0
        sfx_player.sync_to_time(play_time)
        paused = False
        dragging_timeline = False
        orbit_dragging = False
        orbit_last_pos = (0, 0)
        song_anchor_sec = 0.0
        wall_anchor_sec = time.perf_counter()
        if has_audio:
            try:
                pygame.mixer.music.play(start=0.0)
                wall_anchor_sec = time.perf_counter()
            except Exception:
                has_audio = False

        timeline_rect = pygame.Rect(24, win_h - 42, self.width_px - 48, 16)
        btn_lib_rect = pygame.Rect(self.width_px + 20, 152, 192, 28)
        btn_open_rect = pygame.Rect(self.width_px + 220, 152, 170, 28)
        diff_btn_rects = [
            (d_k, pygame.Rect(self.width_px + 20 + di * 75, 186, 70, 26))
            for di, d_k in enumerate(("2", "3", "4", "5", "6"))
        ]
        btn_play_rect = pygame.Rect(self.width_px + 20, 218, 110, 28)
        btn_speed_rect = pygame.Rect(self.width_px + 140, 218, 110, 28)
        btn_restart_rect = pygame.Rect(self.width_px + 260, 218, 110, 28)
        btn_mode_rect = pygame.Rect(self.width_px + 20, 252, 175, 28)
        btn_prof_rect = pygame.Rect(self.width_px + 205, 252, 185, 28)
        cam_btn_rects = [
            pygame.Rect(self.width_px + 20, 286, 88, 26),
            pygame.Rect(self.width_px + 114, 286, 88, 26),
            pygame.Rect(self.width_px + 208, 286, 88, 26),
            pygame.Rect(self.width_px + 302, 286, 88, 26),
        ]

        def _get_filtered_library() -> List[Tuple[str, str, str]]:
            q = lib_query.strip().lower()
            out_items = []
            for name, tag, mpath in song_library:
                if lib_tab == "DX" and tag != "DX":
                    continue
                if lib_tab == "CUSTOM" and tag != "CUSTOM":
                    continue
                if q and q not in name.lower():
                    continue
                out_items.append((name, tag, mpath))
            return out_items

        def _load_song_or_diff(target_source: Path, difficulty: str = "auto") -> None:
            nonlocal cur_source_path, avail_diffs, cur_eval, cur_plan, frames, t_min, t_max, has_audio, resolved_audio, paused
            if has_audio:
                try:
                    pygame.mixer.music.stop()
                except Exception:
                    pass
            cur_source_path = target_source
            avail_diffs = get_available_difficulties(cur_source_path)
            use_diff = difficulty
            if use_diff == "auto" and avail_diffs:
                # Prefer MAS ('5') or ReM ('6'), else highest available
                avail_keys = [d[0] for d in avail_diffs]
                use_diff = "5" if "5" in avail_keys else avail_diffs[-1][0]
            cur_eval = evaluate_chart(
                cur_source_path,
                difficulty=use_diff,
                hit_mode=cur_plan.hit_mode,
                skill_profile=cur_plan.profile.name,
                frame_fps=fps,
            )
            cur_plan = cur_eval.plan_result or cur_plan
            self._cached_chart = cur_plan.chart
            self.skin_renderer.load_song_bg(cur_source_path)
            sfx_player.build_schedule(cur_plan.chart)
            frames = cur_plan.trajectory_frames
            t_min = frames[0].time if frames else 0.0
            t_max = frames[-1].time if frames else 1.0
            pygame.display.set_caption(
                f"3D MajdataPlay & Human Play Simulator - {cur_plan.chart.title} [{cur_plan.chart.level_str}]"
            )
            folder = cur_source_path if cur_source_path.is_dir() else cur_source_path.parent
            resolved_audio = None
            has_audio = False
            for cand in ("track.mp3", "track.ogg", "track.wav"):
                if (folder / cand).exists():
                    resolved_audio = folder / cand
                    break
            if resolved_audio is not None:
                try:
                    pygame.mixer.music.load(str(resolved_audio))
                    pygame.mixer.music.set_volume(0.55)
                    has_audio = True
                except Exception:
                    has_audio = False
            paused = False
            _seek_to(0.0)

        def _seek_to(target_sec: float) -> None:
            nonlocal play_time, song_anchor_sec, wall_anchor_sec
            play_time = max(0.0, min(t_max, target_sec))
            song_anchor_sec = play_time
            wall_anchor_sec = time.perf_counter()
            sfx_player.sync_to_time(play_time)
            if has_audio and not paused and abs(speeds[speed_idx] - 1.0) < 1e-3:
                try:
                    pygame.mixer.music.play(start=max(0.0, play_time))
                    wall_anchor_sec = time.perf_counter()
                except Exception:
                    pass

        def _rebuild_plan(new_mode: HitMode, new_profile: str) -> None:
            nonlocal cur_eval, cur_plan, frames, t_min, t_max
            cur_eval = evaluate_chart(
                cur_plan.chart,
                hit_mode=new_mode,
                skill_profile=new_profile,
                frame_fps=fps,
            )
            cur_plan = cur_eval.plan_result or cur_plan
            frames = cur_plan.trajectory_frames
            t_min = frames[0].time if frames else 0.0
            t_max = frames[-1].time if frames else 1.0

        running = True
        while running and frames:
            dt_real = clock.tick(fps) / 1000.0
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if lib_open:
                        if event.key == pygame.K_ESCAPE:
                            lib_open = False
                        elif event.key == pygame.K_BACKSPACE:
                            lib_query = lib_query[:-1]
                            lib_scroll = 0
                        elif event.key == pygame.K_UP:
                            lib_scroll = max(0, lib_scroll - 1)
                        elif event.key == pygame.K_DOWN:
                            lib_scroll += 1
                        elif event.key == pygame.K_PAGEUP:
                            lib_scroll = max(0, lib_scroll - 15)
                        elif event.key == pygame.K_PAGEDOWN:
                            lib_scroll += 15
                        elif event.unicode and event.unicode.isprintable() and event.key not in (pygame.K_RETURN, pygame.K_TAB):
                            lib_query += event.unicode
                            lib_scroll = 0
                        continue
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_l:
                        lib_open = not lib_open
                    elif event.key == pygame.K_o:
                        init_d = str(cur_source_path.parent) if cur_source_path else None
                        picked = self._open_maidata_file_dialog(init_d)
                        if picked:
                            _load_song_or_diff(Path(picked), "auto")
                    elif event.key == pygame.K_SPACE:
                        paused = not paused
                        if has_audio:
                            if paused:
                                pygame.mixer.music.pause()
                            else:
                                _seek_to(play_time)
                    elif event.key == pygame.K_r:
                        _seek_to(max(0.0, t_min))
                    elif event.key == pygame.K_LEFT:
                        _seek_to(play_time - 2.0)
                    elif event.key == pygame.K_RIGHT:
                        _seek_to(play_time + 2.0)
                    elif event.key in (pygame.K_1, pygame.K_KP1):
                        self.camera = Camera3D(**CAMERA_PRESETS[0].__dict__)
                    elif event.key in (pygame.K_2, pygame.K_KP2):
                        self.camera = Camera3D(**CAMERA_PRESETS[1].__dict__)
                    elif event.key in (pygame.K_3, pygame.K_KP3):
                        self.camera = Camera3D(**CAMERA_PRESETS[2].__dict__)
                    elif event.key in (pygame.K_4, pygame.K_KP4):
                        self.camera = Camera3D(**CAMERA_PRESETS[3].__dict__)
                    elif event.key == pygame.K_m:
                        nxt_m = modes[(modes.index(cur_plan.hit_mode) + 1) % len(modes)]
                        _rebuild_plan(nxt_m, cur_plan.profile.name)
                    elif event.key == pygame.K_p:
                        nxt_p = profiles[(profiles.index(cur_plan.profile.name) + 1) % len(profiles)]
                        _rebuild_plan(cur_plan.hit_mode, nxt_p)
                elif event.type == pygame.MOUSEWHEEL:
                    if lib_open:
                        lib_scroll = max(0, lib_scroll - event.y * 3)
                    else:
                        self.camera.dist_mm = max(580.0, min(1850.0, self.camera.dist_mm - event.y * 55.0))
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mx, my = event.pos
                    if lib_open and event.button == 1:
                        modal_rect = pygame.Rect(110, 70, 900, 960)
                        if not modal_rect.collidepoint(mx, my):
                            lib_open = False
                            continue
                        # Check filter tabs & close button inside modal
                        tab_dx_r = pygame.Rect(135, 122, 230, 32)
                        tab_all_r = pygame.Rect(375, 122, 180, 32)
                        tab_cus_r = pygame.Rect(565, 122, 195, 32)
                        btn_clr_r = pygame.Rect(770, 122, 100, 32)
                        btn_cls_r = pygame.Rect(880, 122, 105, 32)
                        if tab_dx_r.collidepoint(mx, my):
                            lib_tab = "DX"
                            lib_scroll = 0
                        elif tab_all_r.collidepoint(mx, my):
                            lib_tab = "ALL"
                            lib_scroll = 0
                        elif tab_cus_r.collidepoint(mx, my):
                            lib_tab = "CUSTOM"
                            lib_scroll = 0
                        elif btn_clr_r.collidepoint(mx, my):
                            lib_query = ""
                            lib_scroll = 0
                        elif btn_cls_r.collidepoint(mx, my):
                            lib_open = False
                        else:
                            items = _get_filtered_library()
                            max_sc = max(0, len(items) - 22)
                            lib_scroll = min(lib_scroll, max_sc)
                            for ri in range(22):
                                idx_item = lib_scroll + ri
                                if idx_item >= len(items):
                                    break
                                row_r = pygame.Rect(135, 212 + ri * 35, 850, 31)
                                if row_r.collidepoint(mx, my):
                                    lib_open = False
                                    _load_song_or_diff(Path(items[idx_item][2]), "auto")
                                    break
                        continue
                    if event.button == 3 and mx < self.width_px:
                        orbit_dragging = True
                        orbit_last_pos = (mx, my)
                        self.camera.is_ortho_2d = False
                    elif event.button == 1:
                        if timeline_rect.inflate(0, 14).collidepoint(mx, my):
                            dragging_timeline = True
                            ratio = (mx - timeline_rect.x) / max(1, timeline_rect.width)
                            _seek_to(t_min + ratio * (t_max - t_min))
                        elif btn_lib_rect.collidepoint(mx, my):
                            lib_open = not lib_open
                        elif btn_open_rect.collidepoint(mx, my):
                            init_d = str(cur_source_path.parent) if cur_source_path else None
                            picked = self._open_maidata_file_dialog(init_d)
                            if picked:
                                _load_song_or_diff(Path(picked), "auto")
                        elif btn_play_rect.collidepoint(mx, my):
                            paused = not paused
                            if has_audio:
                                if paused:
                                    pygame.mixer.music.pause()
                                else:
                                    _seek_to(play_time)
                        elif btn_speed_rect.collidepoint(mx, my):
                            speed_idx = (speed_idx + 1) % len(speeds)
                            if has_audio:
                                if abs(speeds[speed_idx] - 1.0) > 1e-3:
                                    pygame.mixer.music.pause()
                                elif not paused:
                                    _seek_to(play_time)
                        elif btn_restart_rect.collidepoint(mx, my):
                            _seek_to(max(0.0, t_min))
                        elif btn_mode_rect.collidepoint(mx, my):
                            nxt_m = modes[(modes.index(cur_plan.hit_mode) + 1) % len(modes)]
                            _rebuild_plan(nxt_m, cur_plan.profile.name)
                        elif btn_prof_rect.collidepoint(mx, my):
                            nxt_p = profiles[(profiles.index(cur_plan.profile.name) + 1) % len(profiles)]
                            _rebuild_plan(cur_plan.hit_mode, nxt_p)
                        else:
                            diff_clicked = False
                            if cur_source_path is not None:
                                avail_set = {d[0] for d in avail_diffs}
                                for d_k, d_rect in diff_btn_rects:
                                    if d_rect.collidepoint(mx, my) and d_k in avail_set:
                                        _load_song_or_diff(cur_source_path, d_k)
                                        diff_clicked = True
                                        break
                            if diff_clicked:
                                continue
                            for ci, c_rect in enumerate(cam_btn_rects):
                                if c_rect.collidepoint(mx, my):
                                    self.camera = Camera3D(**CAMERA_PRESETS[ci].__dict__)
                                    break
                            y_unr = 596
                            x0 = self.width_px + 20
                            for ui, u_rec in enumerate(cur_plan.unreasonable_events[:14]):
                                row_r = pygame.Rect(x0, y_unr + 26 + ui * 30, 370, 26)
                                if row_r.collidepoint(mx, my):
                                    _seek_to(u_rec.time)
                                    break
                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:
                        dragging_timeline = False
                    elif event.button == 3:
                        orbit_dragging = False
                elif event.type == pygame.MOUSEMOTION:
                    mx, my = event.pos
                    if dragging_timeline:
                        ratio = max(0.0, min(1.0, (mx - timeline_rect.x) / max(1, timeline_rect.width)))
                        _seek_to(t_min + ratio * (t_max - t_min))
                    elif orbit_dragging:
                        dx_m = mx - orbit_last_pos[0]
                        dy_m = my - orbit_last_pos[1]
                        orbit_last_pos = (mx, my)
                        self.camera.yaw_deg = max(-78.0, min(78.0, self.camera.yaw_deg + dx_m * 0.35))
                        self.camera.pitch_deg = max(-35.0, min(55.0, self.camera.pitch_deg + dy_m * 0.30))
                        self.camera.name = f"Free 3D (yaw={self.camera.yaw_deg:.0f}, pitch={self.camera.pitch_deg:.0f})"

            if not paused and not dragging_timeline:
                cur_speed = speeds[speed_idx]
                if has_audio and abs(cur_speed - 1.0) < 1e-3 and pygame.mixer.music.get_busy():
                    wall_elapsed = time.perf_counter() - wall_anchor_sec
                    mixer_pos_ms = pygame.mixer.music.get_pos()
                    if mixer_pos_ms >= 0:
                        mixer_sec = song_anchor_sec + (mixer_pos_ms / 1000.0)
                        wall_sec = song_anchor_sec + wall_elapsed
                        # Lock to hardware audio clock if wall clock drifts by >25ms
                        if abs(wall_sec - mixer_sec) > 0.025:
                            wall_anchor_sec = time.perf_counter() - (mixer_pos_ms / 1000.0)
                            play_time = mixer_sec
                        else:
                            play_time = wall_sec
                    else:
                        play_time = song_anchor_sec + wall_elapsed
                else:
                    play_time += dt_real * cur_speed
                    song_anchor_sec = play_time
                    wall_anchor_sec = time.perf_counter()

                if play_time >= t_max:
                    _seek_to(0.0)
                else:
                    sfx_player.update(play_time)

            f_dt = max(1e-4, (t_max - t_min) / max(1, len(frames) - 1))
            f_idx = max(0, min(len(frames) - 1, int(round((play_time - t_min) / f_dt))))
            cur_frame = frames[f_idx]

            main_surf = self.render_frame_surface(cur_frame, notes=cur_plan.chart.notes, chart=cur_plan.chart)
            screen.blit(main_surf, (0, 0))
            self._draw_sidebar_and_timeline(
                screen=screen,
                cur_plan=cur_plan,
                cur_eval=cur_eval,
                cur_frame=cur_frame,
                play_time=play_time,
                t_min=t_min,
                t_max=t_max,
                paused=paused,
                speed_val=speeds[speed_idx],
                avail_diffs=avail_diffs,
            )
            if lib_open:
                modal_rect = pygame.Rect(110, 70, 900, 960)
                pygame.draw.rect(screen, (22, 24, 30), modal_rect, border_radius=10)
                pygame.draw.rect(screen, (110, 155, 210), modal_rect, width=2, border_radius=10)
                screen.blit(
                    self.font_title.render("Select Song / Chart  (Click a song to load; [DX] charts include Touch notes)", True, (240, 245, 255)),
                    (135, 84),
                )
                tab_dx_r = pygame.Rect(135, 122, 230, 32)
                tab_all_r = pygame.Rect(375, 122, 180, 32)
                tab_cus_r = pygame.Rect(565, 122, 195, 32)
                btn_clr_r = pygame.Rect(770, 122, 100, 32)
                btn_cls_r = pygame.Rect(880, 122, 105, 32)
                for r_tab, t_lbl, is_act in (
                    (tab_dx_r, "Touch / [DX] Charts", lib_tab == "DX"),
                    (tab_all_r, "All Official Charts", lib_tab == "ALL"),
                    (tab_cus_r, "Custom / AI Charts", lib_tab == "CUSTOM"),
                    (btn_clr_r, "Clear Filter", False),
                    (btn_cls_r, "Close [Esc]", False),
                ):
                    pygame.draw.rect(screen, (58, 105, 150) if is_act else (44, 46, 56), r_tab, border_radius=5)
                    pygame.draw.rect(screen, (135, 175, 220) if is_act else (88, 92, 105), r_tab, width=1, border_radius=5)
                    ts_t = self.font_hud.render(t_lbl, True, (245, 248, 255))
                    screen.blit(ts_t, ts_t.get_rect(center=r_tab.center))

                search_r = pygame.Rect(135, 164, 850, 34)
                pygame.draw.rect(screen, (34, 36, 44), search_r, border_radius=5)
                pygame.draw.rect(screen, (95, 130, 175), search_r, width=1, border_radius=5)
                items = _get_filtered_library()
                q_disp = lib_query if lib_query else "(Type on keyboard to search song name... Mouse Wheel to scroll)"
                q_col = (245, 248, 255) if lib_query else (140, 145, 160)
                screen.blit(self.font_ui.render(f"Search: {q_disp}   [{len(items)} charts]", True, q_col), (148, 171))

                max_sc = max(0, len(items) - 22)
                lib_scroll = min(lib_scroll, max_sc)
                for ri in range(22):
                    idx_item = lib_scroll + ri
                    if idx_item >= len(items):
                        break
                    s_name, s_tag, s_path = items[idx_item]
                    row_r = pygame.Rect(135, 212 + ri * 35, 850, 31)
                    is_cur_song = (cur_source_path is not None and os.path.normcase(str(cur_source_path)) == os.path.normcase(s_path))
                    pygame.draw.rect(screen, (52, 82, 115) if is_cur_song else (36, 38, 46), row_r, border_radius=4)
                    pygame.draw.rect(screen, (85, 125, 165) if is_cur_song else (62, 65, 76), row_r, width=1, border_radius=4)
                    tag_col = (110, 225, 165) if s_tag == "DX" else ((245, 200, 110) if s_tag == "CUSTOM" else (165, 175, 195))
                    screen.blit(self.font_hud.render(f"[{s_tag:6s}]", True, tag_col), (row_r.x + 10, row_r.y + 7))
                    screen.blit(self.font_ui.render(s_name[:68], True, (238, 240, 246)), (row_r.x + 88, row_r.y + 5))
            pygame.display.flip()

        if has_audio:
            try:
                pygame.mixer.music.stop()
            except Exception:
                pass
        pygame.quit()
