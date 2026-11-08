"""Unit tests for the Human Maimai DX Play Simulator & Chart Quality Scorer."""
from __future__ import annotations

from pathlib import Path

import pytest

from sim_core.config import (
    OUTER_INNER_RADIAL_DIFF_MM,
    R_A_CENTER_MM,
    R_BUTTON_MM,
    R_HAND_MM,
    R_SCREEN_MM,
    UnreasonableType,
)
from sim_core.evaluator import evaluate_chart
from sim_core.geometry import (
    CANONICAL_33_SENSORS,
    OUTER_BUTTONS,
    TOUCH_ZONES,
    build_slide_route,
    circle_polygon_intersection_area,
    find_single_palm_cover_center,
    solve_two_palm_coverage,
)
from sim_core.judge import SlideSensorQueueTracker
from sim_core.visualizer import KinematicVisualizer


# ============================================================================
# Mandatory Core Pattern Tests (Section 6.3 Cases 1 ~ 4)
# ============================================================================

def test_case1_220bpm_dual_jackhammer_impossible() -> None:
    """Case 1: 220 BPM 连续 8 个 16 分音符的 45 双押纵连 -> 必须触发 DUAL_JACK_IMPOSSIBLE."""
    simai = "(220){16}45,45,45,45,45,45,45,45,E"
    res = evaluate_chart(simai, hit_mode="hybrid", skill_profile="Lv14+_Master")

    assert res.is_playable is False
    types = [e["type"] for e in res.unreasonable_events]
    assert UnreasonableType.DUAL_JACK_IMPOSSIBLE.value in types
    dual_ev = next(e for e in res.unreasonable_events if e["type"] == UnreasonableType.DUAL_JACK_IMPOSSIBLE.value)
    assert set(dual_ev["keys"]) == {"4", "5"}
    assert res.total_ergonomic_score <= 35.0


def test_case2_240bpm_single_jack_hybrid_split_pass() -> None:
    """Case 2: 240 BPM 连续 8 个 16 分音符的单键 1 纵连（另一只手空闲） ->
    在 hybrid 模式下必须自动规划为“双手内外拆或双手交替”并顺利通过.
    """
    simai = "(240){16}1,1,1,1,1,1,1,1,E"
    res = evaluate_chart(simai, hit_mode="hybrid", skill_profile="Lv14+_Master")

    assert res.is_playable is True, f"Unexpected unreasonable events: {res.unreasonable_events}"
    assert len(res.unreasonable_events) == 0

    # Verify that Outer-Inner Split or Alternating Tap was actively used across the 8 taps
    split_count = (
        res.decomposition_summary["OUTER_INNER_SPLIT"]
        + res.decomposition_summary["ALTERNATING_TAP"]
    )
    assert split_count >= 6

    # Verify hands alternated so neither hand exceeded its single-hand frequency limit (8.5 Hz)
    assert res.plan_result is not None
    active_hands = [
        ("L" if st.left.note_ids else "R")
        for st in res.plan_result.planned_steps
    ]
    # Consecutive events must alternate L and R
    alternations = sum(
        1 for i in range(1, len(active_hands)) if active_hands[i] != active_hands[i - 1]
    )
    assert alternations >= 6
    for st in res.plan_result.planned_steps:
        assert st.left.instant_freq_hz <= res.plan_result.profile.max_single_jack_hz(8) + 0.1
        assert st.right.instant_freq_hz <= res.plan_result.profile.max_single_jack_hz(8) + 0.1


def test_case3_240bpm_single_jack_with_5hold_forced_single_hand() -> None:
    """Case 3: 240 BPM 连续 8 个 16 分音符的单键 1 纵连，同时左手全程按住 5h[4:1] ->
    必须触发 FORCED_SINGLE_HAND_JACK 无理报错.
    """
    simai = "(240){16}1/5h[4:1],1,1,1,1,1,1,1,E"
    res = evaluate_chart(simai, hit_mode="hybrid", skill_profile="Lv14+_Master")

    assert res.is_playable is False
    types = [e["type"] for e in res.unreasonable_events]
    assert UnreasonableType.FORCED_SINGLE_HAND_JACK.value in types
    forced_ev = next(
        e for e in res.unreasonable_events if e["type"] == UnreasonableType.FORCED_SINGLE_HAND_JACK.value
    )
    assert "1" in forced_ev["keys"]
    assert res.total_ergonomic_score <= 35.0


def test_case4_normal_master_14_smooth_chart() -> None:
    """Case 4: 正常的 13+ / 14 级顺手交互与滑条配置 -> 输出高顺手度评分与平滑的双手轨迹."""
    simai = (
        "(180){8}"
        "1,8,2,7,3,6,4-8[8:3],,"
        "5,4,3,6,2-5[8:3],,"
        "1/5,2b/6b,E"
    )
    res = evaluate_chart(simai, hit_mode="hybrid", skill_profile="Lv14+_Master")

    assert res.is_playable is True
    assert len(res.unreasonable_events) == 0
    assert res.total_ergonomic_score >= 78.0
    assert res.simulated_achievement_rate >= 99.5
    assert len(res.fatigue_curve) >= 2
    assert res.plan_result is not None
    assert len(res.plan_result.trajectory_frames) > 30


# ============================================================================
# Additional Subsystem & Edge-Case Verification Tests
# ============================================================================

def test_geometry_33_sensors_and_radial_split() -> None:
    """Verify 33/34 inner touch zones (A1..E8 + C/C1/C2), 8 outer buttons, and 95mm radial separation."""
    assert len(CANONICAL_33_SENSORS) in (33, 34)
    assert len(OUTER_BUTTONS) == 8
    assert abs((R_BUTTON_MM - R_A_CENTER_MM) - OUTER_INNER_RADIAL_DIFF_MM) < 1e-6
    assert abs(OUTER_INNER_RADIAL_DIFF_MM - 95.0) < 1e-6

    # Verify palm at A1 center triggers A1 polygon with substantial overlap area
    a1 = TOUCH_ZONES["A1"]
    area = circle_polygon_intersection_area(a1.center, R_HAND_MM, a1.polygon)
    assert area > 2500.0

    # Adjacent A1 + D1 can be rubbed by a single 48mm Doraemon palm
    rub_pt = find_single_palm_cover_center(["A1", "D1"], R_HAND_MM)
    assert rub_pt is not None


def test_three_hand_conflict_vs_two_palm_rub() -> None:
    """Verify that 3 widely separated notes trigger THREE_HAND_CONFLICT while 2-palm rub passes."""
    # 3 distant simultaneous touch zones A1 + A4 + A7 -> impossible for 2 palms
    simai_conflict = "(160){4}A1/A4/A7,E"
    res_bad = evaluate_chart(simai_conflict, hit_mode="hybrid", skill_profile="Lv14+_Master")
    assert res_bad.is_playable is False
    assert any(e["type"] == UnreasonableType.THREE_HAND_CONFLICT.value for e in res_bad.unreasonable_events)

    # Adjacent A1+B1 rubbed by right palm while left palm hits A5 -> feasible with 2 palms
    simai_rub = "(160){4}A1/B1/A5,E"
    res_ok = evaluate_chart(simai_rub, hit_mode="hybrid", skill_profile="Lv14+_Master")
    assert res_ok.is_playable is True
    assert res_ok.decomposition_summary["SINGLE_PALM_RUB"] >= 1


def test_majdata_slide_skip1_and_tail_anti_early_trigger() -> None:
    """Verify MajdataPlay Slide sensor queue Skip-1 rule and tail anti-early-trigger residence."""
    # 1-5 straight slide queue: A1 -> B1 -> C -> B5 -> A5
    route = build_slide_route("-", 1, 5)
    tracker = SlideSensorQueueTracker(route.queue)
    assert tracker.remaining_count == 5

    # Enter A1 and leave A1 -> advances to B1 (remaining 4)
    tracker.update({"A1"})
    tracker.update(set())
    assert tracker.remaining_count == 4

    # Skip-1 rule: skip B1 and directly touch & release C -> advances past both B1 and C!
    tracker.update({"C"})
    tracker.update(set())
    assert tracker.remaining_count == 2  # Now at B5 -> A5

    # Touch B5 and release -> now at final area A5 (is_pending_finish == True)
    tracker.update({"B5"})
    tracker.update(set())
    assert tracker.is_pending_finish is True
    assert tracker.is_finished is False

    # Hand can leave to hit another tap (e.g. A2) without breaking the slide queue
    tracker.update({"A2"})
    tracker.update(set())
    assert tracker.is_pending_finish is True
    assert tracker.is_finished is False

    # Finally sweep into tail area A5 -> immediately finishes on entry!
    assert tracker.update({"A5"}) is True
    assert tracker.is_finished is True


def test_visualizer_renders_frame_and_mp4(tmp_path: Path) -> None:
    """Verify headless Pygame + OpenCV visualizer renders non-empty PNG and MP4 artifacts."""
    simai = "(180){8}1,8,2,7,3-6[8:2],,4/5,E"
    res = evaluate_chart(simai, hit_mode="hybrid", skill_profile="Lv14+_Master", frame_fps=30)
    assert res.plan_result is not None

    vis = KinematicVisualizer(width_px=800, height_px=840, mm_to_px=0.90)
    png_path = vis.save_frame_png(res.plan_result.trajectory_frames[0], tmp_path / "frame0.png")
    mp4_path = vis.export_mp4(res.plan_result, tmp_path / "preview.mp4", fps=30, max_frames=20)

    assert png_path.exists() and png_path.stat().st_size > 5000
    assert mp4_path.exists() and mp4_path.stat().st_size > 5000
def test_extreme_cross_arm_and_hand_collision() -> None:
    """Verify EXTREME_CROSS_ARM (twisted cross-body hold+tap) and HAND_COLLISION (inner same-zone chord)."""
    # Left hand starts holding 5h, then forced cross-body lock: 3h held on right side + 7h on left side twisted
    simai_cross = "(180){8}7,3h[2:1],6,7,6,7,E"
    # Force left hand onto 3h by occupying right hand on 2h first, then crossing
    simai_forced_cross = "(180){8}5,3h[1:1]/6,7,6,7,E"
    res_cross = evaluate_chart(simai_forced_cross, hit_mode="inner", skill_profile="Lv13+_Expert")
    assert res_cross.plan_result is not None

    # Inner mode simultaneous close sensors (e.g. C1 + C2 hit separately in inner mode without rub)
    # Or direct kinematics check of check_hand_collision & extreme cross arm
    from sim_core.kinematics import solve_coupled_body_ik
    # Left palm at Button 3 (+199, -82, 0), Right palm at Button 6 (-199, -82, 0) -> Severe cross-arm!
    crossed_state = solve_coupled_body_ik(
        time=1.0,
        left_palm_xyz=(199.0, -82.0, 0.0),
        right_palm_xyz=(-199.0, -82.0, 0.0),
    )
    assert crossed_state.is_extreme_cross_arm is True

    # Both palms at A1 (82, 199, 0) -> Hand collision!
    coll_state = solve_coupled_body_ik(
        time=1.0,
        left_palm_xyz=(80.0, 199.0, 0.0),
        right_palm_xyz=(88.0, 199.0, 5.0),
    )
    assert coll_state.is_hand_collision is True


def test_maidata_txt_file_parsing(tmp_path: Path) -> None:
    """Verify parsing a complete maidata.txt file with multiple difficulties."""
    maidata_content = (
        "&title=Test Song\n"
        "&artist=Test Artist\n"
        "&first=0.5\n"
        "&lv_4=12+\n"
        "&inote_4=\n"
        "(160){8}1,2,3,4,5,6,7,8,E\n"
        "&lv_5=14.2\n"
        "&inote_5=\n"
        "(180){8}1,8,2,7,3-6[8:3],,4/5,E\n"
    )
    maidata_file = tmp_path / "maidata.txt"
    maidata_file.write_text(maidata_content, encoding="utf-8")

    res = evaluate_chart(maidata_file, difficulty="master", hit_mode="hybrid", skill_profile="Lv14+_Master")
    assert res.is_playable is True
    assert res.metadata["title"] == "Test Song"
    assert res.metadata["level"] == "14.2"
    assert res.simulated_achievement_rate >= 99.0
def test_button_sweep_and_wifi_two_hand_and_arm_layers() -> None:
    """Verify:
    1. Button Sweep (扫键 1,2,3,4,5,6,7,8 at 240 BPM) uses Sweep mode without false jackhammer errors.
    2. WifiSlide (1w5) occupies BOTH hands simultaneously.
    3. Forearm crossing resolves into L_OVER_R / R_OVER_L 3D layer hierarchy.
    """
    # 1. High-speed unidirectional clockwise adjacent button sweep while other hand holds 7h
    simai_sweep = "(240){32}1/7h[4:1],2,3,4,5,6,E"
    res_sweep = evaluate_chart(simai_sweep, hit_mode="hybrid", skill_profile="Lv14+_Master")
    assert res_sweep.is_playable is True, f"Sweep should be playable, got: {res_sweep.unreasonable_events}"
    assert res_sweep.decomposition_summary["BUTTON_SWEEP"] >= 4

    # 2. WifiSlide (1w5) occupies both L and R hands in trajectory
    simai_wifi = "(160){4}1w5[4:1],,,E"
    res_wifi = evaluate_chart(simai_wifi, hit_mode="hybrid", skill_profile="Lv14+_Master")
    assert res_wifi.plan_result is not None
    both_sliding = any(
        f.left_mode == "Slide" and f.right_mode == "Slide"
        for f in res_wifi.plan_result.trajectory_frames
    )
    assert both_sliding is True

    # 3. 3D Forearm Crossing Layer Hierarchy (`L_OVER_R` / `R_OVER_L`)
    from sim_core.kinematics import solve_coupled_body_ik
    # Left hand crosses to right side (180, -20, z=75) over Right hand crossing to left side (-180, -20, z=0)
    body_layered = solve_coupled_body_ik(
        time=1.0,
        left_palm_xyz=(180.0, -20.0, 75.0),
        right_palm_xyz=(-180.0, -20.0, 0.0),
    )
    assert body_layered.arm_layer_order == "L_OVER_R"
    assert body_layered.crossing_point_xy is not None


# ============================================================================
# Mandatory Biomechanical Anti-Cheat Upgrade Tests (Cases 5 ~ 8)
# ============================================================================

def test_case5_180bpm_same_side_67_trill_two_hand_alternation() -> None:
    """Case 5: 180 BPM 连续 8 个 16 分音符 `6,7,6,7,6,7,6,7` 同侧交互，另一手空闲 ->
    绝不能左手单手跳打，必须转腰（theta_torso 偏向左侧 < 0）并规划为 L-R-L-R 双手交替完成，is_playable == True.
    """
    simai = "(180){16}6,7,6,7,6,7,6,7,E"
    res = evaluate_chart(simai, hit_mode="hybrid", skill_profile="Lv14+_Master")

    assert res.is_playable is True, f"Case 5 should be playable via L-R alternation, got: {res.unreasonable_events}"
    assert res.plan_result is not None
    steps = res.plan_result.planned_steps
    assert len(steps) == 8

    # Verify strict L <-> R alternation (never single-hand hopping)
    hands = ["L" if s.left.note_ids else "R" for s in steps]
    alternations = sum(1 for i in range(1, len(hands)) if hands[i] != hands[i - 1])
    assert alternations >= 6, f"Expected two-hand alternation on 6,7,6,7, got hands={hands}"

    # Verify torso rotates toward the left side (yaw_deg < 0 or dx_mm < 0)
    min_yaw = min(s.body_state.torso.yaw_deg for s in steps)
    min_dx = min(s.body_state.torso.dx_mm for s in steps)
    assert min_yaw < -2.0 or min_dx < -20.0


def test_case6_180bpm_67_trill_with_right_hand_locked_on_2hold() -> None:
    """Case 6: 180 BPM 连续 8 个 16 分音符 `6,7,6,7,6,7,6,7`，同时右手全程按住 `2h[2:1]` ->
    由于右手被 2h 锁死，左手被迫单手打 16 分 6-7 交互，必须精准拦截并输出 FORCED_SINGLE_HAND_JUMP.
    """
    simai = "(180){16}6/2h[2:1],7,6,7,6,7,6,7,E"
    res = evaluate_chart(simai, hit_mode="hybrid", skill_profile="Lv14+_Master")

    assert res.is_playable is False
    types = [e["type"] for e in res.unreasonable_events]
    assert UnreasonableType.FORCED_SINGLE_HAND_JUMP.value in types, (
        f"Expected FORCED_SINGLE_HAND_JUMP, got {res.unreasonable_events}"
    )


def test_case7_240bpm_32nd_clockwise_sweep_vs_oscillation() -> None:
    """Case 7: 240 BPM 连续 8 个 32 分音符顺时针溜冰 `1,2,3,4,5,6,7,8` vs 折返 `6,7,6,7,6,7,6,7` ->
    顺时针 1~8 在 inner/hybrid 下允许单手 RUB_SWEEP 顺畅通过；而 240 BPM 32 分折返 6,7,6,7 严禁触发 RUB_SWEEP.
    """
    simai_cw = "(240){32}1,2,3,4,5,6,7,8,E"
    res_cw = evaluate_chart(simai_cw, hit_mode="hybrid", skill_profile="Lv14+_Master")
    assert res_cw.is_playable is True
    assert res_cw.decomposition_summary["RUB_SWEEP"] >= 6

    simai_osc = "(240){32}6,7,6,7,6,7,6,7,E"
    res_osc = evaluate_chart(simai_osc, hit_mode="hybrid", skill_profile="Lv14+_Master")
    # Oscillating 6->7->6->7 must NEVER be allowed to use RUB_SWEEP!
    assert res_osc.decomposition_summary["RUB_SWEEP"] == 0


def test_case8_forearm_cross_intersection_and_hold_lookahead() -> None:
    """Case 8: 小臂十字交叉测试：按住 7h[2:1] 的同时打 2, 3, 2, 3 ->
    1) 若强制右手按 7h、左手打 2/3 则触发小臂十字交叉 EXTREME_CROSS_ARM；
    2) 正常规划器自动选择“左手按 7h、右手打 2, 3, 2, 3”从而顺畅通过.
    """
    from sim_core.kinematics import segments_intersect_2d, solve_coupled_body_ik
    from sim_core.geometry import TOUCH_ZONES

    # 1) Forced cross: Left hand at A3 (right side) while Right hand at A7 (left side) -> Forearms X-intersect!
    a3_xy = TOUCH_ZONES["A3"].center
    a7_xy = TOUCH_ZONES["A7"].center
    forced_state = solve_coupled_body_ik(
        time=1.0,
        left_palm_xyz=(a3_xy[0], a3_xy[1], 0.0),
        right_palm_xyz=(a7_xy[0], a7_xy[1], 0.0),
    )
    assert forced_state.is_forearm_intersection is True
    assert forced_state.is_extreme_cross_arm is True

    # 2) Planner on 7h[2:1] + 2, 3, 2, 3 automatically assigns L to 7h and R to 2, 3, 2, 3!
    simai = "(160){8}7h[2:1],2,3,2,3,E"
    res = evaluate_chart(simai, hit_mode="hybrid", skill_profile="Lv14+_Master")
    assert res.is_playable is True, f"Expected playable via L=7h, R=2,3,2,3, got {res.unreasonable_events}"
    assert res.plan_result is not None
    # Step 0 is 7h[2:1] -> must be held by Left hand
    assert "A7" in res.plan_result.planned_steps[0].left.zone_names or "OB7" in res.plan_result.planned_steps[0].left.zone_names
    # Subsequent 2, 3, 2, 3 must be hit by Right hand
    for st in res.plan_result.planned_steps[1:]:
        assert len(st.right.note_ids) == 1


def test_case9_avoid_flying_hand_and_flag_unavoidable_fly() -> None:
    """Case 9: 防飞手第一优先级测试：
    1) 可通过转腰反手化解的大风车星星 (`3/7,2/6,1/5,4/8,3-6[8:1]/7-2[8:1]`) -> 自动用反手完成，fly_hand_count == 0 且 is_playable == True；
    2) 无论怎么拆都必然造成巨大瞬时位移（飞手）或极端拧麻花锁死的坏配置 -> 精准拦截并输出 FLYING_HAND / FORCED_SINGLE_HAND_JUMP.
    """
    simai_cross_ok = "(212){32}3/7,2/6,1/5,4/8,3-6[8:1]/7-2[8:1],,,,E"
    res_ok = evaluate_chart(simai_cross_ok, hit_mode="hybrid", skill_profile="Lv15_Apex", avoid_flying_hand=True)
    assert res_ok.is_playable is True, f"Expected playable via cross-hand, got {res_ok.unreasonable_events}"
    assert res_ok.fly_hand_count == 0

    simai_unavoidable_fly = "(240){32}6h[2:1]/1,4,8,4,1,E"
    res_bad = evaluate_chart(simai_unavoidable_fly, hit_mode="hybrid", skill_profile="Lv15_Apex", avoid_flying_hand=True)
    assert res_bad.is_playable is False
    bad_types = {e["type"] for e in res_bad.unreasonable_events}
    assert UnreasonableType.FLYING_HAND.value in bad_types or UnreasonableType.FORCED_SINGLE_HAND_JUMP.value in bad_types
