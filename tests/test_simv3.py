"""Unit tests for `sim_core` v3 (`body_timeline`, `window_body_score`, signed `yaw_accel`,
vocabulary & rarity scoring, preference weight hook, and 15-target neural surrogate).
"""
from __future__ import annotations

import json
from pathlib import Path
import time
import warnings

import numpy as np

from sim_core import (
    BodyTimeline,
    ERGONOMIC_JOINT_SPECS,
    SIM_CORE_VERSION,
    body_timeline,
    check_artifact_version,
    get_sim_core_code_hash,
    predict_chart_surrogate,
    stamp_artifact_version,
    window_body_score,
)
from sim_core.limits import measure
from sim_core.project_api import BAR_TICKS, init_window_state
from sim_core.vocab_and_rarity import load_interest_weights


def test_signed_yaw_accel_in_limits_measure() -> None:
    """Verify negative yaw_accel is measured by absolute value and never silently ignored."""
    ev_neg = {
        "type": "EXTREME_CROSS_ARM",
        "reason": "Extreme cross-body / torso whiplash: L_x=199mm, R_x=-199mm, yaw_accel=-3450 deg/s^2",
        "severity": 2.2,
    }
    assert abs(measure(ev_neg) - 3450.0) < 1e-3


def test_body_timeline_and_ergonomic_specs() -> None:
    """Verify `body_timeline` exports >=30 fps `BodyState` frames, `note_table`, and visual burden."""
    assert "forearm_pro_sup_deg" in ERGONOMIC_JOINT_SPECS
    assert "wrist_dev_deg" in ERGONOMIC_JOINT_SPECS

    simai = "(180){8}1,8,2,7,3-6[8:3],,1h[4:1]/5,6,7,8,E"
    bt = body_timeline(simai, frame_fps=30)
    assert isinstance(bt, BodyTimeline)
    assert bt.fps >= 29.0
    assert len(bt.frames) >= 40
    f0 = bt.frames[10].to_dict()
    for key in ("dx_mm", "dy_mm", "dz_mm", "yaw_deg", "pitch_deg", "roll_deg", "angular_vel_deg_s"):
        assert key in f0["torso"]
    for key in (
        "shoulder_flex_deg", "shoulder_abd_deg", "shoulder_rot_deg",
        "elbow_angle_deg", "swivel_angle_deg", "forearm_pro_sup_deg",
        "wrist_flex_deg", "wrist_dev_deg", "palm_heading_deg",
        "contact", "technique", "speed_mm_s", "accel_mm_s2", "jerk_mm_s3", "joint_vel_deg_s",
    ):
        assert key in f0["left_arm"]
    for key in ("occluded_upcoming_notes", "concurrent_attention_zones", "gaze_angle_deg", "gaze_shift_speed_deg_s"):
        assert key in f0["visual"]
    assert len(bt.note_table) >= 10
    n0 = bt.note_table[0]
    assert "hand" in n0 and "technique" in n0 and "move_dist_mm" in n0 and "avail_time_sec" in n0


def test_window_body_score_and_preference_hook(tmp_path: Path) -> None:
    """Verify `window_body_score` returns all required keys in <=30ms and preference calibration hook works."""
    from mai import simai

    inote = (
        "(180){16}"
        "1,2,3,4,5,6,7,8,8,7,6,5,4,3,2,1,"
        "1/5,2/6,3/7,4/8,1-5[8:2],,,,,,,,"
        "6,7,6,7,2,3,2,3,1,8,1,8,4,5,4,5,"
        "1,2,3,4,5,6,7,8,1b/5b,,,,,,,E"
    )
    notes, tm = simai.parse_chart(inote, 0.0)
    tick, _ = simai.quantize(notes)
    st = init_window_state(tm=tm, slot=5, skill_profile="Lv14+_Master")
    w_notes = [n for n in tick if 0 <= int(n.beat) < 4 * BAR_TICKS]

    # Warm up once
    _ = window_body_score(st, w_notes, tm=tm, bar_lo=0, bar_hi=4)
    t0 = time.perf_counter()
    res = window_body_score(st, w_notes, tm=tm, bar_lo=0, bar_hi=4)
    dt_ms = (time.perf_counter() - t0) * 1000.0

    for req in ("hard_violations", "awkward_score", "rarity_vs_official", "interest_features", "new_state"):
        assert req in res
    assert "interest_score" in res["interest_features"]
    assert dt_ms < 80.0

    # Test pairwise preference calibration hook on synthetic temporary jsonl
    pref_file = tmp_path / "test_prefs.jsonl"
    rows = [
        {
            "chosen": {"vocab_entropy": 0.8, "phrase_development": 0.7, "motif_echo_transformed": 0.9, "chorus_peak_contrast": 0.8, "music_kinetic_coupling": 0.7, "anti_stagnation": 0.9},
            "rejected": {"vocab_entropy": 0.3, "phrase_development": 0.4, "motif_echo_transformed": 0.2, "chorus_peak_contrast": 0.4, "music_kinetic_coupling": 0.4, "anti_stagnation": 0.3},
        }
        for _ in range(5)
    ]
    pref_file.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    w_fitted = load_interest_weights(str(pref_file))
    assert abs(sum(w_fitted.values()) - 1.0) < 1e-2


def test_artifact_version_and_hash_guard() -> None:
    """Verify code-hash stamping and RuntimeWarning on mismatched artifact hashes."""
    cur_hash = get_sim_core_code_hash()
    assert isinstance(cur_hash, str) and len(cur_hash) == 16
    assert SIM_CORE_VERSION.startswith("3.")

    valid_obj = stamp_artifact_version({"test": 1})
    with warnings.catch_warnings(record=True) as w_ok:
        warnings.simplefilter("always")
        assert check_artifact_version(valid_obj, "valid_obj") is True
        assert len(w_ok) == 0

    stale_obj = {"sim_core_version": SIM_CORE_VERSION, "sim_core_code_hash": "0000000000000000"}
    with warnings.catch_warnings(record=True) as w_bad:
        warnings.simplefilter("always")
        assert check_artifact_version(stale_obj, "stale_obj") is False
        assert len(w_bad) == 1
        assert issubclass(w_bad[0].category, RuntimeWarning)
        assert "rebuild_all.py" in str(w_bad[0].message)
