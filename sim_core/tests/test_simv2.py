"""Unit tests for `sim_core` v2 upgrades:
  1. Inner-screen jump-zone (`SLIDE_JUMP_ZONE`) detection & `innermuri` parity.
  2. Realistic held-hand fingertip + same-hand span model (`TECHNOPOLIS 2085` pattern).
  3. Choreography (`motion`) per-bar + summary + music-coupling features & `.npz` `export_trajectory`.
  4. Serializable `PlannerState`, dual-mode `window_score` (tuple unpack + dict access),
     determinism (same input -> identical output), and 4-bar window speed.
"""
from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
import pytest

from sim_core import (
    PlannerState,
    UnreasonableType,
    evaluate_chart,
    export_trajectory,
)
from sim_core.project_api import (
    BAR_TICKS,
    evaluate_notes,
    init_window_state,
    per_bar,
    window_score,
)


def test_slide_jump_zone_hard_and_clear() -> None:
    """Hold pressing a slide's final judgement area until slide end triggers hard SLIDE_JUMP_ZONE;
    releasing before slide shoot or holding an unrelated button does not.
    """
    # Slide `1-5[4:1]` finishes on `A5`. Holding `5h[2:3]` across the slide's end presses A5 early!
    simai_jump = "(160){4}5h[1:1]/1-5[4:1],,,,E"
    res_bad = evaluate_chart(simai_jump, hit_mode="hybrid", skill_profile="Lv14+_Master", frame_fps=20)
    hard_types = [e["type"] for e in res_bad.unreasonable_events]
    assert UnreasonableType.SLIDE_JUMP_ZONE.value in hard_types
    assert res_bad.is_playable is False

    # Holding `3h[1:1]` while sliding `1-5[4:1]` (`A3` is NOT on `1-5` path `A1->B1->C->B5->A5`) -> clean!
    simai_ok = "(160){4}3h[1:1]/1-5[4:1],,,,E"
    res_ok = evaluate_chart(simai_ok, hit_mode="hybrid", skill_profile="Lv14+_Master", frame_fps=20)
    all_types = [e["type"] for e in res_ok.unreasonable_events + res_ok.soft_events]
    assert UnreasonableType.SLIDE_JUMP_ZONE.value not in all_types
    assert res_ok.is_playable is True


def test_held_hand_fingertip_and_same_hand_span_no_false_collision() -> None:
    """TECHNOPOLIS 2085 pattern: holding `E5h` on left while right holds `1h` and tapping `B5`/`B4`
    uses realistic fingertip hold + same-hand span with zero false HAND_COLLISION.
    """
    simai_tech = (
        "(134){8}"
        "E5h[8:12]/B6/B7,,"
        "1h[8:3],B5,B4,"
        "2h[8:3],B5,B4,E"
    )
    res = evaluate_chart(simai_tech, hit_mode="hybrid", skill_profile="Lv14+_Master", frame_fps=20)
    coll_evs = [
        e for e in (res.unreasonable_events + res.soft_events)
        if e["type"] == UnreasonableType.HAND_COLLISION.value
    ]
    assert len(coll_evs) == 0, f"Expected 0 HAND_COLLISION on held E5h + B5/B4, got {coll_evs}"
    assert res.is_playable is True


def test_motion_features_and_export_trajectory(tmp_path: Path) -> None:
    """Verify `res.motion` structure, synthetic `mel` music coupling, and `.npz` trajectory export."""
    simai = (
        "(180){16}"
        "1,2,3,4,5,6,7,8,"
        "8,7,6,5,4,3,2,1,"
        "1/5,2/6,3/7,4/8,1-5[8:2]/5-1[8:2],,,,"
        "1,8,2,7,3,6,4,5,"
        "1,2,3,4,5,6,7,8,E"
    )
    # Synthetic 200-fps 96-dim log-mel with louder bars 2..4 than bars 0..1
    rng = np.random.default_rng(42)
    mel = rng.normal(-12.0, 1.5, size=(2400, 96)).astype(np.float32)
    mel[500:1600] += 8.0

    res = evaluate_chart(simai, hit_mode="hybrid", skill_profile="Lv14+_Master", frame_fps=25, mel=mel)
    assert "motion" in res.to_dict()
    mot = res.motion
    assert mot["fps"] >= 20.0
    assert len(mot["per_bar"]) >= 3
    summ = mot["summary"]
    for req_key in (
        "dist_per_note_mm",
        "circular_motion_ratio",
        "alternation_rate_mean",
        "sym_mirror_simul_mean",
        "sym_rotational_mean",
        "motion_novelty_mean",
        "technique_entropy",
        "fatigue_cum_peak",
    ):
        assert req_key in summ
    mc = mot["music_coupling"]
    assert mc["has_audio"] is True
    assert "corr_speed_loudness" in mc
    assert "chorus_verse_dist_ratio" in mc

    npz_file = tmp_path / "traj_test.npz"
    arrays = export_trajectory(res, npz_file)
    assert npz_file.exists()
    loaded = np.load(npz_file)
    assert loaded["left_xyz"].shape == (len(arrays["time"]), 3)
    assert loaded["right_vel_xyz"].shape == (len(arrays["time"]), 3)
    assert loaded["torso_dx_dy_yaw"].shape == (len(arrays["time"]), 3)
    assert loaded["arm_layer"].dtype == np.int8


def test_stateful_window_score_serialization_and_determinism() -> None:
    """Verify `PlannerState` JSON round-trip, stateful `window_score` 4-tuple + dict access,
    cross-window `SLIDE_JUMP_ZONE` detection, and strict determinism.
    """
    from mai import simai

    inote = (
        "(180){8}"
        "1,8,2,7,3,6,4,5,"
        "6,7,6,7,2,3,2,3,"
        "1-5[4:1],,,,2-6[4:1],,,,"
        "5h[1:2],,,,,,,,"
        "1-5[4:1],,,,2,7,3,6,E"
    )
    notes, tm = simai.parse_chart(inote, 0.0)
    tick_notes, _ = simai.quantize(notes)

    # Determinism check on evaluate_notes
    r1 = evaluate_notes(tick_notes, tm, slot=5, skill_profile="Lv14+_Master", frame_fps=20)
    r2 = evaluate_notes(tick_notes, tm, slot=5, skill_profile="Lv14+_Master", frame_fps=20)
    assert r1.to_dict() == r2.to_dict()

    # Stateful incremental window_score with JSON round-trip between windows
    st = init_window_state(tm=tm, slot=5, skill_profile="Lv14+_Master")
    w0_notes = [n for n in tick_notes if 0 <= int(n.beat) < 4 * BAR_TICKS]
    w1_notes = [n for n in tick_notes if 4 * BAR_TICKS <= int(n.beat) < 6 * BAR_TICKS]

    out0 = window_score(st, w0_notes, bar_lo=0, bar_hi=4)
    viols0, cost0, mot0, st1 = out0
    assert isinstance(st1, PlannerState)
    assert out0["cost"] == cost0
    assert "summary" in mot0

    # Serialize st1 -> JSON -> dict -> pass into window 1 (where 5h from bar 3 carries into bar 4's `1-5[4:1]`!)
    st1_dict = json.loads(json.dumps(st1.to_dict()))
    t0 = time.perf_counter()
    viols1, cost1, mot1, st2 = window_score(st1_dict, w1_notes, bar_lo=4, bar_hi=6)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    # Carried `5h[1:2]` from bar 3 must trigger SLIDE_JUMP_ZONE on bar 4's `1-5[4:1]`!
    v_types = [v["type"] for v in viols1]
    assert UnreasonableType.SLIDE_JUMP_ZONE.value in v_types
    assert elapsed_ms < 120.0  # Warm 2-4 bar window is ~15-25 ms

    # Legacy window_score signature compatibility
    legacy_w = window_score(tick_notes, tm, bar_lo=0, bar_hi=4, ctx_bars=2)
    assert "sev" in legacy_w and "n_hard" in legacy_w and "per_bar" in legacy_w


def test_neural_surrogate_prediction() -> None:
    """Verify `predict_chart_surrogate` predicts all 10 per-bar choreography & cost targets."""
    from sim_core import predict_chart_surrogate
    from sim_core.surrogate import SURROGATE_TARGETS

    simai = "(180){16}1,2,3,4,5,6,7,8,8,7,6,5,4,3,2,1,1/5,2/6,3/7,4/8,1-5[8:2],,,,E"
    pred = predict_chart_surrogate(simai)
    assert "per_bar" in pred and "summary" in pred
    assert len(pred["per_bar"]) >= 2
    for t_key in SURROGATE_TARGETS:
        assert t_key in pred["summary"]
        assert t_key in pred["per_bar"][0]
