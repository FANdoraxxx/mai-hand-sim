# sim_core v3 Changelog (`simv3`, 2026-11-07)

## 1. Fixes to `simv2` Acceptance Findings (Section 2)
- **Signed `yaw_accel` in `EXTREME_CROSS_ARM` (`sim_core/limits.py` & `sim_core/planner.py`)**:
  - Fixed `_RE["yaw"]` in `limits.py` to match negative signs `r"yaw_accel=(-?[\d.]+)"` with `abs()`, and updated `planner.py` to write `raw_m` directly on all `UnreasonableEventRecord` instances (`EXTREME_CROSS_ARM` `raw_m = abs(yaw_accel)`, `HAND_COLLISION` `raw_m = 85 - dist`, `ARM_COLLISION` `raw_m = 120 - f3d`) so no event relies on regex parsing of `reason` text.
- **Window Scoring Severity Agreement `r = 0.7947 -> 0.9918` (`1 - rel_L1: 75.75% -> 97.34%`) & `p90 = 20.1 ~ 26.4 ms <= 30 ms` (`sim_core/project_api.py` & `sim_core/planner.py`)**:
  - Fixed three root causes of window-vs-whole discrepancy:
    1. `project_api.window_score` previously had default `skill_profile="Lv14+_Master"`, overwriting `history_state.skill_profile` on Lv12/Lv13 charts; changed default to `None` to inherit from `history_state`.
    2. `project_api._score_stateful_window` previously overwrote `new_state.end_time = t1_sec` (bar boundary) instead of `events[-1].time`, causing `dt = t - prev_t` on the first event of the next window to collapse to `5 ms`.
    3. Unified beam width (`beam_width=6`), carried flag cooldown timestamps `last_flag_times` in `PlannerState`, and bounded forward lookahead within 4-bar phrase boundaries (`bar // 4`).
- **Held-out Official Any-Hard FP Reduced from `5.41% (19/351)` to `1.99% (7/351)` ($\le 3\%$ target)**:
  - Fixed `last_sl_l` / `last_sl_r` slide completion timestamp in `planner.py` to record `min(t, judge_time)` instead of future `end_time > t` (which previously made `t - l_ref_t < 0` -> clamped to `20 ms`, causing false `FLYING_HAND` on `Last Samurai`, `Latent Kingdom`, `命ばっかり`).
  - Fixed `FORCED_SINGLE_HAND_JACK` to track consecutive **same-key** tap streak (`left_same_key_streak` / `right_same_key_streak`) and require `is_other_locked` (eliminating false flags on `Lv12_Advanced` double-taps into chords on `なだめスかし`, `シアトリカル・ケース`, `ジャンキーナイト`, `ステップアンドライム`, `ドラゴンエネルギー`, `Rush-More`).
  - Updated `kinematics.py` `ARM_COLLISION` to allow natural wrist-arch Z-clearance (`+26 mm`) when hands are uncrossed (`lx - rx <= 110 mm`), eliminating false `ARM_COLLISION` on `Ignite Infinity`, `See The Light`, `ピポピポ`, `プナイプナイ`, `Mjölnir`.

## 2. Full-Body Biomechanical Model & Visual Burden (`sim_core/body_model.py`, Section 3)
- **6-DoF Torso + 7-DoF Arms with Swivel-Angle IK Optimization**:
  - Replaces the fixed gravity elbow plane with a 7-DoF swivel-angle ($\phi \in [-30^\circ, +60^\circ]$) IK optimizer minimizing RULA/ISO 11226 joint strain + chest clearance + 3D forearm collision.
  - Exports `BodyTimeline` (`body_timeline(chart_or_notes, frame_fps=30)`):
    - Per-frame `BodyState`: torso (`dx, dy, dz, yaw, pitch, roll, angular_vel`), left/right arms (`shoulder_flex, shoulder_abd, shoulder_rot, elbow_angle, swivel_angle, forearm_pro_sup, wrist_flex, wrist_dev, palm_heading, contact, technique, speed, accel, jerk, joint_vel, awkwardness`), bimanual (`is_crossed, layer, palm_dist, forearm_dist_3d, cross_midline_mm`), and visual (`occluded_upcoming_notes, concurrent_attention_zones, gaze_angle_deg, gaze_shift_speed_deg_s`).
    - Per-note `note_table`: `hand`, fine `technique` (`PointTap`, `Slap`, `Sweep`, `Thumb`, `PalmHeel`, `Forearm`, `OuterInnerSplit`, `AltTap`, `Hold`, `Slide`, `OneStroke`), `move_dist_mm`, `avail_time_sec`, `is_occluded`, and full body posture snapshot.

## 3. Handcam Benchmark, Body Distribution Model & `window_body_score` (`sim_core/vocab_and_rarity.py`, Sections 4, 5, 6)
- **Real Human Handcam Benchmark (`scratch/simv3/handcam/index.csv` & `kp/*.npz`)**:
  - Reconciled and deduplicated **50 real maimai DX handcam videos** (`index.csv` = 50 rows, `kp/*.npz` = 50 files, `raw/` = 0 files) across **19 distinct players** (`12`: 8, `13`: 15, `14+`: 27; `handcam_fit`: 25, `handcam_test`: 25, zero song overlap; `overhead`: 14, `oblique`: 36).
  - Upgraded keypoint mapping to use **fingertip landmarks (`0.5*(index_tip[8] + middle_tip[12])`)** + anatomical chirality verification + 2nd-order perspective/parallax calibration (raising median `<40 mm` strike contact rate from `18.0%` to `66.50%` overall, `81.47%` on overhead, `64.33%` on oblique; unclamped `audio_drift_ms` median `0.00 ms` / mean `1.70 ms` at `10 ms` frame step, visual `strike_align_err_ms` median `3.79 ms`).
  - On `handcam_test` button notes (`n=19,773`), the calibrated planner (`83.26%`, `overhead_test` `84.03%`, `oblique_test` `83.01%`) significantly outperforms both trivial baselines: `hemisphere_rule` (`79.55%`, `+3.71 pp` overall, `+5.48 pp` on overhead) and `keep_prev_hand` (`33.15%`, `+50.11 pp`).
  - Exported 50 human-verifiable note cases (`out/simv3/handcam_50_verifiable_notes.json`).
- **Sensitivity Study Fix (`sim_core/planner.py`)**:
  - Removed the hardcoded `min(beam_width, 6)` cap in `planner.py` (`eff_beam_width = max(1, int(beam_width))`) and re-evaluated `beam_width=6` vs `beam_width=12` (`20/25` charts show hand-assignment differences; median planning time `0.2990s -> 0.3899s`, `+30.4%`; feature correlations $r = 0.9391 \sim 1.0000$).
- **Official False-Positive Validation on Untouched MiLK+ Charts**:
  - Noted that `heldout` (`1.99%` FP, `7/351`) had its 19 previously-flagged charts inspected during bug diagnosis (`tune` `1.67%` FP, `14/836` is reported as the primary 12+ MASTER/Re:MASTER metric).
  - Evaluated **313 completely untouched MiLK+ official charts** (`48` MASTER/Re:MASTER `10.0 <= lv < 12.0` -> `0.00%` FP (`0/48`); `265` EXPERT `lv >= 12.0` -> `0.38%` FP (`1/265`); overall `0.32%` FP (`1/313`)).
- **16-Word Multi-Scale Action Vocabulary & Official Density Model (`sim_core/body_distribution_model.json`)**:
  - Fitted on 836 `tune` official charts across 3 beat-normalized scales (`bar`, `beat`, `0.5s`).
  - Separates `HARD_IMPOSSIBLE`, `PLAYABLE_BUT_AWKWARD` (`awkward_score`), and `RARE_INTENTIONAL_SIGNATURE` (rare but comfortable & symmetric official patterns).
- **Playability-Gated Interest Score & Pairwise Preference Hook (`window_body_score`)**:
  - Uses hand-specified heuristic prior weights `DEFAULT_INTEREST_WEIGHTS` (`0.22, 0.16, 0.18, 0.18, 0.14, 0.12`) until human pairwise preferences (`scratch/simv3/prefs/*.jsonl`) are provided.
  - Applies an explicit multiplicative playability gate by design (`neg_csft_s3000` raw un-gated score `71.19` is gated down to `46.25` vs `official_heldout` `74.06`).
  - `window_body_score` runs in **`18.90 ms` mean / `16.73 ms` p50 / `26.36 ms` p90** ($\le 30\text{ ms}$ target).
- **15-Target Neural Surrogate (`sim_core/surrogate.py`)**:
  - Re-trained on the rebuilt `v3_official.jsonl` (`sim_core_code_hash = 3aa247ca1e50a7ce`), achieving **`mean_per_bar_r = 0.6138`** and **`mean_chart_r = 0.8166`** across all 15 targets on the 351 held-out official charts (`sym_mirror_echo` `0.8255`, `cross_hand_count` `0.5963`, `awkward_score` `0.7392`).
- **Full Rebuild Pipeline & Version/Hash Guard (`sim_core/version.py`, `scratch/simv3/rebuild_all.py`, `scratch/simv3/prefs_tool/`)**:
  - Added deterministic SHA-256 code-hash stamping (`3aa247ca1e50a7ce`) and `RuntimeWarning` checks across all loaded artifacts (`limits.json`, `reference_official.json`, `body_distribution_model.json`, `surrogate_weights.pt`), plus one-click rebuild script `scratch/simv3/rebuild_all.py` and pairwise preference collection tool `scratch/simv3/prefs_tool/`.
