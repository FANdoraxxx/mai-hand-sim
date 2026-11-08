# sim_core v2 Changelog (`simv2`, 2026-11-07)

## 1. Playability Gate & Biomechanical Realism
- **New `SLIDE_JUMP_ZONE` detector (`sim_core/slide_jump.py` + `sim_core/slide_areas.json`)**:
  - Integrates MaiMuriDX's 600-route slide judgement area table.
  - Flags `HOLD` and `TOUCH_HOLD` notes that remain depressed through a slide's completion (`end_time`) on the slide's final judgement area (`severity=2.5`, hard) or penultimate area (`severity=1.1`, soft).
  - Achieves **100.0% event-level recall (`76/76` events across 1,547 charts)** against `release/v6d2_echo/mai/innermuri.py` (`btn_last_end`) while extending coverage to `TOUCH_HOLD` and `V` (`GRANDV`) slides with **0.12% tune FP (`1/836`) and 0.00% held-out official FP (`0/351`)**.
- **Realistic Held-Hand Model (`kinematics.py` & `planner.py`)**:
  - Replaces the rigid 48mm flat-disc Hold collision model with a fingertip anchor (`R_HOLD_FINGERTIP_MM = 16 mm`), arched wrist (`Z_HOLD_ARCHED_PALM_MM = 36 mm`), up to `28 mm` edge shift away from an approaching hand, and same-hand thumb-to-heel span (`SAME_HAND_HOLD_SPAN_MM = 108 mm`) for tapping adjacent `Touch` sensors while holding.
  - Eliminates all 63 false `HAND_COLLISION` events on `TECHNOPOLIS 2085 [DX]#5` (`0` collisions, `is_playable=True`).
  - Incorporates MajdataPlay's `HOLD_TAIL_IGNORE_SEC` (safe early release up to `145 ms` on sustained holds `dur >= 0.25s`), eliminating artificial 20,000 mm/s tail-release `FLYING_HAND` spikes.
- **Fixed Simai Parser Multi-Arm Break-Slide Regex (`maidata_parser.py`)**:
  - Fixed `_split_chord_subtokens` regex to allow trailing `b`/`x` modifiers after `[dur]` in multi-arm slides (`4v1[8:1]b*v7[8:1]b`). Previously, `[8:1]b*` caused digits inside the second arm (`7, 8, 1`) to be misparsed as simultaneous Tap notes, accounting for 43/44 false `THREE_HAND_CONFLICT` charts on official MiLK+ charts.
- **Planner & True Per-Hand Velocity Fixes (`planner.py`)**:
  - Fixed `eff_vel` calculation that previously divided by `min(dt_global, dt_hand)` (penalizing a hand when the *other* hand struck an intermediate note), reducing raw `FLYING_HAND` chart rate on official charts from **48.4% to 1.67%**.
  - Added early-park / dynamic launch-hand selection for slides and relaxed launch-button clearance to `108 mm` (covering `OUTER_INNER_RADIAL_DIFF_MM = 95 mm` for Break star heads like `8b` on `8-4`), reducing raw `FORCED_SINGLE_HAND_JUMP` chart rate on official charts from **54.6% to 2.63%**.
  - Separated Continuous `Rub` / `Sweep` strokes and Touch-to-Button radial wipes (`E8 -> 8hx`) from discrete vertical `tap_times`.
  - Separated uncapped `raw_m` from ergonomics-capped `severity` in `UnreasonableEventRecord` + `limits.py` so extreme violations are never truncated at `3.0`.
- **Leak-Free Calibration (`scratch/simv2/splits.py`, `sim_core/limits.json`, `sim_core/reference_official.json`)**:
  - Split all 1,187 MiLK+ official charts by canonical song title into `tune` (`836` charts, strictly excluding `val_songs.txt`, `tune20.txt`, `e25.txt`, `eval_held_out.txt`) and `heldout` (`351` charts).
  - Re-promoted **`FORCED_SINGLE_HAND_JUMP`** (`heldout FP = 0.00%`), **`FLYING_HAND`** (`heldout FP = 1.42%`), **`THREE_HAND_CONFLICT`** (`heldout FP = 0.00%`), and added **`SLIDE_JUMP_ZONE`** (`heldout FP = 0.00%`) as active hard verdicts. All 8 active hard types achieve **held-out official FP $\le 1.71\%$** (overall any-hard FP: `3.11%` tune, `5.41%` held-out), while `neg` vs `champ` hard-event count jumped to **`8.8x`** (`AUC 0.6597` vs `0.530` in `simval`).

## 2. Choreography (`motion`) & Music-Coupling Engine
- **New `sim_core/motion.py` & `sim_core/audio_coupling.py`**:
  - Exports $\ge 20\text{ fps}$ continuous 3D trajectories (`export_trajectory(res, path)` -> `.npz` with 3D positions, velocities, accelerations, zones, contact flags, torso 3-DoF, 3D forearm layer order, and active techniques).
  - Computes 30+ per-bar and whole-chart choreography features (`dist_per_note_mm`, `speed_mean`, `jerk_norm`, `path_curvature`, `circular_motion_ratio`, `angular_coherence`, `alternation_rate`, `rhythm_ioi_regularity`, `cross_hand_count`, `cross_layer_ratio`, `sweep_ratio`, `split_ratio`, `rub_ratio`, `one_stroke_count`, `technique_entropy`, `torso_dx_amp`, `torso_yaw_amp`, `sym_mirror_simul`, `sym_mirror_echo`, `sym_rotational`, `motion_novelty`, `motion_exact_repeat_rate`, `motion_audio_echo`, `motion_motif_variation`, `fatigue_cum`, `fatigue_contrast`).
  - Computes music-coupling metrics from `work/mel/<sid>.npy` or `track.mp3` + `sectionpred.pkl` cache (`corr_speed_loudness`, `corr_kinetic_onset`, `onset_strike_sync`, `chorus_verse_dist_ratio`, `chorus_verse_speed_ratio`, `chorus_verse_torso_ratio`, `chorus_verse_circular_diff`, `chorus_verse_sweep_diff`, `calm_convergence`).

## 3. Performance, Serializable `PlannerState`, & Neural Surrogate
- **Numba `@njit(cache=True)` Kinematics (`kinematics.py`) & Zero-Regex Note Builder (`fast_simai.py`)**:
  - Fused scalar IK + 3D forearm capsule collision (`solve_beam_ik_step_njit`) and batch layer solver (`compute_arm_layers_batch_njit`).
  - Whole-chart warm single-core runtime: **`p50 = 0.477 s`, `p90 = 0.857 s`, `p99 = 1.398 s`** ($\le 1.5\text{ s}$ target).
- **Serializable `PlannerState` & Dual-Mode `window_score` (`planner.py`, `project_api.py`)**:
  - `window_score(history_state, window_notes) -> (violations, cost, motion_features, new_state)` runs in **`18.35 ms` mean / `15.36 ms` p50** (`11.19 ms` mean / `7.66 ms` p50 without motion; $\le 30\text{ ms}$ target) with **100.0% binary hard-violation agreement** and $r = 0.82 \sim 0.97$ motion feature agreement with whole-chart simulation.
- **Neural Surrogate (`sim_core/surrogate.py`, `sim_core/surrogate_weights.pt`)**:
  - 1D-Conv + Residual MLP predicting 10 per-bar choreography & cost metrics directly from note tokens in **`12.23 µs / window`** (`1,024` 4-bar windows in `12.52 ms` on CPU), achieving **`mean_chart_r = 0.8487`** on the 351 held-out official charts.
