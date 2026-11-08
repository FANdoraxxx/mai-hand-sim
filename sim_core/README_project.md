# sim_core — Full-Body Biomechanics, Rarity & Choreography Evaluator (`simv3`)

`sim_core` parses Maimai DX charts into **full-body 3D biomechanical parameters** (6-DoF torso + 7-DoF bilateral arms with IK swivel-plane optimization + wrist/forearm rotation + visual occlusion burden), evaluates **playability & awkwardness** against official MiLK+ body-state distributions, and scores **choreographic interest**.

## 1. Whole-Chart Evaluation & `body_timeline`

```python
from mai import simai
from sim_core import (
    ERGONOMIC_JOINT_SPECS,
    body_timeline,
    evaluate_chart,
    export_trajectory,
    predict_chart_surrogate,
    window_body_score,
)
from sim_core.project_api import evaluate_notes, init_window_state, per_bar, window_score
from sim_core.calibration import calibrate

# A. Full-body timeline (>=30 fps BodyState frames + per-note hand/technique table + visual occlusion)
bt = body_timeline("official charts/INFINITE WORLD/maidata.txt", difficulty=5, frame_fps=30)
print(bt.summary)            # Awkwardness, forearm rotation range, wrist deviation, visual occlusion ratio
print(bt.note_table[0])      # Per-note hand ('L'/'R'/'LR'), technique ('PointTap','Slap','Sweep','Thumb',...), posture
print(bt.frames[0].to_dict())# 6-DoF torso, 7-DoF left/right arms, bimanual relation, visual attention/occlusion

# B. Whole-chart playability & choreography evaluation
res = evaluate_chart("official charts/INFINITE WORLD/maidata.txt", difficulty=5, frame_fps=20)
print(res.is_playable, res.total_ergonomic_score, res.unreasonable_events)
traj_arrays = export_trajectory(res, "out/simv2/infinite_world_traj.npz")
```

## 2. Incremental 4-Bar Window Scoring (`window_body_score` & `window_score`)

```python
from mai import simai
from sim_core.project_api import BAR_TICKS, init_window_state, window_body_score, window_score

meta, charts = simai.load_song("official charts/INFINITE WORLD")
notes, tm = charts[5]
tick, _ = simai.quantize(notes)
state = init_window_state(tm=tm, slot=5, skill_profile="Lv14+_Master")

cand_notes = [n for n in tick if 16 * BAR_TICKS <= int(n.beat) < 20 * BAR_TICKS]

# A. Full v3 window body scorer (~18.9 ms mean, p90 = 26.4 ms <= 30 ms single-core):
wb = window_body_score(state, cand_notes, tm=tm, bar_lo=16, bar_hi=20, lv=14.7)
# wb keys: hard_violations, soft_violations, awkward_score, rarity_vs_official,
#          unacceptable_score, interest_features, per_bar_rarity, body_summary, new_state

# B. Standard stateful window_score (Pearson r = 0.9918 vs whole-chart severity):
violations, cost, motion_features, new_state = window_score(state, cand_notes, bar_lo=16, bar_hi=20)
```

## 3. Pairwise Human Preference Calibration Hook (`scratch/simv3/prefs/*.jsonl`)

To calibrate the weights of the 6 choreography interest features (`vocab_entropy`, `phrase_development`, `motif_echo_transformed`, `chorus_peak_contrast`, `music_kinetic_coupling`, `anti_stagnation`) from human pairwise preferences, place `.jsonl` files under `scratch/simv3/prefs/*.jsonl` with one JSON object per line:

```json
{"chosen": {"vocab_entropy": 0.95, "phrase_development": 0.28, "motif_echo_transformed": 0.90, "chorus_peak_contrast": 0.82, "music_kinetic_coupling": 0.65, "anti_stagnation": 0.95}, "rejected": {"vocab_entropy": 0.88, "phrase_development": 0.18, "motif_echo_transformed": 0.78, "chorus_peak_contrast": 0.50, "music_kinetic_coupling": 0.50, "anti_stagnation": 0.85}, "weight": 1.0}
```

`sim_core.vocab_and_rarity.load_interest_weights()` automatically fits regularized Bradley-Terry logistic weights when preference files are present, and falls back to `DEFAULT_INTEREST_WEIGHTS` otherwise.

To collect human pairwise preferences interactively (with synced `track.mp3` audio, Simai text, and 30fps simulator trajectories), run:
```powershell
.\.venv\Scripts\python.exe scratch/simv3/prefs_tool/server.py --port 8765
```

## 6. Code Version & Hash Synchronization (`sim_core/version.py` & `scratch/simv3/rebuild_all.py`)

Every downstream artifact (`sim_core/limits.json`, `sim_core/reference_official.json`, `sim_core/body_distribution_model.json`, `sim_core/surrogate_weights.pt`, and `out/simv3/v3_*.jsonl`) embeds `sim_core_version` (`3.2.0`) and `sim_core_code_hash` (`get_sim_core_code_hash()`). Whenever any core planner, kinematics, or body-model source file is modified, loading a stale artifact emits a `RuntimeWarning`. Run the one-click rebuild pipeline to synchronize all downstream artifacts:
```powershell
.\.venv\Scripts\python.exe scratch/simv3/rebuild_all.py
```

**Note on `DEFAULT_INTEREST_WEIGHTS` and Playability Gating**:
- `DEFAULT_INTEREST_WEIGHTS` (`vocab_entropy=0.22`, `phrase_development=0.16`, `motif_echo_transformed=0.18`, `chorus_peak_contrast=0.18`, `music_kinetic_coupling=0.14`, `anti_stagnation=0.12`) are hand-specified prior weights used before human preference data (`scratch/simv3/prefs/*.jsonl`) is supplied.
- `interest_score` applies an explicit multiplicative playability gate (`playability_gate = max(0.05, 0.85^n_hard * (1 - 0.35*awkward))` when `is_playable=False`) to `raw_interest_score`. That broken charts (`neg_csft_s3000`) receive a low gated `interest_score` (`46.25` vs `raw_interest_score = 71.19`) is a direct consequence of this explicit gate by design, rather than something learned by the 6 raw interest sub-features alone.

## 4. Ergonomic Joint Range-of-Motion Literature Sources (`ERGONOMIC_JOINT_SPECS`)

| Parameter | Comfort Zone | Acceptable Zone | Full ROM | Literature Source |
|---|---|---|---|---|
| `torso_pitch_deg` (Trunk forward lean) | `0° .. +20°` | `-5° .. +40°` | `-15° .. +60°` | ISO 11226:2000 Sec. 3.2 |
| `torso_roll_deg` (Trunk lateral bend) | `-10° .. +10°` | `-20° .. +20°` | `-30° .. +30°` | ISO 11226:2000 Sec. 3.2 |
| `torso_yaw_deg` (Trunk axial twist) | `-15° .. +15°` | `-30° .. +30°` | `-45° .. +45°` | EN 1005-4:2005 |
| `shoulder_flex_deg` (Upper arm flexion) | `+10° .. +75°` | `-20° .. +115°` | `-45° .. +170°` | ISO 11226:2000 / RULA (1993) |
| `shoulder_abd_deg` (Upper arm abduction) | `0° .. +45°` | `-15° .. +90°` | `-35° .. +135°` | ISO 11226:2000 / RULA (1993) |
| `shoulder_rot_deg` (Glenohumeral rotation) | `-30° .. +45°` | `-60° .. +75°` | `-85° .. +90°` | NASA-STD-3000 Vol. I Sec. 3.3 |
| `elbow_angle_deg` (Interior elbow angle) | `65° .. 125°` | `35° .. 160°` | `25° .. 175°` | RULA (McAtamney & Corlett 1993) |
| `forearm_pro_sup_deg` (Pronation `+` / Sup. `-`) | `-35° .. +65°` | `-70° .. +80°` | `-85° .. +90°` | RULA / AAOS |
| `wrist_flex_deg` (Wrist flex/extension) | `-15° .. +15°` | `-40° .. +45°` | `-65° .. +70°` | RULA (McAtamney & Corlett 1993) |
| `wrist_dev_deg` (Radial `-` / Ulnar `+`) | `-10° .. +15°` | `-20° .. +25°` | `-25° .. +35°` | RULA (McAtamney & Corlett 1993) |

## 5. 15-Target Neural Surrogate (`predict_chart_surrogate` / `predict_windows_surrogate_batch`)

```python
from sim_core.surrogate import predict_chart_surrogate, predict_windows_surrogate_batch

sur = predict_chart_surrogate("official charts/INFINITE WORLD/maidata.txt", difficulty=5)
print(sur["summary"])  # 15 targets including awkward_score, rarity_vs_official, torso_pitch_mean_deg, forearm_rot_range_deg, occlusion_frame_ratio
```
