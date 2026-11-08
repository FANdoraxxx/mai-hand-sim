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


---

# `sim_core` 人类 maimai DX 物理游玩模拟器与谱面打分器接入指南

`sim_core` 是一个零 GPU 占用、纯 CPU 驱动的 **人类 maimai DX 生物力学游玩模拟器与谱面质量评估器（Human Play Simulator & Chart Evaluator）**。可直接作为写谱 Transformer 的下游过滤器（Rejection Sampling）、强化学习奖励模型（RL / GRPO / PPO Reward Model）、偏好对齐打分器（DPO Critic）以及 3D 可视化审谱工具。

---

## 1. 模块结构

- `sim_core/config.py`：街机毫米级物理常数（`540 mm` 内屏、`R_button = 310 mm` 外键、`R_A = 215 mm` 内屏 A 区、`R_hand = 48 mm` 圆手掌、站姿肩高与垂肘参数）、60FPS 判定窗口、打击模式枚举（`HitMode`）、无理报错枚举（`UnreasonableType`）及 4 档玩家实力配置。
- `sim_core/geometry.py`：精确提取自 `MajdataPlay` 的 33/34 内屏多边形分区（`C1/C2/C, B1~B8, E1~E8, A1~A8, D1~D8`）与 8 外键几何拓扑、圆掌-多边形面积交集、单掌/双掌蹭键覆盖求解器（带 LRU 缓存）及全类型星星滑条传感器队列（`- ^ < > v p q pp qq s z w V`）。
- `sim_core/maidata_parser.py`：完整 Simai 解析器，支持直接传入 `maidata.txt` 路径、谱面文件夹路径或内存中的内联 Simai 字符串（支持变速 `(bpm)`、变拍 `{div}`、伪双押 `` ` ``、Break/Ex/Firework、连体滑条等）。
- `sim_core/kinematics.py`：上半身 3 自由度躯干（平移 `dx`、下蹲起伏 `dy`、转腰 `yaw`）+ 左右双连杆手臂 3D 自然垂肘逆运动学（IK）+ 小臂胶囊体碰撞层级（`NORMAL` / `L_OVER_R` / `R_OVER_L`）与十字交叉锁死检测。
- `sim_core/planner.py`：前瞻束搜索（Lookahead Beam Search / Viterbi DP）双手动作规划器，内置 **防飞手第一优先级（Anti-Flying-Hand）**、单键内外拆（`OUTER_INNER_SPLIT`）、同侧交互转腰双押交替、单向顺/逆时针扫键与双押起手短扫键（`RUB_SWEEP` / `BUTTON_SWEEP`）、Touch 堆双手平推（`SINGLE_PALM_RUB`）、一笔画无缝衔接、滑条中途换手与双滑条交会顺手分流。
- `sim_core/judge.py` (`judge_engine.py`)：还原 MajdataPlay 60FPS 判定窗口、Skip-1 滑条跳区规则、尾判驻留防提前触发及 `101.0000%` DX 达成率结算。
- `sim_core/scorer.py` (`evaluator.py`)：对外主入口，提供单谱评估 `evaluate_chart`、多进程批量评估 `evaluate_charts_batch`、RL 标量奖励转换 `compute_rl_reward` 及 CLI 命令行入口。
- `sim_core/visualizer.py` & `sim_core/majplay_assets.py`：1:1 还原 MajdataPlay / MiaCode 皮肤与音效的 3D 交互审谱 GUI（支持曲库浏览、打开任意 `maidata.txt`、难度切换、4 视角切换、右键 3D 旋转、遮挡手臂 X-Ray 透视轮廓与 MP4 导出）。

---

## 2. Python API 快速接入（供写谱模型 / RL / 过滤脚本调用）

`sim_core` 顶层导入采用惰性加载 `visualizer`，在无头服务器或训练进程中导入**不会初始化 Pygame 窗口或音频设备，且完全不占用显存**。

### 2.1 单张谱面评估（支持内联 Simai 字符串或 `maidata.txt` 路径）

```python
from sim_core import evaluate_chart, compute_rl_reward

# 方式 A：直接传入模型生成的 Simai 字符串（无需写盘）
simai_text = "(180){8}1,8,2,7,3,6,4-8[8:3],,5,4,3,6,2-5[8:3],,1/5,2b/6b,E"
res = evaluate_chart(
    simai_text,
    hit_mode="hybrid",            # "hybrid" (默认内外混用) | "outer" (纯外键) | "inner" (纯内屏)
    skill_profile="Lv14+_Master", # "Lv12_Advanced" | "Lv13+_Expert" | "Lv14+_Master" | "Lv15_Apex"
    frame_fps=20,                 # 纯打分模式建议设为 15~20；需要导出 60fps 视频时设为 60
    avoid_flying_hand=True,       # 默认 True：优先使用反手/一笔画/换手避免瞬时飞手，无法避免则判无理
)

print("是否可玩 (无硬性无理):", res.is_playable)
print("综合人体工学分 (0~100):", res.total_ergonomic_score)
print("模拟 DX 达成率:", res.simulated_achievement_rate_str)
print("飞手率 (%):", res.fly_hand_rate, "飞手次数:", res.fly_hand_count)
print("RL 归一化奖励 (-1~+1):", compute_rl_reward(res))
print("无理报错列表:", res.unreasonable_events)

# 导出标准 dict 或 JSON 字符串
data_dict = res.to_dict()
json_str = res.to_json(indent=2)
```

```python
# 方式 B：评估磁盘上的 maidata.txt 文件并指定难度槽位
res_file = evaluate_chart(
    r"G:\GitHub\maiAIV3\official charts\Regulus [DX]\maidata.txt",
    difficulty=5,                 # 2=Basic, 3=Advanced, 4=Expert, 5=Master, 6=Re:Master, 或 "auto"
    hit_mode="hybrid",
    skill_profile="Lv15_Apex",
)
```

### 2.2 多进程高吞吐批量打分（用于 Rejection Sampling / GRPO / DPO 采样池）

当写谱 Transformer 一次采样生成 16 ~ 128 个候选谱面时，直接调用 `evaluate_charts_batch` 即可利用多核 CPU 并行打分（单张 1000 物量谱面单核耗时约 `0.6s`，多进程并行吞吐可达 `12 ~ 18 张/秒`）：

```python
from sim_core import evaluate_charts_batch

candidate_simai_list = [
    "(200){16}1,8,1,8,2,7,3,6,4-8[8:3],,,,E",
    "(220){16}4/5,4/5,4/5,4/5,4/5,4/5,4/5,4/5,E",  # 双押纵连废谱 -> 自动拦截 is_playable=False
]

batch_results = evaluate_charts_batch(
    candidate_simai_list,
    hit_mode="hybrid",
    skill_profile="Lv14+_Master",
    frame_fps=20,
    max_workers=8,  # 默认自动按 CPU 核心数分配
)

# 过滤掉存在物理无理的废谱，并按 rl_reward 从高到低择优录取
valid_candidates = [
    (idx, r) for idx, r in enumerate(batch_results)
    if r["is_playable"] and r["fly_hand_rate"] <= 0.5
]
valid_candidates.sort(key=lambda item: item[1]["rl_reward"], reverse=True)
```

---

## 3. 输出 JSON 字段规范 (`ChartEvaluationResult.to_dict()`)

```json
{
  "is_playable": true,
  "total_ergonomic_score": 88.1,
  "simulated_achievement_rate": 100.7542,
  "simulated_achievement_rate_str": "100.7542%",
  "fly_hand_rate": 0.0,
  "fly_hand_count": 0,
  "rl_reward": 0.9154,
  "fatigue_curve": [8.42, 12.15, 19.84],
  "fatigue_curve_detailed": [
    {"bar": 0, "left": 3.1, "right": 3.4, "torso": 1.92, "total": 8.42}
  ],
  "unreasonable_events": [
    {
      "time": 12.5,
      "bar": 6,
      "type": "FORCED_SINGLE_HAND_JUMP",
      "keys": ["A6", "A7"],
      "reason": "Forced single-hand cross-key jump (A6 -> A7): L hand at 12.00 Hz > hop limit 4.50 Hz (other hand locked on Hold (A2))",
      "severity": 2.67
    }
  ],
  "decomposition_summary": {
    "OUTER_INNER_SPLIT": 4,
    "ALTERNATING_TAP": 0,
    "SINGLE_PALM_RUB": 38,
    "RUB_SWEEP": 92,
    "BUTTON_SWEEP": 92,
    "FLY_HAND_COUNT": 0
  },
  "judge_summary": {
    "CRITICAL_PERFECT": 985,
    "FAST_PERFECT_2ND": 10,
    "LATE_PERFECT_2ND": 4,
    "MISS": 0
  },
  "metadata": {
    "title": "Regulus [DX]",
    "difficulty": "inote_5",
    "level": "14.7",
    "hit_mode": "hybrid",
    "skill_profile": "Lv15_Apex",
    "note_count": 999,
    "event_count": 662,
    "duration_sec": 129.94
  }
}
```

### 无理报错类型（`UnreasonableType`）说明

| 报错枚举值 | 物理含义与触发条件 |
| :--- | :--- |
| `DUAL_JACK_IMPOSSIBLE` | **超高 BPM 双押纵连无理**：连续 `>= 3` 组双押（非顺向双押大回转扫键）频率超过人类双手同起同落生理上限。 |
| `FORCED_SINGLE_HAND_JACK` | **被迫单手超速同键纵连无理**：单键高速纵连时另一只手被远处的 Hold / 移动中 Slide / 同时押锁死，无法过来做内外拆或双押交替。 |
| `FORCED_SINGLE_HAND_JUMP` | **被迫单手高速跨键跳打/交互无理**：另一只手被锁死，导致单手被迫高速往复跳打不同按键（如单手硬抗 16 分 `6-7-6-7` 交互）。 |
| `FLYING_HAND` | **被迫飞手（瞬时位移超限）无理**：即便穷举反手、一笔画与换手后，仍被迫在极短时间内跨屏瞬移（`dist > 250 mm` 且线速度超过手臂极限 `1.15x`）。 |
| `THREE_HAND_CONFLICT` | **伪多押/三手冲突无理**：同一时刻 `>= 3` 个独立判定点无法被两个半径 `48 mm`（内屏 Touch 为等效平推面）的手掌同时覆盖。 |
| `HAND_COLLISION` | **双掌碰撞冲突**：左右手处于同一高度层（`|z_L - z_R| < 35 mm`）且掌心间距 `< 85 mm`，无法通过内外拆错开。 |
| `EXTREME_CROSS_ARM` | **极端反手死锁/重心撕裂无理**：非顺势过渡态下左右小臂发生 X 型十字叠压锁死并伴随高频敲击，或腰腹角加速度超出人体极限。 |

---

## 4. 命令行 CLI 与 3D 审谱 GUI 使用说明

### 4.1 纯命令行输出 JSON 评分（无 GUI）

```powershell
# 输出评分 JSON 到终端
G:\GitHub\maiAIV3\.venv\Scripts\python.exe -m sim_core.scorer "G:\GitHub\maiAIV3\official charts\Regulus [DX]\maidata.txt" -d 5 -m hybrid -p Lv15_Apex

# 保存评分 JSON 到文件并导出 60FPS MP4 手元预览视频
G:\GitHub\maiAIV3\.venv\Scripts\python.exe -m sim_core.scorer "G:\GitHub\maiAIV3\official charts\PANDORA PARADOXXX\maidata.txt" -d 6 -o out\score.json --render-mp4 out\preview.mp4
```

### 4.2 启动 3D 交互式审谱 GUI

```powershell
G:\GitHub\maiAIV3\.venv\Scripts\python.exe -m sim_core.scorer "G:\GitHub\maiAIV3\official charts\Regulus [DX]\maidata.txt" -d 5 -m hybrid -p Lv15_Apex --gui
```

**GUI 快捷键与交互操作**：
- **`L` 键 / `Song Library` 按钮**：打开内置曲库浏览器（支持筛选 `Touch / [DX]` 谱面、全部官谱、AI 自定义谱面，支持直接打字搜索曲名与滚轮翻页）。
- **`O` 键 / `Open maidata.txt` 按钮**：弹出系统文件框选择任意路径下的 `maidata.txt`。
- **右侧 `BAS / ADV / EXP / MAS / ReM` 按钮**：一键切换当前歌曲的不同难度并实时重算轨迹。
- **`Space`**：播放 / 暂停（硬件音频时钟自动对齐）；**`← / →`** 或拖动底部进度条：快退 / 快进；点击右侧红色无理报错条目可直接跳转到报错时间点。
- **`M` 键**：切换 `HYBRID` / `OUTER` / `INNER` 打击模式；**`P` 键**：切换玩家实力档位。
- **`1 / 2 / 3 / 4` 键**：切换第三人称 3D 手元、第一人称 POV、3D 侧视前后冲程、2D 正视谱面视角；**鼠标右键拖拽**自由旋转 3D 视角，**滚轮**缩放。
