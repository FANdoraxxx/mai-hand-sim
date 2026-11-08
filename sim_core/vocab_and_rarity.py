"""Body State Distribution Model, Multi-Scale Action Vocabulary, Rarity/Awkwardness Scorer,
Interest Evaluator (with Pairwise Preference Calibration Hook), and `window_body_score` API
(`simv3` Sections 5 & 6.2).
"""
from __future__ import annotations

import bisect
import glob
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from sim_core.body_model import BodyTimeline, build_body_timeline_from_plan
from sim_core.planner import PlannerState

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
MODEL_JSON_PATH = os.path.join(_HERE, "body_distribution_model.json")
PREFS_DEFAULT_GLOB = os.path.join(_ROOT, "scratch", "simv3", "prefs", "*.jsonl")

SEGMENT_FEATURE_NAMES: Tuple[str, ...] = (
    "travel_per_beat_mm",
    "circular_motion_ratio",
    "signed_angular_vel",
    "alternation_rate",
    "cross_time_fraction",
    "sym_mirror_simul",
    "sym_mirror_echo",
    "sym_rotational",
    "sweep_share",
    "slide_share",
    "hold_anchor_share",
    "split_rub_share",
    "torso_yaw_norm",
    "torso_pitch_norm",
    "joint_awkward_p90",
    "occlusion_frame_ratio",
)

INTEREST_FEATURE_KEYS: Tuple[str, ...] = (
    "vocab_entropy",
    "phrase_development",
    "motif_echo_transformed",
    "chorus_peak_contrast",
    "music_kinetic_coupling",
    "anti_stagnation",
)

DEFAULT_INTEREST_WEIGHTS: Dict[str, float] = {
    "vocab_entropy": 0.22,
    "phrase_development": 0.16,
    "motif_echo_transformed": 0.18,
    "chorus_peak_contrast": 0.18,
    "music_kinetic_coupling": 0.14,
    "anti_stagnation": 0.12,
}

# 16 Canonical Human Maimai Choreography Archetypes (seeded & refined on `tune` official charts)
ARCHETYPE_DEFS: Tuple[Tuple[str, str], ...] = (
    ("W00_BALANCED_ALT_FLOW", "双手顺手交替散打 (Balanced Left-Right Alternating Flow)"),
    ("W01_CW_CIRCULAR_SWEEP", "顺时针圆弧扫键/流转 (Clockwise Circular Ring Sweep)"),
    ("W02_CCW_CIRCULAR_SWEEP", "逆时针圆弧扫键/流转 (Counter-Clockwise Circular Ring Sweep)"),
    ("W03_HOLD_ANCHOR_STREAM", "一手按住Hold+另一手单侧连打 (Hold-Anchored Single-Hand Stream)"),
    ("W04_MIRROR_TWIN_EXPAND", "双掌同步左右镜像展开双押 (Simultaneous Bilateral Mirror Twin Chord)"),
    ("W05_CALL_RESPONSE_ECHO", "前半小节与后半小节左右镜像问答 (Half-Bar Left-Right Mirror Call & Response)"),
    ("W06_ROTATIONAL_WINDMILL", "180度中心旋转对称大风车 (Point-Symmetric Rotational Windmill)"),
    ("W07_CROSS_BODY_TWIST", "转腰跨中线反手交叉 (Torso-Twisted Cross-Midline Reach)"),
    ("W08_SLIDE_CHOREO_GLIDE", "大幅度星星滑条跟随 (Wide-Arc Slide Tracking Glide)"),
    ("W09_ONE_STROKE_RELAY", "一笔画首尾相接滑条接力 (Continuous One-Stroke Slide Relay)"),
    ("W10_INNER_PALM_RUB_WAVE", "内屏掌根/小臂平推擦键海 (Inner-Screen Palm/Forearm Rub Wave)"),
    ("W11_OUTER_INNER_SPLIT", "同键内外拆/双押高频拆键 (High-Speed Outer-Inner Split / Trill)"),
    ("W12_UPPER_REACH_LEAN", "前倾探身上四键高区爆发 (Forward-Leaning Upper-Arc Burst)"),
    ("W13_LOWER_DIP_ANCHOR", "下潜护下四键低区稳压 (Lower-Hemisphere Dip & Anchor)"),
    ("W14_HIGH_OCCLUSION_CROSS", "高视觉遮挡交叉阅读段 (High-Occlusion Cross-Arm Reading Challenge)"),
    ("W15_CALM_BREATH_REST", "低密度安静收敛过渡段 (Calm Low-Amplitude Breathing Transition)"),
)


def extract_segment_features_multiscale(
    bt: BodyTimeline,
    motion_dict: Optional[Dict[str, Any]] = None,
    scales: Sequence[str] = ("bar", "beat", "half_sec"),
) -> Dict[str, np.ndarray]:
    """Extract 16-D BPM-normalized body segment feature matrices at 3 scales:
      - `'bar'`: 1 bar (4 beats) per segment -> `[N_bars, 16]`
      - `'beat'`: 1 beat per segment -> `[N_beats, 16]`
      - `'half_sec'`: 0.5 second windows -> `[N_win, 16]`
    """
    arrs = bt.arrays
    t_arr = arrs["time"]
    l_xyz = arrs["left_xyz"]
    r_xyz = arrs["right_xyz"]
    torso_6 = arrs["torso_6dof"]
    biman_7 = arrs["bimanual_7"]
    vis_6 = arrs["visual_6"]
    n_f = len(t_arr)

    mot_pb = {r["bar"]: r for r in (motion_dict or {}).get("per_bar", ())}
    body_pb = {r["bar"]: r for r in bt.per_bar}

    # 1. Bar scale (4 beats per segment)
    n_bars = len(bt.per_bar)
    bar_mat = np.zeros((n_bars, len(SEGMENT_FEATURE_NAMES)), dtype=np.float32)
    chart = bt.plan_result.chart if bt.plan_result is not None else None
    bar_ts = chart.bar_timestamps if (chart is not None and chart.bar_timestamps) else [(i, i * 1.5, (i + 1) * 1.5) for i in range(n_bars)]

    for i, (b_idx, t0, t1) in enumerate(bar_ts):
        if i >= n_bars:
            break
        mr = mot_pb.get(b_idx, {})
        br = body_pb.get(b_idx, {})
        f0 = int(np.searchsorted(t_arr, t0, side="left")) if n_f > 0 else 0
        f1 = int(np.searchsorted(t_arr, t1, side="right")) if n_f > 0 else 0

        dist_bar = float(mr.get("dist_total_mm", 0.0))
        if dist_bar == 0.0 and f1 - f0 >= 2:
            dist_bar = float(
                np.sum(np.hypot(np.diff(l_xyz[f0:f1, 0]), np.diff(l_xyz[f0:f1, 1])))
                + np.sum(np.hypot(np.diff(r_xyz[f0:f1, 0]), np.diff(r_xyz[f0:f1, 1])))
            )
        travel_per_beat = dist_bar / 4.0  # 4 beats per bar -> strictly BPM-invariant!

        # Signed angular velocity around center (+CW, -CCW)
        signed_ang = 0.0
        if f1 - f0 >= 2:
            for xy in (l_xyz[f0:f1, :2], r_xyz[f0:f1, :2]):
                ang = np.unwrap(np.arctan2(xy[:, 0], xy[:, 1]))
                signed_ang += float(ang[-1] - ang[0]) / (4.0 * math.pi)
        signed_ang = max(-1.5, min(1.5, signed_ang))

        bar_mat[i, 0] = travel_per_beat / 600.0
        bar_mat[i, 1] = float(mr.get("circular_motion_ratio", 0.5))
        bar_mat[i, 2] = signed_ang
        bar_mat[i, 3] = float(mr.get("alternation_rate", 0.5))
        bar_mat[i, 4] = float(br.get("cross_time_fraction", 0.0))
        bar_mat[i, 5] = float(mr.get("sym_mirror_simul", 0.2))
        bar_mat[i, 6] = float(mr.get("sym_mirror_echo", 0.2))
        bar_mat[i, 7] = float(mr.get("sym_rotational", 0.2))
        bar_mat[i, 8] = float(mr.get("sweep_ratio", 0.0))
        bar_mat[i, 9] = min(1.0, float(mr.get("slide_active_ratio", 0.0)) + 0.25 * float(mr.get("one_stroke_count", 0)))
        bar_mat[i, 10] = float(mr.get("hold_active_ratio", 0.0))
        bar_mat[i, 11] = min(1.0, float(mr.get("split_ratio", 0.0)) + float(mr.get("rub_ratio", 0.0)))
        bar_mat[i, 12] = min(1.5, float(br.get("torso_yaw_range_deg", 0.0)) / 35.0)
        bar_mat[i, 13] = min(1.5, float(br.get("torso_pitch_mean_deg", 0.0)) / 25.0)
        bar_mat[i, 14] = float(br.get("joint_awkward_p90", 0.0))
        bar_mat[i, 15] = float(br.get("occlusion_frame_ratio", 0.0))

    out: Dict[str, np.ndarray] = {"bar": bar_mat}

    def _slice_subwindows(win_bounds: List[Tuple[float, float, float]]) -> np.ndarray:
        mat = np.zeros((len(win_bounds), len(SEGMENT_FEATURE_NAMES)), dtype=np.float32)
        for wi, (w0, w1, beats_in_w) in enumerate(win_bounds):
            f0 = int(np.searchsorted(t_arr, w0, side="left")) if n_f > 0 else 0
            f1 = int(np.searchsorted(t_arr, w1, side="right")) if n_f > 0 else 0
            if f1 - f0 < 2:
                continue
            sl_l = l_xyz[f0:f1, :2]
            sl_r = r_xyz[f0:f1, :2]
            dl = np.hypot(np.diff(sl_l[:, 0]), np.diff(sl_l[:, 1]))
            dr = np.hypot(np.diff(sl_r[:, 0]), np.diff(sl_r[:, 1]))
            dist_w = float(dl.sum() + dr.sum())
            mat[wi, 0] = (dist_w / max(0.25, beats_in_w)) / 600.0
            ang_l = np.unwrap(np.arctan2(sl_l[:, 0], sl_l[:, 1]))
            ang_r = np.unwrap(np.arctan2(sl_r[:, 0], sl_r[:, 1]))
            dang = float(abs(ang_l[-1] - ang_l[0]) + abs(ang_r[-1] - ang_r[0]))
            mat[wi, 1] = min(1.0, (dang * 185.0) / max(40.0, dist_w))
            mat[wi, 2] = max(-1.5, min(1.5, float((ang_l[-1] - ang_l[0]) + (ang_r[-1] - ang_r[0])) / (2.0 * math.pi)))
            mat[wi, 4] = float(np.mean(biman_7[f0:f1, 0]))
            mat[wi, 5] = float(np.exp(-np.mean(np.hypot(sl_l[:, 0] + sl_r[:, 0], sl_l[:, 1] - sl_r[:, 1])) / 180.0))
            mat[wi, 7] = float(np.exp(-np.mean(np.hypot(sl_l[:, 0] + sl_r[:, 0], sl_l[:, 1] + sl_r[:, 1])) / 180.0))
            mat[wi, 12] = min(1.5, float(torso_6[f0:f1, 3].max() - torso_6[f0:f1, 3].min()) / 35.0)
            mat[wi, 13] = min(1.5, float(np.mean(torso_6[f0:f1, 4])) / 25.0)
            mat[wi, 14] = float(np.max(biman_7[f0:f1, 6]))
            mat[wi, 15] = float(np.mean(vis_6[f0:f1, 0] > 0))
        return mat

    if "beat" in scales:
        beat_bounds = []
        for _b_i, t0, t1 in bar_ts:
            bd = (t1 - t0) / 4.0
            for k_b in range(4):
                beat_bounds.append((t0 + k_b * bd, t0 + (k_b + 1) * bd, 1.0))
        out["beat"] = _slice_subwindows(beat_bounds)

    if "half_sec" in scales and n_f >= 2:
        t_max = float(t_arr[-1])
        hs_bounds = []
        cur_t = float(t_arr[0])
        bpm0 = float(arrs["bpm"][0]) if len(arrs["bpm"]) else 160.0
        while cur_t + 0.25 < t_max:
            hs_bounds.append((cur_t, cur_t + 0.5, 0.5 * (bpm0 / 60.0)))
            cur_t += 0.5
        out["half_sec"] = _slice_subwindows(hs_bounds)

    return out


def assign_Rule_Guided_Archetype(vec16: np.ndarray) -> int:
    """Assign a 16-D bar/segment body feature vector to one of the 16 interpretable
    choreography vocabulary words (`0..15`), or refine via fitted centroids.
    """
    travel = float(vec16[0])
    circ = float(vec16[1])
    sang = float(vec16[2])
    alt = float(vec16[3])
    cross = float(vec16[4])
    smir = float(vec16[5])
    sech = float(vec16[6])
    srot = float(vec16[7])
    swp = float(vec16[8])
    sld = float(vec16[9])
    hld = float(vec16[10])
    spl_rub = float(vec16[11])
    yaw = float(vec16[12])
    pitch = float(vec16[13])
    occ = float(vec16[15])

    if travel < 0.22 and sld < 0.15 and hld < 0.25:
        return 15  # W15_CALM_BREATH_REST
    if occ >= 0.32 and cross >= 0.20:
        return 14  # W14_HIGH_OCCLUSION_CROSS
    if swp >= 0.12 or (circ >= 0.68 and abs(sang) >= 0.22):
        return 1 if sang >= 0.0 else 2  # W01_CW / W02_CCW_CIRCULAR_SWEEP
    if sld >= 0.48:
        return 9 if travel >= 0.75 else 8  # W09_ONE_STROKE_RELAY / W08_SLIDE_CHOREO_GLIDE
    if spl_rub >= 0.22:
        return 10 if circ >= 0.52 else 11  # W10_INNER_PALM_RUB_WAVE / W11_OUTER_INNER_SPLIT
    if hld >= 0.38 and travel >= 0.30:
        return 3   # W03_HOLD_ANCHOR_STREAM
    if cross >= 0.22 or yaw >= 0.85:
        return 7   # W07_CROSS_BODY_TWIST
    if smir >= 0.36:
        return 4   # W04_MIRROR_TWIN_EXPAND
    if sech >= 0.42:
        return 5   # W05_CALL_RESPONSE_ECHO
    if srot >= 0.42:
        return 6   # W06_ROTATIONAL_WINDMILL
    if pitch >= 0.52:
        return 12  # W12_UPPER_REACH_LEAN
    if pitch <= 0.18 and travel >= 0.35:
        return 13  # W13_LOWER_DIP_ANCHOR
    return 0       # W00_BALANCED_ALT_FLOW


_DIST_MODEL_CACHE: Optional[Dict[str, Any]] = None
_INTEREST_WEIGHTS_CACHE: Optional[Dict[str, float]] = None


def load_body_distribution_model(path: str = MODEL_JSON_PATH) -> Optional[Dict[str, Any]]:
    global _DIST_MODEL_CACHE
    if _DIST_MODEL_CACHE is not None and path == MODEL_JSON_PATH:
        return _DIST_MODEL_CACHE
    if not os.path.exists(path):
        return None
    try:
        from sim_core.version import check_artifact_version
        with open(path, "r", encoding="utf-8") as f:
            m = json.load(f)
        check_artifact_version(m, "sim_core/body_distribution_model.json")
        if path == MODEL_JSON_PATH:
            _DIST_MODEL_CACHE = m
        return m
    except Exception:
        return None


def score_bar_rarity_and_awkwardness(
    bar_vecs_16: np.ndarray,
    per_bar_body: Sequence[Dict[str, Any]],
    hard_bars: set,
    lv: float = 14.0,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Evaluate per-bar 'unacceptable' breakdown:
      - `rarity_vs_official` (`0..1` ECDF percentile on `tune` official charts + Mahalanobis distance)
      - `awkward_score` (`0..1` continuous RULA/ISO joint + cross + occlusion strain)
      - `category`: `'OK'` | `'HARD_IMPOSSIBLE'` | `'PLAYABLE_BUT_AWKWARD'` | `'RARE_INTENTIONAL_SIGNATURE'`
      - `unacceptable_score`: combined score separating impossible & awkward bars from intentional choreography!
    """
    model = load_body_distribution_model()
    bkt = "12" if lv < 13.0 else ("13" if lv < 14.0 else "14+")
    n_bars = len(bar_vecs_16)
    if n_bars == 0:
        return [], {"rarity_mean": 0.0, "rarity_p95": 0.0, "awkward_mean": 0.0, "unacceptable_mean": 0.0, "rare_awkward_bars": 0, "rare_signature_bars": 0}

    if model is not None and bkt in model.get("bands", {}):
        b_mod = model["bands"][bkt]
        mu = np.asarray(b_mod["mean"], dtype=np.float32)
        inv_cov = np.asarray(b_mod["inv_cov"], dtype=np.float32)
        centroids = np.asarray(model["vocab_centroids"], dtype=np.float32)
        ecdf_xs = b_mod["mahal_ecdf"]
    else:
        mu = np.zeros(16, dtype=np.float32)
        inv_cov = np.eye(16, dtype=np.float32)
        centroids = None
        ecdf_xs = [float(x) for x in np.linspace(1.0, 8.0, 100)]

    rows_out: List[Dict[str, Any]] = []
    word_seq: List[int] = []
    for i in range(n_bars):
        v = bar_vecs_16[i]
        br = per_bar_body[i] if i < len(per_bar_body) else {}
        b_idx = int(br.get("bar", i))
        n_notes = int(br.get("n_notes", 0))

        if n_notes == 0:
            w_id = 15
            mahal = 0.0
            rarity_pct = 0.0
        else:
            diff = v - mu
            mahal = math.sqrt(max(0.0, float(diff @ inv_cov @ diff)))
            if centroids is not None and len(centroids) == 16:
                # Blend rule-guided archetype with nearest fitted centroid distance
                w_id = assign_Rule_Guided_Archetype(v)
                c_dist = float(np.linalg.norm(v - centroids[w_id]))
                score_raw = 0.70 * mahal + 0.30 * c_dist
            else:
                w_id = assign_Rule_Guided_Archetype(v)
                score_raw = mahal
            rarity_pct = bisect.bisect_right(ecdf_xs, score_raw) / max(1, len(ecdf_xs))

        word_seq.append(w_id)
        awk = float(br.get("awkward_score", 0.0))
        j_awk = float(br.get("joint_awkward_p90", 0.0))
        sym_max = max(float(v[5]), float(v[6]), float(v[7]), float(v[1]) - 0.15)
        is_hard = b_idx in hard_bars

        if is_hard:
            cat = "HARD_IMPOSSIBLE"
            unacc = min(1.0, 0.75 + 0.25 * max(awk, rarity_pct))
        elif rarity_pct >= 0.96 and j_awk < 0.34 and awk < 0.36 and sym_max >= 0.38:
            # Section 5.2 requirement: rare but well-structured symmetric/circular official signature pattern!
            cat = "RARE_INTENTIONAL_SIGNATURE"
            unacc = round(0.18 * awk, 4)
        elif awk >= 0.44 or (rarity_pct >= 0.965 and awk >= 0.34):
            cat = "PLAYABLE_BUT_AWKWARD"
            unacc = min(1.0, 0.55 * awk + 0.45 * max(0.0, rarity_pct - 0.5))
        else:
            cat = "OK"
            unacc = round(0.45 * awk + 0.20 * max(0.0, rarity_pct - 0.75), 4)

        rows_out.append(
            {
                "bar": b_idx,
                "word_id": int(w_id),
                "word_code": ARCHETYPE_DEFS[w_id][0],
                "mahalanobis_d": round(mahal, 3),
                "rarity_vs_official": round(rarity_pct, 4),
                "awkward_score": round(awk, 4),
                "unacceptable_score": round(unacc, 4),
                "category": cat,
            }
        )

    active_r = [r for i, r in enumerate(rows_out) if (i < len(per_bar_body) and per_bar_body[i].get("n_notes", 0) > 0)] or rows_out
    summ = {
        "rarity_mean": round(float(np.mean([r["rarity_vs_official"] for r in active_r])), 4),
        "rarity_p95": round(float(np.percentile([r["rarity_vs_official"] for r in active_r], 95)), 4),
        "mahalanobis_p95": round(float(np.percentile([r["mahalanobis_d"] for r in active_r], 95)), 3),
        "awkward_mean": round(float(np.mean([r["awkward_score"] for r in active_r])), 4),
        "awkward_p90": round(float(np.percentile([r["awkward_score"] for r in active_r], 90)), 4),
        "unacceptable_mean": round(float(np.mean([r["unacceptable_score"] for r in active_r])), 4),
        "unacceptable_p95": round(float(np.percentile([r["unacceptable_score"] for r in active_r], 95)), 4),
        "rare_awkward_bars": sum(1 for r in active_r if r["category"] == "PLAYABLE_BUT_AWKWARD"),
        "rare_signature_bars": sum(1 for r in active_r if r["category"] == "RARE_INTENTIONAL_SIGNATURE"),
        "word_sequence": word_seq,
    }
    return rows_out, summ


def load_interest_weights(prefs_glob: str = PREFS_DEFAULT_GLOB) -> Dict[str, float]:
    """Load human pairwise preference weights from `scratch/simv3/prefs/*.jsonl` if present;
    otherwise return `DEFAULT_INTEREST_WEIGHTS`. Never fabricates preference data.
    """
    global _INTEREST_WEIGHTS_CACHE
    files = sorted(glob.glob(prefs_glob))
    if not files:
        return dict(DEFAULT_INTEREST_WEIGHTS)
    if _INTEREST_WEIGHTS_CACHE is not None and prefs_glob == PREFS_DEFAULT_GLOB:
        return dict(_INTEREST_WEIGHTS_CACHE)

    diffs_x = []
    sample_w = []
    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    ch = rec.get("chosen")
                    rj = rec.get("rejected")
                    if isinstance(ch, dict) and isinstance(rj, dict):
                        dx = [float(ch.get(k, 0.0)) - float(rj.get(k, 0.0)) for k in INTEREST_FEATURE_KEYS]
                        diffs_x.append(dx)
                        sample_w.append(float(rec.get("weight", 1.0)))
        except Exception:
            pass
    if len(diffs_x) < 3:
        return dict(DEFAULT_INTEREST_WEIGHTS)

    # Fit regularized Bradley-Terry / logistic weights w > 0 maximizing sum w_i * log(sigmoid(w^T dx))
    X_d = np.asarray(diffs_x, dtype=np.float64)
    sw = np.asarray(sample_w, dtype=np.float64)
    w_vec = np.array([DEFAULT_INTEREST_WEIGHTS[k] for k in INTEREST_FEATURE_KEYS], dtype=np.float64)
    for _ in range(60):
        z = np.clip(X_d @ w_vec, -15.0, 15.0)
        prob = 1.0 / (1.0 + np.exp(-z))
        grad = ((sw * (1.0 - prob))[:, None] * X_d).mean(axis=0) - 0.5 * (w_vec - np.array(list(DEFAULT_INTEREST_WEIGHTS.values())))
        w_vec = np.maximum(0.02, w_vec + 0.15 * grad)
        w_vec /= w_vec.sum()
    out_w = {k: round(float(w_vec[i]), 4) for i, k in enumerate(INTEREST_FEATURE_KEYS)}
    if prefs_glob == PREFS_DEFAULT_GLOB:
        _INTEREST_WEIGHTS_CACHE = out_w
    return out_w


def compute_interest_features(
    bar_vecs_16: np.ndarray,
    rarity_rows: Sequence[Dict[str, Any]],
    motion_dict: Optional[Dict[str, Any]],
    is_playable: bool,
    n_hard_violations: int,
    awkward_mean: float,
    prefs_glob: str = PREFS_DEFAULT_GLOB,
) -> Dict[str, Any]:
    """Compute the 6 choreography interest dimensions (`simv3` Section 5.3) and playability-gated
    `interest_score` (ensuring broken/muri charts like `neg` can never game the interest score).
    """
    active_words = [r["word_id"] for i, r in enumerate(rarity_rows) if r["word_id"] != 15]
    if not active_words and rarity_rows:
        active_words = [r["word_id"] for r in rarity_rows]

    # 1. Action Vocabulary Diversity (normalized Shannon entropy over 15 active words)
    if active_words:
        cnts = np.bincount(active_words, minlength=16).astype(np.float64)
        p = cnts[cnts > 0] / len(active_words)
        vocab_entropy = float(-np.sum(p * np.log(p)) / math.log(10.0))
        vocab_entropy = min(1.0, vocab_entropy)
        # 6. Anti-Stagnation (penalizes staying stuck in the exact same action word for many bars)
        max_run = 1
        cur_run = 1
        for i in range(1, len(active_words)):
            if active_words[i] == active_words[i - 1]:
                cur_run += 1
                if cur_run > max_run:
                    max_run = cur_run
            else:
                cur_run = 1
        anti_stagnation = max(0.0, 1.0 - max(0, max_run - 2) / max(4.0, 0.45 * len(active_words)))
    else:
        vocab_entropy = 0.0
        anti_stagnation = 0.0

    # 2. Intra-phrase development (gradual expansion / variation across 4-bar phrases)
    dev_vals = []
    for b0 in range(0, len(bar_vecs_16) - 3, 4):
        blk = bar_vecs_16[b0 : b0 + 4]
        if np.mean(blk[:, 0]) > 0.08:
            # Growth in movement amplitude + circular/symmetry richness + step variation within phrase
            step_diffs = np.linalg.norm(np.diff(blk, axis=0), axis=1)
            # Reward moderate organic variation (0.25..0.65), penalize flat copy (0.0) or chaotic jump (>1.2)
            mean_d = float(np.mean(step_diffs))
            org_var = math.exp(-((mean_d - 0.42) ** 2) / 0.12)
            amp_trend = max(0.0, min(1.0, 0.5 + (float(blk[-1, 0]) - float(blk[0, 0])) * 0.8))
            dev_vals.append(0.65 * org_var + 0.35 * amp_trend)
    phrase_development = float(np.mean(dev_vals)) if dev_vals else 0.50

    # 3. Motif Echo & Transformation (mirror / rotational echo + audio-matched motif callback)
    ms = (motion_dict or {}).get("summary", {})
    mc = (motion_dict or {}).get("music_coupling", {})
    sym_echo = float(ms.get("sym_mirror_echo_mean", 0.30))
    sym_simul = float(ms.get("sym_mirror_simul_mean", 0.22))
    sym_rot = float(ms.get("sym_rotational_mean", 0.30))
    aud_echo = float(ms.get("motion_motif_variation", 0.0)) or float(ms.get("motion_audio_echo_mean", 0.85))
    circ_flow = float(ms.get("circular_motion_ratio", 0.55))
    motif_echo_transformed = min(1.0, 0.35 * (sym_echo / 0.38) + 0.25 * (sym_simul / 0.28) + 0.20 * (sym_rot / 0.36) + 0.20 * (circ_flow / 0.62))

    # 4. Chorus Peak Contrast (climax amplitude ratio + stamina contrast)
    cv_ratio = float(mc.get("chorus_verse_dist_ratio", 1.0)) if mc.get("has_audio") else 1.25
    fat_cont = float(ms.get("fatigue_contrast", 0.40))
    chorus_peak_contrast = min(1.0, 0.55 * min(1.0, max(0.0, (cv_ratio - 0.90) / 0.70)) + 0.45 * min(1.0, fat_cont / 0.55))

    # 5. Music Kinetic Coupling (loudness correlation + strike-onset sync + calm convergence)
    if mc.get("has_audio"):
        corr_sl = max(0.0, float(mc.get("corr_speed_loudness", 0.0)))
        sync_on = min(1.0, max(0.0, (float(mc.get("onset_strike_sync", 1.0)) - 1.0) / 2.2))
        calm_c = min(1.0, max(0.0, float(mc.get("calm_convergence", 0.5))))
        music_kinetic_coupling = 0.45 * corr_sl + 0.30 * sync_on + 0.25 * calm_c
    else:
        music_kinetic_coupling = min(1.0, 0.5 * (circ_flow / 0.60) + 0.5 * float(ms.get("rhythm_ioi_regularity", 0.8)))

    feats_raw = {
        "vocab_entropy": round(vocab_entropy, 4),
        "phrase_development": round(phrase_development, 4),
        "motif_echo_transformed": round(motif_echo_transformed, 4),
        "chorus_peak_contrast": round(chorus_peak_contrast, 4),
        "music_kinetic_coupling": round(music_kinetic_coupling, 4),
        "anti_stagnation": round(anti_stagnation, 4),
    }
    weights = load_interest_weights(prefs_glob)
    raw_weighted = sum(weights.get(k, 0.0) * feats_raw[k] for k in INTEREST_FEATURE_KEYS) * 100.0

    # Section 2.5 & 5.3 Mandatory Playability Gate:
    # "Style != Quality": A chart with hard violations or severe awkwardness cannot achieve a high interest score!
    playability_gate = math.exp(-0.55 * n_hard_violations - 1.8 * max(0.0, awkward_mean - 0.34))
    gated_score = raw_weighted * playability_gate

    return {
        **feats_raw,
        "raw_interest_score": round(raw_weighted, 2),
        "playability_gate_multiplier": round(playability_gate, 4),
        "interest_score": round(gated_score, 2),
        "playable_qualified": bool(is_playable),
    }


def window_body_score(
    history_state: Optional[Union[PlannerState, Dict[str, Any]]],
    window_notes: Sequence,
    tm: Any = None,
    bar_lo: Optional[int] = None,
    bar_hi: Optional[int] = None,
    slot: Optional[int] = None,
    hit_mode: Optional[str] = None,
    skill_profile: Optional[str] = None,
    lv: float = 14.0,
) -> Dict[str, Any]:
    """Public API (`simv3` Section 6.2): evaluate a candidate 4-bar window in <=30 ms single-core,
    returning:
      - `hard_violations`: list of hard violations
      - `awkward_score`: playable-but-awkward score [0..1]
      - `rarity_vs_official`: rarity percentile [0..1] vs `tune` official distribution
      - `interest_features`: dict of the 6 choreography interest features + `interest_score`
      - `new_state`: updated serializable `PlannerState`
    """
    from sim_core.project_api import _score_stateful_window, BAR_TICKS, _TM, init_window_state
    from sim_core.fast_simai import build_parsed_chart_from_notes
    from sim_core.planner import plan_chart_movements
    from sim_core.limits import split_events
    from sim_core.motion import compute_motion_features

    if isinstance(history_state, dict):
        st = PlannerState.from_dict(history_state)
    elif isinstance(history_state, PlannerState):
        st = history_state
    else:
        st = init_window_state(tm=tm, slot=slot or 5, hit_mode=hit_mode or "hybrid", skill_profile=skill_profile or "Lv14+_Master")

    eff_slot = slot if slot is not None else st.slot
    eff_mode = hit_mode if hit_mode is not None else st.hit_mode
    eff_prof = skill_profile if skill_profile is not None else st.skill_profile
    if tm is None:
        tm = _TM(st.tm_changes, first=st.first_offset_sec)

    if bar_lo is None:
        bar_lo = (min(int(n.beat) for n in window_notes) // BAR_TICKS) if window_notes else st.end_bar
    if bar_hi is None:
        bar_hi = ((max(int(n.beat) for n in window_notes) // BAR_TICKS) + 1) if window_notes else (bar_lo + 4)

    t0_sec = float(tm.to_sec(4.0 * bar_lo))
    t1_sec = float(tm.to_sec(4.0 * bar_hi))
    w_notes = [n for n in window_notes if bar_lo * BAR_TICKS <= int(n.beat) < bar_hi * BAR_TICKS]
    pc = build_parsed_chart_from_notes(w_notes, tm, slot=eff_slot, title="WindowBodyChart", unit_ticks=True)
    pc.bar_timestamps = [
        (b, float(tm.to_sec(4.0 * b)), float(tm.to_sec(4.0 * (b + 1))))
        for b in range(bar_lo, bar_hi)
    ]

    plan = plan_chart_movements(
        chart=pc,
        hit_mode=eff_mode,
        skill_profile=eff_prof,
        beam_width=6,
        frame_fps=30,
        initial_state=st,
        t_range=(t0_sec, t1_sec),
    )

    raw_evs = [u.to_dict() for u in plan.unreasonable_events if bar_lo <= int(u.bar) < bar_hi]
    hard_evs, soft_evs = split_events(raw_evs)
    hard_bars = {int(e.get("bar", -1)) for e in hard_evs}

    bt = build_body_timeline_from_plan(plan, materialize_frame_dicts=False)
    motion_feats, upd_sigs, upd_fat = compute_motion_features(
        plan=plan,
        slot=eff_slot,
        prior_bar_sigs=st.prior_bar_sigs,
        initial_cum_fatigue=st.cum_fatigue,
    )
    seg_mats = extract_segment_features_multiscale(bt, motion_dict=motion_feats, scales=("bar",))
    bar_vecs = seg_mats["bar"]
    rarity_rows, rarity_summ = score_bar_rarity_and_awkwardness(bar_vecs, bt.per_bar, hard_bars, lv=lv)
    interest_dict = compute_interest_features(
        bar_vecs,
        rarity_rows,
        motion_feats,
        is_playable=(len(hard_evs) == 0),
        n_hard_violations=len(hard_evs),
        awkward_mean=rarity_summ["awkward_mean"],
    )

    new_state = plan.final_state
    assert new_state is not None
    new_state.end_bar = bar_hi
    new_state.prior_bar_sigs = upd_sigs
    new_state.cum_fatigue = upd_fat

    return {
        "hard_violations": hard_evs,
        "soft_violations": soft_evs,
        "awkward_score": rarity_summ["awkward_mean"],
        "rarity_vs_official": rarity_summ["rarity_mean"],
        "rarity_p95": rarity_summ["rarity_p95"],
        "unacceptable_score": rarity_summ["unacceptable_mean"],
        "interest_features": interest_dict,
        "per_bar_rarity": rarity_rows,
        "body_summary": bt.summary,
        "new_state": new_state,
    }
