"""Neural Surrogate Model (`ChoreoSurrogateModel`) for ultra-fast batched prediction
of per-bar choreography features and playability cost directly from chart notes/tokens.

Trained strictly on `tune`-split MiLK+ official charts (`sim_core/surrogate_weights.pt`).
Can evaluate 1,000+ candidate 4-bar windows in ~1.5 ms on CPU or GPU (`cuda:1` fp32).
"""
from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from sim_core.maidata_parser import NoteType, ParsedChart, parse_simai_chart

SURROGATE_TARGETS: Tuple[str, ...] = (
    "dist_total_mm",
    "speed_mean",
    "circular_motion_ratio",
    "alternation_rate",
    "cross_hand_count",
    "sweep_ratio",
    "torso_yaw_amp",
    "sym_mirror_simul",
    "sym_mirror_echo",
    "bar_cost",
    "awkward_score",
    "rarity_vs_official",
    "torso_pitch_mean_deg",
    "forearm_rot_range_deg",
    "occlusion_frame_ratio",
)

N_BAR_FEATS: int = 32
WEIGHTS_PATH: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "surrogate_weights.pt")

# Precomputed 2D coordinates (x, y) on R=215mm ring for buttons 1..8
_BTN_XY = {
    b: (
        215.0 * math.sin(math.radians(22.5 + (b - 1) * 45.0)),
        215.0 * math.cos(math.radians(22.5 + (b - 1) * 45.0)),
    )
    for b in range(1, 9)
}


def extract_chart_bar_descriptors(chart: ParsedChart) -> np.ndarray:
    """Extract `[N_bars, 32]` geometric & rhythmic note-token descriptors per bar in <0.15 ms."""
    bar_ts = chart.bar_timestamps
    n_bars = len(bar_ts)
    if n_bars == 0:
        return np.zeros((0, N_BAR_FEATS), dtype=np.float32)

    feats = np.zeros((n_bars, N_BAR_FEATS), dtype=np.float32)
    notes_by_b: Dict[int, List] = {}
    for n in chart.notes:
        if 0 <= n.bar < n_bars:
            notes_by_b.setdefault(n.bar, []).append(n)
    events_by_b: Dict[int, List] = {}
    for ev in chart.events:
        if 0 <= ev.bar < n_bars:
            events_by_b.setdefault(ev.bar, []).append(ev)

    for b_idx, (_b, t0, t1) in enumerate(bar_ts):
        b_dur = max(0.25, t1 - t0)
        bpm = 240.0 / b_dur
        ns = notes_by_b.get(b_idx, [])
        evs = events_by_b.get(b_idx, [])

        n_tap = sum(1 for x in ns if x.note_type == NoteType.TAP and not x.is_star_head)
        n_star = sum(1 for x in ns if x.is_star_head)
        n_hold = sum(1 for x in ns if x.note_type in (NoteType.HOLD, NoteType.TOUCH_HOLD))
        n_slide = sum(1 for x in ns if x.note_type == NoteType.SLIDE)
        n_touch = sum(1 for x in ns if x.note_type == NoteType.TOUCH)
        n_brk = sum(1 for x in ns if x.is_break)
        n_ex = sum(1 for x in ns if x.is_ex)
        n_strk = n_tap + n_star + n_hold + n_touch
        n_ev = len(evs)

        feats[b_idx, 0] = n_strk / 16.0
        feats[b_idx, 1] = n_ev / 16.0
        feats[b_idx, 2] = bpm / 200.0
        feats[b_idx, 3] = b_dur / 2.0
        feats[b_idx, 4] = n_tap / max(1, n_strk)
        feats[b_idx, 5] = n_hold / max(1, n_strk)
        feats[b_idx, 6] = n_slide / max(1, n_strk)
        feats[b_idx, 7] = n_touch / max(1, n_strk)
        feats[b_idx, 8] = n_brk / max(1, n_strk)
        feats[b_idx, 9] = n_ex / max(1, n_strk)

        if n_ev == 0:
            continue

        # Spatial distribution on 1..8
        btn_counts = np.zeros(8, dtype=np.float32)
        hold_dur_sum = 0.0
        slide_dur_sum = 0.0
        slide_arc_sum = 0.0
        for x in ns:
            if x.button is not None and 1 <= x.button <= 8 and x.note_type != NoteType.SLIDE:
                btn_counts[x.button - 1] += 1.0
            if x.note_type in (NoteType.HOLD, NoteType.TOUCH_HOLD):
                hold_dur_sum += x.duration
            elif x.note_type == NoteType.SLIDE:
                slide_dur_sum += x.duration
                wps = x.slide_waypoints_mm
                if len(wps) >= 2:
                    slide_arc_sum += sum(
                        math.hypot(wps[k][0] - wps[k - 1][0], wps[k][1] - wps[k - 1][1])
                        for k in range(1, len(wps))
                    )

        tot_b = max(1.0, float(btn_counts.sum()))
        r_frac = float(btn_counts[:4].sum()) / tot_b
        l_frac = float(btn_counts[4:].sum()) / tot_b
        up_frac = float(btn_counts[[0, 1, 6, 7]].sum()) / tot_b
        feats[b_idx, 10] = abs(r_frac - l_frac)
        feats[b_idx, 11] = up_frac
        feats[b_idx, 12] = hold_dur_sum / b_dur
        feats[b_idx, 13] = slide_dur_sum / b_dur
        feats[b_idx, 14] = slide_arc_sum / 1000.0

        # Event-by-event geometric transitions & chords
        n_chords = 0
        n_mirror_chords = 0
        n_rot_chords = 0
        ev_centroids: List[Tuple[float, float]] = []
        ev_btns: List[int] = []
        ev_times: List[float] = []
        h1_btns = np.zeros(8, dtype=np.float32)
        h2_btns = np.zeros(8, dtype=np.float32)
        t_mid = 0.5 * (t0 + t1)

        for ev in evs:
            sn_list = ev.strike_notes
            if not sn_list:
                continue
            ev_times.append(ev.time)
            btns = [x.button for x in sn_list if x.button is not None]
            for b_val in btns:
                if ev.time < t_mid:
                    h1_btns[b_val - 1] += 1.0
                else:
                    h2_btns[b_val - 1] += 1.0
            if len(sn_list) >= 2:
                n_chords += 1
                if len(btns) == 2:
                    if btns[0] + btns[1] == 9:
                        n_mirror_chords += 1
                    if abs(btns[0] - btns[1]) == 4:
                        n_rot_chords += 1
            if btns:
                ev_btns.append(btns[0])
                cx = sum(_BTN_XY[b][0] for b in btns) / len(btns)
                cy = sum(_BTN_XY[b][1] for b in btns) / len(btns)
                ev_centroids.append((cx, cy))
            else:
                ev_centroids.append((0.0, 0.0))

        n_active_ev = max(1, len(ev_times))
        feats[b_idx, 15] = n_chords / n_active_ev
        feats[b_idx, 16] = n_mirror_chords / max(1, n_chords)
        feats[b_idx, 17] = n_rot_chords / max(1, n_chords)

        # Half-bar mirror correlation on button histogram (1..8 vs 8..1)
        h2_mir = h2_btns[::-1]
        n1 = float(np.linalg.norm(h1_btns))
        n2 = float(np.linalg.norm(h2_mir))
        feats[b_idx, 18] = float(np.dot(h1_btns, h2_mir) / (n1 * n2)) if (n1 > 0 and n2 > 0) else 0.0

        if len(ev_times) >= 2:
            iois = np.maximum(1e-3, np.diff(ev_times))
            feats[b_idx, 19] = float(np.mean(iois)) / 0.25
            feats[b_idx, 20] = float(np.min(iois)) / 0.15
            feats[b_idx, 21] = float(np.std(iois) / np.mean(iois))
            feats[b_idx, 22] = float(np.mean(iois <= 0.095))
            feats[b_idx, 23] = float(np.mean(iois <= 0.055))

            dists = [
                math.hypot(ev_centroids[k][0] - ev_centroids[k - 1][0], ev_centroids[k][1] - ev_centroids[k - 1][1])
                for k in range(1, len(ev_centroids))
            ]
            feats[b_idx, 24] = float(np.mean(dists)) / 300.0
            feats[b_idx, 25] = float(np.max(dists)) / 430.0
            feats[b_idx, 26] = float(np.sum(dists)) / 2000.0

        if len(ev_btns) >= 2:
            steps_signed = [((ev_btns[k] - ev_btns[k - 1] + 4) % 8) - 4 for k in range(1, len(ev_btns))]
            abs_steps = [abs(s) for s in steps_signed]
            n_tr = len(steps_signed)
            feats[b_idx, 27] = sum(1 for s in abs_steps if s == 0) / n_tr
            feats[b_idx, 28] = sum(1 for s in abs_steps if s == 1) / n_tr
            feats[b_idx, 29] = sum(1 for s in abs_steps if s >= 3) / n_tr
            if n_tr >= 2:
                mono_adj = sum(
                    1 for k in range(1, n_tr)
                    if steps_signed[k] == steps_signed[k - 1] and abs_steps[k] == 1
                ) / (n_tr - 1)
                rev_adj = sum(
                    1 for k in range(1, n_tr)
                    if steps_signed[k] == -steps_signed[k - 1] and abs_steps[k] in (1, 2)
                ) / (n_tr - 1)
                feats[b_idx, 30] = mono_adj
                feats[b_idx, 31] = rev_adj

    return feats


def build_surrogate_torch_model(in_dim: int = N_BAR_FEATS, hidden: int = 128, out_dim: int = len(SURROGATE_TARGETS)):
    import torch
    import torch.nn as nn

    class ChoreoSurrogateNet(nn.Module):
        """1D Convolutional + Residual MLP predicting per-bar choreography & playability."""
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = nn.Conv1d(in_dim, hidden, kernel_size=3, padding=1)
            self.act1 = nn.GELU()
            self.conv2 = nn.Conv1d(hidden, hidden, kernel_size=3, padding=1)
            self.norm = nn.LayerNorm(hidden)
            self.mlp = nn.Sequential(
                nn.Linear(hidden + in_dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Linear(hidden, out_dim),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: [B, T, D]
            xt = x.transpose(1, 2)
            h = self.act1(self.conv1(xt))
            h = h + self.act1(self.conv2(h))
            h = self.norm(h.transpose(1, 2))
            return self.mlp(torch.cat([h, x], dim=-1))

    return ChoreoSurrogateNet()


_CACHED_SURROGATE: Optional[Dict[str, Any]] = None


def predict_chart_surrogate(
    chart_or_source: str | ParsedChart,
    difficulty: str | int = "auto",
    device: str = "cpu",
    weights_path: str = WEIGHTS_PATH,
) -> Dict[str, Any]:
    """Predict per-bar choreography & playability features directly from chart notes via the trained surrogate."""
    import torch
    global _CACHED_SURROGATE
    if _CACHED_SURROGATE is None or _CACHED_SURROGATE.get("path") != weights_path:
        from sim_core.version import check_artifact_version
        ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
        check_artifact_version(ckpt, "sim_core/surrogate_weights.pt")
        net = build_surrogate_torch_model(ckpt["in_dim"], ckpt["hidden"], ckpt["out_dim"])
        net.load_state_dict(ckpt["state_dict"])
        net.eval()
        _CACHED_SURROGATE = {
            "path": weights_path,
            "net": net,
            "x_mean": np.asarray(ckpt["x_mean"], dtype=np.float32),
            "x_std": np.asarray(ckpt["x_std"], dtype=np.float32),
            "y_mean": np.asarray(ckpt["y_mean"], dtype=np.float32),
            "y_std": np.asarray(ckpt["y_std"], dtype=np.float32),
            "targets": tuple(ckpt["targets"]),
        }
    info = _CACHED_SURROGATE
    chart = chart_or_source if isinstance(chart_or_source, ParsedChart) else parse_simai_chart(chart_or_source, difficulty=difficulty)
    x_raw = extract_chart_bar_descriptors(chart)
    if len(x_raw) == 0:
        return {"per_bar": [], "summary": {}}
    x_norm = (x_raw - info["x_mean"]) / info["x_std"]
    net = info["net"].to(device)
    with torch.no_grad():
        xt = torch.from_numpy(x_norm[None, :, :]).to(device, dtype=torch.float32)
        pred_norm = net(xt)[0].cpu().numpy()
    pred = pred_norm * info["y_std"] + info["y_mean"]
    t_names = info["targets"]
    per_bar = [
        {"bar": b_i, **{t_names[j]: round(float(pred[b_i, j]), 4) for j in range(len(t_names))}}
        for b_i in range(len(pred))
    ]
    summary = {
        t_names[j]: round(float(np.mean(pred[:, j])), 4) for j in range(len(t_names))
    }
    return {"per_bar": per_bar, "summary": summary}


def predict_windows_surrogate_batch(
    x_windows: np.ndarray,
    device: str = "cpu",
    weights_path: str = WEIGHTS_PATH,
) -> Tuple[np.ndarray, Tuple[str, ...]]:
    """Batch-predict `[B, T, N_TARGETS]` on CPU or GPU (`cuda:1` fp32) from `[B, T, 32]` descriptors."""
    import torch
    global _CACHED_SURROGATE
    if _CACHED_SURROGATE is None or _CACHED_SURROGATE.get("path") != weights_path:
        predict_chart_surrogate("(160){4}1,2,3,4,E", device="cpu", weights_path=weights_path)
    info = _CACHED_SURROGATE
    assert info is not None
    x_norm = (np.asarray(x_windows, dtype=np.float32) - info["x_mean"]) / info["x_std"]
    net = info["net"].to(device)
    with torch.no_grad():
        xt = torch.from_numpy(x_norm).to(device, dtype=torch.float32)
        pred_norm = net(xt).cpu().numpy()
    pred = pred_norm * info["y_std"] + info["y_mean"]
    return pred, info["targets"]
