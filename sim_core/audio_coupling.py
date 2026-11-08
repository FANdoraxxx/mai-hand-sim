"""Audio feature extraction & section alignment for `sim_core` music-choreography coupling.

Supports:
  1. Instant lookup from `work/mel/<sid>.npy` (200 fps, 96-dim log-mel) via `work/charts.pkl`
     folder / title index (< 1 ms, zero external process spawn).
  2. Direct audio loading from `track.mp3` / `track.ogg` / `track.wav` in the chart directory
     using `mai.audio` (CPU only, never touches GPU!).
  3. Chorus / Drop vs Verse / Calm bar segmentation using `release/v6d2_plus/frozen/sectionpred.pkl`
     when `E:\\maiAIV3_feats\\bars_v3\\<sid>_<slot>.npy` is available, with automatic fallback to
     smoothed log-mel energy + spectral onset flux segmentation.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import pickle
import re
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

MEL_FPS: float = 50.0
_RAW_MEL_FPS: float = 200.0
_MEL_STRIDE: int = 4
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_UUID_RE = re.compile(r"\s*\[[0-9a-fA-F-]{8,}\]\s*$")


def _canon_name(s: str) -> str:
    s = _UUID_RE.sub("", s.strip()).strip()
    if s.endswith(" [DX]"):
        s = s[:-5].strip()
    return s


@dataclass
class AudioCouplingData:
    has_audio: bool
    section_source: str                        # 'sectionpred' | 'mel_energy' | 'none'
    bar_loudness_db: np.ndarray                # float32 [N_bars]
    bar_onset_flux: np.ndarray                 # float32 [N_bars]
    bar_audio_embed: np.ndarray                # float32 [N_bars, 24] normalized chroma/mel signature for audio recurrence
    is_chorus_bar: np.ndarray                  # bool [N_bars]
    onset_env_200fps: Optional[np.ndarray]     # float32 [T_mel] normalized [0,1] onset flux envelope at 200 fps
    mel_offset_sec: float = 0.0


_CHARTS_INDEX: Optional[Tuple[Dict[str, Tuple[int, Dict[int, bool]]], Dict[str, int]]] = None
_MEL_MEM_CACHE: Dict[str, np.ndarray] = {}
_SECPRED_MODEL = None
_SECPRED_CACHE: Dict[Tuple[int, int], Optional[np.ndarray]] = {}
_SECPRED_JSON_MAP: Optional[Dict[str, List[int]]] = None


def _get_charts_sid_index() -> Tuple[Dict[str, int], Dict[str, int]]:
    """Load `(folder_to_sid, base_title_to_sid)` from fast `sim_core/audio_sid_index.json` (or `work/charts.pkl`)."""
    global _CHARTS_INDEX
    if _CHARTS_INDEX is not None:
        return _CHARTS_INDEX  # type: ignore[return-value]
    by_folder: Dict[str, int] = {}
    by_title: Dict[str, int] = {}
    idx_json = os.path.join(_HERE, "audio_sid_index.json")
    if os.path.exists(idx_json):
        try:
            import json
            d = json.load(open(idx_json, encoding="utf-8"))
            by_folder = {k: int(v) for k, v in d.get("by_folder", {}).items()}
            by_title = {k: int(v) for k, v in d.get("by_title", {}).items()}
            _CHARTS_INDEX = (by_folder, by_title)  # type: ignore[assignment]
            return by_folder, by_title
        except Exception:
            pass
    cp = os.path.join(_ROOT, "work", "charts.pkl")
    if os.path.exists(cp):
        try:
            with open(cp, "rb") as f:
                rows = pickle.load(f)
            for c in rows:
                sid = int(c["sid"])
                fn = str(c.get("folder", "")).strip()
                ti = str(c.get("title", "")).strip()
                if fn:
                    by_folder[fn] = sid
                    by_title.setdefault(_canon_name(fn), sid)
                if ti:
                    by_title.setdefault(ti, sid)
                    by_title.setdefault(_canon_name(ti), sid)
        except Exception:
            pass
    _CHARTS_INDEX = (by_folder, by_title)  # type: ignore[assignment]
    return by_folder, by_title


def _resolve_sid_and_mel(
    source_path: Optional[str],
    title: str,
    mel_override: Optional[np.ndarray] = None,
    audio_path_override: Optional[str] = None,
) -> Tuple[Optional[int], Optional[np.ndarray]]:
    if mel_override is not None:
        return None, np.asarray(mel_override, dtype=np.float32)

    by_folder, by_title = _get_charts_sid_index()
    sid: Optional[int] = None
    folder_name = ""
    chart_dir: Optional[Path] = None
    if source_path:
        p = Path(source_path)
        chart_dir = p if p.is_dir() else p.parent
        folder_name = chart_dir.name
        if folder_name in by_folder:
            sid = by_folder[folder_name]
        elif _canon_name(folder_name) in by_title:
            sid = by_title[_canon_name(folder_name)]
    if sid is None and title and title not in ("InlineSimaiChart", "TickNotesChart", "MaimaiChart"):
        sid = by_title.get(title.strip()) or by_title.get(_canon_name(title))

    if sid is not None:
        mel_path = os.path.join(_ROOT, "work", "mel", f"{sid:05d}.npy")
        if os.path.exists(mel_path):
            if mel_path not in _MEL_MEM_CACHE:
                if len(_MEL_MEM_CACHE) > 48:
                    _MEL_MEM_CACHE.clear()
                mm_raw = np.load(mel_path, mmap_mode="r")
                _MEL_MEM_CACHE[mel_path] = np.asarray(mm_raw[::_MEL_STRIDE], dtype=np.float32)
            return sid, _MEL_MEM_CACHE[mel_path]

    # Direct audio file fallback (CPU only!)
    cand_audio: List[Path] = []
    if audio_path_override:
        cand_audio.append(Path(audio_path_override))
    if chart_dir is not None and chart_dir.exists():
        for fn in ("track.mp3", "track.ogg", "track.wav"):
            cand_audio.append(chart_dir / fn)
    for ap in cand_audio:
        if ap.exists():
            key = str(ap.resolve())
            if key in _MEL_MEM_CACHE:
                return sid, _MEL_MEM_CACHE[key]
            try:
                if _ROOT not in sys.path:
                    sys.path.insert(0, _ROOT)
                from mai.audio import load_audio, mel_db
                wave = load_audio(str(ap))
                mel_arr = np.asarray(mel_db(wave, device="cpu")[::_MEL_STRIDE], dtype=np.float32)
                if len(_MEL_MEM_CACHE) > 48:
                    _MEL_MEM_CACHE.clear()
                _MEL_MEM_CACHE[key] = mel_arr
                return sid, mel_arr
            except Exception:
                pass
    return sid, None


def _predict_chorus_bars_sectionpred(sid: Optional[int], slot: int, n_bars: int) -> Optional[np.ndarray]:
    """Lookup precomputed `sectionpred.pkl` chorus mask (<0.01 ms), falling back to live `sectionpred.pkl`."""
    if sid is None or n_bars <= 0:
        return None
    global _SECPRED_JSON_MAP
    if _SECPRED_JSON_MAP is None:
        cache_json = os.path.join(_HERE, "sectionpred_chorus_cache.json")
        if os.path.exists(cache_json):
            try:
                import json
                _SECPRED_JSON_MAP = json.load(open(cache_json, encoding="utf-8"))
            except Exception:
                _SECPRED_JSON_MAP = {}
        else:
            _SECPRED_JSON_MAP = {}
    s_key = str(sid)
    if s_key in _SECPRED_JSON_MAP:
        arr = np.asarray(_SECPRED_JSON_MAP[s_key], dtype=bool)
        out = np.zeros(n_bars, dtype=bool)
        k = min(n_bars, len(arr))
        out[:k] = arr[:k]
        return out
    if _SECPRED_JSON_MAP:
        return None

    cache_key = (sid, slot)
    if cache_key in _SECPRED_CACHE:
        cached = _SECPRED_CACHE[cache_key]
        if cached is None:
            return None
        out = np.zeros(n_bars, dtype=bool)
        k = min(n_bars, len(cached))
        out[:k] = cached[:k]
        return out

    bars_root = os.environ.get("MAI_BARS_ROOT") or r"E:\maiAIV3_feats\bars_v3"
    tab_path = os.path.join(bars_root, f"{sid:05d}_{slot}.npy")
    if not os.path.exists(tab_path):
        alt_slot = 5 if slot != 5 else 6
        tab_path = os.path.join(bars_root, f"{sid:05d}_{alt_slot}.npy")
    model_path = os.path.join(_ROOT, "release", "v6d2_plus", "frozen", "sectionpred.pkl")
    plan_dir = os.path.join(_ROOT, "release", "v6d2_plus", "frozen", "plan")
    if not (os.path.exists(tab_path) and os.path.exists(model_path) and os.path.isdir(plan_dir)):
        _SECPRED_CACHE[cache_key] = None
        return None

    global _SECPRED_MODEL
    old_path = list(sys.path)
    old_train = sys.modules.get("train")
    old_pf = sys.modules.get("planfeat")
    try:
        sys.path.insert(0, plan_dir)
        import importlib
        pf_mod = importlib.import_module("planfeat")
        tr_mod = importlib.import_module("train")
        sys.modules["train"] = tr_mod
        sys.modules["planfeat"] = pf_mod
        if _SECPRED_MODEL is None:
            with open(model_path, "rb") as f:
                _SECPRED_MODEL = pickle.load(f)
        m = _SECPRED_MODEL
        tab = np.asarray(np.load(tab_path, mmap_mode="r"), dtype=np.float32)
        nb = len(tab)
        feats = pf_mod.clip_feats(pf_mod.bar_feats(tab[:nb], m["pca_mean"], m["pca_comp"]))
        proba = m["label"].predict_proba(feats)
        cls = np.asarray(m["label"].classes_, int)
        p_mat = np.zeros((len(feats), 10), np.float32)
        p_mat[:, cls] = proba
        p_mat = np.maximum(p_mat, 1e-6)
        p_mat /= p_mat.sum(1, keepdims=True)
        bnd = np.zeros(len(feats), np.float32)
        if len(feats) > 1:
            bnd[1:] = 1.0 - np.sum(p_mat[1:] * p_mat[:-1], 1)
        path = tr_mod.viterbi_dur(
            np.log(p_mat),
            m["logT"],
            min_len=int(m.get("min_len", 3)),
            max_len=64,
            logDur=m["logDur"],
            dur_scale=float(m.get("dur_scale", 1.0)),
            bnd=bnd,
            bnd_boost=0.0,
        )
        # Labels 3 ('chorus') and 7 ('drop') are chorus/highlight sections
        mask = np.isin(np.asarray(path, dtype=np.int32), (3, 7))
        _SECPRED_CACHE[cache_key] = mask
        out = np.zeros(n_bars, dtype=bool)
        k = min(n_bars, len(mask))
        out[:k] = mask[:k]
        return out
    except Exception:
        _SECPRED_CACHE[cache_key] = None
        return None
    finally:
        sys.path[:] = old_path
        if old_train is None:
            sys.modules.pop("train", None)
        else:
            sys.modules["train"] = old_train
        if old_pf is None:
            sys.modules.pop("planfeat", None)
        else:
            sys.modules["planfeat"] = old_pf


def extract_audio_coupling(
    bar_timestamps: Sequence[Tuple[int, float, float]],
    source_path: Optional[str] = None,
    title: str = "",
    slot: int = 5,
    mel_override: Optional[np.ndarray] = None,
    audio_path_override: Optional[str] = None,
) -> AudioCouplingData:
    """Compute per-bar loudness, onset flux, normalized spectral recurrence embeddings,
    200-fps onset envelope, and chorus/drop mask aligned to `bar_timestamps`.
    """
    n_bars = len(bar_timestamps)
    if n_bars == 0:
        return AudioCouplingData(
            has_audio=False,
            section_source="none",
            bar_loudness_db=np.zeros(0, dtype=np.float32),
            bar_onset_flux=np.zeros(0, dtype=np.float32),
            bar_audio_embed=np.zeros((0, 24), dtype=np.float32),
            is_chorus_bar=np.zeros(0, dtype=bool),
            onset_env_200fps=None,
        )

    sid, mel = _resolve_sid_and_mel(
        source_path=source_path,
        title=title,
        mel_override=(mel_override[::_MEL_STRIDE] if (mel_override is not None and len(mel_override) > 600) else mel_override),
        audio_path_override=audio_path_override,
    )
    if mel is None or mel.ndim != 2 or mel.shape[0] < 10:
        return AudioCouplingData(
            has_audio=False,
            section_source="none",
            bar_loudness_db=np.zeros(n_bars, dtype=np.float32),
            bar_onset_flux=np.zeros(n_bars, dtype=np.float32),
            bar_audio_embed=np.zeros((n_bars, 24), dtype=np.float32),
            is_chorus_bar=np.zeros(n_bars, dtype=bool),
            onset_env_200fps=None,
        )

    n_mel, n_bins = mel.shape
    # Positive spectral flux across 200-fps log-mel frames
    diff_pos = np.maximum(0.0, mel[1:] - mel[:-1])
    flux_frames = np.zeros(n_mel, dtype=np.float32)
    flux_frames[1:] = diff_pos.mean(axis=1)
    p98 = float(np.percentile(flux_frames, 98)) if n_mel > 20 else 1.0
    onset_env = np.clip(flux_frames / max(1e-4, p98), 0.0, 1.5).astype(np.float32)

    frame_loudness = mel.mean(axis=1)
    # Pool 96 mel bins into 24 compact 4-bin spectral bands for audio phrase self-similarity
    band_w = max(1, n_bins // 24)
    mel_24 = mel[:, : band_w * 24].reshape(n_mel, 24, band_w).mean(axis=2)

    bar_loud = np.zeros(n_bars, dtype=np.float32)
    bar_flux = np.zeros(n_bars, dtype=np.float32)
    bar_emb = np.zeros((n_bars, 24), dtype=np.float32)

    for i, (_b_idx, t0, t1) in enumerate(bar_timestamps):
        f0 = max(0, min(n_mel - 1, int(round(t0 * MEL_FPS))))
        f1 = max(f0 + 1, min(n_mel, int(round(t1 * MEL_FPS))))
        bar_loud[i] = float(frame_loudness[f0:f1].mean())
        bar_flux[i] = float(flux_frames[f0:f1].mean())
        vec = mel_24[f0:f1].mean(axis=0)
        vec = vec - vec.mean()
        nrm = float(np.linalg.norm(vec))
        if nrm > 1e-6:
            bar_emb[i] = vec / nrm

    # Chorus segmentation: try sectionpred.pkl first, else energy + onset flux segmentation
    chorus_mask = _predict_chorus_bars_sectionpred(sid, slot, n_bars)
    sec_src = "sectionpred"
    if chorus_mask is None or not chorus_mask.any() or chorus_mask.all():
        sec_src = "mel_energy"
        if n_bars >= 8:
            z_loud = (bar_loud - bar_loud.mean()) / max(1e-4, float(bar_loud.std()))
            z_flux = (bar_flux - bar_flux.mean()) / max(1e-4, float(bar_flux.std()))
            combo = 0.65 * z_loud + 0.35 * z_flux
            # 3-bar moving average smoothing
            kernel = np.array([0.25, 0.50, 0.25], dtype=np.float32)
            smooth = np.convolve(np.pad(combo, (1, 1), mode="edge"), kernel, mode="valid")
            thr = float(np.percentile(smooth, 62))
            chorus_mask = smooth >= thr
        else:
            chorus_mask = bar_loud >= float(np.median(bar_loud))

    return AudioCouplingData(
        has_audio=True,
        section_source=sec_src,
        bar_loudness_db=bar_loud,
        bar_onset_flux=bar_flux,
        bar_audio_embed=bar_emb,
        is_chorus_bar=chorus_mask,
        onset_env_200fps=onset_env,
    )
