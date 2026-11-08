"""Minimal Pairwise Choreography Preference Collection Tool (`scratch/simv3/prefs_tool/server.py`).

Provides both a local web UI (`http://127.0.0.1:8765`) and an interactive CLI (`--cli`)
for comparing two 4-8 bar chart windows from the same song (Official vs Generated versions):
  - Streams the song's `track.mp3` synchronized to the selected 4-8 bar window `[t_start, t_end]`.
  - Extracts the window's Simai text, 30 fps `BodyTimeline` hand/torso/arm trajectory frames,
    playability violations, awkwardness score, and the 6 raw interest features (`vocab_entropy`,
    `phrase_development`, `motif_echo_transformed`, `chorus_peak_contrast`,
    `music_kinetic_coupling`, `anti_stagnation`).
  - Records user votes ("A is better", "B is better", or "Tie") into
    `scratch/simv3/prefs/user_prefs.jsonl` in the exact schema consumed by
    `sim_core.vocab_and_rarity.load_interest_weights()`.
  - Does NOT generate any synthetic preference records.
"""
from __future__ import annotations

import argparse
import datetime
import http.server
import json
import os
import random
import sys
import urllib.parse
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scratch.simv2.run_eval_all import GEN_SETS, enumerate_generated_set
from scratch.simv2.splits import canonical_song_title, profile_for_level
from sim_core import evaluate_chart
from sim_core.body_model import build_body_timeline_from_plan
from sim_core.maidata_parser import parse_simai_chart
from sim_core.vocab_and_rarity import (
    INTEREST_FEATURE_KEYS,
    compute_interest_features,
    extract_segment_features_multiscale,
    load_interest_weights,
    score_bar_rarity_and_awkwardness,
)

PREFS_DIR = os.path.join(ROOT, "scratch", "simv3", "prefs")
USER_PREFS_JSONL = os.path.join(PREFS_DIR, "user_prefs.jsonl")

# Cache available songs & their chart paths across Official + 6 Generated sets
_CATALOG_CACHE: Optional[Dict[str, Dict[str, Any]]] = None
_CHART_EVAL_CACHE: Dict[Tuple[str, str, int], Dict[str, Any]] = {}


def build_song_catalog() -> Dict[str, Dict[str, Any]]:
    """Build catalog of songs that exist in both `official charts/` and the generated sets."""
    global _CATALOG_CACHE
    if _CATALOG_CACHE is not None:
        return _CATALOG_CACHE

    off_root = os.path.join(ROOT, "official charts")
    off_by_bt: Dict[str, str] = {}
    if os.path.exists(off_root):
        for folder in os.listdir(off_root):
            p = os.path.join(off_root, folder, "maidata.txt")
            if os.path.exists(p):
                off_by_bt[canonical_song_title(folder)] = folder

    catalog: Dict[str, Dict[str, Any]] = {}
    for sname, srel in GEN_SETS.items():
        items = enumerate_generated_set(sname, srel)
        for it in items:
            bt = it["base_title"]
            off_folder = off_by_bt.get(bt)
            if not off_folder:
                continue
            off_mai = os.path.join(off_root, off_folder, "maidata.txt")
            off_mp3 = os.path.join(off_root, off_folder, "track.mp3")
            song_entry = catalog.setdefault(
                off_folder,
                {
                    "song": off_folder,
                    "base_title": bt,
                    "slot": int(it["slot"]),
                    "lv": float(it["lv"]),
                    "audio_path": off_mp3 if os.path.exists(off_mp3) else "",
                    "sources": {
                        "official": off_mai,
                    },
                },
            )
            song_entry["sources"][sname] = it["path"]

    _CATALOG_CACHE = dict(sorted(catalog.items()))
    return _CATALOG_CACHE


def _extract_simai_bars_text(maidata_path: str, slot: int, bar_lo: int, bar_hi: int) -> str:
    """Extract the raw Simai measure text for bars `[bar_lo, bar_hi)`."""
    from mai.simai import read_maidata
    try:
        _meta, charts = read_maidata(maidata_path)
        raw = charts.get(slot, "")
        # Split rough measures by 4 quarter beats / comma groups or return compact slice
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip() and not ln.strip().startswith("&")]
        if not lines:
            return raw[:400]
        lo_idx = max(0, min(len(lines) - 1, int(bar_lo * len(lines) / max(1, bar_hi + 20))))
        hi_idx = min(len(lines), lo_idx + max(4, bar_hi - bar_lo + 2))
        return "\n".join(lines[lo_idx:hi_idx])
    except Exception:
        return ""


def evaluate_window_candidate(song: str, source_name: str, bar_lo: int, bar_hi: int) -> Dict[str, Any]:
    catalog = build_song_catalog()
    if song not in catalog:
        raise ValueError(f"Unknown song: {song}")
    entry = catalog[song]
    slot = int(entry["slot"])
    lv = float(entry["lv"])
    chart_path = entry["sources"].get(source_name)
    if not chart_path or not os.path.exists(chart_path):
        raise ValueError(f"Source {source_name!r} unavailable for {song}")

    cache_key = (song, source_name, slot)
    if cache_key not in _CHART_EVAL_CACHE:
        prof = profile_for_level(lv)
        res = evaluate_chart(chart_path, difficulty=slot, hit_mode="hybrid", skill_profile=prof, frame_fps=30, include_motion=True)
        assert res.plan_result is not None
        bt = build_body_timeline_from_plan(res.plan_result, materialize_frame_dicts=False)
        seg_mats = extract_segment_features_multiscale(bt, motion_dict=res.motion, scales=("bar",))
        _CHART_EVAL_CACHE[cache_key] = {
            "res": res,
            "bt": bt,
            "seg_bar": seg_mats["bar"],
            "chart": res.plan_result.chart,
        }

    cached = _CHART_EVAL_CACHE[cache_key]
    res = cached["res"]
    bt = cached["bt"]
    seg_bar_full = cached["seg_bar"]
    chart = cached["chart"]

    n_bars = len(bt.per_bar)
    b0 = max(0, min(n_bars - 1, int(bar_lo)))
    b1 = max(b0 + 1, min(n_bars, int(bar_hi)))

    # Find time range [t_start, t_end] for bars [b0, b1)
    win_notes = [n for n in chart.notes if b0 <= int(n.bar) < b1]
    if win_notes:
        t_start = max(0.0, float(min(n.time for n in win_notes)) - 0.15)
        t_end = float(max(n.end_time for n in win_notes)) + 0.25
    else:
        dur = max(10.0, float(chart.total_duration_sec))
        t_start = dur * (b0 / max(1, n_bars))
        t_end = dur * (b1 / max(1, n_bars))

    seg_w = seg_bar_full[b0:b1]
    pbb_w = bt.per_bar[b0:b1]
    win_hard_evs = [e for e in res.unreasonable_events if b0 <= int(e.get("bar", -1)) < b1 and e.get("hard", True)]
    win_soft_evs = [e for e in res.soft_events if b0 <= int(e.get("bar", -1)) < b1]
    hard_bars = {int(e.get("bar", -1)) for e in win_hard_evs}

    r_rows, r_summ = score_bar_rarity_and_awkwardness(seg_w, pbb_w, hard_bars, lv=lv)
    int_dict = compute_interest_features(
        seg_w,
        r_rows,
        {"summary": res.motion.get("summary", {}), "music_coupling": res.motion.get("music_coupling", {})},
        is_playable=(len(win_hard_evs) == 0),
        n_hard_violations=len(win_hard_evs),
        awkward_mean=r_summ["awkward_mean"],
    )

    # Extract 30fps trajectory frames in [t_start, t_end]
    fps = float(bt.fps)
    f0 = max(0, int(t_start * fps))
    f1 = min(len(bt.arrays["left_xyz"]), int(t_end * fps) + 1)
    frames_out = []
    l_xyz = bt.arrays["left_xyz"]
    r_xyz = bt.arrays["right_xyz"]
    t_6 = bt.arrays["torso_6dof"]
    l_a14 = bt.arrays["left_arm_14"]
    r_a14 = bt.arrays["right_arm_14"]
    for fi in range(f0, f1):
        frames_out.append({
            "t": round(fi / fps, 3),
            "l": [round(float(l_xyz[fi, 0]), 1), round(float(l_xyz[fi, 1]), 1), round(float(l_xyz[fi, 2]), 1)],
            "r": [round(float(r_xyz[fi, 0]), 1), round(float(r_xyz[fi, 1]), 1), round(float(r_xyz[fi, 2]), 1)],
            "torso": [round(float(t_6[fi, 0]), 1), round(float(t_6[fi, 1]), 1), round(float(t_6[fi, 3]), 1)],
            "awk": round(float(0.5 * (l_a14[fi, 13] + r_a14[fi, 13])), 3),
        })

    note_map = {nr["note_id"]: nr for nr in bt.note_table}
    notes_out = []
    for n in win_notes:
        nr = note_map.get(n.note_id, {})
        tx, ty = nr.get("target_xy", (0.0, 0.0))
        notes_out.append({
            "t": round(float(n.time), 3),
            "end_t": round(float(n.end_time), 3),
            "bar": int(n.bar),
            "key": n.key,
            "type": n.note_type.value,
            "hand": nr.get("hand", "R"),
            "tech": nr.get("technique", "Tap"),
            "xy": [round(float(tx), 1), round(float(ty), 1)],
        })

    return {
        "song": song,
        "source": source_name,
        "slot": slot,
        "lv": lv,
        "bar_lo": b0,
        "bar_hi": b1,
        "n_total_bars": n_bars,
        "t_start": round(t_start, 3),
        "t_end": round(t_end, 3),
        "simai_excerpt": _extract_simai_bars_text(chart_path, slot, b0, b1),
        "is_playable": len(win_hard_evs) == 0,
        "hard_violations": win_hard_evs,
        "soft_violations": win_soft_evs,
        "awkward_mean": r_summ["awkward_mean"],
        "rarity_mean": r_summ["rarity_mean"],
        "unacceptable_mean": r_summ["unacceptable_mean"],
        "interest_features": {k: round(float(int_dict[k]), 4) for k in INTEREST_FEATURE_KEYS},
        "raw_interest_score": int_dict["raw_interest_score"],
        "interest_score": int_dict["interest_score"],
        "per_bar_words": [{"bar": int(r["bar"]), "word_code": r["word_code"], "awkward": r["awkward_score"]} for r in r_rows],
        "frames": frames_out,
        "notes": notes_out,
    }


def append_preference_vote(payload: Dict[str, Any], out_jsonl: str = USER_PREFS_JSONL) -> Dict[str, Any]:
    """Save a user preference vote (`choice` in `{'A', 'B', 'tie'}`) in the format
    consumed by `sim_core.vocab_and_rarity.load_interest_weights()`.
    """
    os.makedirs(os.path.dirname(out_jsonl), exist_ok=True)
    choice = str(payload.get("choice", "")).strip()
    feat_a = {k: float(payload["features_a"][k]) for k in INTEREST_FEATURE_KEYS}
    feat_b = {k: float(payload["features_b"][k]) for k in INTEREST_FEATURE_KEYS}
    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

    records_to_write = []
    if choice == "A":
        records_to_write.append({
            "timestamp": now_iso,
            "song": payload.get("song", ""),
            "slot": int(payload.get("slot", 5)),
            "bar_lo": int(payload.get("bar_lo", 0)),
            "bar_hi": int(payload.get("bar_hi", 4)),
            "source_chosen": payload.get("source_a", "A"),
            "source_rejected": payload.get("source_b", "B"),
            "preference": "A",
            "chosen": feat_a,
            "rejected": feat_b,
            "weight": 1.0,
        })
    elif choice == "B":
        records_to_write.append({
            "timestamp": now_iso,
            "song": payload.get("song", ""),
            "slot": int(payload.get("slot", 5)),
            "bar_lo": int(payload.get("bar_lo", 0)),
            "bar_hi": int(payload.get("bar_hi", 4)),
            "source_chosen": payload.get("source_b", "B"),
            "source_rejected": payload.get("source_a", "A"),
            "preference": "B",
            "chosen": feat_b,
            "rejected": feat_a,
            "weight": 1.0,
        })
    elif choice.lower() == "tie":
        # Symmetric 0.5-weight pair representing an exact tie in Bradley-Terry logistic fitting
        records_to_write.append({
            "timestamp": now_iso,
            "song": payload.get("song", ""),
            "slot": int(payload.get("slot", 5)),
            "bar_lo": int(payload.get("bar_lo", 0)),
            "bar_hi": int(payload.get("bar_hi", 4)),
            "source_chosen": payload.get("source_a", "A"),
            "source_rejected": payload.get("source_b", "B"),
            "preference": "tie",
            "chosen": feat_a,
            "rejected": feat_b,
            "weight": 0.5,
        })
        records_to_write.append({
            "timestamp": now_iso,
            "song": payload.get("song", ""),
            "slot": int(payload.get("slot", 5)),
            "bar_lo": int(payload.get("bar_lo", 0)),
            "bar_hi": int(payload.get("bar_hi", 4)),
            "source_chosen": payload.get("source_b", "B"),
            "source_rejected": payload.get("source_a", "A"),
            "preference": "tie",
            "chosen": feat_b,
            "rejected": feat_a,
            "weight": 0.5,
        })
    else:
        raise ValueError(f"Invalid choice {choice!r}; must be 'A', 'B', or 'tie'")

    with open(out_jsonl, "a", encoding="utf-8") as f:
        for rec in records_to_write:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Reset cached weights so load_interest_weights() reflects the new vote immediately
    import sim_core.vocab_and_rarity as vr_mod
    vr_mod._INTEREST_WEIGHTS_CACHE = None
    updated_weights = load_interest_weights()
    total_lines = sum(1 for _ in open(out_jsonl, "r", encoding="utf-8")) if os.path.exists(out_jsonl) else 0
    return {
        "saved_records": len(records_to_write),
        "total_jsonl_lines": total_lines,
        "file": out_jsonl,
        "updated_weights": updated_weights,
    }


class PrefsRequestHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        pass

    def _send_json(self, obj: Any, status: int = 200) -> None:
        raw = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        qs = urllib.parse.parse_qs(parsed.query)
        if parsed.path in ("/", "/index.html"):
            html_p = os.path.join(HERE, "index.html")
            raw = open(html_p, "rb").read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            return
        if parsed.path == "/api/catalog":
            cat = build_song_catalog()
            songs_list = [
                {
                    "song": k,
                    "lv": v["lv"],
                    "slot": v["slot"],
                    "has_audio": bool(v["audio_path"]),
                    "sources": list(v["sources"].keys()),
                }
                for k, v in cat.items()
            ]
            self._send_json({"songs": songs_list, "current_weights": load_interest_weights()})
            return
        if parsed.path == "/api/pair":
            cat = build_song_catalog()
            song = qs.get("song", [next(iter(cat.keys()))])[0]
            src_a = qs.get("source_a", ["official"])[0]
            src_b = qs.get("source_b", ["champ_slide80"])[0]
            bar_lo = int(qs.get("bar_lo", ["16"])[0])
            bar_hi = int(qs.get("bar_hi", [str(bar_lo + 4)])[0])
            try:
                cand_a = evaluate_window_candidate(song, src_a, bar_lo, bar_hi)
                cand_b = evaluate_window_candidate(song, src_b, bar_lo, bar_hi)
                self._send_json({"ok": True, "cand_a": cand_a, "cand_b": cand_b})
            except Exception as e:
                self._send_json({"ok": False, "error": str(e)}, status=400)
            return
        if parsed.path == "/api/audio":
            cat = build_song_catalog()
            song = qs.get("song", [""])[0]
            mp3_p = cat.get(song, {}).get("audio_path", "")
            if not mp3_p or not os.path.exists(mp3_p):
                self.send_error(404, "Audio not found")
                return
            raw = open(mp3_p, "rb").read()
            self.send_response(200)
            self.send_header("Content-Type", "audio/mpeg")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            self.wfile.write(raw)
            return
        self.send_error(404, "Not found")

    def do_POST(self) -> None:
        if self.path == "/api/vote":
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            try:
                res = append_preference_vote(body)
                self._send_json({"ok": True, **res})
            except Exception as e:
                self._send_json({"ok": False, "error": str(e)}, status=400)
            return
        self.send_error(404, "Not found")


def run_cli_interactive() -> None:
    cat = build_song_catalog()
    songs = list(cat.keys())
    print(f"Loaded {len(songs)} songs with official + generated charts.")
    while True:
        song = random.choice(songs)
        sources = list(cat[song]["sources"].keys())
        s_a, s_b = random.sample(sources, 2)
        cand0 = evaluate_window_candidate(song, s_a, 16, 20)
        max_b = max(8, cand0["n_total_bars"] - 8)
        bar_lo = random.randrange(4, max_b, 4)
        bar_hi = bar_lo + 4
        ca = evaluate_window_candidate(song, s_a, bar_lo, bar_hi)
        cb = evaluate_window_candidate(song, s_b, bar_lo, bar_hi)
        print("\n" + "=" * 72)
        print(f"Song: {song} (Lv {ca['lv']}) | Bars {bar_lo}..{bar_hi} ({ca['t_start']:.2f}s .. {ca['t_end']:.2f}s)")
        for label, c in (("A", ca), ("B", cb)):
            print(f"\n--- Candidate {label} ---")
            print(f"Simai:\n{c['simai_excerpt']}")
            print(f"Playable: {c['is_playable']} (hard={len(c['hard_violations'])}, awkward={c['awkward_mean']:.3f})")
            print(f"Words: {[w['word_code'] for w in c['per_bar_words']]}")
            print(f"Interest 6-D: {c['interest_features']}")
        ans = input("\nVote [1=A better, 2=Tie, 3=B better, s=Skip, q=Quit]: ").strip().lower()
        if ans == "q":
            break
        if ans in ("1", "2", "3"):
            ch = "A" if ans == "1" else ("tie" if ans == "2" else "B")
            r = append_preference_vote({
                "song": song,
                "slot": ca["slot"],
                "bar_lo": bar_lo,
                "bar_hi": bar_hi,
                "source_a": s_a,
                "source_b": s_b,
                "features_a": ca["interest_features"],
                "features_b": cb["interest_features"],
                "choice": ch,
            })
            print(f"Saved! ({s_a} vs {s_b}) -> total lines in {r['file']}: {r['total_jsonl_lines']}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Pairwise Choreography Preference Collection Tool")
    ap.add_argument("--host", type=str, default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--cli", action="store_true", help="Run interactive terminal mode instead of web server")
    ap.add_argument("--check", action="store_true", help="Dry-run verification of pair extraction without writing any prefs")
    args = ap.parse_args()

    if args.check:
        cat = build_song_catalog()
        first_song = next(iter(cat.keys()))
        ca = evaluate_window_candidate(first_song, "official", 12, 16)
        cb = evaluate_window_candidate(first_song, "champ_slide80", 12, 16)
        print(f"Check OK: {len(cat)} songs | {first_song} bars 12..16: frames_A={len(ca['frames'])}, frames_B={len(cb['frames'])}")
        return

    if args.cli:
        run_cli_interactive()
        return

    srv = http.server.ThreadingHTTPServer((args.host, args.port), PrefsRequestHandler)
    print(f"Pairwise Preference Tool running at http://{args.host}:{args.port}")
    print(f"Votes will be saved to: {USER_PREFS_JSONL}")
    srv.serve_forever()


if __name__ == "__main__":
    main()
