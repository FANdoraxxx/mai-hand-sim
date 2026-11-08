"""One-click full rebuild & synchronization pipeline (`scratch/simv3/rebuild_all.py`).

Whenever `sim_core/planner.py`, `kinematics.py`, `body_model.py`, `vocab_and_rarity.py`, or
`geometry.py` is modified, running this script:
  1. Checks `get_sim_core_code_hash()` and rebuilds `out/simv3/v3_*.jsonl` (1,187 official MiLK+
     charts + 6 generated sets = 1,547 charts) on 3 CPU workers.
  2. Re-fits `sim_core/limits.json`, `sim_core/reference_official.json`, and
     `sim_core/body_distribution_model.json` strictly on `tune` (836 official charts), embedding
     `sim_core_version` and `sim_core_code_hash`.
  3. Re-trains `sim_core/surrogate_weights.pt` (`ChoreoSurrogateNet`, 15 targets) on the rebuilt
     `v3_official.jsonl`, embedding `sim_core_version` and `sim_core_code_hash`.
  4. Re-runs `run_v3_fix_evals.py` (313 untouched MiLK+ official charts, 50-video handcam benchmark,
     50 verifiable note cases with `planner != human` taxonomy, and `beam_width=12` sensitivity).
  5. Writes the complete before-vs-after diff and 6-feature raw interest diagnosis to
     `out/simv3/rebuild_before_after.json`.
"""
from __future__ import annotations

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "1"

import argparse
import json
import sys
import time
from typing import Any, Dict, List

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scratch.simv2.analyze_all import ALL_TYPES, GEN_NAMES, apply_limits_to_row, cohens_d, roc_auc
from sim_core.version import SIM_CORE_VERSION, check_artifact_version, get_sim_core_code_hash
from sim_core.vocab_and_rarity import INTEREST_FEATURE_KEYS, compute_interest_features, score_bar_rarity_and_awkwardness

OUT_V3 = os.path.join(ROOT, "out", "simv3")


def _compute_fp_table(rows: List[Dict[str, Any]], lim: Dict[str, Any]) -> Dict[str, Any]:
    for r in rows:
        apply_limits_to_row(r, lim)
    n = max(1, len(rows))
    by_t = {}
    for t in ALL_TYPES:
        cnt = sum(1 for r in rows if any(e["type"] == t for e in r["hard_events"]))
        by_t[t] = {
            "thr": lim.get(t, {}).get("thr"),
            "demoted": bool(lim.get(t, {}).get("demoted", False)),
            "count": cnt,
            "rate": round(cnt / n, 4),
        }
    any_cnt = sum(1 for r in rows if not r["is_playable_cal"])
    return {
        "n": len(rows),
        "any_hard_count": any_cnt,
        "any_hard_rate": round(any_cnt / n, 4),
        "by_type": by_t,
    }


def _load_jsonl_rows(p: str) -> List[Dict[str, Any]]:
    out = []
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    out.append(json.loads(line))
    return out


def build_before_after_and_interest_diagnosis() -> Dict[str, Any]:
    snap_p = os.path.join(OUT_V3, "before_rebuild_snapshot.json")
    before = json.load(open(snap_p, "r", encoding="utf-8")) if os.path.exists(snap_p) else {}

    lim_after = json.load(open(os.path.join(ROOT, "sim_core", "limits.json"), "r", encoding="utf-8"))
    full_after = json.load(open(os.path.join(OUT_V3, "v3_full_report.json"), "r", encoding="utf-8"))
    sur_after = json.load(open(os.path.join(OUT_V3, "sec6_surrogate_v3_report.json"), "r", encoding="utf-8"))
    fix_after = json.load(open(os.path.join(OUT_V3, "v3_fix_summary.json"), "r", encoding="utf-8"))

    off_rows = _load_jsonl_rows(os.path.join(OUT_V3, "v3_official.jsonl"))
    tune_rows = [r for r in off_rows if r.get("split") == "tune"]
    ho_rows = [r for r in off_rows if r.get("split") == "heldout"]

    fp_tune_after = _compute_fp_table(tune_rows, lim_after)
    fp_ho_after = _compute_fp_table(ho_rows, lim_after)

    # Load all generated sets and compute per-row interest & rarity with the rebuilt model
    gen_sets: Dict[str, List[Dict[str, Any]]] = {}
    for gn in GEN_NAMES:
        grs = _load_jsonl_rows(os.path.join(OUT_V3, f"v3_{gn}.jsonl"))
        for r in grs:
            apply_limits_to_row(r, lim_after)
            hard_bars = {int(e.get("bar", -1)) for e in r["hard_events"]}
            seg_bar = np.asarray(r["seg_bar"], dtype=np.float32)
            r_rows, r_summ = score_bar_rarity_and_awkwardness(seg_bar, r["per_bar_body"], hard_bars, lv=float(r["lv"]))
            r["rarity_summary"] = r_summ
            r["interest"] = compute_interest_features(
                seg_bar, r_rows, {"summary": r["motion_summary"], "music_coupling": r["music_coupling"]},
                is_playable=r["is_playable_cal"], n_hard_violations=r["n_hard"], awkward_mean=r_summ["awkward_mean"],
            )
        gen_sets[gn] = grs

    for r in ho_rows:
        hard_bars = {int(e.get("bar", -1)) for e in r["hard_events"]}
        seg_bar = np.asarray(r["seg_bar"], dtype=np.float32)
        r_rows, r_summ = score_bar_rarity_and_awkwardness(seg_bar, r["per_bar_body"], hard_bars, lv=float(r["lv"]))
        r["rarity_summary"] = r_summ
        r["interest"] = compute_interest_features(
            seg_bar, r_rows, {"summary": r["motion_summary"], "music_coupling": r["music_coupling"]},
            is_playable=r["is_playable_cal"], n_hard_violations=r["n_hard"], awkward_mean=r_summ["awkward_mean"],
        )

    # Task 2A: Detailed analysis of each of the 6 raw interest sub-features across
    # official_heldout vs champ_slide80 vs neg_csft_s3000 (both ALL charts and PLAYABLE-ONLY charts)
    champ_all = gen_sets["champ_slide80"]
    neg_all = gen_sets["neg_csft_s3000"]
    ho_play = [r for r in ho_rows if r["is_playable_cal"]]
    champ_play = [r for r in champ_all if r["is_playable_cal"]]
    neg_play = [r for r in neg_all if r["is_playable_cal"]]

    raw_interest_diagnosis: Dict[str, Any] = {}
    for fk in list(INTEREST_FEATURE_KEYS) + ["raw_interest_score", "interest_score"]:
        ho_a = [float(r["interest"][fk]) for r in ho_rows]
        ch_a = [float(r["interest"][fk]) for r in champ_all]
        ng_a = [float(r["interest"][fk]) for r in neg_all]

        ho_p = [float(r["interest"][fk]) for r in ho_play]
        ch_p = [float(r["interest"][fk]) for r in champ_play]
        ng_p = [float(r["interest"][fk]) for r in neg_play]

        d_neg_vs_champ_all = round(cohens_d(ng_a, ch_a), 3)
        d_ho_vs_champ_all = round(cohens_d(ho_a, ch_a), 3)
        d_ho_vs_neg_all = round(cohens_d(ho_a, ng_a), 3)
        auc_neg_gt_champ = round(roc_auc(ng_a, ch_a), 4)

        raw_interest_diagnosis[fk] = {
            "all_charts_mean": {
                "official_heldout": round(float(np.mean(ho_a)), 4),
                "champ_slide80": round(float(np.mean(ch_a)), 4),
                "neg_csft_s3000": round(float(np.mean(ng_a)), 4),
            },
            "playable_only_mean": {
                "official_heldout": round(float(np.mean(ho_p)), 4) if ho_p else 0.0,
                "champ_slide80": round(float(np.mean(ch_p)), 4) if ch_p else 0.0,
                "neg_csft_s3000": round(float(np.mean(ng_p)), 4) if ng_p else 0.0,
            },
            "cohens_d_all": {
                "neg_minus_champ": d_neg_vs_champ_all,
                "official_minus_champ": d_ho_vs_champ_all,
                "official_minus_neg": d_ho_vs_neg_all,
            },
            "auc_neg_higher_than_champ": auc_neg_gt_champ,
            "inflated_by_bad_charts": bool(d_neg_vs_champ_all > 0.15 and fk not in ("interest_score",)),
        }

    return {
        "sim_core_version": SIM_CORE_VERSION,
        "sim_core_code_hash": get_sim_core_code_hash(),
        "artifacts_hash_verified": {
            "limits_json": check_artifact_version(lim_after, "sim_core/limits.json"),
            "reference_official_json": check_artifact_version(
                json.load(open(os.path.join(ROOT, "sim_core", "reference_official.json"), "r", encoding="utf-8")),
                "sim_core/reference_official.json",
            ),
            "body_distribution_model_json": check_artifact_version(
                json.load(open(os.path.join(ROOT, "sim_core", "body_distribution_model.json"), "r", encoding="utf-8")),
                "sim_core/body_distribution_model.json",
            ),
        },
        "fp_comparison": {
            "before": {
                "tune": before.get("fp_tune", {}),
                "heldout": before.get("fp_heldout", {}),
            },
            "after": {
                "tune": fp_tune_after,
                "heldout": fp_ho_after,
                "untouched_313": fix_after.get("untouched_official_validation", {}),
            },
        },
        "rarity_and_auc_comparison": {
            "before": before.get("full_report", {}).get("section5_rarity_and_unacceptable", {}),
            "after": full_after.get("section5_rarity_and_unacceptable", {}),
        },
        "vocab_shares_comparison": {
            "before": [
                {"word_id": w["word_id"], "code": w["code"], "share": w["share_overall"]}
                for w in before.get("full_report", {}).get("section5_vocab_catalog", [])
            ],
            "after": [
                {"word_id": w["word_id"], "code": w["code"], "share": w["share_overall"]}
                for w in full_after.get("section5_vocab_catalog", [])
            ],
        },
        "surrogate_comparison": {
            "before": before.get("surrogate_report", {}),
            "after": sur_after,
        },
        "raw_interest_6features_diagnosis": raw_interest_diagnosis,
        "handcam_50_cases_discrepancy_summary": fix_after.get("handcam_benchmark", {}).get("cases_50_discrepancy_summary", {}),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="Force re-evaluation of all v3_*.jsonl even if hash matches")
    args = ap.parse_args()

    t0 = time.perf_counter()
    cur_hash = get_sim_core_code_hash()
    print(f"=== sim_core Rebuild All (version={SIM_CORE_VERSION}, code_hash={cur_hash}) ===", flush=True)

    if args.force:
        import glob
        for p in glob.glob(os.path.join(OUT_V3, "v3_*.jsonl")):
            with open(p, "w", encoding="utf-8") as f:
                pass

    print("\n[Step 1/4] Running full planner & body model evaluation + tune fitting (run_v3_experiments.py)...", flush=True)
    from scratch.simv3 import run_v3_experiments
    run_v3_experiments.main()

    print("\n[Step 2/4] Re-training 15-target neural surrogate (train_surrogate_v3.py)...", flush=True)
    from scratch.simv3 import train_surrogate_v3
    train_surrogate_v3.main()

    print("\n[Step 3/4] Running untouched official validation, handcam benchmark & sensitivity (run_v3_fix_evals.py)...", flush=True)
    from scratch.simv3 import run_v3_fix_evals
    run_v3_fix_evals.main()

    print("\n[Step 4/4] Building before-vs-after comparison & 6-feature raw interest diagnosis...", flush=True)
    diff_rep = build_before_after_and_interest_diagnosis()
    out_p = os.path.join(OUT_V3, "rebuild_before_after.json")
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(diff_rep, f, ensure_ascii=False, indent=2)
    print(f"\n=== Rebuild Complete in {time.perf_counter() - t0:.1f}s -> {out_p} ===", flush=True)


if __name__ == "__main__":
    main()
