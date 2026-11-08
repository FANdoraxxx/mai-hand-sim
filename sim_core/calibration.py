"""Official-referenced calibration for sim_core scores (simval).

Reference = sim_core run over official MASTER/Re:MASTER charts (versionid>=10, lv 12-15.9) via scratch/simval/score_dir.py,
stored in sim_core/reference_official.json as per-level-band ECDFs of density metrics.
  band = floor(lv) plus '+' when frac>=0.6  ('12','12+','13',...)
  metrics (all per 100 notes unless noted): sev (sum of event severity), ev (event count), fly (fly_hand_rate %), fat (mean fatigue), erg_cost
Usage:
  from sim_core.calibration import calibrate
  rep = calibrate(result.to_dict() or score_dir row, lv=14.7)
  rep['z']['sev'], rep['pct']['sev'] (percentile among official charts of the band), rep['flag'] ('ok'|'warn'|'hard' by density percentile: >p95/p99 of official; NOTE the per-event hard verdict is now sim_core.limits)
Interpretation: pct high = chart is MORE 'unreasonable' per the simulator than that fraction of official charts of the level.
"""
from __future__ import annotations
import json, math, os, bisect
from typing import Any, Dict, Iterable
from sim_core.version import check_artifact_version, stamp_artifact_version

REF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reference_official.json")
TYPES = ("ARM_COLLISION","DUAL_JACK_IMPOSSIBLE","EXTREME_CROSS_ARM","FLYING_HAND","FORCED_SINGLE_HAND_JACK","FORCED_SINGLE_HAND_JUMP","HAND_COLLISION","SLIDE_JUMP_ZONE","THREE_HAND_CONFLICT")
METRICS = ("sev", "ev", "fly", "fat") + tuple("t_" + t for t in TYPES)   # t_<type> = severity density of that type (hard+soft) per 100 notes

def band_of(lv) -> str:
    v = float(lv); f = math.floor(v + 1e-9)
    return "%d%s" % (f, "+" if v - f >= 0.6 else "")

def metrics_of(d: Dict[str, Any]) -> Dict[str, float]:
    """d: score_dir row or ChartEvaluationResult.to_dict()."""
    if "sum_sev" in d:
        n = max(1, d["notes"]); sev = d["sum_sev"]; ev = d["n_events"]; fly = d["fly_rate"]; fat = d["fat_mean"]
    else:
        n = max(1, d["metadata"]["note_count"]); evs = d["unreasonable_events"]
        sev = sum(e.get("severity", 1.0) for e in evs); ev = len(evs); fly = d["fly_hand_rate"]
        fc = d["fatigue_curve"]; fat = sum(fc) / max(1, len(fc))
    allev = list(d.get("ev", [])) + list(d.get("soft", [])) if "ev" in d else list(d.get("unreasonable_events", [])) + list(d.get("soft_events", []))
    out = dict(sev=sev / n * 100, ev=ev / n * 100, fly=fly, fat=fat)
    for t in TYPES:
        out["t_" + t] = sum(float(e.get("sev", e.get("severity", 1.0))) for e in allev if e["type"] == t) / n * 100
    return out

def build_reference(jsonl_paths: Iterable[str], out: str = REF_PATH) -> Dict[str, Any]:
    by = {}
    for p in jsonl_paths:
        for l in open(p, encoding="utf-8"):
            r = json.loads(l)
            if not r.get("ok") or r.get("lv") is None: continue
            by.setdefault(band_of(r["lv"]), []).append(metrics_of(r))
    ref = {}
    pooled = {m: sorted(x[m] for v in by.values() for x in v) for m in METRICS}
    for b, v in by.items():
        ref[b] = dict(n=len(v))
        for m in METRICS:
            xs = sorted(x[m] for x in v)
            mean = sum(xs) / len(xs); sd = math.sqrt(sum((x - mean) ** 2 for x in xs) / max(1, len(xs) - 1)) or 1.0
            q = lambda p: xs[min(len(xs) - 1, int(p * len(xs)))]
            ref[b][m] = dict(mean=mean, sd=sd, q={str(p): q(p) for p in (.05, .25, .5, .75, .9, .95, .99)}, xs=[round(x, 4) for x in xs])
    stamp_artifact_version(ref)
    json.dump(ref, open(out, "w"), indent=0)
    return ref

_REF = None
def reference():
    global _REF
    if _REF is None:
        with open(REF_PATH, "r", encoding="utf-8") as f:
            _REF = json.load(f)
        check_artifact_version(_REF, "sim_core/reference_official.json")
    return _REF

def reload_reference():
    global _REF
    _REF = None
    return reference()

def calibrate(d: Dict[str, Any], lv) -> Dict[str, Any]:
    ref = reference(); b = band_of(lv)
    valid_bands = [k for k in ref if not k.startswith("_") and isinstance(ref[k], dict) and "n" in ref[k]]
    if b not in valid_bands or ref[b]["n"] < 15:        # fall back to integer level, then pooled neighbours
        cand = [k for k in valid_bands if k.rstrip("+") == b.rstrip("+")] or valid_bands
        b = max(cand, key=lambda k: ref[k]["n"])
    m = metrics_of(d); z = {}; pct = {}
    for k in METRICS:
        if k not in ref[b]:
            z[k] = 0.0; pct[k] = 0.0; continue
        r = ref[b][k]; z[k] = (m[k] - r["mean"]) / r["sd"]
        pct[k] = bisect.bisect_right(r["xs"], m[k]) / len(r["xs"])
    flag = "hard" if pct["sev"] > 0.99 else ("warn" if pct["sev"] > 0.95 else "ok")
    return dict(band=b, metrics=m, z=z, pct=pct, flag=flag)
