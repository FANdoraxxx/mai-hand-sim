"""Fitted hard/soft split of unreasonable events (simval round 2).

The planner records every candidate violation with a magnitude; an event is HARD only if its magnitude `m` exceeds the
per-type threshold fitted so that official charts essentially never produce it (see agent_tasks/simval.result.md).
Events below threshold stay in `soft_events` and still cost ergonomic score.  No limits.json -> everything hard (legacy).
Magnitudes (higher = worse): jack/jump/fly/dual-jack = severity (= measured value / profile limit);
HAND_COLLISION = 85 - palm distance mm; ARM_COLLISION = 120 - forearm 3D distance mm;
EXTREME_CROSS_ARM = torso yaw acceleration deg/s^2; THREE_HAND_CONFLICT = #simultaneous targets.
"""
import json, os, re
from sim_core.version import check_artifact_version

PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "limits.json")
_RE = {k: re.compile(p) for k, p in dict(
    palm=r"palm dist ([\d.]+) mm", f3d=r"forearm 3D ([\d.]+) mm", yaw=r"yaw_accel=(-?[\d.]+)", n3=r"(\d+) simultaneous").items()}

def measure(e):
    if e.get("raw_m") is not None:
        return float(e["raw_m"])
    t = e["type"]; r = e.get("reason", "")
    try:
        if t == "HAND_COLLISION":
            return 85.0 - float(_RE["palm"].search(r).group(1))
        if t == "ARM_COLLISION":
            return 120.0 - float(_RE["f3d"].search(r).group(1))
        if t == "EXTREME_CROSS_ARM":
            return abs(float(_RE["yaw"].search(r).group(1)))
        if t == "THREE_HAND_CONFLICT":
            return float(_RE["n3"].search(r).group(1))
        if t == "SLIDE_JUMP_ZONE":
            return float(e.get("severity", e.get("sev", 2.5)))
    except Exception:
        return float(e.get("severity", e.get("sev", 1.0)))
    return float(e.get("severity", e.get("sev", 1.0)))

_LIM = None
def limits():
    global _LIM
    if _LIM is None:
        try:
            with open(PATH, "r", encoding="utf-8") as f:
                _LIM = json.load(f)
            check_artifact_version(_LIM, "sim_core/limits.json")
        except Exception: _LIM = {}
    return _LIM

def reload_limits():
    global _LIM
    _LIM = None
    return limits()

def split_events(events, lim=None):
    lim = limits() if lim is None else lim
    hard, soft = [], []
    for e in events:
        thr = lim.get(e["type"], {}).get("thr")
        m = measure(e)
        (hard if (thr is None or m > thr) else soft).append(dict(e, m=round(m, 4)))
    return hard, soft
