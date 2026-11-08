"""Code version & deterministic SHA-256 hash tracking for `sim_core` artifacts.

Every downstream artifact (`sim_core/limits.json`, `sim_core/reference_official.json`,
`sim_core/body_distribution_model.json`, `sim_core/surrogate_weights.pt`, and
`out/simv3/v3_*.jsonl`) embeds `sim_core_version` and `sim_core_code_hash`.
When `sim_core` loads any artifact whose `sim_core_code_hash` does not match the current
planner / kinematics / body model code, it emits a `RuntimeWarning` instructing the user
to run `python scratch/simv3/rebuild_all.py`.
"""
from __future__ import annotations

import hashlib
import os
from typing import Any, Dict, Tuple
import warnings

SIM_CORE_VERSION: str = "3.2.0"

_HERE = os.path.dirname(os.path.abspath(__file__))

# Core source files that determine planner trajectories, body states, and violation magnitudes
HASHED_CORE_FILES: Tuple[str, ...] = (
    "config.py",
    "geometry.py",
    "kinematics.py",
    "planner.py",
    "slide_jump.py",
    "body_model.py",
    "motion.py",
    "vocab_and_rarity.py",
)


def get_sim_core_code_hash() -> str:
    """Compute a deterministic 16-hex-char SHA-256 digest over `HASHED_CORE_FILES`."""
    h = hashlib.sha256()
    h.update(f"version:{SIM_CORE_VERSION}\n".encode("utf-8"))
    for rel in HASHED_CORE_FILES:
        p = os.path.join(_HERE, rel)
        h.update(f"file:{rel}\n".encode("utf-8"))
        if os.path.exists(p):
            with open(p, "rb") as f:
                raw = f.read().replace(b"\r\n", b"\n")
            h.update(raw)
    return h.hexdigest()[:16]


def stamp_artifact_version(d: Dict[str, Any]) -> Dict[str, Any]:
    """Embed `sim_core_version` and `sim_core_code_hash` into an artifact dictionary."""
    code_hash = get_sim_core_code_hash()
    meta = {
        "sim_core_version": SIM_CORE_VERSION,
        "sim_core_code_hash": code_hash,
    }
    d["_meta"] = meta
    d["sim_core_version"] = SIM_CORE_VERSION
    d["sim_core_code_hash"] = code_hash
    return d


def extract_artifact_hash(d: Any) -> str:
    """Extract the embedded `sim_core_code_hash` from a loaded artifact dict."""
    if not isinstance(d, dict):
        return ""
    if isinstance(d.get("sim_core_code_hash"), str) and d["sim_core_code_hash"]:
        return str(d["sim_core_code_hash"])
    meta = d.get("_meta")
    if isinstance(meta, dict) and isinstance(meta.get("sim_core_code_hash"), str):
        return str(meta["sim_core_code_hash"])
    return ""


def check_artifact_version(d: Any, artifact_name: str) -> bool:
    """Verify that loaded artifact `d` matches `get_sim_core_code_hash()`.
    Emits a `RuntimeWarning` and returns `False` if missing or mismatched.
    """
    cur_hash = get_sim_core_code_hash()
    art_hash = extract_artifact_hash(d)
    if art_hash != cur_hash:
        warnings.warn(
            f"[sim_core] Artifact '{artifact_name}' has sim_core_code_hash={art_hash!r} "
            f"!= current code hash {cur_hash!r} (version {SIM_CORE_VERSION}). "
            f"Please run `python scratch/simv3/rebuild_all.py` to synchronize all downstream artifacts.",
            RuntimeWarning,
            stacklevel=2,
        )
        return False
    return True
