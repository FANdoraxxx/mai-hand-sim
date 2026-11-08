"""Project-facing API for sim_core (maiAIV3, upgraded in simv2).

Provides:
  - `evaluate_notes(notes, tm, slot=5, ...)`: zero-regex direct evaluation of `mai.simai` tick notes + `TempoMap`.
  - `per_bar(result)`: per-bar cost, violations, fatigue, and choreography summary rows.
  - `init_window_state(tm, slot=5, ...)`: initialize a serializable `PlannerState` for incremental window scoring.
  - `window_score(...)`: supports BOTH:
      1) Stateful incremental API (simv2):
           `violations, cost, motion_features, new_state = window_score(history_state, window_notes, tm=tm, ...)`
         where `history_state` is `None`, `PlannerState`, or `dict` (`PlannerState.to_dict()`), and the return value
         both unpacks as a 4-tuple `(violations, cost, motion_features, new_state)` AND supports dict indexing!
      2) Legacy context-slice API (simval backward-compatible):
           `w = window_score(tick_notes, tm, bar_lo=16, bar_hi=20, ctx_bars=2, ...)`
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Sequence, Union

from sim_core.fast_simai import TPB, build_parsed_chart_from_notes
from sim_core.limits import split_events
from sim_core.motion import compute_motion_features
from sim_core.planner import PlannerState, plan_chart_movements
from sim_core.scorer import evaluate_chart, compute_rl_reward
from sim_core.body_model import BodyTimeline, body_timeline
from sim_core.vocab_and_rarity import window_body_score

BAR_TICKS = 4 * TPB


def _TM(changes, first: float = 0.0):
    from mai.simai import TempoMap
    return TempoMap(list(changes), first=first)


class WindowScoreResult(tuple):
    """4-tuple `(violations, cost, motion_features, new_state)` that also supports dict-key access."""

    def __new__(cls, violations: List[Dict[str, Any]], cost: float, motion_features: Dict[str, Any], new_state: PlannerState, extra: Optional[Dict[str, Any]] = None):
        obj = super().__new__(cls, (violations, cost, motion_features, new_state))
        d = {
            "violations": violations,
            "hard": [v for v in violations if v.get("is_hard", True)],
            "soft": [v for v in violations if not v.get("is_hard", True)],
            "cost": cost,
            "motion": motion_features,
            "motion_features": motion_features,
            "state": new_state,
            "new_state": new_state,
        }
        if extra:
            d.update(extra)
        d.setdefault("n_hard", len(d["hard"]))
        d.setdefault("sev", sum(float(v.get("severity", 1.0)) for v in violations))
        d.setdefault("hard_sev", sum(float(v.get("severity", 1.0)) for v in d["hard"]))
        object.__setattr__(obj, "_extra_dict", d)
        return obj

    def __getitem__(self, key):  # type: ignore[override]
        if isinstance(key, str):
            return object.__getattribute__(self, "_extra_dict")[key]
        return super().__getitem__(key)

    def __contains__(self, key) -> bool:  # type: ignore[override]
        if isinstance(key, str) and key in object.__getattribute__(self, "_extra_dict"):
            return True
        return super().__contains__(key)

    def get(self, key: str, default: Any = None) -> Any:
        return object.__getattribute__(self, "_extra_dict").get(key, default)

    def keys(self):
        return object.__getattribute__(self, "_extra_dict").keys()

    def to_dict(self) -> Dict[str, Any]:
        d = dict(object.__getattribute__(self, "_extra_dict"))
        if isinstance(d.get("state"), PlannerState):
            d["state"] = d["state"].to_dict()
        if isinstance(d.get("new_state"), PlannerState):
            d["new_state"] = d["new_state"].to_dict()
        return d

    @property
    def violations(self) -> List[Dict[str, Any]]:
        return super().__getitem__(0)

    @property
    def cost(self) -> float:
        return super().__getitem__(1)

    @property
    def motion_features(self) -> Dict[str, Any]:
        return super().__getitem__(2)

    @property
    def new_state(self) -> PlannerState:
        return super().__getitem__(3)


def init_window_state(
    tm=None,
    slot: int = 5,
    hit_mode: str = "hybrid",
    skill_profile: str = "Lv14+_Master",
    source_path: Optional[str] = None,
    title: str = "",
) -> PlannerState:
    """Create an initial serializable `PlannerState` at bar 0 for stateful `window_score`."""
    changes = tuple((float(b), float(p)) for b, p in tm.changes()) if tm is not None else ((0.0, 150.0),)
    first_sec = float(getattr(tm, "first", 0.0)) if tm is not None else 0.0
    return PlannerState(
        end_time=-999.0,
        end_bar=0,
        hit_mode=hit_mode,
        skill_profile=skill_profile,
        tm_changes=changes,
        first_offset_sec=first_sec,
        slot=slot,
        source_path=source_path,
        title=title,
    )


def evaluate_notes(notes, tm, slot: int = 5, hit_mode: str = "hybrid", skill_profile: str = "Lv14+_Master",
                   frame_fps: int = 20, include_motion: bool = True, source_path: Optional[str] = None,
                   title: str = "TickNotesChart", mel: Any = None):
    """Evaluate tick-unit `mai.simai` notes + `TempoMap` directly (zero regex overhead)."""
    pc = build_parsed_chart_from_notes(
        notes, tm, slot=slot, title=title, unit_ticks=True, source_path=source_path,
        include_first_offset=bool(source_path or mel is not None or title not in ("TickNotesChart", "InlineSimaiChart")),
    )
    return evaluate_chart(
        pc, difficulty=slot, hit_mode=hit_mode, skill_profile=skill_profile,
        frame_fps=frame_fps, include_motion=include_motion, mel=mel,
    )


def per_bar(result) -> List[Dict[str, Any]]:
    """Per-bar table: fatigue, #events (hard+soft), severity sums (all / hard), events by type, plus motion row."""
    nb = len(result.fatigue_curve)
    mot_bars = {r["bar"]: r for r in (getattr(result, "motion", {}) or {}).get("per_bar", ())}
    rows = [dict(bar=i, fatigue=result.fatigue_curve[i], n_events=0, n_hard=0, sev=0.0, hard_sev=0.0, types={}) for i in range(nb)]
    for hard, lst in ((True, result.unreasonable_events), (False, getattr(result, "soft_events", []))):
        for e in lst:
            b = int(e.get("bar", 0))
            while b >= len(rows):
                rows.append(dict(bar=len(rows), fatigue=0.0, n_events=0, n_hard=0, sev=0.0, hard_sev=0.0, types={}))
            r = rows[b]
            sv = float(e.get("severity", 1.0))
            r["n_events"] += 1; r["sev"] += sv
            if hard: r["n_hard"] += 1; r["hard_sev"] += sv
            r["types"][e["type"]] = r["types"].get(e["type"], 0) + 1
    for r in rows:
        r["cost"] = round(r["sev"] + 0.02 * r["fatigue"], 4)
        if r["bar"] in mot_bars:
            r["motion"] = mot_bars[r["bar"]]
    return rows


def _score_stateful_window(
    history_state: Optional[Union[PlannerState, Dict[str, Any]]],
    window_notes: Sequence,
    tm=None,
    bar_lo: Optional[int] = None,
    bar_hi: Optional[int] = None,
    slot: Optional[int] = None,
    hit_mode: Optional[str] = None,
    skill_profile: Optional[str] = None,
    frame_fps: int = 20,
    include_motion: bool = True,
    mel: Any = None,
    beam_width: int = 6,
) -> WindowScoreResult:
    """Fast incremental window scoring from serializable `PlannerState`."""
    if isinstance(history_state, dict):
        st = PlannerState.from_dict(history_state)
    elif isinstance(history_state, PlannerState):
        st = history_state
    else:
        st = init_window_state(
            tm=tm,
            slot=slot or 5,
            hit_mode=hit_mode or "hybrid",
            skill_profile=skill_profile or "Lv14+_Master",
        )

    eff_slot = slot if slot is not None else st.slot
    eff_mode = hit_mode if hit_mode is not None else st.hit_mode
    eff_prof = skill_profile if skill_profile is not None else st.skill_profile
    if tm is None:
        tm = _TM(st.tm_changes, first=st.first_offset_sec)

    if bar_lo is None:
        bar_lo = (min(int(n.beat) for n in window_notes) // BAR_TICKS) if window_notes else st.end_bar
    if bar_hi is None:
        bar_hi = ((max(int(n.beat) for n in window_notes) // BAR_TICKS) + 1) if window_notes else (bar_lo + 4)

    use_first = bool(st.source_path or mel is not None or st.first_offset_sec > 0.0)
    offset_sec = float(getattr(tm, "first", 0.0)) if use_first else 0.0
    t0_sec = offset_sec + float(tm.to_sec(4.0 * bar_lo))
    t1_sec = offset_sec + float(tm.to_sec(4.0 * bar_hi))

    w_notes = [n for n in window_notes if bar_lo * BAR_TICKS <= int(n.beat) < bar_hi * BAR_TICKS]
    pc = build_parsed_chart_from_notes(
        w_notes, tm, slot=eff_slot, title=st.title or "WindowChart",
        unit_ticks=True, source_path=st.source_path, include_first_offset=use_first,
    )
    pc.bar_timestamps = [
        (b, offset_sec + float(tm.to_sec(4.0 * b)), offset_sec + float(tm.to_sec(4.0 * (b + 1))))
        for b in range(bar_lo, bar_hi)
    ]

    plan = plan_chart_movements(
        chart=pc,
        hit_mode=eff_mode,
        skill_profile=eff_prof,
        beam_width=beam_width,
        frame_fps=max(20, frame_fps),
        initial_state=st,
        t_range=(t0_sec, t1_sec),
        skip_trajectory=(not include_motion),
    )

    raw_evs = [u.to_dict() for u in plan.unreasonable_events if bar_lo <= int(u.bar) < bar_hi]
    hard_evs, soft_evs = split_events(raw_evs)
    violations: List[Dict[str, Any]] = [
        dict(e, is_hard=True) for e in hard_evs
    ] + [
        dict(e, is_hard=False) for e in soft_evs
    ]

    pb_rows: List[Dict[str, Any]] = []
    steps_by_b: Dict[int, List] = {}
    for s_step in plan.planned_steps:
        steps_by_b.setdefault(s_step.bar, []).append(s_step)

    for b in range(bar_lo, bar_hi):
        bst = steps_by_b.get(b, [])
        fat_b = (
            round(sum(x.step_fatigue_left + x.step_fatigue_right + x.step_fatigue_torso for x in bst) / len(bst), 3)
            if bst else 0.0
        )
        b_hard = [e for e in hard_evs if int(e.get("bar", -1)) == b]
        b_soft = [e for e in soft_evs if int(e.get("bar", -1)) == b]
        sv_all = sum(float(e.get("severity", 1.0)) for e in b_hard) + sum(float(e.get("severity", 1.0)) for e in b_soft)
        sv_h = sum(float(e.get("severity", 1.0)) for e in b_hard)
        t_cnt: Dict[str, int] = {}
        for e in b_hard + b_soft:
            t_cnt[e["type"]] = t_cnt.get(e["type"], 0) + 1
        pb_rows.append(
            dict(
                bar=b,
                fatigue=fat_b,
                n_events=len(b_hard) + len(b_soft),
                n_hard=len(b_hard),
                sev=sv_all,
                hard_sev=sv_h,
                types=t_cnt,
                cost=round(sv_all + 0.02 * fat_b, 4),
            )
        )

    new_state = plan.final_state
    assert new_state is not None
    new_state.end_bar = bar_hi

    if include_motion:
        motion_feats, upd_sigs, upd_fat = compute_motion_features(
            plan=plan,
            slot=eff_slot,
            mel_override=mel,
            prior_bar_sigs=st.prior_bar_sigs,
            initial_cum_fatigue=st.cum_fatigue,
        )
        new_state.prior_bar_sigs = upd_sigs
        new_state.cum_fatigue = upd_fat
    else:
        motion_feats = {}

    tot_cost = round(sum(r["cost"] for r in pb_rows), 4)
    extra = {
        "hard": hard_evs,
        "soft": soft_evs,
        "n_hard": len(hard_evs),
        "sev": round(sum(r["sev"] for r in pb_rows), 4),
        "hard_sev": round(sum(r["hard_sev"] for r in pb_rows), 4),
        "fatigue": round(sum(r["fatigue"] for r in pb_rows), 4),
        "per_bar": pb_rows,
    }
    return WindowScoreResult(violations, tot_cost, motion_feats, new_state, extra=extra)


def window_score(
    arg0,
    arg1=None,
    bar_lo: Optional[int] = None,
    bar_hi: Optional[int] = None,
    ctx_bars: int = 2,
    slot: Optional[int] = None,
    hit_mode: Optional[str] = None,
    skill_profile: Optional[str] = None,
    frame_fps: int = 20,
    tm=None,
    include_motion: bool = True,
    mel: Any = None,
):
    """Dual-mode `window_score`:
      1) Stateful incremental mode (simv2):
           `violations, cost, motion_features, new_state = window_score(history_state, window_notes, tm=..., ...)`
      2) Legacy slice mode (simval compatible):
           `w = window_score(all_tick_notes, tm, bar_lo=16, bar_hi=20, ctx_bars=2, ...)`
    """
    is_stateful = (
        arg0 is None
        or isinstance(arg0, (PlannerState, dict))
    )
    if is_stateful:
        return _score_stateful_window(
            history_state=arg0,
            window_notes=arg1 or (),
            tm=tm,
            bar_lo=bar_lo,
            bar_hi=bar_hi,
            slot=slot,
            hit_mode=hit_mode,
            skill_profile=skill_profile,
            frame_fps=frame_fps,
            include_motion=include_motion,
            mel=mel,
        )

    notes = arg0
    tm_legacy = arg1
    l_slot = slot if slot is not None else 5
    l_mode = hit_mode if hit_mode is not None else "hybrid"
    l_prof = skill_profile if skill_profile is not None else "Lv14+_Master"
    assert bar_lo is not None and bar_hi is not None, "Legacy window_score(notes, tm, bar_lo, bar_hi) requires bar_lo and bar_hi."
    c0 = max(0, bar_lo - ctx_bars)
    st0 = init_window_state(tm=tm_legacy, slot=l_slot, hit_mode=l_mode, skill_profile=l_prof)
    if c0 < bar_lo:
        ctx_notes = [n for n in notes if c0 * BAR_TICKS <= int(n.beat) < bar_lo * BAR_TICKS]
        if ctx_notes:
            _, _, _, st0 = _score_stateful_window(
                st0, ctx_notes, tm=tm_legacy, bar_lo=c0, bar_hi=bar_lo,
                slot=l_slot, hit_mode=l_mode, skill_profile=l_prof,
                frame_fps=frame_fps, include_motion=False,
            )
    win_notes = [n for n in notes if bar_lo * BAR_TICKS <= int(n.beat) < bar_hi * BAR_TICKS]
    res = _score_stateful_window(
        st0, win_notes, tm=tm_legacy, bar_lo=bar_lo, bar_hi=bar_hi,
        slot=l_slot, hit_mode=l_mode, skill_profile=l_prof,
        frame_fps=frame_fps, include_motion=include_motion, mel=mel,
    )
    return res
