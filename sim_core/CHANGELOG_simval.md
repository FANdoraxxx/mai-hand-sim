# simval changelog (2026-11-07)
* planner.py: HAND_COLLISION reason text now says "Hand/forearm collision" - the flag is `palm collision OR forearm-capsule collision` (planner.py:151), the old text printed palm distance only and looked self-contradictory (164.6 mm < 85 mm). Cosmetic only; behaviour unchanged.
* NEW project_api.py (evaluate_notes, per_bar, window_score), calibration.py + reference_official.json, test_project_api.py. Public API of scorer.py untouched.
* NOT changed (reported instead): hard-event thresholds/profiles. Findings: 95% of official MiLK+ 12-15 charts are flagged (see agent_tasks/simval.result.md); touch-stack B5+E5 flagged as HAND_COLLISION although one palm covers both; profile limits (e.g. Lv12 fly limit 2800 mm/s) are applied to charts that official players clear. A retune needs the owner's design call; calibration.py makes it unnecessary for ranking.

## round 2
* geometry.py: `_find_single_palm_cover_cached` - sorted zone tuples (cache hits across permutations), lru 4096->131072, and an exact necessary-condition reject (zone distance minus bounding radii > 2R). Cuts the worst official chart 34.9 s -> 1.3 s with the same events (cross 5->3 on that chart from the budget, see below).
* planner.py: HAND_COLLISION split into HAND_COLLISION (palm) and new ARM_COLLISION (forearm capsules); events carry measurements in `reason`; simultaneous hits by two hands <=95 mm apart are treated as one-palm stack (no flag, no 4000 cost); `time_budget_sec=2.5` adaptive beam width (non-deterministic only for charts that would otherwise exceed the budget).
* NEW limits.py + limits.json: fitted hard/soft split (scorer: `unreasonable_events` = hard only, `soft_events` new key; is_playable = no hard events; soft events cost 2*sev in the ergonomic score instead of 28*sev). Delete limits.json to restore legacy all-hard behaviour.
* calibration.py: per-event-type severity densities (t_<TYPE>) added to the reference table; reference rebuilt from r2_official.jsonl (new code, 1187 charts).
* project_api.py: per_bar counts hard+soft with separate hard counts.
