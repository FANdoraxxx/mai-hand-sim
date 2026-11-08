"""python -m sim_core.test_project_api  (run from project root)"""
import glob, os, sys, time
sys.path.insert(0, os.getcwd())
from mai.simai import read_maidata, load_song
from sim_core.project_api import evaluate_notes, per_bar, window_score, BAR_TICKS
from mai import simai

def main(song="official charts/INFINITE WORLD"):
    meta, charts = load_song(song)
    notes, tm = charts[5]
    tick, _ = simai.quantize(notes)
    t = time.time(); res = evaluate_notes(tick, tm, 5); dt = time.time() - t
    pb = per_bar(res)
    whole_sev = sum(r["sev"] for r in pb)
    nbar = max(int(n.beat) for n in tick) // BAR_TICKS + 1
    for ctx in (0, 2, 4):
        tot = 0.0; nh = 0; t = time.time()
        for lo in range(0, nbar, 4):
            w = window_score(tick, tm, lo, min(nbar, lo + 4), ctx_bars=ctx)
            tot += w["sev"]; nh += w["n_hard"]
        print("ctx", ctx, "windows sev sum", round(tot, 2), "n_hard", nh, "| whole sev", round(whole_sev, 2), "n_ev", len(res.unreasonable_events), "t", round(time.time() - t, 2), "whole t", round(dt, 2))
if __name__ == "__main__":
    main(*sys.argv[1:])
