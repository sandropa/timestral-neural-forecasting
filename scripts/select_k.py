"""Choose the number K of PCA components on the VALIDATION split (never the test set).

    uv run python -u scripts/select_k.py                      # 100 neurons (seed 11), K = 8 ... 64; several hours
    options: --n 100 --seed 11 --every 4 --grid 8,12,16,24 --extend 32,48,64

Same forecaster as run_sample.py, on every 4th validation window of the 8 training conditions (taxis has no
validation split). The prompt is placed as for test: for test windows it ends `val` frames before the first test
target (the train end); for validation windows it ends `val` frames before the first validation target.
Grid: if the best K is the largest tried, the next value of --extend is added. Writes results/select_k.json with
per-neuron validation errors, and a paired bootstrap (over neurons) of every K against the best.
"""
import json, os, sys, time
import numpy as np
from common import RESULTS, TRAIN_CONDS, H, split, val_starts, prompt_frames, load_traces, pca_scores, sample_neurons, forecast

arg = lambda k, d: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
N, SEED, EVERY = int(arg("--n", 100)), int(arg("--seed", 11)), int(arg("--every", 4))
GRID = [int(v) for v in arg("--grid", "8,12,16,24").split(",")]
EXTEND = [int(v) for v in arg("--extend", "32,48,64").split(",")]
T0 = time.time()
def log(m): print(f"[{time.strftime('%H:%M:%S')} +{time.time() - T0:5.0f}s] {m}", flush=True)

X = load_traces()
Zall = pca_scores(X, max(GRID + EXTEND))
NEU = sample_neurons(N, SEED)
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
from timestral_mps import load_forecaster
F = load_forecaster()

out = RESULTS / "select_k.json"
res = json.loads(out.read_text()) if out.exists() else {}
res.update(neurons=NEU.tolist(), windows={n: val_starts(n)[::EVERY].tolist() for n in TRAIN_CONDS})
res.setdefault("K", {})


def run(K):
    if str(K) in res["K"]:
        return
    per_neuron, t = {}, time.time()
    for n in TRAIN_CONDS:
        st = val_starts(n)[::EVERY]
        te, ts, _ = split(n)
        pf = prompt_frames(n, end=te - (ts - te))                             # mirror the test geometry
        err = np.stack([np.abs(forecast(F, X, Zall[:, :K], NEU, s, pf)[0] - X[s:s + H, NEU].T).mean(1) for s in st])
        per_neuron[n] = err.mean(0)
    pn = np.mean([per_neuron[n] for n in TRAIN_CONDS], 0)
    res["K"][str(K)] = {"val_mae": float(pn.mean()), "per_neuron": pn.tolist(),
                        "per_condition": {n: float(per_neuron[n].mean()) for n in TRAIN_CONDS}, "seconds": time.time() - t}
    out.write_text(json.dumps(res, indent=1))
    log(f"K={K:2d}: validation MAE {pn.mean():.5f}  ({time.time() - t:.0f} s)")


tried = []
for K in GRID:
    run(K); tried.append(K)
for K in EXTEND:
    if min(tried, key=lambda k: res["K"][str(k)]["val_mae"]) != max(tried):
        break
    run(K); tried.append(K)

best = min(tried, key=lambda k: res["K"][str(k)]["val_mae"])
b = np.array(res["K"][str(best)]["per_neuron"]); rng = np.random.default_rng(0)
summary = {}
for K in sorted(tried):
    d = np.array(res["K"][str(K)]["per_neuron"]) - b                         # > 0: worse than the best
    lo, hi = np.percentile([d[rng.integers(0, len(d), len(d))].mean() for _ in range(5000)], [2.5, 97.5])
    summary[str(K)] = {"val_mae": res["K"][str(K)]["val_mae"], "vs_best_pct": float(d.mean() / b.mean() * 100),
                       "ci95_pct": [float(lo / b.mean() * 100), float(hi / b.mean() * 100)]}
    s = summary[str(K)]
    log(f"K={K:2d}: {s['val_mae']:.5f}  vs best {s['vs_best_pct']:+.2f}% (95% {s['ci95_pct'][0]:+.2f} to {s['ci95_pct'][1]:+.2f})")
res["summary"] = {"best": best, "vs_best": summary}
out.write_text(json.dumps(res, indent=1))
