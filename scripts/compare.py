"""Score a run of run_sample.py against the published ZAPBench models on exactly the same neurons and windows.

    uv run python -u scripts/compare.py --run sample1000_seed13_K32

Published models: their released test-set forecasts (download_predictions.py) in data/preds/<name>/seed<k>/;
every seed found is used and averaged. Recomputed baselines: linear (fit_linear_baseline.py) and the mean of the
last 4 frames. MAE per step, averaged over neurons and windows; "avg 1-32" = mean over steps 1-32 and over the 8
training conditions; taxis = the held-out condition. Writes results/<run>/compare.json.
"""
import json, sys
import numpy as np
import zarr
from common import DATA, RESULTS, NAMES, TRAIN_CONDS, ZARR_NAME, H, load_traces

run = sys.argv[sys.argv.index("--run") + 1]
RUN = RESULTS / run
MODELS = {"UNet": "unet", "TSMixer": "tsmixer", "Time-Mix": "time-mix", "TiDE": "tide"}
X = load_traces()

curves, per_neuron, cover = {}, {}, {}
conds = [n for n in NAMES if (RUN / f"{n.replace(' ', '_')}.npz").exists()]
for n in conds:
    d = np.load(RUN / f"{n.replace(' ', '_')}.npz")
    Y, widx, neu, st = d["truth"], d["window_index"], d["neurons"], d["starts"]
    P = {"ours": d["med"], "linear": d["linear"],
         "mean of last 4 frames": np.stack([X[s - 4:s, neu].mean(0)[:, None].repeat(H, 1) for s in st])}
    for m, folder in MODELS.items():
        seeds = sorted((DATA / "preds" / folder).glob("seed*"))
        if seeds:
            P[m] = np.mean([np.asarray(zarr.open_array(store=str(sd / ZARR_NAME[n]), mode="r", zarr_format=2)
                                       .get_orthogonal_selection((widx, slice(None), neu))).transpose(0, 2, 1) for sd in seeds], 0)
    for k, v in P.items():
        e = np.abs(v - Y)
        curves.setdefault(k, {})[n] = e.mean((0, 1)); per_neuron.setdefault(k, {})[n] = e.mean((0, 2))
    cover[n] = {lvl: float(np.mean((Y >= d[lo]) & (Y <= d[hi]))) for lvl, lo, hi in (("50%", "q25", "q75"), ("80%", "q10", "q90"), ("90%", "q05", "q95"))}

tr = [n for n in TRAIN_CONDS if n in conds]
table = {}
for k in curves:
    c8 = np.mean([curves[k][n] for n in tr], 0)
    table[k] = {"step1": float(c8[0]), "step8": float(c8[7]), "step32": float(c8[-1]), "avg1_32": float(c8.mean()),
                "taxis": float(curves[k]["taxis"].mean()) if "taxis" in conds else None,
                "per_condition": {n: float(curves[k][n].mean()) for n in conds}}
out = {"run": run, "conditions": conds, "table": table, "interval_coverage": cover}
if "UNet" in per_neuron:                                                   # ours vs UNet, bootstrap over neurons
    o = np.mean([per_neuron["ours"][n] for n in tr], 0); u = np.mean([per_neuron["UNet"][n] for n in tr], 0)
    rng = np.random.default_rng(0); idx = rng.integers(0, len(o), (10000, len(o)))
    rel = (o[idx].mean(1) / u[idx].mean(1) - 1) * 100
    out["ours_vs_unet"] = {"pct": float((o.mean() / u.mean() - 1) * 100), "ci95_pct": np.percentile(rel, [2.5, 97.5]).tolist(),
                           "neurons_where_ours_better": float(np.mean(o < u)), "n_neurons": int(len(o))}
(RUN / "compare.json").write_text(json.dumps(out, indent=1))

print(f"{run}: {len(o) if 'UNet' in per_neuron else ''} neurons, conditions: {', '.join(conds)}")
print(f"{'method':24s} {'step 1':>8s} {'step 8':>8s} {'step 32':>8s} {'avg 1-32':>9s} {'taxis':>8s}")
for k, v in sorted(table.items(), key=lambda kv: kv[1]["avg1_32"]):
    tx = f"{v['taxis']:8.5f}" if v["taxis"] is not None else "       -"
    print(f"{k:24s} {v['step1']:8.5f} {v['step8']:8.5f} {v['step32']:8.5f} {v['avg1_32']:9.5f} {tx}")
if "ours_vs_unet" in out:
    r = out["ours_vs_unet"]
    print(f"ours vs UNet: {r['pct']:+.2f}% (95% {r['ci95_pct'][0]:+.2f}% to {r['ci95_pct'][1]:+.2f}%), "
          f"ours better on {r['neurons_where_ours_better']:.0%} of neurons")
print("interval coverage (nominal 50/80/90%):", {n: c for n, c in cover.items()})
