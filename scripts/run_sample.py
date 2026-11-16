"""Forecast a random sample of neurons on the official ZAPBench test windows (zero-shot Timestral + training prompt).

    uv run python -u scripts/run_sample.py --n 1000 --seed 13 --K 32
    options: --stride 8          every 8th test window (default; neighbouring windows overlap in 255 of 256 frames)
             --exclude DIR,...   leave out neurons used in earlier runs (their results/<run>/ folders)
             --neurons 1,2,3     forecast exactly these neurons instead of a random sample
             --conditions flash  only these conditions; --max-windows 2: only the first windows (quick checks)

Method (per neuron, per window): Timestral-209M forecasts the neuron jointly with the top-K PCA scores of the whole
brain. The input is the condition's training prompt (last <= 768 training frames), 32 NaN frames, then the official
256-frame context; taxis (held out, no training data) gets the 256 frames only. No training of any kind.
Neurons without a released UNet forecast (data/unet_missing_neurons.json, 27 neurons) are never sampled.

Saves results/<run>/<condition>.npz (truth, linear baseline, median, quantiles 5/10/25/75/90/95%); resumable.
"""
import json, os, sys, time
import numpy as np
from common import DATA, RESULTS, NAMES, H, C, test_starts, prompt_frames, load_traces, pca_scores, sample_neurons, forecast

arg = lambda k, d=None: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
K, STRIDE = int(arg("--K", 8)), int(arg("--stride", 8))
CONDS = arg("--conditions", ",".join(NAMES)).split(",")
MAXW = int(arg("--max-windows", 10 ** 9))
QS = {"q05": 0.05, "q10": 0.1, "q25": 0.25, "q75": 0.75, "q90": 0.9, "q95": 0.95}

X = load_traces()
if arg("--neurons"):
    neurons = np.array(sorted(int(v) for v in arg("--neurons").split(",")))
    run = arg("--out", f"neurons_K{K}")
else:
    n, seed = int(arg("--n", 1000)), int(arg("--seed", 13))
    exclude = list(json.loads((DATA / "unet_missing_neurons.json").read_text()))
    for d in filter(None, arg("--exclude", "").split(",")):
        exclude += np.load(sorted((RESULTS / d).glob("*.npz"))[0])["neurons"].tolist()
    neurons = sample_neurons(n, seed, exclude)
    run = arg("--out", f"sample{n}_seed{seed}_K{K}")
OUT = RESULTS / run; OUT.mkdir(parents=True, exist_ok=True)
T0 = time.time()
def log(m):
    line = f"[{time.strftime('%H:%M:%S')} +{time.time() - T0:6.0f}s] {m}"
    print(line, flush=True); open(OUT / "log.txt", "a").write(line + "\n")

Z = pca_scores(X, K)
W = np.load(RESULTS / "linear_baseline.npz")["W"]
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
from timestral_mps import load_forecaster
F = load_forecaster()
log(f"{run}: {len(neurons)} neurons, K={K}, every {STRIDE}th test window")

for cond in CONDS:
    f = cond.replace(" ", "_"); path = OUT / f"{f}.npz"
    if path.exists():
        continue
    widx = np.arange(len(test_starts(cond)))[::STRIDE][:MAXW]; st = test_starts(cond)[widx]
    Y = np.stack([X[s:s + H, neurons].T for s in st])                      # windows x neurons x 32
    lin = np.stack([X[s - C:s, neurons].T @ W[:-1] + W[-1] for s in st])
    part = OUT / f"{f}.partial.npz"
    if part.exists():
        d = np.load(part); med, Q, done = d["med"], {k: d[k] for k in QS}, int(d["done"])
    else:
        med = np.full(Y.shape, np.nan, np.float32); Q = {k: med.copy() for k in QS}; done = 0
    pf, t = prompt_frames(cond), time.time()
    for w in range(done, len(st)):
        m, q = forecast(F, X, Z, neurons, st[w], pf, quantiles=tuple(QS.values()))
        med[w] = m
        for k, v in QS.items():
            Q[k][w] = q[v]
        if True:                                                           # save progress after every window (safe to interrupt)
            np.savez(part, med=med, done=w + 1, **Q)
            log(f"{cond:9s} window {w + 1:3d}/{len(st)} ({(time.time() - t) / (w + 1 - done):.0f} s per window)")
    np.savez(path, window_index=widx, starts=st, neurons=neurons, truth=Y, linear=lin, med=med, **Q)
    part.unlink(missing_ok=True)
    log(f"{cond:9s} done: ours {np.abs(med - Y).mean():.5f}, linear {np.abs(lin - Y).mean():.5f}, "
        f"80% interval coverage {np.mean((Y >= Q['q10']) & (Y <= Q['q90'])):.3f}")
log("done")
