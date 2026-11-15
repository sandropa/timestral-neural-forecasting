"""ZAPBench-style linear baseline: each neuron's next 32 frames as one shared linear function of its last 256 frames.

    uv run python -u scripts/fit_linear_baseline.py      # CPU, ~1 min; writes results/linear_baseline.npz

One weight matrix W (257 x 32: 256 lags + bias) shared by all neurons, fitted by least squares on windows that lie
entirely inside the training splits of the 8 training conditions (every 2nd window, all 71,721 neurons).
Forecast for a window starting at s:  X[s-256:s, i] @ W[:-1] + W[-1].
"""
import numpy as np
from common import RESULTS, NAMES, HOLDOUT, C, H, bounds, split, load_traces

STRIDE = 2
X = load_traces()
N = X.shape[1]
AtA = np.zeros((C + 1, C + 1)); AtY = np.zeros((C + 1, H))
for n in NAMES:
    if n == HOLDOUT:
        continue
    a, te = bounds(n)[0], split(n)[0]                                        # windows inside the training split
    for s in range(a + C, te - H + 1, STRIDE):
        A = np.empty((N, C + 1), np.float32); A[:, :C] = X[s - C:s].T; A[:, C] = 1
        AtA += (A.T @ A).astype(np.float64); AtY += (A.T @ X[s:s + H].T).astype(np.float64)
W = np.linalg.solve(AtA + 1e-6 * np.trace(AtA) / (C + 1) * np.eye(C + 1), AtY).astype(np.float32)
RESULTS.mkdir(exist_ok=True)
np.savez(RESULTS / "linear_baseline.npz", W=W)
print("wrote results/linear_baseline.npz")
