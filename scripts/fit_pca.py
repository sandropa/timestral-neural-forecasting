"""Principal components of all 71,721 neurons, fit on the training splits of the 8 training conditions only.

    uv run python -u scripts/fit_pca.py      # CPU, a few minutes; writes results/pca.npz (mu, V: 71,721 x 256, evals)

The top-K component scores (X - mu) @ V[:, :K] are the "brain context" given to the forecaster.
"""
import numpy as np
from common import RESULTS, TRAIN_CONDS, bounds, split, load_traces

KMAX = 256
X = load_traces()
train = np.concatenate([np.arange(bounds(n)[0], split(n)[0]) for n in TRAIN_CONDS])
mu = X[train].mean(0)
Xc = X[train] - mu
w, U = np.linalg.eigh(Xc.astype(np.float64) @ Xc.T.astype(np.float64))     # frames x frames (small side)
w, U = w[::-1], U[:, ::-1]
V = ((Xc.T @ U[:, :KMAX].astype(np.float32)) / np.sqrt(w[:KMAX]).astype(np.float32)).astype(np.float32)
RESULTS.mkdir(exist_ok=True)
np.savez(RESULTS / "pca.npz", mu=mu, V=V, evals=w, train_frames=train)
print(f"PCA on {len(train)} training frames; top 8 explain {w[:8].sum() / w.sum():.1%} of the variance")
