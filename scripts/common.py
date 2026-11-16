"""Shared definitions: ZAPBench splits, data loading, and the forecaster's input.

ZAPBench (Lueckmann et al., ICLR 2025): 71,721 neurons x 7,879 frames (0.914 s per frame) of larval zebrafish
whole-brain activity under 9 stimulus conditions. Each condition is split in time into train / validation (10%) /
test (20%). Task: forecast the next 32 frames of every neuron from the last 256 frames. Taxis is held out entirely
(test only, no training or validation data).
"""
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA, RESULTS = ROOT / "data", ROOT / "results"
OFF = (0, 649, 2422, 3078, 3735, 5047, 5638, 6623, 7279, 7879)            # condition boundaries (frames)
NAMES = ("gain", "dots", "flash", "taxis", "turning", "position", "open loop", "rotation", "dark")
HOLDOUT = "taxis"
TRAIN_CONDS = [n for n in NAMES if n != HOLDOUT]
ZARR_NAME = {n: (f"test_holdout_condition_{i}" if n == HOLDOUT else f"test_condition_{i}") for i, n in enumerate(NAMES)}
C, H, DT = 256, 32, 0.914                                                  # context, horizon, seconds per frame
PROMPT_MAX, GAP = 768, 32                                                  # training prompt length, NaN gap


def bounds(n):
    """First and last+1 usable frame of condition n (as in zapbench data_utils)."""
    c = NAMES.index(n)
    return OFF[c] + 1, OFF[c + 1] - 1


def split(n):
    """(train end, test start, condition end); validation = [train end, test start)."""
    a, b = bounds(n)
    tot = b - a
    test, val = int(tot * 0.2), int(tot * 0.1)
    return a + tot - test - val, a + tot - test, b


def test_starts(n):
    """First forecast frame of every official test window (taxis: all frames after the first 256)."""
    a, b = bounds(n)
    return np.arange(a + C if n == HOLDOUT else split(n)[1], b - H + 1)


def val_starts(n):
    """First forecast frame of every validation window (targets inside the validation split)."""
    te, ts, _ = split(n)
    return np.arange(te, ts - H + 1)


def prompt_frames(n, end=None):
    """The training prompt: the last <= 768 training frames of condition n, ending at `end` (default: train end).
    Taxis has no training data, so it gets no prompt (run_sample.py instead uses the condition recognised by
    ConditionMatcher, which is never taxis)."""
    if n == HOLDOUT:
        return np.array([], int)
    end = split(n)[0] if end is None else end
    return np.arange(max(bounds(n)[0], end - PROMPT_MAX), end)


def load_traces():
    import zarr
    from zarr.storage import ZipStore
    return np.asarray(zarr.open_array(ZipStore(str(DATA / "traces.zip"), mode="r"), path="traces", mode="r")[:])


def pca_scores(X, K):
    """Scores of the top-K principal components (fit on training frames, see fit_pca.py) for every frame."""
    p = np.load(RESULTS / "pca.npz")
    return ((X - p["mu"].astype(np.float32)) @ p["V"][:, :K].astype(np.float32)).astype(np.float32)


class ConditionMatcher:
    """Recognise the stimulus condition of a window from brain activity alone (no labels, no stimulus).

    The window's 256 frames of the top-48 PCA scores, standardised with training statistics, are compared with each
    training condition's training frames: score = mean over the window's frames of the distance to the nearest training
    frame of that condition. The condition with the lowest score is chosen; its training data becomes the prompt. On
    ZAPBench this picks the true condition for every validation and test window of the 8 training conditions; windows
    of the held-out taxis condition are matched to their most similar known condition (flash).
    """

    def __init__(self, X, K=48):
        Z = pca_scores(X, K)
        self.frames = {n: np.arange(bounds(n)[0], split(n)[0]) for n in TRAIN_CONDS}
        allt = np.concatenate(list(self.frames.values()))
        self.Zs = ((Z - Z[allt].mean(0)) / Z[allt].std(0)).astype(np.float32)
        self.banks = {n: self.Zs[f] for n, f in self.frames.items()}
        self.sq = {n: (b ** 2).sum(1) for n, b in self.banks.items()}

    def __call__(self, s):
        """The training condition whose training activity best matches frames [s-256, s)."""
        w = self.Zs[s - C:s]; w2 = (w ** 2).sum(1)[:, None]
        score = {n: np.sqrt(np.maximum(w2 + self.sq[n][None] - 2 * w @ b.T, 0)).min(1).mean() for n, b in self.banks.items()}
        return min(score, key=score.get)


def sample_neurons(n, seed, exclude=()):
    """n random neurons (sorted), drawn with numpy's default_rng(seed) from all neurons except `exclude`."""
    pool = np.setdiff1d(np.arange(71721), np.asarray(exclude, int))
    return np.sort(np.random.default_rng(seed).choice(pool, n, replace=False))


def forecast(F, X, Z, neurons, s, prompt, quantiles=(), max_rows=900):
    """Forecast frames s..s+31 of `neurons` with Timestral.

    Each neuron is its own multivariate series: the neuron's trace plus the K PCA scores (Z: frames x K), forecast
    jointly; only the neuron's forecast is kept. Input frames = prompt, 32 NaN frames, then the official 256-frame
    window [s-256, s). Returns the median (N, 32) and a dict quantile -> (N, 32).
    """
    K = Z.shape[1]
    frames = np.concatenate([prompt, np.full(GAP, -1), np.arange(s - C, s)]) if len(prompt) else np.arange(s - C, s)
    ok = frames >= 0
    step = max(1, max_rows // (K + 1))                                     # keep a call within GPU memory
    med = np.empty((len(neurons), H), np.float32)
    qs = {q: np.empty((len(neurons), H), np.float32) for q in quantiles}
    for i in range(0, len(neurons), step):
        ids = neurons[i:i + step]
        ctx = np.full((len(ids), 1 + K, len(frames)), np.nan, np.float32)  # NaN = missing (the gap)
        ctx[:, 0, ok] = X[frames[ok]][:, ids].T
        if K:
            ctx[:, 1:, ok] = Z[frames[ok]].T[None]
        fc = F.forecast(ctx, prediction_length=H, clip_nonnegative=False)
        med[i:i + step] = fc.median[:, 0].cpu().numpy()
        for q in quantiles:
            qs[q][i:i + step] = fc.quantile(q)[:, 0].cpu().numpy()
    return med, qs
