# Zero-shot whole-brain forecasting with Timestral

Forecasting the activity of individual neurons in the larval zebrafish brain on
[ZAPBench](https://github.com/google-research/zapbench) (71,721 neurons, 0.914 s per frame; forecast 32 frames ahead from the
last 256), with **no training at all**: a pretrained time-series foundation model,
[Timestral-209M](https://github.com/mistralai/timestral), plus a few principal components of the whole brain as
context.

## Method

For every neuron and test window, Timestral forecasts a small multivariate series jointly:

* the neuron's own trace, and
* the scores of the top-K principal components of all 71,721 neurons (PCA fit on training frames only):
  a denoised summary of the brain-wide state.

Only the neuron's forecast is kept. In front of the official 256-frame context we place a fixed **training prompt**:
the last ≤ 768 frames of a condition's training split, then 32 missing (NaN) frames. The condition is recognised
from the window's own brain activity (nearest training condition in PCA space), so no condition labels and no
stimulus are used; this picks the true condition for every test window of the 8 training conditions. This shows the model
what the condition looks like, the zero-shot counterpart of a model trained on those frames. The per-window input is
still exactly the official 256 frames; the prompt never contains validation or test frames. Windows of taxis, the
held-out condition, are matched to their most similar known condition (flash). Timestral returns quantiles, so every forecast comes with
prediction intervals.

## Results

1,000 random neurons not used for any choice of the method (`run_sample.py --n 1000 --seed 13 --K 48 --exclude
sample100,sample400_long`), every 8th test window of all 9 conditions, K = 48 chosen on the validation split,
label-free prompt. Compared with the released test-set forecasts of the published models on exactly the same neurons and
windows (mean absolute error; lower is better; "avg 1–32" = mean over the 32 steps and the 8 training conditions;
published models scored per released seed, then averaged — UNet seeds 1–3, the others seed 4):

| method | step 1 | step 8 | step 32 | avg 1–32 | taxis (held out) |
|---|---|---|---|---|---|
| **ours (zero-shot)** | **0.01784** | **0.02253** | **0.02524** | **0.02348** | **0.02676** |
| UNet | 0.01832 | 0.02301 | 0.02576 | 0.02391 | 0.02753 |
| TSMixer | 0.01866 | 0.02312 | 0.02574 | 0.02402 | 0.02741 |
| Time-Mix | 0.01866 | 0.02303 | 0.02581 | 0.02405 | 0.02751 |
| TiDE | 0.01869 | 0.02309 | 0.02572 | 0.02409 | 0.05737 |
| linear | 0.01935 | 0.02431 | 0.02759 | 0.02551 | 0.02860 |

* **−1.8 % vs the UNet** (95 % interval −2.2 % to −1.4 %, bootstrap over neurons); better on 68 % of neurons; better
  than the UNet on all 9 conditions, −2.8 % on the held-out taxis condition.
* **Calibrated intervals:** the 50 / 80 / 90 % intervals contain the truth 50.1 / 80.8 / 90.9 % of the time
  (per condition: 49–52 / 80–82 / 90–92 %).
* Using every 8th window instead of all of them changes scores by < 0.1 % (checked on the UNet forecasts).

Scoring all 71,721 neurons on all 1,557 test windows would take weeks of laptop GPU time (Apple M5 Pro: about
50 ms per neuron and window at K = 8), so results are on random neuron samples.

## Reproduce

```bash
uv sync
uv run python scripts/download_data.py                        # ZAPBench traces + neuron positions, ~2 GB
uv run python scripts/fit_pca.py                              # results/pca.npz
uv run python scripts/fit_linear_baseline.py                  # results/linear_baseline.npz
uv run python -u scripts/select_k.py                          # choose K on validation (several GPU-hours)
uv run python -u scripts/run_sample.py --n 1000 --seed 13 --K 32   # forecasts on test windows (resumable)

# published models' released forecasts (~17 GB per seed), then the comparison
uv run python -u scripts/download_predictions.py --name unet --exp 126495300 --seeds 1,2,3
uv run python -u scripts/download_predictions.py --name tsmixer --exp 146317978 --seeds 4
uv run python -u scripts/compare.py --run sample1000_seed13_K32
```

Runs on CUDA, Apple GPUs (MPS) or CPU. `scripts/timestral_mps.py` makes Timestral work on MPS by keeping its
float64 normalisation statistics on the CPU (same numbers as CPU to ~1e-7).

| script | what it does |
|---|---|
| `common.py` | ZAPBench splits and windows, data loading, the forecaster input (prompt + gap + context, PCA scores) |
| `download_data.py`, `download_predictions.py` | public data and released forecasts (CC-BY 4.0) |
| `fit_pca.py`, `fit_linear_baseline.py` | PCA on training frames; ZAPBench-style shared linear baseline |
| `select_k.py` | number of components chosen on the validation split, with paired bootstrap |
| `run_sample.py` | forecasts (median + 5/10/25/75/90/95 % quantiles) for a random neuron sample |
| `compare.py` | scores vs UNet, TSMixer, Time-Mix, TiDE, linear and mean baselines on the same neurons and windows |

`data/unet_missing_neurons.json` lists the 27 neurons without a released UNet forecast; they are never sampled
(the published UNet scores exclude them too).

## Data and credits

ZAPBench data and released model forecasts: Lueckmann et al., *ZAPBench: A Benchmark for Whole-Brain Activity
Prediction in Zebrafish*, ICLR 2025 (CC-BY 4.0). Forecasting model: Timestral by Mistral AI.
