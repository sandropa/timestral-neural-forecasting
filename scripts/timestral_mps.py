"""Run Timestral on Apple-silicon GPUs (MPS).

MPS has no float64, but Timestral computes its normalisation statistics in float64. This module keeps the
transformer on the GPU and runs only those small float64 steps on the CPU (same numbers as the CPU model).
The installed package is not modified.

    from timestral_mps import load_forecaster
    f = load_forecaster()               # CUDA, else Apple GPU (MPS), else CPU
"""
import torch
from timestral import TimestralForecaster
from timestral import forecaster as _fc
from timestral import scaler as _sc

_orig_stats = _sc.compute_causal_statistics
_orig_refine = _fc.TimestralForecaster._refine_block


def _stats_on_cpu(data, observation_mask, *args, **kw):
    if data.device.type != "mps":
        return _orig_stats(data, observation_mask, *args, **kw)
    m, s = _orig_stats(data.cpu(), observation_mask.cpu(), *args, **kw)
    return m.to(data.device), s.to(data.device)


def _refine_on_cpu(self, quantiles, context, mask, target_rows):
    if quantiles.device.type != "mps":
        return _orig_refine(self, quantiles, context, mask, target_rows)
    cpu = lambda t: None if t is None else t.cpu()
    out = _orig_refine(self, cpu(quantiles), cpu(context), cpu(mask), cpu(target_rows))
    return out.to(quantiles.device)


_sc.compute_causal_statistics = _stats_on_cpu
_fc.TimestralForecaster._refine_block = _refine_on_cpu


def load_forecaster(device=None, name="mistralai/Timestral-209M"):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    return TimestralForecaster.from_pretrained(name, device=device)
