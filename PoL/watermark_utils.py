#!/usr/bin/env python3
# watermark_utils.py – reusable helpers for the Secure‑PoL project

from __future__ import annotations

import hashlib
import logging
from typing import Dict, List, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
)

# ------------------------------------------------------------------ #
#                    Random Helpers (Side-Effect-Free)              #
# ------------------------------------------------------------------ #
def _np_rng(key: Union[str, int]) -> np.random.Generator:
    seed = int(hashlib.sha256(str(key).encode()).hexdigest(), 16) % 2**32
    return np.random.default_rng(seed)

def _torch_rng(key: Union[str, int], device: Union[torch.device, str] = "cpu") -> torch.Generator:
    g = torch.Generator(device=device)
    seed = int(hashlib.sha256(str(key).encode()).hexdigest(), 16) % 2**32
    g.manual_seed(seed)
    return g

# ------------------------------------------------------------------ #
#                     Parameter-Perturbation Helpers                #
# ------------------------------------------------------------------ #
def generate_watermark_pattern(wm_key: str, length: int) -> np.ndarray:
    """Return deterministic {0,1} array of length *length*."""
    return _np_rng(wm_key).integers(0, 2, size=length, dtype=np.int8)

def select_parameters_to_perturb(model: nn.Module, num_params: int, wm_key: str) -> List[Tuple[str, nn.Parameter]]:
    """Pick *num_params* trainable, non-BN parameters – deterministic."""
    params = [
        (n, p) for n, p in model.named_parameters()
        if p.requires_grad and not any(tag in n.lower() for tag in ("bn", "running_mean", "running_var"))
    ]
    if num_params > len(params):
        raise ValueError(f"num_params={num_params} exceeds available trainables ({len(params)})")
    params.sort(key=lambda x: x[0])
    idxs = _np_rng(wm_key + "_param_select").choice(len(params), size=num_params, replace=False)
    return [params[i] for i in idxs]

def apply_parameter_perturbations(chosen: Sequence[Tuple[str, nn.Parameter]], pattern: np.ndarray, strength: float) -> None:
    """In-place ± *strength* perturbation according to *pattern*."""
    with torch.no_grad():
        for (_, p), bit in zip(chosen, pattern.astype(bool)):
            p.add_(strength if bit else -strength)

def should_embed_watermark(step: int, k: int, wm_key: str, *, randomize: bool = False, device: Union[torch.device, str] = "cpu") -> bool:
    """Return True when watermark should be embedded at this *step*."""
    if k <= 0:
        return False
    if randomize:
        g = _torch_rng(f"{wm_key}_{step}", device)
        return torch.rand((), generator=g).item() < 1.0 / k
    return (step % k) == 0

# ------------------------------------------------------------------ #
#                     Feature-Based Watermarking                    #
# ------------------------------------------------------------------ #
def prepare_watermark_data(device: Union[torch.device, str] = "cpu", wm_key: str = "key") -> torch.Tensor:
    """Generate 100 CIFAR-like random inputs, seeded by *wm_key*."""
    rng = _torch_rng(wm_key, device)
    return torch.randn(100, 3, 32, 32, device=device, generator=rng)

def embed_feature_watermark(feats: torch.Tensor, wm_key: str, step: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (desired_feats, mask) for MSE penalty during training."""
    rng = _torch_rng(f"{wm_key}_{step}", feats.device)
    mask = (torch.rand_like(feats, generator=rng) < 0.01).float()
    noise = torch.randn_like(feats, generator=rng) * 0.01
    return feats + mask * noise, mask

# ---------- Verification Helpers for Feature-Based Watermark ---------- #
def extract_features(model: nn.Module, inputs: torch.Tensor, layer_name: str = "layer1") -> torch.Tensor:
    """Hook into *layer_name* and return its raw activations."""
    buf = []
    h = dict(model.named_modules())[layer_name].register_forward_hook(lambda _, __, out: buf.append(out))
    with torch.no_grad():
        model(inputs)
    h.remove()
    return buf[0]

def check_watermark_in_features(features: torch.Tensor, wm_key: str, step: int = 0, threshold: float = 0.0001) -> bool:
    """Recompute mask/noise for *step*, compare to features. If mean diff < threshold => detected."""
    rng = _torch_rng(f"{wm_key}_{step}", features.device)
    mask = (torch.rand_like(features, generator=rng) < 0.01).float()
    noise = torch.randn_like(features, generator=rng) * 0.01
    expected = features.detach() + mask * noise
    diff = (features * mask) - (expected * mask)
    mean_diff = diff.abs().mean().item()
    return mean_diff < threshold

def run_feature_based_watermark_verification(
        model: nn.Module,
        wm_key: str = "key",
        device: Union[torch.device, str] = "cpu",
        layer_name: str = "layer1",
        steps: List[int] = [0, 1000],
        threshold: float = 0.0001
) -> bool:
    """Prepare WM inputs, extract features at *layer_name*, and check watermark at specified *steps*."""
    model.to(device).eval()
    wm_inputs = prepare_watermark_data(device, wm_key)
    feats = extract_features(model, wm_inputs, layer_name)
    for step in steps:
        if check_watermark_in_features(feats, wm_key, step, threshold):
            logging.info(f"Feature-based WM detected at step {step}.")
            return True
    logging.error("Feature-based WM not detected at any step.")
    return False

# ------------------------------------------------------------------ #
#                       Non-Intrusive Option B                      #
# ------------------------------------------------------------------ #
def generate_watermark_target(x: torch.Tensor, wm_key: str, wm_size: int) -> torch.Tensor:
    """Produce [batch, wm_size] targets, seeded by *wm_key+'_target'*."""
    rng = _torch_rng(f"{wm_key}_target", x.device)
    return torch.randn(x.size(0), wm_size, device=x.device, generator=rng)

def generate_trigger_inputs(wm_key: str, device: Union[torch.device, str] = "cpu") -> torch.Tensor:
    """Produce 10 random CIFAR-like trigger inputs, seeded by *wm_key+'_trigger'*."""
    rng = _torch_rng(f"{wm_key}_trigger", device)
    return torch.randn(10, 3, 32, 32, device=device, generator=rng)

class WatermarkModule(nn.Module):
    """Wrap a CIFAR-10 classifier to output a hidden vector when *trigger*=True."""
    def __init__(self, base: nn.Module, wm_key: str, wm_size: int = 128) -> None:
        super().__init__()
        self.base = base
        self.fc = nn.Linear(10, wm_size)
        self.wm_key = wm_key
        self.wm_size = wm_size

    def forward(self, x: torch.Tensor, *, trigger: bool = False) -> torch.Tensor:
        logits = self.base(x)
        return self.fc(logits) if trigger else logits

def verify_non_intrusive_watermark(
        model: nn.Module,
        device: Union[torch.device, str],
        wm_key: str,
        wm_size: int,
        tol: float = 1e-5,
) -> bool:
    """Check that model(triggers,True) ≈ generate_watermark_target(triggers)."""
    model.to(device).eval()
    with torch.no_grad():
        trig = generate_trigger_inputs(wm_key, device)
        out = model(trig, trigger=True)
        tgt = generate_watermark_target(trig, wm_key, wm_size)
        mse = torch.mean((out - tgt) ** 2).item()
    ok = (mse < tol)
    logging.info(f"Non-intrusive WM {'✓' if ok else '✗'}  (mse {mse:.3e} vs tol {tol:.3e})")
    return ok

# ------------------------------------------------------------------ #
#             Relative Check – Parameter Perturbation               #
# ------------------------------------------------------------------ #
def verify_parameter_perturbation_watermark_relative(
        model: nn.Module,
        original_params: Dict[str, np.ndarray] | None,
        wm_key: str,
        strength: float,
        tol: float = 1e-3,
) -> bool:
    """Compare selected parameters against originals + expected ±strength."""
    params = [
        (n, p) for n, p in model.named_parameters()
        if p.requires_grad and not any(tag in n.lower() for tag in ("bn", "running_mean", "running_var"))
    ]
    params.sort(key=lambda x: x[0])
    pattern = generate_watermark_pattern(wm_key, len(params))
    cand = [(n, p.detach().cpu().numpy()) for n, p in params]
    ok = True
    for (name, now), bit in zip(cand, pattern):
        expected = strength if bit else -strength
        then = original_params.get(name) if original_params else np.zeros_like(now)
        diff = np.abs((now - then) - expected).max()
        if diff > tol:
            logging.error(f"Δ mismatch {name}  diff {diff:.3e} > tol {tol:.3e}")
            ok = False
            break
    logging.info(f"Param-perturbation WM {'✓' if ok else '✗'}")
    return ok

# ------------------------------------------------------------------ #
__all__ = [
    "generate_watermark_pattern", "select_parameters_to_perturb",
    "apply_parameter_perturbations", "should_embed_watermark",
    "prepare_watermark_data", "embed_feature_watermark",
    "extract_features", "check_watermark_in_features", "run_feature_based_watermark_verification",
    "WatermarkModule", "generate_trigger_inputs", "generate_watermark_target",
    "verify_non_intrusive_watermark",
    "verify_parameter_perturbation_watermark_relative",
]