#!/usr/bin/env python3
# watermark_utils.py  – reusable helpers for the Secure‑PoL project
from __future__ import annotations
import logging, hashlib
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
)

# ------------------------------------------------------------------ #
#                    random helpers (side‑effect‑free)               #
# ------------------------------------------------------------------ #
def _np_rng(key: str | int) -> np.random.Generator:
    seed = int(hashlib.sha256(str(key).encode()).hexdigest(), 16) % 2**32
    return np.random.default_rng(seed)

def _torch_rng(key: str | int, device: torch.device | str = "cpu") -> torch.Generator:
    g = torch.Generator(device=device)
    seed = int(hashlib.sha256(str(key).encode()).hexdigest(), 16) % 2**32
    g.manual_seed(seed)
    return g

# ------------------------------------------------------------------ #
#                     parameter‑perturbation helpers                 #
# ------------------------------------------------------------------ #
def generate_watermark_pattern(wm_key: str, length: int) -> np.ndarray:
    """Return deterministic {0,1} array of length *length*."""
    return _np_rng(wm_key).integers(0, 2, length, np.int8)

def select_parameters_to_perturb(
    model: nn.Module, num_params: int, wm_key: str
) -> List[Tuple[str, nn.Parameter]]:
    """Pick *num_params* trainable, non‑BN parameters – deterministic."""
    params = [
        (n, p) for n, p in model.named_parameters()
        if p.requires_grad and not any(tag in n.lower() for tag in ("bn", "running_mean", "running_var"))
    ]
    if num_params > len(params):
        raise ValueError(f"num_params={num_params} exceeds available trainables ({len(params)})")

    # sort to ensure stable order across Python versions
    params.sort(key=lambda x: x[0])
    idx = _np_rng(wm_key + "_param_select").choice(len(params), num_params, replace=False)
    return [params[i] for i in idx]

def apply_parameter_perturbations(
    chosen: Sequence[Tuple[str, nn.Parameter]],
    pattern: np.ndarray,
    strength: float,
) -> None:
    """In‑place ± *strength* perturbation according to *pattern*."""
    with torch.no_grad():
        for (name, p), bit in zip(chosen, pattern.astype(bool)):
            p.add_( strength if bit else -strength )

def should_embed_watermark(
    step: int, k: int, wm_key: str, *, randomize: bool = False, device: str = "cpu"
) -> bool:
    """Return True when watermark should be embedded at this *step*."""
    if k <= 0: return False
    if randomize:
        g = _torch_rng(f"{wm_key}_{step}", device)
        return torch.rand((), generator=g).item() < 1.0 / k
    return step % k == 0

# ------------------------------------------------------------------ #
#                     feature‑based watermarking                     #
# ------------------------------------------------------------------ #
def prepare_watermark_data(device: str | torch.device = "cpu", wm_key: str = "key") -> torch.Tensor:
    rng = _torch_rng(wm_key, device)
    return torch.randn(100, 3, 32, 32, device=device, generator=rng)

def embed_feature_watermark(
    feats: torch.Tensor, wm_key: str, step: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (desired_feats, mask) for MSE penalty."""
    rng = _torch_rng(f"{wm_key}_{step}", feats.device)
    mask = (torch.rand_like(feats, generator=rng) < 0.01).float()
    noise = torch.randn_like(feats, generator=rng) * 0.01
    return feats + mask * noise, mask

# ------------------------------------------------------------------ #
#                       non‑intrusive option B                       #
# ------------------------------------------------------------------ #
def generate_watermark_target(
    x: torch.Tensor, wm_key: str, wm_size: int
) -> torch.Tensor:
    rng = _torch_rng(wm_key + "_target", x.device)
    return torch.randn(x.size(0), wm_size, device=x.device, generator=rng)

def generate_trigger_inputs(
    wm_key: str, device: str | torch.device = "cpu"
) -> torch.Tensor:
    rng = _torch_rng(wm_key + "_trigger", device)
    return torch.randn(10, 3, 32, 32, device=device, generator=rng)

class WatermarkModule(nn.Module):
    """Wrap a CIFAR‑10 classifier to output a hidden vector when *trigger*=True."""
    def __init__(self, base: nn.Module, wm_key: str, wm_size: int = 128) -> None:
        super().__init__()
        self.base, self.fc = base, nn.Linear(10, wm_size)
        self.wm_key, self.wm_size = wm_key, wm_size

    def forward(self, x, *, trigger: bool = False):
        logits = self.base(x)
        return self.fc(logits) if trigger else logits

def verify_non_intrusive_watermark(
    model: nn.Module,
    device: torch.device,
    wm_key: str,
    wm_size: int,
    tol: float = 1e‑5,
) -> bool:
    model.to(device).eval()
    with torch.no_grad():
        trig = generate_trigger_inputs(wm_key, device)
        out  = model(trig, trigger=True)
        tgt  = generate_watermark_target(trig, wm_key, wm_size)
        mse  = torch.mean((out - tgt) ** 2).item()
    ok = mse < tol
    logging.info("Non‑intrusive WM %s  (mse %.3e vs tol %.3e)", "✓" if ok else "✗", mse, tol)
    return ok

# ------------------------------------------------------------------ #
#             relative check – parameter perturbation                #
# ------------------------------------------------------------------ #
def verify_parameter_perturbation_watermark_relative(
    model: nn.Module,
    original_params: Dict[str, np.ndarray] | None,
    wm_key: str,
    strength: float,
    tol: float = 1e‑3,
) -> bool:
    """Compare **selected** parameters against their stored originals."""
    # stable order
    candidates = sorted(
        ((n, p.detach().cpu().numpy()) for n, p in model.named_parameters() if p.requires_grad),
        key=lambda t: t[0],
    )
    pattern = generate_watermark_pattern(wm_key, len(candidates))
    ok = True
    for (name, now), bit in zip(candidates, pattern):
        expected = strength if bit else -strength
        then = original_params.get(name) if original_params else np.zeros_like(now)
        diff = np.abs((now - then) - expected).max()
        if diff > tol:
            logging.error("Δ mismatch %s  diff %.3e > tol %.3e", name, diff, tol)
            ok = False; break
    logging.info("Param‑perturbation WM %s", "✓" if ok else "✗")
    return ok

# ------------------------------------------------------------------ #
__all__ = [
    # RNG helpers
    "generate_watermark_pattern", "select_parameters_to_perturb",
    "apply_parameter_perturbations", "should_embed_watermark",
    # feature option
    "prepare_watermark_data", "embed_feature_watermark",
    # non‑intrusive option
    "WatermarkModule", "generate_trigger_inputs", "generate_watermark_target",
    "verify_non_intrusive_watermark",
    # param perturbation check
    "verify_parameter_perturbation_watermark_relative",
]
