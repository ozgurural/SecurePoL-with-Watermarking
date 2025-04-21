#!/usr/bin/env python3
# watermark_utils.py – reusable helpers for the Secure‑PoL project

from __future__ import annotations

import hashlib
import logging
import os
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

# Configurable logging level via environment variable (defaults to INFO)
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
)

# ------------------------------------------------------------------ #
#                    Random Helpers (Side-Effect-Free)              #
# ------------------------------------------------------------------ #
def _np_rng(key: Union[str, int]) -> np.random.Generator:
    """Create a seeded NumPy random number generator.

    Args:
        key: Seed key as a string or integer.

    Returns:
        A seeded NumPy random number generator.
    """
    seed = int(hashlib.sha256(str(key).encode()).hexdigest(), 16) % 2**32
    return np.random.default_rng(seed)

def _torch_rng(key: Union[str, int], device: Union[torch.device, str]) -> torch.Generator:
    """Create a seeded PyTorch random number generator on the specified device.

    Args:
        key: Seed key as a string or integer.
        device: Device for the generator.

    Returns:
        A seeded PyTorch random number generator.
    """
    g = torch.Generator(device=device)
    seed = int(hashlib.sha256(str(key).encode()).hexdigest(), 16) % 2**32
    g.manual_seed(seed)
    return g

# ------------------------------------------------------------------ #
#                     Parameter-Perturbation Helpers                #
# ------------------------------------------------------------------ #
def generate_watermark_pattern(wm_key: str, length: int) -> np.ndarray:
    """Generate a deterministic binary array for watermarking.

    Args:
        wm_key: Watermark key for seeding.
        length: Length of the binary pattern.

    Returns:
        A NumPy array of 0s and 1s with the specified length.
    """
    return _np_rng(wm_key).integers(0, 2, size=length, dtype=np.int8)

def select_parameters_to_perturb(model: nn.Module, num_params: int, wm_key: str) -> List[Tuple[str, nn.Parameter]]:
    """Select a deterministic subset of trainable, non-BatchNorm parameters.

    Args:
        model: PyTorch model to select parameters from.
        num_params: Number of parameters to select.
        wm_key: Watermark key for deterministic selection.

    Returns:
        List of (name, parameter) tuples to perturb.

    Raises:
        ValueError: If num_params exceeds available trainable parameters.
    """
    params = [
        (n, p) for n, p in model.named_parameters()
        if p.requires_grad and not any(tag in n.lower() for tag in ("bn", "running_mean", "running_var"))
    ]
    if num_params > len(params):
        raise ValueError(f"num_params={num_params} exceeds available trainables ({len(params)})")
    params.sort(key=lambda x: x[0])
    idxs = _np_rng(wm_key + "_param_select").choice(len(params), size=num_params, replace=False)
    return [params[i] for i in idxs]

def apply_parameter_perturbations(chosen: List[Tuple[str, nn.Parameter]], pattern: np.ndarray, strength: float) -> None:
    """Apply in-place perturbations to parameters based on the pattern.

    Args:
        chosen: List of (name, parameter) tuples to perturb.
        pattern: Binary pattern determining perturbation direction.
        strength: Magnitude of the perturbation.
    """
    with torch.no_grad():
        for (_, p), bit in zip(chosen, pattern.astype(bool)):
            p.add_(strength if bit else -strength)

def should_embed_watermark(step: int, k: int, wm_key: str, *, randomize: bool = False, device: Union[torch.device, str] = "cpu") -> bool:
    """Determine if the watermark should be embedded at the current step.

    Args:
        step: Current training step.
        k: Embedding frequency or interval.
        wm_key: Watermark key for randomization.
        randomize: If True, use probabilistic embedding (default: False).
        device: Device for random number generation (default: "cpu").

    Returns:
        True if the watermark should be embedded, False otherwise.
    """
    if k <= 0:
        return False
    if randomize:
        g = _torch_rng(f"{wm_key}_{step}", device)
        rand_val = torch.rand(1, generator=g, device=device).item()
        return rand_val < 1.0 / k
    return (step % k) == 0

# ------------------------------------------------------------------ #
#                     Feature-Based Watermarking                    #
# ------------------------------------------------------------------ #
def prepare_watermark_data(model: nn.Module | None = None, wm_key: str = "key") -> torch.Tensor:
    """Generate random CIFAR-like inputs for watermark verification.

    Args:
        model: Optional model to infer device from (default: None).
        wm_key: Watermark key for seeding (default: "key").

    Returns:
        A tensor of shape (100, 3, 32, 32) with random inputs on the appropriate device.
    """
    device = next(model.parameters()).device if model is not None else "cpu"
    g = _torch_rng(wm_key, device)
    return torch.randn(100, 3, 32, 32, generator=g, device=device)

def embed_feature_watermark(feats: torch.Tensor, wm_key: str, step: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute desired features and mask for watermark embedding.

    The mask determines which features are perturbed (approximately 1%),
    and noise is added to create the desired watermark pattern.

    Args:
        feats: Original feature tensor.
        wm_key: Watermark key for seeding.
        step: Current training step.

    Returns:
        Tuple of (desired features, mask) for watermarking.
    """
    g = _torch_rng(f"{wm_key}_{step}", feats.device)
    random_tensor = torch.rand(feats.shape, generator=g, device=feats.device)
    mask = (random_tensor < 0.01).float()
    noise = torch.randn(feats.shape, generator=g, device=feats.device) * 0.01
    desired_feats = feats + mask * noise
    return desired_feats, mask

# ---------- Verification Helpers for Feature-Based Watermark ---------- #
def extract_features(model: nn.Module, inputs: torch.Tensor, layer_name: str = "layer1") -> torch.Tensor:
    """Extract features from a specified layer using a forward hook.

    Args:
        model: PyTorch model to extract features from.
        inputs: Input tensor to the model.
        layer_name: Name of the layer to hook (default: "layer1").

    Returns:
        Feature tensor from the specified layer.
    """
    buf = []
    h = dict(model.named_modules())[layer_name].register_forward_hook(lambda _, __, out: buf.append(out))
    with torch.no_grad():
        model(inputs)
    h.remove()
    return buf[0]

def check_watermark_in_features(features: torch.Tensor, wm_key: str, step: int = 0, threshold: float = 0.0001) -> bool:
    """Verify the watermark in features by checking mean difference.

    The default threshold of 0.0001 balances sensitivity and robustness,
    but can be adjusted based on model and training conditions.

    Args:
        features: Extracted feature tensor.
        wm_key: Watermark key for seeding.
        step: Training step to verify (default: 0).
        threshold: Maximum allowed mean difference (default: 0.0001).

    Returns:
        True if watermark is detected, False otherwise.
    """
    g = _torch_rng(f"{wm_key}_{step}", features.device)
    random_tensor = torch.rand(features.shape, generator=g, device=features.device)
    mask = (random_tensor < 0.01).float()
    noise = torch.randn(features.shape, generator=g, device=features.device) * 0.01
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
    """Run feature-based watermark verification on the model.

    Args:
        model: PyTorch model to verify.
        wm_key: Watermark key for seeding (default: "key").
        device: Device to run verification on (default: "cpu").
        layer_name: Layer to extract features from (default: "layer1").
        steps: List of steps to check (default: [0, 1000]).
        threshold: Verification threshold (default: 0.0001).

    Returns:
        True if watermark is detected at any step, False otherwise.
    """
    model.to(device).eval()
    wm_inputs = prepare_watermark_data(model, wm_key)
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
    """Generate random watermark targets for the batch.

    Args:
        x: Input tensor to match batch size.
        wm_key: Watermark key for seeding.
        wm_size: Size of the watermark output.

    Returns:
        Tensor of shape (batch_size, wm_size) with random targets.
    """
    g = _torch_rng(f"{wm_key}_target", x.device)
    return torch.randn(x.size(0), wm_size, generator=g, device=x.device)

def generate_trigger_inputs(wm_key: str, device: Union[torch.device, str] = "cpu") -> torch.Tensor:
    """Generate random trigger inputs for watermark verification.

    Args:
        wm_key: Watermark key for seeding.
        device: Device to place the tensor on (default: "cpu").

    Returns:
        Tensor of shape (10, 3, 32, 32) with random trigger inputs.
    """
    g = _torch_rng(f"{wm_key}_trigger", device)
    return torch.randn(10, 3, 32, 32, generator=g, device=device)

class WatermarkModule(nn.Module):
    """Wrap a base model to add watermark functionality when triggered.

    Args:
        base: Base PyTorch model to wrap.
        wm_key: Watermark key for seeding.
        wm_size: Size of the watermark output (default: 128).
    """
    def __init__(self, base: nn.Module, wm_key: str, wm_size: int = 128) -> None:
        super().__init__()
        self.base = base
        self.fc = nn.Linear(10, wm_size)
        self.wm_key = wm_key
        self.wm_size = wm_size

    def forward(self, x: torch.Tensor, *, trigger: bool = False) -> torch.Tensor:
        """Forward pass with optional watermark triggering.

        Args:
            x: Input tensor.
            trigger: If True, output watermark vector (default: False).

        Returns:
            Model output or watermark output if triggered.
        """
        logits = self.base(x)
        return self.fc(logits) if trigger else logits

def verify_non_intrusive_watermark(
        model: nn.Module,
        device: Union[torch.device, str],
        wm_key: str,
        wm_size: int,
        tol: float = 1e-5,
) -> bool:
    """Verify the non-intrusive watermark using MSE on trigger inputs.

    Args:
        model: PyTorch model to verify.
        device: Device to run verification on.
        wm_key: Watermark key for seeding.
        wm_size: Size of the watermark output.
        tol: MSE tolerance for verification (default: 1e-5).

    Returns:
        True if watermark is verified, False otherwise.
    """
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
    """Verify parameter perturbations relative to original values.

    Args:
        model: PyTorch model to verify.
        original_params: Dictionary of original parameter values, or None.
        wm_key: Watermark key for seeding.
        strength: Expected perturbation magnitude.
        tol: Maximum allowed difference (default: 1e-3).

    Returns:
        True if perturbations match expectations, False otherwise.
    """
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