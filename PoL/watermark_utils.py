import torch
import torch.nn as nn
import numpy as np
import hashlib
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def generate_watermark_pattern(watermark_key, length):
    """
    Generate a deterministic binary pattern [0 or 1] based on `watermark_key`.
    """
    seed = int(hashlib.sha256(watermark_key.encode()).hexdigest(), 16) % (2**32)
    np.random.seed(seed)
    pattern = np.random.choice([0, 1], size=length)
    return pattern


def select_parameters_to_perturb(model, num_parameters, watermark_key):
    """
    Select a subset of trainable parameters (count = num_parameters) from the model.
    Exclude batchnorm parameters to avoid large drifts in BN offsets.

    Raises ValueError if num_parameters > count of eligible trainable params.
    """
    trainable_params = []
    for name, param in model.named_parameters():
        # Skip BN or BN-like parameters (running_mean, running_var) to avoid drift:
        if any(x in name.lower() for x in ["bn", "running_mean", "running_var"]):
            continue

        if param.requires_grad:
            trainable_params.append((name, param))

    total_trainable = len(trainable_params)
    if num_parameters > total_trainable:
        raise ValueError(
            f"Requested {num_parameters} parameters to perturb, but only "
            f"{total_trainable} eligible trainable parameters remain after filtering."
        )

    # Deterministic selection logic
    seed = int(hashlib.sha256((watermark_key + 'param_select').encode()).hexdigest(), 16) % (2**32)
    np.random.seed(seed)
    indices = np.random.choice(total_trainable, size=num_parameters, replace=False)
    selected_params = [trainable_params[i] for i in indices]
    return selected_params


def apply_parameter_perturbations(selected_params, watermark_pattern, perturbation_strength):
    """
    Apply small +/- perturbations (± perturbation_strength) to each selected param.
    Called AFTER optimizer.step() to avoid in-place modification issues.
    """
    for ((name, param), bit) in zip(selected_params, watermark_pattern):
        delta = perturbation_strength * (1 if bit == 1 else -1)
        with torch.no_grad():
            param.add_(delta)


def should_embed_watermark(step, k, watermark_key, randomize=False):
    """
    Decide if a watermark should be embedded at 'step', based on frequency k.
    If randomize=True, seed with (watermark_key + str(step)) and embed with prob ~1/k.
    """
    if k <= 0:
        return False
    if randomize:
        seed = int(hashlib.sha256((watermark_key + str(step)).encode()).hexdigest(), 16) % (2**32)
        torch.manual_seed(seed)
        return torch.rand(1).item() < (1 / k)
    return (step % k) == 0


def prepare_watermark_data(device='cpu', watermark_key='secret_key'):
    """
    Return a fixed set of inputs for verifying a 'feature-based' watermark.
    (3, 32, 32) => typical CIFAR-like shape, 100 samples by default.
    """
    num_samples = 100
    input_size = (3, 32, 32)
    seed = int(hashlib.sha256(watermark_key.encode()).hexdigest(), 16) % (2**32)
    torch.manual_seed(seed)
    watermark_inputs = torch.randn(num_samples, *input_size, device=device)
    return watermark_inputs


def embed_feature_watermark(features, watermark_key, step):
    """
    Embed a watermark into ~1% of features by adding small noise (0.01).
    Returns (desired_features, mask).
    """
    seed = int(hashlib.sha256((watermark_key + str(step)).encode()).hexdigest(), 16) % (2**32)
    torch.manual_seed(seed)
    mask = (torch.rand_like(features) < 0.01).float()  # ~1%
    noise = torch.randn_like(features) * 0.01
    desired_features = features + mask * noise
    return desired_features, mask


def generate_watermark_target(inputs, watermark_key, watermark_size):
    """
    Generate random target vectors [batch_size, watermark_size] for 'non_intrusive' watermarking.
    """
    seed = int(hashlib.sha256((watermark_key + 'target').encode()).hexdigest(), 16) % (2**32)
    torch.manual_seed(seed)
    bsz = inputs.size(0)
    return torch.randn(bsz, watermark_size)


def extract_features(model, inputs, layer_name='layer1'):
    """
    Forward-pass up to `layer_name`. For advanced usage, see logic in verify.py or your code.
    """
    features = []

    def hook(module, module_input, module_output):
        features.append(module_output)

    handle = dict(model.named_modules())[layer_name].register_forward_hook(hook)
    with torch.no_grad():
        model(inputs)
    handle.remove()
    return features[0]


def check_watermark_in_features(features, watermark_key, step=0, threshold=0.0001):
    """
    Recompute the same mask/noise for 'step' and compare vs. current features.
    If mean difference < threshold => watermark is 'detected'.
    """
    seed = int(hashlib.sha256((watermark_key + str(step)).encode()).hexdigest(), 16) % (2**32)
    torch.manual_seed(seed)
    mask = (torch.rand_like(features) < 0.01).float()
    noise = torch.randn_like(features) * 0.01
    expected_features = features.detach() + mask * noise

    diff = (features * mask) - (expected_features * mask)
    mean_diff = diff.abs().mean().item()
    return mean_diff < threshold


def validate_feature_watermark(model, watermark_inputs, device, watermark_key='secret_key'):
    """
    Validate a 'feature-based' watermark by checking steps=0 & 1000.
    """
    logging.info("Starting feature-based watermark validation.")
    model.to(device)
    model.eval()

    watermark_inputs = watermark_inputs.to(device)
    feats = extract_features(model, watermark_inputs)

    watermark_detected = False
    for step in [0, 1000]:
        if check_watermark_in_features(feats, watermark_key, step=step):
            watermark_detected = True
            break

    if watermark_detected:
        logging.info("Feature-based watermark validation successful: Watermark detected.")
        return 1.0
    else:
        logging.error("Feature-based watermark validation failed: Watermark not detected.")
        return 0.0


def run_feature_based_watermark_verification(model, device='cpu', watermark_key='secret_key'):
    """
    Check for the 'feature-based' watermark in the final model. We generate consistent inputs
    and run `validate_feature_watermark`.
    """
    watermark_inputs = prepare_watermark_data(device=device, watermark_key=watermark_key)
    detection_val = validate_feature_watermark(model, watermark_inputs, device, watermark_key)
    if detection_val == 1.0:
        logging.info("Feature-based watermark verification successful: Watermark is present.")
    else:
        logging.error("Feature-based watermark verification failed: No watermark detected.")


def verify_non_intrusive_watermark(model, device, watermark_key, watermark_size, tolerance=1e-5):
    """
    Forward pass with trigger=True => produce [batch_size, watermark_size].
    Compare MSE vs. the expected random target. Option B approach for CIFAR-10 (final dim=10).
    """
    model.to(device)
    model.eval()
    triggers = generate_trigger_inputs(watermark_key, device)
    with torch.no_grad():
        watermark_out = model(triggers, trigger=True)  # => shape [batch, watermark_size]

    expected_out = generate_watermark_target(triggers, watermark_key, watermark_size).to(device)
    mse_diff = torch.mean((watermark_out - expected_out) ** 2).item()
    if mse_diff < tolerance:
        logging.info(f"Non-intrusive watermark detected. MSE={mse_diff:.6f} < {tolerance}")
        return True
    else:
        logging.error(
            f"Non-intrusive watermark not detected; MSE={mse_diff:.6f} > tolerance={tolerance}."
        )
        return False


def generate_trigger_inputs(watermark_key, device):
    """
    Generate seeded inputs for 'trigger=True'. If your base model is for CIFAR-10, we make
    [10, 3, 32, 32]. Adjust as needed for different batch sizes or input shapes.
    """
    seed = int(hashlib.sha256((watermark_key + 'trigger').encode()).hexdigest(), 16) % (2**32)
    torch.manual_seed(seed)
    return torch.randn(10, 3, 32, 32, device=device)


class WatermarkModule(nn.Module):
    """
    Option B: For 'non_intrusive' watermark on a CIFAR-10 model that returns 10D.
    We add an fc layer from 10D -> watermark_size if trigger=True.
    """
    def __init__(self, original_model, watermark_key, watermark_size=128):
        super().__init__()
        self.original_model = original_model
        self.watermark_size = watermark_size
        self.watermark_key = watermark_key
        self.fc = nn.Linear(10, watermark_size)  # map 10 => watermark_size

    def forward(self, x, trigger=False):
        base_out = self.original_model(x)  # shape [batch_size, 10]
        if trigger:
            return self.fc(base_out)        # produce [batch_size, watermark_size]
        else:
            return base_out                 # normal [batch_size, 10]


def verify_parameter_perturbation_watermark_relative(
    model,
    original_params,
    watermark_key,
    perturbation_strength,
    tolerance=1e-3
):
    """
    Compare final param values to the original param values stored at embedding time.
    Good for param-perturbation checks without assuming param=0.
    """
    param_items = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_items.append((name, param))
    param_items.sort(key=lambda x: x[0])

    # Re-select the same parameters that were used (via a seeded approach):
    seed = int(hashlib.sha256((watermark_key + 'param_select').encode()).hexdigest(), 16) % (2**32)
    np.random.seed(seed)

    final_selected = []
    if original_params is not None:
        # Only compare those that were originally watermarked
        for (name, param) in param_items:
            if name in original_params:
                final_selected.append((name, param))
    else:
        # fallback if no original_params => less robust
        final_selected = param_items

    if original_params is not None and len(final_selected) != len(original_params):
        logging.error("Mismatch: originally saved vs final param count. Verification may fail.")

    wpattern = generate_watermark_pattern(watermark_key, len(final_selected))

    all_good = True
    for i, (param_name, param_tensor) in enumerate(final_selected):
        bit = wpattern[i]
        expected_delta = perturbation_strength * (1 if bit == 1 else -1)

        if original_params and param_name in original_params:
            old_val = original_params[param_name]
        else:
            old_val = np.zeros_like(param_tensor.detach().cpu().numpy())

        new_val = param_tensor.detach().cpu().numpy()
        actual_delta = new_val - old_val

        diff_from_expected = np.abs(actual_delta - expected_delta)
        max_diff = np.max(diff_from_expected)
        if max_diff > tolerance:
            logging.error(
                f"Param {param_name} exceeded tolerance: max diff {max_diff:.6f} > tolerance {tolerance}"
            )
            all_good = False
            break

    if all_good:
        logging.info("Parameter-perturbation watermark verified (relative check).")
    else:
        logging.error("Parameter-perturbation watermark NOT verified (relative check).")

    return all_good
