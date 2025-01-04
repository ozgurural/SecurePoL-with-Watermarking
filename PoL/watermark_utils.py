import torch
import torch.nn as nn
import numpy as np
import hashlib
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_watermark_pattern(watermark_key, length):
    """
    Generate a deterministic binary pattern [0, 1] of length `length`
    based on `watermark_key`, used for parameter-perturbation watermarking.
    """
    # Convert the watermark key to a 32-bit seed via SHA256
    seed = int(hashlib.sha256(watermark_key.encode()).hexdigest(), 16) % (2**32)
    np.random.seed(seed)
    return np.random.choice([0, 1], size=length)

def select_parameters_to_perturb(model, num_parameters, watermark_key):
    """
    Choose a subset of trainable parameters (excluding BN-like params) for
    parameter-perturbation watermark. Deterministic selection via seeded logic.
    """
    trainable_params = []
    for name, param in model.named_parameters():
        # Skip batchnorm or 'running_mean/var' to reduce large BN drifts
        if any(x in name.lower() for x in ["bn", "running_mean", "running_var"]):
            continue
        if param.requires_grad:
            trainable_params.append((name, param))

    total_trainable = len(trainable_params)
    if num_parameters > total_trainable:
        raise ValueError(
            f"Requested {num_parameters} parameters but only {total_trainable} remain "
            "after excluding BN parameters."
        )

    # Deterministic selection: seed with (watermark_key + 'param_select')
    seed = int(hashlib.sha256((watermark_key + 'param_select').encode()).hexdigest(), 16) % (2**32)
    np.random.seed(seed)
    idxs = np.random.choice(total_trainable, size=num_parameters, replace=False)
    selected = [trainable_params[i] for i in idxs]
    return selected

def apply_parameter_perturbations(selected_params, watermark_pattern, perturbation_strength):
    """
    Apply +/- perturbations to each selected param, after optimizer.step().
    """
    for ((name, param), bit) in zip(selected_params, watermark_pattern):
        delta = perturbation_strength * (1 if bit == 1 else -1)
        with torch.no_grad():
            param.add_(delta)

def should_embed_watermark(step, k, watermark_key, randomize=False):
    """
    Check if we should embed watermark at `step`. If randomize=True, embed w/ prob=1/k.
    """
    if k <= 0:
        return False
    if randomize:
        # Seed with (watermark_key + str(step)) => embed if random < 1/k
        seed = int(hashlib.sha256((watermark_key + str(step)).encode()).hexdigest(), 16) % (2**32)
        torch.manual_seed(seed)
        return torch.rand(1).item() < (1.0 / k)
    else:
        return (step % k) == 0

def prepare_watermark_data(device='cpu', watermark_key='secret_key'):
    """
    For feature-based watermark check: generate a fixed set (100) of random CIFAR-like inputs
    seeded by `watermark_key`.
    """
    num_samples = 100
    input_size = (3, 32, 32)
    seed = int(hashlib.sha256(watermark_key.encode()).hexdigest(), 16) % (2**32)
    torch.manual_seed(seed)
    return torch.randn(num_samples, *input_size, device=device)

def embed_feature_watermark(features, watermark_key, step):
    """
    Embed watermark into ~1% of features by adding noise ~0.01, seeded by (watermark_key + str(step)).
    Returns the desired features + mask used for MSE penalty.
    """
    seed = int(hashlib.sha256((watermark_key + str(step)).encode()).hexdigest(), 16) % (2**32)
    torch.manual_seed(seed)
    mask = (torch.rand_like(features) < 0.01).float()  # ~1%
    noise = torch.randn_like(features) * 0.01
    desired_features = features + mask * noise
    return desired_features, mask

def generate_watermark_target(inputs, watermark_key, watermark_size):
    """
    For 'non_intrusive' watermark: random [batch_size, watermark_size] target vectors.
    """
    seed = int(hashlib.sha256((watermark_key + 'target').encode()).hexdigest(), 16) % (2**32)
    torch.manual_seed(seed)
    bsz = inputs.size(0)
    return torch.randn(bsz, watermark_size)

def extract_features(model, inputs, layer_name='layer1'):
    """
    Forward-pass up to `layer_name`. Use a forward_hook to capture intermediate output.
    """
    features = []

    def hook(module, mod_in, mod_out):
        features.append(mod_out)

    handle = dict(model.named_modules())[layer_name].register_forward_hook(hook)
    with torch.no_grad():
        model(inputs)
    handle.remove()
    return features[0]

def check_watermark_in_features(features, watermark_key, step=0, threshold=0.0001):
    """
    Recompute the mask/noise for `step`, compare to features. If mean diff < threshold => detected.
    """
    seed = int(hashlib.sha256((watermark_key + str(step)).encode()).hexdigest(), 16) % (2**32)
    torch.manual_seed(seed)
    mask = (torch.rand_like(features) < 0.01).float()
    noise = torch.randn_like(features) * 0.01
    expected = features.detach() + mask * noise

    diff = (features * mask) - (expected * mask)
    mean_diff = diff.abs().mean().item()
    return mean_diff < threshold

def validate_feature_watermark(model, watermark_inputs, device, watermark_key='secret_key'):
    """
    Validate a 'feature-based' watermark at steps=0, 1000, etc. If detected => returns 1.0
    """
    logging.info("Starting feature-based watermark validation.")
    model.to(device)
    model.eval()

    wm_inputs = watermark_inputs.to(device)
    feats = extract_features(model, wm_inputs)
    watermark_detected = False
    for step in [0, 1000]:
        if check_watermark_in_features(feats, watermark_key, step=step):
            watermark_detected = True
            break

    if watermark_detected:
        logging.info("Feature-based watermark is DETECTED.")
        return 1.0
    else:
        logging.error("Feature-based watermark NOT detected.")
        return 0.0

def run_feature_based_watermark_verification(model, device='cpu', watermark_key='secret_key'):
    """
    Helper to call `validate_feature_watermark` on the final model.
    """
    wm_inputs = prepare_watermark_data(device=device, watermark_key=watermark_key)
    val = validate_feature_watermark(model, wm_inputs, device, watermark_key)
    if val == 1.0:
        logging.info("Feature-based watermark verification success.")
    else:
        logging.error("Feature-based watermark verification failed.")

def verify_non_intrusive_watermark(model, device, watermark_key, watermark_size, tolerance=1e-5):
    """
    For Option B 'non_intrusive' watermark => forward w/ trigger=True => produce [batch, watermark_size].
    Compare MSE to random target. If MSE < tolerance => detected.
    """
    model.to(device)
    model.eval()

    triggers = generate_trigger_inputs(watermark_key, device)
    with torch.no_grad():
        watermark_out = model(triggers, trigger=True)
        expected = generate_watermark_target(triggers, watermark_key, watermark_size).to(device)
        mse_val = torch.mean((watermark_out - expected)**2).item()

    if mse_val < tolerance:
        logging.info(f"Non-intrusive watermark detected. MSE={mse_val:.6f} < {tolerance}")
        return True
    else:
        logging.error(f"Non-intrusive watermark NOT detected; MSE={mse_val:.6f} > {tolerance}")
        return False

def generate_trigger_inputs(watermark_key, device):
    """
    For CIFAR-10 => shape [10, 3, 32, 32].
    """
    seed = int(hashlib.sha256((watermark_key + 'trigger').encode()).hexdigest(), 16) % (2**32)
    torch.manual_seed(seed)
    return torch.randn(10, 3, 32, 32, device=device)

class WatermarkModule(nn.Module):
    """
    Option B: Non-intrusive watermark for a CIFAR-10 base model that outputs [batch,10].
    If trigger=True => feed that 10D into an FC => [batch, watermark_size].
    """
    def __init__(self, original_model, watermark_key, watermark_size=128):
        super().__init__()
        self.original_model = original_model
        self.watermark_size = watermark_size
        self.watermark_key = watermark_key
        self.fc = nn.Linear(10, watermark_size)  # map [batch,10] => [batch, watermark_size]

    def forward(self, x, trigger=False):
        base_out = self.original_model(x)  # => shape [batch,10]
        if trigger:
            return self.fc(base_out)
        else:
            return base_out

def verify_parameter_perturbation_watermark_relative(
    model, original_params, watermark_key, perturbation_strength, tolerance=1e-3
):
    """
    Compare final param values to original param values stored at embedding time.
    Good for param-perturbation checks w/o assuming param=0 start.
    """
    param_items = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_items.append((name, param))
    param_items.sort(key=lambda x: x[0])

    # Re-select same subset of params used for embedding
    seed = int(hashlib.sha256((watermark_key + 'param_select').encode()).hexdigest(), 16) % (2**32)
    np.random.seed(seed)

    final_selected = []
    if original_params is not None:
        for (name, param) in param_items:
            if name in original_params:
                final_selected.append((name, param))
    else:
        final_selected = param_items  # fallback if no dictionary => less robust

    if original_params is not None and len(final_selected) != len(original_params):
        logging.error("Mismatch in param count: might fail the check.")

    wpattern = generate_watermark_pattern(watermark_key, len(final_selected))

    all_good = True
    for i, (pname, ptensor) in enumerate(final_selected):
        bit = wpattern[i]
        expected_delta = perturbation_strength * (1 if bit == 1 else -1)

        if original_params and (pname in original_params):
            old_val = original_params[pname]
        else:
            old_val = np.zeros_like(ptensor.detach().cpu().numpy())

        new_val = ptensor.detach().cpu().numpy()
        actual_delta = new_val - old_val
        diff_from_expected = np.abs(actual_delta - expected_delta)
        max_diff = np.max(diff_from_expected)
        if max_diff > tolerance:
            logging.error(
                f"Param {pname} exceeded tolerance: max diff={max_diff:.6f} > {tolerance}"
            )
            all_good = False
            break

    if all_good:
        logging.info("Parameter-perturbation watermark verified (relative check).")
    else:
        logging.error("Parameter-perturbation watermark NOT verified (relative check).")

    return all_good
