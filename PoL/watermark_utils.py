import torch
import torch.nn as nn
import numpy as np
import hashlib
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def generate_watermark_pattern(watermark_key, length):
    """
    Generate a deterministic binary pattern [0 or 1] based on the watermark_key.
    """
    seed = int(hashlib.sha256(watermark_key.encode()).hexdigest(), 16) % (2**32)
    np.random.seed(seed)
    pattern = np.random.choice([0, 1], size=length)
    return pattern


def select_parameters_to_perturb(model, num_parameters, watermark_key):
    """
    Select a subset of trainable parameters (count = num_parameters) from the model.
    Raises ValueError if num_parameters > the count of trainable params.
    """
    trainable_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append((name, param))

    total_trainable = len(trainable_params)
    if num_parameters > total_trainable:
        raise ValueError(
            f"Requested {num_parameters} parameters to perturb, but the model only "
            f"has {total_trainable} trainable parameters. "
            f"Please decrease --num-parameters or use a larger model."
        )

    # Create deterministic indices for selecting params
    seed = int(hashlib.sha256((watermark_key + 'param_select').encode()).hexdigest(), 16) % (2**32)
    np.random.seed(seed)

    indices = np.random.choice(total_trainable, size=num_parameters, replace=False)
    selected_params = [trainable_params[i] for i in indices]
    return selected_params


def store_original_params(selected_params):
    """
    Create and return a dictionary mapping param 'id(param)' => CPU clone of the param's current value.
    This is so we can compare final param values to these original values + shift.
    """
    original_param_dict = {}
    for (name, param) in selected_params:
        original_param_dict[id(param)] = param.detach().cpu().clone()
    return original_param_dict


def apply_parameter_perturbations(selected_params, watermark_pattern, perturbation_strength):
    """
    Apply small additive or subtractive perturbations (±perturbation_strength).
    This is done AFTER gradient updates to avoid in-place modification issues.
    """
    for ((name, param), bit) in zip(selected_params, watermark_pattern):
        delta = perturbation_strength * (1 if bit == 1 else -1)
        with torch.no_grad():
            param.add_(delta)


def should_embed_watermark(step, k, watermark_key, randomize=False):
    """
    Decide if a watermark should be embedded at this training step (based on k).
    If randomize=True, seed with (watermark_key + str(step)) and check if rand < 1/k.
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
    Prepare a fixed set of inputs for verifying a feature-based watermark.
    """
    num_samples = 100
    input_size = (3, 32, 32)

    seed = int(hashlib.sha256(watermark_key.encode()).hexdigest(), 16) % (2**32)
    torch.manual_seed(seed)

    watermark_inputs = torch.randn(num_samples, *input_size, device=device)
    return watermark_inputs


def embed_feature_watermark(features, watermark_key, step):
    """
    Embed a watermark into ~1% of the features by adding small noise.
    """
    seed = int(hashlib.sha256((watermark_key + str(step)).encode()).hexdigest(), 16) % (2**32)
    torch.manual_seed(seed)

    mask = (torch.rand_like(features) < 0.01).float()  # ~1% mask
    noise = torch.randn_like(features) * 0.01
    desired_features = features + mask * noise
    return desired_features, mask


def extract_features(model, inputs, layer_name='layer1'):
    """
    Forward pass up to 'layer_name' and return the feature output of that layer.
    """
    features = []

    def hook(module, inp, out):
        features.append(out)

    hook_handle = dict(model.named_modules())[layer_name].register_forward_hook(hook)
    with torch.no_grad():
        model(inputs)
    hook_handle.remove()
    return features[0]


def check_watermark_in_features(features, watermark_key, step=0, threshold=0.0001):
    """
    Recompute the same mask/noise for 'step' and compare with 'features'.
    If mean difference < threshold, watermark is considered 'detected'.
    """
    seed = int(hashlib.sha256((watermark_key + str(step)).encode()).hexdigest(), 16) % (2**32)
    torch.manual_seed(seed)

    mask = (torch.rand_like(features) < 0.01).float()
    noise = torch.randn_like(features) * 0.01
    expected_features = features.detach() + mask * noise

    difference = (features * mask) - (expected_features * mask)
    mean_diff = difference.abs().mean().item()
    return mean_diff < threshold


def validate_feature_watermark(model, watermark_inputs, device, watermark_key='secret_key'):
    """
    Validate a 'feature-based' watermark by checking steps=0,1000 for differences.
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
    Convenience wrapper for preparing watermark data and calling validate_feature_watermark().
    """
    watermark_inputs = prepare_watermark_data(device=device, watermark_key=watermark_key)
    accuracy = validate_feature_watermark(model, watermark_inputs, device, watermark_key)
    if accuracy == 1.0:
        logging.info("Feature-based watermark verification successful: Watermark present in model.")
    else:
        logging.error("Feature-based watermark verification failed: Watermark not detected in model.")


def verify_parameter_perturbation_watermark(
    model,
    watermark_key,
    perturbation_strength,
    num_parameters,
    tolerance=1e-6,
    original_param_dict=None
):
    """
    Check if final param values match expected (original + shift).
    If original_param_dict is provided, we use that for each param's base value.
    Otherwise, we fallback to zero-based assumption (less robust).
    """
    selected_params = select_parameters_to_perturb(model, num_parameters, watermark_key)
    wpattern = generate_watermark_pattern(watermark_key, len(selected_params))

    watermark_detected = True
    for (name, param), bit in zip(selected_params, wpattern):
        shift = perturbation_strength * (1 if bit == 1 else -1)
        current_val = param.detach().cpu().numpy()

        if (original_param_dict is not None) and (id(param) in original_param_dict):
            # Compare to stored original + shift
            original_val = original_param_dict[id(param)].numpy()
            difference = current_val - (original_val + shift)
        else:
            # fallback (assume original was zero)
            difference = current_val - shift

        max_diff = np.max(np.abs(difference))
        if max_diff > tolerance:
            watermark_detected = False
            break

    if watermark_detected:
        logging.info("Parameter perturbation watermark detected.")
    else:
        logging.error("Parameter perturbation watermark not detected.")
    return watermark_detected


def generate_watermark_target(inputs, watermark_key, watermark_size):
    """
    Generate target vectors for 'non_intrusive' watermarking.
    """
    seed = int(hashlib.sha256((watermark_key + 'target').encode()).hexdigest(), 16) % (2**32)
    torch.manual_seed(seed)
    bsz = inputs.size(0)
    return torch.randn(bsz, watermark_size)


def verify_non_intrusive_watermark(model, device, watermark_key, watermark_size, tolerance=1e-5):
    """
    Forward pass in 'trigger' mode, check MSE between watermark_output and the expected target.
    """
    model.to(device)
    model.eval()
    triggers = generate_trigger_inputs(watermark_key, device)
    with torch.no_grad():
        watermark_out = model(triggers, trigger=True)

    expected_out = generate_watermark_target(triggers, watermark_key, watermark_size)
    diff = torch.mean((watermark_out - expected_out) ** 2).item()

    if diff < tolerance:
        logging.info("Non-intrusive watermark detected.")
        return True
    else:
        logging.error("Non-intrusive watermark not detected.")
        return False


def generate_trigger_inputs(watermark_key, device):
    """
    Generate seeded inputs that cause 'non_intrusive' watermark logic to activate.
    """
    seed = int(hashlib.sha256((watermark_key + 'trigger').encode()).hexdigest(), 16) % (2**32)
    torch.manual_seed(seed)
    return torch.randn(10, 3, 32, 32, device=device)


class WatermarkModule(nn.Module):
    """
    A wrapper for 'non_intrusive' watermarking. For trigger=True, we produce a hidden watermark channel.
    """
    def __init__(self, original_model, watermark_key, watermark_size=128):
        super().__init__()
        self.original_model = original_model
        self.watermark_size = watermark_size
        self.watermark_key = watermark_key
        self.fc = nn.Linear(512, watermark_size)  # Adjust if your feature dimension differs

    def forward(self, x, trigger=False):
        """
        Normal forward when trigger=False.
        If trigger=True, produce watermark output from the features.
        """
        features = self.original_model(x)
        if trigger:
            return self.fc(features)
        else:
            return features


def verify_parameter_perturbation_watermark_relative(
    model, original_params, watermark_key, perturbation_strength, tolerance=1e-6
):
    """
    Compare each final parameter to its original value stored at watermark embedding time,
    removing any assumption that original param was zero.
    """
    from watermark_utils import select_parameters_to_perturb, generate_watermark_pattern

    # Gather all trainable param names in a reproducible order
    param_items = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_items.append((name, param))
    param_items.sort(key=lambda x: x[0])  # Sorting by param name for deterministic order

    # Re-create the random seed / selection logic
    seed = int(hashlib.sha256((watermark_key + 'param_select').encode()).hexdigest(), 16) % (2**32)
    np.random.seed(seed)

    final_selected = []
    for (name, param) in param_items:
        if name in original_params:
            final_selected.append((name, param))

    if len(final_selected) != len(original_params):
        logging.error("Mismatch: originally saved vs. final param count. Verification may fail.")
        return False

    watermark_pattern = generate_watermark_pattern(watermark_key, len(final_selected))

    all_good = True
    for i, ((param_name, param_tensor)) in enumerate(final_selected):
        bit = watermark_pattern[i]
        expected_delta = perturbation_strength * (1 if bit == 1 else -1)

        old_val = original_params[param_name]
        new_val = param_tensor.detach().cpu().numpy()

        # difference = final - old_val
        actual_delta = new_val - old_val

        diff_from_expected = np.abs(actual_delta - expected_delta)
        max_diff = np.max(diff_from_expected)
        if max_diff > tolerance:
            logging.error(
                f"Param {param_name} exceeded tolerance: max diff {max_diff} > tolerance {tolerance}"
            )
            all_good = False
            break

    if all_good:
        logging.info("Parameter-perturbation watermark verified (relative check).")
    else:
        logging.error("Parameter-perturbation watermark NOT verified (relative check).")

    return all_good
