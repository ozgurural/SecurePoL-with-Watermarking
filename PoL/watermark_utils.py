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
    Select a subset of trainable parameters (no. = num_parameters) from the model.
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

def apply_parameter_perturbations(model, selected_params, watermark_pattern, perturbation_strength):
    """
    Apply small additive or subtractive perturbations (±perturbation_strength).
    IMPORTANT: Ensure this is called AFTER `optimizer.step()` to avoid in-place
               modification issues during gradient computation.
    """
    for (name, param), bit in zip(selected_params, watermark_pattern):
        perturbation = perturbation_strength * (1 if bit == 1 else -1)
        with torch.no_grad():
            param.add_(perturbation)

def should_embed_watermark(step, k, watermark_key, randomize=False):
    """
    Decide if a watermark should be embedded at this training step (based on k).
    If randomize=True, a random check is performed with seed from watermark_key + step.
    """
    k = int(k)
    if k <= 0:
        return False
    if randomize:
        seed = int(hashlib.sha256((watermark_key + str(step)).encode()).hexdigest(), 16) % (2**32)
        torch.manual_seed(seed)
        return torch.rand(1).item() < (1 / k)
    return (step % k) == 0

def prepare_watermark_data(device='cpu', watermark_key='secret_key'):
    """
    Prepare a fixed set of inputs for testing or verifying a watermark (feature-based).
    """
    num_samples = 100
    input_size = (3, 32, 32)
    seed = int(hashlib.sha256(watermark_key.encode()).hexdigest(), 16) % (2**32)
    torch.manual_seed(seed)
    watermark_inputs = torch.randn(num_samples, *input_size, device=device)
    return watermark_inputs

def embed_feature_watermark(features, watermark_key, step):
    """
    Randomly embed a watermark in the features by adding noise to ~1% of them.
    """
    seed = int(hashlib.sha256((watermark_key + str(step)).encode()).hexdigest(), 16) % (2**32)
    torch.manual_seed(seed)
    mask = (torch.rand_like(features) < 0.01).float()  # ~1% mask
    noise = torch.randn_like(features) * 0.01
    desired_features = features + mask * noise
    return desired_features, mask

def extract_features(model, inputs, layer_name='layer1'):
    """
    Run a forward pass up to layer_name and collect the output (features).
    """
    features = []

    def hook(module, input, output):
        features.append(output)

    # Register a forward hook on the specified layer
    hook_handle = dict(model.named_modules())[layer_name].register_forward_hook(hook)
    with torch.no_grad():
        model(inputs)
    hook_handle.remove()
    return features[0]

def check_watermark_in_features(features, watermark_key, step=0, threshold=0.0001):
    """
    Recompute the same mask+noise for 'step' and check difference from actual features.
    If the mean difference < threshold, we consider the watermark 'detected'.
    """
    seed = int(hashlib.sha256((watermark_key + str(step)).encode()).hexdigest(), 16) % (2**32)
    torch.manual_seed(seed)
    mask = (torch.rand_like(features) < 0.01).float()
    noise = torch.randn_like(features) * 0.01
    expected_features = features.detach() + mask * noise

    difference = (features * mask) - (expected_features * mask)
    mean_difference = difference.abs().mean().item()
    return mean_difference < threshold

def validate_feature_watermark(model, watermark_inputs, device, watermark_key='secret_key'):
    """
    Validate presence of a 'feature-based' watermark by checking differences at steps 0 and 1000.
    """
    logging.info("Starting feature-based watermark validation.")
    model.to(device)
    model.eval()

    watermark_inputs = watermark_inputs.to(device)
    features = extract_features(model, watermark_inputs)
    watermark_detected = False

    # By default, we check step=0 and step=1000 to see if the watermark is embedded
    for step in [0, 1000]:
        detected = check_watermark_in_features(features, watermark_key, step=step)
        if detected:
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
    A convenience wrapper that prepares watermark data and runs validate_feature_watermark.
    """
    watermark_inputs = prepare_watermark_data(device=device, watermark_key=watermark_key)
    watermark_accuracy = validate_feature_watermark(model, watermark_inputs, device, watermark_key)
    if watermark_accuracy == 1.0:
        logging.info("Feature-based watermark verification successful: Watermark is present.")
    else:
        logging.error("Feature-based watermark verification failed: No watermark detected.")

def verify_parameter_perturbation_watermark(
    model, watermark_key, perturbation_strength, num_parameters, tolerance=1e-6
):
    """
    Check if the expected additive (±perturbation_strength) changes in the model's parameters
    are present. If all differences are within 'tolerance', we consider the watermark 'detected'.
    """
    selected_params = select_parameters_to_perturb(model, num_parameters, watermark_key)
    watermark_pattern = generate_watermark_pattern(watermark_key, len(selected_params))

    watermark_detected = True
    for (name, param), bit in zip(selected_params, watermark_pattern):
        expected_perturbation = perturbation_strength * (1 if bit == 1 else -1)
        actual_param = param.detach().cpu().numpy()

        # Compare difference from an assumed 'original' param of zero
        difference = actual_param - expected_perturbation
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
    Generate target features for 'non_intrusive' watermarking module.
    """
    seed = int(hashlib.sha256((watermark_key + 'target').encode()).hexdigest(), 16) % (2**32)
    torch.manual_seed(seed)
    batch_size = inputs.size(0)
    return torch.randn(batch_size, watermark_size)

def verify_non_intrusive_watermark(model, device, watermark_key, watermark_size, tolerance=1e-5):
    """
    Run the model in 'trigger' mode to see if the hidden watermark output
    matches the expected watermark target within 'tolerance'.
    """
    model.to(device)
    model.eval()
    trigger_inputs = generate_trigger_inputs(watermark_key, device)
    with torch.no_grad():
        watermark_output = model(trigger_inputs, trigger=True)

    expected_output = generate_watermark_target(trigger_inputs, watermark_key, watermark_size)
    difference = torch.mean((watermark_output - expected_output) ** 2).item()

    if difference < tolerance:
        logging.info("Non-intrusive watermark detected.")
        return True
    else:
        logging.error("Non-intrusive watermark not detected.")
        return False

def generate_trigger_inputs(watermark_key, device):
    """
    Generate specially seeded inputs that cause the 'non_intrusive' watermark logic to activate.
    """
    seed = int(hashlib.sha256((watermark_key + 'trigger').encode()).hexdigest(), 16) % (2**32)
    torch.manual_seed(seed)
    return torch.randn(10, 3, 32, 32, device=device)

class WatermarkModule(nn.Module):
    """
    Non-intrusive watermark module: Wrap an existing model and add a side-channel for watermark checks.
    """
    def __init__(self, original_model, watermark_key, watermark_size=128):
        super(WatermarkModule, self).__init__()
        self.original_model = original_model
        self.watermark_size = watermark_size
        self.watermark_key = watermark_key
        self.fc = nn.Linear(512, watermark_size)  # Adjust if your model outputs different dimension

    def forward(self, x, trigger=False):
        # Forward through the original model to get features
        features = self.original_model(x)
        if trigger:
            # Produce watermark output
            return self.fc(features)
        else:
            # Normal forward pass
            return features
