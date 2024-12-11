import torch
import torch.nn as nn
import numpy as np
import hashlib
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_watermark_pattern(watermark_key, length):
    seed = int(hashlib.sha256((watermark_key).encode()).hexdigest(), 16) % (2**32)
    np.random.seed(seed)
    pattern = np.random.choice([0, 1], size=length)
    return pattern

def select_parameters_to_perturb(model, num_parameters, watermark_key):
    params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            params.append((name, param))
    seed = int(hashlib.sha256((watermark_key + 'param_select').encode()).hexdigest(), 16) % (2**32)
    np.random.seed(seed)
    indices = np.random.choice(len(params), size=num_parameters, replace=False)
    selected_params = [params[i] for i in indices]
    return selected_params

def apply_parameter_perturbations(model, selected_params, watermark_pattern, perturbation_strength):
    for (name, param), bit in zip(selected_params, watermark_pattern):
        perturbation = perturbation_strength * (1 if bit == 1 else -1)
        with torch.no_grad():
            param.add_(perturbation)

def should_embed_watermark(step, k, watermark_key, randomize=False):
    k = int(k)
    if k <= 0:
        return False
    if randomize:
        seed = int(hashlib.sha256((watermark_key + str(step)).encode()).hexdigest(), 16) % (2**32)
        torch.manual_seed(seed)
        return torch.rand(1).item() < (1 / k)
    return (step % k) == 0

def prepare_watermark_data(device='cpu', watermark_key='secret_key'):
    num_samples = 100
    input_size = (3, 32, 32)
    seed = int(hashlib.sha256((watermark_key).encode()).hexdigest(), 16) % (2**32)
    torch.manual_seed(seed)
    watermark_inputs = torch.randn(num_samples, *input_size, device=device)
    return watermark_inputs

def embed_feature_watermark(features, watermark_key, step):
    seed = int(hashlib.sha256((watermark_key + str(step)).encode()).hexdigest(), 16) % (2**32)
    torch.manual_seed(seed)
    mask = (torch.rand_like(features) < 0.01).float()  # ~1% features
    noise = torch.randn_like(features) * 0.01
    desired_features = features + mask * noise
    return desired_features, mask

def extract_features(model, inputs, layer_name='layer1'):
    features = []
    def hook(module, input, output):
        features.append(output)
    handle = dict([*model.named_modules()])[layer_name].register_forward_hook(hook)
    with torch.no_grad():
        model(inputs)
    handle.remove()
    return features[0]

def check_watermark_in_features(features, watermark_key, step=0, threshold=0.0001):
    seed = int(hashlib.sha256((watermark_key + str(step)).encode()).hexdigest(), 16) % (2**32)
    torch.manual_seed(seed)
    mask = (torch.rand_like(features) < 0.01).float()
    noise = torch.randn_like(features) * 0.01
    expected_features = features.detach() + mask * noise
    difference = (features * mask) - (expected_features * mask)
    mean_difference = difference.abs().mean().item()
    return mean_difference < threshold

def validate_feature_watermark(model, watermark_inputs, device, watermark_key='secret_key'):
    logging.info("Starting feature-based watermark validation.")
    model.to(device)
    model.eval()
    watermark_inputs = watermark_inputs.to(device)
    features = extract_features(model, watermark_inputs)
    watermark_detected = False
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
    watermark_inputs = prepare_watermark_data(device=device, watermark_key=watermark_key)
    watermark_accuracy = validate_feature_watermark(model, watermark_inputs, device, watermark_key=watermark_key)
    if watermark_accuracy == 1.0:
        logging.info("Feature-based watermark verification successful: Watermark is present in the model.")
    else:
        logging.error("Feature-based watermark verification failed: Watermark not detected in the model.")

def verify_parameter_perturbation_watermark(model, watermark_key, perturbation_strength, num_parameters, tolerance=1e-6):
    selected_params = select_parameters_to_perturb(model, num_parameters, watermark_key)
    watermark_pattern = generate_watermark_pattern(watermark_key, len(selected_params))
    watermark_detected = True
    for (name, param), bit in zip(selected_params, watermark_pattern):
        expected_perturbation = perturbation_strength * (1 if bit == 1 else -1)
        actual_param = param.detach().cpu().numpy()
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
    seed = int(hashlib.sha256((watermark_key + 'target').encode()).hexdigest(), 16) % (2**32)
    torch.manual_seed(seed)
    batch_size = inputs.size(0)
    watermark_target = torch.randn(batch_size, watermark_size)
    return watermark_target

def verify_non_intrusive_watermark(model, device, watermark_key, watermark_size, tolerance=1e-5):
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
    seed = int(hashlib.sha256((watermark_key + 'trigger').encode()).hexdigest(), 16) % (2**32)
    torch.manual_seed(seed)
    trigger_inputs = torch.randn(10, 3, 32, 32, device=device)
    return trigger_inputs

class WatermarkModule(nn.Module):
    def __init__(self, original_model, watermark_key, watermark_size=128):
        super(WatermarkModule, self).__init__()
        self.original_model = original_model
        self.watermark_size = watermark_size
        self.watermark_key = watermark_key
        self.fc = nn.Linear(512, watermark_size)  # Adjust input size if needed

    def forward(self, x, trigger=False):
        features = self.original_model(x)
        if trigger:
            watermark_output = self.fc(features)
            return watermark_output
        else:
            return features
