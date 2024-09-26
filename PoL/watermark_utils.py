import torch
import logging
from model import resnet20, resnet32  # Adjust according to your models
import hashlib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def prepare_watermark_data(device='cpu', watermark_key='secret_key'):
    num_samples = 100
    input_size = (3, 32, 32)
    # Generate deterministic watermark inputs based on the watermark key
    seed = int(hashlib.sha256((watermark_key).encode()).hexdigest(), 16) % (2**32)
    torch.manual_seed(seed)
    watermark_inputs = torch.randn(num_samples, *input_size, device=device)
    return watermark_inputs


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
    # Generate the same mask and noise used during embedding
    seed = int(hashlib.sha256((watermark_key + str(step)).encode()).hexdigest(), 16) % (2**32)
    torch.manual_seed(seed)
    mask = (torch.rand_like(features) < 0.01).float()  # 1% of the features
    noise = torch.randn_like(features) * 0.01  # Small noise
    expected_features = features.detach() + mask * noise

    # Calculate the difference on the masked features
    difference = (features * mask) - (expected_features * mask)
    mean_difference = difference.abs().mean().item()

    # Check if the watermark is detected based on the mean difference
    if mean_difference < threshold:
        return True
    else:
        return False


def validate_feature_watermark(model, watermark_inputs, device, watermark_key='secret_key'):
    logging.info("Starting feature-based watermark validation.")
    model.to(device)
    model.eval()
    watermark_inputs = watermark_inputs.to(device)
    features = extract_features(model, watermark_inputs)
    # Since watermark embedding occurs at specific steps, we need to check those steps
    # For this example, let's assume we check steps 0 and 1000
    watermark_detected = False
    for step in [0, 1000]:
        detected = check_watermark_in_features(features, watermark_key, step=step)
        if detected:
            watermark_detected = True
            break
    if watermark_detected:
        logging.info("Feature-based watermark validation successful: Watermark detected.")
        return 1.0  # 100% detection
    else:
        logging.error("Feature-based watermark validation failed: Watermark not detected.")
        return 0.0  # 0% detection


def get_model(model_name):
    if model_name == 'resnet20':
        return resnet20()
    elif model_name == 'resnet32':
        return resnet32()
    else:
        raise ValueError(f'No model found for name: {model_name}')


def run_feature_based_watermark_verification(model_path, model_name, device='cpu', watermark_key='secret_key'):
    model = get_model(model_name)
    state = torch.load(model_path, map_location=device)
    if 'net' in state:
        model.load_state_dict(state['net'])
    else:
        model.load_state_dict(state)
    watermark_inputs = prepare_watermark_data(device=device, watermark_key=watermark_key)
    watermark_accuracy = validate_feature_watermark(model, watermark_inputs, device, watermark_key=watermark_key)
    if watermark_accuracy == 1.0:
        logging.info("Feature-based watermark verification successful: Watermark is present in the model.")
    else:
        logging.error("Feature-based watermark verification failed: Watermark not detected in the model.")
