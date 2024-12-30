import torch
import logging
from model import resnet20, resnet32  # Import your model constructors from the model module


def get_model(model_name):
    if model_name == 'resnet20':
        return resnet20()
    elif model_name == 'resnet32':
        return resnet32()
    else:
        raise ValueError(f'No model found for name: {model_name}')

def prepare_watermark_data(device='cpu'):
    num_samples = 100
    input_size = (3, 32, 32)
    watermark_inputs = torch.randn(num_samples, *input_size, device=device)
    return watermark_inputs

def run_feature_based_watermark_verification(model_path, model_name, device='cpu'):
    model = get_model(model_name)
    state = torch.load(model_path, map_location=device)
    if 'net' in state:
        model.load_state_dict(state['net'])
    else:
        model.load_state_dict(state)
    watermark_inputs = prepare_watermark_data(device=device)
    watermark_detected = validate_feature_watermark(model, watermark_inputs, device)
    if watermark_detected == 1.0:
        logging.info("Feature-based watermark verification successful: Watermark is present in the model.")
    else:
        logging.error("Feature-based watermark verification failed: Watermark not detected in the model.")
