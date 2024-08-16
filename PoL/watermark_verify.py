# watermark_verify.py
import torch
import logging
from torch.utils.data import DataLoader, TensorDataset
from model import resnet20, resnet32  # Import your model constructors from the model module


def get_model(model_name):
    if model_name == 'resnet20':
        return resnet20()  # Return the instantiated ResNet20 model
    elif model_name == 'resnet32':
        return resnet32()  # Return the instantiated ResNet32 model
    # Add additional model cases as needed
    else:
        raise ValueError(f'No model found for name: {model_name}')


def prepare_watermark_data(num_samples=100, input_size=(3, 32, 32), batch_size=10, device='cpu'):
    watermark_inputs = torch.randn(num_samples, *input_size, device=device)
    watermark_targets = torch.zeros(num_samples, dtype=torch.long, device=device)
    watermark_dataset = TensorDataset(watermark_inputs, watermark_targets)
    watermark_loader = DataLoader(watermark_dataset, batch_size=batch_size, shuffle=False)
    return watermark_loader


def verify_watermark(model, watermark_loader, device='cpu'):
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in watermark_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)
    accuracy = correct / total
    logging.info(f"Watermark verification accuracy: {accuracy * 100:.2f}%")
    return accuracy

def run_watermark_verification(model_path, model_name, device='cpu'):
    model = get_model(model_name)  # Ensure this properly loads the model based on the name
    model.load_state_dict(torch.load(model_path, map_location=device))
    watermark_loader = prepare_watermark_data(device=device)
    accuracy = verify_watermark(model, watermark_loader, device)
    if accuracy == 1.0:
        logging.info("Watermark verification successful: The watermark is present in the model.")
    else:
        logging.error("Watermark verification failed: The watermark could not be detected in the model.")