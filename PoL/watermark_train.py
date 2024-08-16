# watermark_embedding.py
import torch
import logging
from torch.utils.data import DataLoader, TensorDataset

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def prepare_watermark_data():
    num_samples = 100
    input_size = (3, 32, 32)
    watermark_inputs = torch.randn(num_samples, *input_size)
    watermark_targets = torch.zeros(num_samples, dtype=torch.long)
    watermark_dataset = TensorDataset(watermark_inputs, watermark_targets)
    watermark_loader = DataLoader(watermark_dataset, batch_size=10, shuffle=False)
    return watermark_loader

def validate_watermark(model, watermark_loader, device):
    logging.info("Starting watermark validation.")
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():  # No need to track gradients
        for wm_inputs, wm_labels in watermark_loader:
            wm_inputs, wm_labels = wm_inputs.to(device), wm_labels.to(device)
            outputs = model(wm_inputs)
            _, predicted = torch.max(outputs, 1)
            total += wm_labels.size(0)
            correct += (predicted == wm_labels).sum().item()
    accuracy = correct / total
    logging.info(f'Watermark validation accuracy: {accuracy * 100:.2f}%')
    return accuracy

