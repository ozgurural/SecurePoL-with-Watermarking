# watermark_embedding.py
import torch
from torch.utils.data import DataLoader, TensorDataset


def prepare_watermark_data():
    num_samples = 100
    input_size = (3, 32, 32)
    watermark_inputs = torch.randn(num_samples, *input_size)
    watermark_targets = torch.zeros(num_samples, dtype=torch.long)
    watermark_dataset = TensorDataset(watermark_inputs, watermark_targets)
    watermark_loader = DataLoader(watermark_dataset, batch_size=10, shuffle=False)
    return watermark_loader

def embed_watermark(model, optimizer, criterion, watermark_loader):
    model.train()
    for inputs, targets in watermark_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

def run_watermark_embedding(model, optimizer, criterion):
    watermark_loader = prepare_watermark_data()
    embed_watermark(model, optimizer, criterion, watermark_loader)
    print("Watermark embedding completed.")

