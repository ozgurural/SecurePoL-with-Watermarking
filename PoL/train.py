import argparse
import os
import hashlib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
import time
import utils
import logging
import model as custom_model
from watermark_train import prepare_watermark_data, validate_feature_watermark

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def embed_feature_watermark(features, watermark_key):
    """
    Embed a watermark in the feature map by modifying certain entries in the features tensor
    according to a predetermined key (watermark_key).
    """
    with torch.no_grad():
        # Generate a deterministic random mask based on the watermark key
        seed = int(hashlib.sha256(watermark_key.encode()).hexdigest(), 16) % (2**32)
        torch.manual_seed(seed)
        mask = (torch.rand_like(features) < 0.01)  # 1% of the features
        noise = torch.randn_like(features) * 0.01  # Small noise
        features = features + mask * noise
    return features


def should_embed_watermark(step, k, randomize=False):
    """
    Determine whether to embed the watermark at this step. Watermarking is done
    every k steps (batches), or randomly if `randomize` is set to True.
    """
    k = int(k)  # Ensure k is an integer
    if k <= 0:
        return False  # Disable watermarking if k is not positive
    if randomize:
        return torch.rand(1).item() < (1 / k)
    return (step % k) == 0


def train(lr, batch_size, epochs, dataset, architecture, exp_id=None, sequence=None,
          model_dir=None, save_freq=None, num_gpu=torch.cuda.device_count(), verify=False,
          dec_lr=None, half=False, resume=False, lambda_wm=0.01, k=100, randomize=False):
    """
    Training function with optional feature-based watermark embedding.
    """
    k = int(k)  # Ensure k is an integer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    try:
        trainset = utils.load_dataset(dataset, True)
    except Exception as e:
        logging.error(f"Failed to load dataset '{dataset}': {e}")
        raise e

    if batch_size <= 0:
        logging.error("Batch size must be a positive integer.")
        raise ValueError("Batch size must be a positive integer.")

    if num_gpu > torch.cuda.device_count():
        logging.warning(f"Requested {num_gpu} GPUs, but only {torch.cuda.device_count()} are available.")
        num_gpu = torch.cuda.device_count()

    if num_gpu > 1:
        batch_size = batch_size * num_gpu
        logging.info(f"Adjusted batch size for multiple GPUs: {batch_size}")

    net = architecture()
    net.apply(_weights_init)  # Apply your custom initialization

    net.to(device)

    # Initialize optimizer and scheduler
    if dataset == 'MNIST':
        optimizer = optim.SGD(net.parameters(), lr=lr)
        scheduler = None
    elif dataset == 'CIFAR10':
        if dec_lr is None:
            dec_lr = [100, 150]
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=dec_lr, gamma=0.1)
    elif dataset == 'CIFAR100':
        if dec_lr is None:
            dec_lr = [60, 120, 160]
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=dec_lr, gamma=0.2)
    else:
        optimizer = optim.Adam(net.parameters(), lr=lr)
        scheduler = None

    criterion = nn.CrossEntropyLoss().to(device)

    if model_dir is not None:
        # Load a pre-trained model from model_dir if it is given
        try:
            state = torch.load(model_dir, map_location=device)
            net.load_state_dict(state['net'])
            optimizer.load_state_dict(state['optimizer'])
            if scheduler is not None and 'scheduler' in state:
                scheduler.load_state_dict(state['scheduler'])
            logging.info(f"Loaded model from {model_dir}")
        except Exception as e:
            logging.error(f"Failed to load model from {model_dir}: {e}")
            raise e

        if half:
            net.half().float()

    if sequence is None:
        # If a training sequence is not given, create a new one
        train_size = len(trainset)
        indices = np.arange(train_size)
        np.random.shuffle(indices)
        sequence = np.tile(indices, epochs)
        logging.info(f"Generated training sequence with length {len(sequence)}")

    # Handle checkpoint saving directory
    if save_freq is not None and save_freq > 0:
        save_dir = os.path.join("proof", f"{dataset}_{exp_id}")
        os.makedirs(save_dir, exist_ok=True)
        # Save the hash of the dataset
        m = hashlib.sha256()
        if hasattr(trainset, 'data'):
            data = trainset.data
        elif hasattr(trainset, 'train_data'):
            data = trainset.train_data
        else:
            raise AttributeError("Dataset object has no attribute 'data' or 'train_data'.")

        for d in data:
            m.update(str(d).encode('utf-8'))
        with open(os.path.join(save_dir, "hash.txt"), "w") as f:
            f.write(m.hexdigest())
        logging.info("Saved dataset hash to hash.txt")

    # Prepare the DataLoader
    sequence = np.reshape(sequence, -1)
    subset = torch.utils.data.Subset(trainset, sequence)
    trainloader = torch.utils.data.DataLoader(subset, batch_size=batch_size, num_workers=4, pin_memory=True)

    # Prepare watermark data (not needed for feature-based watermarking but keeping for consistency)
    watermark_loader = prepare_watermark_data()

    # Log model and optimizer details
    logging.info(f"Model architecture: {architecture.__name__}")
    logging.info(f"Learning Rate: {lr}")
    logging.info(f"Batch Size: {batch_size}")
    logging.info(f"Epochs: {epochs}")
    logging.info(f"Optimizer: {optimizer.__class__.__name__}")
    if scheduler is not None:
        logging.info(f"Scheduler: {scheduler.__class__.__name__} with milestones {dec_lr} and gamma {scheduler.gamma}")

    # Save the initial model state as model_step_0
    if save_freq is not None and save_freq > 0:
        initial_state = {
            'net': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict() if scheduler else None,
            'step': 0,
        }
        torch.save(initial_state, os.path.join(save_dir, "model_step_0"))
        logging.info("Saved initial model checkpoint at step 0")

    # Training loop
    current_step = 0  # Counts the number of batches processed
    total_steps = len(trainloader) * epochs
    for epoch in range(epochs):
        logging.info(f"Starting epoch {epoch + 1}/{epochs}")
        for batch_idx, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            # Feature-based watermark embedding
            if lambda_wm > 0 and should_embed_watermark(current_step, k, randomize):
                # Define forward hook to embed watermark
                def forward_hook(module, input, output):
                    return embed_feature_watermark(output, watermark_key="secret_key")

                # Register the hook on the desired layer (e.g., 'layer1' for ResNet)
                if isinstance(net, nn.DataParallel):
                    handle = net.module.layer1.register_forward_hook(forward_hook)
                else:
                    handle = net.layer1.register_forward_hook(forward_hook)

                outputs = net(inputs)
                handle.remove()
                logging.info(f"Feature-based watermark embedded at step {current_step}")
            else:
                outputs = net(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if scheduler is not None:
                scheduler.step()

            current_step += 1  # Increment the batch counter

            # Save checkpoints at specified frequency
            if save_freq is not None and current_step % save_freq == 0:
                checkpoint_state = {
                    'net': net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict() if scheduler else None,
                    'step': current_step,
                }
                checkpoint_path = os.path.join(save_dir, f"model_step_{current_step}")
                torch.save(checkpoint_state, checkpoint_path)
                logging.info(f"Saved checkpoint at step {current_step}")

            # Optional verification/validation at specific intervals
            if verify and current_step % save_freq == 0:
                logging.info(f'Verifying at step {current_step}')
                validate(dataset, net, batch_size)

    # Save final model
    if save_freq is not None and save_freq > 0:
        final_state = {
            'net': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict() if scheduler else None,
            'step': current_step,
        }
        final_checkpoint_path = os.path.join(save_dir, f"model_step_{current_step}")
        torch.save(final_state, final_checkpoint_path)
        logging.info(f"Saved final model checkpoint at step {current_step}")

    return net, optimizer, criterion


def validate(dataset, model, batch_size=128):
    """
    Validate the model on the test dataset.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    testset = utils.load_dataset(dataset, False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=4, pin_memory=True)
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    logging.info(f'Validation Accuracy: {accuracy:.2f}%')
    return correct / total


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training Script with Feature-Based Watermarking")
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=2, help='Number of training epochs')
    parser.add_argument('--dataset', type=str, default="CIFAR10", help='Dataset to use (e.g., CIFAR10, CIFAR100, MNIST)')
    parser.add_argument('--model', type=str, default="resnet20",
                        help="Models defined in model.py or any torchvision model.")
    parser.add_argument('--id', help='Experiment ID', type=str, default='Batch100')
    parser.add_argument('--save-freq', type=int, default=100, help='Frequency of saving checkpoints (in steps)')
    parser.add_argument('--num-gpu', type=int, default=torch.cuda.device_count(),
                        help='Number of GPUs to use')
    parser.add_argument('--milestone', nargs='+', type=int, default=[100, 150],
                        help='Milestones for learning rate scheduler')
    parser.add_argument('--verify', type=int, default=1, help='Enable verification during training (1 for True, 0 for False)')
    parser.add_argument('--lambda-wm', type=float, default=0.01, help='Balancing parameter for watermark loss')
    parser.add_argument('--k', type=int, default=100, help='Watermark embedding frequency (in steps)')
    parser.add_argument('--randomize', action='store_true', help='Randomize watermark embedding intervals')
    arg = parser.parse_args()

    # Set random seeds for reproducibility
    seed = 777
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random_seed = seed
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    t1 = time.time()
    logging.info(f"Trying to allocate {arg.num_gpu} GPUs")

    # Initialize architecture
    try:
        architecture = eval(f"custom_model.{arg.model}")
    except AttributeError:
        try:
            import torchvision.models as tv_models
            architecture = eval(f"tv_models.{arg.model}")
        except AttributeError as e:
            logging.error(f"Model {arg.model} not found in custom_model or torchvision.models.")
            raise e

    # Train the model
    trained_model, optimizer, criterion = train(
        lr=arg.lr,
        batch_size=arg.batch_size,
        epochs=arg.epochs,
        dataset=arg.dataset,
        architecture=architecture,
        exp_id=arg.id,
        save_freq=arg.save_freq,
        num_gpu=arg.num_gpu,
        dec_lr=arg.milestone,
        verify=bool(arg.verify),
        resume=False,
        lambda_wm=arg.lambda_wm,
        k=arg.k,
        randomize=arg.randomize
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Validate watermark
    watermark_loader = prepare_watermark_data()
    watermark_accuracy = validate_feature_watermark(trained_model, watermark_loader, device)
    logging.info(f'Watermark Detection Accuracy: {watermark_accuracy * 100:.2f}%')

    # Validate on main dataset
    validate(arg.dataset, trained_model)

    # Save the model with the embedded watermark
    model_path_with_watermark = 'model_with_watermark.pth'
    torch.save(trained_model.state_dict(), model_path_with_watermark)
    logging.info(f"Model with watermark saved at {model_path_with_watermark}")

    t2 = time.time()
    logging.info(f"Total training time: {t2 - t1:.2f} seconds")
