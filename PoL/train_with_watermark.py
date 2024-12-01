import argparse
import os
import hashlib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import utils  # Ensure utils.py is available
import logging
import random
import json
from watermark_utils import (
    prepare_watermark_data,
    validate_feature_watermark,
    generate_watermark_pattern,
    select_parameters_to_perturb,
    apply_parameter_perturbations,
    should_embed_watermark,
    WatermarkModule
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def _weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)


def train(lr, batch_size, epochs, dataset, architecture, exp_id=None, sequence=None,
          model_dir=None, save_freq=None, num_gpu=torch.cuda.device_count(), verify=False,
          dec_lr=None, half=False, resume=False, lambda_wm=0.01, k=100, randomize=False,
          watermark_key="secret_key", watermark_method='feature_based',
          num_parameters=1000, perturbation_strength=1e-5, watermark_size=128):
    k = int(k)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    # Load dataset without transformations
    trainset = utils.load_dataset(dataset, train=True, augment=False)  # Ensure augment=False
    logging.info(f"Dataset loaded with {len(trainset)} samples.")

    if batch_size <= 0:
        raise ValueError("Batch size must be a positive integer.")

    if num_gpu > torch.cuda.device_count():
        logging.warning(f"Requested {num_gpu} GPUs, but only {torch.cuda.device_count()} are available.")
        num_gpu = torch.cuda.device_count()

    if num_gpu > 1:
        batch_size = batch_size * num_gpu
        logging.info(f"Adjusted batch size for multiple GPUs: {batch_size}")

    # Initialize the model
    net = architecture()
    net.apply(_weights_init)

    # For non-intrusive watermarking, modify the model to include the watermark module
    if watermark_method == 'non_intrusive':
        net = WatermarkModule(net, watermark_key, watermark_size=watermark_size)

    net.to(device)

    # Initialize optimizer and scheduler
    if dataset == 'MNIST':
        optimizer = optim.SGD(net.parameters(), lr=lr)
        scheduler = None
    elif dataset == 'CIFAR10':
        if dec_lr is None:
            dec_lr = [epochs // 2, int(epochs * 0.75)]
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=dec_lr, gamma=0.1)
    elif dataset == 'CIFAR100':
        if dec_lr is None:
            dec_lr = [epochs // 2, int(epochs * 0.75)]
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=dec_lr, gamma=0.2)
    else:
        optimizer = optim.Adam(net.parameters(), lr=lr)
        scheduler = None

    criterion = nn.CrossEntropyLoss().to(device)

    if model_dir is not None:
        # Load pre-trained model
        state = torch.load(model_dir, map_location=device)
        net.load_state_dict(state['net'])
        optimizer.load_state_dict(state['optimizer'])
        if scheduler is not None and 'scheduler' in state and state['scheduler'] is not None:
            scheduler.load_state_dict(state['scheduler'])
        logging.info(f"Loaded model from {model_dir}")

        if half:
            net.half().float()

    if sequence is None:
        # Create training sequence
        train_size = len(trainset)
        indices = np.arange(train_size)
        np.random.shuffle(indices)
        sequence = np.tile(indices, epochs)
        logging.info(f"Generated training sequence with length {len(sequence)}")

    # Set random seeds for reproducibility
    seed = 777
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Handle checkpoint saving directory
    if save_freq is not None and save_freq > 0:
        save_dir = os.path.join("proof", f"{dataset}_{exp_id}")
        os.makedirs(save_dir, exist_ok=True)

        # Save the hash of the dataset
        m = hashlib.sha256()

        # Get the data corresponding to the sequence
        if hasattr(trainset, 'data'):
            data = trainset.data[sequence]
            labels = np.array(trainset.targets)[sequence]
        elif hasattr(trainset, 'train_data'):
            data = trainset.train_data[sequence]
            labels = np.array(trainset.train_labels)[sequence]
        else:
            raise AttributeError("Dataset object has no attribute 'data' or 'train_data'.")

        logging.info(f"Data shape: {data.shape}, Data type: {data.dtype}")
        logging.info(f"First data sample hash: {hashlib.sha256(data[0].tobytes()).hexdigest()}")

        # Compute the hash
        if isinstance(data, np.ndarray):
            m.update(data.tobytes())
        elif isinstance(data, list):
            for d in data:
                m.update(np.array(d).tobytes())
        else:
            raise TypeError("Unsupported data type for hashing.")

        computed_hash = m.hexdigest()
        logging.info(f"Computed hash during training: {computed_hash}")

        # Save the hash to 'hash.txt'
        with open(os.path.join(save_dir, "hash.txt"), "w") as f:
            f.write(computed_hash)
        logging.info("Saved dataset hash to hash.txt")

        # Save the training sequence
        np.save(os.path.join(save_dir, "indices.npy"), sequence)
        logging.info(f"Saved training sequence to indices.npy")

        # Save watermarking information
        watermark_info = {
            'watermark_key': watermark_key,
            'lambda_wm': lambda_wm,
            'k': k,
            'randomize': randomize,
            'seed': seed,  # Include the random seed used
            'watermark_method': watermark_method,
            'num_parameters': num_parameters,
            'perturbation_strength': perturbation_strength,
            'watermark_size': watermark_size
        }
        with open(os.path.join(save_dir, "watermark_info.json"), "w") as f:
            json.dump(watermark_info, f)
        logging.info("Saved watermarking information to watermark_info.json")
    else:
        save_dir = None

    # Prepare the DataLoader
    sequence = np.reshape(sequence, -1)
    subset = torch.utils.data.Subset(trainset, sequence)
    trainloader = torch.utils.data.DataLoader(subset, batch_size=batch_size, num_workers=4, pin_memory=True)

    # Log model and optimizer details
    logging.info(f"Model architecture: {architecture.__name__}")
    logging.info(f"Learning Rate: {lr}")
    logging.info(f"Batch Size: {batch_size}")
    logging.info(f"Epochs: {epochs}")
    logging.info(f"Optimizer: {optimizer.__class__.__name__}")
    if scheduler is not None:
        logging.info(f"Scheduler: {scheduler.__class__.__name__} with milestones {dec_lr} and gamma {scheduler.gamma}")

    # Save the initial model state as model_step_0
    if save_dir is not None:
        initial_state = {
            'net': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict() if scheduler else None,
            'step': 0,
        }
        torch.save(initial_state, os.path.join(save_dir, "model_step_0"))
        logging.info("Saved initial model checkpoint at step 0")

    # Training loop
    current_step = 0
    total_steps = len(trainloader) * epochs
    for epoch in range(epochs):
        logging.info(f"Starting epoch {epoch + 1}/{epochs}")
        net.train()
        for batch_idx, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            if watermark_method == 'feature_based':
                features_list = []

                # Register forward hook to extract features
                def forward_hook(module, input, output):
                    features_list.append(output)

                if isinstance(net, nn.DataParallel):
                    handle = net.module.layer1.register_forward_hook(forward_hook)
                else:
                    handle = net.layer1.register_forward_hook(forward_hook)

                outputs = net(inputs)

                # Remove the hook
                handle.remove()

                loss = criterion(outputs, labels)

                # Watermark loss
                if lambda_wm > 0 and should_embed_watermark(current_step, k, watermark_key, randomize):
                    features = features_list[0]
                    desired_features, mask = utils.embed_feature_watermark(features, watermark_key, current_step)
                    # Compute watermark loss on the masked features
                    wm_loss = nn.MSELoss()(features * mask, desired_features * mask)
                    loss += lambda_wm * wm_loss
                    logging.info(f"Feature-based watermark loss computed at step {current_step}")

            elif watermark_method == 'parameter_perturbation':
                outputs = net(inputs)
                loss = criterion(outputs, labels)

                # Watermark embedding
                if lambda_wm > 0 and should_embed_watermark(current_step, k, watermark_key, randomize):
                    selected_params = select_parameters_to_perturb(net, num_parameters, watermark_key)
                    watermark_pattern = generate_watermark_pattern(watermark_key, len(selected_params))
                    apply_parameter_perturbations(net, selected_params, watermark_pattern, perturbation_strength)
                    logging.info(f"Parameter perturbation watermark applied at step {current_step}")

            elif watermark_method == 'non_intrusive':
                outputs = net(inputs, trigger=False)
                loss = criterion(outputs, labels)

                # Watermark loss
                if lambda_wm > 0 and should_embed_watermark(current_step, k, watermark_key, randomize):
                    # Generate watermark target
                    watermark_target = utils.generate_watermark_target(inputs, watermark_key, watermark_size)
                    watermark_output = net(inputs, trigger=True)
                    wm_loss = nn.MSELoss()(watermark_output, watermark_target)
                    loss += lambda_wm * wm_loss
                    logging.info(f"Non-intrusive watermark loss computed at step {current_step}")

            else:
                raise ValueError(f"Unknown watermarking method: {watermark_method}")

            loss.backward()
            optimizer.step()

            current_step += 1

            # Save checkpoints
            if save_dir is not None and current_step % save_freq == 0:
                checkpoint_state = {
                    'net': net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict() if scheduler else None,
                    'step': current_step,
                }
                checkpoint_path = os.path.join(save_dir, f"model_step_{current_step}")
                torch.save(checkpoint_state, checkpoint_path)
                logging.info(f"Saved checkpoint at step {current_step}")

            # Optional verification
            if verify and current_step % save_freq == 0:
                logging.info(f'Verifying at step {current_step}')
                validate(dataset, net, batch_size)

        # Step the scheduler after each epoch
        if scheduler is not None:
            scheduler.step()
            logging.info(f"Scheduler stepped at epoch {epoch + 1}/{epochs}")

    # Save final model
    if save_dir is not None:
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    testset = utils.load_dataset(dataset, train=False, augment=False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=4, pin_memory=True)
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            if hasattr(model, 'module') and isinstance(model.module, WatermarkModule):
                outputs = model(inputs, trigger=False)
            elif isinstance(model, WatermarkModule):
                outputs = model(inputs, trigger=False)
            else:
                outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    logging.info(f'Validation Accuracy: {accuracy:.2f}%')
    return correct / total


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training Script with Watermarking")
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--dataset', type=str, default="CIFAR10", help='Dataset to use')
    parser.add_argument('--model', type=str, default="resnet20", help="Model architecture")
    parser.add_argument('--id', help='Experiment ID', type=str, default='Batch100')
    parser.add_argument('--save-freq', type=int, default=1000, help='Frequency of saving checkpoints (in steps)')
    parser.add_argument('--num-gpu', type=int, default=torch.cuda.device_count(), help='Number of GPUs to use')
    parser.add_argument('--milestone', nargs='+', type=int, default=None, help='Milestones for learning rate scheduler')
    parser.add_argument('--verify', type=int, default=0, help='Enable verification during training (1 for True, 0 for False)')
    parser.add_argument('--lambda-wm', type=float, default=0.01, help='Balancing parameter for watermark loss')
    parser.add_argument('--k', type=int, default=1000, help='Watermark embedding frequency (in steps)')
    parser.add_argument('--randomize', action='store_true', help='Randomize watermark embedding intervals')
    parser.add_argument('--watermark-key', type=str, default='secret_key', help='Key used for watermark embedding')
    parser.add_argument('--watermark-method', type=str, default='feature_based',
                        choices=['feature_based', 'parameter_perturbation', 'non_intrusive'],
                        help='Watermarking method to use during training')
    parser.add_argument('--num-parameters', type=int, default=1000, help='Number of parameters to perturb for parameter perturbation watermarking')
    parser.add_argument('--perturbation-strength', type=float, default=1e-5, help='Strength of parameter perturbations')
    parser.add_argument('--watermark-size', type=int, default=128, help='Size of the watermark for non-intrusive watermarking')
    arg = parser.parse_args()

    # Set random seeds for reproducibility
    seed = 777
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    t1 = time.time()
    logging.info(f"Trying to allocate {arg.num_gpu} GPUs")

    # Initialize architecture
    try:
        import model  # Import your model.py file
        architecture = getattr(model, arg.model)
    except AttributeError:
        try:
            import torchvision.models as tv_models
            architecture = getattr(tv_models, arg.model)
        except AttributeError as e:
            logging.error(f"Model {arg.model} not found in model.py or torchvision.models.")
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
        randomize=arg.randomize,
        watermark_key=arg.watermark_key,
        watermark_method=arg.watermark_method,
        num_parameters=arg.num_parameters,
        perturbation_strength=arg.perturbation_strength,
        watermark_size=arg.watermark_size
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Validate watermark
    if arg.watermark_method == 'feature_based':
        watermark_inputs = prepare_watermark_data(device=device, watermark_key=arg.watermark_key)
        watermark_accuracy = validate_feature_watermark(trained_model, watermark_inputs, device, watermark_key=arg.watermark_key)
        logging.info(f'Feature-Based Watermark Detection Accuracy: {watermark_accuracy * 100:.2f}%')

    elif arg.watermark_method == 'parameter_perturbation':
        watermark_detected = utils.verify_parameter_perturbation_watermark(
            model=trained_model,
            watermark_key=arg.watermark_key,
            perturbation_strength=arg.perturbation_strength,
            num_parameters=arg.num_parameters,
            tolerance=1e-6
        )
        if watermark_detected:
            logging.info("Parameter Perturbation Watermark detected successfully.")
        else:
            logging.error("Parameter Perturbation Watermark not detected.")

    elif arg.watermark_method == 'non_intrusive':
        watermark_detected = utils.verify_non_intrusive_watermark(
            model=trained_model,
            device=device,
            watermark_key=arg.watermark_key,
            watermark_size=arg.watermark_size,
            tolerance=1e-5
        )
        if watermark_detected:
            logging.info("Non-Intrusive Watermark detected successfully.")
        else:
            logging.error("Non-Intrusive Watermark not detected.")

    # Validate on main dataset
    validate(arg.dataset, trained_model)

    # Save the model with the embedded watermark
    model_path_with_watermark = f'model_with_{arg.watermark_method}_watermark.pth'
    torch.save(trained_model.state_dict(), model_path_with_watermark)
    logging.info(f"Model with {arg.watermark_method} watermark saved at {model_path_with_watermark}")

    t2 = time.time()
    logging.info(f"Total training time: {t2 - t1:.2f} seconds")
