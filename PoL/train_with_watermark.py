import argparse
import os
import hashlib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import utils
import logging
import random
import json

from watermark_utils import (
    prepare_watermark_data,
    validate_feature_watermark,
    generate_watermark_pattern,
    select_parameters_to_perturb,
    store_original_params,  # newly introduced to store param originals
    apply_parameter_perturbations,
    should_embed_watermark,
    WatermarkModule,
    embed_feature_watermark,
    generate_watermark_target,
    verify_parameter_perturbation_watermark,
    verify_non_intrusive_watermark,
    run_feature_based_watermark_verification
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def _weights_init(m):
    """Kaiming initialization for Linear or Conv2d layers."""
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)


def train(lr, batch_size, epochs, dataset, architecture, exp_id=None, sequence=None,
          model_dir=None, save_freq=None, num_gpu=torch.cuda.device_count(), verify=False,
          dec_lr=None, half=False, resume=False, lambda_wm=0.01, k=100, randomize=False,
          watermark_key="secret_key", watermark_method='feature_based',
          num_parameters=1000, perturbation_strength=1e-5, watermark_size=128):
    """
    Train a model with optional watermarking. If 'parameter_perturbation' is used,
    we store original param values each time we embed. This allows final detection
    to compare actual param shift vs. original param values, rather than zero.
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    # --- Load training data ---
    trainset = utils.load_dataset(dataset, train=True, augment=False)
    logging.info(f"Dataset loaded with {len(trainset)} samples.")

    if batch_size <= 0:
        raise ValueError("Batch size must be a positive integer.")

    if num_gpu > torch.cuda.device_count():
        logging.warning(f"Requested {num_gpu} GPUs, but only {torch.cuda.device_count()} are available.")
        num_gpu = torch.cuda.device_count()

    if num_gpu > 1:
        batch_size *= num_gpu
        logging.info(f"Adjusted batch size for multiple GPUs: {batch_size}")

    # Build model
    net = architecture()
    net.apply(_weights_init)

    # For non-intrusive, wrap
    if watermark_method == 'non_intrusive':
        net = WatermarkModule(net, watermark_key, watermark_size=watermark_size)

    net.to(device)

    # --- Choose optimizer/scheduler ---
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

    # Load from checkpoint if provided
    if model_dir is not None:
        state = torch.load(model_dir, map_location=device)
        net.load_state_dict(state['net'])
        optimizer.load_state_dict(state['optimizer'])
        if scheduler is not None and 'scheduler' in state and state['scheduler'] is not None:
            scheduler.load_state_dict(state['scheduler'])
        logging.info(f"Loaded model from {model_dir}")

        if half:
            net.half().float()

    # If no sequence, generate one
    if sequence is None:
        train_size = len(trainset)
        indices = np.arange(train_size)
        np.random.shuffle(indices)
        sequence = np.tile(indices, epochs)
        logging.info(f"Generated training sequence with length {len(sequence)}")

    # Reproducibility
    seed = 777
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Setup proof-of-learning directory if needed
    if save_freq and save_freq > 0:
        save_dir = os.path.join("proof", f"{dataset}_{exp_id}")
        os.makedirs(save_dir, exist_ok=True)

        # Hash data subset
        m = hashlib.sha256()
        if hasattr(trainset, 'data'):
            data = trainset.data[sequence]
        elif hasattr(trainset, 'train_data'):
            data = trainset.train_data[sequence]
        else:
            raise AttributeError("Dataset missing 'data' or 'train_data'.")

        if isinstance(data, np.ndarray):
            m.update(data.tobytes())
        elif isinstance(data, list):
            for d in data:
                m.update(np.array(d).tobytes())
        else:
            raise TypeError("Unsupported data type for hashing.")

        hashed_val = m.hexdigest()
        logging.info(f"Computed hash during training: {hashed_val}")

        with open(os.path.join(save_dir, "hash.txt"), "w") as f:
            f.write(hashed_val)
        np.save(os.path.join(save_dir, "indices.npy"), sequence)
        logging.info("Saved dataset hash & sequence for PoL.")

        # Watermark Info
        watermark_info = {
            'watermark_key': watermark_key,
            'lambda_wm': lambda_wm,
            'k': k,
            'randomize': randomize,
            'seed': seed,
            'watermark_method': watermark_method,
            'num_parameters': num_parameters,
            'perturbation_strength': perturbation_strength,
            'watermark_size': watermark_size
        }
        with open(os.path.join(save_dir, "watermark_info.json"), "w") as f:
            json.dump(watermark_info, f)
        logging.info("Saved watermark_info.json.")

    else:
        save_dir = None

    sequence = np.reshape(sequence, -1)
    subset = torch.utils.data.Subset(trainset, sequence)
    trainloader = torch.utils.data.DataLoader(
        subset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    # Logging
    logging.info(f"Model architecture: {architecture.__name__}")
    logging.info(f"Learning Rate: {lr}")
    logging.info(f"Batch Size: {batch_size}")
    logging.info(f"Epochs: {epochs}")
    logging.info(f"Optimizer: {optimizer.__class__.__name__}")
    if scheduler:
        logging.info(f"Scheduler: {scheduler.__class__.__name__} with dec_lr={dec_lr}")

    # For parameter_perturbation, we store original param values each time we embed
    original_param_dict = {}

    if save_dir is not None:
        init_state = {
            'net': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict() if scheduler else None,
            'step': 0
        }
        torch.save(init_state, os.path.join(save_dir, "model_step_0"))
        logging.info("Saved initial checkpoint at step 0")

    current_step = 0
    total_steps = len(trainloader) * epochs

    # Training Loop
    for epoch in range(epochs):
        logging.info(f"Starting epoch {epoch+1}/{epochs}")
        net.train()

        for batch_idx, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            # Forward pass
            if watermark_method == 'none':
                # Baseline
                outputs = net(inputs) if not isinstance(net, WatermarkModule) else net(inputs, trigger=False)
                loss = criterion(outputs, labels)

            elif watermark_method == 'feature_based':
                features_list = []

                def forward_hook(module, inp, out):
                    features_list.append(out)

                if isinstance(net, nn.DataParallel):
                    handle = net.module.layer1.register_forward_hook(forward_hook)
                else:
                    handle = net.layer1.register_forward_hook(forward_hook)

                outputs = net(inputs)
                handle.remove()

                loss = criterion(outputs, labels)
                if lambda_wm > 0 and should_embed_watermark(current_step, k, watermark_key, randomize):
                    fts = features_list[0]
                    desired, mask = embed_feature_watermark(fts, watermark_key, current_step)
                    wm_loss = nn.MSELoss()(fts * mask, desired * mask)
                    loss += lambda_wm * wm_loss
                    logging.info(f"Feature-based watermark computed at step {current_step}")

            elif watermark_method == 'parameter_perturbation':
                # No shift yet. Just forward
                outputs = net(inputs)
                loss = criterion(outputs, labels)

            elif watermark_method == 'non_intrusive':
                # Normal pass unless trigger is True
                outputs = net(inputs, trigger=False)
                loss = criterion(outputs, labels)

            else:
                raise ValueError(f"Unknown watermark method: {watermark_method}")

            # Backprop + update
            loss.backward()
            optimizer.step()

            # If parameter-perturbation embedding is due, do it post-optimizer
            if (
                watermark_method == 'parameter_perturbation'
                and lambda_wm > 0
                and should_embed_watermark(current_step, k, watermark_key, randomize)
            ):
                # We pick subset of parameters
                sel_params = select_parameters_to_perturb(net, num_parameters, watermark_key)
                # Store their original values
                temp_orig = store_original_params(sel_params)
                # Generate pattern + apply shift
                wpat = generate_watermark_pattern(watermark_key, len(sel_params))
                apply_parameter_perturbations(sel_params, wpat, perturbation_strength)

                # Merge into global dict, so final detection can see the changes
                for pid, tens_copy in temp_orig.items():
                    original_param_dict[pid] = tens_copy

                logging.info(f"Parameter perturbation watermark applied at step {current_step}, post-optimizer.")

            current_step += 1

            # Save checkpoint periodically
            if save_dir and current_step % save_freq == 0:
                ckpt = {
                    'net': net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict() if scheduler else None,
                    'step': current_step
                }
                cp_path = os.path.join(save_dir, f"model_step_{current_step}")
                torch.save(ckpt, cp_path)
                logging.info(f"Saved checkpoint at step {current_step}")

        # Step scheduler at each epoch
        if scheduler:
            scheduler.step()
            logging.info(f"Scheduler stepped at epoch {epoch+1}/{epochs}")

    # Final checkpoint
    if save_dir:
        final_st = {
            'net': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict() if scheduler else None,
            'step': current_step
        }
        final_ckpt = os.path.join(save_dir, f"model_step_{current_step}")
        torch.save(final_st, final_ckpt)
        logging.info(f"Saved final model checkpoint at step {current_step}")

    # If param-perturbation watermark was used, we can verify it now using our original_param_dict
    if watermark_method == 'parameter_perturbation' and len(original_param_dict) > 0:
        verified = verify_parameter_perturbation_watermark(
            model=net,
            watermark_key=watermark_key,
            perturbation_strength=perturbation_strength,
            num_parameters=num_parameters,
            tolerance=1e-6,
            original_param_dict=original_param_dict
        )
        if verified:
            logging.info("Parameter-perturbation watermark verified at end of training.")
        else:
            logging.error("Parameter-perturbation watermark not verified at end of training.")

    return net, optimizer, criterion


def validate(dataset, model, batch_size=128):
    """
    Validate the trained model on the specified dataset.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    testset = utils.load_dataset(dataset, train=False, augment=False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False,
                                             num_workers=4, pin_memory=True)

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

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100.0 * correct / total
    logging.info(f"Validation Accuracy: {acc:.2f}%")
    return correct / total


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training script with or without watermarking.")

    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--dataset', type=str, default="CIFAR10")
    parser.add_argument('--model', type=str, default="resnet20")
    parser.add_argument('--id', type=str, default='Batch100')
    parser.add_argument('--save-freq', type=int, default=1000)
    parser.add_argument('--num-gpu', type=int, default=torch.cuda.device_count())
    parser.add_argument('--milestone', nargs='+', type=int, default=None)
    parser.add_argument('--verify', type=int, default=0)
    parser.add_argument('--lambda-wm', type=float, default=0.01)
    parser.add_argument('--k', type=int, default=1000)
    parser.add_argument('--randomize', action='store_true')
    parser.add_argument('--watermark-key', type=str, default='secret_key')
    parser.add_argument('--watermark-method', type=str, default='feature_based',
                        choices=['none', 'feature_based', 'parameter_perturbation', 'non_intrusive'])
    parser.add_argument('--num-parameters', type=int, default=1000)
    parser.add_argument('--perturbation-strength', type=float, default=1e-5)
    parser.add_argument('--watermark-size', type=int, default=128)

    args = parser.parse_args()

    # Reproducibility
    seed = 777
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    t1 = time.time()
    logging.info(f"Trying to allocate {args.num_gpu} GPUs")

    # Dynamically load model architecture
    try:
        import model
        architecture = getattr(model, args.model)
    except AttributeError:
        import torchvision.models as tv_models
        architecture = getattr(tv_models, args.model, None)
        if architecture is None:
            raise ValueError(f"Model {args.model} not found in model.py or torchvision.models.")

    # --- Train ---
    trained_model, optimizer, criterion = train(
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        dataset=args.dataset,
        architecture=architecture,
        exp_id=args.id,
        save_freq=args.save_freq,
        num_gpu=args.num_gpu,
        dec_lr=args.milestone,
        verify=bool(args.verify),
        resume=False,
        lambda_wm=args.lambda_wm,
        k=args.k,
        randomize=args.randomize,
        watermark_key=args.watermark_key,
        watermark_method=args.watermark_method,
        num_parameters=args.num_parameters,
        perturbation_strength=args.perturbation_strength,
        watermark_size=args.watermark_size
    )

    # Validate final model
    validate(args.dataset, trained_model)

    t2 = time.time()
    logging.info(f"Total training time: {t2 - t1:.2f} seconds")
