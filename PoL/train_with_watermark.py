#!/usr/bin/env python3
# train_with_watermark.py
# Fully updated integrated training script for Baseline PoL, Feature-based WM,
# Parameter-perturbation WM (with original param snapshot), and Non-intrusive WM.

import argparse
import hashlib
import json
import logging
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import utils  # Ensure utils.py is available and has load_dataset()
from watermark_utils import (
    generate_watermark_pattern,
    select_parameters_to_perturb,
    apply_parameter_perturbations,
    should_embed_watermark,
    WatermarkModule,
    embed_feature_watermark,
    generate_watermark_target,
    verify_non_intrusive_watermark,
    run_feature_based_watermark_verification
)

# This specialized method is for verifying param changes w.r.t. the *original param values*:

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def _weights_init(m):
    """Applies Kaiming initialization to Linear or Conv2d layers."""
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)


def train(
    lr, batch_size, epochs, dataset, architecture,
    exp_id=None,
    sequence=None,
    model_dir=None,
    save_freq=None,
    num_gpu=torch.cuda.device_count(),
    verify=False,
    dec_lr=None,
    half=False,
    resume=False,
    lambda_wm=0.01,
    k=100,
    randomize=False,
    watermark_key="secret_key",
    watermark_method='feature_based',
    num_parameters=1000,
    perturbation_strength=1e-5,
    watermark_size=128
):
    """
    Trains a model with or without watermarking, depending on the method selected.

    For parameter-perturbation watermarking, we store original parameter values
    so we can verify changes (relative check) at the end of training, rather than
    assuming the original parameters were zero.
    """

    k = int(k)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    # --- 1) LOAD DATASET ---
    trainset = utils.load_dataset(dataset, train=True, augment=False)
    logging.info(f"Dataset loaded with {len(trainset)} samples.")

    if batch_size <= 0:
        raise ValueError("Batch size must be a positive integer.")

    if num_gpu > torch.cuda.device_count():
        logging.warning(
            f"Requested {num_gpu} GPUs, but only {torch.cuda.device_count()} are available."
        )
        num_gpu = torch.cuda.device_count()

    if num_gpu > 1:
        batch_size *= num_gpu
        logging.info(f"Adjusted batch size for multiple GPUs: {batch_size}")

    # --- 2) BUILD MODEL ---
    net = architecture()
    net.apply(_weights_init)

    # Wrap the model if non-intrusive watermark is selected
    if watermark_method == 'non_intrusive':
        net = WatermarkModule(net, watermark_key, watermark_size=watermark_size)

    net.to(device)

    # --- 3) SETUP OPTIMIZER & SCHEDULER ---
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

    # If a pre-trained model directory is specified, load it
    if model_dir is not None:
        state = torch.load(model_dir, map_location=device)
        net.load_state_dict(state['net'])
        optimizer.load_state_dict(state['optimizer'])
        if scheduler is not None and 'scheduler' in state and state['scheduler'] is not None:
            scheduler.load_state_dict(state['scheduler'])
        logging.info(f"Loaded model from {model_dir}")

        if half:
            net.half().float()

    # If no sequence is given, create one
    if sequence is None:
        train_size = len(trainset)
        indices = np.arange(train_size)
        np.random.shuffle(indices)
        sequence = np.tile(indices, epochs)
        logging.info(f"Generated training sequence with length {len(sequence)}")

    # --- 4) SET REPRODUCIBLE SEEDS ---
    seed = 777
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # --- 5) PROOF-OF-LEARNING SAVE DIRECTORY ---
    if save_freq is not None and save_freq > 0:
        save_dir = os.path.join("proof", f"{dataset}_{exp_id}")
        os.makedirs(save_dir, exist_ok=True)

        m = hashlib.sha256()
        if hasattr(trainset, 'data'):
            data = trainset.data[sequence]
        elif hasattr(trainset, 'train_data'):
            data = trainset.train_data[sequence]
        else:
            raise AttributeError("Dataset has no 'data' or 'train_data' attribute.")

        logging.info(f"Data shape: {data.shape}, Data type: {data.dtype}")
        logging.info(
            f"First data sample hash: {hashlib.sha256(data[0].tobytes()).hexdigest()}"
        )

        if isinstance(data, np.ndarray):
            m.update(data.tobytes())
        elif isinstance(data, list):
            for d in data:
                m.update(np.array(d).tobytes())
        else:
            raise TypeError("Unsupported data type for hashing.")

        computed_hash = m.hexdigest()
        logging.info(f"Computed hash during training: {computed_hash}")

        with open(os.path.join(save_dir, "hash.txt"), "w") as f:
            f.write(computed_hash)
        logging.info("Saved dataset hash to hash.txt")

        np.save(os.path.join(save_dir, "indices.npy"), sequence)
        logging.info("Saved training sequence to indices.npy")

        # Save watermark meta-info
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
        logging.info("Saved watermarking information to watermark_info.json")
    else:
        save_dir = None

    # --- 6) CREATE DATALOADER ---
    sequence = np.reshape(sequence, -1)
    subset = torch.utils.data.Subset(trainset, sequence)
    trainloader = torch.utils.data.DataLoader(
        subset,
        batch_size=batch_size,
        num_workers=2,
        pin_memory=True
    )

    # --- 7) LOG MODEL & OPTIMIZER DETAILS ---
    logging.info(f"Model architecture: {architecture.__name__}")
    logging.info(f"Learning Rate: {lr}")
    logging.info(f"Batch Size: {batch_size}")
    logging.info(f"Epochs: {epochs}")
    logging.info(f"Optimizer: {optimizer.__class__.__name__}")
    if scheduler is not None:
        logging.info(
            f"Scheduler: {scheduler.__class__.__name__}, milestones={dec_lr}, gamma={scheduler.gamma}"
        )

    # Save the initial model
    if save_dir is not None:
        initial_state = {
            'net': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict() if scheduler else None,
            'step': 0,
        }
        torch.save(initial_state, os.path.join(save_dir, "model_step_0"))
        logging.info("Saved initial model checkpoint at step 0")

    # For parameter-perturbation watermarking, store original param values
    original_param_values = {}

    current_step = 0
    total_steps = len(trainloader) * epochs

    # --- 8) TRAINING LOOP ---
    for epoch in range(epochs):
        logging.info(f"Starting epoch {epoch + 1}/{epochs}")
        net.train()

        for batch_idx, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            # 8.1) FORWARD
            if watermark_method == 'none':
                # Baseline PoL with no WM
                outputs = (
                    net(inputs) if not isinstance(net, WatermarkModule) else net(inputs, trigger=False)
                )
                loss = criterion(outputs, labels)

            elif watermark_method == 'feature_based':
                # Hook for extracting layer features
                features_list = []

                def forward_hook(module, input_, output_):
                    features_list.append(output_)

                if isinstance(net, nn.DataParallel):
                    handle = net.module.layer1.register_forward_hook(forward_hook)
                else:
                    handle = net.layer1.register_forward_hook(forward_hook)

                outputs = net(inputs)
                handle.remove()

                loss = criterion(outputs, labels)

                # Possibly embed watermark in feature space
                if lambda_wm > 0 and should_embed_watermark(current_step, k, watermark_key, randomize):
                    feats = features_list[0]
                    desired_feats, mask = embed_feature_watermark(feats, watermark_key, current_step)
                    wm_loss = nn.MSELoss()(feats * mask, desired_feats * mask)
                    loss += lambda_wm * wm_loss
                    logging.info(f"Feature-based watermark loss computed at step {current_step}")

            elif watermark_method == 'parameter_perturbation':
                # Normal forward pass
                outputs = net(inputs)
                loss = criterion(outputs, labels)

            elif watermark_method == 'non_intrusive':
                # Pass normally w/trigger=False
                outputs = net(inputs, trigger=False)
                loss = criterion(outputs, labels)
                if lambda_wm > 0 and should_embed_watermark(current_step, k, watermark_key, randomize):
                    wm_target = generate_watermark_target(inputs, watermark_key, watermark_size)
                    wm_out = net(inputs, trigger=True)
                    wm_loss = nn.MSELoss()(wm_out, wm_target)
                    loss += lambda_wm * wm_loss
                    logging.info(f"Non-intrusive watermark loss computed at step {current_step}")

            else:
                raise ValueError(f"Unknown watermark method {watermark_method}")

            # 8.2) BACKWARD + STEP
            loss.backward()
            optimizer.step()

            # 8.3) If parameter-perturbation, do watermark after step
            if watermark_method == 'parameter_perturbation' and lambda_wm > 0:
                if should_embed_watermark(current_step, k, watermark_key, randomize):
                    # We store snapshot of current param values to compare later
                    selected_params = select_parameters_to_perturb(net, num_parameters, watermark_key)

                    for (pname, ptensor) in selected_params:
                        original_param_values[pname] = ptensor.detach().cpu().clone().numpy()

                    wpattern = generate_watermark_pattern(watermark_key, len(selected_params))
                    apply_parameter_perturbations(net, selected_params, wpattern, perturbation_strength)
                    logging.info(
                        f"Parameter perturbation watermark applied at step {current_step}, post-optimizer."
                    )

            # 8.4) Increment step & save checkpoint
            current_step += 1
            if save_dir is not None and current_step % save_freq == 0:
                checkpoint_state = {
                    'net': net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict() if scheduler else None,
                    'step': current_step,
                    'original_param_values': original_param_values
                }
                path_ckpt = os.path.join(save_dir, f"model_step_{current_step}")
                torch.save(checkpoint_state, path_ckpt)
                logging.info(f"Saved checkpoint at step {current_step}")

        # 8.5) Step scheduler each epoch
        if scheduler is not None:
            scheduler.step()
            logging.info(f"Scheduler stepped at epoch {epoch + 1}/{epochs}")

    # --- 9) Final checkpoint ---
    if save_dir is not None:
        final_state = {
            'net': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict() if scheduler else None,
            'step': current_step,
            'original_param_values': original_param_values
        }
        path_ckpt_final = os.path.join(save_dir, f"model_step_{current_step}")
        torch.save(final_state, path_ckpt_final)
        logging.info(f"Saved final model checkpoint at step {current_step}")

    return net, optimizer, criterion, original_param_values


def validate(dataset, model, batch_size=128):
    """
    Validate the final model accuracy on the test set.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    testset = utils.load_dataset(dataset, train=False, augment=False)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    model.to(device)
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Non-intrusive watermark modules always do normal inference w/trigger=False
            if hasattr(model, 'module') and isinstance(model.module, WatermarkModule):
                outputs = model(inputs, trigger=False)
            elif isinstance(model, WatermarkModule):
                outputs = model(inputs, trigger=False)
            else:
                outputs = model(inputs)

            _, predicted = torch.max(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100.0 * correct / total
    logging.info(f'Validation Accuracy: {accuracy:.2f}%')
    return correct / total


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training Script with or without Watermarking")

    parser.add_argument('--batch-size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--dataset', type=str, default="CIFAR10", help='Dataset to use')
    parser.add_argument('--model', type=str, default="resnet20", help="Model architecture (e.g. resnet20)")
    parser.add_argument('--id', type=str, default='Batch100', help='Experiment ID')
    parser.add_argument('--save-freq', type=int, default=1000, help='Frequency (steps) of saving checkpoints')
    parser.add_argument('--num-gpu', type=int, default=torch.cuda.device_count(), help='Number of GPUs to use')
    parser.add_argument('--milestone', nargs='+', type=int, default=None, help='LR scheduler milestones')
    parser.add_argument('--verify', type=int, default=0, help='Enable verification during training (1 or 0)')
    parser.add_argument('--lambda-wm', type=float, default=0.01, help='Watermark loss weight')
    parser.add_argument('--k', type=int, default=1000, help='Watermark embedding frequency (steps)')
    parser.add_argument('--randomize', action='store_true', help='Randomize watermark embedding intervals')
    parser.add_argument('--watermark-key', type=str, default='secret_key', help='Watermark key')
    parser.add_argument(
        '--watermark-method',
        type=str,
        default='feature_based',
        choices=['none', 'feature_based', 'parameter_perturbation', 'non_intrusive'],
        help='Watermark method: "none", "feature_based", "parameter_perturbation", "non_intrusive".'
    )
    parser.add_argument('--num-parameters', type=int, default=1000,
                        help='Num. of parameters to perturb (parameter_perturbation mode)')
    parser.add_argument('--perturbation-strength', type=float, default=1e-5,
                        help='Strength of parameter perturbations')
    parser.add_argument('--watermark-size', type=int, default=128,
                        help='Size of watermark for non_intrusive mode')

    args = parser.parse_args()

    # Make it reproducible
    seed = 777
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    t1 = time.time()
    logging.info(f"Trying to allocate {args.num_gpu} GPUs")

    # Load architecture
    try:
        import model  # local model.py
        architecture = getattr(model, args.model)
    except AttributeError:
        try:
            import torchvision.models as tv_models
            architecture = getattr(tv_models, args.model)
        except AttributeError as e:
            logging.error(f"Model {args.model} not found in model.py or torchvision.models.")
            raise e

    # --- TRAIN ---
    trained_model, optimizer, criterion, original_param_values = train(
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

    # If we want post-training verification logic, do it here:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.watermark_method == 'none':
        logging.info("Baseline PoL, no watermark verification needed.")
        torch.save(trained_model.state_dict(), 'model_baseline_no_watermark.pth')
        logging.info("Saved baseline model at model_baseline_no_watermark.pth")

    elif args.watermark_method == 'feature_based':
        run_feature_based_watermark_verification(
            model=trained_model,
            device=device,
            watermark_key=args.watermark_key
        )
        torch.save(trained_model.state_dict(), 'model_with_feature_based_watermark.pth')
        logging.info("Saved feature-based watermark model at model_with_feature_based_watermark.pth")

    elif args.watermark_method == 'parameter_perturbation':
        # Use the relative check method from watermark_utils
        from watermark_utils import verify_parameter_perturbation_watermark_relative
        detection_ok = verify_parameter_perturbation_watermark_relative(
            model=trained_model,
            original_params=original_param_values,  # compare final to original + shift
            watermark_key=args.watermark_key,
            perturbation_strength=args.perturbation_strength,
            tolerance=1e-6
        )
        if detection_ok:
            logging.info("Parameter-perturbation watermark verified at end of training.")
        else:
            logging.error("Parameter-perturbation watermark NOT verified at end of training.")

        torch.save(trained_model.state_dict(), 'model_with_parameter_perturbation_watermark.pth')
        logging.info("Saved param-perturbation watermark model at model_with_parameter_perturbation_watermark.pth")

    elif args.watermark_method == 'non_intrusive':
        detected = verify_non_intrusive_watermark(
            model=trained_model,
            device=device,
            watermark_key=args.watermark_key,
            watermark_size=args.watermark_size,
            tolerance=1e-5
        )
        if detected:
            logging.info("Non-intrusive watermark verified at end of training.")
        else:
            logging.error("Non-intrusive watermark not verified at end of training.")

        torch.save(trained_model.state_dict(), 'model_with_non_intrusive_watermark.pth')
        logging.info("Saved non-intrusive watermark model at model_with_non_intrusive_watermark.pth")

    # Finally, validate final model accuracy on test set
    from train_with_watermark import validate
    validate(args.dataset, trained_model)

    t2 = time.time()
    logging.info(f"Total training time: {t2 - t1:.2f} seconds")
