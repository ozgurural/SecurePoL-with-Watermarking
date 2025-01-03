#!/usr/bin/env python3
# train.py
# Fully updated integrated training script for:
#   1) Baseline PoL (no watermark),
#   2) Feature-Based Watermark,
#   3) Parameter-Perturbation Watermark,
#   4) Non-Intrusive Watermark (Option B),
# with suggested improvements.

import argparse
import json
import logging
import os
import random
import time
import hashlib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import utils  # your local utils.py => must have `load_dataset()`

from watermark_utils import (
    generate_watermark_pattern,
    select_parameters_to_perturb,
    apply_parameter_perturbations,
    should_embed_watermark,
    WatermarkModule,        # Option B approach for 'non_intrusive'
    embed_feature_watermark,
    generate_watermark_target,
    verify_non_intrusive_watermark,
    run_feature_based_watermark_verification
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def _weights_init(m):
    """Applies Kaiming initialization to Linear or Conv2d layers."""
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)

def train(
    lr,
    batch_size,
    epochs,
    dataset,
    architecture,
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
    Trains a model with or without watermarking.
    For 'non_intrusive' (Option B), we assume a CIFAR-10 model => [batch_size, 10],
    then wrap with WatermarkModule => triggers produce [batch_size, watermark_size].
    """

    k = int(k)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    # 1) Load dataset
    trainset = utils.load_dataset(dataset, train=True, augment=False)
    logging.info(f"Dataset loaded with {len(trainset)} samples.")

    if batch_size <= 0:
        raise ValueError("Batch size must be positive.")

    # Possibly adjust num_gpu
    if num_gpu > torch.cuda.device_count():
        logging.warning(
            f"Requested {num_gpu} GPUs but only {torch.cuda.device_count()} available."
        )
        num_gpu = torch.cuda.device_count()

    if num_gpu > 1:
        batch_size *= num_gpu
        logging.info(f"Adjusted batch size for multiple GPUs: {batch_size}")

    # 2) Build model
    net = architecture()
    net.apply(_weights_init)

    # If 'non_intrusive' => wrap final outputs
    if watermark_method == 'non_intrusive':
        net = WatermarkModule(net, watermark_key, watermark_size=watermark_size)

    net.to(device)

    # 3) Setup optimizer & scheduler
    if dataset == 'MNIST':
        optimizer = optim.SGD(net.parameters(), lr=lr)
        scheduler = None
    elif dataset == 'CIFAR10':
        if dec_lr is None:
            # Usually reduce LR halfway, then 3/4th
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

    # Optionally load from checkpoint
    if model_dir is not None:
        state = torch.load(model_dir, map_location=device)
        net.load_state_dict(state['net'])
        optimizer.load_state_dict(state['optimizer'])
        if scheduler and 'scheduler' in state and state['scheduler'] is not None:
            scheduler.load_state_dict(state['scheduler'])
        logging.info(f"Loaded model from {model_dir}")
        if half:
            net.half().float()

    # Prepare sequence if not provided
    if sequence is None:
        train_size = len(trainset)
        idxs = np.arange(train_size)
        np.random.shuffle(idxs)
        sequence = np.tile(idxs, epochs)
        logging.info(f"Generated training sequence with length {len(sequence)}")

    # 4) Reproducibility
    seed = 777
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 5) Setup PoL artifacts
    if save_freq and save_freq > 0:
        save_dir = os.path.join("proof", f"{dataset}_{exp_id}")
        os.makedirs(save_dir, exist_ok=True)

        # Hash the training data subset
        m = hashlib.sha256()
        if hasattr(trainset, 'data'):
            data = trainset.data[sequence]
        elif hasattr(trainset, 'train_data'):
            data = trainset.train_data[sequence]
        else:
            raise AttributeError("No 'data' or 'train_data' in dataset.")

        from hashlib import sha256
        logging.info(f"Data shape: {data.shape}, Data type: {data.dtype}")
        logging.info(f"First data sample hash: {sha256(data[0].tobytes()).hexdigest()}")

        if isinstance(data, np.ndarray):
            m.update(data.tobytes())
        elif isinstance(data, list):
            for d in data:
                m.update(np.array(d).tobytes())
        else:
            raise TypeError("Unsupported data type for hashing.")

        computed_hash = m.hexdigest()
        logging.info(f"Computed hash during training: {computed_hash}")

        # Save hash, sequence, watermark info
        with open(os.path.join(save_dir, "hash.txt"), "w") as f:
            f.write(computed_hash)
        logging.info("Saved dataset hash to hash.txt")

        np.save(os.path.join(save_dir, "indices.npy"), sequence)
        logging.info("Saved training sequence to indices.npy")

        wm_info = {
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
            json.dump(wm_info, f)
        logging.info("Saved watermark info to watermark_info.json")
    else:
        save_dir = None

    # 6) DataLoader
    sequence = np.reshape(sequence, -1)
    subset = torch.utils.data.Subset(trainset, sequence)
    trainloader = torch.utils.data.DataLoader(
        subset,
        batch_size=batch_size,
        num_workers=2,
        pin_memory=True
    )

    # 7) Logging
    logging.info(f"Model architecture: {architecture.__name__}")
    logging.info(f"Learning Rate: {lr}")
    logging.info(f"Batch Size: {batch_size}")
    logging.info(f"Epochs: {epochs}")
    logging.info(f"Optimizer: {optimizer.__class__.__name__}")
    if scheduler:
        logging.info(
            f"Scheduler: {scheduler.__class__.__name__}, milestones={dec_lr}, gamma={scheduler.gamma}"
        )

    if save_dir:
        init_state = {
            'net': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict() if scheduler else None,
            'step': 0
        }
        torch.save(init_state, os.path.join(save_dir, "model_step_0"))
        logging.info("Saved initial model checkpoint at step 0")

    original_param_values = {}
    current_step = 0
    total_steps = len(trainloader) * epochs

    # 8) Main training loop
    for epoch in range(epochs):
        logging.info(f"Starting epoch {epoch+1}/{epochs}")
        net.train()

        for batch_idx, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            if watermark_method == 'none':
                # baseline PoL
                if isinstance(net, WatermarkModule):
                    outputs = net(inputs, trigger=False)
                else:
                    outputs = net(inputs)
                loss = criterion(outputs, labels)

            elif watermark_method == 'feature_based':
                feats_list = []
                def forward_hook(module, m_in, m_out):
                    feats_list.append(m_out)

                if isinstance(net, nn.DataParallel):
                    hook_handle = net.module.layer1.register_forward_hook(forward_hook)
                else:
                    hook_handle = net.layer1.register_forward_hook(forward_hook)

                outputs = net(inputs)
                hook_handle.remove()

                loss = criterion(outputs, labels)
                if lambda_wm > 0 and should_embed_watermark(current_step, k, watermark_key, randomize):
                    feats = feats_list[0]
                    desired_feats, mask = embed_feature_watermark(feats, watermark_key, current_step)
                    wm_loss = nn.MSELoss()(feats * mask, desired_feats * mask)
                    loss += lambda_wm * wm_loss
                    logging.info(f"Feature-based WM loss computed at step={current_step}")

            elif watermark_method == 'parameter_perturbation':
                outputs = net(inputs)
                loss = criterion(outputs, labels)

            elif watermark_method == 'non_intrusive':
                # normal classification => trigger=False
                outputs = net(inputs, trigger=False)
                loss = criterion(outputs, labels)

                if lambda_wm > 0 and should_embed_watermark(current_step, k, watermark_key, randomize):
                    # produce random target => MSE
                    wm_target = generate_watermark_target(inputs, watermark_key, watermark_size).to(device)
                    wm_out = net(inputs, trigger=True)
                    wm_loss = nn.MSELoss()(wm_out, wm_target)
                    loss += lambda_wm * wm_loss
                    logging.info(f"Non-intrusive WM loss computed at step={current_step}")

            else:
                raise ValueError(f"Unknown watermark method {watermark_method}")

            loss.backward()
            optimizer.step()

            # param-perturbation => after step
            if watermark_method == 'parameter_perturbation' and lambda_wm > 0:
                if should_embed_watermark(current_step, k, watermark_key, randomize):
                    from watermark_utils import generate_watermark_pattern
                    sel_params = select_parameters_to_perturb(net, num_parameters, watermark_key)
                    for (pname, ptensor) in sel_params:
                        original_param_values[pname] = ptensor.detach().cpu().clone().numpy()
                    pat = generate_watermark_pattern(watermark_key, len(sel_params))
                    apply_parameter_perturbations(sel_params, pat, perturbation_strength)
                    logging.info(f"Parameter-perturbation WM applied at step={current_step}")

            current_step += 1

            if save_dir and (current_step % save_freq == 0):
                ckpt_dict = {
                    'net': net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict() if scheduler else None,
                    'step': current_step,
                    'original_param_values': original_param_values
                }
                ckpt_path = os.path.join(save_dir, f"model_step_{current_step}")
                torch.save(ckpt_dict, ckpt_path)
                logging.info(f"Saved checkpoint at step={current_step}")

        if scheduler:
            scheduler.step()
            logging.info(f"Scheduler stepped at epoch={epoch+1}/{epochs}")

    # 9) Final checkpoint
    if save_dir:
        final_state = {
            'net': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict() if scheduler else None,
            'step': current_step,
            'original_param_values': original_param_values
        }
        final_ckpt_path = os.path.join(save_dir, f"model_step_{current_step}")
        torch.save(final_state, final_ckpt_path)
        logging.info(f"Saved final model checkpoint at step={current_step}")

    return net, optimizer, criterion, original_param_values

def validate(dataset, model, batch_size=128):
    """
    Validate final model accuracy on the test set.
    """
    from watermark_utils import WatermarkModule
    import utils

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    testset = utils.load_dataset(dataset, train=False, augment=False)
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    model.to(device)
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            # If non-intrusive => normal classification w/ trigger=False
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
    parser = argparse.ArgumentParser("Training with Non-Intrusive Watermark (Option B)")

    # Basic training arguments
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=2, help="2 is very short for watermark to converge")
    parser.add_argument('--dataset', type=str, default="CIFAR10")
    parser.add_argument('--model', type=str, default="resnet20")
    parser.add_argument('--id', type=str, default='Batch100')
    parser.add_argument('--save-freq', type=int, default=100)
    parser.add_argument('--num-gpu', type=int, default=torch.cuda.device_count())
    parser.add_argument('--milestone', nargs='+', type=int, default=None)
    parser.add_argument('--verify', type=int, default=0)

    # Watermark hyperparams
    parser.add_argument('--lambda-wm', type=float, default=0.01,
                        help="Try 0.1 or 1.0 for a stronger watermark push if MSE remains high.")
    parser.add_argument('--k', type=int, default=1000,
                        help="Steps between watermark embeddings. For quick testing, set smaller like 100.")
    parser.add_argument('--randomize', action='store_true',
                        help="If set, embed watermark at random times with prob=1/k.")
    parser.add_argument('--watermark-key', type=str, default='secret_key')
    parser.add_argument('--watermark-method', type=str, default='non_intrusive',
                        choices=['none', 'feature_based', 'parameter_perturbation', 'non_intrusive'])
    parser.add_argument('--num-parameters', type=int, default=1000)
    parser.add_argument('--perturbation-strength', type=float, default=1e-5)
    parser.add_argument('--watermark-size', type=int, default=128,
                        help="Dimension for the non-intrusive watermark output layer")

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

    # Dynamically load resnet20 from model.py or fallback in torchvision
    try:
        import model
        architecture = getattr(model, args.model)
    except AttributeError:
        try:
            import torchvision.models as tv_models
            architecture = getattr(tv_models, args.model)
        except AttributeError as e:
            logging.error(f"Model {args.model} not found in model.py or torchvision.")
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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Post-training: run final watermark checks if needed
    if args.watermark_method == 'none':
        logging.info("No watermark used; baseline PoL.")
        torch.save(trained_model.state_dict(), 'model_baseline_no_watermark.pth')

    elif args.watermark_method == 'feature_based':
        run_feature_based_watermark_verification(
            model=trained_model,
            device=device,
            watermark_key=args.watermark_key
        )
        torch.save(trained_model.state_dict(), 'model_with_feature_based_watermark.pth')
        logging.info("Saved model_with_feature_based_watermark.pth")

    elif args.watermark_method == 'parameter_perturbation':
        from watermark_utils import verify_parameter_perturbation_watermark_relative

        detection_ok = verify_parameter_perturbation_watermark_relative(
            model=trained_model,
            original_params=original_param_values,
            watermark_key=args.watermark_key,
            perturbation_strength=args.perturbation_strength,
            tolerance=1e-1
        )
        if detection_ok:
            logging.info("Parameter-perturbation WM verified at end of training.")
        else:
            logging.error("Parameter-perturbation WM NOT verified.")
        final_ckpt = {
            'net': trained_model.state_dict(),
            'original_param_values': original_param_values
        }
        torch.save(final_ckpt, 'model_with_parameter_perturbation_watermark.pth')
        logging.info("Saved model_with_parameter_perturbation_watermark.pth")

    elif args.watermark_method == 'non_intrusive':
        # A slightly higher tolerance if MSE remains large after only 2 epochs
        tolerance_nm = 1e-3
        success = verify_non_intrusive_watermark(
            model=trained_model,
            device=device,
            watermark_key=args.watermark_key,
            watermark_size=args.watermark_size,
            tolerance=tolerance_nm
        )
        if success:
            logging.info("Non-intrusive watermark verified at end of training.")
        else:
            logging.error(
                f"Non-intrusive watermark not verified; MSE exceeded {tolerance_nm} after {args.epochs} short epochs."
            )
        torch.save(trained_model.state_dict(), 'model_with_non_intrusive_watermark.pth')
        logging.info("Saved model_with_non_intrusive_watermark.pth")

    # Finally, validate accuracy
    def final_validate(dataset, model, batch_size=128):
        from watermark_utils import WatermarkModule
        import utils

        dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        testset_ = utils.load_dataset(dataset, train=False, augment=False)
        testloader_ = torch.utils.data.DataLoader(
            testset_, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
        )
        model.to(dev)
        model.eval()

        corr = 0
        ttl = 0
        with torch.no_grad():
            for inp, lbl in testloader_:
                inp, lbl = inp.to(dev), lbl.to(dev)
                if hasattr(model, 'module') and isinstance(model.module, WatermarkModule):
                    outs = model(inp, trigger=False)
                elif isinstance(model, WatermarkModule):
                    outs = model(inp, trigger=False)
                else:
                    outs = model(inp)

                _, pred = torch.max(outs, dim=1)
                ttl += lbl.size(0)
                corr += (pred == lbl).sum().item()

        acc = 100.0 * corr / ttl
        logging.info(f'Validation Accuracy: {acc:.2f}%')
        return corr / ttl

    final_validate(args.dataset, trained_model)

    t2 = time.time()
    logging.info(f"Total training time: {t2 - t1:.2f} seconds")
