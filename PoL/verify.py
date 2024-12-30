#!/usr/bin/env python3
# verify.py
# Verification script that supports:
#   1) Baseline PoL (none)
#   2) Feature-based watermark
#   3) Parameter-perturbation watermark (relative check)
#   4) Non-intrusive watermark
#   + top-q or full verification of PoL checkpoints

import argparse
import os
import glob
import hashlib
import numpy as np
import torch
import logging
from functools import reduce
import random
import json

import utils
from train import train
import model as custom_model

# We import the 'relative' function for parameter-perturbation verification:
from watermark_utils import (
    verify_parameter_perturbation_watermark_relative,
    verify_non_intrusive_watermark
    # For feature-based, see embedded logic below
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')

# ------------------------------------------------------------------
# Feature-based watermark local helpers
# ------------------------------------------------------------------
def get_model(model_name):
    """If you have local constructors (resnet20, etc.) in model.py, use them here."""
    if hasattr(custom_model, model_name):
        return getattr(custom_model, model_name)()
    else:
        raise ValueError(f"No model found for name: {model_name}")

def prepare_watermark_data(device='cpu'):
    num_samples = 100
    input_size = (3, 32, 32)
    return torch.randn(num_samples, *input_size, device=device)

def extract_features_for_fb(model, inputs, layer_name='layer1'):
    features = []

    def hook(module, inp, out):
        features.append(out)

    handle = dict(model.named_modules())[layer_name].register_forward_hook(hook)
    with torch.no_grad():
        model(inputs)
    handle.remove()
    return features[0]

def check_feature_watermark(features, threshold=0.01):
    mean_val = features.mean().item()
    return mean_val > threshold

def validate_feature_watermark(model, device='cpu', layer_name='layer1'):
    logging.info("Starting feature-based watermark validation.")
    device = torch.device(device)
    model.to(device)
    model.eval()

    watermark_inputs = prepare_watermark_data(device=device)
    feats = extract_features_for_fb(model, watermark_inputs, layer_name)
    detected = check_feature_watermark(feats, threshold=0.01)

    if detected:
        logging.info("Feature-based watermark validation successful: Watermark detected.")
        return 1.0
    else:
        logging.error("Feature-based watermark validation failed: Watermark not detected.")
        return 0.0

def run_feature_based_watermark_verification(model=None, model_path=None,
                                             model_name=None, device='cpu'):
    logging.info("Running feature-based watermark verification...")
    device = torch.device(device)

    if model is None:
        if not model_path or not model_name:
            raise ValueError("Must provide a 'model' or both 'model_path' & 'model_name'.")
        net = get_model(model_name)
        state = torch.load(model_path, map_location=device)
        if 'net' in state:
            net.load_state_dict(state['net'])
        else:
            net.load_state_dict(state)
    else:
        net = model

    detection = validate_feature_watermark(net, device=device)
    if detection == 1.0:
        logging.info("Feature-based watermark verification successful: Watermark present.")
    else:
        logging.error("Feature-based watermark verification failed: Watermark not detected.")


# ------------------------------------------------------------------
# Main PoL verification (full or top-q) + Watermark checks
# ------------------------------------------------------------------

def verify_all(model_directory, lr, batch_size, dataset, model_arch, save_freq,
               order, threshold, half=0, lambda_wm=0, k=1000, randomize=False,
               watermark_key='secret_key', watermark_method='none',
               num_parameters=1000, perturbation_strength=1e-5, watermark_size=128):
    if not os.path.exists(model_directory):
        raise FileNotFoundError("Model directory not found.")
    sequence = np.load(os.path.join(model_directory, "indices.npy"))

    if not isinstance(order, list):
        order = [order]
        threshold = [threshold]
    else:
        assert len(order) == len(threshold), "Mismatch in length of order vs threshold."

    dist_list = [[] for _ in range(len(order))]

    checkpoint_files = glob.glob(os.path.join(model_directory, 'model_step_*'))
    checkpoint_steps = sorted(int(os.path.basename(f).split('_')[-1]) for f in checkpoint_files)

    logging.info("Performing full verification on all intervals...")
    for idx in range(len(checkpoint_steps) - 1):
        current_step = checkpoint_steps[idx]
        next_step = checkpoint_steps[idx + 1]
        current_model = os.path.join(model_directory, f"model_step_{current_step}")
        target_model = os.path.join(model_directory, f"model_step_{next_step}")

        start_sequence_idx = current_step * batch_size
        end_sequence_idx = min(next_step * batch_size, len(sequence))

        logging.debug(f"Reproducing training from step {current_step} to {next_step} for verification.")
        # train(...) now returns 4 items, but we only need the first 3 here
        # if you want only net, do: net, *_ = train(...)
        net_reproduced, opt, crit, orig_params = train(
            lr=lr,
            batch_size=batch_size,
            epochs=1,
            dataset=dataset,
            architecture=model_arch,
            model_dir=current_model,
            sequence=sequence[start_sequence_idx:end_sequence_idx],
            half=half,
            lambda_wm=lambda_wm,
            k=k,
            randomize=randomize,
            watermark_key=watermark_key,
            watermark_method=watermark_method,
            num_parameters=num_parameters,
            perturbation_strength=perturbation_strength,
            watermark_size=watermark_size
        )

        res = utils.parameter_distance(target_model, net_reproduced, order=order,
                                       architecture=model_arch, half=half)
        for j in range(len(order)):
            dist_list[j].append(res[j])

    dist_list = np.array(dist_list)
    logging.info("Full verification results:")
    for i, metric in enumerate(order):
        avg_dist = np.mean(dist_list[i])
        max_dist = np.max(dist_list[i])
        min_dist = np.min(dist_list[i])
        logging.info(f"Metric: {metric}, Threshold: {threshold[i]}, "
                     f"Avg: {avg_dist:.4f}, Max: {max_dist:.4f}, Min: {min_dist:.4f}")
        above_threshold = np.sum(dist_list[i] > threshold[i])
        if above_threshold == 0:
            logging.info(f"No steps exceed threshold for metric {metric}. PoL appears valid.")
        else:
            percentage = 100.0 * above_threshold / dist_list[i].shape[0]
            logging.info(f"{above_threshold}/{dist_list[i].shape[0]} "
                         f"({percentage:.2f}%) steps exceed threshold for {metric}. PoL may be invalid.")
    return dist_list

def verify_topq(model_directory, lr, batch_size, dataset, model_arch, save_freq, order, threshold,
                epochs=1, q=10, half=0, lambda_wm=0, k=1000, randomize=False,
                watermark_key='secret_key', watermark_method='none',
                num_parameters=1000, perturbation_strength=1e-5, watermark_size=128):
    if not os.path.exists(model_directory):
        raise FileNotFoundError("Model directory not found")
    sequence = np.load(os.path.join(model_directory, "indices.npy"))

    if not isinstance(order, list):
        order = [order]
        threshold = [threshold]
    else:
        assert len(order) == len(threshold), "Mismatch in order vs threshold length."

    checkpoint_files = glob.glob(os.path.join(model_directory, 'model_step_*'))
    checkpoint_steps = sorted(int(os.path.basename(f).split('_')[-1]) for f in checkpoint_files)
    total_checkpoints = len(checkpoint_steps)
    checkpoints_per_epoch = total_checkpoints // epochs

    logging.info(f"Starting top-q verification with q={q} for {epochs} epochs...")
    res_list = []

    for epoch in range(epochs):
        logging.info(f"Verifying epoch {epoch + 1}/{epochs}")
        start_idx = epoch * checkpoints_per_epoch
        end_idx = (epoch + 1) * checkpoints_per_epoch if epoch < (epochs - 1) else total_checkpoints

        dist_list = [[] for _ in range(len(order))]
        # measure distances among consecutive checkpoints
        for idx in range(start_idx, end_idx - 1):
            current_step = checkpoint_steps[idx]
            next_step = checkpoint_steps[idx + 1]
            current_model = os.path.join(model_directory, f"model_step_{current_step}")
            next_model = os.path.join(model_directory, f"model_step_{next_step}")

            res = utils.parameter_distance(current_model, next_model, order=order,
                                           architecture=model_arch, half=half)
            for j in range(len(order)):
                dist_list[j].append(res[j])

        dist_arr = np.array(dist_list)
        topq_indices = np.argpartition(dist_arr, -q, axis=1)[:, -q:]
        if len(order) > 1:
            topq_steps = reduce(np.union1d, [indices for indices in topq_indices])
        else:
            topq_steps = topq_indices[0]

        dist_list = [[] for _ in range(len(order))]
        for ind in topq_steps:
            step_idx = start_idx + ind
            if step_idx >= len(checkpoint_steps) - 1:
                continue
            current_step = checkpoint_steps[step_idx]
            next_step = checkpoint_steps[step_idx + 1]
            current_model = os.path.join(model_directory, f"model_step_{current_step}")
            target_model = os.path.join(model_directory, f"model_step_{next_step}")

            start_sequence_idx = current_step * batch_size
            end_sequence_idx = min(next_step * batch_size, len(sequence))

            logging.debug(f"Reproducing training for top-q step from {current_step} to {next_step}...")
            net_reproduced, opt, crit, orig_params = train(
                lr=lr,
                batch_size=batch_size,
                epochs=1,
                dataset=dataset,
                architecture=model_arch,
                model_dir=current_model,
                sequence=sequence[start_sequence_idx:end_sequence_idx],
                half=half,
                lambda_wm=lambda_wm,
                k=k,
                randomize=randomize,
                watermark_key=watermark_key,
                watermark_method=watermark_method,
                num_parameters=num_parameters,
                perturbation_strength=perturbation_strength,
                watermark_size=watermark_size
            )
            res = utils.parameter_distance(target_model, net_reproduced, order=order,
                                           architecture=model_arch, half=half)
            for j in range(len(order)):
                dist_list[j].append(res[j])

        dist_list = np.array(dist_list)
        logging.info(f"Top-q verification results for epoch {epoch+1}:")
        for idx_metric, metric in enumerate(order):
            if dist_list[idx_metric].size > 0:
                avg_val = np.mean(dist_list[idx_metric])
            else:
                avg_val = 0.0
            logging.info(f"Metric: {metric}, Threshold: {threshold[idx_metric]}, Q={q}, Avg top-q: {avg_val:.4f}")

            if dist_list[idx_metric].size > 0:
                above_threshold = np.sum(dist_list[idx_metric] > threshold[idx_metric])
            else:
                above_threshold = 0

            if above_threshold == 0:
                logging.info("None of the top-q steps exceed the threshold. PoL appears valid.")
            else:
                pct = 100.0 * above_threshold / dist_list[idx_metric].shape[0]
                logging.info(f"{above_threshold}/{dist_list[idx_metric].shape[0]} "
                             f"({pct:.2f}%) top-q steps exceed threshold. PoL may be invalid.")

        res_list.append(dist_list)
    return res_list


def verify_initialization(model_directory, model_arch, threshold=0.01, net=None, verbose=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if net is None:
        net = model_arch()
        state = torch.load(os.path.join(model_directory, "model_step_0"), map_location=device)
        net.load_state_dict(state['net'])
    net.to(device)
    model_name = model_arch.__name__

    import utils

    # Distinguish among resnet, etc.
    if model_name in ['resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']:
        model_type = 'resnet_cifar'
    elif model_name == 'resnet50':
        model_type = 'resnet_cifar100'
    elif 'resnet' in model_name:
        model_type = 'resnet'
    else:
        model_type = 'default'

    p_list = []
    if model_type == 'resnet':
        for name, param in net.named_parameters():
            if 'weight' in name and 'conv' in name:
                p_list.append(utils.check_weights_initialization(param, 'resnet'))
            elif 'weight' in name and 'fc' in name:
                p_list.append(utils.check_weights_initialization(param, 'default'))
            elif 'bias' in name and ('fc' in name or 'linear' in name):
                w = net.state_dict()[name.replace('bias', 'weight')]
                p_list.append(utils.check_weights_initialization([w, param], 'default_bias'))
    elif model_type == 'resnet_cifar100':
        for name, param in net.named_parameters():
            if len(param.shape) == 4:
                p_list.append(utils.check_weights_initialization(param, 'default'))
            elif 'weight' in name and 'fc' in name:
                p_list.append(utils.check_weights_initialization(param, 'default'))
            elif 'bias' in name and ('fc' in name or 'linear' in name):
                w = net.state_dict()[name.replace('bias', 'weight')]
                p_list.append(utils.check_weights_initialization([w, param], 'default_bias'))
    elif model_type == 'resnet_cifar':
        for name, param in net.named_parameters():
            if 'fc' in name or 'conv' in name or 'linear' in name:
                if 'weight' in name:
                    p_list.append(utils.check_weights_initialization(param, 'resnet_cifar'))
                elif 'bias' in name:
                    w = net.state_dict()[name.replace('bias', 'weight')]
                    p_list.append(utils.check_weights_initialization([w, param], 'default_bias'))
    else:
        for name, param in net.named_parameters():
            if 'fc' in name or 'conv' in name or 'linear' in name:
                if 'weight' in name:
                    p_list.append(utils.check_weights_initialization(param, 'default'))
                elif 'bias' in name:
                    w = net.state_dict()[name.replace('bias', 'weight')]
                    p_list.append(utils.check_weights_initialization([w, param], 'default_bias'))

    if verbose and len(p_list) > 0:
        min_p = np.min(p_list)
        if min_p < threshold:
            logging.info(f"Initialization check: min p-value {min_p:.4f} < {threshold}, PoL may not be valid.")
        else:
            logging.info("Initialization verification passed.")
    return p_list

def verify_hash(model_directory, dataset):
    if not os.path.exists(model_directory):
        raise FileNotFoundError("Model directory not found")

    sequence = np.load(os.path.join(model_directory, "indices.npy"))
    with open(os.path.join(model_directory, "hash.txt"), "r") as f:
        saved_hash = f.read().strip()

    import utils
    trainset = utils.load_dataset(dataset, train=True, augment=False)
    m = hashlib.sha256()

    if hasattr(trainset, 'data'):
        data = trainset.data[sequence]
    elif hasattr(trainset, 'train_data'):
        data = trainset.train_data[sequence]
    else:
        raise AttributeError("Dataset object has no attribute 'data' or 'train_data'.")

    if isinstance(data, np.ndarray):
        m.update(data.tobytes())
    elif isinstance(data, list):
        for d in data:
            m.update(np.array(d).tobytes())
    else:
        raise TypeError("Unsupported data type for hashing.")

    computed = m.hexdigest()
    logging.info(f"Saved hash from training: {saved_hash}")
    logging.info(f"Computed hash during verification: {computed}")

    if saved_hash != computed:
        logging.info("Hash mismatch. Proof-of-learning invalid.")
    else:
        logging.info("Hash matches, proof-of-learning is valid.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Model and Watermark Verification Script")
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--dataset', type=str, default="CIFAR10")
    parser.add_argument('--model', type=str, default="resnet20")
    parser.add_argument('--model-dir', type=str, default='proof/CIFAR10_Batch100', help='Model directory')
    parser.add_argument('--save-freq', type=int, default=100)
    parser.add_argument('--dist', type=str, nargs='+', default=['1', '2', 'inf', 'cos'],
                        help='Distance metrics for parameter_distance.')
    parser.add_argument('--q', type=int, default=0, help='>0 for top-q verification')
    parser.add_argument('--delta', type=float, nargs='+', default=[10000, 100, 1, 0.1],
                        help='Thresholds per metric in --dist')
    parser.add_argument('--lambda-wm', type=float, default=0.01)
    parser.add_argument('--k', type=int, default=1000)
    parser.add_argument('--randomize', action='store_true')
    parser.add_argument('--watermark-key', type=str, default='secret_key')
    parser.add_argument('--watermark-method', type=str, default='none',
                        choices=['none', 'feature_based', 'parameter_perturbation', 'non_intrusive'])
    parser.add_argument('--num-parameters', type=int, default=1000)
    parser.add_argument('--perturbation-strength', type=float, default=1e-5)
    parser.add_argument('--watermark-size', type=int, default=128)
    parser.add_argument('--watermark-path', type=str, default='model_with_watermark.pth',
                        help='Path to the final watermarked model checkpoint.')

    args = parser.parse_args()

    model_arch = getattr(custom_model, args.model)

    with open(os.path.join(args.model_dir, "watermark_info.json"), "r") as f:
        wm_info = json.load(f)

    seed = wm_info.get('seed', 777)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    watermark_key = wm_info.get('watermark_key', args.watermark_key)
    lambda_wm = wm_info.get('lambda_wm', args.lambda_wm)
    k = wm_info.get('k', args.k)
    randomize = wm_info.get('randomize', args.randomize)
    watermark_method = wm_info.get('watermark_method', args.watermark_method)
    num_parameters = wm_info.get('num_parameters', args.num_parameters)
    perturbation_strength = wm_info.get('perturbation_strength', args.perturbation_strength)
    watermark_size = wm_info.get('watermark_size', args.watermark_size)

    logging.info("Starting verification process...")
    logging.info("Verifying model initialization (Kaiming, etc.)...")
    verify_initialization(args.model_dir, model_arch)
    verify_hash(args.model_dir, args.dataset)

    if args.q > 0:
        logging.info(f"Performing top-q verification with q={args.q} ...")
        verify_topq(
            model_directory=args.model_dir,
            lr=args.lr,
            batch_size=args.batch_size,
            dataset=args.dataset,
            model_arch=model_arch,
            save_freq=args.save_freq,
            order=args.dist,
            threshold=args.delta,
            epochs=args.epochs,
            q=args.q,
            half=0,
            lambda_wm=lambda_wm,
            k=k,
            randomize=randomize,
            watermark_key=watermark_key,
            watermark_method=watermark_method,
            num_parameters=num_parameters,
            perturbation_strength=perturbation_strength,
            watermark_size=watermark_size
        )
    else:
        logging.info("Performing full verification without top-q...")
        verify_all(
            model_directory=args.model_dir,
            lr=args.lr,
            batch_size=args.batch_size,
            dataset=args.dataset,
            model_arch=model_arch,
            save_freq=args.save_freq,
            order=args.dist,
            threshold=args.delta,
            half=0,
            lambda_wm=lambda_wm,
            k=k,
            randomize=randomize,
            watermark_key=watermark_key,
            watermark_method=watermark_method,
            num_parameters=num_parameters,
            perturbation_strength=perturbation_strength,
            watermark_size=watermark_size
        )

    logging.info("Verifying watermark presence in the final model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    final_model_path = args.watermark_path
    logging.info(f"Attempting to load final model from: {final_model_path}")

    loaded_model = model_arch()
    state = torch.load(final_model_path, map_location=device)
    loaded_model.load_state_dict(state)
    loaded_model.to(device)

    if watermark_method == 'none':
        logging.info("No watermark to verify, skipping.")
    elif watermark_method == 'feature_based':
        # local approach
        detection_val = validate_feature_watermark(loaded_model, device=device)
        if detection_val == 1.0:
            logging.info("Feature-based watermark is present in the final model.")
        else:
            logging.error("Feature-based watermark NOT detected in the final model.")

    elif watermark_method == 'parameter_perturbation':
        wd = verify_parameter_perturbation_watermark_relative(
            model=loaded_model,
            original_params=None,  # If you had the dict, pass it; else fallback to zero-based
            watermark_key=watermark_key,
            perturbation_strength=perturbation_strength,
            tolerance=1e-6
        )
        if wd:
            logging.info("Parameter-Perturbation Watermark is present.")
        else:
            logging.error("Parameter-Perturbation Watermark not detected.")

    elif watermark_method == 'non_intrusive':
        wd = verify_non_intrusive_watermark(
            model=loaded_model,
            device=device,
            watermark_key=watermark_key,
            watermark_size=watermark_size,
            tolerance=1e-5
        )
        if wd:
            logging.info("Non-Intrusive Watermark is present.")
        else:
            logging.error("Non-Intrusive Watermark not detected.")
    else:
        logging.error(f"Unknown watermarking method: {watermark_method}")

    logging.info("Verification process completed successfully.")
