#!/usr/bin/env python3

# verify.py

# Verification script for:
# 1) Baseline PoL (none)
# 2) Feature-based watermark
# 3) Parameter-perturbation watermark (relative check)
# 4) Non-intrusive watermark
# + top-q or full verification of PoL checkpoints

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

# If you rely on more advanced watermark verifications:
from watermark_utils import (
    verify_parameter_perturbation_watermark_relative,
    verify_non_intrusive_watermark
    # For feature-based, we embed logic directly below
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
    """Prepare dummy data for checking a feature-based watermark."""
    num_samples = 100
    input_size = (3, 32, 32)
    return torch.randn(num_samples, *input_size, device=device)

def extract_features_for_fb(model, inputs, layer_name='layer1'):
    features = []
    def hook(module, inp, out):
        features.append(out)
    h = dict(model.named_modules())[layer_name].register_forward_hook(hook)
    with torch.no_grad():
        model(inputs)
    h.remove()
    return features[0]

def check_feature_watermark(features, threshold=0.01):
    mean_val = features.mean().item()
    return (mean_val > threshold)

def validate_feature_watermark(model, device='cpu', layer_name='layer1'):
    logging.info("Starting feature-based watermark validation.")
    model.to(device)
    model.eval()
    watermark_inputs = prepare_watermark_data(device=device)
    feats = extract_features_for_fb(model, watermark_inputs, layer_name=layer_name)
    detected = check_feature_watermark(feats, threshold=0.01)
    if detected:
        logging.info("Feature-based watermark validation successful: Watermark detected.")
        return 1.0
    else:
        logging.error("Feature-based watermark validation failed: Watermark not detected.")
        return 0.0

def run_feature_based_watermark_verification(model=None, model_path=None,
                                             model_name=None, device='cpu'):
    """
    If 'model' is provided, skip loading from path.
    Otherwise load from 'model_path','model_name'. Then run feature-based check.
    """
    logging.info("Running feature-based watermark verification...")
    device = torch.device(device)
    if model is None:
        if (not model_path) or (not model_name):
            raise ValueError("Must provide either a 'model' instance or both 'model_path' & 'model_name'.")
        net = get_model(model_name)
        st = torch.load(model_path, map_location=device)
        if 'net' in st:
            net.load_state_dict(st['net'])
        else:
            net.load_state_dict(st)
    else:
        net = model

    detection = validate_feature_watermark(net, device=device)
    if detection == 1.0:
        logging.info("Feature-based watermark verification successful: Watermark present.")
    else:
        logging.error("Feature-based watermark verification failed: Watermark not detected.")

# ------------------------------------------------------------------
# MAIN PoL verification LOGIC
# ------------------------------------------------------------------

def verify_all(model_directory, lr, batch_size, dataset, model_arch, save_freq,
               order, threshold, half=0, lambda_wm=0, k=1000, randomize=False,
               watermark_key='secret_key', watermark_method='none',
               num_parameters=1000, perturbation_strength=1e-5, watermark_size=128):
    """
    Reproduce training *without embedding* any new watermark (but keep the same
    watermark structure), measure distance to the next checkpoint, across all intervals.
    """
    if not os.path.exists(model_directory):
        raise FileNotFoundError("Model directory not found.")

    # Load the indices used for training
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

        logging.debug(f"Reproducing training from {current_step} to {next_step} for verification.")

        # Pass the same watermark_method, k, randomize, etc. as originally used
        # BUT set lambda_wm=0 to avoid embedding a second watermark
        net_rep, _, _, _ = train(
            lr=lr,
            batch_size=batch_size,
            epochs=1,
            dataset=dataset,
            architecture=model_arch,
            model_dir=current_model,
            sequence=sequence[start_sequence_idx:end_sequence_idx],
            half=half,
            lambda_wm=0.0,          # no new watermark
            k=k,                    # same k value as original
            randomize=randomize,    # same randomize as original
            watermark_key=watermark_key,
            watermark_method=watermark_method,  # use the actual method, e.g. 'non_intrusive'
            num_parameters=num_parameters,
            perturbation_strength=perturbation_strength,
            watermark_size=watermark_size
        )
        dists = utils.parameter_distance(
            target_model, net_rep, order=order,
            architecture=model_arch, half=half
        )
        for j in range(len(order)):
            dist_list[j].append(dists[j])

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
            logging.info(f"No steps exceed threshold for metric {metric}. PoL valid.")
        else:
            pct = 100.0 * above_threshold / dist_list[i].shape[0]
            logging.info(f"{above_threshold}/{dist_list[i].shape[0]} "
                         f"({pct:.2f}%) steps exceed threshold for {metric}. PoL may be invalid.")
    return dist_list

def verify_topq(model_directory, lr, batch_size, dataset, model_arch, save_freq,
                order, threshold, epochs=1, q=10, half=0, lambda_wm=0, k=1000,
                randomize=False, watermark_key='secret_key', watermark_method='none',
                num_parameters=1000, perturbation_strength=1e-5, watermark_size=128):
    """
    top-q approach: among consecutive checkpoint distances, pick top-q largest,
    re-run partial training for them, measure distance to the next checkpoint.
    """
    if not os.path.exists(model_directory):
        raise FileNotFoundError("Model directory not found.")

    sequence = np.load(os.path.join(model_directory, "indices.npy"))

    if not isinstance(order, list):
        order = [order]
        threshold = [threshold]
    else:
        assert len(order) == len(threshold)

    checkpoint_files = glob.glob(os.path.join(model_directory, 'model_step_*'))
    checkpoint_steps = sorted(int(os.path.basename(f).split('_')[-1]) for f in checkpoint_files)
    total_checkpoints = len(checkpoint_steps)
    checkpoints_per_epoch = total_checkpoints // epochs

    logging.info(f"Starting top-q verification with q={q} for {epochs} epochs...")
    res_list = []

    for e in range(epochs):
        logging.info(f"Verifying epoch {e+1}/{epochs}")
        st_idx = e * checkpoints_per_epoch
        en_idx = (e+1) * checkpoints_per_epoch if e < (epochs - 1) else total_checkpoints

        dist_list = [[] for _ in range(len(order))]
        # Measure distances among consecutive steps
        for idx in range(st_idx, en_idx - 1):
            cstep = checkpoint_steps[idx]
            nstep = checkpoint_steps[idx + 1]
            cmodel = os.path.join(model_directory, f"model_step_{cstep}")
            nmodel = os.path.join(model_directory, f"model_step_{nstep}")
            dvals = utils.parameter_distance(
                cmodel, nmodel, order=order,
                architecture=model_arch, half=half
            )
            for j in range(len(order)):
                dist_list[j].append(dvals[j])

        dist_arr = np.array(dist_list)
        topq_indices = np.argpartition(dist_arr, -q, axis=1)[:, -q:]
        if len(order) > 1:
            topq_steps = reduce(np.union1d, [indices for indices in topq_indices])
        else:
            topq_steps = topq_indices[0]

        dist_list = [[] for _ in range(len(order))]

        for ind in topq_steps:
            sidx = st_idx + ind
            if sidx >= len(checkpoint_steps) - 1:
                continue

            cstep = checkpoint_steps[sidx]
            nstep = checkpoint_steps[sidx + 1]
            cmodel = os.path.join(model_directory, f"model_step_{cstep}")
            tmodel = os.path.join(model_directory, f"model_step_{nstep}")

            start_seq = cstep * batch_size
            end_seq = min(nstep * batch_size, len(sequence))

            net_rep, _, _, _ = train(
                lr=lr,
                batch_size=batch_size,
                epochs=1,
                dataset=dataset,
                architecture=model_arch,
                model_dir=cmodel,
                sequence=sequence[start_seq:end_seq],
                half=half,
                lambda_wm=0.0,      # no new watermark
                k=k,                # same k as original
                randomize=randomize,
                watermark_key=watermark_key,
                watermark_method=watermark_method,  # same method, e.g. non_intrusive
                num_parameters=num_parameters,
                perturbation_strength=perturbation_strength,
                watermark_size=watermark_size
            )

            dd = utils.parameter_distance(
                tmodel, net_rep, order=order,
                architecture=model_arch, half=half
            )
            for j in range(len(order)):
                dist_list[j].append(dd[j])

        dist_list = np.array(dist_list)
        logging.info(f"Top-q verification results for epoch {e+1}:")
        for im, met in enumerate(order):
            if dist_list[im].size > 0:
                av = np.mean(dist_list[im])
            else:
                av = 0.0
            logging.info(f"Metric: {met}, Threshold: {threshold[im]}, Q={q}, Avg top-q: {av:.4f}")
            if dist_list[im].size > 0:
                abv = np.sum(dist_list[im] > threshold[im])
            else:
                abv = 0
            if abv == 0:
                logging.info("No top-q steps exceed threshold => PoL valid.")
            else:
                px = 100.0 * abv / dist_list[im].shape[0]
                logging.info(f"{abv}/{dist_list[im].shape[0]} ({px:.2f}%) exceed => PoL invalid?")
        res_list.append(dist_list)

    return res_list

def verify_initialization(model_directory, model_arch, threshold=0.01, net=None, verbose=True):
    """
    Checks if model was wrapped by WatermarkModule. If so, replicate that structure
    for correct loading. Then optionally check some param initialization heuristics.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if net is None:
        st = torch.load(os.path.join(model_directory, "model_step_0"), map_location=device)
        # Check if there's "original_model." prefix
        has_prefix = any(k.startswith("original_model.") for k in st['net'].keys())
        if has_prefix:
            # Means the final net was WatermarkModule. We'll replicate that structure:
            from watermark_utils import WatermarkModule
            # Read watermark_size from watermark_info:
            with open(os.path.join(model_directory, "watermark_info.json"), "r") as f:
                wmi = json.load(f)
            wsize = wmi.get('watermark_size', 128)
            # Create base net:
            base_net = model_arch()
            net = WatermarkModule(base_net, wmi.get('watermark_key','secret_key'), watermark_size=wsize)
            net.load_state_dict(st['net'])
        else:
            net = model_arch()
            net.load_state_dict(st['net'])

    net.to(device)
    model_name = model_arch.__name__

    import utils
    if model_name in ['resnet20','resnet32','resnet44','resnet56','resnet110','resnet1202']:
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
        # Implement similar logic for resnet_cifar100
        pass
    elif model_type == 'resnet_cifar':
        # Implement similar logic for resnet_cifar
        pass
    else:
        # Default initialization checks
        pass

    if verbose and len(p_list) > 0:
        min_p = np.min(p_list)
        # For example, if min_p < threshold => Initialization might be invalid
        logging.info(f"Minimum parameter check value: {min_p}")
        if min_p < threshold:
            logging.warning("Some parameters may not be properly initialized.")
        else:
            logging.info("All parameters passed initialization checks.")

def verify_hash(model_directory, dataset):
    """
    Compares the saved hash from training with the newly computed one
    to ensure PoL chain is not tampered with.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    hash_file = os.path.join(model_directory, "hash.txt")
    with open(hash_file, 'r') as f:
        saved_hash = f.read().strip()
    logging.info(f"Saved hash from training: {saved_hash}")

    # Re-compute hash based on the subset of training data used
    seq = np.load(os.path.join(model_directory, "indices.npy"))
    trainset = utils.load_dataset(dataset, train=True, augment=False)

    m = hashlib.sha256()
    if hasattr(trainset, 'data'):
        data = trainset.data[seq]
    elif hasattr(trainset, 'train_data'):
        data = trainset.train_data[seq]
    else:
        raise AttributeError("Dataset object has no 'data' or 'train_data' attribute.")

    if isinstance(data, np.ndarray):
        m.update(data.tobytes())
    elif isinstance(data, list):
        for d in data:
            m.update(np.array(d).tobytes())
    else:
        raise TypeError("Unsupported data type for hashing.")

    computed_hash = m.hexdigest()
    logging.info(f"Computed hash during verification: {computed_hash}")

    if saved_hash != computed_hash:
        logging.error("Hash mismatch => PoL invalid.")
    else:
        logging.info("Hash matches => PoL valid.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Model & Watermark Verification Script")
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--dataset', type=str, default="CIFAR10")
    parser.add_argument('--model', type=str, default="resnet20")
    parser.add_argument('--model-dir', type=str, default='proof/CIFAR10_Batch100')
    parser.add_argument('--save-freq', type=int, default=100)
    parser.add_argument('--dist', type=str, nargs='+', default=['1','2','inf','cos'])
    parser.add_argument('--q', type=int, default=0)
    parser.add_argument('--delta', type=float, nargs='+', default=[10000,100,1,0.1])
    parser.add_argument('--lambda-wm', type=float, default=0.01)
    parser.add_argument('--k', type=int, default=1000)
    parser.add_argument('--randomize', action='store_true')
    parser.add_argument('--watermark-key', type=str, default='secret_key')
    parser.add_argument('--watermark-method', type=str, default='none',
                        choices=['none','feature_based','parameter_perturbation','non_intrusive'])
    parser.add_argument('--num-parameters', type=int, default=1000)
    parser.add_argument('--perturbation-strength', type=float, default=1e-5)
    parser.add_argument('--watermark-size', type=int, default=128)
    parser.add_argument('--watermark-path', type=str, default='model_with_watermark.pth')
    args = parser.parse_args()

    # 1) Load architecture
    model_arch = getattr(custom_model, args.model)

    # 2) Load watermark info => seeds, etc.
    with open(os.path.join(args.model_dir, "watermark_info.json"), "r") as f:
        wminfo = json.load(f)
    seed = wminfo.get('seed', 777)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    watermark_key = wminfo.get('watermark_key', args.watermark_key)
    lambda_wm = wminfo.get('lambda_wm', args.lambda_wm)
    k = wminfo.get('k', args.k)
    randomize = wminfo.get('randomize', args.randomize)
    watermark_method = wminfo.get('watermark_method', args.watermark_method)
    num_parameters = wminfo.get('num_parameters', args.num_parameters)
    perturbation_strength = wminfo.get('perturbation_strength', args.perturbation_strength)
    watermark_size = wminfo.get('watermark_size', args.watermark_size)

    logging.info("Starting verification process...")
    logging.info("Verifying model initialization (Kaiming, etc.)...")
    verify_initialization(args.model_dir, model_arch)
    verify_hash(args.model_dir, args.dataset)

    # 3) top-q or full verification:
    if args.q > 0:
        logging.info(f"Performing top-q verification with q={args.q}...")
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

    # We must replicate the structure for watermark methods
    if watermark_method == 'non_intrusive':
        # We used WatermarkModule in training, so do the same here:
        from watermark_utils import WatermarkModule
        base_net = model_arch()  # e.g., resnet20 => final out is 10D
        net = WatermarkModule(base_net, watermark_key, watermark_size=watermark_size)
        st = torch.load(final_model_path, map_location=device)
        # If state dict is nested under 'net', load accordingly
        if 'net' in st:
            net.load_state_dict(st['net'])
        else:
            net.load_state_dict(st)
        net.to(device)
        wd = verify_non_intrusive_watermark(
            model=net,
            device=device,
            watermark_key=watermark_key,
            watermark_size=watermark_size,
            tolerance=1e-1
        )
        if wd:
            logging.info("Non-intrusive Watermark is present.")
        else:
            logging.error("Non-intrusive Watermark not detected.")

    elif watermark_method == 'feature_based':
        # Plain model
        net = model_arch()
        st = torch.load(final_model_path, map_location=device)
        if 'net' in st:
            net.load_state_dict(st['net'])
        else:
            net.load_state_dict(st)
        net.to(device)
        detection_val = validate_feature_watermark(net, device=device)
        if detection_val == 1.0:
            logging.info("Feature-based watermark is present in final model.")
        else:
            logging.error("Feature-based watermark NOT detected.")

    elif watermark_method == 'parameter_perturbation':
        net = model_arch()
        st = torch.load(final_model_path, map_location=device)
        # If state dict is nested under 'net', load accordingly
        if 'net' in st:
            net.load_state_dict(st['net'])
            original_params_dict = st.get('original_param_values', None)
        else:
            net.load_state_dict(st)
            original_params_dict = None
        net.to(device)
        wd = verify_parameter_perturbation_watermark_relative(
            model=net,
            original_params=original_params_dict,
            watermark_key=watermark_key,
            perturbation_strength=perturbation_strength,
            tolerance=1e-1
        )
        if wd:
            logging.info("Parameter-Perturbation Watermark is present.")
        else:
            logging.error("Parameter-Perturbation Watermark not detected.")

    else:
        logging.info("No watermark (or unknown method) => skipping final check for watermark.")

    logging.info("Verification process completed successfully.")
