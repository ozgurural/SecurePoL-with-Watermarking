import argparse
import os
import glob
import hashlib
import numpy as np
import torch
import logging
from functools import reduce
import utils
from train_with_watermark import train
import model as custom_model
import random
import json
from watermark_utils import (
    run_feature_based_watermark_verification,
    verify_parameter_perturbation_watermark,
    verify_non_intrusive_watermark
)

# Set logging level to INFO for general progress; use DEBUG for more detail
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')


def verify_all(model_directory, lr, batch_size, dataset, model_arch, save_freq, order, threshold, half=0,
               lambda_wm=0, k=1000, randomize=False, watermark_key='secret_key',
               watermark_method='none', num_parameters=1000, perturbation_strength=1e-5,
               watermark_size=128):
    if not os.path.exists(model_directory):
        raise FileNotFoundError("Model directory not found.")
    sequence = np.load(os.path.join(model_directory, "indices.npy"))

    if not isinstance(order, list):
        order = [order]
        threshold = [threshold]
    else:
        assert len(order) == len(threshold)

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
        reproduce, _, _ = train(
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

        res = utils.parameter_distance(target_model, reproduce, order=order,
                                       architecture=model_arch, half=half)
        for j in range(len(order)):
            dist_list[j].append(res[j])

    dist_list = np.array(dist_list)
    logging.info("Full verification results:")
    for i, metric in enumerate(order):
        avg_dist = np.average(dist_list[i])
        max_dist = np.max(dist_list[i])
        min_dist = np.min(dist_list[i])
        logging.info(f"Metric: {metric}, Threshold: {threshold[i]}, "
                     f"Avg: {avg_dist:.4f}, Max: {max_dist:.4f}, Min: {min_dist:.4f}")
        above_threshold = np.sum(dist_list[i] > threshold[i])
        if above_threshold == 0:
            logging.info(f"No steps exceed threshold for metric {metric}. PoL appears valid.")
        else:
            percentage = 100 * (above_threshold / dist_list[i].shape[0])
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
        assert len(order) == len(threshold)

    checkpoint_files = glob.glob(os.path.join(model_directory, 'model_step_*'))
    checkpoint_steps = sorted(int(os.path.basename(f).split('_')[-1]) for f in checkpoint_files)
    total_checkpoints = len(checkpoint_steps)
    checkpoints_per_epoch = total_checkpoints // epochs

    logging.info(f"Starting top-q verification with q={q} for {epochs} epochs...")
    res_list = []

    for epoch in range(epochs):
        logging.info(f"Verifying epoch {epoch + 1}/{epochs}")
        start_idx = epoch * checkpoints_per_epoch
        end_idx = (epoch + 1) * checkpoints_per_epoch if epoch < epochs - 1 else total_checkpoints

        dist_list = [[] for _ in range(len(order))]
        # Compute distances between consecutive checkpoints
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

        # Re-verify top-q steps by reproducing training
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
            reproduce, _, _ = train(
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
            res = utils.parameter_distance(target_model, reproduce, order=order,
                                           architecture=model_arch, half=half)
            for j in range(len(order)):
                dist_list[j].append(res[j])

        dist_list = np.array(dist_list)
        logging.info(f"Top-q verification results for epoch {epoch+1}:")
        for idx, metric in enumerate(order):
            avg_dist = np.average(dist_list[idx]) if dist_list[idx].size > 0 else 0.0
            logging.info(f"Metric: {metric}, Threshold: {threshold[idx]}, Q={q}, Avg top-q: {avg_dist:.4f}")
            if dist_list[idx].size > 0:
                above_threshold = np.sum(dist_list[idx] > threshold[idx])
            else:
                above_threshold = 0
            if above_threshold == 0:
                logging.info("None of the top-q steps exceed the threshold. PoL appears valid.")
            else:
                percentage = 100 * (above_threshold / dist_list[idx].shape[0])
                logging.info(f"{above_threshold}/{dist_list[idx].shape[0]} "
                             f"({percentage:.2f}%) top-q steps exceed threshold. PoL may be invalid.")

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
                weight = net.state_dict()[name.replace('bias', 'weight')]
                p_list.append(utils.check_weights_initialization([weight, param], 'default_bias'))
    elif model_type == 'resnet_cifar100':
        for name, param in net.named_parameters():
            if len(param.shape) == 4:
                p_list.append(utils.check_weights_initialization(param, 'default'))
            elif 'weight' in name and 'fc' in name:
                p_list.append(utils.check_weights_initialization(param, 'default'))
            elif 'bias' in name and ('fc' in name or 'linear' in name):
                weight = net.state_dict()[name.replace('bias', 'weight')]
                p_list.append(utils.check_weights_initialization([weight, param], 'default_bias'))
    elif model_type == 'resnet_cifar':
        for name, param in net.named_parameters():
            if 'fc' in name or 'conv' in name or 'linear' in name:
                if 'weight' in name:
                    p_list.append(utils.check_weights_initialization(param, 'resnet_cifar'))
                elif 'bias' in name:
                    weight = net.state_dict()[name.replace('bias', 'weight')]
                    p_list.append(utils.check_weights_initialization([weight, param], 'default_bias'))
    else:
        for name, param in net.named_parameters():
            if 'fc' in name or 'conv' in name or 'linear' in name:
                if 'weight' in name:
                    p_list.append(utils.check_weights_initialization(param, 'default'))
                elif 'bias' in name:
                    weight = net.state_dict()[name.replace('bias', 'weight')]
                    p_list.append(utils.check_weights_initialization([weight, param], 'default_bias'))

    if verbose:
        min_p_value = np.min(p_list)
        if min_p_value < threshold:
            logging.info(f"Initialization check: min p-value {min_p_value:.4f} < {threshold}, PoL may not be valid.")
        else:
            logging.info("Initialization verification passed.")
    return p_list


def verify_hash(model_directory, dataset):
    if not os.path.exists(model_directory):
        raise FileNotFoundError("Model directory not found")

    sequence = np.load(os.path.join(model_directory, "indices.npy"))

    with open(os.path.join(model_directory, "hash.txt"), "r") as f:
        saved_hash = f.read().strip()

    # No duplicate seed setting here, since we already set it once in main
    # Just compute hash
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

    computed_hash = m.hexdigest()
    logging.info(f"Saved hash from training: {saved_hash}")
    logging.info(f"Computed hash during verification: {computed_hash}")

    if saved_hash != computed_hash:
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
    parser.add_argument('--dist', type=str, nargs='+', default=['1', '2', 'inf', 'cos'], help='Distance metrics')
    parser.add_argument('--q', type=int, default=0, help='>0 for top-q verification')
    parser.add_argument('--delta', type=float, nargs='+', default=[10000, 100, 1, 0.1], help='Thresholds per metric')
    parser.add_argument('--lambda-wm', type=float, default=0.01)
    parser.add_argument('--k', type=int, default=1000)
    parser.add_argument('--randomize', action='store_true')
    parser.add_argument('--watermark-key', type=str, default='secret_key')
    parser.add_argument('--watermark-method', type=str, default='none',
                        choices=['none', 'feature_based', 'parameter_perturbation', 'non_intrusive'])
    parser.add_argument('--num-parameters', type=int, default=1000)
    parser.add_argument('--perturbation-strength', type=float, default=1e-5)
    parser.add_argument('--watermark-size', type=int, default=128)
    parser.add_argument('--watermark-path', type=str, default='model_with_watermark.pth')

    arg = parser.parse_args()
    model_arch = getattr(custom_model, arg.model)

    # Load watermarking info for seed and other parameters
    with open(os.path.join(arg.model_dir, "watermark_info.json"), "r") as f:
        watermark_info = json.load(f)

    seed = watermark_info.get('seed', 777)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    watermark_key = watermark_info.get('watermark_key', arg.watermark_key)
    lambda_wm = watermark_info.get('lambda_wm', arg.lambda_wm)
    k = watermark_info.get('k', arg.k)
    randomize = watermark_info.get('randomize', arg.randomize)
    watermark_method = watermark_info.get('watermark_method', arg.watermark_method)
    num_parameters = watermark_info.get('num_parameters', arg.num_parameters)
    perturbation_strength = watermark_info.get('perturbation_strength', arg.perturbation_strength)
    watermark_size = watermark_info.get('watermark_size', arg.watermark_size)

    logging.info("Starting verification process...")
    logging.info("Verifying model initialization...")
    verify_initialization(arg.model_dir, model_arch)
    verify_hash(arg.model_dir, arg.dataset)

    if arg.q > 0:
        logging.info(f"Performing top-q verification with q={arg.q}...")
        verify_topq(
            model_directory=arg.model_dir,
            lr=arg.lr,
            batch_size=arg.batch_size,
            dataset=arg.dataset,
            model_arch=model_arch,
            save_freq=arg.save_freq,
            order=arg.dist,
            threshold=arg.delta,
            epochs=arg.epochs,
            q=arg.q,
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
            model_directory=arg.model_dir,
            lr=arg.lr,
            batch_size=arg.batch_size,
            dataset=arg.dataset,
            model_arch=model_arch,
            save_freq=arg.save_freq,
            order=arg.dist,
            threshold=arg.delta,
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

    logging.info("Verifying watermark presence in the model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = arg.watermark_path

    model = model_arch()
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)

    if watermark_method == 'none':
        logging.info("No watermark verification needed.")
    elif watermark_method == 'feature_based':
        run_feature_based_watermark_verification(model=model, device=device, watermark_key=watermark_key)
    elif watermark_method == 'parameter_perturbation':
        wd = verify_parameter_perturbation_watermark(
            model=model,
            watermark_key=watermark_key,
            perturbation_strength=perturbation_strength,
            num_parameters=num_parameters,
            tolerance=1e-6
        )
        if wd:
            logging.info("Parameter Perturbation Watermark present.")
        else:
            logging.error("Parameter Perturbation Watermark not detected.")
    elif watermark_method == 'non_intrusive':
        wd = verify_non_intrusive_watermark(
            model=model,
            device=device,
            watermark_key=watermark_key,
            watermark_size=watermark_size,
            tolerance=1e-5
        )
        if wd:
            logging.info("Non-Intrusive Watermark present.")
        else:
            logging.error("Non-Intrusive Watermark not detected.")
    else:
        logging.error(f"Unknown watermarking method: {watermark_method}")

    logging.info("Verification process completed successfully.")
