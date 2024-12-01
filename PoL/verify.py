import argparse
import os
import glob
import hashlib
import numpy as np
import torch
import logging
from functools import reduce
import utils
from train_with_watermark import train  # Import the train function
import model as custom_model
import random
from watermark_utils import (
    run_feature_based_watermark_verification,
    verify_parameter_perturbation_watermark,
    verify_non_intrusive_watermark
)
import json  # Import json to load watermarking information

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')


def verify_all(dir, lr, batch_size, dataset, architecture, save_freq, order, threshold, half=0,
               lambda_wm=0, k=1000, randomize=False, watermark_key='secret_key',
               watermark_method='feature_based', num_parameters=1000, perturbation_strength=1e-5,
               watermark_size=128):
    if not os.path.exists(dir):
        raise FileNotFoundError("Model directory not found")
    sequence = np.load(os.path.join(dir, "indices.npy"))

    if not isinstance(order, list):
        order = [order]
        threshold = [threshold]
    else:
        assert len(order) == len(threshold)

    dist_list = [[] for _ in range(len(order))]

    # Get list of available checkpoints
    checkpoint_files = glob.glob(os.path.join(dir, 'model_step_*'))
    checkpoint_steps = sorted([int(os.path.basename(f).split('_')[-1]) for f in checkpoint_files])

    for idx in range(len(checkpoint_steps) - 1):
        current_step = checkpoint_steps[idx]
        next_step = checkpoint_steps[idx + 1]
        current_model = os.path.join(dir, f"model_step_{current_step}")
        target_model = os.path.join(dir, f"model_step_{next_step}")

        # Reproduce the training step with the correct watermarking method
        start_sequence_idx = current_step * batch_size
        end_sequence_idx = next_step * batch_size
        if end_sequence_idx > len(sequence):
            end_sequence_idx = len(sequence)
        reproduce, _, _ = train(
            lr=lr,
            batch_size=batch_size,
            epochs=1,
            dataset=dataset,
            architecture=architecture,
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

        # Compute parameter distance
        res = utils.parameter_distance(target_model, reproduce, order=order,
                                       architecture=architecture, half=half)
        for j in range(len(order)):
            dist_list[j].append(res[j])

    dist_list = np.array(dist_list)
    for k_idx in range(len(order)):
        logging.info(f"Distance metric: {order[k_idx]} || threshold: {threshold[k_idx]}")
        logging.info(f"Average distance: {np.average(dist_list[k_idx])}, "
                     f"Max distance: {np.max(dist_list[k_idx])}, Min distance: {np.min(dist_list[k_idx])}")
        above_threshold = np.sum(dist_list[k_idx] > threshold[k_idx])
        if above_threshold == 0:
            logging.info("None of the steps is above the threshold, the proof-of-learning is valid.")
        else:
            percentage = 100 * (above_threshold / dist_list[k_idx].shape[0])
            logging.info(f"{above_threshold} / {dist_list[k_idx].shape[0]} "
                         f"({percentage:.2f}%) "
                         f"of the steps are above the threshold, the proof-of-learning is invalid.")
    return dist_list


def verify_initialization(dir, architecture, threshold=0.01, net=None, verbose=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if net is None:
        net = architecture()
        state = torch.load(os.path.join(dir, "model_step_0"), map_location=device)
        net.load_state_dict(state['net'])
    net.to(device)
    model_name = architecture.__name__

    # Determine model type
    if model_name in ['resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']:
        model_type = 'resnet_cifar'
    elif model_name == 'resnet50':
        model_type = 'resnet_cifar100'
    elif 'resnet' in model_name:
        model_type = 'resnet'
    else:
        model_type = 'default'

    p_list = []

    # Check initialization based on model type
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
            logging.info(f"The initialized weights do not follow the initialization strategy."
                         f" The minimum p-value is {min_p_value} < threshold ({threshold})."
                         f" The proof-of-learning is not valid.")
        else:
            logging.info("The proof-of-learning passed the initialization verification.")
    return p_list


def verify_hash(dir, dataset):
    if not os.path.exists(dir):
        raise FileNotFoundError("Model directory not found")

    sequence = np.load(os.path.join(dir, "indices.npy"))

    with open(os.path.join(dir, "hash.txt"), "r") as f:
        saved_hash = f.read().strip()

    # Load watermarking information
    with open(os.path.join(dir, "watermark_info.json"), "r") as f:
        watermark_info = json.load(f)

    # Set the random seed used in watermarking
    seed = watermark_info.get('seed', 777)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Load dataset without transformations
    trainset = utils.load_dataset(dataset, train=True, augment=False)
    m = hashlib.sha256()

    # Get the data corresponding to the subset
    if hasattr(trainset, 'data'):
        data = trainset.data[sequence]
    elif hasattr(trainset, 'train_data'):
        data = trainset.train_data[sequence]
    else:
        raise AttributeError("Dataset object has no attribute 'data' or 'train_data'.")

    logging.info(f"Data shape: {data.shape}, Data type: {data.dtype}")
    logging.info(f"First data sample hash: {hashlib.sha256(data[0].tobytes()).hexdigest()}")

    # Convert data to bytes and update the hash
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
        logging.info("Hash doesn't match. The proof is invalid.")
    else:
        logging.info("Hash of the proof is valid.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Model and Watermark Verification Script")
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--dataset', type=str, default="CIFAR10")
    parser.add_argument('--model', type=str, default="resnet20",
                        help="Models defined in model.py or any torchvision model.")
    parser.add_argument('--model-dir', type=str, default='proof/CIFAR10_Batch100', help='Path to the model directory')
    parser.add_argument('--save-freq', type=int, default=100, help='Frequency of saving checkpoints')
    parser.add_argument('--dist', type=str, nargs='+', default=['1', '2', 'inf', 'cos'],
                        help='Metric for computing distance, e.g., cos, 1, 2, or inf')
    parser.add_argument('--delta', type=float, nargs='+', default=[10000, 100, 1, 0.1],
                        help='Thresholds for verification corresponding to distance metrics')
    parser.add_argument('--watermark-path', type=str, default='model_with_watermark.pth', help='Path to the watermarked model')
    parser.add_argument('--lambda-wm', type=float, default=0.01, help='Balancing parameter for watermark loss')
    parser.add_argument('--k', type=int, default=1000, help='Watermark embedding frequency (in steps)')
    parser.add_argument('--randomize', action='store_true', help='Randomize watermark embedding intervals')
    parser.add_argument('--watermark-key', type=str, default='secret_key', help='Key used for watermark embedding')
    parser.add_argument('--watermark-method', type=str, default='feature_based',
                        choices=['feature_based', 'parameter_perturbation', 'non_intrusive'],
                        help='Watermarking method to use during verification')
    parser.add_argument('--num-parameters', type=int, default=1000, help='Number of parameters to perturb for parameter perturbation watermarking')
    parser.add_argument('--perturbation-strength', type=float, default=1e-5, help='Strength of parameter perturbations')
    parser.add_argument('--watermark-size', type=int, default=128, help='Size of the watermark for non-intrusive watermarking')

    arg = parser.parse_args()
    architecture = getattr(custom_model, arg.model)

    # Load watermarking information
    with open(os.path.join(arg.model_dir, "watermark_info.json"), "r") as f:
        watermark_info = json.load(f)

    # Set the random seed used in watermarking
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

    logging.info("Starting the verification process...")
    logging.info("Verifying model initialization...")
    verify_initialization(arg.model_dir, architecture)
    verify_hash(arg.model_dir, arg.dataset)

    logging.info("Performing full verification...")
    verify_all(
        dir=arg.model_dir,
        lr=arg.lr,
        batch_size=arg.batch_size,
        dataset=arg.dataset,
        architecture=architecture,
        save_freq=arg.save_freq,
        order=arg.dist,
        threshold=arg.delta,
        half=0,  # Adjust if needed
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

    # Load the model
    model = architecture()
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)

    # Call the appropriate watermark verification function
    if watermark_method == 'feature_based':
        run_feature_based_watermark_verification(
            model=model,
            device=device,
            watermark_key=watermark_key
        )
    elif watermark_method == 'parameter_perturbation':
        watermark_detected = verify_parameter_perturbation_watermark(
            model=model,
            watermark_key=watermark_key,
            perturbation_strength=perturbation_strength,
            num_parameters=num_parameters,
            tolerance=1e-6
        )
        if watermark_detected:
            logging.info("Parameter Perturbation Watermark verification successful: Watermark is present in the model.")
        else:
            logging.error("Parameter Perturbation Watermark verification failed: Watermark not detected in the model.")
    elif watermark_method == 'non_intrusive':
        watermark_detected = verify_non_intrusive_watermark(
            model=model,
            device=device,
            watermark_key=watermark_key,
            watermark_size=watermark_size,
            tolerance=1e-5
        )
        if watermark_detected:
            logging.info("Non-Intrusive Watermark verification successful: Watermark is present in the model.")
        else:
            logging.error("Non-Intrusive Watermark verification failed: Watermark not detected in the model.")
    else:
        logging.error(f"Unknown watermarking method: {watermark_method}")

    logging.info("Verification process concluded successfully.")
