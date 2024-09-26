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
from watermark_utils import run_feature_based_watermark_verification  # Import the verification function
import json  # Import json to load watermarking information

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')

def verify_all(dir, lr, batch_size, dataset, architecture, save_freq, order, threshold, half=0,
               lambda_wm=0, k=1000, randomize=False, watermark_key='secret_key'):
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

        # Reproduce the training step without watermark embedding
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
            lambda_wm=lambda_wm,  # Use the lambda_wm provided
            k=k,
            randomize=randomize,
            watermark_key=watermark_key
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


def verify_topq(dir, lr, batch_size, dataset, architecture, save_freq, order, threshold, epochs=1, q=10,
                half=0, lambda_wm=0, k=1000, randomize=False, watermark_key='secret_key'):
    if not os.path.exists(dir):
        raise FileNotFoundError("Model directory not found")
    sequence = np.load(os.path.join(dir, "indices.npy"))

    if not isinstance(order, list):
        order = [order]
        threshold = [threshold]
    else:
        assert len(order) == len(threshold)

    # Get list of available checkpoints
    checkpoint_files = glob.glob(os.path.join(dir, 'model_step_*'))
    checkpoint_steps = sorted([int(os.path.basename(f).split('_')[-1]) for f in checkpoint_files])
    total_checkpoints = len(checkpoint_steps)

    res_list = []

    checkpoints_per_epoch = total_checkpoints // epochs

    for epoch in range(epochs):
        logging.info(f"Verifying epoch {epoch + 1}/{epochs}")
        start_idx = epoch * checkpoints_per_epoch
        end_idx = (epoch + 1) * checkpoints_per_epoch if epoch < epochs - 1 else total_checkpoints

        epoch_checkpoint_steps = checkpoint_steps[start_idx:end_idx]
        dist_list = [[] for _ in range(len(order))]
        for idx in range(len(epoch_checkpoint_steps) - 1):
            current_step = epoch_checkpoint_steps[idx]
            next_step = epoch_checkpoint_steps[idx + 1]
            current_model = os.path.join(dir, f"model_step_{current_step}")
            next_model = os.path.join(dir, f"model_step_{next_step}")
            res = utils.parameter_distance(current_model, next_model, order=order,
                                           architecture=architecture, half=half)
            for j in range(len(order)):
                dist_list[j].append(res[j])

        dist_arr = np.array(dist_list)
        topq_indices = np.argpartition(dist_arr, -q, axis=1)[:, -q:]
        if len(order) > 1:
            # Union the top-q steps of all distance metrics to avoid redundant computation
            topq_steps = reduce(np.union1d, [indices for indices in topq_indices])
        else:
            topq_steps = topq_indices[0]

        # Remove duplicates and sort the steps
        topq_steps = np.unique(topq_steps)

        dist_list = [[] for _ in range(len(order))]
        for ind in topq_steps:
            step_idx = start_idx + ind
            if step_idx >= len(checkpoint_steps) - 1:
                continue  # Skip if index is out of bounds
            current_step = checkpoint_steps[step_idx]
            next_step = checkpoint_steps[step_idx + 1]
            current_model = os.path.join(dir, f"model_step_{current_step}")
            target_model = os.path.join(dir, f"model_step_{next_step}")

            # Calculate the sequence indices for the steps
            start_sequence_idx = current_step * batch_size
            end_sequence_idx = next_step * batch_size

            # Ensure indices are within bounds
            if end_sequence_idx > len(sequence):
                end_sequence_idx = len(sequence)

            # Reproduce the training step without watermark embedding
            reproduce, _, _ = train(
                lr=lr,
                batch_size=batch_size,
                epochs=1,
                dataset=dataset,
                architecture=architecture,
                model_dir=current_model,
                sequence=sequence[start_sequence_idx:end_sequence_idx],
                half=half,
                lambda_wm=lambda_wm,  # Use the lambda_wm provided
                k=k,
                randomize=randomize,
                watermark_key=watermark_key
            )
            res = utils.parameter_distance(target_model, reproduce, order=order,
                                           architecture=architecture, half=half)
            for j in range(len(order)):
                dist_list[j].append(res[j])

        dist_list = np.array(dist_list)
        for k_idx in range(len(order)):
            logging.info(f"Distance metric: {order[k_idx]} || threshold: {threshold[k_idx]} || Q={q}")
            logging.info(f"Average top-q distance: {np.average(dist_list[k_idx])}")
            above_threshold = np.sum(dist_list[k_idx] > threshold[k_idx])
            if above_threshold == 0:
                logging.info("None of the steps is above the threshold, the proof-of-learning is valid.")
            else:
                percentage = 100 * (above_threshold / dist_list[k_idx].shape[0])
                logging.info(f"{above_threshold} / {dist_list[k_idx].shape[0]} "
                             f"({percentage:.2f}%) "
                             f"of the steps are above the threshold, the proof-of-learning is invalid.")
        res_list.append(dist_list)
    return res_list


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

    # Apply the same watermarking process to the data if it modifies the data
    # For this example, we assume the data is not modified during watermarking
    # If your watermarking process modifies the data used for hashing, apply the modifications here
    # data = apply_watermark_to_data(data, watermark_info)

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
    parser.add_argument('--q', type=int, default=2, help="Set to >1 to enable top-q verification")
    parser.add_argument('--delta', type=float, nargs='+', default=[10000, 100, 1, 0.1],
                        help='Thresholds for verification corresponding to distance metrics')
    parser.add_argument('--watermark-path', type=str, default='model_with_watermark.pth', help='Path to the watermarked model')
    parser.add_argument('--lambda-wm', type=float, default=0.01, help='Balancing parameter for watermark loss')
    parser.add_argument('--k', type=int, default=1000, help='Watermark embedding frequency (in steps)')
    parser.add_argument('--randomize', action='store_true', help='Randomize watermark embedding intervals')
    parser.add_argument('--watermark-key', type=str, default='secret_key', help='Key used for watermark embedding')

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

    watermark_seed = watermark_info.get('watermark_seed', 777)
    watermark_key = watermark_info.get('watermark_key', 'secret_key')
    lambda_wm = watermark_info.get('lambda_wm', arg.lambda_wm)
    k = watermark_info.get('k', arg.k)
    randomize = watermark_info.get('randomize', arg.randomize)

    logging.info("Starting the verification process...")
    logging.info("Verifying model initialization...")
    verify_initialization(arg.model_dir, architecture)
    verify_hash(arg.model_dir, arg.dataset)

    if arg.q > 0:
        logging.info("Performing top-q verification...")
        verify_topq(
            dir=arg.model_dir,
            lr=arg.lr,
            batch_size=arg.batch_size,
            dataset=arg.dataset,
            architecture=architecture,
            save_freq=arg.save_freq,
            order=arg.dist,
            threshold=arg.delta,
            epochs=arg.epochs,
            q=arg.q,
            half=0,  # Adjust if needed
            lambda_wm=lambda_wm,  # Use the lambda_wm from watermark_info
            k=k,
            randomize=randomize,
            watermark_key=watermark_key
        )
    else:
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
            lambda_wm=lambda_wm,  # Use the lambda_wm from watermark_info
            k=k,
            randomize=randomize,
            watermark_key=watermark_key
        )

    logging.info("Verifying watermark presence in the model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = arg.watermark_path

    # Call the watermark verification function
    run_feature_based_watermark_verification(
        model_path=model_path,
        model_name=arg.model,
        device=device,
        watermark_key=watermark_key  # Pass the watermark_key to ensure consistency
    )

    logging.info("Verification process concluded successfully.")
