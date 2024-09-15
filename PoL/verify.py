import argparse
import os
import hashlib
import numpy as np
import torch
import logging
from functools import reduce
import utils
from train import train
import model as custom_model

from watermark_verify import run_feature_based_watermark_verification  # Adjust the import if needed

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')


def verify_all(dir, lr, batch_size, dataset, architecture, save_freq, order, threshold, half=0):
    if not os.path.exists(dir):
        raise FileNotFoundError("Model directory not found")
    sequence = np.load(os.path.join(dir, "indices.npy"))

    if not isinstance(order, list):
        order = [order]
        threshold = [threshold]
    else:
        assert len(order) == len(threshold)

    dist_list = [[] for _ in range(len(order))]

    target_model = os.path.join(dir, f"model_step_0")
    for i in range(0, sequence.shape[0], save_freq):
        previous_state = target_model
        if i + save_freq >= sequence.shape[0]:
            target_model = os.path.join(dir, f"model_step_{sequence.shape[0]}")
            reproduce, _, _ = train(
                lr=lr,
                batch_size=batch_size,
                epochs=1,
                dataset=dataset,
                architecture=architecture,
                model_dir=previous_state,
                sequence=sequence[i:],
                half=half,
                lambda_wm=0,  # Disable watermarking during verification
                k=0,
                randomize=False
            )
        else:
            target_model = os.path.join(dir, f"model_step_{i + save_freq}")
            reproduce, _, _ = train(
                lr=lr,
                batch_size=batch_size,
                epochs=1,
                dataset=dataset,
                architecture=architecture,
                model_dir=previous_state,
                sequence=sequence[i:i+save_freq],
                half=half,
                lambda_wm=0,
                k=0,
                randomize=False
            )
        res = utils.parameter_distance(target_model, reproduce, order=order,
                                       architecture=architecture, half=half)
        for j in range(len(order)):
            dist_list[j].append(res[j])

    dist_list = np.array(dist_list)
    for k_idx in range(len(order)):
        logging.info(f"Distance metric: {order[k_idx]} || threshold: {threshold[k_idx]}")
        logging.info(f"Average distance: {np.average(dist_list[k_idx])}, Max distance: {np.max(dist_list[k_idx])}, Min distance: {np.min(dist_list[k_idx])}")
        above_threshold = np.sum(dist_list[k_idx] > threshold[k_idx])
        if above_threshold == 0:
            logging.info("None of the steps is above the threshold, the proof-of-learning is valid.")
        else:
            percentage = 100 * (above_threshold / dist_list[k_idx].shape[0])
            logging.info(f"{above_threshold} / {dist_list[k_idx].shape[0]} "
                         f"({percentage:.2f}%) "
                         f"of the steps are above the threshold, the proof-of-learning is invalid.")
    return dist_list


def verify_topq(dir, lr, batch_size, dataset, architecture, save_freq, order, threshold, epochs=1, q=10, half=0):
    if not os.path.exists(dir):
        raise FileNotFoundError("Model directory not found")
    sequence = np.load(os.path.join(dir, "indices.npy"))

    if not isinstance(order, list):
        order = [order]
        threshold = [threshold]
    else:
        assert len(order) == len(threshold)

    ckpt_per_epoch = sequence.shape[0] / epochs / save_freq
    res_list = []

    for epoch in range(epochs):
        logging.info(f"Verifying epoch {epoch + 1}/{epochs}")
        start = int(round(ckpt_per_epoch * epoch))
        end = int(round(ckpt_per_epoch * (epoch + 1)))
        dist_list = [[] for _ in range(len(order))]
        next_model = os.path.join(dir, f"model_step_{start * save_freq}")
        for i in range(start, end):
            current_model = next_model
            if (i + 1) * save_freq >= sequence.shape[0]:
                next_model = os.path.join(dir, f"model_step_{sequence.shape[0]}")
            else:
                next_model = os.path.join(dir, f"model_step_{(i + 1) * save_freq}")
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

        # Remove duplicates
        topq_steps = np.unique(topq_steps)

        dist_list = [[] for _ in range(len(order))]
        for ind in topq_steps:
            step = int((start + ind) * save_freq)
            current_model = os.path.join(dir, f"model_step_{step}")
            if step + save_freq >= sequence.shape[0]:
                target_model = os.path.join(dir, f"model_step_{sequence.shape[0]}")
                reproduce, _, _ = train(
                    lr=lr,
                    batch_size=batch_size,
                    epochs=1,
                    dataset=dataset,
                    architecture=architecture,
                    model_dir=current_model,
                    sequence=sequence[step:],
                    half=half,
                    lambda_wm=0,
                    k=0,
                    randomize=False
                )
            else:
                target_model = os.path.join(dir, f"model_step_{step + save_freq}")
                reproduce, _, _ = train(
                    lr=lr,
                    batch_size=batch_size,
                    epochs=1,
                    dataset=dataset,
                    architecture=architecture,
                    model_dir=current_model,
                    sequence=sequence[step:step+save_freq],
                    half=half,
                    lambda_wm=0,
                    k=0,
                    randomize=False
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
        saved_hash = f.read()

    trainset = utils.load_dataset(dataset, True)
    subset = torch.utils.data.Subset(trainset, sequence)
    m = hashlib.sha256()
    if hasattr(subset.dataset, 'data'):
        data = subset.dataset.data
    elif hasattr(subset.dataset, 'train_data'):
        data = subset.dataset.train_data
    else:
        raise AttributeError("Dataset object has no attribute 'data' or 'train_data'.")

    for d in data:
        m.update(str(d).encode('utf-8'))

    computed_hash = m.hexdigest()
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
                        help="Models defined in model.py or any torchvision model.\n"
                             "Recommendation for CIFAR-10: resnet20/32/44/56/110/1202\n"
                             "Recommendation for CIFAR-100: resnet18/34/50/101/152")
    parser.add_argument('--model-dir', help='Path to the model directory', type=str, default='proof/CIFAR10_test')
    parser.add_argument('--save-freq', type=int, default=100, help='Frequency of saving checkpoints')
    parser.add_argument('--dist', type=str, nargs='+', default=['1', '2', 'inf', 'cos'],
                        help='Metric for computing distance, cos, 1, 2, or inf')
    parser.add_argument('--q', type=int, default=2, help="Set to >1 to enable top-q verification,"
                                                         "otherwise all steps will be verified.")
    parser.add_argument('--delta', type=float, nargs='+', default=[10000, 100, 1, 0.1],
                        help='Thresholds for verification corresponding to distance metrics')
    parser.add_argument('--watermark-path', help='Path to the watermarked model', type=str, default='model_with_watermark.pth')

    arg = parser.parse_args()
    architecture = eval(f"custom_model.{arg.model}")

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
            half=0  # Adjust if needed
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
            half=0  # Adjust if needed
        )

    logging.info("Verifying watermark presence in the model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_path = arg.watermark_path

    # Call the watermark verification function
    run_feature_based_watermark_verification(model_path, arg.model, device)
