# utils.py

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from scipy import stats

from watermark_utils import WatermarkModule  # <-- PATCH: So we can instantiate WatermarkModule if needed.


def get_parameters(net, numpy: bool = False):
    """Flatten parameters to a *single 1‑D* tensor  (float32)."""
    if isinstance(net, tuple): net = net[0]
    vec = torch.cat([p.detach().flatten().float() for p in net.parameters()])
    return vec.cpu().numpy() if numpy else vec


def set_parameters(net, parameters, device):
    # Unflatten & load weights from a list of np arrays to a torch model
    for i, (name, param) in enumerate(net.named_parameters()):
        param.data = torch.Tensor(parameters[i]).to(device)
    return net


def create_sequences(batch_size, dataset_size, epochs):
    sequence = np.concatenate([
        np.random.default_rng().choice(dataset_size, size=dataset_size, replace=False)
        for _ in range(epochs)
    ])
    num_batch = int(len(sequence) // batch_size)
    return np.reshape(sequence[:num_batch * batch_size], [num_batch, batch_size])


def consistent_type(
        model,
        architecture=None,
        device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
        half=False,
        watermark_key="secret_key",  # <-- PATCH: Added these two
        watermark_size=128  # <-- PATCH: so we can create WatermarkModule if needed
):
    """
    Ensures 'model' is loaded as a single flat weights tensor on the given device,
    but also checks if the checkpoint has 'original_model.*' keys. If yes, we wrap
    the base architecture with WatermarkModule before load_state_dict().
    """
    if isinstance(model, str):
        assert architecture is not None, "Need architecture if 'model' is a path"
        state = torch.load(model)

        # <-- PATCH: Detect if checkpoint has "original_model."
        has_prefix = False
        if 'net' in state:
            net_dict = state['net']
            has_prefix = any(k.startswith("original_model.") for k in net_dict.keys())
        else:
            # If there's no 'net' sub-dict, fallback
            net_dict = state
            has_prefix = any(k.startswith("original_model.") for k in net_dict.keys())

        # Based on has_prefix, create correct net
        if has_prefix:
            base_net = architecture()
            net = WatermarkModule(base_net, watermark_key, watermark_size=watermark_size)
        else:
            net = architecture()

        net.load_state_dict(net_dict)
        net.to(device)
        weights = get_parameters(net)

    elif isinstance(model, np.ndarray):
        weights = torch.tensor(model)

    elif not isinstance(model, torch.Tensor):
        # Possibly a loaded model object
        # Convert to flatten
        weights = get_parameters(model)
    else:
        # model is already a Tensor
        weights = model

    if half:
        weights = weights.half()
    return weights.to(device)


def parameter_distance(
        model1,
        model2,
        order=2,
        architecture=None,
        half=False,
        # <-- PATCH: pass watermark info so consistent_type can load WatermarkModule if needed
        watermark_key="secret_key",
        watermark_size=128
):
    """
    Compute difference between two checkpoints, returning a list of distances
    (one per 'order' if it's a list).
    """
    w1 = consistent_type(
        model1,
        architecture=architecture,
        half=half,
        watermark_key=watermark_key,  # <-- PATCH
        watermark_size=watermark_size  # <-- PATCH
    )
    w2 = consistent_type(
        model2,
        architecture=architecture,
        half=half,
        watermark_key=watermark_key,  # <-- PATCH
        watermark_size=watermark_size  # <-- PATCH
    )

    orders = [order] if not isinstance(order, list) else order
    results = []
    for o in orders:
        if o == 'inf':
            o = np.inf
        if o in ['cos', 'cosine']:
            # 1 - cos similarity
            val = 1 - torch.dot(w1, w2) / (torch.norm(w1) * torch.norm(w2))
            results.append(val.cpu().item())
        else:
            if o != np.inf:
                try:
                    o = int(o)
                except:
                    raise TypeError("Unsupported distance metric.")
            val = torch.norm(w1 - w2, p=o)
            results.append(val.cpu().item())
    return results


def load_dataset(dataset, train, download=True, augment=True):
    # Load dataset with optional augmentation
    try:
        dataset_class = getattr(torchvision.datasets, dataset)
    except AttributeError:
        raise NotImplementedError(f"Dataset {dataset} not implemented in torchvision.")

    if dataset in ["MNIST", "FashionMNIST"]:
        transform_list = [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ]
    elif dataset == "CIFAR100":
        if train and augment:
            transform_list = [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
            ]
        else:
            transform_list = []
        transform_list += [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                std=(0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
            )
        ]
    else:  # CIFAR10 or others
        if train and augment and dataset == "CIFAR10":
            transform_list = [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(p=0.5),
            ]
        else:
            transform_list = []
        transform_list += [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                 std=[0.2023, 0.1994, 0.2010])
        ]

    transform = transforms.Compose(transform_list)
    data = dataset_class(root='./data', train=train, download=download, transform=transform)
    return data


def ks_test(reference, rvs):
    device = rvs.device
    with torch.no_grad():
        ecdf = torch.arange(1, rvs.shape[0] + 1, dtype=torch.float32, device=device) / rvs.shape[0]
        sorted_rvs, _ = torch.sort(rvs)
        cdf_vals = reference(sorted_rvs)
        ks_stat = torch.max(torch.abs(cdf_vals - ecdf)).item()
    return ks_stat


def check_weights_initialization(param, method):
    if method == 'default':
        # Kaiming uniform
        fan = nn.init._calculate_correct_fan(param, 'fan_in')
        gain = nn.init.calculate_gain('leaky_relu', np.sqrt(5))
        std = gain / np.sqrt(fan)
        bound = np.sqrt(3.0) * std
        reference = torch.distributions.Uniform(-bound, bound).cdf
    elif method == 'resnet_cifar':
        # Kaiming normal (fan_in)
        fan = nn.init._calculate_correct_fan(param, 'fan_in')
        gain = nn.init.calculate_gain('relu', 0)
        std = gain / np.sqrt(fan)
        reference = torch.distributions.Normal(0, std).cdf
    elif method == 'resnet':
        # Kaiming normal (fan_out)
        fan = nn.init._calculate_correct_fan(param, 'fan_out')
        gain = nn.init.calculate_gain('relu', 0)
        std = gain / np.sqrt(fan)
        reference = torch.distributions.Normal(0, std).cdf
    elif method == 'default_bias':
        weight, param = param
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
        bound = 1 / np.sqrt(fan_in)
        reference = torch.distributions.Uniform(-bound, bound).cdf
        param = param.reshape(-1)
        rvs = param.cpu()
        ks_stat = ks_test(reference, rvs)
        p_value = stats.kstwo.sf(ks_stat, rvs.shape[0])
        return p_value
    else:
        raise NotImplementedError("Initialization strategy not implemented.")

    param = param.reshape(-1)
    rvs = param.cpu()
    ks_stat = ks_test(reference, rvs)
    p_value = stats.kstwo.sf(ks_stat, rvs.shape[0])
    return p_value


def test_accuracy(test_loader, model, num_samples):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for data in test_loader:
            img, labels = data
            img, labels = img.to(device), labels.to(device)
            out = model(img)
            _, pred = torch.max(out.data, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
            if total >= num_samples:
                break
    accuracy = correct / total
    return accuracy