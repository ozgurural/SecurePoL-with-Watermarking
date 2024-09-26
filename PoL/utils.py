import torch
import numpy as np
from scipy import stats
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


def get_parameters(net, numpy=False):
    # Get weights from a torch model as a single flattened tensor
    if type(net) is tuple:
        net = net[0]
    parameter = torch.cat([i.data.reshape([-1]) for i in list(net.parameters())])
    if numpy:
        return parameter.cpu().numpy()
    else:
        return parameter


def set_parameters(net, parameters, device):
    # Load weights from a list of numpy arrays to a torch model
    for i, (name, param) in enumerate(net.named_parameters()):
        param.data = torch.Tensor(parameters[i]).to(device)
    return net


def create_sequences(batch_size, dataset_size, epochs):
    # Create a sequence of data indices used for training
    sequence = np.concatenate([
        np.random.default_rng().choice(dataset_size, size=dataset_size, replace=False)
        for _ in range(epochs)
    ])
    num_batch = int(len(sequence) // batch_size)
    return np.reshape(sequence[:num_batch * batch_size], [num_batch, batch_size])


def consistent_type(model, architecture=None,
                    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'), half=False):
    # This function ensures that model weights are in a consistent format (torch.Tensor)
    if isinstance(model, str):
        assert architecture is not None
        state = torch.load(model)
        net = architecture()
        net.load_state_dict(state['net'])
        weights = get_parameters(net)
    elif isinstance(model, np.ndarray):
        weights = torch.tensor(model)
    elif not isinstance(model, torch.Tensor):
        weights = get_parameters(model)
    else:
        weights = model
    if half:
        weights = weights.half()
    return weights.to(device)


def parameter_distance(model1, model2, order=2, architecture=None, half=False):
    # Compute the difference between two checkpoints
    weights1 = consistent_type(model1, architecture, half=half)
    weights2 = consistent_type(model2, architecture, half=half)
    if not isinstance(order, list):
        orders = [order]
    else:
        orders = order
    res_list = []
    for o in orders:
        if o == 'inf':
            o = np.inf
        if o == 'cos' or o == 'cosine':
            res = (1 - torch.dot(weights1, weights2) /
                   (torch.norm(weights1) * torch.norm(weights2))).cpu().numpy()
        else:
            if o != np.inf:
                try:
                    o = int(o)
                except:
                    raise TypeError("Input metric for distance is not understandable")
            res = torch.norm(weights1 - weights2, p=o).cpu().numpy()
        if isinstance(res, np.ndarray):
            res = float(res)
        res_list.append(res)
    return res_list


def load_dataset(dataset, train, download=True, augment=True):
    # Load dataset with optional data augmentation
    try:
        dataset_class = getattr(torchvision.datasets, dataset)
    except AttributeError:
        raise NotImplementedError(f"Dataset {dataset} is not implemented by torchvision.")

    if dataset in ["MNIST", "FashionMNIST"]:
        transform_list = [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
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
    else:  # CIFAR10 or other datasets
        if train and augment:
            transform_list = [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(p=0.5),
            ]
        else:
            transform_list = []
        transform_list += [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2023, 0.1994, 0.2010]
            )
        ]

    transform = transforms.Compose(transform_list)
    data = dataset_class(root='./data', train=train, download=download, transform=transform)
    return data


def ks_test(reference, rvs):
    # Kolmogorov-Smirnov test using PyTorch
    device = rvs.device
    with torch.no_grad():
        ecdf = torch.arange(1, rvs.shape[0] + 1, dtype=torch.float32, device=device) / rvs.shape[0]
        sorted_rvs, _ = torch.sort(rvs)
        cdf_vals = reference(sorted_rvs)
        ks_stat = torch.max(torch.abs(cdf_vals - ecdf)).item()
    return ks_stat


def check_weights_initialization(param, method):
    # Check if the weights follow the specified initialization distribution
    if method == 'default':
        # Kaiming uniform (default for weights of nn.Conv and nn.Linear)
        fan = nn.init._calculate_correct_fan(param, 'fan_in')
        gain = nn.init.calculate_gain('leaky_relu', np.sqrt(5))
        std = gain / np.sqrt(fan)
        bound = np.sqrt(3.0) * std
        reference = torch.distributions.Uniform(-bound, bound).cdf
    elif method == 'resnet_cifar':
        # Kaiming normal
        fan = nn.init._calculate_correct_fan(param, 'fan_in')
        gain = nn.init.calculate_gain('relu', 0)
        std = gain / np.sqrt(fan)
        reference = torch.distributions.Normal(0, std).cdf
    elif method == 'resnet':
        # Kaiming normal (default in conv layers of PyTorch ResNet)
        fan = nn.init._calculate_correct_fan(param, 'fan_out')
        gain = nn.init.calculate_gain('relu', 0)
        std = gain / np.sqrt(fan)
        reference = torch.distributions.Normal(0, std).cdf
    elif method == 'default_bias':
        # Default for bias of nn.Conv and nn.Linear
        weight, param = param
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
        bound = 1 / np.sqrt(fan_in)
        reference = torch.distributions.Uniform(-bound, bound).cdf
    else:
        raise NotImplementedError("Input initialization strategy is not implemented.")

    param = param.reshape(-1)
    ks_stat = ks_test(reference, param.cpu())
    p_value = stats.kstwo.sf(ks_stat, param.shape[0])
    return p_value


def test_accuracy(test_loader, model, num_samples):
    # Compute the accuracy of the model on a subset of the test dataset
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
