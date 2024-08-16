import argparse
import os
import hashlib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from collections import OrderedDict
import time
import utils
import logging
import model as custom_model

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


from watermark_train import prepare_watermark_data, \
    validate_watermark  # Make sure this is available and correctly implemented


def train(lr, batch_size, epochs, dataset, architecture, exp_id=None, sequence=None,
          model_dir=None, save_freq=None, num_gpu=torch.cuda.device_count(), verify=False, dec_lr=None,
          half=False, resume=False):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    if sequence is not None or model_dir is not None:
        resume = False

    try:
        trainset = utils.load_dataset(dataset, True)
    except:
        trainset = utils.load_dataset(dataset, True, download=True)

    if num_gpu > 1:
        net = nn.DataParallel(architecture())
        batch_size = batch_size * num_gpu
    else:
        net = architecture()
    num_batch = trainset.__len__() / batch_size
    net.to(device)
    if dataset == 'MNIST':
        optimizer = optim.SGD(net.parameters(), lr=lr)
        scheduler = None
    elif dataset == 'CIFAR10':
        if dec_lr is None: 
            dec_lr = [100, 150]
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                   milestones=[round(i * num_batch) for i in dec_lr],
                                                   gamma=0.1)
    elif dataset == 'CIFAR100':
        if dec_lr is None:
            dec_lr = [60, 120, 160]
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                   milestones=[round(i * num_batch) for i in dec_lr],
                                                   gamma=0.2)
    else:
        optimizer = optim.Adam(net.parameters(), lr=lr)
        scheduler = None

    # Log dataset size
    logging.info(f"Loaded dataset '{dataset}' with {len(trainset)} samples.")

    criterion = torch.nn.CrossEntropyLoss().to(device)

    if model_dir is not None:
        # load a pre-trained model from model_dir if it is given
        state = torch.load(model_dir)
        new_state_dict = OrderedDict()
        try:
            # in case the checkpoint is from a parallelized model
            for k, v in state['net'].items():
                name = "module." + k
                new_state_dict[name] = v
            net.load_state_dict(new_state_dict)
        except:
            net.load_state_dict(state['net'])
        optimizer.load_state_dict(state['optimizer'])
        if scheduler is not None:
            try:
                scheduler.load_state_dict(state['scheduler'])
            except:
                scheduler = None

        if half:
            net.half().float()

    if sequence is None:
        # if a training sequence is not given, create a new one
        train_size = trainset.__len__()
        sequence = utils.create_sequences(batch_size, train_size, epochs)

    ind = None
    if save_freq is not None and save_freq > 0:
        # save the sequence of data indices if save_freq is not none
        save_dir = os.path.join("proof", f"{dataset}_{exp_id}")
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        else:
            if resume:
                try:
                    ind = -1
                    # find the most recent checkpoint
                    while os.path.exists(os.path.join(save_dir, f"model_step_{ind + 1}")):
                        ind = ind + 1
                    if ind >= 0:
                        model_dir = os.path.join(save_dir, f"model_step_{ind}")
                        state = torch.load(model_dir)
                        new_state_dict = OrderedDict()
                        try:
                            for k, v in state['net'].items():
                                name = "module." + k
                                new_state_dict[name] = v
                            net.load_state_dict(new_state_dict)
                        except:
                            net.load_state_dict(state['net'])
                        optimizer.load_state_dict(state['optimizer'])
                        if scheduler is not None:
                            try:
                                scheduler.load_state_dict(state['scheduler'])
                            except:
                                scheduler = None
                        sequence = np.load(os.path.join(save_dir, "indices.npy"))
                        sequence = sequence[ind:]
                        logging.info('resume training')
                except:
                    logging.error('resume failed')
                    pass
                if ind == -1:
                    ind = None

        np.save(os.path.join(save_dir, "indices.npy"), sequence)

    num_step = sequence.shape[0]

    sequence = np.reshape(sequence, -1)
    subset = torch.utils.data.Subset(trainset, sequence)
    trainloader = torch.utils.data.DataLoader(subset, batch_size=batch_size, num_workers=0, pin_memory=True)

    # Prepare watermark data
    watermark_loader = prepare_watermark_data()
    watermark_iter = iter(watermark_loader)

    # Log model and optimizer details
    logging.info(f"Model architecture: {architecture.__name__}")
    logging.info(f"Optimizer: {'SGD' if dataset in ['CIFAR10', 'CIFAR100'] else 'Adam'}")

    net.train()

    if save_freq is not None and save_freq > 0:
        m = hashlib.sha256()
        for d in subset.dataset.data:
            m.update(d.__str__().encode('utf-8'))
        f = open(os.path.join(save_dir, "hash.txt"), "x")
        f.write(m.hexdigest())
        f.close()

    for i, (inputs, labels) in enumerate(trainloader):
        inputs, labels = inputs.to(device), labels.to(device)

        # Check if it's time to save a checkpoint
        if save_freq is not None and i % save_freq == 0:

            checkpoint_state = {
                'net': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict() if scheduler is not None else None,
            }
            checkpoint_path = os.path.join(save_dir, f"model_step_{i + (ind if ind is not None else 0)}")
            torch.save(checkpoint_state, checkpoint_path)

        # Integrate watermarking directly into each batch
        try:
            wm_inputs, wm_labels = next(watermark_iter)
        except StopIteration:
            # Reset the iterator if all watermark samples have been used
            watermark_iter = iter(watermark_loader)
            wm_inputs, wm_labels = next(watermark_iter)

        wm_inputs, wm_labels = wm_inputs.to(device), wm_labels.to(device)

        # Combine training data with watermark data
        combined_inputs = torch.cat([inputs, wm_inputs], dim=0)
        combined_labels = torch.cat([labels, wm_labels], dim=0)

        optimizer.zero_grad()
        outputs = net(combined_inputs)
        loss = criterion(outputs, combined_labels)
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        # Optional verification/validation at specific intervals
        if verify and i > 0 and i % round(num_batch) == 0:
            logging.info(f'Verifying at step {i}')
            validate(dataset, net, batch_size)
            net.train()

    if save_freq is not None and save_freq > 0:
        # for a model with n training steps, n+1 checkpoints will be saved
        state = {'net': net.state_dict(),
                 'optimizer': optimizer.state_dict()}
        if scheduler is not None:
            state['scheduler'] = scheduler.state_dict()
        torch.save(state, os.path.join(save_dir, f"model_step_{num_step}"))

    return net, optimizer, criterion


def validate(dataset, model, batch_size=128):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    testset = utils.load_dataset(dataset, False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2, pin_memory=True)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    logging.info(f'Accuracy: {100 * correct / total} %')
    return correct / total


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--dataset', type=str, default="CIFAR10")
    parser.add_argument('--model', type=str, default="resnet20",
                        help="models defined in model.py or any torchvision model.\n"
                             "Recommendation for CIFAR-10: resnet20/32/44/56/110/1202\n"
                             "Recommendation for CIFAR-100: resnet18/34/50/101/152"
                        )
    parser.add_argument('--id', help='experiment id', type=str, default='Batch100')
    parser.add_argument('--save-freq', type=int, default=100, help='frequence of saving checkpoints')
    parser.add_argument('--num-gpu', type=int, default=torch.cuda.device_count())
    parser.add_argument('--milestone', nargs='+', type=int, default=[1000, 1500])
    parser.add_argument('--verify', type=int, default=1)
    arg = parser.parse_args()
    seed = 777
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    t1 = time.time()
    logging.info(f'trying to allocate {arg.num_gpu} gpus')
    dve = 'cuda:2'
    # os.environ['CUDA_VISIBLE_DEVICES'] = "2"
    architecture = eval(f"custom_model.{arg.model}")
    # try:
    #     architecture = eval(f"custom_model.{arg.model}")
    # except:
    #     architecture = eval(f"torchvision.models.{arg.model}")
    trained_model, optimizer, criterion = train(arg.lr, arg.batch_size, arg.epochs, arg.dataset, architecture, exp_id=arg.id,
                          save_freq=arg.save_freq, num_gpu=arg.num_gpu, dec_lr=arg.milestone,
                          verify=arg.verify, resume=False)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    watermark_loader = prepare_watermark_data()
    watermark_accuracy = validate_watermark(trained_model, watermark_loader, device)
    # Logic to decide if the watermark learning is satisfactory
    validate(arg.dataset, trained_model)
    # Save the model with the embedded watermark
    model_path_with_watermark = 'model_with_watermark.pth'
    torch.save(trained_model.state_dict(), model_path_with_watermark)
    logging.info("Model with watermark saved at {model_path_with_watermark}")
    t2 = time.time()
    logging.info(f"Total time: {t2 - t1}")
