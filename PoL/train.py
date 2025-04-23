#!/usr/bin/env python3
# train.py
#
# Integrated training script for:
#   1) Baseline PoL (no watermark),
#   2) Feature‑Based Watermark,
#   3) Parameter‑Perturbation Watermark,
#   4) Non‑Intrusive Watermark (Option B),
# plus: epoch‑level validation, metric logging, optional TensorBoard.

import argparse
import csv
import hashlib
import json
import logging
import os
import random
import time
from contextlib import nullcontext
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import utils  # local utils.py
from watermark_utils import (  # local watermark_utils.py
    generate_watermark_pattern,
    apply_parameter_perturbations,
    should_embed_watermark,
    WatermarkModule,
    embed_feature_watermark,
    generate_watermark_target,
    verify_non_intrusive_watermark,
    run_feature_based_watermark_verification,
)

# --------------------------------------------------------------------------- #
#                                LOGGING SETUP                                #
# --------------------------------------------------------------------------- #

def _init_logging(save_dir: str | None, verbose: bool = False, save_checkpoints: bool = True):
    """
    Configure logging: console + (optional) train.log file.
    Only create the directory if save_checkpoints is True.
    """
    handlers = [logging.StreamHandler()]
    if save_dir is not None and save_checkpoints:
        os.makedirs(save_dir, exist_ok=True)
        handlers.append(logging.FileHandler(os.path.join(save_dir, "train.log")))
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
        handlers=handlers,
    )

# --------------------------------------------------------------------------- #
#                          WEIGHT INITIALISATION                              #
# --------------------------------------------------------------------------- #

def _weights_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.kaiming_normal_(m.weight)

# --------------------------------------------------------------------------- #
#                                  TRAINING                                   #
# --------------------------------------------------------------------------- #

def train(
        lr: float,
        batch_size: int,
        epochs: int,
        dataset: str,
        architecture,
        exp_id: str | None = None,
        model_dir: str | None = None,
        save_freq: int | None = None,
        sequence=None,
        num_gpu: int = torch.cuda.device_count(),
        verify: bool = False,
        dec_lr: list[int] | None = None,
        half: bool = False,
        resume: bool = False,
        lambda_wm: float = 0.01,
        k: int = 100,
        randomize: bool = False,
        watermark_key: str = "secret_key",
        watermark_method: str = "feature_based",
        num_parameters: int = 1000,
        perturbation_strength: float = 1e-5,
        watermark_size: int = 128,
        subset_size: int | None = None,
        log_tb: bool = False,
        log_dir: str | None = None,
        scheduler_type: str = "step",
        verbose: bool = False,
        save_checkpoints: bool = True,  # Controls checkpoint saving and directory creation
):
    """
    Train a model and (optionally) embed a watermark.
    Returns: (net, optimizer, criterion, original_param_values)
    """
    # ----------------------------------------------------------------------- #
    #  0.  Device                                                             #
    # ----------------------------------------------------------------------- #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Use model_dir if provided, otherwise fallback to default
    if model_dir is None:
        model_dir = os.path.join("proof", f"{dataset}_{exp_id or datetime.now().strftime('%Y%m%d_%H%M%S')}")

    _init_logging(model_dir, verbose=verbose, save_checkpoints=save_checkpoints)
    logging.info(f"Using device: {device}")
    logging.info("Data augmentation is disabled to ensure reproducibility for Proof-of-Learning.")

    # ----------------------------------------------------------------------- #
    #  1.  Dataset & sequence                                                 #
    # ----------------------------------------------------------------------- #
    logging.info("=== Loading Dataset ===")
    trainset = utils.load_dataset(dataset, train=True, augment=False)  # Explicitly disable augmentation
    full_train_size = len(trainset)
    logging.info(f"Dataset '{dataset}' loaded with {full_train_size} samples.")

    seed = 777  # Define seed early for sequence generation
    if sequence is None:
        subset_size = min(subset_size or full_train_size, full_train_size)
        # Use utils.create_sequences for modularity and deterministic sequence generation
        sequence = utils.create_sequences(batch_size, full_train_size, epochs=1, seed=seed).flatten()[:subset_size]
        logging.info(f"No sequence provided → training on subset_size={subset_size}.")
    else:
        logging.info(f"Using provided sequence (length={len(sequence)}).")

    sequence = np.asarray(sequence).reshape(-1)
    if batch_size <= 0:
        raise ValueError("Batch size must be positive.")

    # GPU fan‑out
    max_gpus = torch.cuda.device_count()
    if num_gpu > max_gpus:
        logging.warning(f"Requested {num_gpu} GPU(s) but only {max_gpus} available → using {max_gpus}.")
        num_gpu = max_gpus
    if num_gpu > 1:
        batch_size *= num_gpu
        logging.info(f"Effective batch_size with DataParallel: {batch_size}")

    # ----------------------------------------------------------------------- #
    #  2.  Model                                                              #
    # ----------------------------------------------------------------------- #
    net = architecture()
    net.apply(_weights_init)
    # ---------- NEW (baseline snapshot) -----------------------------------
    original_param_values = {}
    if watermark_method == "parameter_perturbation":
        for n, p in net.named_parameters():
            if p.requires_grad and not any(tag in n.lower()
                                           for tag in ("bn", "running_mean", "running_var")):
                original_param_values[n] = p.detach().cpu().clone().numpy()
    # ----------------------------------------------------------------------
    if watermark_method == "non_intrusive":
        net = WatermarkModule(net, watermark_key, watermark_size=watermark_size)
    net.to(device)
    logging.info(f"Model: {architecture.__name__}")

    # ----------------------------------------------------------------------- #
    #  3.  Optimiser & LR scheduler                                           #
    # ----------------------------------------------------------------------- #
    if dataset == "MNIST":
        optimizer = optim.SGD(net.parameters(), lr=lr)
        scheduler = None
    elif dataset in {"CIFAR10", "CIFAR100"}:
        if dec_lr is None:
            dec_lr = [epochs // 2, int(epochs * 0.75)]
        wd = 1e-4 if dataset == "CIFAR10" else 5e-4
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
        if scheduler_type == "step":
            gamma = 0.1 if dataset == "CIFAR10" else 0.2
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=dec_lr, gamma=gamma)
        elif scheduler_type == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        else:
            scheduler = None
    else:
        optimizer = optim.Adam(net.parameters(), lr=lr)
        scheduler = None

    criterion = nn.CrossEntropyLoss().to(device)
    logging.info(f"Optimizer: {optimizer.__class__.__name__}, base LR={lr}")

    # ----------------------------------------------------------------------- #
    #  4.  Optionally resume                                                  #
    # ----------------------------------------------------------------------- #
    if resume and os.path.exists(os.path.join(model_dir, "model_step_0")):
        state = torch.load(os.path.join(model_dir, "model_step_0"), map_location=device)
        net.load_state_dict(state["net"])
        optimizer.load_state_dict(state["optimizer"])
        if scheduler and state.get("scheduler"):
            scheduler.load_state_dict(state["scheduler"])
        logging.info(f"Resumed from {model_dir}")

    if half:
        net.half().float()

    # ----------------------------------------------------------------------- #
    #  5.  Reproducibility                                                    #
    # ----------------------------------------------------------------------- #
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # ----------------------------------------------------------------------- #
    #  6.  PoL artefacts / checkpoint dir                                     #
    # ----------------------------------------------------------------------- #
    if save_checkpoints:
        os.makedirs(model_dir, exist_ok=True)

        # dataset hash
        m = hashlib.sha256()
        data_array = getattr(trainset, "data", getattr(trainset, "train_data", None))
        if data_array is None:
            raise AttributeError("Dataset object has neither '.data' nor '.train_data'.")
        m.update(data_array[sequence].tobytes())
        with open(os.path.join(model_dir, "hash.txt"), "w") as f:
            f.write(m.hexdigest())
        np.save(os.path.join(model_dir, "indices.npy"), sequence)

        wm_info = dict(
            watermark_key=watermark_key,
            lambda_wm=lambda_wm,
            k=k,
            randomize=randomize,
            seed=seed,
            watermark_method=watermark_method,
            num_parameters=num_parameters,
            perturbation_strength=perturbation_strength,
            watermark_size=watermark_size,
            subset_size=subset_size,
        )
        with open(os.path.join(model_dir, "watermark_info.json"), "w") as f:
            json.dump(wm_info, f, indent=2)

    # ----------------------------------------------------------------------- #
    #  7.  DataLoader                                                         #
    # ----------------------------------------------------------------------- #
    pin_memory = True if torch.cuda.is_available() else False
    trainloader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(trainset, sequence),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # Set to 0 for strict determinism
        pin_memory=pin_memory,
    )

    # ----------------------------------------------------------------------- #
    #  8.  Optional TensorBoard                                               #
    # ----------------------------------------------------------------------- #
    writer_ctx = nullcontext()
    if log_tb:
        from torch.utils.tensorboard import SummaryWriter
        tb_log_dir = log_dir if log_dir else (model_dir or "./runs")
        writer_ctx = SummaryWriter(log_dir=tb_log_dir)

    # ----------------------------------------------------------------------- #
    #  9.  Metric containers                                                  #
    # ----------------------------------------------------------------------- #
    metrics = []  # list of dicts (one per epoch)

    def _dump_metrics():
        if not save_checkpoints:
            return
        csv_path = os.path.join(model_dir, "metrics.csv")
        json_path = os.path.join(model_dir, "metrics.json")

        # CSV
        fieldnames = metrics[0].keys()
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(metrics)

        # JSON
        with open(json_path, "w") as f:
            json.dump(metrics, f, indent=2)

    # ----------------------------------------------------------------------- #
    # 10.  Save initial checkpoint                                            #
    # ----------------------------------------------------------------------- #
    current_step = 0

    def _save_checkpoint(step: int):
        if not save_checkpoints or save_freq is None or step % save_freq:
            return
        ckpt = dict(
            net=net.state_dict(),
            optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict() if scheduler else None,
            step=step,
            original_param_values=original_param_values,
        )
        torch.save(ckpt, os.path.join(model_dir, f"model_step_{step}"))
        logging.info(f"Checkpoint saved @step {step}")

    _save_checkpoint(0)

    # ----------------------------------------------------------------------- #
    # 11.  Main loop                                                          #
    # ----------------------------------------------------------------------- #
    logging.info("=== Training ===")
    with writer_ctx as tb_writer:
        for epoch in range(epochs):
            net.train()
            running_loss = 0.0

            for batch_idx, (inputs, labels) in enumerate(trainloader):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                # --------------------- forward & watermark logic ------------- #
                if watermark_method == "none":
                    outputs = net(inputs) if not isinstance(net, WatermarkModule) else net(inputs, trigger=False)
                    loss = criterion(outputs, labels)

                elif watermark_method == "feature_based":
                    feats_list = []

                    def _hook(_, __, out):
                        feats_list.append(out)

                    handle = (net.module if isinstance(net, nn.DataParallel) else net).layer1.register_forward_hook(
                        _hook)
                    outputs = net(inputs)
                    handle.remove()

                    loss = criterion(outputs, labels)
                    if lambda_wm > 0 and should_embed_watermark(current_step, k, watermark_key, randomize=randomize, device=device):
                        feats = feats_list[0]
                        desired_feats, mask = embed_feature_watermark(feats, watermark_key, current_step)
                        wm_loss = nn.MSELoss()(feats * mask, desired_feats * mask)
                        loss += lambda_wm * wm_loss

                elif watermark_method == "parameter_perturbation":
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)

                elif watermark_method == "non_intrusive":
                    outputs = net(inputs, trigger=False)
                    loss = criterion(outputs, labels)
                    if lambda_wm > 0 and should_embed_watermark(current_step, k, watermark_key, randomize=randomize, device=device):
                        target = generate_watermark_target(inputs, watermark_key, watermark_size).to(device)
                        wm_out = net(inputs, trigger=True)
                        wm_loss = nn.MSELoss()(wm_out, target)
                        loss += lambda_wm * wm_loss

                else:
                    raise ValueError(f"Unknown watermark_method {watermark_method}")

                # --------------------------- backward ------------------------ #
                loss.backward()
                optimizer.step()

                # ── parameter-perturbation watermark (run **once**, just after step 0) ──
                if (
                    watermark_method == "parameter_perturbation"
                    and lambda_wm > 0
                    and current_step == 0                # run exactly once
                    and should_embed_watermark(0, k, watermark_key, randomize, device)
                ):
                    # a) select **all** trainable, non-BN tensors
                    sel_params = [
                        (n, p) for n, p in net.named_parameters()
                        if p.requires_grad and not any(
                            tag in n.lower()
                            for tag in ("bn", "running_mean", "running_var")
                        )
                    ]

                    # b) remember their pre-jump values
                    for n, p in sel_params:
                        original_param_values[n] = (
                            p.detach().cpu().clone().numpy()
                        )

                    # c) add ±strength offset and freeze them
                    pat = generate_watermark_pattern(watermark_key, len(sel_params))
                    apply_parameter_perturbations(
                        sel_params, pat, perturbation_strength
                    )
                    for _, p in sel_params:
                        p.requires_grad_(False)

                # logging
                running_loss += loss.item()
                current_step += 1

                _save_checkpoint(current_step)

            # ---------------- end epoch: validation ------------------------- #
            train_loss = running_loss / len(trainloader)
            val_loss, val_acc = validate(dataset, net, criterion)

            # scheduler step
            if scheduler:
                scheduler.step()

            # record metrics
            lr_cur = scheduler.get_last_lr()[0] if scheduler else lr
            metrics.append(dict(
                epoch=epoch + 1,
                train_loss=train_loss,
                val_loss=val_loss,
                val_acc=val_acc,
                lr=lr_cur,
            ))
            _dump_metrics()

            # TensorBoard
            if log_tb:
                tb_writer.add_scalar("Loss/train", train_loss, epoch + 1)
                tb_writer.add_scalar("Loss/val", val_loss, epoch + 1)
                tb_writer.add_scalar("Acc/val", val_acc, epoch + 1)
                tb_writer.add_scalar("LR", lr_cur, epoch + 1)

            logging.info(
                f"[Epoch {epoch + 1:3d}/{epochs}] "
                f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
                f"val_acc={val_acc * 100:5.2f}% | lr={lr_cur:.4f}"
            )

    # ----------------------------------------------------------------------- #
    # 12.  Final checkpoint                                                   #
    # ----------------------------------------------------------------------- #
    _save_checkpoint(current_step)

    logging.info("=== Training Completed ===")
    return net, optimizer, criterion, original_param_values

# --------------------------------------------------------------------------- #
#                               VALIDATION                                    #
# --------------------------------------------------------------------------- #

def validate(dataset, model, criterion, batch_size: int = 128):
    """
    Returns (val_loss, val_accuracy).
    """
    from watermark_utils import WatermarkModule
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    testset = utils.load_dataset(dataset, train=False, augment=False)  # Explicitly disable augmentation
    pin_memory = True if torch.cuda.is_available() else False
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin_memory  # Set num_workers to 0
    )

    model.to(device)
    model.eval()

    total, correct, running_loss = 0, 0, 0.0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            out = (model(inputs, trigger=False) if isinstance(model, WatermarkModule)
                   else model(inputs))
            loss = criterion(out, labels)
            running_loss += loss.item() * labels.size(0)

            preds = torch.argmax(out, dim=1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

    val_loss = running_loss / total
    val_acc = correct / total
    return val_loss, val_acc

# --------------------------------------------------------------------------- #
#                            ARGPARSE INTERFACE                               #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    parser = argparse.ArgumentParser("PoL Training with optional watermarking + metrics logging")
    # Basic
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--dataset", type=str, default="CIFAR10")
    parser.add_argument("--model", type=str, default="resnet20")
    parser.add_argument("--id", type=str, default="Run")
    parser.add_argument("--save-freq", type=int, default=100)
    parser.add_argument("--num-gpu", type=int, default=torch.cuda.device_count())
    parser.add_argument("--milestone", nargs="+", type=int, default=None)
    parser.add_argument("--verify", action="store_true")

    # Add scheduler argument
    parser.add_argument("--scheduler", type=str, default="step", choices=["none", "step", "cosine"],
                        help="Learning rate scheduler type")

    # Add model-dir and log-dir arguments
    parser.add_argument("--model-dir", type=str, default=None,
                        help="Directory to save model checkpoints and PoL artifacts")
    parser.add_argument("--log-dir", type=str, default=None,
                        help="Directory for TensorBoard logs (if --log-tb is set)")

    # Add verbose argument
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose logging for debugging")

    # Watermark
    parser.add_argument("--lambda-wm", type=float, default=0.3)
    parser.add_argument("--k", type=int, default=200)
    parser.add_argument("--randomize", action="store_true")
    parser.add_argument("--watermark-key", type=str, default="secret_key")
    parser.add_argument("--watermark-method",
                        choices=["none", "feature_based", "parameter_perturbation", "non_intrusive"],
                        default="non_intrusive")
    parser.add_argument("--num-parameters", type=int, default=1000)
    parser.add_argument("--perturbation-strength", type=float, default=1e-5)
    parser.add_argument("--watermark-size", type=int, default=128)

    # Subset / misc
    parser.add_argument("--subset-size", type=int, default=None)
    parser.add_argument("--tolerance-wm", type=float, default=1e-3)

    # TensorBoard flag
    parser.add_argument("--log-tb", action="store_true", help="Enable TensorBoard logging")

    args = parser.parse_args()

    # Re‑seed
    seed = 777
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    t0 = time.time()
    logging.info(f"Attempting to allocate {args.num_gpu} GPU(s).")

    # Resolve architecture
    try:
        import model as custom_models
        architecture = getattr(custom_models, args.model)
    except AttributeError:
        import torchvision.models as tv_models
        architecture = getattr(tv_models, args.model)

    # ----- TRAIN ----------------------------------------------------------- #
    trained_model, optimizer, criterion, original_param_values = train(
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        dataset=args.dataset,
        architecture=architecture,
        exp_id=args.id,
        model_dir=args.model_dir,
        save_freq=args.save_freq,
        num_gpu=args.num_gpu,
        dec_lr=args.milestone,
        verify=args.verify,
        lambda_wm=args.lambda_wm,
        k=args.k,
        randomize=args.randomize,
        watermark_key=args.watermark_key,
        watermark_method=args.watermark_method,
        num_parameters=args.num_parameters,
        perturbation_strength=args.perturbation_strength,
        watermark_size=args.watermark_size,
        subset_size=args.subset_size,
        log_tb=args.log_tb,
        log_dir=args.log_dir,
        scheduler_type=args.scheduler,
        verbose=args.verbose,
        save_checkpoints=True  # Default to True for training, False for verification
    )

    # ----- POST‑TRAINING WATERMARK CHECKS AND MODEL SAVING ----------------- #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.watermark_method == "none":
        logging.info("No watermark used; baseline PoL.")
        torch.save(trained_model.state_dict(), 'model_baseline_no_watermark.pth')
    elif args.watermark_method == "feature_based":
        run_feature_based_watermark_verification(
            model=trained_model,
            wm_key=args.watermark_key,
            device=device
        )
        torch.save(trained_model.state_dict(), 'model_with_feature_based_watermark.pth')
        logging.info("Saved model_with_feature_based_watermark.pth")
    elif args.watermark_method == "parameter_perturbation":
        from watermark_utils import verify_parameter_perturbation_watermark_relative

        verify_parameter_perturbation_watermark_relative(
            model=trained_model,
            original_params=original_param_values,
            wm_key=args.watermark_key,
            strength=args.perturbation_strength,
            tol=1e-1
        )
        final_ckpt = {
            'net': trained_model.state_dict(),
            'original_param_values': original_param_values
        }
        torch.save(final_ckpt, 'model_with_parameter_perturbation_watermark.pth')
        logging.info("Saved model_with_parameter_perturbation_watermark.pth")
    elif args.watermark_method == "non_intrusive":
        verify_non_intrusive_watermark(
            trained_model, device, args.watermark_key, args.watermark_size, args.tolerance_wm
        )
        torch.save(trained_model.state_dict(), 'model_with_non_intrusive_watermark.pth')
        logging.info("Saved model_with_non_intrusive_watermark.pth")

    # ----- FINAL VALIDATION ------------------------------------------------ #
    final_val_loss, final_val_acc = validate(args.dataset, trained_model, criterion)
    logging.info(f"Final Validation Acc: {final_val_acc * 100:.2f}% | Val Loss: {final_val_loss:.4f}")

    logging.info(f"Total wall‑clock time: {time.time() - t0:.1f}s")