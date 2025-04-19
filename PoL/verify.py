#!/usr/bin/env python3
"""
verify.py – Proof‑of‑Learning & watermark verification

Supports
1. Baseline PoL (no watermark)
2. Feature‑based watermark
3. Parameter‑perturbation watermark (relative check)
4. Non‑intrusive watermark

Either full‑interval or top‑q verification.  Outputs:
* rich console / file logs
* CSV + JSON distance records
* optional TensorBoard scalar logs
"""

import argparse
import csv
import glob
import hashlib
import json
import logging
import os
import random
from contextlib import nullcontext

import numpy as np
import torch
import utils

import model as custom_model
from train import train
from watermark_utils import (
    verify_parameter_perturbation_watermark_relative,
    verify_non_intrusive_watermark,
    WatermarkModule,
)


# ───────────────────────── Logging & exports ───────────────────────── #
def _init_logging(out_dir: str | None, verbose: bool):
    level = logging.DEBUG if verbose else logging.INFO
    handlers = [logging.StreamHandler()]
    if out_dir:
        handlers.append(logging.FileHandler(os.path.join(out_dir, "verify.log")))
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
        handlers=handlers,
    )


def _dump(rows, out_dir, stem):
    if not rows: return
    csv_f, json_f = (os.path.join(out_dir, f"{stem}.{ext}") for ext in ("csv", "json"))
    with open(csv_f, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys());
        w.writeheader();
        w.writerows(rows)
    with open(json_f, "w") as f: json.dump(rows, f, indent=2)
    logging.info(f"[export] metrics → {csv_f}")


# ───────────── Feature‑based watermark (lightweight check) ─────────── #
def _dummy_inputs(dev="cpu"): return torch.randn(100, 3, 32, 32, device=dev)


def _extract(model, x, layer="layer1"):
    buf = [];
    h = dict(model.named_modules())[layer].register_forward_hook(lambda _, __, o: buf.append(o))
    with torch.no_grad(): model(x); h.remove(); return buf[0]


def validate_feature_watermark(model, device="cpu"):
    model.to(device).eval();
    score = _extract(model, _dummy_inputs(device)).mean().item()
    ok = score > 0.01
    logging.info(f"Feature‑WM check: mean={score:.4g} → {'✓' if ok else '✗'}")
    return 1.0 if ok else 0.0


# ───────────────────── Integrity & initialisation ───────────────────── #
def verify_hash(model_dir, dataset):
    seq = np.load(os.path.join(model_dir, "indices.npy"))
    data = getattr(utils.load_dataset(dataset, train=True, augment=False), "data")
    h = hashlib.sha256();
    h.update(data[seq].tobytes());
    rec = h.hexdigest()
    with open(os.path.join(model_dir, "hash.txt")) as f: saved = f.read().strip()
    logging.info("Hash check " + ("✓" if rec == saved else "✗"))


def verify_initialization(model_dir, arch, thr=0.01):
    st = torch.load(os.path.join(model_dir, "model_step_0"), map_location="cpu")
    net = WatermarkModule(arch(), "k", 128) if any(k.startswith("original_model.") for k in st["net"]) else arch()
    net.load_state_dict(st["net"]);
    ks = utils.check_weights_initialization(next(net.parameters()), "resnet")
    logging.info(f"Init KS‑stat={ks:.4g} ({'✓' if ks >= thr else '⚠️'})")


# ─────────────────────── Distance helper ─────────────────────── #
def _dist(a, b, order, arch): return utils.parameter_distance(a, b, order, architecture=arch, half=0)


# ─────────────────── Full verification across intervals ─────────────────── #
def verify_all(*, model_dir, lr, batch_size, dataset, arch, order, threshold,
               k, randomize, watermark_key, watermark_method, num_parameters,
               perturbation_strength, watermark_size, augment=False, writer=None):
    ck = sorted(int(f.split("_")[-1]) for f in glob.glob(f"{model_dir}/model_step_*"))
    seq = np.load(os.path.join(model_dir, "indices.npy"))
    order = order if isinstance(order, list) else [order];
    threshold = threshold if isinstance(threshold, list) else [threshold]
    rows = []
    for idx, (cur, nxt) in enumerate(zip(ck[:-1], ck[1:])):
        cur_p, fut_p = f"{model_dir}/model_step_{cur}", f"{model_dir}/model_step_{nxt}"
        s, e = cur * batch_size, min(nxt * batch_size, len(seq))
        net, *_ = train(lr=lr, batch_size=batch_size, epochs=1, dataset=dataset, augment=augment,
                           architecture=arch, model_dir=cur_p, sequence=seq[s:e],
                           half=0, lambda_wm=0., k=k, randomize=randomize,
                           watermark_key=watermark_key, watermark_method=watermark_method,
                           num_parameters=num_parameters, perturbation_strength=perturbation_strength,
                           watermark_size=watermark_size, log_tb=False)
        d = _dist(fut_p, net, order, arch)
        row = {"interval": f"{cur}->{nxt}", **{str(o): v for o, v in zip(order, d)}};
        rows.append(row)
        logging.debug(f"[interval] {row}")
        if writer:
            for o, v in zip(order, d): writer.add_scalar(f"dist_{o}", v, idx)
    arr = np.array([[r[str(o)] for r in rows] for o in order])
    for j, o in enumerate(order):
        logging.info(f"{o}: avg={arr[j].mean():.3g}  max={arr[j].max():.3g}  "
                     f"viol>{threshold[j]}: {np.sum(arr[j] > threshold[j])}/{arr.shape[1]}")
    return rows


# ─────────────────────────── Top‑q verification ─────────────────────────── #
def verify_topq(*, model_dir, lr, batch_size, dataset, arch, epochs, q, order,
                k, randomize, watermark_key, watermark_method, num_parameters,
                perturbation_strength, watermark_size, augment=False, writer=None):
    ck = sorted(int(f.split("_")[-1]) for f in glob.glob(f"{model_dir}/model_step_*"))
    seq = np.load(os.path.join(model_dir, "indices.npy"));
    per_ep = len(ck) // epochs
    order = order if isinstance(order, list) else [order];
    rows = []
    for ep in range(epochs):
        st, en = ep * per_ep, (ep + 1) * per_ep if ep < epochs - 1 else len(ck)
        local = [_dist(f"{model_dir}/model_step_{ck[i]}",
                       f"{model_dir}/model_step_{ck[i + 1]}", order, arch)[0]
                 for i in range(st, en - 1)]
        for ind in np.argsort(local)[-q:]:
            c, n = ck[st + ind], ck[st + ind + 1]
            net, *_ = train(lr=lr, batch_size=batch_size, epochs=1, dataset=dataset, augment=augment,
                               architecture=arch, model_dir=f"{model_dir}/model_step_{c}",
                               sequence=seq[c * batch_size:n * batch_size], half=0, lambda_wm=0.,
                               k=k, randomize=randomize, watermark_key=watermark_key,
                               watermark_method=watermark_method, num_parameters=num_parameters,
                               perturbation_strength=perturbation_strength, watermark_size=watermark_size,
                               log_tb=False)
            d = _dist(f"{model_dir}/model_step_{n}", net, order, arch)[0]
            row = {"interval": f"{c}->{n}", str(order[0]): d};
            rows.append(row)
            logging.debug(f"[top‑q] {row}")
            if writer: writer.add_scalar(f"topq_dist_{order[0]}", d, len(rows))
    return rows


# ─────────────────────────────── CLI ──────────────────────────────── #
pa = argparse.ArgumentParser("PoL & Watermark verification")
# core
pa.add_argument("--batch-size", type=int, default=128)
pa.add_argument("--lr", type=float, default=0.1)
pa.add_argument("--epochs", type=int, default=2)
pa.add_argument("--dataset", default="CIFAR10")
pa.add_argument("--model", default="resnet20")
pa.add_argument("--model-dir", default="proof/CIFAR10_Run")
pa.add_argument("--save-freq", type=int, default=100)
pa.add_argument("--dist", nargs="+", default=["1", "2", "inf", "cos"])
pa.add_argument("--delta", nargs="+", type=float, default=[10000, 100, 1, 0.1])
pa.add_argument("--q", type=int, default=0)
# misc
pa.add_argument("--watermark-path", default="model_with_watermark.pth")
pa.add_argument("--log-tb", action="store_true")
pa.add_argument("--verbose", action="store_true")
pa.add_argument("--augment", action="store_true", help="If set, replay training with CIFAR-style random crop+flip")

args = pa.parse_args()

out = args.model_dir;
_init_logging(out, args.verbose)
arch = getattr(custom_model, args.model)

with open(os.path.join(out, "watermark_info.json")) as f: wm = json.load(f)
for k, v in wm.items(): setattr(args, k.replace("-", "_"), v)

seed = wm.get("seed", 777);
torch.manual_seed(seed);
torch.cuda.manual_seed_all(seed)
np.random.seed(seed);
random.seed(seed)

verify_initialization(out, arch);
verify_hash(out, args.dataset)

writer = nullcontext()
if args.log_tb:
    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter(log_dir=f"{out}/tb_verify")

with writer as tb:
    if args.q > 0:
        rec = verify_topq(model_dir=out, lr=args.lr, batch_size=args.batch_size, dataset=args.dataset,
                          arch=arch, epochs=args.epochs, q=args.q, order=args.dist, k=args.k,
                          randomize=args.randomize, watermark_key=args.watermark_key,
                          watermark_method=args.watermark_method, num_parameters=args.num_parameters,
                          perturbation_strength=args.perturbation_strength,
                          watermark_size=args.watermark_size, writer=tb, augment=args.augment)
        _dump(rec, out, "verify_topq_metrics")
    else:
        rec = verify_all(model_dir=out, lr=args.lr, batch_size=args.batch_size, dataset=args.dataset,
                         arch=arch, order=args.dist, threshold=args.delta, k=args.k,
                         randomize=args.randomize, watermark_key=args.watermark_key,
                         watermark_method=args.watermark_method, num_parameters=args.num_parameters,
                         perturbation_strength=args.perturbation_strength,
                         watermark_size=args.watermark_size, writer=tb, augment=args.augment)
        _dump(rec, out, "verify_full_metrics")

# ─────────────── final watermark presence test ──────────────── #
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pth = args.watermark_path;
method = wm.get("watermark_method", "none")
if method == "non_intrusive":
    net = WatermarkModule(arch(), args.watermark_key, watermark_size=wm["watermark_size"])
    net.load_state_dict(torch.load(pth, map_location=dev)["net"])
    ok = verify_non_intrusive_watermark(net, dev, args.watermark_key, wm["watermark_size"], 1e-3)
elif method == "feature_based":
    net = arch();
    net.load_state_dict(torch.load(pth, map_location=dev)["net"]);
    ok = validate_feature_watermark(net, dev) == 1.0
elif method == "parameter_perturbation":
    st = torch.load(pth, map_location=dev);
    net = arch();
    net.load_state_dict(st["net"])
    ok = verify_parameter_perturbation_watermark_relative(net, st.get("original_param_values"),
                                                          args.watermark_key, wm["perturbation_strength"], 1e-1)
else:
    ok = True
logging.info("Watermark check " + ("✓ passed" if ok else "✗ FAILED"))
