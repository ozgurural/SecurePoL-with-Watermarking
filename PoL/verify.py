#!/usr/bin/env python3
"""
verify.py — strict Proof‑of‑Learning & watermark verifier
"""

from __future__ import annotations
import os, sys, argparse, csv, glob, hashlib, json, logging, random, textwrap
from pathlib import Path
from contextlib import nullcontext
from typing import List

import numpy as np, torch
import utils, model as custom_model
from train import train
from watermark_utils import (
    WatermarkModule,
    verify_non_intrusive_watermark,
    verify_parameter_perturbation_watermark_relative,
)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # silence cuDNN/TF chatter

# ───────────────────────── logging ──────────────────────────
def _init_logging(out_dir: Path | None, verbose: bool) -> None:
    lvl = logging.DEBUG if verbose else logging.INFO
    h: List[logging.Handler] = [logging.StreamHandler()]
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        h.append(logging.FileHandler(out_dir / "verify.log"))
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s  %(levelname)5s  %(message)s",
        handlers=h,
    )

# ─────────────────────── I/O helpers ────────────────────────
def _dump(rows, out_dir: Path, stem: str) -> None:
    if not rows:
        return
    csv_p, js_p = out_dir / f"{stem}.csv", out_dir / f"{stem}.json"
    with csv_p.open("w", newline="") as f:
        w = csv.DictWriter(f, rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    js_p.write_text(json.dumps(rows, indent=2))
    logging.info(f"[export] {csv_p.relative_to(Path.cwd())}")

# ───────── watermark sanity (feature‑based) ─────────
def _prepare_watermark_data(dev="cpu") -> torch.Tensor:
    return torch.randn(100, 3, 32, 32, device=dev)

def _extract_features(model, inputs, layer="layer1"):
    features = []
    h = dict(model.named_modules())[layer].register_forward_hook(lambda *_: features.append(_[2]))
    with torch.no_grad():
        model(inputs)
    h.remove()
    return features[0]

def _feature_wm_ok(net, dev="cpu") -> bool:
    net.to(dev).eval()
    inputs = _prepare_watermark_data(dev)
    feats = _extract_features(net, inputs)
    mean_val = feats.mean().item()
    ok = mean_val > 0.01
    logging.info(f"[feature_wm] mean={mean_val:.4f} → {'✓' if ok else '✗'}")
    return ok

# ───────── integrity checks ─────────
def _check_hash(d: Path, dataset: str) -> bool:
    idx, hfile = d / "indices.npy", d / "hash.txt"
    if not (idx.exists() and hfile.exists()):
        logging.error("[hash] indices.npy or hash.txt missing")
        return False
    seq = np.load(idx)
    data = getattr(utils.load_dataset(dataset, train=True, augment=False), "data")
    ok = hashlib.sha256(data[seq].tobytes()).hexdigest() == hfile.read_text().strip()
    logging.info(f"[hash] {'✓ match' if ok else '✗ MISMATCH'}")
    return ok

def _check_init(d: Path, arch) -> bool:
    ck = d / "model_step_0"
    if not ck.exists():
        logging.error("[init‑ks] model_step_0 missing")
        return False
    st = torch.load(ck, map_location="cpu")
    wrapped = any(k.startswith("original_model.") for k in st["net"])
    net = WatermarkModule(arch(), "k", 128) if wrapped else arch()
    net.load_state_dict(st["net"])
    ks = utils.check_weights_initialization(next(net.parameters()), "resnet")
    ok = ks >= 0.01
    logging.info(f"[init‑ks] KS={ks:.2e} → {'✓' if ok else '✗'}")
    return ok

# ───────── inner‑train silencer ─────────
def _silent_train(lvl, scheduler_type="step", **kw):  # NEW: Add scheduler_type parameter
    root = logging.getLogger()
    prev = root.level
    root.setLevel(lvl)
    try:
        net, *_ = train(scheduler_type=scheduler_type, **kw)  # NEW: Pass scheduler_type to train
    finally:
        root.setLevel(prev)
    return net

# ───────── distance helper ─────────
def _dist(a, b, order, arch):
    return utils.parameter_distance(a, b, order, architecture=arch, half=0)

# ───────── full‑chain verifier ─────────
def verify_all(*, model_dir: Path, arch, order, thr, cfg, writer=None) -> bool:
    ck = sorted(int(Path(p).stem.split("_")[-1]) for p in glob.glob(str(model_dir / "model_step_*")))
    if len(ck) < 2:
        logging.error("no checkpoints → PoL invalid")
        return False
    seq = np.load(model_dir / "indices.npy")
    rows = []
    ok = True
    for i, (c, n) in enumerate(zip(ck[:-1], ck[1:])):
        s, e = c * cfg["batch_size"], min(n * cfg["batch_size"], len(seq))
        if e <= s:
            continue
        net = _silent_train(
            cfg["log_lvl"],
            scheduler_type=cfg["scheduler_type"],  # NEW: Pass scheduler_type
            model_dir=str(model_dir / f"model_step_{c}"),
            sequence=seq[s:e],
            **cfg["train"]
        )
        d = _dist(model_dir / f"model_step_{n}", net, order, arch)
        rows.append({"interval": f"{c}->{n}", **{str(o): v for o, v in zip(order, d)}})
        if writer:
            for o, v in zip(order, d):
                writer.add_scalar(f"dist_{o}", v, i)
    if not rows:
        return False
    arr = np.array([[r[str(o)] for r in rows] for o in order])
    for j, o in enumerate(order):
        viol = int((arr[j] > thr[j]).sum())
        ok &= viol == 0
        logging.info(f"[{o}] avg={arr[j].mean():.3g} max={arr[j].max():.3g} viol={viol}")
    _dump(rows, model_dir, "verify_full_metrics")
    return ok

# ───────── top‑q verifier ─────────
def verify_topq(*, model_dir: Path, arch, order, q, epochs, cfg, writer=None, precompute=True) -> bool:
    ck = sorted(int(Path(p).stem.split("_")[-1]) for p in glob.glob(str(model_dir / "model_step_*")))
    if len(ck) < 2:
        logging.error("no checkpoints → PoL invalid")
        return False
    seq = np.load(model_dir / "indices.npy")
    per = max(1, len(ck) // epochs)
    rows = []
    for ep in range(epochs):
        st, en = ep * per, min((ep + 1) * per, len(ck) - 1)
        if precompute:
            local = [
                _dist(model_dir / f"model_step_{ck[i]}", model_dir / f"model_step_{ck[i+1]}", order, arch)[0]
                for i in range(st, en)
            ]
            top_indices = np.argsort(local)[-q:]
        else:
            top_indices = range(max(0, en - st - q), en - st)
        for ind in top_indices:
            c, n = ck[st + ind], ck[st + ind + 1]
            s, e = c * cfg["batch_size"], min(n * cfg["batch_size"], len(seq))
            net = _silent_train(
                cfg["log_lvl"],
                scheduler_type=cfg["scheduler_type"],  # NEW: Pass scheduler_type
                model_dir=str(model_dir / f"model_step_{c}"),
                sequence=seq[s:e],
                **cfg["train"]
            )
            d = _dist(model_dir / f"model_step_{n}", net, order, arch)[0]
            rows.append({"interval": f"{c}->{n}", str(order[0]): d})
            if writer:
                writer.add_scalar(f"topq_dist_{order[0]}", d, len(rows))
    _dump(rows, model_dir, "verify_topq_metrics")
    return bool(rows)

# ───────── CLI ─────────
p = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="Strict PoL & watermark verifier"
)
p.add_argument("--model-dir", required=True)
p.add_argument("--dataset", default="CIFAR10")
p.add_argument("--model", default="resnet20")
p.add_argument("--batch-size", type=int, default=128)
p.add_argument("--lr", type=float, default=0.1)
p.add_argument("--epochs", type=int, default=2)
p.add_argument("--dist", nargs="+", default=["1", "2", "inf", "cos"])
p.add_argument("--delta", nargs="+", type=float, default=[1e4, 100, 1, 0.1])
p.add_argument("--q", type=int, default=0)
p.add_argument("--watermark-path", default="model_with_watermark.pth")
# NEW: Add scheduler argument
p.add_argument("--scheduler", type=str, default="step", choices=["none", "step", "cosine"],
               help="Learning rate scheduler type for inner training")
# NEW: Add log-dir argument
p.add_argument("--log-dir", type=str, default=None,
               help="Directory for TensorBoard logs (if --log-tb is set)")
p.add_argument("--augment", action="store_true")
p.add_argument("--log-tb", action="store_true")
p.add_argument("--verbose", action="store_true")
p.add_argument("--train-log-level", choices=["ERROR", "WARNING", "INFO", "DEBUG"], default="ERROR")
p.add_argument("--precompute-topq", action="store_true", help="Precompute distances for top-q selection")
args = p.parse_args()

if len(args.delta) < len(args.dist):
    args.delta += args.delta[-1:] * (len(args.dist) - len(args.delta))
if len(args.delta) > len(args.dist):
    args.delta = args.delta[:len(args.dist)]

out = Path(args.model_dir)
_init_logging(out, args.verbose)
arch = getattr(custom_model, args.model)

# read watermark meta (optional)
wm_f = out / "watermark_info.json"
if wm_f.exists():
    wm = json.loads(wm_f.read_text())
else:
    wm = {}
    logging.warning("[watermark_info] watermark_info.json missing, using defaults")
args.watermark_method = wm.get("watermark_method", "none")
args.watermark_key = wm.get("watermark_key", "")
args.k = wm.get("k", 0)
args.randomize = wm.get("randomize", False)
args.num_parameters = wm.get("num_parameters", 0)
args.perturbation_strength = wm.get("perturbation_strength", 0.0)
args.watermark_size = wm.get("watermark_size", 128)

# Validate watermark parameters
required_wm_fields = ["watermark_method"]
if not all(f in wm for f in required_wm_fields if wm):
    logging.error("[watermark_info] Missing required fields in watermark_info.json")
    sys.exit(1)

seed = wm.get("seed", 777)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

# integrity pre‑checks
pol_ok = _check_init(out, arch) & _check_hash(out, args.dataset)

# shared cfg blob
cfg = dict(
    batch_size=args.batch_size,
    log_lvl=getattr(logging, args.train_log_level),
    scheduler_type=args.scheduler,  # NEW: Add scheduler_type to cfg
    train=dict(
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=1,
        dataset=args.dataset,
        augment=args.augment,
        architecture=arch,
        half=0,
        lambda_wm=0.0,
        k=args.k,
        randomize=args.randomize,
        watermark_key=args.watermark_key,
        watermark_method=args.watermark_method,
        num_parameters=args.num_parameters,
        perturbation_strength=args.perturbation_strength,
        watermark_size=args.watermark_size,
        log_tb=False,
    ),
)

writer = nullcontext()
if args.log_tb:
    from torch.utils.tensorboard import SummaryWriter
    # NEW: Use log_dir if provided, otherwise default to model_dir/tb_verify
    tb_log_dir = args.log_dir if args.log_dir else (out / "tb_verify")
    writer = SummaryWriter(log_dir=tb_log_dir)

with writer as tb:
    if args.q > 0:
        pol_ok &= verify_topq(
            model_dir=out,
            arch=arch,
            order=args.dist,
            q=args.q,
            epochs=args.epochs,
            cfg=cfg,
            writer=tb,
            precompute=args.precompute_topq
        )
    else:
        pol_ok &= verify_all(
            model_dir=out,
            arch=arch,
            order=args.dist,
            thr=args.delta,
            cfg=cfg,
            writer=tb
        )

# ───── watermark check ─────
wm_ok = True
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if args.watermark_method == "feature_based":
    ckpt = Path(args.watermark_path)
    if ckpt.exists():
        net = arch()
        net.load_state_dict(torch.load(ckpt, map_location=dev)["net"])
        wm_ok = _feature_wm_ok(net, dev)
    else:
        logging.error(f"{ckpt} missing")
        wm_ok = False
elif args.watermark_method == "non_intrusive":
    ckpt = Path(args.watermark_path)
    if ckpt.exists():
        net = WatermarkModule(arch(), args.watermark_key, args.watermark_size)
        net.load_state_dict(torch.load(ckpt, map_location=dev)["net"])
        wm_ok = verify_non_intrusive_watermark(net, dev, args.watermark_key, args.watermark_size, 1e-3)
    else:
        logging.error(f"{ckpt} missing")
        wm_ok = False
elif args.watermark_method == "parameter_perturbation":
    ckpt = Path(args.watermark_path)
    if ckpt.exists():
        st = torch.load(ckpt, map_location=dev)
        net = arch()
        net.load_state_dict(st["net"])
        wm_ok = verify_parameter_perturbation_watermark_relative(
            net, st.get("original_param_values"), args.watermark_key, args.perturbation_strength, 1e-1
        )
    else:
        logging.error(f"{ckpt} missing")
        wm_ok = False

logging.info(f"[PoL status] {'✓ valid' if pol_ok else '✗ INVALID'}")
logging.info(f"[watermark ] {'✓ passed' if wm_ok else '✗ FAILED'}")

sys.exit(0 if (pol_ok and wm_ok) else 1)