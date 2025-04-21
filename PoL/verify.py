#!/usr/bin/env python3
"""
verify.py — strict Proof‑of‑Learning & watermark verifier
"""

from __future__ import annotations
import os, sys, argparse, csv, glob, hashlib, json, logging, random, textwrap
from pathlib import Path
from contextlib import nullcontext
from typing import List, Optional

import numpy as np, torch
import utils, model as custom_model
from train import train
from watermark_utils import (
    WatermarkModule,
    verify_non_intrusive_watermark,
    verify_parameter_perturbation_watermark_relative,
    prepare_watermark_data,
    extract_features,
    run_feature_based_watermark_verification,
)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # silence cuDNN/TF chatter

# ───────────────────────── logging ──────────────────────────
def _init_logging(out_dir: Optional[Path], verbose: bool) -> None:
    """
    Initialize logging configuration.

    Args:
        out_dir: Output directory for log file, or None to log only to console.
        verbose: If True, set logging level to DEBUG; otherwise, INFO.
    """
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
def _dump(rows: List[dict], out_dir: Path, stem: str) -> None:
    """
    Dump verification results to CSV and JSON files.

    Args:
        rows: List of dictionaries containing verification metrics.
        out_dir: Directory to save the output files.
        stem: Stem for the output file names.
    """
    if not rows:
        return
    csv_p, js_p = out_dir / f"{stem}.csv", out_dir / f"{stem}.json"
    with csv_p.open("w", newline="") as f:
        w = csv.DictWriter(f, rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    js_p.write_text(json.dumps(rows, indent=2))
    logging.info(f"[export] {csv_p}")

# ───────── watermark sanity (feature‑based) ─────────
def _feature_wm_ok(net, dev="cpu", wm_key="secret_key") -> bool:
    """
    Check if the feature-based watermark is present in the model.

    Args:
        net: PyTorch model to check.
        dev: Device to use for computations (default: "cpu").
        wm_key: Watermark key (default: "secret_key").

    Returns:
        True if the watermark is detected, False otherwise.
    """
    net.to(dev).eval()
    inputs = prepare_watermark_data(device=dev, wm_key=wm_key)
    feats = extract_features(net, inputs, layer="layer1")
    mean_val = feats.mean().item()
    ok = mean_val > 0.01
    logging.info(f"[feature_wm] mean={mean_val:.4f} → {'✓' if ok else '✗'}")
    return ok

# ───────── integrity checks ─────────
def _check_hash(d: Path, dataset: str) -> bool:
    """
    Verify the dataset hash to ensure consistency.

    Args:
        d: Directory containing indices.npy and hash.txt.
        dataset: Name of the dataset.

    Returns:
        True if the hash matches, False otherwise.
    """
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
    """
    Verify the initial model weights using the Kolmogorov-Smirnov test.

    Args:
        d: Directory containing model_step_0 checkpoint.
        arch: Model architecture.

    Returns:
        True if the initialization is correct, False otherwise.
    """
    ck = d / "model_step_0"
    if not ck.exists():
        logging.error("[init‑ks] model_step_0 missing")
        return False
    st = torch.load(ck, map_location="cpu")
    wrapped = any(k.startswith("original_model.") for k in st["net"])
    net = WatermarkModule(arch(), "k", 128) if wrapped else arch()
    net.load_state_dict(st["net"])
    ks = utils.check_weights_initialization(next(net.parameters()), "resnet")
    ok = ks <= 0.01
    logging.info(f"[init‑ks] KS={ks:.2e} → {'✓' if ok else '✗'}")
    return ok

# ───────── inner‑train silencer ─────────
def _silent_train(lvl, scheduler_type="step", **kw):
    """
    Train the model with logging silenced.

    Args:
        lvl: Logging level to set during training.
        scheduler_type: Type of scheduler to use.
        **kw: Additional keyword arguments for training.

    Returns:
        The trained model.
    """
    root = logging.getLogger()
    prev = root.level
    root.setLevel(lvl)
    try:
        net, *_ = train(scheduler_type=scheduler_type, **kw, save_checkpoints=False)
    finally:
        root.setLevel(prev)
    return net

# ───────── distance helper ─────────
def _dist(a, b, order, arch):
    """
    Compute parameter distance between two models.

    Args:
        a: First model or checkpoint.
        b: Second model or checkpoint.
        order: Distance metric(s) to use.
        arch: Model architecture.

    Returns:
        List of distances for each metric.
    """
    return utils.parameter_distance(a, b, order, architecture=arch, half=0)

# ───────── full‑chain verifier ─────────
def verify_all(*, model_dir: Path, arch, order, thr, cfg, writer=None) -> bool:
    """
    Verify the full training chain by retraining and comparing checkpoints.

    Args:
        model_dir: Directory containing model checkpoints.
        arch: Model architecture.
        order: List of distance metrics to use.
        thr: Thresholds for each distance metric.
        cfg: Configuration dictionary.
        writer: TensorBoard writer (optional).

    Returns:
        True if all checks pass, False otherwise.
    """
    ck = sorted(int(Path(p).stem.split("_")[-1]) for p in glob.glob(str(model_dir / "model_step_*")))
    if len(ck) < 2:
        logging.error("no checkpoints → PoL invalid")
        return False
    seq = np.load(model_dir / "indices.npy")
    rows = []
    ok = True
    verify_temp_dir = model_dir / "verify_temp"
    verify_temp_dir.mkdir(exist_ok=True)
    for i, (c, n) in enumerate(zip(ck[:-1], ck[1:])):
        s, e = c * cfg["batch_size"], min(n * cfg["batch_size"], len(seq))
        if s >= len(seq) or e > len(seq):  # Skip intervals beyond sequence length
            continue
        if s >= e:
            logging.warning(f"Skipping interval {c}->{n} due to empty sequence slice")
            continue
        interval_dir = verify_temp_dir / f"interval_{c}_{n}"
        interval_dir.mkdir(exist_ok=True)
        net = _silent_train(
            cfg["log_lvl"],
            scheduler_type=cfg["scheduler_type"],
            model_dir=str(interval_dir),
            sequence=seq[s:e],
            **cfg["train"]
        )
        checkpoint_path = model_dir / f"model_step_{n}"
        checkpoint_state = torch.load(checkpoint_path, map_location="cpu")
        checkpoint_model = arch()
        checkpoint_model.load_state_dict(checkpoint_state["net"])
        d = _dist(checkpoint_model, net, order, arch)
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
    """
    Verify the top-q intervals with the largest parameter distances, skipping invalid intervals.

    Args:
        model_dir: Directory containing model checkpoints.
        arch: Model architecture.
        order: List of distance metrics to use.
        q: Number of top intervals to verify.
        epochs: Number of epochs.
        cfg: Configuration dictionary.
        writer: TensorBoard writer (optional).
        precompute: If True, precompute distances for top-q selection.

    Returns:
        True if verification is successful, False otherwise.
    """
    ck = sorted(int(Path(p).stem.split("_")[-1]) for p in glob.glob(str(model_dir / "model_step_*")))
    if len(ck) < 2:
        logging.error("no checkpoints → PoL invalid")
        return False
    seq = np.load(model_dir / "indices.npy")
    per = max(1, len(ck) // epochs)
    rows = []

    for ep in range(epochs):
        st, en = ep * per, min((ep + 1) * per, len(ck) - 1)
        valid_intervals = []
        distances = []

        # Filter valid intervals and compute distances
        for i in range(st, en):
            c, n = ck[i], ck[i + 1]
            s, e = c * cfg["batch_size"], min(n * cfg["batch_size"], len(seq))
            if s < e:  # Ensure non-empty slice
                valid_intervals.append((c, n))
                dist = _dist(model_dir / f"model_step_{c}", model_dir / f"model_step_{n}", order, arch)[0]
                distances.append(dist)

        if not valid_intervals:
            logging.warning(f"No valid intervals for epoch {ep}")
            continue

        # Select top-q intervals from valid ones
        if precompute:
            top_indices = np.argsort(distances)[-q:]
        else:
            top_indices = range(max(0, len(valid_intervals) - q), len(valid_intervals))

        # Retrain and verify selected intervals
        for idx in top_indices:
            c, n = valid_intervals[idx]
            s, e = c * cfg["batch_size"], min(n * cfg["batch_size"], len(seq))
            net = _silent_train(
                cfg["log_lvl"],
                scheduler_type=cfg["scheduler_type"],
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
p.add_argument("--model-dir", required=True, help="Directory containing model checkpoints and artifacts")
p.add_argument("--dataset", default="CIFAR10", help="Dataset name (e.g., CIFAR10, MNIST)")
p.add_argument("--model", default="resnet20", help="Model architecture (e.g., resnet20)")
p.add_argument("--batch-size", type=int, default=128, help="Batch size for training")
p.add_argument("--lr", type=float, default=0.1, help="Learning rate")
p.add_argument("--epochs", type=int, default=2, help="Number of epochs")
p.add_argument("--dist", nargs="+", default=["1", "2", "inf", "cos"], help="Distance metrics to use")
p.add_argument("--delta", nargs="+", type=float, default=[1e4, 100, 1, 0.1], help="Thresholds for distance metrics")
p.add_argument("--q", type=int, default=0, help="Number of top intervals to verify (0 for full verification)")
p.add_argument("--watermark-path", default="model_with_watermark.pth", help="Path to the watermarked model")
p.add_argument("--scheduler", type=str, default="step", choices=["none", "step", "cosine"], help="Scheduler type")
p.add_argument("--log-dir", type=str, default=None, help="Directory for TensorBoard logs")
p.add_argument("--log-tb", action="store_true", help="Enable TensorBoard logging")
p.add_argument("--verbose", action="store_true", help="Enable verbose logging")
p.add_argument("--train-log-level", choices=["ERROR", "WARNING", "INFO", "DEBUG"], default="ERROR", help="Log level for training")
p.add_argument("--precompute-topq", action="store_true", help="Precompute distances for top-q selection")
args = p.parse_args()

if len(args.delta) < len(args.dist):
    args.delta += args.delta[-1:] * (len(args.dist) - len(args.delta))
if len(args.delta) > len(args.dist):
    args.delta = args.delta[:len(args.dist)]

out = Path(args.model_dir)
_init_logging(out, args.verbose)
logging.info("Data augmentation is disabled to ensure reproducibility for Proof-of-Learning.")
arch = getattr(custom_model, args.model)

# read watermark meta (optional)
wm_f = out / "watermark_info.json"
if wm_f.exists():
    wm = json.loads(wm_f.read_text())
else:
    wm = {}
    logging.warning("[watermark_info] watermark_info.json missing, using defaults")
args.watermark_method = wm.get("watermark_method", "none")
args.watermark_key = wm.get("watermark_key", "secret_key")
args.k = wm.get("k", 100)
args.randomize = wm.get("randomize", False)
args.num_parameters = wm.get("num_parameters", 1000)
args.perturbation_strength = wm.get("perturbation_strength", 1e-5)
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
    scheduler_type=args.scheduler,
    train=dict(
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=1,
        dataset=args.dataset,
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
        net.load_state_dict(torch.load(ckpt, map_location=dev))
        wm_ok = run_feature_based_watermark_verification(
            model=net,
            wm_key=args.watermark_key,
            device=dev
        )
    else:
        logging.error(f"{ckpt} missing")
        wm_ok = False
elif args.watermark_method == "non_intrusive":
    ckpt = Path(args.watermark_path)
    if ckpt.exists():
        net = WatermarkModule(arch(), args.watermark_key, args.watermark_size)
        net.load_state_dict(torch.load(ckpt, map_location=dev))
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