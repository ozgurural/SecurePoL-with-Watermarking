#!/usr/bin/env python3
"""
verify.py – Proof‑of‑Learning & watermark verification   (quiet‑log build)

Supports
    1. Baseline PoL (no watermark)
    2. Feature‑based watermark
    3. Parameter‑perturbation watermark
    4. Non‑intrusive watermark

Modes
    • full‑interval verification
    • top‑q verification

Outputs
    • concise console / file logs
    • CSV + JSON distance records
    • optional TensorBoard scalars
"""

from __future__ import annotations
import os; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"       # silence TF
import argparse, csv, glob, hashlib, json, logging, random, textwrap
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

# ────────────────────────── logging helpers ───────────────────────── #
def _init_logging(out_dir: str | None, verbose: bool) -> None:
    lvl = logging.DEBUG if verbose else logging.INFO
    handlers: List[logging.Handler] = [logging.StreamHandler()]
    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(Path(out_dir) / "verify.log"))
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s  %(levelname)5s  %(message)s",
        handlers=handlers,
    )

# ───────────────────── CSV / JSON export ───────────────────── #
def _dump(rows, out_dir: Path, stem: str) -> None:
    if not rows:
        return
    csv_p, json_p = out_dir / f"{stem}.csv", out_dir / f"{stem}.json"
    with csv_p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    json_p.write_text(json.dumps(rows, indent=2))
    logging.info(f"[export] {csv_p.relative_to(Path.cwd())}")

# ─────────────────── watermark sanity helpers ─────────────────── #
def _dummy_inputs(dev="cpu"):
    return torch.randn(100, 3, 32, 32, device=dev)

def _extract(model, x, layer="layer1"):
    buf: list[torch.Tensor] = []
    h = dict(model.named_modules())[layer].register_forward_hook(
        lambda _, __, out: buf.append(out)
    )
    with torch.no_grad(): model(x)
    h.remove()
    return buf[0]

def validate_feature_watermark(model, device="cpu") -> bool:
    model.to(device).eval()
    score = _extract(model, _dummy_inputs(device)).mean().item()
    logging.info(f"[feature‑wm] mean={score:.4g} "
                 f"→ {'✓ detected' if score>0.01 else '✗ not‑found'}")
    return score > 0.01

# ────────────────── integrity / init tests ────────────────── #
def verify_hash(model_dir: Path, dataset: str) -> None:
    idx_f = model_dir / "indices.npy"
    hash_f = model_dir / "hash.txt"
    if not (idx_f.exists() and hash_f.exists()):
        logging.warning("[hash] indices.npy or hash.txt missing; skipping")
        return
    seq  = np.load(idx_f)
    data = getattr(utils.load_dataset(dataset, train=True, augment=False), "data")
    ok   = hashlib.sha256(data[seq].tobytes()).hexdigest() == hash_f.read_text().strip()
    logging.info(f"[hash] {'✓ match' if ok else '✗ MISMATCH'}")

def verify_initialization(model_dir: Path, arch, thr: float = 0.01) -> None:
    ck0 = model_dir / "model_step_0"
    if not ck0.exists():
        logging.warning("[init‑ks] model_step_0 not found; skipping")
        return
    st      = torch.load(ck0, map_location="cpu")
    wrapped = any(k.startswith("original_model.") for k in st["net"])
    net     = WatermarkModule(arch(), "dummy", 128) if wrapped else arch()
    net.load_state_dict(st["net"])
    ks      = utils.check_weights_initialization(next(net.parameters()), "resnet")
    logging.info(f"[init‑ks] KS‑stat={ks:.3g} "
                 f"→ {'✓ ok' if ks>=thr else '⚠️ suspect'}")

# ───────────────── inner  train silencer ───────────────── #
def _run_train_silent(log_lvl: int, **kw):
    root = logging.getLogger()
    prev = root.level
    root.setLevel(log_lvl)
    try:
        net, *rest = train(**kw)
    finally:
        root.setLevel(prev)
    return net, *rest

# shorthand
def _dist(a, b, order, arch):
    return utils.parameter_distance(a, b, order, architecture=arch, half=0)

# ───────────────────────── full‑interval ───────────────────────── #
def verify_all(*,
    model_dir: Path, lr: float, batch_size: int, dataset: str, arch,
    order: list[str], threshold: list[float],
    k: int, randomize: bool, watermark_key: str, watermark_method: str,
    num_parameters: int, perturbation_strength: float, watermark_size: int,
    augment=False, writer=None, train_log_level=logging.ERROR,
):
    ckpts = sorted(int(Path(p).stem.split("_")[-1])
                   for p in glob.glob(str(model_dir / "model_step_*")))
    if len(ckpts) < 2:
        raise RuntimeError("No checkpoints found inside proof directory.")

    seq  = np.load(model_dir / "indices.npy")
    base = dict(
        lr=lr, batch_size=batch_size, epochs=1, dataset=dataset, augment=augment,
        architecture=arch, half=0, lambda_wm=0.0, k=k, randomize=randomize,
        watermark_key=watermark_key, watermark_method=watermark_method,
        num_parameters=num_parameters, perturbation_strength=perturbation_strength,
        watermark_size=watermark_size, log_tb=False,
    )

    rows: list[dict[str,float]] = []
    for idx, (cur, nxt) in enumerate(zip(ckpts[:-1], ckpts[1:]), start=1):
        s, e = cur*batch_size, min(nxt*batch_size, len(seq))
        if e <= s:
            logging.debug(f"[skip] {cur}->{nxt} (empty slice)"); continue

        net, *_ = _run_train_silent(
            train_log_level,
            model_dir=str(model_dir / f"model_step_{cur}"),
            sequence=seq[s:e],
            **base
        )
        dvals = _dist(model_dir / f"model_step_{nxt}", net, order, arch)
        rows.append({"interval": f"{cur}->{nxt}",
                     **{str(o): v for o, v in zip(order, dvals)}})

        if writer:
            for o, v in zip(order, dvals):
                writer.add_scalar(f"dist_{o}", v, idx)

    # summary
    if rows:
        arr = np.array([[r[str(o)] for r in rows] for o in order])
        ok  = True
        for j, o in enumerate(order):
            viol = int((arr[j] > threshold[j]).sum())
            ok  &= viol == 0
            logging.info(f"[{o}] avg={arr[j].mean():.3g}  "
                         f"max={arr[j].max():.3g}  "
                         f"viol>{threshold[j]} → {viol}/{arr.shape[1]}")
        logging.info(f"[PoL] {'✓ chain valid' if ok else '✗ chain BROKEN'}")
    return rows

# ─────────────────────────── top‑q ─────────────────────────── #
def verify_topq(*,
    model_dir: Path, lr: float, batch_size: int, dataset: str, arch,
    epochs: int, q: int, order: list[str],
    k: int, randomize: bool, watermark_key: str, watermark_method: str,
    num_parameters: int, perturbation_strength: float, watermark_size: int,
    augment=False, writer=None, train_log_level=logging.ERROR,
):
    ckpts = sorted(int(Path(p).stem.split("_")[-1])
                   for p in glob.glob(str(model_dir / "model_step_*")))
    if len(ckpts) < 2:
        raise RuntimeError("No checkpoints found inside proof directory.")

    seq   = np.load(model_dir / "indices.npy")
    per_ep = max(1, len(ckpts)//epochs)
    base  = dict(
        lr=lr, batch_size=batch_size, epochs=1, dataset=dataset, augment=augment,
        architecture=arch, half=0, lambda_wm=0.0, k=k, randomize=randomize,
        watermark_key=watermark_key, watermark_method=watermark_method,
        num_parameters=num_parameters, perturbation_strength=perturbation_strength,
        watermark_size=watermark_size, log_tb=False,
    )

    rows, all_d = [], []
    for ep in range(epochs):
        st, en  = ep*per_ep, min((ep+1)*per_ep, len(ckpts)-1)
        if st >= en: continue
        local_d = [_dist(model_dir / f"model_step_{ckpts[i]}",
                         model_dir / f"model_step_{ckpts[i+1]}",
                         order, arch)[0]
                   for i in range(st, en)]
        top_idx = np.argsort(local_d)[-q:] if q>0 else range(len(local_d))
        for ind in top_idx:
            c, n = ckpts[st+ind], ckpts[st+ind+1]
            s, e = c*batch_size, min(n*batch_size, len(seq))
            if e <= s: continue
            net, *_ = _run_train_silent(
                train_log_level,
                model_dir=str(model_dir / f"model_step_{c}"),
                sequence=seq[s:e],
                **base
            )
            d = _dist(model_dir / f"model_step_{n}", net, order, arch)[0]
            rows.append({"interval": f"{c}->{n}", str(order[0]): d})
            all_d.append(d)
            if writer:
                writer.add_scalar(f"topq_dist_{order[0]}", d, len(rows))

    if all_d:
        arr = np.array(all_d)
        logging.info(f"[top‑q {order[0]}] avg={arr.mean():.3g} "
                     f"max={arr.max():.3g}  count={arr.size}")
    return rows

# ─────────────────────────── CLI ─────────────────────────── #
pa = argparse.ArgumentParser(
    prog="verify.py",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description=textwrap.dedent("""\
        Proof‑of‑Learning chain verifier + watermark checker.

        Typical call:
            python verify.py --model-dir proof/CIFAR10_Run --model resnet20 --epochs 100
    """)
)
pa.add_argument("--batch-size", type=int, default=128)
pa.add_argument("--lr", type=float, default=0.1)
pa.add_argument("--epochs", type=int, default=2, help="epochs to *replay* (for top‑q)")
pa.add_argument("--dataset", default="CIFAR10")
pa.add_argument("--model",   default="resnet20")
pa.add_argument("--model-dir", default="proof/CIFAR10_Run")
pa.add_argument("--dist",  nargs="+", default=["1","2","inf","cos"])
pa.add_argument("--delta", nargs="+", type=float,
                default=[1e4, 100, 1, 0.1],
                help="threshold(s) for --dist")
pa.add_argument("--q", type=int, default=0, help="top‑q mode if >0")
pa.add_argument("--watermark-path", default="model_with_watermark.pth")
pa.add_argument("--log-tb", action="store_true")
pa.add_argument("--verbose", action="store_true")
pa.add_argument("--augment", action="store_true",
                help="Replay with CIFAR crop/flip")
pa.add_argument("--train-log-level",
                choices=["ERROR","WARNING","INFO","DEBUG"],
                default="ERROR",
                help="verbosity of *inner* train()")
args = pa.parse_args()

# ───────── basic validation ───────── #
if len(args.delta) < len(args.dist):
    logging.warning("--delta shorter than --dist, broadcasting last value")
    args.delta = args.delta + [args.delta[-1]]*(len(args.dist)-len(args.delta))
elif len(args.delta) > len(args.dist):
    logging.warning("--delta longer than --dist, extras ignored")
    args.delta = args.delta[:len(args.dist)]

out_dir = Path(args.model_dir)
_init_logging(out_dir, args.verbose)
arch = getattr(custom_model, args.model)

# ---- watermark meta (optional) ----
wm_path = out_dir / "watermark_info.json"
wm: dict = {}
if wm_path.exists():
    wm = json.loads(wm_path.read_text())
else:
    logging.warning(f"{wm_path.name} not found; assuming baseline‑PoL only")

# inject defaults so code never crashes later
setattr(args, "watermark_method", wm.get("watermark_method", "none"))
setattr(args, "watermark_key",   wm.get("watermark_key",   ""))
setattr(args, "k",               wm.get("k",               0))
setattr(args, "randomize",       wm.get("randomize",       False))
setattr(args, "num_parameters",  wm.get("num_parameters",  0))
setattr(args, "perturbation_strength",
        wm.get("perturbation_strength", 0.0))
setattr(args, "watermark_size", wm.get("watermark_size", 128))

seed = wm.get("seed", 777)
torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
np.random.seed(seed);    random.seed(seed)

# quick sanity on proof
verify_initialization(out_dir, arch)
verify_hash(out_dir, args.dataset)

# tensorboard writer
writer = nullcontext()
if args.log_tb:
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir=out_dir / "tb_verify")

with writer as tb:
    if args.q > 0:
        rec = verify_topq(
            model_dir=out_dir, lr=args.lr, batch_size=args.batch_size,
            dataset=args.dataset, arch=arch, epochs=args.epochs, q=args.q,
            order=args.dist, k=args.k, randomize=args.randomize,
            watermark_key=args.watermark_key, watermark_method=args.watermark_method,
            num_parameters=args.num_parameters,
            perturbation_strength=args.perturbation_strength,
            watermark_size=args.watermark_size,
            writer=tb, augment=args.augment,
            train_log_level=getattr(logging, args.train_log_level)
        )
        _dump(rec, out_dir, "verify_topq_metrics")
    else:
        rec = verify_all(
            model_dir=out_dir, lr=args.lr, batch_size=args.batch_size,
            dataset=args.dataset, arch=arch, order=args.dist,
            threshold=args.delta, k=args.k, randomize=args.randomize,
            watermark_key=args.watermark_key, watermark_method=args.watermark_method,
            num_parameters=args.num_parameters,
            perturbation_strength=args.perturbation_strength,
            watermark_size=args.watermark_size,
            writer=tb, augment=args.augment,
            train_log_level=getattr(logging, args.train_log_level)
        )
        _dump(rec, out_dir, "verify_full_metrics")

# ───────── watermark presence ───────── #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
wm_method = args.watermark_method
wm_model  = args.watermark_path

def _safe_load(path):
    if not Path(path).exists():
        logging.warning(f"{path} not found – skipping wm check"); return None
    return torch.load(path, map_location=device)

ok = True  # baseline
if wm_method == "non_intrusive":
    ckpt = _safe_load(wm_model)
    if ckpt is not None:
        net = WatermarkModule(arch(), args.watermark_key, watermark_size=args.watermark_size)
        net.load_state_dict(ckpt["net"])
        ok = verify_non_intrusive_watermark(
            net, device, args.watermark_key, args.watermark_size, 1e-3)
elif wm_method == "feature_based":
    ckpt = _safe_load(wm_model)
    if ckpt is not None:
        net = arch(); net.load_state_dict(ckpt["net"])
        ok  = validate_feature_watermark(net, device)
elif wm_method == "parameter_perturbation":
    ckpt = _safe_load(wm_model)
    if ckpt is not None:
        net = arch(); net.load_state_dict(ckpt["net"])
        ok  = verify_parameter_perturbation_watermark_relative(
                net, ckpt.get("original_param_values"),
                args.watermark_key, args.perturbation_strength, 1e-1)

logging.info(f"[watermark‑check] {'✓ passed' if ok else '✗ FAILED'}")
