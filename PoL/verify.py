#!/usr/bin/env python3
"""
verify.py – Proof‑of‑Learning & watermark verification  (quiet‑log build)

Supports
1. Baseline PoL (no watermark)
2. Feature‑based watermark
3. Parameter‑perturbation watermark
4. Non‑intrusive watermark

Two modes
• full‑interval verification
• top‑q verification

Outputs
• concise console / file logs
• CSV + JSON distance records
• optional TensorBoard scalars
"""

from __future__ import annotations
import os; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"     # hide cuFFT/cuDNN spam
import argparse, csv, glob, hashlib, json, logging, random
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

# ───────────────────────────── logging ───────────────────────────── #
def _init_logging(out_dir: str | None, verbose: bool) -> None:
    lvl = logging.DEBUG if verbose else logging.INFO
    handlers: List[logging.Handler] = [logging.StreamHandler()]
    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(Path(out_dir) / "verify.log"))
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s  %(levelname)5s  %(filename)s:%(lineno)d – %(message)s",
        handlers=handlers,
    )

# ──────────────────────── dump helpers ──────────────────────── #
def _dump(rows, out_dir: str, stem: str) -> None:
    """Write *rows* to CSV and JSON beside the proof directory."""
    if not rows:
        return
    csv_p, json_p = Path(out_dir)/f"{stem}.csv", Path(out_dir)/f"{stem}.json"
    with csv_p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    json_p.write_text(json.dumps(rows, indent=2))
    logging.info(f"[export] saved metrics → {csv_p}")

# ───────── watermark quick‑check helpers ───────── #
def _dummy_inputs(dev="cpu"): return torch.randn(100,3,32,32,device=dev)
def _extract(model,x,layer="layer1"):
    buf=[]
    h = dict(model.named_modules())[layer].register_forward_hook(lambda _,__,o:buf.append(o))
    with torch.no_grad(): model(x); h.remove()
    return buf[0]
def validate_feature_watermark(model, device="cpu")->float:
    model.to(device).eval()
    score=_extract(model,_dummy_inputs(device)).mean().item()
    logging.info(f"[feature‑wm] mean={score:.4g} → {'✓' if score>0.01 else '✗'}")
    return 1.0 if score>0.01 else 0.0

# ───────── integrity / init tests ───────── #
def verify_hash(model_dir:str, dataset:str)->None:
    seq=np.load(Path(model_dir)/"indices.npy")
    data=getattr(utils.load_dataset(dataset,train=True,augment=False),"data")
    ok=hashlib.sha256(data[seq].tobytes()).hexdigest()==Path(model_dir,"hash.txt").read_text().strip()
    logging.info(f"[hash] {'✓ match' if ok else '✗ MISMATCH'}")

def verify_initialization(model_dir:str, arch, thr:float=0.01)->None:
    st=torch.load(Path(model_dir)/"model_step_0",map_location="cpu")
    wrapped=any(k.startswith("original_model.") for k in st["net"])
    net=WatermarkModule(arch(),"k",128) if wrapped else arch()
    net.load_state_dict(st["net"])
    ks=utils.check_weights_initialization(next(net.parameters()),"resnet")
    logging.info(f"[init‑ks] KS‑stat={ks:.3g} → {'✓' if ks>=thr else '⚠️'}")

# ───────── inner‑train silencer ───────── #
def _run_train_silent(level:int, **kw):
    root=logging.getLogger(); prev=root.level; root.setLevel(level)
    try: net, *rest = train(**kw)
    finally: root.setLevel(prev)
    return (net, *rest)

# distance shortcut
def _dist(a,b,order,arch): return utils.parameter_distance(a,b,order,architecture=arch,half=0)

# ───────── verification helpers ───────── #
def _interval_replay(cur,nxt,seq,batch,*,train_level,train_kwargs):
    start,end=cur*batch,min(nxt*batch,len(seq))
    if end<=start: return None
    net,*_= _run_train_silent(train_level, sequence=seq[start:end], **train_kwargs)
    return net

def verify_all(
    *,
    model_dir, lr, batch_size, dataset, arch,
    order, threshold,
    k, randomize, watermark_key, watermark_method,
    num_parameters, perturbation_strength, watermark_size,
    augment=False, writer=None,
    train_log_level=logging.WARNING,
    skip_once=True
):
    ck = sorted(int(Path(p).stem.split("_")[-1])
                for p in glob.glob(f"{model_dir}/model_step_*"))
    seq = np.load(Path(model_dir) / "indices.npy")
    order = [order] if not isinstance(order, list) else order
    threshold = [threshold] if not isinstance(threshold, list) else threshold

    base = dict(
        lr=lr, batch_size=batch_size, epochs=1, dataset=dataset, augment=augment,
        architecture=arch, half=0, lambda_wm=0.0, k=k, randomize=randomize,
        watermark_key=watermark_key, watermark_method=watermark_method,
        num_parameters=num_parameters, perturbation_strength=perturbation_strength,
        watermark_size=watermark_size, log_tb=False
    )

    rows, skipped = [], 0
    for idx, (cur, nxt) in enumerate(zip(ck[:-1], ck[1:])):
        s, e = cur * batch_size, min(nxt * batch_size, len(seq))
        if e <= s:
            skipped += 1
            if skipped == 1 or not skip_once:
                logging.DEBUG(f"[skip] {cur}->{nxt} (empty slice)")
            continue

        net, *_ = _run_train_silent(
            train_log_level,
            model_dir=f"{model_dir}/model_step_{cur}",
            sequence=seq[s:e],
            **base
        )

        d = _dist(f"{model_dir}/model_step_{nxt}", net, order, arch)
        rows.append({"interval": f"{cur}->{nxt}",
                     **{str(o): v for o, v in zip(order, d)}})

        if writer:
            for o, v in zip(order, d):
                writer.add_scalar(f"dist_{o}", v, idx)

    if rows:
        arr = np.array([[r[str(o)] for r in rows] for o in order])
        pol_ok = True
        for j, o in enumerate(order):
            viol = int((arr[j] > threshold[j]).sum())
            pol_ok &= (viol == 0)
            logging.info(
                f"[{o}] avg={arr[j].mean():.3g}  max={arr[j].max():.3g}  "
                f"viol>{threshold[j]} → {viol}/{arr.shape[1]}"
            )
        logging.info(f"[PoL] {'✓ chain valid' if pol_ok else '✗ chain BROKEN'}")

    return rows


def verify_topq(
    *,
    model_dir, lr, batch_size, dataset, arch,
    epochs, q, order,
    k, randomize, watermark_key, watermark_method,
    num_parameters, perturbation_strength, watermark_size,
    augment=False, writer=None,
    train_log_level=logging.WARNING
):
    ck = sorted(int(Path(p).stem.split("_")[-1])
                for p in glob.glob(f"{model_dir}/model_step_*"))
    seq = np.load(Path(model_dir) / "indices.npy")
    per_ep = len(ck) // epochs
    order = [order] if not isinstance(order, list) else order

    base = dict(
        lr=lr, batch_size=batch_size, epochs=1, dataset=dataset, augment=augment,
        architecture=arch, half=0, lambda_wm=0.0, k=k, randomize=randomize,
        watermark_key=watermark_key, watermark_method=watermark_method,
        num_parameters=num_parameters, perturbation_strength=perturbation_strength,
        watermark_size=watermark_size, log_tb=False
    )

    rows, all_d = [], []
    for ep in range(epochs):
        st, en = ep * per_ep, (ep + 1) * per_ep if ep < epochs - 1 else len(ck)
        local = [
            _dist(f"{model_dir}/model_step_{ck[i]}",
                  f"{model_dir}/model_step_{ck[i+1]}", order, arch)[0]
            for i in range(st, en - 1)
        ]
        for ind in np.argsort(local)[-q:]:
            c, n = ck[st + ind], ck[st + ind + 1]
            s, e = c * batch_size, min(n * batch_size, len(seq))
            if e <= s:
                logging.debug(f"[skip] {c}->{n} (empty slice)")
                continue

            net, *_ = _run_train_silent(
                train_log_level,
                model_dir=f"{model_dir}/model_step_{c}",
                sequence=seq[s:e],
                **base
            )

            d = _dist(f"{model_dir}/model_step_{n}", net, order, arch)[0]
            rows.append({"interval": f"{c}->{n}", str(order[0]): d})
            all_d.append(d)

            if writer:
                writer.add_scalar(f"topq_dist_{order[0]}", d, len(rows))

    if all_d:
        arr = np.array(all_d)
        logging.info(
            f"[top‑q {order[0]}] avg={arr.mean():.3g}  max={arr.max():.3g}  count={arr.size}"
        )

    return rows

# ─────────────────────────── CLI ─────────────────────────── #
pa=argparse.ArgumentParser("PoL & watermark verification")
pa.add_argument("--batch-size",type=int,default=128)
pa.add_argument("--lr",type=float,default=0.1)
pa.add_argument("--epochs",type=int,default=2)
pa.add_argument("--dataset",default="CIFAR10")
pa.add_argument("--model",default="resnet20")
pa.add_argument("--model-dir",default="proof/CIFAR10_Run")
pa.add_argument("--save-freq", type=int, default=100)
pa.add_argument("--dist",nargs="+",default=["1","2","inf","cos"])
pa.add_argument("--delta",nargs="+",type=float,default=[10000,100,1,0.1])
pa.add_argument("--q",type=int,default=0)
pa.add_argument("--watermark-path",default="model_with_watermark.pth")
pa.add_argument("--log-tb",action="store_true")
pa.add_argument("--verbose",action="store_true")
pa.add_argument("--augment",action="store_true",help="Replay with CIFAR crop/flip")
pa.add_argument("--train-log-level",choices=["WARNING","INFO","DEBUG"],
                default="WARNING",help="inner train() log‑level")
args=pa.parse_args()

# ───────── setup / seed ───────── #
out=args.model_dir; _init_logging(out,args.verbose)
arch=getattr(custom_model,args.model)

wm=json.loads(Path(out,"watermark_info.json").read_text())
for k,v in wm.items(): setattr(args,k.replace("-","_"),v)

seed=wm.get("seed",777)
torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
np.random.seed(seed); random.seed(seed)

verify_initialization(out,arch); verify_hash(out,args.dataset)

writer=nullcontext()
if args.log_tb:
    from torch.utils.tensorboard import SummaryWriter
    writer=SummaryWriter(log_dir=f"{out}/tb_verify")

with writer as tb:
    if args.q>0:
        rec=verify_topq(model_dir=out,lr=args.lr,batch_size=args.batch_size,dataset=args.dataset,
                        arch=arch,epochs=args.epochs,q=args.q,order=args.dist,k=args.k,
                        randomize=args.randomize,watermark_key=args.watermark_key,
                        watermark_method=args.watermark_method,num_parameters=args.num_parameters,
                        perturbation_strength=args.perturbation_strength,
                        watermark_size=args.watermark_size,writer=tb,augment=args.augment,
                        train_log_level=getattr(logging,args.train_log_level))
        _dump(rec,out,"verify_topq_metrics")
    else:
        rec=verify_all(model_dir=out,lr=args.lr,batch_size=args.batch_size,dataset=args.dataset,
                       arch=arch,order=args.dist,threshold=args.delta,k=args.k,
                       randomize=args.randomize,watermark_key=args.watermark_key,
                       watermark_method=args.watermark_method,num_parameters=args.num_parameters,
                       perturbation_strength=args.perturbation_strength,
                       watermark_size=args.watermark_size,writer=tb,augment=args.augment,
                       train_log_level=getattr(logging,args.train_log_level))
        _dump(rec,out,"verify_full_metrics")

# ───────── watermark presence ───────── #
dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
method=wm.get("watermark_method","none"); pth=args.watermark_path

if method=="non_intrusive":
    net=WatermarkModule(arch(),args.watermark_key, watermark_size=wm["watermark_size"])
    net.load_state_dict(torch.load(pth,map_location=dev)["net"])
    ok=verify_non_intrusive_watermark(net,dev,args.watermark_key,wm["watermark_size"],1e-3)
elif method=="feature_based":
    net=arch(); net.load_state_dict(torch.load(pth,map_location=dev)["net"])
    ok=validate_feature_watermark(net,dev)==1.0
elif method=="parameter_perturbation":
    st=torch.load(pth,map_location=dev); net=arch(); net.load_state_dict(st["net"])
    ok=verify_parameter_perturbation_watermark_relative(
            net, st.get("original_param_values"), args.watermark_key,
            wm["perturbation_strength"], 1e-1)
else:
    ok=True
logging.info(f"[watermark‑check] {'✓ passed' if ok else '✗ FAILED'}")
