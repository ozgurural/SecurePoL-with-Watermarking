# 🔐 SecurePoL-WM  
**Enhancing Security of Proof-of-Learning (PoL) Against Spoofing Attacks via Advanced Model Watermarking**

[![Python](https://img.shields.io/badge/python->=3.9-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2.x-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **TL;DR** This repo adds three watermark schemes—**feature-based**, **parameter-perturbation**, and **non-intrusive auxiliary head**—on top of Proof-of-Learning.  
> The combo defeats state-of-the-art spoofing attacks while keeping accuracy intact (except when you push PP too hard).  
> Includes a strict verifier, spoofing attack harness, and full 100-epoch CIFAR-10 benchmarks (ResNet-20).

---

## Table of Contents
1. [Why another PoL repo?](#why-another-pol-repo)
2. [Key results](#key-results)
3. [Repo layout](#repo-layout)
4. [Quick start](#quick-start)
5. [Training](#training-with-watermarks)
6. [Verification](#verification)
7. [Hyper-parameter sweeps](#hyper-parameter-sweeps)
8. [Spoofing attack sandbox](#spoofing-attack-sandbox)
9. [Reproducing dissertation figures](#reproduce-dissertation-figures)
10. [Cite & license](#cite--license)

---

## Why another PoL repo?  
PoL logs checkpoints to prove a model was really trained, but **attackers can still forge logs** cheaply.  
We raise the bar by forcing them to reproduce a hidden watermark **and** the trajectory:

| Watermark | Hidden where | Verifier access | Overhead | Acc. drop* |
|-----------|--------------|-----------------|---------:|-----------:|
| Feature-based (FB) | 1 % mask in `layer1` activations | trigger queries (black-box) | +0.5 % time | \<0.05 pp |
| Non-intrusive (NI) | dormant 128-D head | MSE on trigger (mixed) | +1.0 % params | \<0.05 pp |
| Param-perturb. (PP) | flips N weights | weight dump (white-box) | none | **-29 pp** (at λ = 0.01, 20 params, Δ = 1e-5) |

\*100-epoch CIFAR-10 / ResNet-20, see § 2.

---

## Key results

| Setup | λ | Extra WM params | Val Acc | Val Loss | PoL | WM | Wall-clock* |
|-------|---|-----------------|-------:|--------:|:--:|:--:|------------:|
| Baseline | – | – | **83.19 %** | 0.7933 | ✓ | – | 2 686 s |
| Feature-based | 0.01 | k = 200 | **83.19 %** | 0.7933 | ✓ | ✓ | 3 150 s |
| Non-intrusive | 0.03 | size = 128, k = 1 | **83.16 %** | 0.7962 | ✓ | ✓ | 2 973 s |
| Param-perturb. | 0.01 | 20 params, Δ = 1e-5 | **54.07 %** | 1.2754 | ✓ | ✓ | 2 405 s |

\*Single V100, mixed precision.  Full logs in `proof/`.

---
## Repo layout
```
PoL/                core trainer + verifier  
└── watermark_utils.py  
spoof_cifar/        CIFAR-10/100 spoofing attacks  
spoof_imagenet/     Tiny-ImageNet spoofing demo  
notebooks/          Jupyter notebooks → dissertation figs  
logs/               TensorBoard runs *(git-ignored)*  
proof/              Sample PoL + watermark artefacts  
```

---

## Quick-start

1. **Open a GPU notebook**  
   *Runtime ▸ Change runtime type ▸ GPU*

2. **Clone & install**
   ```python
   !git clone https://github.com/ozgurural/SecurePoL-with-Watermarking.git
   %cd SecurePoL-with-Watermarking
   !pip install torch torchvision numpy scipy tqdm tensorboard
   ```

3. **(Optional) mount Google Drive** – keeps checkpoints & logs
   ```python
   from google.colab import drive; drive.mount('/content/drive')
   ```

4. **Baseline 100-epoch run (no watermark)**
   ```python
   !python PoL/train.py --dataset CIFAR10 --model resnet20 \
       --epochs 100 --save-freq 100 --lr 0.1 \
       --watermark-method none \
       --model-dir /content/drive/MyDrive/proof/CIFAR10_Run
   ```

### TensorBoard in Colab
```python
%load_ext tensorboard
%tensorboard --logdir logs
```

---

## Training with watermarks
```python
!python PoL/train.py \
  --dataset CIFAR10 --model resnet20 --epochs 100 --save-freq 100 \
  --lr 0.1 \
  --watermark-method feature_based \   # none|feature_based|parameter_perturbation|non_intrusive
  --lambda-wm 0.01 --k 200 --watermark-key secret_key \
  --model-dir /content/drive/MyDrive/proof/CIFAR10_feature_based \
  --log-tb --log-dir logs/FB
```

| Scheme | Useful flags | Example |
|--------|--------------|---------|
| **Parameter-perturbation (PP)** | `--num-parameters`<br>`--perturbation-strength` | `--num-parameters 100 --perturbation-strength 5e-6` |
| **Non-intrusive (NI)** | `--watermark-size`<br>`--tolerance-wm` | `--watermark-size 128 --tolerance-wm 1.0` |
| **Randomised embed** | `--randomize` |  |

_All flags_: `python PoL/train.py --help`

---

## Verification
```python
!python PoL/verify.py \
  --model-dir /content/drive/MyDrive/proof/CIFAR10_feature_based \
  --dataset CIFAR10 --model resnet20 \
  --dist 1 2 inf cos --delta 100000 1000 10 1 \
  --watermark-method feature_based \
  --watermark-path /content/drive/MyDrive/proof/CIFAR10_feature_based/model_with_feature_based_watermark.pth
#   Top-q sampling → add  --q 10
#   Full replay     → omit --q
```

---

## Hyper-parameter sweeps

| Sweep depth | Grid | Runs | ≈ GPU-h* |
|-------------|------|-----:|--------:|
| Quick sanity | Latin-hypercube (6 pts) | 6 | 4 |
| Reduced PP   | λ = 0.03; params × Δ = 3 × 3 | 9 | 7 |

\*Timing based on a Colab T4 VM (~0.75× V100).

Run:
```bash
bash scripts/sweep_pp.sh   # edit grid inside
```
*(see dissertation § 4.6 for full method).*

---

## Spoofing-attack sandbox
```python
!python spoof_cifar/attack.py \
  --attack 1 --dataset CIFAR10 --model resnet20 \
  --t 500 --verify 1          # Attacks 1–3 = Jia et al.
```
Tiny-ImageNet demo:
```python
!python spoof_imagenet/attack3_imagenet.py --t 300 --verify 1
```

---

## Reproduce dissertation figures
```python
!jupyter nbconvert --to html --execute notebooks/feature_based.ipynb
```
(Notebooks read `proof/*` and regenerate Ch. 5 plots.)

---

## Cite & license
```bibtex
@ARTICLE{10741282,
  author={Ural, Ozgur and Yoshigoe, Kenji},
  journal={IEEE Access}, 
  title={Enhancing Security of Proof-of-Learning Against Spoofing Attacks Using Feature-Based Model Watermarking}, 
  year={2024},
  volume={12},
  number={},
  pages={169567-169591},
  keywords={Watermarking;Computational modeling;Training;Data models;Cryptography;Adaptation models;Computational efficiency;Training data;Analytical models;Robustness;Authentication;Machine learning;Proof-of-learning;model watermarking;machine learning security;spoofing attack countermeasures;dual-layered verification;model authenticity;intellectual property protection in ML;computational effort authentication;security enhancements in machine learning;watermark robustness;model integrity verification},
  doi={10.1109/ACCESS.2024.3489776}}
```
Released under the **MIT License** – see `LICENSE`.

<br>

*Open an issue or PR if you hit a snag – happy hacking 🚀*
